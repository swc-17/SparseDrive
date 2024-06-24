from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from shapely.geometry import Polygon

from mmcv.utils import print_log
from mmdet.datasets import build_dataset, build_dataloader

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners


def check_collision(ego_box, boxes):
    '''
        ego_box: tensor with shape [7], [x, y, z, w, l, h, yaw]
        boxes: tensor with shape [N, 7]
    '''
    if  boxes.shape[0] == 0:
        return False

    # follow uniad, add a 0.5m offset
    ego_box[0] += 0.5 * torch.cos(ego_box[6])
    ego_box[1] += 0.5 * torch.sin(ego_box[6])
    ego_corners_box = box3d_to_corners(ego_box.unsqueeze(0))[0, [0, 3, 7, 4], :2]
    corners_box = box3d_to_corners(boxes)[:, [0, 3, 7, 4], :2]
    ego_poly = Polygon([(point[0], point[1]) for point in ego_corners_box])
    for i in range(len(corners_box)):
        box_poly =  Polygon([(point[0], point[1]) for point in corners_box[i]])
        collision = ego_poly.intersects(box_poly)
        if collision:
            return True

    return False

def get_yaw(traj):
    start = traj[0]
    end = traj[-1]
    dist = torch.linalg.norm(end - start, dim=-1)
    if dist < 0.5:
        return traj.new_ones(traj.shape[0]) * np.pi / 2

    zeros = traj.new_zeros((1, 2))
    traj_cat = torch.cat([zeros, traj], dim=0)
    yaw = traj.new_zeros(traj.shape[0]+1)
    yaw[..., 1:-1] = torch.atan2(
        traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
        traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    return yaw[1:]

class PlanningMetric():
    def __init__(
        self,
        n_future=6,
        compute_on_step: bool = False,
    ):
        self.W = 1.85
        self.H = 4.084

        self.n_future = n_future
        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future)
        self.obj_box_col = torch.zeros(self.n_future)
        self.L2 = torch.zeros(self.n_future)
        self.total = torch.tensor(0)

    def evaluate_single_coll(self, traj, fut_boxes):
        n_future = traj.shape[0]
        yaw = get_yaw(traj)
        ego_box = traj.new_zeros((n_future, 7))
        ego_box[:, :2] = traj
        ego_box[:, 3:6] = ego_box.new_tensor([self.H, self.W, 1.56])
        ego_box[:, 6] = yaw
        collision = torch.zeros(n_future, dtype=torch.bool)

        for t in range(n_future):
            ego_box_t = ego_box[t].clone()
            boxes = fut_boxes[t][0].clone()
            collision[t] = check_collision(ego_box_t, boxes)
        return collision

    def evaluate_coll(self, trajs, gt_trajs, fut_boxes):
        B, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=trajs.device)
        obj_box_coll_sum = torch.zeros(n_future, device=trajs.device)

        assert B == 1, 'only supprt bs=1'
        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], fut_boxes)
            box_coll = self.evaluate_single_coll(trajs[i], fut_boxes)
            box_coll = torch.logical_and(box_coll, torch.logical_not(gt_box_coll))
            
            obj_coll_sum += gt_box_coll.long()
            obj_box_coll_sum += box_coll.long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        '''
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        '''
        return torch.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)) 

    def update(self, trajs, gt_trajs, gt_trajs_mask, fut_boxes):
        assert trajs.shape == gt_trajs.shape
        trajs[..., 0] = - trajs[..., 0]
        gt_trajs[..., 0] = - gt_trajs[..., 0]
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], fut_boxes)

        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(dim=0)
        self.total +=len(trajs)

    def compute(self):
        return {
            'obj_col': self.obj_col / self.total,
            'obj_box_col': self.obj_box_col / self.total,
            'L2' : self.L2 / self.total
        }


def planning_eval(results, eval_config, logger):
    dataset = build_dataset(eval_config)
    dataloader = build_dataloader(
            dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)
    planning_metrics = PlanningMetric()
    for i, data in enumerate(tqdm(dataloader)):
        sdc_planning = data['gt_ego_fut_trajs'].cumsum(dim=-2).unsqueeze(1)
        sdc_planning_mask = data['gt_ego_fut_masks'].unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)
        command = data['gt_ego_fut_cmd'].argmax(dim=-1).item()
        fut_boxes = data['fut_boxes']
        if not sdc_planning_mask.all(): ## for incomplete gt, we do not count this sample
            continue
        res = results[i]
        pred_sdc_traj = res['img_bbox']['final_planning'].unsqueeze(0)
        planning_metrics.update(pred_sdc_traj[:, :6, :2], sdc_planning[0,:, :6, :2], sdc_planning_mask[0,:, :6, :2], fut_boxes)
       
    planning_results = planning_metrics.compute()
    planning_metrics.reset()
    from prettytable import PrettyTable
    planning_tab = PrettyTable()
    metric_dict = {}

    planning_tab.field_names = [
    "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s", "avg"]
    for key in planning_results.keys():
        value = planning_results[key].tolist()
        new_values = []
        for i in range(len(value)):
            new_values.append(np.array(value[:i+1]).mean())
        value = new_values
        avg = [value[1], value[3], value[5]]
        avg = sum(avg) / len(avg)
        value.append(avg)
        metric_dict[key] = avg
        row_value = []
        row_value.append(key)
        for i in range(len(value)):
            if 'col' in key:
                row_value.append('%.3f' % float(value[i]*100) + '%')
            else:
                row_value.append('%.4f' % float(value[i]))
        planning_tab.add_row(row_value)

    print_log('\n'+str(planning_tab), logger=logger)
    return metric_dict
