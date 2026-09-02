import random
import math
import os
from multiprocessing import Pool
from os import path as osp
import cv2
import tempfile
import copy
import prettytable
import pickle
import json

import numpy as np
# np.set_printoptions(suppress=True)
import torch
from torch.utils.data import Dataset
import pyquaternion
from pyquaternion import Quaternion
from shapely.geometry import LineString
from .evaluation.detection.nuscenes_styled_eval_utils import (
    DetectionMetrics, 
    EvalBoxes, 
    DetectionBox,
    center_distance,
    accumulate,
    DetectionMetricDataList,
    calc_ap, 
    calc_tp, 
    quaternion_yaw,
)


import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from .utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)

Discrete_Actions_DICT = {
    0:  (0, 0, 1, False),
    1:  (0.7, -0.5, 0, False),
    2:  (0.7, -0.3, 0, False),
    3:  (0.7, -0.2, 0, False),
    4:  (0.7, -0.1, 0, False),
    5:  (0.7, 0, 0, False),
    6:  (0.7, 0.1, 0, False),
    7:  (0.7, 0.2, 0, False),
    8:  (0.7, 0.3, 0, False),
    9:  (0.7, 0.5, 0, False),
    10: (0.3, -0.7, 0, False),
    11: (0.3, -0.5, 0, False),
    12: (0.3, -0.3, 0, False),
    13: (0.3, -0.2, 0, False),
    14: (0.3, -0.1, 0, False),
    15: (0.3, 0, 0, False),
    16: (0.3, 0.1, 0, False),
    17: (0.3, 0.2, 0, False),
    18: (0.3, 0.3, 0, False),
    19: (0.3, 0.5, 0, False),
    20: (0.3, 0.7, 0, False),
    21: (0, -1, 0, False),
    22: (0, -0.6, 0, False),
    23: (0, -0.3, 0, False),
    24: (0, -0.1, 0, False),
    25: (1, 0, 0, False),
    26: (0, 0.1, 0, False),
    27: (0, 0.3, 0, False),
    28: (0, 0.6, 0, False),
    29: (0, 1.0, 0, False),
    30: (0.5, -0.5, 0, True),
    31: (0.5, -0.3, 0, True),
    32: (0.5, -0.2, 0, True),
    33: (0.5, -0.1, 0, True),
    34: (0.5, 0, 0, True),
    35: (0.5, 0.1, 0, True),
    36: (0.5, 0.2, 0, True),
    37: (0.5, 0.3, 0, True),
    38: (0.5, 0.5, 0, True),
}

@DATASETS.register_module()
class B2D3DDataset(Dataset):
    CLASSES = [
        'car',
        'van',
        'truck',
        'bicycle',
        'traffic_sign',
        'traffic_cone',
        'traffic_light',
        'pedestrian',
        'others',
    ]
    MAP_CLASSES = [
        'Broken',
        'Solid',
        'SolidSolid',
        'Center',
        'TrafficLight',
        'StopSign',
    ]
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        map_classes=None,
        name_mapping=None,
        load_interval=1,
        modality=None,
        sample_interval=5,
        past_frames=2,
        future_frames=6,
        test_mode=False,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        work_dir=None,
        eval_config=None,
        plan_config={},
        path_mode="lidar",
        vel_mode="ego",
        time_points=None,
    ):
        self.load_interval = load_interval
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0
        self.sample_interval = sample_interval
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.plan_config = plan_config
        self.path_mode = path_mode
        self.vel_mode = vel_mode
        self.time_points = time_points

        if classes is not None:
            self.CLASSES = classes
        if map_classes is not None: 
            self.map_classes = map_classes
        self.NameMapping = name_mapping
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        if with_seq_flag:
            self._set_sequence_group_flag()
        
        self.work_dir = work_dir
        self.eval_config = eval_config
        self.eval_cfg = {
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "mean_ap_weight": 5,
            "class_names":['car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian'],
            "tp_metrics":['trans_err', 'scale_err', 'orient_err', 'vel_err'],
            "err_name_maping":{'trans_err': 'mATE','scale_err': 'mASE','orient_err': 'mAOE','vel_err': 'mAVE','attr_err': 'mAAE'},
            "class_range":{'car':(50,50),'van':(50,50),'truck':(50,50),'bicycle':(40,40),'traffic_sign':(30,30),'traffic_cone':(30,30),'traffic_light':(30,30),'pedestrian':(40,40)}
        }

    def __len__(self):
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.sequences_split_num == -1:
            self.flag = np.arange(len(self.data_infos))
            return
        
        res = []

        curr_folder = self.data_infos[0]["folder"]
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx]["folder"] != curr_folder:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
                curr_folder = self.data_infos[idx]["folder"]
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data["index"] = idx
        data = self.pipeline(data)
        return data

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = data[:: self.load_interval]
        return data_infos
    
    def anno2geom(self, annos):
        map_geoms = {}
        for label, anno_list in annos.items():
            map_geoms[label] = []
            for anno in anno_list:
                geom = LineString(anno)
                map_geoms[label].append(geom)
        return map_geoms
    
    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            token=info['token'],
            timestamp=info['timestamp'] / 1e6,
            town_name=info["town_name"],
        )

        def similar_array(p1, p2):
            if p1.shape != p2.shape:
                return False
            if np.abs(p1-p2).max() < 0.1:
                return True
            if np.abs(p1-np.flip(p2, axis=0)).max() < 0.1:
                return True
            return False

        def not_too_short(p):
            if np.abs(p - p[0]).max() < 0.2:
                return False
            else:
                return True

        def deduplicate_arrays(arr_list):
            if not arr_list:
                return []
            
            unique_arrays = [arr_list[0]]
            for arr in arr_list[1:]:
                if not_too_short(arr) and (not any(similar_array(arr, u_arr) for u_arr in unique_arrays)):
                    unique_arrays.append(arr)
            return unique_arrays

        map_annos = {i: info["map_annos"][self.MAP_CLASSES.index(cls)] for i, cls in enumerate(self.map_classes)}
        map_annos = {i: deduplicate_arrays(anno) for i, anno in map_annos.items()}
        map_geoms = self.anno2geom(map_annos)
        input_dict["map_infos"] = map_annos
        input_dict["map_geoms"] = map_geoms

        if self.modality['use_camera']:
            image_paths = []
            depth_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsic = []
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            lidar2global =  self.invert_pose(info['sensors']['LIDAR_TOP']['world2lidar'])
            for sensor_type, cam_info in info['sensors'].items():
                if not 'CAM' in sensor_type:
                    continue
                img_path = osp.join(self.data_root,cam_info['data_path'])
                image_paths.append(img_path)
                depth_path = img_path.replace('rgb_','depth_').replace('.jpg','.png')
                depth_paths.append(depth_path)
                # obtain lidar to image transformation matrix
                cam2ego = cam_info['cam2ego']
                intrinsic = copy.deepcopy(cam_info["intrinsic"])
                cam_intrinsic.append(intrinsic)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2cam = (self.invert_pose(cam2ego) @ lidar2ego).T
                lidar2img = viewpad @ lidar2cam.T
                lidar2img_rts.append(lidar2img)
                lidar2cam_rts.append(lidar2cam)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    depth_filename=depth_paths,
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                    cam_intrinsic=cam_intrinsic,
                    lidar2global=lidar2global,
                )
            )

        annos = self.get_ann_info(index)
        input_dict.update(annos)

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        mask = (info['num_points'] != 0)
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        for i in range(len(gt_names_3d)):
            if gt_names_3d[i] in self.NameMapping.keys():
                gt_names_3d[i] = self.NameMapping[gt_names_3d[i]]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)        

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

        if "gt_ids" in info:
            instance_inds = np.array(info["gt_ids"], dtype=np.int64)[mask]
            anns_results["instance_inds"] = instance_inds

        gt_agent_fut_trajs, gt_agent_fut_masks = self.get_fut_agent(index, self.sample_interval, self.future_frames)
        anns_results['gt_agent_fut_trajs'] = gt_agent_fut_trajs[mask]
        anns_results['gt_agent_fut_masks'] = gt_agent_fut_masks[mask]

        (
            ego_his_trajs, 
            ego_fut_trajs, 
            ego_fut_masks, 
            command
        ) = self.get_ego_trajs(index,self.sample_interval,self.past_frames,self.future_frames)
        anns_results['gt_ego_fut_trajs'] = ego_fut_trajs
        anns_results['gt_ego_fut_masks'] = ego_fut_masks
        anns_results['gt_ego_fut_cmd'] = command

        if self.time_points is not None:
            gt_traj, gt_traj_mask = self.get_trajs(index, self.time_points)
            anns_results["gt_traj"] = gt_traj
            anns_results["gt_traj_mask"] = gt_traj_mask

        for key, value in self.plan_config.items():
            if key == "lat":
                num_pts = value["num_pts"]
                interval = value["interval"]
                spatial_waypoints, spatial_mask = self.get_future_waypoints(index, num_pts, interval, mode=self.path_mode)
                anns_results[f"gt_{key}"] = spatial_waypoints
                anns_results[f"gt_{key}_mask"] = spatial_mask
            elif key == "lon":
                vel_seq, vel_seq_mask = self.get_vel_seq(index, value["time_points"], mode=self.vel_mode)
                anns_results[f"gt_{key}"] = vel_seq
                anns_results[f"gt_{key}_mask"] = vel_seq_mask
            elif key == "traj":
                gt_traj, gt_traj_mask = self.get_trajs(index, value["time_points"])
                anns_results["gt_traj"] = gt_traj
                anns_results["gt_traj_mask"] = gt_traj_mask
            elif key == "collision":
                continue

        route_near, route_far = self.get_future_route(index)
        anns_results["route"] = route_near

        lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
        global2lidar =  info['sensors']['LIDAR_TOP']['world2lidar']
        ego2global = self.invert_pose(global2lidar) @ self.invert_pose(lidar2ego)
        ego_translation = ego2global[:3,3]

        yaw = info['ego_yaw']
        raw_theta = -(yaw-np.pi/2)
        theta_to_lidar = raw_theta
        rotation_matrix = np.array(
            [[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)], [np.sin(theta_to_lidar), np.cos(theta_to_lidar)]])

        command_near_xy = np.array(
            [info['command_near_xy'][0] - ego_translation[0], info['command_near_xy'][1] - ego_translation[1]])
        local_command_xy_near = rotation_matrix @ command_near_xy
        anns_results["tp_near"] = local_command_xy_near

        command_far_xy = np.array(
            [info['command_far_xy'][0] - ego_translation[0], info['command_far_xy'][1] - ego_translation[1]])
        local_command_xy_far = rotation_matrix @ command_far_xy
        anns_results["tp_far"] = local_command_xy_far

        ego_status = np.zeros(10)
        ego_status[:3] = info["ego_accel"]
        ego_status[3:6] = info["ego_rotation_rate"]
        ego_status[6:9] = info["ego_vel"]
        anns_results["ego_status"] = ego_status.astype(np.float32)
        anns_results["fut_boxes"] = [np.zeros((0, 7))] * 6
      
        return anns_results

    def get_future_waypoints(self, idx, num_pts=20, interval=1, mode="lidar"):
        # 初始化变量
        points = [np.array([0.0, 0.0])]  # 起点 (0,0)
        current_length = 0.0
        future_idx = 1  # 从当前帧的下一个帧开始
        last_point = np.array([0.0, 0.0])
        
        def get_ego_trans(info):
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            global2lidar =  info['sensors']['LIDAR_TOP']['world2lidar']
            ego2global = self.invert_pose(global2lidar) @ self.invert_pose(lidar2ego)
            g2e =lidar2ego @ global2lidar
            return g2e

        # 获取当前帧的世界坐标系到激光雷达坐标系的变换矩阵
        cur_frame = self.data_infos[idx]
        if mode == "ego":
            world2lidar_cur = get_ego_trans(cur_frame)
        else:
            world2lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        
        # 循环获取未来帧，直到路径长度达到20米或数据结束
        if interval == "mix":
            max_dis = 20
        else:
            max_dis = num_pts * interval
        while current_length < max_dis:
            # 计算下一帧的索引
            next_idx = idx + future_idx
            
            # 检查索引是否有效
            if next_idx < 0 or next_idx >= len(self.data_infos):
                break
                
            # 检查是否在同一文件夹/场景中
            next_frame = self.data_infos[next_idx]
            if next_frame['folder'] != cur_frame['folder']:
                break
            
            # 获取下一帧到当前帧的变换
            if mode == "ego":
                world2lidar_next = get_ego_trans(next_frame)
                next2cur_lidar = world2lidar_cur @ np.linalg.inv(world2lidar_next)
                xy = next2cur_lidar[[1,0], 3]
                xy[...,0] *= -1
            else:
                world2lidar_next = next_frame['sensors']['LIDAR_TOP']['world2lidar']
                next2cur_lidar = world2lidar_cur @ np.linalg.inv(world2lidar_next)
                xy = next2cur_lidar[0:2, 3]
            
            # 计算到上一个点的距离
            ## make sure forward
            xy[1] = max(last_point[1], xy[1])

            distance = np.linalg.norm(xy - last_point)
            last_point = xy
            
            # 添加到路径点
            points.append(xy)
            current_length += distance
            
            # 移动到下一帧
            future_idx += 1
        
        # 将点列表转换为numpy数组
        points = np.array(points)
        waypoints = self.equal_spacing_route(points, num_pts, interval)
        
        # 创建有效性掩码
        mask = np.ones(num_pts, dtype=np.float32)
        if interval == "mix":
            x = list(np.arange(1, 11, 1) * 0.5) + list(np.arange(6, 11, 1)) + list(np.arange(12, 21, 2))
            for i, dist in enumerate(x):
                if dist > current_length:
                    break
            if x[-1] > current_length:
                mask[i:] = 0
        else:
            valid_points = min(num_pts, int(np.floor(current_length / interval)))
            mask[valid_points:] = 0
        
        return waypoints, mask

    def get_trajs(self, idx, time_points, mode="ego"):
        def get_ego_trans(info):
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            global2lidar =  info['sensors']['LIDAR_TOP']['world2lidar']
            ego2global = self.invert_pose(global2lidar) @ self.invert_pose(lidar2ego)
            g2e =lidar2ego @ global2lidar
            return g2e

        # 获取当前帧的世界坐标系到激光雷达坐标系的变换矩阵
        cur_frame = self.data_infos[idx]
        if mode == "ego":
            world2lidar_cur = get_ego_trans(cur_frame)
        else:
            world2lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        

        last_point = np.array([0.0, 0.0])
        mask = np.zeros(len(time_points), dtype=np.float32)
        trajs = np.zeros((len(time_points), 2), dtype=np.float32)
        for i, time_point in enumerate(time_points):
            end_time = time_point[-1]
            next_idx = idx + int(end_time / 0.1)
      
            if next_idx < 0 or next_idx >= len(self.data_infos):
                break
            next_frame = self.data_infos[next_idx]
            if next_frame['folder'] != cur_frame['folder']:
                break
            
            # 获取下一帧到当前帧的变换
            if mode == "ego":
                world2lidar_next = get_ego_trans(next_frame)
                next2cur_lidar = world2lidar_cur @ np.linalg.inv(world2lidar_next)
                xy = next2cur_lidar[[1,0], 3]
                xy[...,0] *= -1
            else:
                world2lidar_next = next_frame['sensors']['LIDAR_TOP']['world2lidar']
                next2cur_lidar = world2lidar_cur @ np.linalg.inv(world2lidar_next)
                xy = next2cur_lidar[0:2, 3]
            
            ## make sure forward
            xy[1] = max(last_point[1], xy[1])
            last_point = xy
            trajs[i] = xy
            mask[i] = 1
            
        return trajs, mask

    def get_future_route(self, idx):
        info = self.data_infos[idx]
        lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
        global2lidar =  info['sensors']['LIDAR_TOP']['world2lidar']
        ego2global = self.invert_pose(global2lidar) @ self.invert_pose(lidar2ego)
        ego_translation = ego2global[:3,3]

        yaw = info['ego_yaw']
        raw_theta = -(yaw-np.pi/2)
        theta_to_lidar = raw_theta
        rotation_matrix = np.array(
            [[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)], [np.sin(theta_to_lidar), np.cos(theta_to_lidar)]])

        token = info["folder"]

        def target_point_to_local(target_point):
            target_point = np.array(
                [target_point[0] - ego_translation[0], target_point[1] - ego_translation[1]])
            target_point = rotation_matrix @ target_point
            return target_point
 
        route_near = []
        route_far = []
        
        while idx < len(self.data_infos):
            
            info_cur = self.data_infos[idx]
            if info_cur['folder'] != token:
                break
            
            tp_near = target_point_to_local(info_cur["command_near_xy"])
            route_near.append(tp_near)
            tp_far = target_point_to_local(info_cur["command_far_xy"])
            route_far.append(tp_far)

            idx += 1

        route_near = np.array(route_near)
        route_far = np.array(route_far)
        route_near = route_near[np.concatenate(([True], np.any(np.diff(route_near, axis=0) != 0, axis=1)))]

        return route_near, route_far
        

    def equal_spacing_route(self, points, num_pts=20, interval=1):
        route = np.concatenate((np.zeros_like(points[:1]),  points)) # Add 0 to front
        shift = np.roll(route, 1, axis=0) # Shift by 1
        shift[0] = shift[1] # Set wraparound value to 0

        dists = np.linalg.norm(route-shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists))*1e-4 # Prevents dists not being strictly increasing

        if interval == "mix":
            x = list(np.arange(1, 11, 1) * 0.5) + list(np.arange(6, 11, 1)) + list(np.arange(12, 21, 2))
        else:
            x = np.arange(1, (num_pts + 1), 1) * interval
        interp_points = np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T

        return interp_points

    def get_fut_agent(self, idx, sample_rate, frames):
        adj_idx_list = range(idx,idx+(frames+1)*sample_rate,sample_rate)
        cur_frame = self.data_infos[idx]
        cur_boxes = cur_frame['gt_boxes'].copy()
        box_ids = cur_frame['gt_ids']

        future_track = np.zeros((len(box_ids),frames+1,2))
        future_mask = np.zeros((len(box_ids),frames+1))
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            box_id = box_ids[i]
            cur_box2lidar = world2lidar_lidar_cur @ cur_frame['npc2world'][i]
            cur_xy = cur_box2lidar[0:2,3]
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx < 0 or adj_idx >= len(self.data_infos):
                    break
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame ['folder']:
                    break
                if len(np.where(adj_frame['gt_ids']==box_id)[0])==0:
                    break
                assert len(np.where(adj_frame['gt_ids']==box_id)[0]) == 1 , np.where(adj_frame['gt_ids']==box_id)[0]
                adj_idx = np.where(adj_frame['gt_ids']==box_id)[0][0]
                adj_box2lidar = world2lidar_lidar_cur @ adj_frame['npc2world'][adj_idx]
                adj_xy = adj_box2lidar[0:2,3]
                if j > 0:
                    last_xy = future_track[i,j-1,:]
                    distance = np.linalg.norm(last_xy - adj_xy)
                    if distance > 10:
                        break
                future_track[i,j,:] = adj_xy
                future_mask[i,j] = 1

        future_track_offset = future_track[:,1:,:] - future_track[:,:-1,:]
        future_mask_offset = future_mask[:,1:]
        future_track_offset[future_mask_offset==0] = 0

        return future_track_offset.astype(np.float32), future_mask_offset.astype(np.float32)

    def get_ego_trajs(self,idx,sample_rate,past_frames,future_frames):
        adj_idx_list = range(idx-past_frames*sample_rate,idx+(future_frames+1)*sample_rate,sample_rate)
        cur_frame = self.data_infos[idx]
        full_adj_track = np.zeros((past_frames+future_frames+1,2))
        full_adj_adj_mask = np.zeros(past_frames+future_frames+1)
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if adj_idx <0 :
                continue
            if adj_idx>=len(self.data_infos):
                break
            adj_frame = self.data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame['folder']:
                break
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2,3]
            full_adj_track[j,0:2] = xy
            full_adj_adj_mask[j] = 1
        offset_track = full_adj_track[1:] - full_adj_track[:-1]
        for j in range(past_frames-1,-1,-1):
            if full_adj_adj_mask[j] == 0:
                offset_track[j] = offset_track[j+1]
        for j in range(past_frames,past_frames+future_frames,1):

            if full_adj_adj_mask[j+1] == 0 :
                offset_track[j] = 0
        command = self.command2hot(cur_frame['command_near'])
        offset_track = offset_track.astype(np.float32)
        return offset_track[:past_frames].copy(), offset_track[past_frames:].copy(), full_adj_adj_mask[-future_frames:].copy(), command
    
    def command2hot(self,command,max_dim=6):
        if command < 0:
            command = 4
        command -= 1
        cmd_one_hot = np.zeros(max_dim)
        cmd_one_hot[command] = 1
        return cmd_one_hot

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose

    def get_vel_seq(self, index, time_points, mode="ego"):
        vel_seq = []
        mask = []
        cur_info = self.data_infos[index]
        for time_point in time_points:
            start, end = time_point
            frame_idxs = list(range(index + int(start / 0.1), index + int(end / 0.1) + 1))

            def check_in_range(cur_info, frame_idxs):
                for idx in frame_idxs:
                    if idx < 0 or idx >= len(self.data_infos):
                        return False
                    if self.data_infos[idx]["folder"] != cur_info["folder"]:
                        return False
                return True
            
            def get_ego_trans(info):
                if mode == "ego":
                    lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
                    global2lidar =  info['sensors']['LIDAR_TOP']['world2lidar']
                    ego2global = self.invert_pose(global2lidar) @ self.invert_pose(lidar2ego)
                    ego_translation = ego2global[:2,3]
                    return ego_translation
                if mode == "lidar":
                    global2lidar =  info['sensors']['LIDAR_TOP']['world2lidar']
                    lidar2global = self.invert_pose(global2lidar)
                    lidar_translation = lidar2global[:2, 3]
                    return lidar2global
            
            if not check_in_range(cur_info, frame_idxs):
                vel_seq.append(0)
                mask.append(0)
            else:
                if mode == "ego" or mode == "lidar":
                    locs = np.array([get_ego_trans(self.data_infos[i]) for i in frame_idxs])
                    offset = locs[1:] - locs[:-1]
                    dist = np.linalg.norm(offset, axis=-1).sum()
                    vel = dist / (end - start)
                    vel_seq.append(vel)
                    mask.append(1)
                if mode == "vel":
                    vels = [self.data_infos[i]["ego_vel"][0] for i in frame_idxs]
                    vels = [max(0, v) for v in vels]
                    vel = sum(vels) / len(vels)                   
                    vel_seq.append(vel)
                    mask.append(1)

        return np.array(vel_seq, dtype=np.float32), np.array(mask, dtype=np.float32)

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            box3d = det['boxes_3d']
            scores = det['scores_3d']
            labels = det['labels_3d']
            box_gravity_center = box3d[:, :3]
            box_dims = box3d[:, 3:6]
            box_yaw = box3d[:, 6]
            sample_token = self.data_infos[sample_id]['token']

            for i in range(len(box3d)):
                quat = list(Quaternion(axis=[0, 0, 1], radians=box_yaw[i]))
                velocity = [box3d[i, 7].item(),box3d[i, 8].item()]
                name = mapped_class_names[labels[i]]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box_gravity_center[i].tolist(),
                    size=box_dims[i].tolist(),
                    rotation=quat,
                    velocity=velocity,
                    detection_name=name,
                    detection_score=scores[i].item(),
                    attribute_name=name)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):

        with open(result_path) as f:
            result_data = json.load(f)
        pred_boxes = EvalBoxes.deserialize(result_data['results'], DetectionBox)
        meta = result_data['meta']

        gt_boxes = self.load_gt()

        # Build list of (class_name, dist_th) tasks and run accumulate in parallel
        tasks = [
            (gt_boxes, pred_boxes, cls, center_distance, dt)
            for cls in self.eval_cfg['class_names']
            for dt in self.eval_cfg['dist_ths']
        ]
        n_workers = min(12, len(tasks))  # 50% of 24 cores
        print(f'Running {len(tasks)} accumulate tasks with {n_workers} workers...')
        with Pool(n_workers) as pool:
            results = pool.starmap(accumulate, tasks)

        metric_data_list = DetectionMetricDataList()
        for (gt_boxes_, pred_boxes_, cls, _dist_fn, dt), md in zip(tasks, results):
            metric_data_list.set(cls, dt, md)
        metrics = DetectionMetrics(self.eval_cfg)

        for class_name in self.eval_cfg['class_names']:
            # Compute APs.
            for dist_th in self.eval_cfg['dist_ths']:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.eval_cfg['min_recall'], self.eval_cfg['min_precision'])
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in self.eval_cfg['tp_metrics']:
                metric_data = metric_data_list[(class_name, self.eval_cfg['dist_th_tp'])]
                tp = calc_tp(metric_data, self.eval_cfg['min_recall'], metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = meta.copy()
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        #print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err']))        

        detail = dict()
        metric_prefix = 'bbox_NuScenes'
        for name in self.eval_cfg['class_names']:
            for k, v in metrics_summary['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,self.eval_cfg['err_name_maping'][k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics_summary['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics_summary['mean_ap']


        return detail

    def load_gt(self):
        all_annotations = EvalBoxes()
        for i in range(len(self.data_infos)):
            sample_boxes = []
            sample_data = self.data_infos[i]

            gt_boxes = sample_data['gt_boxes']
            
            for j in range(gt_boxes.shape[0]):
                class_name = self.NameMapping[sample_data['gt_names'][j]]
                if not class_name in self.eval_cfg['class_range'].keys():
                    continue
                range_x, range_y = self.eval_cfg['class_range'][class_name]
                if abs(gt_boxes[j,0]) > range_x or abs(gt_boxes[j,1]) > range_y:
                    continue
                sample_boxes.append(DetectionBox(
                    sample_token=sample_data['token'],
                    translation=gt_boxes[j,0:3],
                    size=gt_boxes[j,3:6],
                    rotation=list(Quaternion(axis=[0, 0, 1], radians=gt_boxes[j,6])),
                    velocity=gt_boxes[j,7:9],
                    num_pts=int(sample_data['num_points'][j]),
                    detection_name=self.NameMapping[sample_data['gt_names'][j]],
                    detection_score=-1.0,  
                    attribute_name=self.NameMapping[sample_data['gt_names'][j]]
                ))
            all_annotations.add_boxes(sample_data['token'], sample_boxes)
        return all_annotations

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = jsonfile_prefix
                result_files.update(
                    {
                        name: self._format_bbox(
                            results_, tmp_file_, tracking=tracking
                        )
                    }
                )
        return result_files, tmp_dir

    def format_map_results(self, results, prefix=None):
        submissions = {'results': {},}
        
        for j, pred in enumerate(results):
            '''
            For each case, the result should be formatted as Dict{'vectors': [], 'scores': [], 'labels': []}
            'vectors': List of vector, each vector is a array([[x1, y1], [x2, y2] ...]),
                contain all vectors predicted in this sample.
            'scores: List of score(float), 
                contain scores of all instances in this sample.
            'labels': List of label(int), 
                contain labels of all instances in this sample.
            '''
            if pred is None: # empty prediction
                continue
            pred = pred['img_bbox']

            single_case = {'vectors': [], 'scores': [], 'labels': []}
            token = self.data_infos[j]['token']
            for i in range(len(pred['scores'])):
                score = pred['scores'][i]
                label = pred['labels'][i]
                vector = pred['vectors'][i]

                # A line should have >=2 points
                if len(vector) < 2:
                    continue
                
                single_case['vectors'].append(vector)
                single_case['scores'].append(score)
                single_case['labels'].append(label)
            
            submissions['results'][token] = single_case
        
        out_path = osp.join(prefix, 'submission_vector.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def format_motion_results(self, results, jsonfile_prefix=None, tracking=False, thresh=None):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det['img_bbox'], threshold=None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
                filter_with_cls_range=False,
            )
            for i, box in enumerate(boxes):
                if thresh is not None and box.score < thresh:
                    continue
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = B2D3DDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = B2D3DDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )
                nusc_anno.update(
                    dict(
                        trajs=det['img_bbox']['trajs_3d'][i].numpy(),
                    )
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        return nusc_submissions 

    def _evaluate_single_motion(self,
                         results,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from .evaluation.motion.motion_eval_uniad import NuScenesEval as NuScenesEvalMotion

        output_dir = result_path
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEvalMotion(
            nusc,
            config=copy.deepcopy(self.det3d_eval_configs),
            result_path=results,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            seconds=6)
        metrics = nusc_eval.main(render_curves=False)
        
        MOTION_METRICS = ['EPA', 'min_ade_err', 'min_fde_err', 'miss_rate_err']
        class_names = ['car', 'pedestrian']

        table = prettytable.PrettyTable()
        table.field_names = ["class names"] + MOTION_METRICS
        for class_name in class_names:
            row_data = [class_name]
            for m in MOTION_METRICS:
                row_data.append('%.4f' % metrics[f'{class_name}_{m}'])
            table.add_row(row_data)
        print_log('\n'+str(table), logger=logger)
        return metrics

    def evaluate(
        self,
        results,
        eval_mode,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        res_path = "results.pkl"
        res_path = osp.join(self.work_dir, res_path)
        print('All Results write to', res_path)
        mmcv.dump(results, res_path)

        results_dict = dict()
        if eval_mode['with_det']:
            self.tracking = eval_mode["with_tracking"]
            self.tracking_threshold = eval_mode["tracking_threshold"]
            for metric in ["detection", "tracking"]:
                tracking = metric == "tracking"
                if tracking and not self.tracking:
                    continue
                result_files, tmp_dir = self.format_results(
                    results, jsonfile_prefix=self.work_dir, tracking=tracking
                )

                if isinstance(result_files, dict):
                    for name in result_names:
                        ret_dict = self._evaluate_single(
                            result_files[name], tracking=tracking
                        )
                    results_dict.update(ret_dict)
                elif isinstance(result_files, str):
                    ret_dict = self._evaluate_single(
                        result_files, tracking=tracking
                    )
                    results_dict.update(ret_dict)

                if tmp_dir is not None:
                    tmp_dir.cleanup()

        if eval_mode['with_map']:
            from .evaluation.map.vector_eval import VectorEvaluate
            self.map_evaluator = VectorEvaluate(self.eval_config)
            result_path = self.format_map_results(results, prefix=self.work_dir)
            map_results_dict = self.map_evaluator.evaluate(result_path, logger=logger)
            results_dict.update(map_results_dict)

        if eval_mode['with_motion']:
            thresh = eval_mode["motion_threshhold"]
            result_files = self.format_motion_results(results, jsonfile_prefix=self.work_dir, thresh=thresh)
            motion_results_dict = self._evaluate_single_motion(result_files, self.work_dir, logger=logger)
            results_dict.update(motion_results_dict)
        
        if eval_mode['with_planning']:
            from .evaluation.planning.planning_eval import planning_eval
            planning_results_dict = planning_eval(results, self.eval_config, logger=logger)
            results_dict.update(planning_results_dict)

        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)
        
        # print main metrics for recording
        metric_str = '\n'
        if "bbox_NuScenes/NDS" in results_dict:
            metric_str += f'mAP: {results_dict.get("bbox_NuScenes/mAP"):.4f}\n'
            metric_str += f'mATE: {results_dict.get("bbox_NuScenes/mATE"):.4f}\n'
            metric_str += f'mASE: {results_dict.get("bbox_NuScenes/mASE"):.4f}\n'
            metric_str += f'mAOE: {results_dict.get("bbox_NuScenes/mAOE"):.4f}\n' 
            metric_str += f'mAVE: {results_dict.get("bbox_NuScenes/mAVE"):.4f}\n' 
            # metric_str += f'mAAE: {results_dict.get("bbox_NuScenes/mAAE"):.4f}\n' 
            metric_str += f'NDS: {results_dict.get("bbox_NuScenes/NDS"):.4f}\n\n'
        
        if "img_bbox_NuScenes/amota" in results_dict:
            metric_str += f'AMOTA: {results_dict["img_bbox_NuScenes/amota"]:.4f}\n' 
            metric_str += f'AMOTP: {results_dict["img_bbox_NuScenes/amotp"]:.4f}\n' 
            metric_str += f'RECALL: {results_dict["img_bbox_NuScenes/recall"]:.4f}\n' 
            metric_str += f'MOTAR: {results_dict["img_bbox_NuScenes/motar"]:.4f}\n' 
            metric_str += f'MOTA: {results_dict["img_bbox_NuScenes/mota"]:.4f}\n' 
            metric_str += f'MOTP: {results_dict["img_bbox_NuScenes/motp"]:.4f}\n' 
            metric_str += f'IDS: {results_dict["img_bbox_NuScenes/ids"]}\n\n' 

        if "mAP_normal" in results_dict:
            # metric_str += f'ped_crossing= {results_dict["ped_crossing"]:.4f}\n' 
            # metric_str += f'divider= {results_dict["divider"]:.4f}\n' 
            # metric_str += f'boundary= {results_dict["boundary"]:.4f}\n' 
            metric_str += f'mAP_normal= {results_dict["mAP_normal"]:.4f}\n\n' 

        if "car_EPA" in results_dict:
            metric_str += f'Car / Ped\n' 
            metric_str += f'epa= {results_dict["car_EPA"]:.4f} / {results_dict["pedestrian_EPA"]:.4f}\n'
            metric_str += f'ade= {results_dict["car_min_ade_err"]:.4f} / {results_dict["pedestrian_min_ade_err"]:.4f}\n'
            metric_str += f'fde= {results_dict["car_min_fde_err"]:.4f} / {results_dict["pedestrian_min_fde_err"]:.4f}\n'
            metric_str += f'mr= {results_dict["car_miss_rate_err"]:.4f} / {results_dict["pedestrian_miss_rate_err"]:.4f}\n\n' 

        if "L2" in results_dict:
            metric_str += f'obj_box_col: {(results_dict["obj_box_col"]*100):.3f}%\n'
            metric_str += f'L2: {results_dict["L2"]:.4f}\n\n'
        
        print_log(metric_str, logger=logger)
        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        save_dir = "./" if save_dir is None else save_dir
        save_dir = os.path.join(save_dir, "visual")
        print_log(os.path.abspath(save_dir))
        pipeline = Compose(pipeline)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = None

        for i, result in enumerate(results):
            if "img_bbox" in result.keys():
                result = result["img_bbox"]
            data_info = pipeline(self.get_data_info(i))
            imgs = []

            raw_imgs = data_info["img"]
            lidar2img = data_info["img_metas"].data["lidar2img"]
            pred_bboxes_3d = result["boxes_3d"][
                result["scores_3d"] > self.vis_score_threshold
            ]
            if "instance_ids" in result and self.tracking:
                color = []
                for id in result["instance_ids"].cpu().numpy().tolist():
                    color.append(
                        self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))]
                    )
            elif "labels_3d" in result:
                color = []
                for id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[id])
            else:
                color = (255, 0, 0)

            # ===== draw boxes_3d to images =====
            for j, img_origin in enumerate(raw_imgs):
                img = img_origin.copy()
                if len(pred_bboxes_3d) != 0:
                    img = draw_lidar_bbox3d_on_img(
                        pred_bboxes_3d,
                        img,
                        lidar2img[j],
                        img_metas=None,
                        color=color,
                        thickness=3,
                    )
                imgs.append(img)

            # ===== draw boxes_3d to BEV =====
            bev = draw_lidar_bbox3d_on_bev(
                pred_bboxes_3d,
                bev_size=img.shape[0] * 2,
                color=color,
            )

            # ===== put text and concat =====
            for j, name in enumerate(
                [
                    "front",
                    "front right",
                    "front left",
                    "rear",
                    "rear left",
                    "rear right",
                ]
            ):
                imgs[j] = cv2.rectangle(
                    imgs[j],
                    (0, 0),
                    (440, 80),
                    color=(255, 255, 255),
                    thickness=-1,
                )
                w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                text_x = int(220 - w / 2)
                text_y = int(40 + h / 2)

                imgs[j] = cv2.putText(
                    imgs[j],
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            image = np.concatenate(
                [
                    np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                    np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                ],
                axis=0,
            )
            image = np.concatenate([image, bev], axis=1)

            # ===== save video =====
            if videoWriter is None:
                videoWriter = cv2.VideoWriter(
                    os.path.join(save_dir, "video.avi"),
                    fourcc,
                    7,
                    image.shape[:2][::-1],
                )
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
            videoWriter.write(image)
        videoWriter.release()


def get_T_global(info):
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = pyquaternion.Quaternion(
        info["lidar2ego_rotation"]
    ).rotation_matrix
    lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
    ego2global = np.eye(4)
    ego2global[:3, :3] = pyquaternion.Quaternion(
        info["ego2global_rotation"]
    ).rotation_matrix
    ego2global[:3, 3] = np.array(info["ego2global_translation"])
    return ego2global @ lidar2ego