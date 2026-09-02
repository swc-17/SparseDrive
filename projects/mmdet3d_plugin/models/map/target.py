import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import (BBOX_SAMPLERS, BBOX_ASSIGNERS)
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core import (build_assigner, build_sampler)
from mmdet.core.bbox.assigners import (AssignResult, BaseAssigner)

from ..base_target import BaseTargetWithDenoising


@BBOX_SAMPLERS.register_module()
class SparsePoint3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        assigner=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        dn_size_scale=0.2,
        dn_rot_angle=10,
        dn_trans_scale=1.0,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
        noise_type=[],
        dn_combination="rand",
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60),
    ):
        super(SparsePoint3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.assigner = build_assigner(assigner)
        self.dn_noise_scale = dn_noise_scale
        self.dn_trans_scale = dn_trans_scale
        self.dn_size_scale = dn_size_scale
        self.dn_rot_angle = dn_rot_angle
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn
        self.noise_type = noise_type
        self.dn_combination = dn_combination

        self.num_cls = num_cls
        self.num_sample = num_sample
        self.roi_size = roi_size

    def sample(
        self,
        cls_preds,
        pts_preds,
        cls_targets,
        pts_targets,
    ):
        pts_targets  = [x.flatten(2, 3) if len(x.shape)==4 else x for x in pts_targets]
        indices = []
        for(cls_pred, pts_pred, cls_target, pts_target) in zip(
            cls_preds, pts_preds, cls_targets, pts_targets
        ):
            # normalize to (0, 1)
            pts_pred = self.normalize_line(pts_pred)
            pts_target = self.normalize_line(pts_target)
            preds=dict(lines=pts_pred, scores=cls_pred)
            gts=dict(lines=pts_target, labels=cls_target)
            indice = self.assigner.assign(preds, gts)
            indices.append(indice)
        
        bs, num_pred, num_cls = cls_preds.shape
        output_cls_target = cls_targets[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        output_pts_target = pts_preds.new_zeros(pts_preds.shape)
        output_reg_weights = pts_preds.new_zeros(pts_preds.shape)
        for i, (pred_idx, target_idx, gt_permute_index) in enumerate(indices):
            if len(cls_targets[i]) == 0:
                continue
            permute_idx = gt_permute_index[pred_idx, target_idx]
            output_cls_target[i, pred_idx] = cls_targets[i][target_idx]
            output_pts_target[i, pred_idx] = pts_targets[i][target_idx, permute_idx]
            output_reg_weights[i, pred_idx] = 1

        return output_cls_target, output_pts_target, output_reg_weights

    def normalize_line(self, line):
        if line.shape[0] == 0:
            return line
        
        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        
        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
        line = line - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
        line = line / norm
        line = line.flatten(-2, -1)

        return line

    def get_dn_anchors(self, cls_target, pts_target, gt_instance_id=None):
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None
        
        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            pts_target = [x[: self.max_dn_gt] for x in pts_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]
        
        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None
        
        cls_target = torch.stack([
            F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1) for x in cls_target
        ])
        pts_target_permute = torch.stack([
            F.pad(x.flatten(-2), (0, 0, 0, 0, 0, max_dn_gt - x.shape[0])) 
            for x in pts_target
        ])
        pts_target = pts_target_permute[:, :, 0]
        pts_target = torch.where(
            cls_target[..., None] == -1, pts_target.new_tensor(0), pts_target
        )
        
        if gt_instance_id is not None:
            gt_instance_id = torch.stack([
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1) for x in gt_instance_id
            ])
        
        bs, num_gt, state_dims = pts_target.shape
        
        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            pts_target = pts_target.tile(self.num_dn_groups, 1, 1)
            pts_target_permute = pts_target_permute.tile(self.num_dn_groups, 1, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)
        
        dn_anchor = pts_target.clone()
        
        if self.dn_combination == "all":
            noise_types = self.noise_type
        elif self.dn_combination == "rand":
            noise_types = [np.random.choice(self.noise_type)]

        for noise_type in noise_types:
            if noise_type == 0:  # point noise
                noise = torch.rand_like(dn_anchor) * 2 - 1
                noise *= dn_anchor.new_tensor(self.dn_noise_scale).tile(self.num_sample)
                dn_anchor += noise
                
            elif noise_type == 1:  # location noise
                noise = torch.rand_like(dn_anchor[..., :2]) * 2 - 1
                noise *= dn_anchor.new_tensor(self.dn_trans_scale)
                dn_anchor += noise.tile(1, 1, self.num_sample)
                
            elif noise_type == 2:  # scale noise
                noise = torch.rand_like(dn_anchor[..., :2]) * 2 - 1 
                noise = noise * self.dn_size_scale + 1
                origin = dn_anchor.unflatten(-1, (-1, 2)).mean(dim=-2).tile(1, 1, self.num_sample)
                dn_anchor = (dn_anchor - origin) * noise.tile(1, 1, self.num_sample) + origin
                
            elif noise_type == 3:  # rotation noise
                noise = torch.rand_like(dn_anchor[..., 0]) * 2 - 1 
                noise = noise * self.dn_rot_angle * np.pi / 180
                pts_ = dn_anchor.unflatten(-1, (self.num_sample, 2))
                origin = pts_.mean(dim=-2).unsqueeze(-2)
                rot_mat = torch.stack([
                    torch.stack([torch.cos(noise), torch.sin(noise)]),
                    torch.stack([-torch.sin(noise), torch.cos(noise)]),
                ]).permute(2, 3, 0, 1)
                dn_anchor = rot_mat.unsqueeze(2) @ (pts_ - origin).unsqueeze(-1)
                dn_anchor = (dn_anchor.squeeze(-1) + origin).flatten(-2)
        
        if self.add_neg_dn:
            neg_anchor = pts_target.clone()
            for noise_type in noise_types:
                if noise_type == 0:  # point noise
                    noise_neg = torch.rand_like(neg_anchor) + 1
                    flag = torch.where(
                        torch.rand_like(neg_anchor) > 0.5,
                        neg_anchor.new_tensor(1),
                        neg_anchor.new_tensor(-1),
                    )
                    noise_neg *= flag
                    noise_neg *= neg_anchor.new_tensor(self.dn_noise_scale).tile(self.num_sample)
                    neg_anchor += noise_neg
                    
                elif noise_type == 1:  # location noise
                    noise_neg = torch.rand_like(neg_anchor[..., :2]) + 1
                    flag = torch.where(
                        torch.rand_like(neg_anchor[..., :2]) > 0.5,
                        neg_anchor.new_tensor(1),
                        neg_anchor.new_tensor(-1),
                    )
                    noise_neg *= flag
                    noise_neg *= neg_anchor.new_tensor(self.dn_trans_scale)
                    neg_anchor += noise_neg.tile(1, 1, self.num_sample)
                    
                elif noise_type == 2:  # scale noise
                    noise_neg = torch.rand_like(neg_anchor[..., :2]) + 1
                    flag = torch.where(
                        torch.rand_like(neg_anchor[..., :2]) > 0.5,
                        neg_anchor.new_tensor(1),
                        neg_anchor.new_tensor(-1),
                    )
                    noise_neg *= flag
                    noise_neg = noise_neg * self.dn_size_scale + 1
                    origin = neg_anchor.unflatten(-1, (-1, 2)).mean(dim=-2).tile(1, 1, self.num_sample)
                    neg_anchor = (neg_anchor - origin) * noise_neg.tile(1, 1, self.num_sample) + origin
                    
                elif noise_type == 3:  # rotation noise
                    noise_neg = torch.rand_like(neg_anchor[..., 0]) * 2 + 1 
                    noise_neg = noise_neg * self.dn_rot_angle * np.pi / 180
                    flag = torch.where(
                        torch.rand_like(neg_anchor[..., 0]) > 0.5,
                        neg_anchor.new_tensor(1),
                        neg_anchor.new_tensor(-1),
                    )
                    noise_neg *= flag
                    pts_ = neg_anchor.unflatten(-1, (self.num_sample, 2))
                    origin = pts_.mean(dim=-2).unsqueeze(-2)
                    rot_mat = torch.stack([
                        torch.stack([torch.cos(noise_neg), torch.sin(noise_neg)]),
                        torch.stack([-torch.sin(noise_neg), torch.cos(noise_neg)]),
                    ]).permute(2, 3, 0, 1)
                    neg_anchor = rot_mat.unsqueeze(2) @ (pts_ - origin).unsqueeze(-1)
                    neg_anchor = (neg_anchor.squeeze(-1) + origin).flatten(-2)
            
            dn_anchor = torch.cat([dn_anchor, neg_anchor], dim=1)
            num_gt *= 2
        
        dn_pts_target = torch.zeros_like(dn_anchor)
        dn_cls_target = -torch.ones_like(cls_target) * 3
        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id)
        
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)
        
        cls_pred = dn_anchor.new_ones([dn_anchor.shape[1], self.num_cls])
        for i in range(dn_anchor.shape[0]):
            line_pred = self.normalize_line(dn_anchor[i])
            line_target = self.normalize_line(pts_target_permute[i])
            preds = dict(lines=line_pred, scores=cls_pred)
            gts = dict(lines=line_target, labels=cls_target[i])
            indice = self.assigner.assign(preds, gts, ignore_cls_cost=True)
            anchor_idx, gt_idx, gt_permute_index = indice
            permute_idx = gt_permute_index[anchor_idx, gt_idx]
            dn_pts_target[i, anchor_idx] = pts_target_permute[i, gt_idx, permute_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
            if gt_instance_id is not None:
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]
        
        # from tools.visualization.bev_render import (
        #     color_mapping, 
        # )
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 1, figsize=(30, 60))
        # for i in range(32):
        #     db = dn_pts_target[0, i].reshape(-1, 2).cpu().numpy()
        #     plt.plot(db[:,0], db[:,1], color=color_mapping[i], linewidth=3, marker='o', linestyle='-', markersize=7)
        #     da = dn_anchor[0, i].reshape(-1, 2).cpu().numpy()
        #     plt.plot(da[:,0], da[:,1], color=color_mapping[i], linewidth=3, marker='v', linestyle='-', markersize=7)
        # plt.savefig('dn')
        # import ipdb; ipdb.set_trace()
        # fig, axes = plt.subplots(1, 1, figsize=(30, 60))
        # for i in range(16,32):
        #     db = dn_pts_target[0, i].reshape(-1, 2).cpu().numpy()
        #     plt.plot(db[:,0], db[:,1], color=color_mapping[i], linewidth=3, marker='o', linestyle='-', markersize=7)
        #     da = dn_anchor[0, i].reshape(-1, 2).cpu().numpy()
        #     plt.plot(da[:,0], da[:,1], color=color_mapping[i], linewidth=3, marker='v', linestyle='-', markersize=7)
        # plt.savefig('dn_neg')


        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_pts_target = (
            dn_pts_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        )
        if gt_instance_id is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
        else:
            dn_id_target = None
        
        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )
        
        attn_mask = dn_pts_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        
        return (
            dn_anchor,
            dn_pts_target,
            dn_cls_target.long(),
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def get_dn_anchors_single(self, cls_target, pts_target, gt_instance_id=None):
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None

        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            pts_target = [x[: self.max_dn_gt] for x in pts_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None
        cls_target = torch.stack(
            [
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                for x in cls_target
            ]
        )
        pts_target_permute = torch.stack(
            [F.pad(x.flatten(-2), (0, 0, 0, 0, 0, max_dn_gt - x.shape[0])) for x in pts_target]
        )
        pts_target = pts_target_permute[:, :, 0]
        pts_target = torch.where(
            cls_target[..., None] == -1, pts_target.new_tensor(0), pts_target
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )

        bs, num_gt, state_dims = pts_target.shape

        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            pts_target = pts_target.tile(self.num_dn_groups, 1, 1)
            pts_target_permute = pts_target_permute.tile(self.num_dn_groups, 1, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)

        if self.noise_type == 0: ## add noise to each point
            noise = torch.rand_like(pts_target) * 2 - 1
            noise *= pts_target.new_tensor(self.dn_noise_scale).tile(self.num_sample)
            dn_anchor = pts_target + noise
            if self.add_neg_dn:
                noise_neg = torch.rand_like(pts_target) + 1
                flag = torch.where(
                    torch.rand_like(pts_target) > 0.5,
                    noise_neg.new_tensor(1),
                    noise_neg.new_tensor(-1),
                )
                noise_neg *= flag
                noise_neg *= pts_target.new_tensor(self.dn_noise_scale).tile(self.num_sample)
                dn_anchor = torch.cat([dn_anchor, pts_target + noise_neg], dim=1)
                num_gt *= 2
        elif self.noise_type == 1: # location_noise
            noise = torch.rand_like(pts_target)[..., :2] * 2 - 1
            noise *= pts_target.new_tensor(self.dn_noise_scale)
            dn_anchor = pts_target + noise.tile(1, 1, self.num_sample)
            if self.add_neg_dn:
                noise_neg = torch.rand_like(pts_target[..., :2]) + 1
                flag = torch.where(
                    torch.rand_like(pts_target[..., :2]) > 0.5,
                    noise_neg.new_tensor(1),
                    noise_neg.new_tensor(-1),
                )
                noise_neg *= flag
                noise_neg *= pts_target.new_tensor(self.dn_noise_scale)
                dn_anchor = torch.cat([dn_anchor, pts_target + noise_neg.tile(self.num_sample)], dim=1)
                num_gt *= 2
        elif self.noise_type == 2: # scale_noise
            noise = torch.rand_like(pts_target)[..., :2] * 2 - 1 
            noise = noise * self.dn_size_scale + 1
            origin = pts_target.unflatten(-1, (-1, 2)).mean(dim=-2).tile(1, 1, self.num_sample)
            dn_anchor = (pts_target - origin) * noise.tile(1, 1, self.num_sample) + origin
            if self.add_neg_dn:
                noise_neg = torch.rand_like(pts_target[..., :2]) + 1
                flag = torch.where(
                    torch.rand_like(pts_target[..., :2]) > 0.5,
                    noise_neg.new_tensor(1),
                    noise_neg.new_tensor(-1),
                )
                noise_neg *= flag
                noise_neg = noise_neg * self.dn_size_scale + 1
                dn_anchor = torch.cat([dn_anchor, (pts_target - origin) * noise_neg.tile(1, 1, self.num_sample) + origin], dim=1)
                num_gt *= 2
        elif self.noise_type == 3: # rot_noise
            noise = torch.rand_like(pts_target)[..., 0] * 2 - 1 
            noise = noise * self.dn_rot_angle * np.pi / 180
            pts_target_ = pts_target.unflatten(-1, (self.num_sample, 2))
            origin = pts_target_.mean(dim=-2).unsqueeze(-2)
            rot_mat = torch.stack(
                [
                    torch.stack([torch.cos(noise), torch.sin(noise)]),
                    torch.stack([-torch.sin(noise), torch.cos(noise)]),
                ]
            ).permute(2, 3, 0, 1)
            dn_anchor = rot_mat.unsqueeze(2) @ (pts_target_ - origin).unsqueeze(-1)
            dn_anchor = (dn_anchor.squeeze(-1) + origin).flatten(-2)
            if self.add_neg_dn:
                noise_neg = torch.rand_like(pts_target)[..., 0] * 2 + 1 
                noise_neg = noise_neg * self.dn_rot_angle * np.pi / 180
                flag = torch.where(
                    torch.rand_like(pts_target[..., 0]) > 0.5,
                    noise_neg.new_tensor(1),
                    noise_neg.new_tensor(-1),
                )
                noise_neg *= flag
                rot_mat = torch.stack(
                    [
                        torch.stack([torch.cos(noise_neg), torch.sin(noise_neg)]),
                        torch.stack([-torch.sin(noise_neg), torch.cos(noise_neg)]),
                    ]
                ).permute(2, 3, 0, 1)
                dn_anchor_neg = rot_mat.unsqueeze(2) @ (pts_target_ - origin).unsqueeze(-1)
                dn_anchor_neg = (dn_anchor_neg.squeeze(-1) + origin).flatten(-2)
                dn_anchor = torch.cat([dn_anchor, dn_anchor_neg], dim=1)
                num_gt *= 2

        dn_pts_target = torch.zeros_like(dn_anchor)
        dn_cls_target = -torch.ones_like(cls_target) * 3
        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id)
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        cls_pred = dn_anchor.new_ones([dn_anchor.shape[1], self.num_cls])
        for i in range(dn_anchor.shape[0]):
            line_pred = self.normalize_line(dn_anchor[i])
            line_target = self.normalize_line(pts_target_permute[i])
            preds=dict(lines=line_pred, scores=cls_pred)
            gts=dict(lines=line_target, labels=cls_target[i])
            indice = self.assigner.assign(preds, gts, ignore_cls_cost=True)

            anchor_idx, gt_idx, gt_permute_index = indice
            permute_idx = gt_permute_index[anchor_idx, gt_idx]
            dn_pts_target[i, anchor_idx] = pts_target_permute[i, gt_idx, permute_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
            if gt_instance_id is not None:
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]

        # from tools.visualization.bev_render import (
        #     color_mapping, 
        # )
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 1, figsize=(30, 60))
        # for i in range(32):
        #     db = dn_pts_target[0, i].reshape(-1, 2).cpu().numpy()
        #     plt.plot(db[:,0], db[:,1], color=color_mapping[i], linewidth=3, marker='o', linestyle='-', markersize=7)
        #     da = dn_anchor[0, i].reshape(-1, 2).cpu().numpy()
        #     plt.plot(da[:,0], da[:,1], color=color_mapping[i], linewidth=3, marker='v', linestyle='-', markersize=7)
        # plt.savefig('dn_pos')
        # import ipdb; ipdb.set_trace()
        # fig, axes = plt.subplots(1, 1, figsize=(30, 60))
        # for i in range(16,32):
        #     db = dn_pts_target[0, i].reshape(-1, 2).cpu().numpy()
        #     plt.plot(db[:,0], db[:,1], color=color_mapping[i], linewidth=3, marker='o', linestyle='-', markersize=7)
        #     da = dn_anchor[0, i].reshape(-1, 2).cpu().numpy()
        #     plt.plot(da[:,0], da[:,1], color=color_mapping[i], linewidth=3, marker='v', linestyle='-', markersize=7)
        # plt.savefig('dn_neg')

        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_pts_target = (
            dn_pts_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        )
        if gt_instance_id is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
        else:
            dn_id_target = None
        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )  # valid denotes the items is not from pad.
        attn_mask = dn_pts_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        dn_cls_target = dn_cls_target.long()
        return (
            dn_anchor,
            dn_pts_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def update_dn(
        self,
        instance_feature,
        anchor,
        dn_reg_target,
        dn_cls_target,
        valid_mask,
        dn_id_target,
        num_noraml_anchor,
        temporal_valid_mask,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_metas = None
        if self.dn_metas is None or num_noraml_anchor >= num_anchor:
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # split instance_feature and anchor into non-dn and dn
        num_dn = num_anchor - num_noraml_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, :num_noraml_anchor]
        anchor = anchor[:, :num_noraml_anchor]

        # reshape all dn metas from (bs,num_all_dn,xxx)
        # to (bs, dn_group, num_dn_per_group, xxx)
        num_dn_groups = self.num_dn_groups
        num_dn = num_dn // num_dn_groups
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1)
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1)
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)
        if dn_id_target is not None:
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)

        # update temp_dn_metas by instance_id
        temp_dn_feat = self.dn_metas["dn_instance_feature"]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        temp_dn_id = self.dn_metas["dn_id_target"]

        # bs, num_temp_dn_groups, num_temp_dn, num_dn
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None]
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(dim=3)
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            self.dn_metas["dn_cls_target"].new_tensor(-1),
            self.dn_metas["dn_cls_target"],
        )
        temp_valid_mask = self.dn_metas["valid_mask"]
        temp_dn_anchor = self.dn_metas["dn_anchor"]

        # handle the misalignment the length of temp_dn to dn caused by the
        # change of num_gt, then concat the temp_dn and dn
        temp_dn_metas = [
            temp_dn_feat,
            temp_dn_anchor,
            temp_reg_target,
            temp_cls_target,
            temp_valid_mask,
            temp_dn_id,
        ]
        dn_metas = [
            dn_feat,
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            valid_mask,
            dn_id,
        ]
        output = []
        for i, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)):
            if num_temp_dn < num_dn:
                pad = (0, num_dn - num_temp_dn)
                if temp_meta.dim() == 4:
                    pad = (0, 0) + pad
                else:
                    assert temp_meta.dim() == 3
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                temp_meta = temp_meta[:, :, :num_dn]
            mask = temporal_valid_mask[:, None, None]
            if meta.dim() == 4:
                mask = mask.unsqueeze(dim=-1)
            temp_meta = torch.where(
                mask, temp_meta, meta[:, :num_temp_dn_groups]
            )
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)
            meta = meta.flatten(1, 2)
            output.append(meta)
        output[0] = torch.cat([instance_feature, output[0]], dim=1)
        output[1] = torch.cat([anchor, output[1]], dim=1)
        return output

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        if self.num_temp_dn_groups <= 0:
            return
        num_dn_groups = self.num_dn_groups
        bs, num_dn = dn_instance_feature.shape[:2]
        num_temp_dn = num_dn // num_dn_groups
        temp_group_mask = (
            torch.randperm(num_dn_groups) < self.num_temp_dn_groups
        )
        temp_group_mask = temp_group_mask.to(device=dn_anchor.device)
        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_anchor = dn_anchor.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(
                bs, num_dn_groups, num_temp_dn
            )[:, temp_group_mask]
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature,
            dn_anchor=dn_anchor,
            dn_cls_target=dn_cls_target,
            valid_mask=valid_mask,
            dn_id_target=dn_id_target,
        )


@BBOX_ASSIGNERS.register_module()
class HungarianLinesAssigner(BaseAssigner):
    """
        Computes one-to-one matching between predictions and ground truth.
        This class computes an assignment between the targets and the predictions
        based on the costs. The costs are weighted sum of three components:
        classification cost and regression L1 cost. The
        targets don't include the no_object, so generally there are more
        predictions than targets. After the one-to-one matching, the un-matched
        are treated as backgrounds. Thus each query prediction will be assigned
        with `0` or a positive integer indicating the ground truth index:
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt
        Args:
            cls_weight (int | float, optional): The scale factor for classification
                cost. Default 1.0.
            bbox_weight (int | float, optional): The scale factor for regression
                L1 cost. Default 1.0.
    """

    def __init__(self, cost=dict, **kwargs):
        self.cost = build_match_cost(cost)

    def assign(self,
               preds: dict,
               gts: dict,
               ignore_cls_cost=False,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """
            Computes one-to-one matching based on the weighted costs.
            This method assign each query prediction to a ground truth or
            background. The `assigned_gt_inds` with -1 means don't care,
            0 means negative sample, and positive number is the index (1-based)
            of assigned gt.
            The assignment is done in the following steps, the order matters.
            1. assign every prediction to -1
            2. compute the weighted costs
            3. do Hungarian matching on CPU based on the costs
            4. assign all to 0 (background) first, then for each matched pair
            between predictions and gts, treat this prediction as foreground
            and assign the corresponding gt index (plus 1) to it.
            Args:
                lines_pred (Tensor): predicted normalized lines:
                    [num_query, num_points, 2]
                cls_pred (Tensor): Predicted classification logits, shape
                    [num_query, num_class].

                lines_gt (Tensor): Ground truth lines
                    [num_gt, num_points, 2].
                labels_gt (Tensor): Label of `gt_bboxes`, shape (num_gt,).
                gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                    labelled as `ignored`. Default None.
                eps (int | float, optional): A value added to the denominator for
                    numerical stability. Default 1e-7.
            Returns:
                :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        
        num_gts, num_lines = gts['lines'].size(0), preds['lines'].size(0)
        if num_gts == 0 or num_lines == 0:
            return None, None, None

        # compute the weighted costs
        gt_permute_idx = None # (num_preds, num_gts)
        if self.cost.reg_cost.permute:
            cost, gt_permute_idx = self.cost(preds, gts, ignore_cls_cost)
        else:
            cost = self.cost(preds, gts, ignore_cls_cost)

        ### 
        # if torch.any(torch.isnan(cost)):
        #     print("****************cost nan***************")
        #     cost = torch.nan_to_num(cost, 
        #             nan=1e5,
        #             posinf=1e5,
        #             neginf=-1e5)
        ### 

        # do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu().numpy()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        return matched_row_inds, matched_col_inds, gt_permute_idx