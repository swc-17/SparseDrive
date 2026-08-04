from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..detection3d.decoder import decode_box
from ..instance_bank import topk
from .trajectory_vocab_planner import TrajectoryVocabularyPlanner


def _inject_cjepa_into_iq(temp_feat, temp_anc, pred_feat, proj_bbox, ref_ids, det_output):
    """Overwrite the current-frame slice of stacked IQ tensors at ID-matched positions.

    temp_feat  : (B, N, T, D)  — stacked instance feature queue (in-place modified)
    temp_anc   : (B, N, T, 11) — stacked anchor queue (in-place modified)
    pred_feat  : (B, 50, D)    — C-JEPA predicted agent features (canonical slot order)
    proj_bbox  : (B, 50, 11)   — C-JEPA bboxes projected into current frame lidar
    ref_ids    : (50,) int     — frame t's canonical instance IDs
    det_output : dict with 'instance_id' (B, N)
    """
    cur_ids = det_output.get('instance_id')
    if cur_ids is None:
        return
    B = pred_feat.shape[0]
    for b in range(B):
        for i in range(len(ref_ids)):
            rid = int(ref_ids[i])
            if rid == -1:
                continue
            hits = (cur_ids[b] == rid).nonzero(as_tuple=True)[0]
            if len(hits) > 0:
                j = hits[0].item()
                temp_feat[b, j, -1] = pred_feat[b, i].detach()
                temp_anc[b, j, -1]  = proj_bbox[b, i].detach()


def _project_anchor_torch(bbox, T):
    """Project 11-dim anchors [X,Y,Z,W,L,H,sin,cos,vx,vy,vz] via batched rigid T.

    bbox: (B, N, 11), T: (B, 4, 4) src→dst.  Returns (B, N, 11).
    """
    R3 = T[:, :3, :3]                                                    # (B, 3, 3)
    t3 = T[:, :3, 3].unsqueeze(1)                                        # (B, 1, 3)
    center  = torch.bmm(bbox[:, :, :3], R3.transpose(1, 2)) + t3        # (B, N, 3)
    cos_sin = bbox[:, :, [7, 6]]                                         # (B, N, 2): [cos, sin]
    yaw     = torch.bmm(cos_sin, T[:, :2, :2].transpose(1, 2))          # (B, N, 2): [cos', sin']
    yaw     = yaw[:, :, [1, 0]]                                          # → [sin', cos']
    vel     = torch.bmm(bbox[:, :, 8:], R3.transpose(1, 2))             # (B, N, 3)
    return torch.cat([center, bbox[:, :, 3:6], yaw, vel], dim=-1)


@HEADS.register_module()
class MotionPlanningHead(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
        with_motion_loss=True,
        use_measured_ego_status=False,
        geometric_inputs=False,
        trajectory_vocab=None,
    ):
        super(MotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        # with_motion_loss=False: motion branch still runs forward (shared
        # layers) but receives no supervision — requires
        # find_unused_parameters=True in the training config.
        self.with_motion_loss = with_motion_loss
        # use_measured_ego_status=True: feed data['ego_status'] velocity into
        # the ego anchor instead of the previous frame's predicted status.
        self.use_measured_ego_status = use_measured_ego_status
        self.trajectory_vocab_planner = (
            TrajectoryVocabularyPlanner(**dict(trajectory_vocab))
            if trajectory_vocab is not None else None
        )
        if self.trajectory_vocab_planner is not None and not geometric_inputs:
            raise ValueError("trajectory_vocab requires geometric_inputs=True")

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # V1 fixed plan anchors are replaced entirely by the V2 vocabulary.
        if self.trajectory_vocab_planner is None:
            plan_anchor = np.load(plan_anchor)
            self.plan_anchor = nn.Parameter(
                torch.tensor(plan_anchor, dtype=torch.float32),
                requires_grad=False,
            )
            self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1),
                Linear(embed_dims, embed_dims),
            )

        self.num_det = num_det
        self.num_map = num_map

        # geometric_inputs: the planner consumes ONLY geometry — every
        # image-derived feature it would read (agent/map instance features,
        # CNN ego feature) is replaced by an MLP encoding of the raw
        # anchors/polylines. All other components stay intact.
        self.geometric_inputs = geometric_inputs
        if geometric_inputs:
            self.agent_geo_encoder = nn.Sequential(
                nn.Linear(11, embed_dims), nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True), nn.Linear(embed_dims, embed_dims))
            self.map_geo_encoder = nn.Sequential(
                nn.Linear(40, embed_dims), nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True), nn.Linear(embed_dims, embed_dims))
            # ego CNN removal is handled by the InstanceQueue config
            # (use_cam_ego_feature=False) so no DDP-unused params exist
            assert self.instance_queue is None or \
                not self.instance_queue.use_cam_ego_feature, (
                    "geometric_inputs requires instance_queue "
                    "use_cam_ego_feature=False")

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def _run_planner_variant(
        self,
        instance_feature,
        anchor_embed,
        gnn_key,
        gnn_key_pos,
        temp_instance_feature,
        temp_anchor_embed,
        temp_mask,
        map_instance_feature_selected,
        map_anchor_embed_selected,
        motion_mode_query,
        plan_mode_query,
        num_anchor,
        bs,
        dim,
    ):
        """Stateless replay of the shared motion/planning layers."""
        instance_feature = instance_feature.clone()
        motion_classification, motion_prediction = [], []
        planning_classification, planning_prediction, planning_status = [], [], []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            if op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                ).reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    gnn_key,
                    gnn_key,
                    query_pos=anchor_embed,
                    key_pos=gnn_key_pos,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                motion_query = motion_mode_query + (
                    instance_feature + anchor_embed
                )[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (
                    instance_feature + anchor_embed
                )[:, num_anchor:].unsqueeze(2)
                motion_cls, motion_reg, plan_cls, plan_reg, plan_status = self.layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                )
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)

        return (
            {
                "classification": motion_classification,
                "prediction": motion_prediction,
            },
            {
                "classification": planning_classification,
                "prediction": planning_prediction,
                "status": planning_status,
            },
        )

    @staticmethod
    def _gt_list(value):
        return list(value) if isinstance(value, (list, tuple)) else [value]

    def _make_gt_oracle_variants(
        self,
        det_output,
        map_output,
        metas,
        anchor_encoder,
        det_confidence,
        det_anchors,
        map_confidence,
        map_anchors,
        instance_feature,
        anchor_embed,
        gnn_key,
        gnn_key_pos,
        temp_instance_feature,
        temp_anchor_embed,
        temp_mask,
        map_instance_feature_selected,
        map_anchor_embed_selected,
        motion_mode_query,
        plan_mode_query,
        num_anchor,
        bs,
        dim,
    ):
        """Build paired planner inputs with officially matched GT geometry."""
        required = ("gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts")
        if not all(k in metas for k in required):
            missing = [k for k in required if k not in metas]
            raise KeyError(f"GT planner oracle is missing dataset keys: {missing}")
        if not all(hasattr(self, k) for k in (
            "_gt_det_sampler", "_gt_map_sampler", "_gt_map_anchor_encoder"
        )):
            raise RuntimeError("GT planner oracle samplers/encoder were not attached")
        if bs != 1:
            raise ValueError("GT planner oracle requires samples_per_gpu=1")

        det_top_idx = torch.topk(det_confidence, self.num_det, dim=1).indices
        map_top_idx = torch.topk(map_confidence, self.num_map, dim=1).indices
        det_idx_3d = det_top_idx.unsqueeze(-1).expand(-1, -1, det_anchors.shape[-1])
        map_idx_3d = map_top_idx.unsqueeze(-1).expand(-1, -1, map_anchors.shape[-1])
        det_selected = torch.gather(det_anchors, 1, det_idx_3d)
        map_selected = torch.gather(map_anchors, 1, map_idx_3d)

        # The detector is trained/matched on its first ten regression dimensions;
        # VZ exists in the anchor but has no nuScenes GT supervision.
        det_reg_dim = len(self._gt_det_sampler.reg_weights)
        num_det_cls = det_output["classification"][-1].shape[-1]
        match_topk = getattr(self, "_gt_oracle_match_topk", False)
        if match_topk:
            # Hungarian over only the planner's top-k slots, so every GT can be
            # won by a slot the planner actually consumes.
            det_cls_selected = torch.gather(
                det_output["classification"][-1],
                1,
                det_top_idx.unsqueeze(-1).expand(-1, -1, num_det_cls),
            )
            det_cls_target, det_box_target, _ = self._gt_det_sampler.sample(
                det_cls_selected.float(),
                det_selected[..., :det_reg_dim].float(),
                self._gt_list(metas["gt_labels_3d"]),
                self._gt_list(metas["gt_bboxes_3d"]),
            )
            det_target_label = det_cls_target
            det_match = det_cls_target < num_det_cls
            det_target_selected = det_box_target.to(det_selected.dtype)
        else:
            det_cls_target, det_box_target, _ = self._gt_det_sampler.sample(
                det_output["classification"][-1].float(),
                det_anchors[..., :det_reg_dim].float(),
                self._gt_list(metas["gt_labels_3d"]),
                self._gt_list(metas["gt_bboxes_3d"]),
            )
            det_target_label = torch.gather(det_cls_target, 1, det_top_idx)
            det_match = det_target_label < num_det_cls
            det_target_selected = torch.gather(
                det_box_target,
                1,
                det_top_idx.unsqueeze(-1).expand(-1, -1, det_box_target.shape[-1]),
            ).to(det_selected.dtype)
        # Geometry-only injection replaces X..COS_YAW and keeps the predicted
        # velocities; the *_vel variants additionally replace VX, VY.
        geo_dim = COS_YAW + 1
        det_gt_geo = det_selected.clone()
        det_gt_geo[..., :geo_dim] = torch.where(
            det_match.unsqueeze(-1),
            det_target_selected[..., :geo_dim],
            det_gt_geo[..., :geo_dim],
        )
        det_gt_vel = det_selected.clone()
        det_gt_vel[..., :det_reg_dim] = torch.where(
            det_match.unsqueeze(-1),
            det_target_selected,
            det_gt_vel[..., :det_reg_dim],
        )

        num_map_cls = map_output["classification"][-1].shape[-1]
        if match_topk:
            map_cls_selected = torch.gather(
                map_output["classification"][-1],
                1,
                map_top_idx.unsqueeze(-1).expand(-1, -1, num_map_cls),
            )
            map_cls_target, map_box_target, _ = self._gt_map_sampler.sample(
                map_cls_selected.float(),
                map_selected.float(),
                self._gt_list(metas["gt_map_labels"]),
                self._gt_list(metas["gt_map_pts"]),
            )
            map_target_label = map_cls_target
            map_match = map_cls_target < num_map_cls
            map_target_selected = map_box_target.to(map_selected.dtype)
        else:
            map_cls_target, map_box_target, _ = self._gt_map_sampler.sample(
                map_output["classification"][-1].float(),
                map_anchors.float(),
                self._gt_list(metas["gt_map_labels"]),
                self._gt_list(metas["gt_map_pts"]),
            )
            map_target_label = torch.gather(map_cls_target, 1, map_top_idx)
            map_match = map_target_label < num_map_cls
            map_target_selected = torch.gather(
                map_box_target,
                1,
                map_top_idx.unsqueeze(-1).expand(-1, -1, map_box_target.shape[-1]),
            ).to(map_selected.dtype)
        map_gt = torch.where(
            map_match.unsqueeze(-1), map_target_selected, map_selected
        )
        map_gt_embed = self._gt_map_anchor_encoder(map_gt)

        def det_injected(det_gt):
            det_gt_embed = anchor_encoder(det_gt)
            obj_ae = anchor_embed.clone()
            obj_kpos = gnn_key_pos.clone()
            obj_temp_ae = temp_anchor_embed.clone()
            det_gt_all = det_anchors.clone()
            for b in range(bs):
                slots = det_match[b].nonzero(as_tuple=True)[0]
                queries = det_top_idx[b, slots]
                obj_ae[b, queries] = det_gt_embed[b, slots]
                obj_kpos[b, slots] = det_gt_embed[b, slots]
                obj_temp_ae[b * (num_anchor + 1) + queries, -1] = det_gt_embed[b, slots]
                det_gt_all[b, queries] = det_gt[b, slots]
            obj_motion_anchor = self.get_motion_anchor(
                det_output["classification"][-1].sigmoid(), det_gt_all
            )
            obj_mmq = self.motion_anchor_encoder(
                gen_sineembed_for_position(obj_motion_anchor[..., -1, :])
            )
            inputs = dict(
                anchor_embed=obj_ae.detach(),
                gnn_key_pos=obj_kpos.detach(),
                temp_anchor_embed=obj_temp_ae.detach(),
                motion_mode_query=obj_mmq.detach(),
            )
            return inputs, det_gt_all.detach()

        det_geo, det_geo_anchors = det_injected(det_gt_geo)
        det_vel, det_vel_anchors = det_injected(det_gt_vel)

        oracle_map_kpos = map_anchor_embed_selected.clone()
        oracle_map_kpos[map_match] = map_gt_embed[map_match]

        common = dict(
            gnn_key=gnn_key.detach().clone(),
            temp_instance_feature=temp_instance_feature.detach().clone(),
            temp_mask=temp_mask.detach().clone(),
            map_instance_feature_selected=map_instance_feature_selected.detach().clone(),
            plan_mode_query=plan_mode_query.detach().clone(),
            num_anchor=num_anchor,
            bs=bs,
            dim=dim,
        )
        identity_det = dict(
            anchor_embed=anchor_embed.detach().clone(),
            gnn_key_pos=gnn_key_pos.detach().clone(),
            temp_anchor_embed=temp_anchor_embed.detach().clone(),
            motion_mode_query=motion_mode_query.detach().clone(),
        )
        identity_map = dict(
            map_anchor_embed_selected=map_anchor_embed_selected.detach().clone()
        )
        oracle_map = dict(map_anchor_embed_selected=oracle_map_kpos.detach())
        variants = {
            "identity": {**identity_det, **identity_map},
            "gt_bbox": {**det_geo, **identity_map},
            "gt_bbox_vel": {**det_vel, **identity_map},
            "gt_map": {**identity_det, **oracle_map},
            "gt_bbox_map": {**det_geo, **oracle_map},
            "gt_bbox_map_vel": {**det_vel, **oracle_map},
        }
        for values in variants.values():
            values["instance_feature"] = instance_feature.detach().clone()
        # Injected full-slot det anchors per variant, so a rescore-enabled
        # decode collision-checks against the same geometry the planner saw.
        self._gt_oracle_det_variants = {
            "identity": None,
            "gt_bbox": det_geo_anchors,
            "gt_bbox_vel": det_vel_anchors,
            "gt_map": None,
            "gt_bbox_map": det_geo_anchors,
            "gt_bbox_map_vel": det_vel_anchors,
        }
        for values in variants.values():
            values.update(common)
        matched_xy_l2 = (
            det_selected[..., :2] - det_target_selected[..., :2]
        ).norm(dim=-1)[det_match]
        matched_xyz_l2 = (
            det_selected[..., :3] - det_target_selected[..., :3]
        ).norm(dim=-1)[det_match]
        matched_size_l2 = (
            det_selected[..., W : H + 1].exp()
            - det_target_selected[..., W : H + 1].exp()
        ).norm(dim=-1)[det_match]
        pred_yaw = torch.atan2(
            det_selected[..., SIN_YAW], det_selected[..., COS_YAW]
        )
        target_yaw = torch.atan2(
            det_target_selected[..., SIN_YAW],
            det_target_selected[..., COS_YAW],
        )
        matched_yaw_deg = torch.rad2deg(
            torch.atan2(
                torch.sin(pred_yaw - target_yaw),
                torch.cos(pred_yaw - target_yaw),
            ).abs()
        )[det_match]
        matched_vel_l2 = (
            det_selected[..., VX : VY + 1] - det_target_selected[..., VX : VY + 1]
        ).norm(dim=-1)[det_match]
        map_pts_pred = map_selected.view(*map_selected.shape[:-1], -1, 2)
        map_pts_gt = map_target_selected.view(*map_target_selected.shape[:-1], -1, 2)
        matched_map_pt_l2 = (map_pts_pred - map_pts_gt).norm(dim=-1)[map_match]
        det_selected_label = torch.gather(
            det_output["classification"][-1].argmax(dim=-1), 1, det_top_idx
        )
        map_selected_label = torch.gather(
            map_output["classification"][-1].argmax(dim=-1), 1, map_top_idx
        )
        self._gt_oracle_frame_stats = {
            "bbox_matched": int(det_match.sum().item()),
            "bbox_slots": int(det_match.numel()),
            "bbox_gt": sum(len(x) for x in self._gt_list(metas["gt_labels_3d"])),
            "bbox_class_agree": int(
                (det_selected_label[det_match] == det_target_label[det_match]).sum().item()
            ),
            "map_matched": int(map_match.sum().item()),
            "map_slots": int(map_match.numel()),
            "map_gt": sum(len(x) for x in self._gt_list(metas["gt_map_labels"])),
            "map_class_agree": int(
                (map_selected_label[map_match] == map_target_label[map_match]).sum().item()
            ),
            "bbox_matched_xy_l2": matched_xy_l2.float().cpu().numpy().tolist(),
            "bbox_matched_xyz_l2": matched_xyz_l2.float().cpu().numpy().tolist(),
            "bbox_matched_size_l2": matched_size_l2.float().cpu().numpy().tolist(),
            "bbox_matched_yaw_deg": matched_yaw_deg.float().cpu().numpy().tolist(),
            "bbox_matched_vel_l2": matched_vel_l2.float().cpu().numpy().tolist(),
            "map_matched_point_l2": matched_map_pt_l2.float().cpu().flatten().numpy().tolist(),
            "map_matched_line_l2": matched_map_pt_l2.float().mean(dim=-1).cpu().numpy().tolist(),
        }
        return variants

    def forward(
        self, 
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):   
        # =========== geometric-input substitution ===========
        if self.geometric_inputs:
            det_output = dict(det_output)
            det_output["instance_feature"] = self.agent_geo_encoder(
                det_output["prediction"][-1])
            map_output = dict(map_output)
            map_output["instance_feature"] = self.map_geo_encoder(
                map_output["prediction"][-1])

        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (
            instance_feature_selected,
            anchor_embed_selected,
            detection_geometry_selected,
        ) = topk(
            det_confidence,
            self.num_det,
            instance_feature,
            anchor_embed,
            det_anchors,
        )

        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (
            map_instance_feature_selected,
            map_anchor_embed_selected,
            map_geometry_selected,
        ) = topk(
            map_confidence,
            self.num_map,
            map_instance_feature,
            map_anchor_embed,
            map_anchors,
        )

        # Evaluation-only source boundary. Teacher geometry is deliberately
        # re-encoded through the same path used by external EMPERROR output.
        geometry_source = getattr(self, "_planner_geometry_source", None)
        if geometry_source is not None:
            if not self.geometric_inputs:
                raise RuntimeError("planner geometry gate requires geometric_inputs=True")
            map_encoder = getattr(self, "_planner_map_anchor_encoder", None)
            if map_encoder is None:
                raise RuntimeError("planner geometry gate needs the map anchor encoder")
            generated = getattr(self, "_planner_geometry_for_frame", None)
            from navsim_agent.geometry_gate import (
                GENERATED_GEOMETRY_SOURCES,
                planner_geometry,
            )

            detections, maps = planner_geometry(
                geometry_source,
                detection_geometry_selected,
                map_geometry_selected,
                generated,
            )
            if geometry_source in GENERATED_GEOMETRY_SOURCES:
                source_delta = max(
                    (detections - detection_geometry_selected).abs().max().item(),
                    (maps - map_geometry_selected).abs().max().item(),
                )
                previous_delta = getattr(
                    self, "_planner_geometry_gate_max_source_delta", None
                )
                self._planner_geometry_gate_max_source_delta = max(
                    0.0 if previous_delta is None else previous_delta,
                    source_delta,
                )
            encoded = (
                self.agent_geo_encoder(detections),
                anchor_encoder(detections),
                self.map_geo_encoder(maps),
                map_encoder(maps),
            )
            if geometry_source == "teacher":
                native = (
                    instance_feature_selected,
                    anchor_embed_selected,
                    map_instance_feature_selected,
                    map_anchor_embed_selected,
                )
                tolerance = (
                    1e-3
                    if detections.dtype in (torch.float16, torch.bfloat16)
                    else 1e-5
                )
                errors = [
                    (before - after).abs().max().item()
                    for before, after in zip(native, encoded)
                ]
                previous_error = getattr(
                    self, "_planner_geometry_gate_max_error", None
                )
                self._planner_geometry_gate_max_error = max(
                    0.0 if previous_error is None else previous_error,
                    *errors,
                )
                if any(
                    not torch.allclose(
                        before, after, atol=tolerance, rtol=tolerance
                    )
                    for before, after in zip(native, encoded)
                ):
                    raise RuntimeError("teacher geometry re-encoding failed parity")
            (
                instance_feature_selected,
                anchor_embed_selected,
                map_instance_feature_selected,
                map_anchor_embed_selected,
            ) = encoded
            self._planner_geometry_gate_calls = (
                getattr(self, "_planner_geometry_gate_calls", 0)
                + detections.shape[0]
            )

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
            ego_status=(
                metas.get("ego_status") if self.use_measured_ego_status else None
            ),
        )
        # --- C-JEPA IQ injection (update_iq flag) ---
        if getattr(self, 'cjepa_cfg', None) is not None and self.cjepa_cfg.get('update_iq'):
            cfg = self.cjepa_cfg
            proj_bbox_iq = _project_anchor_torch(cfg['pred_bbox'], cfg['T_rel'])  # (B,51,11) t→t+1
            _inject_cjepa_into_iq(
                temp_instance_feature, temp_anchor,
                cfg['pred_feat'][:, :50],
                proj_bbox_iq[:, :50],
                cfg['ref_ids'],
                det_output,
            )

        if self.geometric_inputs:
            ego_feature = ego_feature + self.agent_geo_encoder(ego_anchor)

        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        if self.trajectory_vocab_planner is None:
            plan_anchor = torch.tile(
                self.plan_anchor[None], (bs, 1, 1, 1, 1)
            )

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
        if self.trajectory_vocab_planner is None:
            plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
            plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)
        else:
            plan_mode_query = None

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)
        self._last_instance_feature_selected = instance_feature_selected.detach()  # (B, 51, 256)
        self._last_anchor_embed_selected     = anchor_embed_selected.detach()       # (B, 51, 256)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # Pre-compute ablated map keys once before the loop.
        if getattr(self, 'ablate_map_zero', False):
            map_instance_feature_selected = torch.zeros_like(map_instance_feature_selected)
            map_anchor_embed_selected     = torch.zeros_like(map_anchor_embed_selected)

        # Pre-compute ablated agent-GNN keys once before the loop.
        if getattr(self, 'ablate_gnn_zero', False):
            # Zero out both instance features and anchor embeddings for the top-50 agents;
            # ego slot (last) is kept unchanged.
            gnn_key     = torch.zeros_like(instance_feature_selected)
            gnn_key_pos = torch.zeros_like(anchor_embed_selected)
            gnn_key[:, -1, :]     = instance_feature_selected[:, -1, :]
            gnn_key_pos[:, -1, :] = anchor_embed_selected[:, -1, :]
        elif getattr(self, 'ablate_gnn_anchor_mean', False):
            # Collapse only anchor positions to mean; instance features unchanged.
            gnn_key     = instance_feature_selected
            gnn_key_pos = anchor_embed_selected.mean(dim=1, keepdim=True).expand_as(anchor_embed_selected)
        elif getattr(self, 'ablate_gnn_both_mean', False):
            # Collapse both instance features and anchor positions to their scene mean.
            gnn_key      = instance_feature_selected.mean(dim=1, keepdim=True).expand_as(instance_feature_selected)
            gnn_key_pos  = anchor_embed_selected.mean(dim=1, keepdim=True).expand_as(anchor_embed_selected)
        elif getattr(self, 'ablate_gnn_permute', False):
            # Permute agent dim so each instance feature is paired with a different
            # anchor position. Fixed seed → same permutation across all samples.
            N = instance_feature_selected.shape[1]
            g = torch.Generator(device='cpu')
            g.manual_seed(42)
            perm = torch.randperm(N, generator=g).to(instance_feature_selected.device)
            gnn_key     = instance_feature_selected[:, perm, :]
            gnn_key_pos = anchor_embed_selected
        elif getattr(self, 'ablate_gnn_ego_only', False):
            # Replace all agent keys with ego feature; box positions unchanged.
            gnn_key     = instance_feature_selected[:, -1:, :].expand_as(instance_feature_selected)
            gnn_key_pos = anchor_embed_selected
        elif getattr(self, 'ablate_gnn_instance_features', False):
            gnn_key     = instance_feature_selected.mean(dim=1, keepdim=True).expand_as(instance_feature_selected)
            gnn_key_pos = anchor_embed_selected
        elif getattr(self, 'cjepa_inject_features', None) is not None:
            # C-JEPA injection: replace top-50+ego K/V and ego Q with C-JEPA predictions.
            # cjepa_inject_features: tuple of (instance_feature (B,51,256), anchor_embed (B,51,256))
            # cjepa_inject_strict: also replace the 900-slot Q top-50 positions with C-JEPA.
            cjepa_feat, cjepa_ae = self.cjepa_inject_features
            instance_feature_selected = torch.cat([cjepa_feat[:, :self.num_det], cjepa_feat[:, -1:]], dim=1)
            anchor_embed_selected     = torch.cat([cjepa_ae[:, :self.num_det],   cjepa_ae[:, -1:]],   dim=1)
            gnn_key     = instance_feature_selected
            gnn_key_pos = anchor_embed_selected
            if getattr(self, 'cjepa_inject_strict', False):
                # Also replace top-50 in the 900-slot Q (affects temporal GNN Q and planning Q)
                top_idx = det_confidence.topk(self.num_det, dim=-1).indices  # (B, num_det)
                for b in range(bs):
                    instance_feature[b, top_idx[b]] = cjepa_feat[b, :self.num_det]
                    anchor_embed[b, top_idx[b]]     = cjepa_ae[b, :self.num_det]
                # Replace ego slot too
                instance_feature[:, num_anchor:] = cjepa_feat[:, -1:]
                anchor_embed[:, num_anchor:]      = cjepa_ae[:, -1:]
            else:
                # Replace only ego slot in the query; 900 agent Q slots keep real features.
                instance_feature = torch.cat([
                    instance_feature[:, :num_anchor, :],
                    cjepa_feat[:, -1:],
                ], dim=1)
                anchor_embed = torch.cat([
                    anchor_embed[:, :num_anchor, :],
                    cjepa_ae[:, -1:],
                ], dim=1)
        elif getattr(self, 'cjepa_withcam_data', None) is not None:
            # C-JEPA withcam mode:
            #   - GNN K/V: C-JEPA predicted features + anchor embeds computed from
            #     predicted bboxes projected into t+1's lidar frame
            #   - Agent Q (temp_gnn, cross_gnn, refine): real camera detection features
            #   - Ego query: C-JEPA prediction (default) OR real camera ego (real_ego=True)
            cjepa_feat, cjepa_pred_bbox, T_rel = self.cjepa_withcam_data
            # project pred_bbox from frame t's lidar → frame t+1's lidar
            proj_bbox = _project_anchor_torch(cjepa_pred_bbox, T_rel)   # (B, 51, 11)
            cjepa_ae  = anchor_encoder(proj_bbox)                        # (B, 51, 256)
            # GNN K/V uses all 51 C-JEPA slots (50 agents + ego)
            gnn_key     = cjepa_feat   # (B, 51, 256)
            gnn_key_pos = cjepa_ae     # (B, 51, 256)
            if not getattr(self, 'cjepa_withcam_real_ego', False):
                # replace ego query with C-JEPA predicted ego (planner + GNN Q)
                instance_feature = torch.cat(
                    [instance_feature[:, :num_anchor], cjepa_feat[:, -1:]], dim=1
                )
                anchor_embed = torch.cat(
                    [anchor_embed[:, :num_anchor], cjepa_ae[:, -1:]], dim=1
                )
            # else: keep real t+1 camera ego query unchanged
        elif getattr(self, 'cjepa_cfg', None) is not None and (
            self.cjepa_cfg.get('use_kv') or self.cjepa_cfg.get('use_ego')
        ):
            # Unified composable C-JEPA flags (use_kv / use_ego).
            # update_iq was already handled above (before anchor_encoder calls).
            cfg       = self.cjepa_cfg
            pred_feat = cfg['pred_feat']           # (B, 51, 256)
            if cfg.get('pred_kpos') is not None:
                cjepa_ae = cfg['pred_kpos']        # (B, 51, 256) pre-computed, skip anchor encoder
            else:
                proj_bbox = _project_anchor_torch(cfg['pred_bbox'], cfg['T_rel'])
                cjepa_ae  = anchor_encoder(proj_bbox)  # (B, 51, 256)

            if cfg.get('use_kv'):
                # use_real_kv_feat: real t+1 instance features as K, WM bbox as K pos encoding
                # use_real_kv_pos:  WM features as K, real t+1 anchor_embed as K pos encoding
                gnn_key     = (instance_feature_selected if cfg.get('use_real_kv_feat')
                               else pred_feat)
                gnn_key_pos = (anchor_embed_selected if cfg.get('use_real_kv_pos')
                               else cjepa_ae)
                # kv_valid_mask: (B, 51) bool — zero out invalid KV slots
                if cfg.get('kv_valid_mask') is not None:
                    m = cfg['kv_valid_mask'].float().unsqueeze(-1)  # (B, 51, 1)
                    gnn_key     = gnn_key * m
                    gnn_key_pos = gnn_key_pos * m
            else:
                gnn_key     = instance_feature_selected
                gnn_key_pos = anchor_embed_selected

            if cfg.get('use_ego'):
                instance_feature = torch.cat(
                    [instance_feature[:, :num_anchor], pred_feat[:, -1:]], dim=1
                )
                anchor_embed = torch.cat(
                    [anchor_embed[:, :num_anchor], cjepa_ae[:, -1:]], dim=1
                )
        else:
            gnn_key     = instance_feature_selected
            gnn_key_pos = anchor_embed_selected

        # Inject pre-stored planner GNN inputs (eliminates upstream non-determinism for replay).
        # Set mpl._inject_planner_inputs = snap dict before calling forward.
        _inj = getattr(self, '_inject_planner_inputs', None)
        if _inj is not None:
            dev = instance_feature.device
            instance_feature              = _inj['if_query'].to(dev)
            anchor_embed                  = _inj['ae_query'].to(dev)
            gnn_key                       = _inj['gnn_key'].to(dev)
            gnn_key_pos                   = _inj['gnn_kpos'].to(dev)
            temp_instance_feature         = _inj['temp_if'].to(dev)
            temp_anchor_embed             = _inj['temp_ae'].to(dev)
            temp_mask                     = _inj['temp_mask'].to(dev)
            map_instance_feature_selected = _inj['map_key'].to(dev)
            map_anchor_embed_selected     = _inj['map_kpos'].to(dev)
            if 'mmq' in _inj:
                motion_mode_query         = _inj['mmq'].to(dev)
            plan_mode_query               = _inj['pmq'].to(dev)
            # Ego-only mode: if_query carries only the ego slot (shape [..., 1, ...]).
            # Set num_anchor=0 so temp_gnn reshape and refine split work correctly,
            # and truncate motion_mode_query to empty to avoid the broadcast mismatch.
            if instance_feature.shape[1] == 1:
                num_anchor = 0
                motion_mode_query = motion_mode_query[:, :0]

        # Ablation: zero 900-agent query (not K/V, not ego slot).
        # Applied after injection so it works on the injected or fresh tensors.
        if getattr(self, 'ablate_query_zero', False):
            instance_feature = torch.cat([
                torch.zeros_like(instance_feature[:, :num_anchor]),
                instance_feature[:, num_anchor:],
            ], dim=1)
            anchor_embed = torch.cat([
                torch.zeros_like(anchor_embed[:, :num_anchor]),
                anchor_embed[:, num_anchor:],
            ], dim=1)

        # Flag: drop agent queries and their temporal history entirely.
        # Only the ego slot (index num_anchor) and K/V tensors are kept.
        # temp has shape (B*N, T, dim); ego rows sit at indices num_anchor, num_anchor+(N+1), …
        if getattr(self, 'drop_agent_queries', False) and num_anchor > 0:
            instance_feature      = instance_feature[:, num_anchor:]
            anchor_embed          = anchor_embed[:, num_anchor:]
            N = num_anchor + 1
            temp_instance_feature = temp_instance_feature[num_anchor::N]
            temp_anchor_embed     = temp_anchor_embed[num_anchor::N]
            temp_mask             = temp_mask[num_anchor::N]
            motion_mode_query     = motion_mode_query[:, :0]
            num_anchor            = 0

        # Snapshot all GNN inputs (for subsequent controlled replay).
        if getattr(self, '_snap_planner', False):
            self._snap_if_query   = instance_feature.detach().clone()              # (B, 901, 256)
            self._snap_ae_query   = anchor_embed.detach().clone()                  # (B, 901, 256)
            self._snap_gnn_key    = gnn_key.detach().clone()                       # (B,  51, 256)
            self._snap_gnn_kpos   = gnn_key_pos.detach().clone()                   # (B,  51, 256)
            self._snap_temp_if    = temp_instance_feature.detach().clone()          # (B*T, N, 256)
            self._snap_temp_ae    = temp_anchor_embed.detach().clone()              # (B*T, N, 256)
            self._snap_temp_mask  = temp_mask.detach().clone()                     # (B*T, N)
            self._snap_map_key    = map_instance_feature_selected.detach().clone() # (B, M, 256)
            self._snap_map_kpos   = map_anchor_embed_selected.detach().clone()     # (B, M, 256)
            self._snap_mmq        = motion_mode_query.detach().clone()             # motion mode queries
            if plan_mode_query is not None:
                self._snap_pmq = plan_mode_query.detach().clone()                  # plan mode queries
            # Raw pre-encoding anchors for ego-frame normalisation in C-JEPA training.
            _, (det_anc_sel,) = topk(det_confidence, self.num_det, det_anchors)
            self._snap_anchor_bbox = torch.cat(
                [det_anc_sel, ego_anchor], dim=1
            ).detach().clone()                                                      # (B, 51, 11)
            _, (map_anc_sel,) = topk(map_confidence, self.num_map, map_anchors)
            self._snap_map_anchor = map_anc_sel.detach().clone()                   # (B, 10, 40)
            T_g = metas.get("img_metas", [{}])[0].get("T_global")
            if T_g is not None:
                self._snap_T_global = torch.as_tensor(
                    np.array(T_g), dtype=torch.float32
                )  # (4, 4)

        gt_oracle_variants = None
        if getattr(self, "_gt_oracle_enabled", False):
            gt_oracle_variants = self._make_gt_oracle_variants(
                det_output=det_output,
                map_output=map_output,
                metas=metas,
                anchor_encoder=anchor_encoder,
                det_confidence=det_confidence,
                det_anchors=det_anchors,
                map_confidence=map_confidence,
                map_anchors=map_anchors,
                instance_feature=instance_feature,
                anchor_embed=anchor_embed,
                gnn_key=gnn_key,
                gnn_key_pos=gnn_key_pos,
                temp_instance_feature=temp_instance_feature,
                temp_anchor_embed=temp_anchor_embed,
                temp_mask=temp_mask,
                map_instance_feature_selected=map_instance_feature_selected,
                map_anchor_embed_selected=map_anchor_embed_selected,
                motion_mode_query=motion_mode_query,
                plan_mode_query=plan_mode_query,
                num_anchor=num_anchor,
                bs=bs,
                dim=dim,
            )

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        vocabulary_output = None
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    gnn_key,
                    gnn_key,
                    query_pos=anchor_embed,
                    key_pos=gnn_key_pos,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                if self.trajectory_vocab_planner is None:
                    plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2)
                else:
                    plan_query = None
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                )
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_status.append(plan_status)
                if self.trajectory_vocab_planner is None:
                    planning_classification.append(plan_cls)
                    planning_prediction.append(plan_reg)
                else:
                    geometry_memory = torch.cat(
                        [
                            gnn_key + gnn_key_pos,
                            map_instance_feature_selected
                            + map_anchor_embed_selected,
                        ],
                        dim=1,
                    )
                    scene_context = (
                        instance_feature + anchor_embed
                    )[:, num_anchor:]
                    vocabulary_output = self.trajectory_vocab_planner(
                        scene_context,
                        geometry_memory,
                        metas.get("gt_ego_fut_cmd"),
                    )
        
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)

        # Snap planner outputs (planning logits from refine layer).
        # These are purely determined by the injected planner inputs and are
        # bit-exact on replay, unlike final_planning which goes through the decoder.
        if getattr(self, '_snap_planner', False) and planning_classification:
            self._snap_plan_cls = planning_classification[-1].detach().clone()  # (B,1,ego_mode,1)
            self._snap_plan_reg = planning_prediction[-1].detach().clone()       # (B,1,ego_mode,ts,2)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {} if vocabulary_output is None else vocabulary_output
        planning_output.update({
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
            # Ego instance representation for C-JEPA state:
            #   ego_feature     : (B, 1, 256) from InstanceQueue CNN encoder
            #   ego_anchor_embed: (B, 1, 256) from AnchorEncoder(ego_anchor)
            "ego_feature":      ego_feature,
            "ego_anchor_embed": ego_anchor_embed,
        })

        self._gt_oracle_outputs = {}
        if gt_oracle_variants is not None:
            for tag, variant_inputs in gt_oracle_variants.items():
                variant_motion, variant_planning = self._run_planner_variant(
                    **variant_inputs
                )
                variant_motion.update(
                    period=motion_output["period"],
                    anchor_queue=motion_output["anchor_queue"],
                )
                variant_planning.update(
                    period=planning_output["period"],
                    anchor_queue=planning_output["anchor_queue"],
                )
                self._gt_oracle_outputs[tag] = (variant_motion, variant_planning)

            identity_planning = self._gt_oracle_outputs["identity"][1]
            self._gt_oracle_frame_stats.update({
                "identity_plan_cls_max": float((
                    identity_planning["classification"][-1]
                    - planning_output["classification"][-1]
                ).abs().max().item()),
                "identity_plan_reg_max": float((
                    identity_planning["prediction"][-1]
                    - planning_output["prediction"][-1]
                ).abs().max().item()),
            })
        return motion_output, planning_output
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):
        loss = {}
        if self.with_motion_loss:
            motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
            loss.update(motion_loss)
        if self.trajectory_vocab_planner is None:
            planning_loss = self.loss_planning(planning_model_outs, data)
        else:
            planning_loss = self.trajectory_vocab_planner.loss(
                planning_model_outs, data
            )
            planning_loss["planning_loss_status_0"] = self.plan_loss_status(
                planning_model_outs["status"][-1].squeeze(1),
                data["ego_status"],
            )
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        no_agents = motion_output["classification"][-1].shape[1] == 0
        if no_agents:
            bs = motion_output["classification"][-1].shape[0]
            motion_result = [
                {"trajs_3d": torch.zeros(0), "trajs_score": torch.zeros(0),
                 "anchor_queue": torch.zeros(0), "period": torch.zeros(0)}
                for _ in range(bs)
            ]
        else:
            motion_result = self.motion_decoder.decode(
                det_output["classification"],
                det_output["prediction"],
                det_output.get("instance_id"),
                det_output.get("quality"),
                motion_output,
            )
        if self.trajectory_vocab_planner is None:
            planning_result = self.planning_decoder.decode(
                det_output,
                motion_output,
                planning_output,
                data,
            )
        else:
            planning_result = self.trajectory_vocab_planner.decode(
                planning_output
            )
            ego_anchor_queue = torch.stack(
                planning_output["anchor_queue"], dim=2
            )
            for batch_idx, result in enumerate(planning_result):
                result["ego_period"] = planning_output["period"][
                    batch_idx
                ].cpu()
                result["ego_anchor_queue"] = decode_box(
                    ego_anchor_queue[batch_idx]
                ).cpu()
                result["ego_feature"] = planning_output["ego_feature"][
                    batch_idx
                ].cpu()
                result["ego_anchor_embed"] = planning_output[
                    "ego_anchor_embed"
                ][batch_idx].cpu()

        if self.trajectory_vocab_planner is None:
            variant_det_anchors = getattr(self, "_gt_oracle_det_variants", {})
            for tag, (variant_motion, variant_planning) in getattr(
                self, "_gt_oracle_outputs", {}
            ).items():
                variant_det = det_output
                injected_anchors = variant_det_anchors.get(tag)
                if injected_anchors is not None:
                    variant_det = {**det_output}
                    variant_det["prediction"] = list(det_output["prediction"][:-1]) + [
                        injected_anchors
                    ]
                decoded = self.planning_decoder.decode(
                    variant_det, variant_motion, variant_planning, data
                )
                for base, variant in zip(planning_result, decoded):
                    base[f"final_planning_{tag}"] = variant["final_planning"]

        return motion_result, planning_result
