import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.core.box3d import *


@PLUGIN_LAYERS.register_module()
class InstanceQueue(nn.Module):
    def __init__(
        self,
        embed_dims,
        queue_length=0,
        tracking_threshold=0,
        feature_map_scale=None,
    ):
        super(InstanceQueue, self).__init__()
        self.embed_dims = embed_dims
        self.queue_length = queue_length
        self.tracking_threshold = tracking_threshold

        kernel_size = tuple([int(x / 2) for x in feature_map_scale])
        self.ego_feature_encoder = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size),
        )
        self.ego_anchor = nn.Parameter(
            torch.tensor([[0, 0.5, -1.84 + 1.56/2, np.log(4.08), np.log(1.73), np.log(1.56), 1, 0, 0, 0, 0],], dtype=torch.float32),
            requires_grad=False,
        )

        self.reset()

    def reset(self):
        self.metas = None
        self.prev_instance_id = None
        self.prev_confidence = None
        self.period = None
        self.instance_feature_queue = []
        self.anchor_queue = []
        self.prev_ego_status = None
        self.ego_period = None
        self.ego_feature_queue = []
        self.ego_anchor_queue = []

    def get(
        self,
        det_output,
        feature_maps,
        metas,
        batch_size,
        mask,
        anchor_handler,
    ):
        if (
            self.period is not None
            and batch_size == self.period.shape[0]
        ):
            if anchor_handler is not None:
                T_temp2cur = feature_maps[0].new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                for i in range(len(self.anchor_queue)):
                    temp_anchor = self.anchor_queue[i]
                    temp_anchor = anchor_handler.anchor_projection(
                        temp_anchor,
                        [T_temp2cur],
                    )[0]
                    self.anchor_queue[i] = temp_anchor
                for i in range(len(self.ego_anchor_queue)):
                    temp_anchor = self.ego_anchor_queue[i]
                    temp_anchor = anchor_handler.anchor_projection(
                        temp_anchor,
                        [T_temp2cur],
                    )[0]
                    self.ego_anchor_queue[i] = temp_anchor
        else:
            self.reset()

        self.prepare_motion(det_output, mask)
        ego_feature, ego_anchor = self.prepare_planning(feature_maps, mask, batch_size)

        # temporal 
        temp_instance_feature = torch.stack(self.instance_feature_queue, dim=2)
        temp_anchor = torch.stack(self.anchor_queue, dim=2)
        temp_ego_feature = torch.stack(self.ego_feature_queue, dim=2)
        temp_ego_anchor = torch.stack(self.ego_anchor_queue, dim=2)

        period = torch.cat([self.period, self.ego_period], dim=1)
        temp_instance_feature = torch.cat([temp_instance_feature, temp_ego_feature], dim=1)
        temp_anchor = torch.cat([temp_anchor, temp_ego_anchor], dim=1)
        num_agent = temp_anchor.shape[1]
        
        temp_mask = torch.arange(len(self.anchor_queue), 0, -1, device=temp_anchor.device)
        temp_mask = temp_mask[None, None].repeat((batch_size, num_agent, 1))
        temp_mask = torch.gt(temp_mask, period[..., None])

        return ego_feature, ego_anchor, temp_instance_feature, temp_anchor, temp_mask

    def prepare_motion(
        self,
        det_output,
        mask,
    ):
        instance_feature = det_output["instance_feature"]
        det_anchors = det_output["prediction"][-1]

        if self.period == None:
            self.period = instance_feature.new_zeros(instance_feature.shape[:2]).long()
        else:
            instance_id = det_output['instance_id']
            prev_instance_id = self.prev_instance_id
            match = instance_id[..., None] == prev_instance_id[:, None]
            if self.tracking_threshold > 0:
                temp_mask = self.prev_confidence > self.tracking_threshold
                match = match * temp_mask.unsqueeze(1)

            for i in range(len(self.instance_feature_queue)):
                temp_feature = self.instance_feature_queue[i]
                temp_feature = (
                    match[..., None] * temp_feature[:, None]
                ).sum(dim=2)
                self.instance_feature_queue[i] = temp_feature

                temp_anchor = self.anchor_queue[i]
                temp_anchor = (
                    match[..., None] * temp_anchor[:, None]
                ).sum(dim=2)
                self.anchor_queue[i] = temp_anchor

            self.period = (
                match * self.period[:, None]
            ).sum(dim=2)

        self.instance_feature_queue.append(instance_feature.detach())
        self.anchor_queue.append(det_anchors.detach())
        self.period += 1

        if len(self.instance_feature_queue) > self.queue_length:
            self.instance_feature_queue.pop(0)
            self.anchor_queue.pop(0)
        self.period = torch.clip(self.period, 0, self.queue_length)

    def prepare_planning(
        self,
        feature_maps,
        mask,
        batch_size,
    ):
        ## ego instance init
        feature_maps_inv = feature_maps_format(feature_maps, inverse=True)
        feature_map = feature_maps_inv[0][-1][:, 0]
        ego_feature = self.ego_feature_encoder(feature_map)
        ego_feature = ego_feature.unsqueeze(1).squeeze(-1).squeeze(-1)

        ego_anchor = torch.tile(
            self.ego_anchor[None], (batch_size, 1, 1)
        )
        if self.prev_ego_status is not None:
            prev_ego_status = torch.where(
                mask[:, None, None],
                self.prev_ego_status,
                self.prev_ego_status.new_tensor(0),
            )
            ego_anchor[..., VY] = prev_ego_status[..., 6]

        if self.ego_period == None:
            self.ego_period = ego_feature.new_zeros((batch_size, 1)).long()
        else:
            self.ego_period = torch.where(
                mask[:, None],
                self.ego_period,
                self.ego_period.new_tensor(0),
            )

        self.ego_feature_queue.append(ego_feature.detach())
        self.ego_anchor_queue.append(ego_anchor.detach())
        self.ego_period += 1
        
        if len(self.ego_feature_queue) > self.queue_length:
            self.ego_feature_queue.pop(0)
            self.ego_anchor_queue.pop(0)
        self.ego_period = torch.clip(self.ego_period, 0, self.queue_length)

        return ego_feature, ego_anchor

    def cache_motion(self, instance_feature, det_output, metas):
        det_classification = det_output["classification"][-1].sigmoid()
        det_confidence = det_classification.max(dim=-1).values
        instance_id = det_output['instance_id']
        self.metas = metas
        self.prev_confidence = det_confidence.detach()
        self.prev_instance_id = instance_id

    def cache_planning(self, ego_feature, ego_status):
        self.prev_ego_status = ego_status.detach()
        self.ego_feature_queue[-1] = ego_feature.detach()
