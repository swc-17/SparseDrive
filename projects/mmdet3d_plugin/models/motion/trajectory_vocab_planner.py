import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _navsim_to_sparsedrive(array):
    """Rotate NAVSIM x-forward/y-left poses into SD x-right/y-forward."""
    array = np.array(array, dtype=np.float32, copy=True)
    x_nav = array[..., 0].copy()
    y_nav = array[..., 1].copy()
    array[..., 0] = -y_nav
    array[..., 1] = x_nav
    if array.shape[-1] > 2:
        array[..., 2] = (array[..., 2] + math.pi / 2 + math.pi) % (
            2 * math.pi
        ) - math.pi
    return array


def _sparsedrive_to_navsim_torch(poses):
    """Inverse of _navsim_to_sparsedrive for (..., 3) pose tensors."""
    x_sd, y_sd, yaw_sd = poses[..., 0], poses[..., 1], poses[..., 2]
    yaw_nav = torch.remainder(yaw_sd - math.pi / 2 + math.pi, 2 * math.pi) - math.pi
    return torch.stack([y_sd, -x_sd, yaw_nav], dim=-1)


# Released V2 default (dataset_version="v2"): the eight EPDMS sub-metrics.
V2_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "lane_keeping",
    "history_comfort",
)


class _CandidateBlock(nn.Module):
    """V2 candidate update with geometry attention replacing image sampling."""

    def __init__(self, embed_dims, feedforward_dims, num_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_dims),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dims, embed_dims),
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, geometry_memory, scene_context):
        query = query + scene_context
        update = self.cross_attn(
            query, geometry_memory, geometry_memory, need_weights=False
        )[0]
        query = self.norms[0](query + self.dropout(update))
        update = self.self_attn(query, query, query, need_weights=False)[0]
        query = self.norms[1](query + self.dropout(update))
        return self.norms[2](query + self.dropout(self.ffn(query)))


class TrajectoryVocabularyPlanner(nn.Module):
    """SparseDriveV2 factorized vocabulary conditioned only on V6 geometry.

    V2's deformable image reads are replaced by attention to the V6 agent/map
    geometry tokens. The vocabulary remains the released 1024 paths x 256
    velocity profiles and produces 8 poses at 0.5 second intervals.
    """

    def __init__(
        self,
        path_anchor,
        velocity_anchor,
        trajectory_anchor,
        embed_dims=256,
        feedforward_dims=1024,
        num_heads=8,
        dropout=0.0,
        path_filter_num=(128, 20),
        velocity_filter_num=(64, 10),
        path_sigma=4.0,
        velocity_sigma=4.0,
        trajectory_sigma=4.0,
        interval=0.5,
        require_full_horizon=False,
        metrics=None,
        metric_loss_weight=5.0,
    ):
        super().__init__()
        if len(path_filter_num) != len(velocity_filter_num):
            raise ValueError("path and velocity filters must have equal length")

        path = _navsim_to_sparsedrive(np.load(path_anchor))
        velocity = np.asarray(np.load(velocity_anchor), dtype=np.float32)
        trajectory_data = np.load(trajectory_anchor)
        trajectory = _navsim_to_sparsedrive(trajectory_data["trajectory"])
        trajectory_mask = np.asarray(
            trajectory_data["trajectory_mask"], dtype=np.float32
        )

        if trajectory.shape[:2] != (len(path), len(velocity)):
            raise ValueError(
                "trajectory vocabulary axes do not match path/velocity anchors"
            )
        if trajectory.shape[-2] != velocity.shape[-1]:
            raise ValueError("trajectory and velocity horizons do not match")

        self.register_buffer("path_vocab", torch.from_numpy(path))
        self.register_buffer("velocity_vocab", torch.from_numpy(velocity))
        self.register_buffer("trajectory_vocab", torch.from_numpy(trajectory))
        self.register_buffer(
            "trajectory_mask", torch.from_numpy(trajectory_mask)
        )

        self.path_filter_num = tuple(path_filter_num)
        self.velocity_filter_num = tuple(velocity_filter_num)
        self.path_sigma = path_sigma
        self.velocity_sigma = velocity_sigma
        self.trajectory_sigma = trajectory_sigma
        self.interval = interval
        self.require_full_horizon = require_full_horizon

        self.path_encoder = nn.Sequential(
            nn.Linear(path.shape[-2] * path.shape[-1], feedforward_dims),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dims, embed_dims),
        )
        self.velocity_encoder = nn.Sequential(
            nn.Linear(velocity.shape[-1], feedforward_dims),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dims, embed_dims),
        )
        self.command_encoder = nn.Linear(3, embed_dims)

        block_args = (embed_dims, feedforward_dims, num_heads, dropout)
        self.path_blocks = nn.ModuleList(
            [_CandidateBlock(*block_args) for _ in self.path_filter_num]
        )
        self.velocity_blocks = nn.ModuleList(
            [_CandidateBlock(*block_args) for _ in self.velocity_filter_num]
        )
        self.path_scorers = nn.ModuleList(
            [self._score_head(embed_dims, feedforward_dims) for _ in self.path_filter_num]
        )
        self.velocity_scorers = nn.ModuleList(
            [
                self._score_head(embed_dims, feedforward_dims)
                for _ in self.velocity_filter_num
            ]
        )
        self.trajectory_block = _CandidateBlock(*block_args)
        self.trajectory_scorer = self._score_head(
            embed_dims, feedforward_dims
        )

        # V2 PDM metric heads: one binary head per EPDMS sub-metric on the
        # composed candidate embedding, supervised by live PDM simulation of
        # the candidates against the sample's metric cache. When enabled,
        # final selection uses the combined metric score instead of the
        # trajectory-imitation score, matching released V2.
        self.metrics = tuple(metrics) if metrics else ()
        self.metric_loss_weight = metric_loss_weight
        if self.metrics and set(V2_METRICS) - set(self.metrics):
            # The selection formula reads all eight v2 sub-metric heads.
            raise ValueError(
                "metrics must include the full v2 EPDMS set: "
                f"{sorted(set(V2_METRICS) - set(self.metrics))} missing"
            )
        if self.metrics:
            self.metric_heads = nn.ModuleDict(
                {
                    metric: self._score_head(embed_dims, feedforward_dims)
                    for metric in self.metrics
                }
            )

    @staticmethod
    def _score_head(embed_dims, feedforward_dims):
        return nn.Sequential(
            nn.Linear(embed_dims, feedforward_dims),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dims, 1),
        )

    @staticmethod
    def _gather(sequence, indices):
        shape = indices.shape + sequence.shape[2:]
        index = indices.reshape(indices.shape + (1,) * (sequence.ndim - 2))
        return torch.gather(sequence, 1, index.expand(shape))

    def forward(self, scene_context, geometry_memory, command=None):
        batch_size = scene_context.shape[0]
        if command is None:
            command = scene_context.new_zeros(batch_size, 3)
        command_context = self.command_encoder(command.to(scene_context.dtype))
        scene_context = scene_context + command_context.unsqueeze(1)

        path_vocab = self.path_vocab.unsqueeze(0).expand(batch_size, -1, -1, -1)
        velocity_vocab = self.velocity_vocab.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        path_feature = self.path_encoder(path_vocab.flatten(-2))
        velocity_feature = self.velocity_encoder(velocity_vocab)
        path_indices = torch.arange(
            path_vocab.shape[1], device=path_vocab.device
        ).unsqueeze(0).expand(batch_size, -1)
        velocity_indices = torch.arange(
            velocity_vocab.shape[1], device=velocity_vocab.device
        ).unsqueeze(0).expand(batch_size, -1)

        path_scores, path_candidates = [], []
        velocity_scores, velocity_candidates = [], []
        for layer_idx, (path_block, velocity_block) in enumerate(
            zip(self.path_blocks, self.velocity_blocks)
        ):
            path_feature = path_block(
                path_feature, geometry_memory, scene_context
            )
            velocity_feature = velocity_block(
                velocity_feature, geometry_memory, scene_context
            )
            path_score = self.path_scorers[layer_idx](path_feature).squeeze(-1)
            velocity_score = self.velocity_scorers[layer_idx](
                velocity_feature
            ).squeeze(-1)
            path_scores.append(path_score)
            velocity_scores.append(velocity_score)
            path_candidates.append(path_vocab)
            velocity_candidates.append(velocity_vocab)

            path_k = min(self.path_filter_num[layer_idx], path_score.shape[1])
            velocity_k = min(
                self.velocity_filter_num[layer_idx], velocity_score.shape[1]
            )
            path_topk = path_score.topk(path_k, dim=1).indices
            velocity_topk = velocity_score.topk(velocity_k, dim=1).indices
            path_feature = self._gather(path_feature, path_topk)
            path_vocab = self._gather(path_vocab, path_topk)
            path_indices = self._gather(path_indices, path_topk)
            velocity_feature = self._gather(velocity_feature, velocity_topk)
            velocity_vocab = self._gather(velocity_vocab, velocity_topk)
            velocity_indices = self._gather(velocity_indices, velocity_topk)

        trajectory = self.trajectory_vocab[
            path_indices.unsqueeze(2), velocity_indices.unsqueeze(1)
        ]
        trajectory_mask = self.trajectory_mask[
            path_indices.unsqueeze(2), velocity_indices.unsqueeze(1)
        ]
        trajectory_feature = (
            path_feature.unsqueeze(2) + velocity_feature.unsqueeze(1)
        ).flatten(1, 2)
        trajectory_feature = self.trajectory_block(
            trajectory_feature, geometry_memory, scene_context
        )
        trajectory_score = self.trajectory_scorer(
            trajectory_feature
        ).squeeze(-1)
        trajectory = trajectory.flatten(1, 2)
        trajectory_mask = trajectory_mask.flatten(1, 2)
        if self.require_full_horizon:
            valid = trajectory_mask.bool().all(dim=-1)
            # Coarse path/velocity pruning is independent, so it can rarely
            # produce a set with no fully valid pair. Fall back to the
            # released V2 behavior instead of killing every scorer gradient.
            valid = valid | ~valid.any(dim=-1, keepdim=True)
            trajectory_score = trajectory_score.masked_fill(~valid, -1e4)
        else:
            valid = torch.ones_like(trajectory_score, dtype=torch.bool)

        output = {
            "path_scores": path_scores,
            "path_candidates": path_candidates,
            "velocity_scores": velocity_scores,
            "velocity_candidates": velocity_candidates,
            "trajectory_scores": trajectory_score,
            "trajectory_candidates": trajectory,
            "trajectory_valid": valid,
            "path_indices": path_indices,
            "velocity_indices": velocity_indices,
        }
        if self.metrics:
            output["metric_logits"] = {
                metric: self.metric_heads[metric](trajectory_feature).squeeze(-1)
                for metric in self.metrics
            }
        return output

    @staticmethod
    def _soft_cross_entropy(logits, target, sample_weight):
        loss = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        sample_weight = sample_weight.to(loss.dtype).reshape(-1)
        return (loss * sample_weight).sum() / sample_weight.sum().clamp(min=1)

    def _path_target(self, trajectory, trajectory_mask):
        """Fallback 1 m path target from the available 4 s trajectory."""
        batch_size = trajectory.shape[0]
        num_path_points = self.path_vocab.shape[-2]
        target = trajectory.new_zeros(batch_size, num_path_points, 3)
        target_mask = trajectory.new_zeros(batch_size, num_path_points)
        sample_distance = torch.arange(
            1, num_path_points + 1, device=trajectory.device,
            dtype=trajectory.dtype,
        )
        origin = trajectory.new_zeros(1, 2)

        for batch_idx in range(batch_size):
            count = int(trajectory_mask[batch_idx].sum().item())
            if count == 0:
                continue
            points = torch.cat([origin, trajectory[batch_idx, :count]], dim=0)
            segments = points[1:] - points[:-1]
            distances = torch.cat(
                [trajectory.new_zeros(1), segments.norm(dim=-1).cumsum(dim=0)]
            )
            total_distance = distances[-1]
            valid = sample_distance <= total_distance
            if not valid.any():
                continue
            right = torch.searchsorted(distances, sample_distance[valid])
            right = right.clamp(min=1, max=len(points) - 1)
            left = right - 1
            span = (distances[right] - distances[left]).clamp(min=1e-6)
            ratio = (sample_distance[valid] - distances[left]) / span
            xy = points[left] + ratio.unsqueeze(-1) * (
                points[right] - points[left]
            )
            target[batch_idx, valid, :2] = xy
            target[batch_idx, valid, 2] = torch.atan2(
                segments[right - 1, 1], segments[right - 1, 0]
            )
            target_mask[batch_idx, valid] = 1
        return target, target_mask

    def loss(self, output, data):
        gt_delta = data["gt_ego_fut_trajs"].float()
        gt_mask = data["gt_ego_fut_masks"].float()
        gt_trajectory = gt_delta.cumsum(dim=-2)
        sample_weight = data.get("gt_ego_fut_cmd_valid")
        if sample_weight is None:
            sample_weight = gt_delta.new_ones(gt_delta.shape[0])

        target_path = data.get("gt_ego_fut_path")
        target_path_mask = data.get("gt_ego_fut_path_mask")
        if target_path is None or target_path_mask is None:
            target_path, target_path_mask = self._path_target(
                gt_trajectory, gt_mask
            )
        else:
            target_path = target_path.float()
            target_path_mask = target_path_mask.float()

        losses = {}
        for idx, (scores, candidates) in enumerate(
            zip(output["path_scores"], output["path_candidates"])
        ):
            distance = (candidates[..., :2] - target_path[:, None, :, :2]).pow(2)
            distance = distance.sum(dim=-1) * target_path_mask[:, None]
            distance = distance.sum(dim=-1) / target_path_mask.sum(
                dim=-1, keepdim=True
            ).clamp(min=1)
            target = (-distance * self.path_sigma * candidates.shape[-2]).softmax(
                dim=-1
            )
            losses[f"planning_path_loss_{idx}"] = self._soft_cross_entropy(
                scores, target, sample_weight
            )

        target_velocity = gt_delta.norm(dim=-1) / self.interval
        for idx, (scores, candidates) in enumerate(
            zip(output["velocity_scores"], output["velocity_candidates"])
        ):
            distance = (candidates - target_velocity[:, None]).abs()
            distance = (distance * gt_mask[:, None]).sum(dim=-1)
            target = (-distance * self.velocity_sigma).softmax(dim=-1)
            losses[f"planning_velocity_loss_{idx}"] = self._soft_cross_entropy(
                scores, target, sample_weight
            )

        candidates = output["trajectory_candidates"]
        distance = (candidates[..., :2] - gt_trajectory[:, None]).pow(2).sum(-1)
        distance = (distance * gt_mask[:, None]).sum(dim=-1)
        if self.require_full_horizon:
            distance = distance.masked_fill(~output["trajectory_valid"], 1e4)
        target = (-distance * self.trajectory_sigma).softmax(dim=-1)
        losses["planning_trajectory_loss"] = self._soft_cross_entropy(
            output["trajectory_scores"], target, sample_weight
        )

        if self.metrics:
            losses.update(
                self._metric_losses(output, data, sample_weight)
            )
        return losses

    @staticmethod
    def _get_scorer():
        """Indirection so tests can stub the navsim-dependent scorer."""
        from .pdm_metric_scorer import get_pdm_sub_scores

        return get_pdm_sub_scores

    def _metric_losses(self, output, data, sample_weight):
        """BCE against live PDM sub-scores of the composed candidates.

        Matches V2's custom_decoder: per-metric BCE (0.5 sub-scores zeroed),
        weight metric_loss_weight, ground truth from simulating candidates
        against the sample's metric cache. Samples without a cache (or with
        cmd_valid == 0) drop out of the loss instead of failing.
        """
        cache_paths = data.get("metric_cache_path")
        candidates = output["trajectory_candidates"]
        # Zero-valued but grad-connected, so DDP always sees the metric
        # heads participate even when a batch has no usable caches.
        zero = sum(
            logits.sum() for logits in output["metric_logits"].values()
        ) * 0.0
        losses = {
            f"planning_metric_{metric}_loss": zero.clone()
            for metric in self.metrics
        }
        if cache_paths is None:
            return losses
        get_pdm_sub_scores = self._get_scorer()

        weight = sample_weight.to(candidates.dtype).reshape(-1)
        paths = [
            path if weight[idx] > 0 else None
            for idx, path in enumerate(cache_paths)
        ]
        candidates_navsim = _sparsedrive_to_navsim_torch(candidates)
        sub_scores = get_pdm_sub_scores(
            candidates_navsim.detach().cpu().double().numpy(), paths
        )
        scored = [idx for idx, s in enumerate(sub_scores) if s is not None]
        if not scored:
            return losses

        for metric in self.metrics:
            logits = output["metric_logits"][metric][scored]
            target = torch.as_tensor(
                np.stack(
                    [sub_scores[idx][metric] for idx in scored]
                ).astype(np.float32)
            ).to(logits.device)
            target[target == 0.5] = 0.0
            losses[f"planning_metric_{metric}_loss"] = (
                F.binary_cross_entropy_with_logits(logits, target)
                * self.metric_loss_weight
            )
        return losses

    def _combined_metric_score(self, metric_logits):
        """V2's EPDMS-style selection score from the metric head sigmoids."""
        multiplicative = (
            metric_logits["no_at_fault_collisions"].sigmoid()
            * metric_logits["drivable_area_compliance"].sigmoid()
            * metric_logits["driving_direction_compliance"].sigmoid()
            * metric_logits["traffic_light_compliance"].sigmoid()
        )
        weighted = (
            5 * metric_logits["time_to_collision_within_bound"].sigmoid()
            + 5 * metric_logits["ego_progress"].sigmoid()
            + 2 * metric_logits["lane_keeping"].sigmoid()
            + 2 * metric_logits["history_comfort"].sigmoid()
        )
        return multiplicative * weighted

    @torch.no_grad()
    def decode(self, output):
        if self.metrics:
            scores = self._combined_metric_score(output["metric_logits"])
        else:
            scores = output["trajectory_scores"].softmax(dim=-1)
        mode = scores.argmax(dim=-1)
        batch = torch.arange(mode.shape[0], device=mode.device)
        final = output["trajectory_candidates"][batch, mode, :, :2]
        results = []
        for batch_idx in range(mode.shape[0]):
            candidate_xy = output["trajectory_candidates"][
                batch_idx, :, :, :2
            ].cpu()
            candidate_score = scores[batch_idx].cpu()
            results.append(
                {
                    # Preserve V1's three-command result shape for existing
                    # visualization tools. The candidates are already
                    # conditioned on the active command, so this is a view.
                    "planning_score": candidate_score.unsqueeze(0).expand(3, -1),
                    "planning": candidate_xy.unsqueeze(0).expand(3, -1, -1, -1),
                    "final_planning": final[batch_idx].cpu(),
                    "path_indices": output["path_indices"][batch_idx].cpu(),
                    "velocity_indices": output["velocity_indices"][
                        batch_idx
                    ].cpu(),
                }
            )
        return results
