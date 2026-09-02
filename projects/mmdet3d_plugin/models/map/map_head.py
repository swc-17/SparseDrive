import torch

from mmcv.runner import force_fp32
from mmcv.utils import build_from_cfg
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from ..detection3d.detection3d_head import Sparse4DHead

__all__ = ["SparseMapHead"]


@HEADS.register_module()
class SparseMapHead(Sparse4DHead):
    """Map head with per-instance traffic-light attributes on stop lines.

    The map refine layer (``SparsePoint3DRefinementModule`` with
    ``with_tl_branch=True``) returns a (bs, num_anchor, 4) tensor via its
    third output slot — 2 traffic-light offset dims (TL BEV position
    relative to the predicted polyline mean) + 2 state logits (0=green,
    1=red) — which the shared ``Sparse4DHead.forward`` threads through as
    ``quality``. This subclass adds the TL losses on top of the standard
    cls/pts losses:

    - offset: L1 between the predicted offset and (gt_tl_xy - detached
      predicted polyline mean), normalized by roi_size like the line loss,
      on matched queries whose GT carries a mapped TL position;
    - state: classification on matched queries whose GT state is known
      (state -1 = unknown is masked out, not a third class).

    GT comes from ``gt_map_tl`` (num_inst, 3) = (tl_x, tl_y, state)
    aligned with ``gt_map_labels`` (non-stop-line instances carry
    (nan, nan, -1)).
    """

    def __init__(
        self,
        *args,
        loss_tl_offset: dict = None,
        loss_tl_state: dict = None,
        gt_tl_key: str = "gt_map_tl",
        roi_size=(30, 60),
        **kwargs,
    ):
        super(SparseMapHead, self).__init__(*args, **kwargs)
        self.gt_tl_key = gt_tl_key
        self.roi_size = roi_size
        self.loss_tl_offset = build_from_cfg(loss_tl_offset, LOSSES)
        self.loss_tl_state = build_from_cfg(loss_tl_state, LOSSES)

    @force_fp32(apply_to=("model_outs",))
    def loss(self, model_outs, data, feature_maps=None):
        output = super(SparseMapHead, self).loss(
            model_outs, data, feature_maps
        )

        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        tl_preds = model_outs["quality"]
        for decoder_idx, (cls, reg, tl) in enumerate(
            zip(cls_scores, reg_preds, tl_preds)
        ):
            if tl is None:
                continue
            reg = reg[..., : len(self.reg_weights)]
            _, reg_target, _, tl_target = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
                tl_targets=data[self.gt_tl_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            # matched queries (same criterion as the base pts loss)
            matched = torch.logical_not(
                torch.all(reg_target == 0, dim=-1)
            )

            bs, num_pred = matched.shape
            roi = tl.new_tensor(
                [self.roi_size[0], self.roi_size[1]]
            )

            # ---- TL offset: relative to the detached predicted polyline
            # mean (identical to inference decoding), normalized by roi
            tl_xy_target = tl_target[..., :2]
            offset_valid = matched & torch.isfinite(tl_xy_target).all(-1)
            num_offset = max(
                reduce_mean(
                    torch.sum(offset_valid).to(dtype=reg.dtype)
                ),
                1.0,
            )
            if offset_valid.any():
                pred_mean = (
                    reg.reshape(bs, num_pred, -1, 2).mean(dim=2).detach()
                )
                offset_target = (tl_xy_target - pred_mean) / roi
                offset_pred = tl[..., :2] / roi
                offset_loss = self.loss_tl_offset(
                    offset_pred[offset_valid],
                    offset_target[offset_valid],
                    avg_factor=num_offset,
                )
            else:
                offset_loss = tl[..., :2].sum() * 0
            output[
                f"{self.task_prefix}_loss_tl_offset_{decoder_idx}"
            ] = offset_loss

            # ---- TL state: green/red on matched queries with known GT
            # state (unknown = -1 is masked, not trained)
            state_target = tl_target[..., 2]
            state_valid = matched & (state_target >= 0)
            num_state = max(
                reduce_mean(
                    torch.sum(state_valid).to(dtype=reg.dtype)
                ),
                1.0,
            )
            if state_valid.any():
                state_loss = self.loss_tl_state(
                    tl[..., 2:4][state_valid],
                    state_target[state_valid].long(),
                    avg_factor=num_state,
                )
            else:
                state_loss = tl[..., 2:4].sum() * 0
            output[
                f"{self.task_prefix}_loss_tl_state_{decoder_idx}"
            ] = state_loss

        return output
