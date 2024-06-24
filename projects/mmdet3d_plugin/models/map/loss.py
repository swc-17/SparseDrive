import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES
from mmdet.models.losses import l1_loss, smooth_l1_loss


@LOSSES.register_module()
class LinesL1Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        """
            L1 loss. The same as the smooth L1 loss
            Args:
                reduction (str, optional): The method to reduce the loss.
                    Options are "none", "mean" and "sum".
                loss_weight (float, optional): The weight of loss.
        """

        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
                shape: [bs, ...]
            target (torch.Tensor): The learning target of the prediction.
                shape: [bs, ...]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None. 
                it's useful when the predictions are not all valid.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.beta > 0:
            loss = smooth_l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        
        else:
            loss = l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        
        num_points = pred.shape[-1] // 2
        loss = loss / num_points

        return loss*self.loss_weight


@LOSSES.register_module()
class SparseLineLoss(nn.Module):
    def __init__(
        self,
        loss_line,
        num_sample=20,
        roi_size=(30, 60),
    ):
        super().__init__()

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.loss_line = build(loss_line, LOSSES)
        self.num_sample = num_sample
        self.roi_size = roi_size

    def forward(
        self,
        line,
        line_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        **kwargs,
    ):

        output = {}
        line = self.normalize_line(line)
        line_target = self.normalize_line(line_target)
        line_loss = self.loss_line(
            line, line_target, weight=weight, avg_factor=avg_factor
        )
        output[f"{prefix}loss_line{suffix}"] = line_loss

        return output

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
