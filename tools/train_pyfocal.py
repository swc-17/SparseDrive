"""Training entry for environments whose mmcv lacks CUDA ops (local + Lilypad).

Replaces mmcv's sigmoid_focal_loss CUDA extension with a numerically
equivalent pure-PyTorch implementation, then runs tools/train.py unchanged.
The declutter matrix trains through this wrapper everywhere (local dry-runs
and cluster runs share one code path). The python focal loss is bit-identical
to mmdet's py_sigmoid_focal_loss reference (verified 2026-07-20).
"""
import os
import runpy
import sys

import torch
import torch.nn.functional as F

import mmdet.models.losses.focal_loss as _fl


def _py_sigmoid_focal_loss_ext(pred, target, gamma=2.0, alpha=0.25,
                               weight=None, reduction='none'):
    """Pure-python equivalent of mmcv ext sigmoid_focal_loss_forward.

    pred: (N, C) logits; target: (N,) long class index where C = background.
    Returns per-element loss (N, C), matching the CUDA op's 'none' reduction.
    """
    num_classes = pred.size(1)
    one_hot = F.one_hot(target.long(), num_classes + 1)[:, :num_classes]
    one_hot = one_hot.to(pred.dtype)
    # numerically stable form, verbatim mmdet py_sigmoid_focal_loss math
    pred_sigmoid = pred.sigmoid()
    pt = (1 - pred_sigmoid) * one_hot + pred_sigmoid * (1 - one_hot)
    focal_weight = (alpha * one_hot + (1 - alpha) * (1 - one_hot)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, one_hot, reduction='none') * focal_weight
    if weight is not None:
        loss = loss * weight
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


_fl._sigmoid_focal_loss = _py_sigmoid_focal_loss_ext

_here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_here, "train.py"), run_name="__main__")
