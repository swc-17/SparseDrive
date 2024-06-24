from typing import Optional, List

import torch

from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class SparsePoint3DDecoder(object):
    def __init__(
        self,
        coords_dim: int = 2,
        score_threshold: Optional[float] = None,
    ):
        super(SparsePoint3DDecoder, self).__init__()
        self.score_threshold = score_threshold
        self.coords_dim = coords_dim

    def decode(
        self,
        cls_scores,
        pts_preds,
        instance_id=None,
        quality=None,
        output_idx=-1,
    ):
        bs, num_pred, num_cls = cls_scores[-1].shape
        cls_scores = cls_scores[-1].sigmoid()
        pts_preds = pts_preds[-1].reshape(bs, num_pred, -1, self.coords_dim)
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            num_pred, dim=1
        )
        cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold
        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            scores = cls_scores[i]
            pts = pts_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                pts = pts[mask[i]]

            output.append(
                {
                    "vectors": [vec.detach().cpu().numpy() for vec in pts],
                    "scores": scores.detach().cpu().numpy(),
                    "labels": category_ids.detach().cpu().numpy(),
                }
            )
        return output