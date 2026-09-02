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
        instance_feature=None,
        anchor_embed=None,
    ):
        bs, num_pred, num_cls = cls_scores[-1].shape
        cls_scores = cls_scores[-1].sigmoid()
        pts_preds = pts_preds[-1].reshape(bs, num_pred, -1, self.coords_dim)
        # traffic-light attribute predictions from the map TL branch
        # (2 offset dims + 2 state logits), threaded via the quality slot
        tl_preds = None
        if quality is not None and quality[-1] is not None:
            tl_preds = quality[-1]
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

            entry = {
                "vectors": [vec.detach().cpu().numpy() for vec in pts],
                "scores": scores.detach().cpu().numpy(),
                "labels": category_ids.detach().cpu().numpy(),
            }
            if tl_preds is not None:
                tl = tl_preds[i, indices[i] // num_cls]
                if self.score_threshold is not None:
                    tl = tl[mask[i]]
                # absolute TL BEV position = polyline mean + offset
                tl_xy = pts.mean(dim=1) + tl[:, :2]
                state_scores = tl[:, 2:4].sigmoid()
                entry["tl_xy"] = tl_xy.detach().cpu().numpy()
                # 0=green, 1=red (meaningful for stop_line instances only)
                entry["tl_state"] = (
                    state_scores.argmax(-1).detach().cpu().numpy()
                )
                entry["tl_score"] = (
                    state_scores.max(-1).values.detach().cpu().numpy()
                )
            # Pass through map instance_feature and anchor_embed for C-JEPA
            if instance_feature is not None:
                feat = instance_feature[i, indices[i] // num_cls]
                if self.score_threshold is not None:
                    feat = feat[mask[i]]
                entry["map_instance_feature"] = feat.detach()
            if anchor_embed is not None:
                emb = anchor_embed[i, indices[i] // num_cls]
                if self.score_threshold is not None:
                    emb = emb[mask[i]]
                entry["map_anchor_embed"] = emb.detach()
            output.append(entry)
        return output