"""Detection mAP as a function of range cutoff, for B2D-schema results.

Adapted from SparseDrive/tools/range_ap_navsim.py for Bench2Drive: maps raw
gt_names through the config's NameMapping to the 8 canonical detection
classes (matching b2d_3d_dataset.py's evaluator), then for each R in
--ranges restricts GT+pred to BEV center distance <= R and computes
nuScenes-style AP at 0.5/1/2/4 m via det_eval.evaluate_detection.

    PYTHONPATH=SparseDrive python tools_b2d/range_ap_b2d.py \
        --infos /path/b2d_infos_val.pkl \
        --results /path/results.pkl \
        --config SparseDrive/projects/configs/sparsedrive_b2d_stage1.py \
        --label b2d_val_stage1_ep20 \
        --ranges 5 10 20 30 40 50 \
        --out work_dirs/range_ap_b2d_val.json
"""

import argparse
import io
import json
import os
import pickle
import sys

import numpy as np


def load_results(path):
    import torch
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", required=True, help="B2D SparseDrive config with class_names / NameMapping")
    ap.add_argument("--label", default="")
    ap.add_argument("--ranges", type=float, nargs="+", default=[5, 10, 20, 30, 40, 50])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    import importlib
    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.evaluation.detection.det_eval import evaluate_detection

    from mmcv import Config
    cfg = Config.fromfile(args.config)
    name_mapping = cfg.NameMapping
    class_names = [c for c in cfg.class_names if c != "others"]

    infos = pickle.load(open(args.infos, "rb"))
    if isinstance(infos, dict):
        infos = infos["infos"]
    print(f"[load] {len(infos)} infos; loading results ...", flush=True)
    results = load_results(args.results)
    assert len(results) == len(infos), (len(results), len(infos))

    gts, preds = [], []
    for info, res in zip(infos, results):
        raw_names = info["gt_names"]
        boxes = info["gt_boxes"]
        keep = np.array([n in name_mapping and name_mapping[n] in class_names for n in raw_names])
        names = np.array([name_mapping[n] for n in raw_names[keep]])
        b = boxes[keep]
        v = b[:, 7:9].copy() if b.shape[1] >= 9 else np.zeros((len(b), 2))
        v[np.isnan(v)] = 0
        gts.append(dict(boxes=b[:, :7], names=names, velocity=v))

        d = res.get("img_bbox", res)
        b3d = d["boxes_3d"]
        b3d = b3d.tensor.numpy() if hasattr(b3d, "tensor") else np.asarray(b3d)
        b3d = np.asarray(b3d, dtype=np.float64)
        preds.append(dict(
            boxes=b3d[:, :7],
            scores=np.asarray(d["scores_3d"], dtype=np.float64),
            labels=np.asarray(d["labels_3d"], dtype=np.int64),
            velocity=b3d[:, 7:9] if b3d.shape[1] >= 9 else np.zeros((len(b3d), 2)),
        ))
    del results

    gt_dist = np.concatenate([np.linalg.norm(g["boxes"][:, :2], axis=1) for g in gts])
    rows = []
    for R in args.ranges:
        met = evaluate_detection(gts, preds, class_names=class_names, class_range=float(R), verbose=False)
        in_r = gt_dist <= R
        row = dict(range_m=R, frames=len(gts), gt_boxes=int(in_r.sum()),
                   mAP=float(met["mean_ap"]), NDS=float(met["nd_score"]),
                   class_ap={c: float(np.nanmean(list(met["label_aps"][c].values()))) for c in class_names})
        rows.append(row)
        print(f"[{args.label}] R<={R:>4.0f} m  GT {row['gt_boxes']:>6d}  mAP {row['mAP']:.4f}  NDS {row['NDS']:.4f}",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(label=args.label, infos=args.infos, results=args.results, class_names=class_names, rows=rows),
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
