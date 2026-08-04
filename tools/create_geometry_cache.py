"""Build the geometry cache for the geometric-planner experiments.

Runs the FROZEN official stage1 model (temporal det + map) sequentially over a
split and stores, per frame, the geometric abstraction of the scene:

    det_anchor   (K_det, 11) fp32  raw decoded boxes [x,y,z,logw,logl,logh,
                                    sin_yaw,cos_yaw,vx,vy,vz]  (lidar frame)
    det_conf     (K_det,)    fp32  max sigmoid class score
    det_label    (K_det,)    int8  argmax class id (selection/debug only)
    det_id       (K_det,)    int64 temporal tracking instance id (-1 = none)
    det_feat     (K_det,256) fp16  image instance feature — C1 CONTROL ONLY
    map_anchor   (K_map, 40) fp32  raw polyline points (20 x 2, roi frame)
    map_conf     (K_map,)    fp32
    map_feat     (K_map,256) fp16  image map feature — C1 CONTROL ONLY
    ego_status   (10,)       fp32  [accel(3), rot_rate(3), vel(3), steer]
                                    (MEASURED, from the infos pkl)
    ego_recursive_v  float   fp32  VY of the ego_anchor actually consumed by
                                    prepare_planning THIS frame — i.e. the
                                    t-1 predicted plan-status velocity cached
                                    by cache_planning (0.0 at scene start).
                                    Only present when the model has a
                                    motion_plan_head (stage2 caches).
    plan_status_pred (10,)   fp32  THIS frame's predicted plan status (last
                                    refine layer); index 6 is the velocity
                                    that becomes t+1's ego_recursive_v.
    T_global     (4,4)       fp32  lidar->global
    token / scene_token / timestamp

One pkl per scene (frames in temporal order). GT for planner training is NOT
duplicated here — join with the infos pkl by token at train time.

Usage:
    PYTHONPATH=. python tools/create_geometry_cache.py --split val
    PYTHONPATH=. python tools/create_geometry_cache.py --split train
"""
import argparse
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(repo_root)
sys.path.insert(0, repo_root)

import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
import projects.mmdet3d_plugin  # noqa: register custom modules
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

SPLITS = {
    "val": "projects/configs/declutter/stage1_official_cache_val.py",
    "train": "projects/configs/declutter/stage1_official_cache_train.py",
}
CKPT = "ckpt/sparsedrive_stage1.pth"
K_DET = 50
K_MAP = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--config", default=None,
                    help="config override (default: frozen stage1 cache config for the split)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out-root", default="data/geometry_cache")
    ap.add_argument("--limit-scenes", type=int, default=0,
                    help="stop after N scenes (smoke test)")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config or SPLITS[args.split])
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=6,
                              dist=False, shuffle=False)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    infos = dataset.data_infos
    out_dir = Path(args.out_root) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    head = model.module.head          # SparseDriveHead (stage1: det+map only)
    original_fwd = head.forward.__func__

    scenes = {}
    frame_idx = [0]

    # ── ego kinematic input hooks (stage2 caches only) ─────────────────────
    # prepare_planning builds the ego_anchor consumed by THIS frame's planner;
    # its VY slot holds prev_ego_status[..., 6] — the plan-status velocity
    # predicted at t-1 and stored by cache_planning at the END of the t-1
    # forward (masked to 0 at scene start). We wrap both methods: the
    # prepare_planning wrapper fires DURING the forward, before cache_planning
    # overwrites prev_ego_status with this frame's prediction, so the captured
    # VY is exactly the recursive input used for this frame.
    ego_holder = {}
    mp_head = getattr(head, "motion_plan_head", None)
    if mp_head is not None:
        from projects.mmdet3d_plugin.core.box3d import VY

        iq = mp_head.instance_queue
        orig_prepare = iq.prepare_planning
        orig_cache_planning = iq.cache_planning

        def wrapped_prepare(*a, **kw):
            ego_feature, ego_anchor = orig_prepare(*a, **kw)
            ego_holder["ego_recursive_v"] = float(ego_anchor[0, 0, VY])
            return ego_feature, ego_anchor

        def wrapped_cache_planning(ego_feature, ego_status):
            status = ego_status.detach().cpu().float().numpy().reshape(-1)
            assert status.shape == (10,), f"plan_status shape {status.shape}"
            ego_holder["plan_status_pred"] = status
            return orig_cache_planning(ego_feature, ego_status)

        iq.prepare_planning = wrapped_prepare
        iq.cache_planning = wrapped_cache_planning

    def topk_gather(conf, k, *tensors):
        top_conf, idx = torch.topk(conf, k, dim=0)
        return top_conf, idx, [t[idx] for t in tensors]

    def hooked_fwd(self, feature_maps, metas):
        det_output, map_output, motion_output, planning_output = original_fwd(
            self, feature_maps, metas)
        info = infos[frame_idx[0]]

        # ── detection: top-K by max sigmoid class score ─────────────────────
        cls = det_output["classification"][-1][0].sigmoid()        # (900, 10)
        conf, labels = cls.max(dim=-1)                             # (900,)
        anchors = det_output["prediction"][-1][0]                  # (900, 11)
        feats = det_output["instance_feature"][0]                  # (900, 256)
        ids = det_output.get("instance_id")
        ids = ids[0] if ids is not None else torch.full_like(conf, -1).long()
        top_conf, idx, (top_anchor, top_feat, top_label, top_id) = topk_gather(
            conf, K_DET, anchors, feats, labels, ids)

        # ── map: top-K by max sigmoid class score ───────────────────────────
        mcls = map_output["classification"][-1][0].sigmoid()       # (100, 3)
        mconf, _ = mcls.max(dim=-1)
        manchors = map_output["prediction"][-1][0]                 # (100, 40)
        mfeats = map_output["instance_feature"][0]                 # (100, 256)
        mtop_conf, midx, (mtop_anchor, mtop_feat) = topk_gather(
            mconf, K_MAP, manchors, mfeats)

        meta = metas["img_metas"][0]
        entry = dict(
            token=info["token"],
            scene_token=info["scene_token"],
            timestamp=float(info["timestamp"]),
            det_anchor=top_anchor.detach().cpu().float().numpy(),
            det_conf=top_conf.detach().cpu().float().numpy(),
            det_label=top_label.detach().cpu().numpy().astype(np.int8),
            det_id=top_id.detach().cpu().numpy().astype(np.int64),
            det_feat=top_feat.detach().cpu().half().numpy(),
            map_anchor=mtop_anchor.detach().cpu().float().numpy(),
            map_conf=mtop_conf.detach().cpu().float().numpy(),
            map_feat=mtop_feat.detach().cpu().half().numpy(),
            ego_status=np.asarray(info["ego_status"], dtype=np.float32),
            T_global=np.asarray(meta["T_global"], dtype=np.float32),
        )
        if mp_head is not None:
            entry["ego_recursive_v"] = np.float32(
                ego_holder.pop("ego_recursive_v"))
            entry["plan_status_pred"] = ego_holder.pop("plan_status_pred")
        scenes.setdefault(info["scene_token"], []).append(entry)
        frame_idx[0] += 1
        return det_output, map_output, motion_output, planning_output

    head.forward = types.MethodType(hooked_fwd, head)

    n_frames = len(infos)
    print(f"[cache] split={args.split}  frames={n_frames}")
    stop_after = None
    if args.limit_scenes:
        # find frame index where the (limit+1)-th scene starts
        seen = []
        for i, info in enumerate(infos):
            if info["scene_token"] not in seen:
                seen.append(info["scene_token"])
            if len(seen) > args.limit_scenes:
                stop_after = i
                break

    with torch.no_grad():
        for i, data in enumerate(mmcv.track_iter_progress(loader)):
            if stop_after is not None and i >= stop_after:
                break
            model(return_loss=False, rescale=True, **data)

    print(f"\n[cache] writing {len(scenes)} scenes -> {out_dir}")
    total = 0
    for scene_token, frames in scenes.items():
        p = out_dir / f"{scene_token}.pkl"
        with open(p, "wb") as f:
            pickle.dump(frames, f)
        total += p.stat().st_size
    print(f"[cache] done: {len(scenes)} scenes, "
          f"{total/1e6:.1f} MB total, "
          f"{total/1e3/max(frame_idx[0],1):.1f} KB/frame")


if __name__ == "__main__":
    main()
