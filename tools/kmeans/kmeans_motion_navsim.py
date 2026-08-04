"""NAVSIM agent-motion anchors — mirrors tools/kmeans/kmeans_motion.py.

Clusters agent-frame future trajectories per NAVSIM class from a converted
NAVSIM info pkl into K modes each, giving anchors of shape
(num_classes, K, FUT_TS, 2) indexed by the NAVSIM class order (the head
does self.motion_anchor[cls_ids], so EVERY class needs a correctly indexed
row — see NAVSIM_SPARSEDRIVE_PORT.md "Regenerate all learned anchors").

NAVSIM deltas vs the nuScenes generator:
  - 7 classes in the fixed NAVSIM order;
  - agent futures carry at most 10 of the FUT_TS=12 slots (NUM_FUTURE=10),
    so full-track selection requires mask.sum() == 10, not 12; the two
    trailing anchor steps stay zero-offset (loss is mask-zeroed there);
  - static classes (traffic_cone, barrier, czone_sign, generic_object) and
    any class with < K usable tracks get an all-zero (static) anchor row,
    the documented intentional zero-motion policy.

Usage:
    python tools/kmeans/kmeans_motion_navsim.py \
        --info data/infos/navsim_infos_navmini.pkl \
        --out data/kmeans/kmeans_motion_6_navsim.npy

TODO(navtrain): regenerate from the converted navtrain split before any
real (non-overfit) training run; navmini-only anchors are acceptable for
the Phase 3 overfit gate only.
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

CLASSES = [
    "vehicle",
    "pedestrian",
    "bicycle",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
]
FUT_TS = 12          # anchor slots (converter FUT_TS)
FUT_AVAILABLE = 10   # NAVSIM windows store 10 future frames

parser = argparse.ArgumentParser()
parser.add_argument("--info", default="data/infos/navsim_infos_navmini.pkl")
parser.add_argument("--out", default="data/kmeans/kmeans_motion_6_navsim.npy")
parser.add_argument("--num-clusters", type=int, default=6)
parser.add_argument("--dist-thresh", type=float, default=55.0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
K = args.num_clusters


def lidar2agent(trajs_offset, boxes):
    """Rotate BEV-frame per-step offsets into each agent's heading frame."""
    origin = np.zeros((trajs_offset.shape[0], 1, 2), dtype=np.float64)
    trajs_offset = np.concatenate([origin, trajs_offset], axis=1)
    trajs = trajs_offset.cumsum(axis=1)
    yaws = -boxes[:, 6]
    rot_sin = np.sin(yaws)
    rot_cos = np.cos(yaws)
    rot_mat_T = np.stack(
        [
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos]),
        ]
    )
    trajs_new = np.einsum("aij,jka->aik", trajs, rot_mat_T)
    return trajs_new[:, 1:]


with open(args.info, "rb") as f:
    data = pickle.load(f)

per_class = {i: [] for i in range(len(CLASSES))}
for info in data["infos"]:
    boxes = info["gt_boxes"]
    if len(boxes) == 0:
        continue
    names = info["gt_names"]
    fut_masks = info["gt_agent_fut_masks"]
    trajs = info["gt_agent_fut_trajs"]
    labels = np.array(
        [CLASSES.index(c) if c in CLASSES else -1 for c in names]
    )
    distance = np.linalg.norm(boxes[:, :2], axis=1)
    full = np.logical_and(
        fut_masks.sum(axis=1) >= FUT_AVAILABLE,
        distance < args.dist_thresh,
    )
    for i in range(len(CLASSES)):
        mask = np.logical_and(labels == i, full)
        if mask.any():
            per_class[i].append(lidar2agent(trajs[mask], boxes[mask]))

clusters = np.zeros((len(CLASSES), K, FUT_TS, 2), dtype=np.float64)
for i, name in enumerate(CLASSES):
    if len(per_class[i]) == 0:
        print(f"{name}: 0 tracks -> zero/static anchor")
        continue
    tracks = np.concatenate(per_class[i], axis=0)
    if tracks.shape[0] < K:
        print(f"{name}: only {tracks.shape[0]} tracks (<{K}) "
              f"-> zero/static anchor")
        continue
    flat = tracks.reshape(-1, FUT_TS * 2)
    centers = KMeans(n_clusters=K, random_state=args.seed).fit(
        flat).cluster_centers_
    clusters[i] = centers.reshape(K, FUT_TS, 2)
    # anchors are cumulative agent-frame positions (lidar2agent cumsums)
    print(f"{name}: {tracks.shape[0]} tracks -> {K} modes "
          f"(max |xy| {np.abs(clusters[i]).max():.2f} m)")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.save(args.out, clusters.astype(np.float64))
print(f"saved {clusters.shape} -> {args.out}")
