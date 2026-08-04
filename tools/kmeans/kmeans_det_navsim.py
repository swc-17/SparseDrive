"""NAVSIM detection anchors — mirrors tools/kmeans/kmeans_det.py.

Clusters gt box centers (within DIS_THRESH of ego) from a converted NAVSIM
info pkl into K anchor centers, then appends the same fixed
[l, w, h, yaw_sin/cos, vel] tail as the nuScenes generator
(ones for dims, zeros for yaw/vel), giving anchors of shape (K, 11).

Usage:
    python tools/kmeans/kmeans_det_navsim.py \
        --info data/infos/navsim_infos_navmini.pkl \
        --out data/kmeans/kmeans_det_900_navsim.npy

TODO(navtrain): regenerate from the converted navtrain split before any
real (non-overfit) training run; navmini-only anchors are acceptable for
the Phase 1 overfit gate only.
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("--info", default="data/infos/navsim_infos_navmini.pkl")
parser.add_argument("--out", default="data/kmeans/kmeans_det_900_navsim.npy")
parser.add_argument("--num-clusters", type=int, default=900)
parser.add_argument("--dist-thresh", type=float, default=55.0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

with open(args.info, "rb") as f:
    data = pickle.load(f)

center = []
for info in data["infos"]:
    boxes = info["gt_boxes"][:, :3]
    if len(boxes) == 0:
        continue
    distance = np.linalg.norm(boxes[:, :2], axis=1)
    center.append(boxes[distance < args.dist_thresh])
center = np.concatenate(center, axis=0)
print(f"{len(center)} box centers from {len(data['infos'])} samples "
      f"({data['metadata']['version']})")

print("clustering ...")
cluster = KMeans(n_clusters=args.num_clusters, random_state=args.seed).fit(
    center).cluster_centers_
others = np.array([1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].repeat(
    args.num_clusters, axis=0)
cluster = np.concatenate([cluster, others], axis=1)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.save(args.out, cluster)
print(f"saved {cluster.shape} -> {args.out}")
