"""NAVSIM map anchors — mirrors tools/kmeans/kmeans_map.py.

Clusters the mean point of every map_annos polyline (all three classes)
from a converted, map-populated NAVSIM info pkl into K centers, then
attaches the same fixed vertical 20-point segment (delta_y in [-4, 4],
delta_x = 0) as the nuScenes generator, giving anchors of shape
(K, num_sample, 2).

Usage:
    python tools/kmeans/kmeans_map_navsim.py \
        --info data/infos/navsim_infos_navmini.pkl \
        --out data/kmeans/kmeans_map_100_navsim.npy

TODO(navtrain): regenerate from the converted navtrain split before any
real (non-overfit) training run; navmini-only anchors are acceptable for
the Phase 2 overfit gate only.
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("--info", default="data/infos/navsim_infos_navmini.pkl")
parser.add_argument("--out", default="data/kmeans/kmeans_map_100_navsim.npy")
parser.add_argument("--num-clusters", type=int, default=100)
parser.add_argument("--num-sample", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

with open(args.info, "rb") as f:
    data = pickle.load(f)
assert data["metadata"].get("map_annos_populated"), \
    "info pkl has no map annotations — rerun the converter with maps"

center = []
for info in data["infos"]:
    for cls, geoms in info["map_annos"].items():
        for geom in geoms:
            center.append(geom.mean(axis=0))
center = np.stack(center, axis=0)
print(f"{len(center)} polyline centers from {len(data['infos'])} samples "
      f"({data['metadata']['version']})")

print("clustering ...")
center = KMeans(n_clusters=args.num_clusters, random_state=args.seed).fit(
    center).cluster_centers_
delta_y = np.linspace(-4, 4, args.num_sample)
delta_x = np.zeros([args.num_sample])
delta = np.stack([delta_x, delta_y], axis=-1)
vecs = (center[:, np.newaxis] + delta[np.newaxis]).astype(np.float32)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.save(args.out, vecs)
print(f"saved {vecs.shape} -> {args.out}")
