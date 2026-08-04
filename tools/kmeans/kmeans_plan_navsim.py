"""NAVSIM ego-planning anchors — mirrors tools/kmeans/kmeans_plan.py.

Clusters cumulative 8-step (4 s at 0.5 s) ego future positions per command
branch [right, left, straight] from a converted NAVSIM info pkl, giving
anchors of shape (3, K, EGO_FUT_TS, 2).

NAVSIM deltas vs the nuScenes generator:
  - EGO_FUT_TS = 8, so full-horizon selection requires mask.sum() == 8;
  - rows with gt_ego_fut_cmd_valid == 0 ('unknown' mapped to straight) are
    EXCLUDED — otherwise unknown commands contaminate the straight-branch
    anchors even though training loss is masked (PORT doc section 4);
  - a command with < K valid samples falls back to the all-command pool,
    same as the nuScenes generator.

Usage:
    python tools/kmeans/kmeans_plan_navsim.py \
        --info data/infos/navsim_infos_navmini.pkl \
        --out data/kmeans/kmeans_plan_6_navsim.npy

TODO(navtrain): regenerate from the converted navtrain split before any
real (non-overfit) training run; navmini-only anchors are acceptable for
the Phase 3 overfit gate only.
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

EGO_FUT_TS = 8
COMMANDS = ["right", "left", "straight"]  # SD command order

parser = argparse.ArgumentParser()
parser.add_argument("--info", default="data/infos/navsim_infos_navmini.pkl")
parser.add_argument("--out", default="data/kmeans/kmeans_plan_6_navsim.npy")
parser.add_argument("--num-clusters", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
K = args.num_clusters

with open(args.info, "rb") as f:
    data = pickle.load(f)

navi_trajs = [[], [], []]
n_invalid = 0
for info in data["infos"]:
    if float(info["gt_ego_fut_cmd_valid"]) == 0.0:
        n_invalid += 1
        continue  # never cluster command-invalid (unknown) rows
    plan_mask = info["gt_ego_fut_masks"]
    if plan_mask.sum() != EGO_FUT_TS:
        continue
    plan_traj = info["gt_ego_fut_trajs"].cumsum(axis=-2)  # (8, 2) positions
    cmd = int(info["gt_ego_fut_cmd"].argmax(axis=-1))
    navi_trajs[cmd].append(plan_traj[None])

print(f"{data['metadata']['version']}: "
      + ", ".join(f"{c}={len(t)}" for c, t in zip(COMMANDS, navi_trajs))
      + f", excluded cmd-invalid={n_invalid}")

all_trajs_flat = []
for t in navi_trajs:
    all_trajs_flat.extend(t)

clusters = []
for cmd_name, trajs in zip(COMMANDS, navi_trajs):
    if len(trajs) < K:
        print(f"WARNING: only {len(trajs)} samples for '{cmd_name}', "
              f"using global pool.")
        trajs = all_trajs_flat
    flat = np.concatenate(trajs, axis=0).reshape(-1, EGO_FUT_TS * 2)
    centers = KMeans(n_clusters=K, random_state=args.seed).fit(
        flat).cluster_centers_
    clusters.append(centers.reshape(K, EGO_FUT_TS, 2))

clusters = np.stack(clusters, axis=0)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.save(args.out, clusters)
print(f"saved {clusters.shape} -> {args.out}")
