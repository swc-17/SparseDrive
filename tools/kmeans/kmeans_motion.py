import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

def lidar2agent(trajs_offset, boxes):
    origin = np.zeros((trajs_offset.shape[0], 1, 2), dtype=np.float32)
    trajs_offset = np.concatenate([origin, trajs_offset], axis=1)
    trajs = trajs_offset.cumsum(axis=1)
    yaws = - boxes[:, 6]
    rot_sin = np.sin(yaws)
    rot_cos = np.cos(yaws)
    rot_mat_T = np.stack(
        [
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos]),
        ]
    )
    trajs_new = np.einsum('aij,jka->aik', trajs, rot_mat_T)
    trajs_new = trajs_new[:, 1:]
    return trajs_new

K = 6
DIS_THRESH = 55

fp = 'data/infos/nuscenes_infos_train.pkl'
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
intention = dict()
for i in range(len(CLASSES)):
    intention[i] = []
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    boxes = info['gt_boxes']
    names = info['gt_names']
    fut_masks = info['gt_agent_fut_masks']
    trajs = info['gt_agent_fut_trajs']
    velos = info['gt_velocity']
    labels = []
    for cat in names:
        if cat in CLASSES:
            labels.append(CLASSES.index(cat))
        else:
            labels.append(-1)
    labels = np.array(labels)
    if len(boxes) == 0:
        continue    
    for i in range(len(CLASSES)):
        cls_mask = (labels == i)
        box_cls = boxes[cls_mask]
        fut_masks_cls = fut_masks[cls_mask]
        trajs_cls = trajs[cls_mask]
        velos_cls = velos[cls_mask]

        distance = np.linalg.norm(box_cls[:, :2], axis=1)
        mask = np.logical_and(
            fut_masks_cls.sum(axis=1) == 12,
            distance < DIS_THRESH,
        )
        trajs_cls = trajs_cls[mask]
        box_cls = box_cls[mask]
        velos_cls = velos_cls[mask]

        trajs_agent = lidar2agent(trajs_cls, box_cls)
        if trajs_agent.shape[0] == 0:
            continue
        intention[i].append(trajs_agent)

clusters = []
for i in range(len(CLASSES)):
    intention_cls = np.concatenate(intention[i], axis=0).reshape(-1, 24)
    if intention_cls.shape[0] < K:
        continue
    cluster = KMeans(n_clusters=K).fit(intention_cls).cluster_centers_
    cluster = cluster.reshape(-1, 12, 2)
    clusters.append(cluster)
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1])
    plt.savefig(f'vis/kmeans/motion_intention_{CLASSES[i]}_{K}', bbox_inches='tight')
    plt.close()

clusters = np.stack(clusters, axis=0)
np.save(f'data/kmeans/kmeans_motion_{K}.npy', clusters)