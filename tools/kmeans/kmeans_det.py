import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

os.makedirs('data/kmeans', exist_ok=True)
os.makedirs('vis/kmeans', exist_ok=True)

K = 900
DIS_THRESH = 55

fp = 'data/infos/nuscenes_infos_train.pkl'
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
center = []
for idx in tqdm(range(len(data_infos))):
    boxes = data_infos[idx]['gt_boxes'][:,:3]
    if len(boxes) == 0:
        continue
    distance = np.linalg.norm(boxes[:, :2], axis=1)
    center.append(boxes[distance < DIS_THRESH])
center = np.concatenate(center, axis=0)
print("start clustering, may take a few minutes.")
cluster = KMeans(n_clusters=K).fit(center).cluster_centers_
plt.scatter(cluster[:,0], cluster[:,1])
plt.savefig(f'vis/kmeans/det_anchor_{K}', bbox_inches='tight')
others = np.array([1,1,1,1,0,0,0,0])[np.newaxis].repeat(K, axis=0)
cluster = np.concatenate([cluster, others], axis=1)
np.save(f'data/kmeans/kmeans_det_{K}.npy', cluster)