import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

K = 100
num_sample = 20

fp = 'data/infos/nuscenes_infos_train.pkl'
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
center = []
for idx in tqdm(range(len(data_infos))):
    for cls, geoms in data_infos[idx]["map_annos"].items():
        for geom in geoms:  
            center.append(geom.mean(axis=0))
center = np.stack(center, axis=0)
center = KMeans(n_clusters=K).fit(center).cluster_centers_
delta_y = np.linspace(-4, 4, num_sample)
delta_x = np.zeros([num_sample])
delta = np.stack([delta_x, delta_y], axis=-1)
vecs = center[:, np.newaxis] + delta[np.newaxis]

for i in range(K):
    x = vecs[i, :, 0]
    y = vecs[i, :, 1]
    plt.plot(x, y, linewidth=1, marker='o', linestyle='-', markersize=2)
plt.savefig(f'vis/kmeans/map_anchor_{K}', bbox_inches='tight')
np.save(f'data/kmeans/kmeans_map_{K}.npy', vecs)