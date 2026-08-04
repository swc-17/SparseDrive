"""Dataset over the frozen-stage1 geometry cache.

Each item provides ONLY geometric quantities (plus image features when the C1
control explicitly asks for them):

    agent_geo   (50, 12)  [x,y,z,logw,logl,logh,sin,cos,vx,vy,vz, conf]
    agent_feat  (50, 256) image instance features   (C1 only, else zeros)
    agent_hist  (50, H, 11) ego-compensated ID-matched past boxes
    agent_hist_mask (50, H) 1 = valid history slot
    agent_fut   (50, F, 2) future xy of the same instance in the CURRENT frame
    agent_fut_mask (50, F)                                 (G3 targets)
    map_geo     (10, 41)  [40 polyline coords, conf]
    map_feat    (10, 256) image map features            (C1 only, else zeros)
    ego_anchor  (11,)     kinematic ego anchor (VY = velocity per variant)
    gt_ego_fut_trajs (6,2), gt_ego_fut_masks (6,), gt_ego_fut_cmd (3,),
    ego_status (10,)
"""
import glob
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

EGO_ANCHOR_BASE = np.array(
    [0, 0.5, -1.84 + 1.56 / 2, np.log(4.08), np.log(1.73), np.log(1.56),
     1, 0, 0, 0, 0], dtype=np.float32)
VY = 9


def _relative_transform(T_global_src, T_global_dst):
    """4x4 transform taking points from src lidar frame to dst lidar frame."""
    return np.linalg.inv(T_global_dst) @ T_global_src


def _transform_boxes(boxes, T):
    """Transform (N, 11) boxes by 4x4 rigid T (numpy)."""
    out = boxes.copy()
    R = T[:3, :3]
    out[:, :3] = boxes[:, :3] @ R.T + T[:3, 3]
    # yaw: rotate [cos, sin] by the 2x2 rotation
    cos_sin = boxes[:, [7, 6]]
    rot = cos_sin @ T[:2, :2].T
    out[:, 6] = rot[:, 1]
    out[:, 7] = rot[:, 0]
    out[:, 8:11] = boxes[:, 8:11] @ R.T
    return out


class GeoCacheDataset(Dataset):
    def __init__(self, cache_dir, infos_pkl, hist_len=4, fut_len=3,
                 load_image_feats=False):
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.load_image_feats = load_image_feats

        # GT lookup by token
        with open(infos_pkl, "rb") as f:
            d = pickle.load(f)
        infos = d["infos"] if isinstance(d, dict) else d
        # match NuScenes3DDataset.load_annotations ordering exactly —
        # planning_eval iterates the mmdet dataset, which sorts by timestamp
        infos = list(sorted(infos, key=lambda e: e["timestamp"]))
        self.gt = {
            i["token"]: dict(
                trajs=np.asarray(i["gt_ego_fut_trajs"], dtype=np.float32),
                masks=np.asarray(i["gt_ego_fut_masks"], dtype=np.float32),
                cmd=np.asarray(i["gt_ego_fut_cmd"], dtype=np.float32),
                status=np.asarray(i["ego_status"], dtype=np.float32),
            )
            for i in infos
        }
        # preserve the dataset (sorted-infos) order for aligned evaluation
        self.token_order = [i["token"] for i in infos]

        # scene pkls
        self.scenes = {}
        self.index = []  # (scene_token, frame_idx)
        for p in sorted(glob.glob(os.path.join(cache_dir, "*.pkl"))):
            with open(p, "rb") as f:
                frames = pickle.load(f)
            st = frames[0]["scene_token"]
            self.scenes[st] = frames
            for j in range(len(frames)):
                self.index.append((st, j))
        # order items to match infos order (evaluation alignment)
        by_token = {self.scenes[s][j]["token"]: (s, j) for s, j in self.index}
        self.index = [by_token[t] for t in self.token_order if t in by_token]
        assert len(self.index) == len(self.token_order), (
            f"cache/info mismatch: {len(self.index)} vs {len(self.token_order)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        st, j = self.index[i]
        frames = self.scenes[st]
        fr = frames[j]
        K = fr["det_anchor"].shape[0]

        agent_geo = np.concatenate(
            [fr["det_anchor"], fr["det_conf"][:, None]], axis=1)  # (50, 12)

        # ── ID-matched history, ego-compensated into the current frame ──────
        H = self.hist_len
        hist = np.zeros((K, H, 11), dtype=np.float32)
        hist_mask = np.zeros((K, H), dtype=np.float32)
        id_now = fr["det_id"]
        for h in range(1, H + 1):
            if j - h < 0:
                break
            past = frames[j - h]
            T = _relative_transform(past["T_global"], fr["T_global"])
            past_boxes = _transform_boxes(past["det_anchor"], T)
            pos = {pid: k for k, pid in enumerate(past["det_id"]) if pid >= 0}
            for k in range(K):
                pid = id_now[k]
                if pid >= 0 and pid in pos:
                    hist[k, h - 1] = past_boxes[pos[pid]]
                    hist_mask[k, h - 1] = 1.0

        # ── ID-matched future (self-supervised forecast targets, G3) ────────
        F = self.fut_len
        fut = np.zeros((K, F, 2), dtype=np.float32)
        fut_mask = np.zeros((K, F), dtype=np.float32)
        for h in range(1, F + 1):
            if j + h >= len(frames):
                break
            nxt = frames[j + h]
            T = _relative_transform(nxt["T_global"], fr["T_global"])
            nxt_boxes = _transform_boxes(nxt["det_anchor"], T)
            pos = {pid: k for k, pid in enumerate(nxt["det_id"]) if pid >= 0}
            for k in range(K):
                pid = id_now[k]
                if pid >= 0 and pid in pos:
                    fut[k, h - 1] = nxt_boxes[pos[pid], :2]
                    fut_mask[k, h - 1] = 1.0

        map_geo = np.concatenate(
            [fr["map_anchor"], fr["map_conf"][:, None]], axis=1)  # (10, 41)

        gt = self.gt[fr["token"]]
        ego_anchor = EGO_ANCHOR_BASE.copy()  # VY filled by the model per variant

        out = dict(
            agent_geo=torch.from_numpy(agent_geo),
            agent_hist=torch.from_numpy(hist),
            agent_hist_mask=torch.from_numpy(hist_mask),
            agent_fut=torch.from_numpy(fut),
            agent_fut_mask=torch.from_numpy(fut_mask),
            map_geo=torch.from_numpy(map_geo),
            ego_anchor=torch.from_numpy(ego_anchor),
            ego_status=torch.from_numpy(gt["status"]),
            gt_ego_fut_trajs=torch.from_numpy(gt["trajs"]),
            gt_ego_fut_masks=torch.from_numpy(gt["masks"]),
            gt_ego_fut_cmd=torch.from_numpy(gt["cmd"]),
        )
        if self.load_image_feats:
            out["agent_feat"] = torch.from_numpy(
                fr["det_feat"].astype(np.float32))
            out["map_feat"] = torch.from_numpy(
                fr["map_feat"].astype(np.float32))
        else:
            out["agent_feat"] = torch.zeros(K, 256)
            out["map_feat"] = torch.zeros(map_geo.shape[0], 256)
        return out
