"""NAVSIM -> SparseDrive info-PKL converter (Phase 0 of the NAVSIM port).

Converts NAVSIM (OpenScene / nuPlan 1.2 schema) log pickles into SparseDrive
info PKLs that implement the full SparseDrive training contract:

- one sample row per official scenario token (navmini: 396, navtest: 12146),
  backed by a deduplicated frame table for temporal replay;
- coordinate contract NAVSIM-local -> SparseDrive BEV:
      x_sd = -y_nav,  y_sd = x_nav,  yaw_sd = wrap(yaw_nav + pi/2)
  applied to box centers/yaw, velocities, agent/ego trajectories, camera
  extrinsics and temporal (ego2global) poses;
- fixed 7-class taxonomy in the order
      vehicle, pedestrian, bicycle, traffic_cone, barrier, czone_sign,
      generic_object;
- deterministic integer instance ids from globally sorted track_tokens;
- agent futures joined by track_token (fut_ts=12, masks stop at first
  absence; NAVSIM windows provide at most 10 future frames);
- 8-step ego future at 0.5 s (4 s horizon) with masks;
- command mapping NAVSIM [left, straight, right, unknown] -> SparseDrive
  one-hot [right, left, straight]; unknown falls back to straight with
  gt_ego_fut_cmd_valid = 0;
- 10-value ego status from can_bus (accel xyz, angular rate xyz, velocity
  xyz, steering=0), asserted against ego_dynamic_state within 1e-6;
- sequence_id grouping over connected log frames (link+timestamp based, NOT
  raw scene_token);
- 8 cameras in the fixed order F0, L0, L1, L2, R0, R1, R2, B0 with full
  intrinsics/extrinsics/distortion.

Phase 2: local vector maps. Each sample's ``map_annos`` is populated from the
nuPlan map GPKGs via ``NavsimMapExtractor`` (ped_crossing / divider /
boundary / stop_line, labels 0/1/2/3), clipped to the same (30, 60) BEV ROI
the nuScenes configs use and expressed in the SparseDrive local frame — the
exact schema ``nuscenes_converter.geom2anno`` emits (dict label -> list of
float64 (N, 2) polylines). ``--no-maps`` restores the Phase-0 empty-dict
behavior.

Traffic lights: each sample also carries ``map_tl_annos``, a float64 (N, 3)
array aligned with ``map_annos[3]`` (the stop lines): mean controlling
traffic-light bulb position in the SD local frame (NaN when the map has no
bulb for the bar) and per-frame state 1=red / 0=green / -1=unknown resolved
from the log's per-lane-connector ``traffic_lights`` field via the GPKG
``lane_connectors.traffic_light_stop_line_fids`` linkage.

The converter reads the raw log pickles directly (plain pickle files); it does
not import navsim or the nuplan devkit.

Example:
    python tools/data_converter/navsim_converter.py \
        --data-root /media/applied/navsim \
        --split mini \
        --filter navmini \
        --output data/infos/navsim_infos_navmini.pkl \
        --overlay-dir docs/phase0_overlays --overlay-num 32
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
from collections import Counter, OrderedDict

import numpy as np
import yaml
from pyquaternion import Quaternion

CONVERTER_VERSION = "navsim_converter_v1.3"  # v1.3: stop_line map class
# (label 3, traffic-light-controlled stop bars) + per-sample map_tl_annos
# (traffic-light position + red/green state aligned with the stop lines).
# v1.2: --workers (per-log multiprocessing, serial-equivalent) +
# --split-json/--split-key frozen log-split restriction + streamed pickle
# write for navtrain-scale outputs

# Map classes in the fixed SparseDrive order (labels 0/1/2/3).
MAP_CLASSES = ("ped_crossing", "divider", "boundary", "stop_line")
STOP_LINE_LABEL = MAP_CLASSES.index("stop_line")
# BEV ROI for local map extraction: x in [-15, 15], y in [-30, 30] in the
# SparseDrive frame — identical to the nuScenes configs' roi_size.
MAP_ROI_SIZE = (30.0, 60.0)

# Fixed class taxonomy and order (do not reorder).
NAVSIM_CLASSES = (
    "vehicle",
    "pedestrian",
    "bicycle",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
)

# Fixed camera order; F0 must be index 0 (planning feature path uses cam 0).
CAMERA_ORDER = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
)

# NAVSIM-local -> SparseDrive BEV rotation.
C3 = np.array(
    [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]]
)
C4 = np.eye(4)
C4[:3, :3] = C3
YAW_OFFSET = np.pi / 2.0

# Radial box retention range (matches SparseDrive's 55 m detection range).
BOX_RANGE_M = 55.0

# Temporal horizons.
NUM_HISTORY = 4   # incl. current
NUM_FUTURE = 10
FUT_TS = 12       # agent future steps (12 slots, at most 10 filled)
EGO_FUT_TS = 8    # ego planning steps at 0.5 s (4 s)

# Valid inter-frame interval (s) for connected-run grouping.
DT_MIN, DT_MAX = 0.49, 0.51


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_yaw(q_wxyz):
    """Yaw of a wxyz quaternion (nuPlan convention, yaw_pitch_roll[0])."""
    return Quaternion(*q_wxyz).yaw_pitch_roll[0]


def pose_to_matrix(translation, rotation_wxyz):
    m = np.eye(4)
    m[:3, :3] = Quaternion(*rotation_wxyz).rotation_matrix
    m[:3, 3] = np.asarray(translation, dtype=np.float64)
    return m


def nav_to_sd_points(pts):
    """Rotate (..., 2 or 3) NAVSIM-local points/vectors into SparseDrive BEV."""
    pts = np.asarray(pts, dtype=np.float64)
    out = pts.copy()
    out[..., 0] = -pts[..., 1]
    out[..., 1] = pts[..., 0]
    return out


def sd_to_nav_points(pts):
    pts = np.asarray(pts, dtype=np.float64)
    out = pts.copy()
    out[..., 0] = pts[..., 1]
    out[..., 1] = -pts[..., 0]
    return out


def coordinate_roundtrip_selftest():
    """Gate: C and its inverse round-trip points/yaw/poses below 1e-5."""
    rng = np.random.RandomState(0)
    pts = rng.uniform(-100, 100, size=(1000, 3))
    err = np.abs(sd_to_nav_points(nav_to_sd_points(pts)) - pts).max()
    assert err < 1e-5, f"point round-trip error {err}"
    # matrix vs formula agreement
    err2 = np.abs((C3 @ pts.T).T - nav_to_sd_points(pts)).max()
    assert err2 < 1e-5, f"C-matrix mismatch {err2}"
    yaw = rng.uniform(-np.pi, np.pi, size=1000)
    yaw_sd = wrap_angle(yaw + YAW_OFFSET)
    err3 = np.abs(wrap_angle(wrap_angle(yaw_sd - YAW_OFFSET) - yaw)).max()
    assert err3 < 1e-5, f"yaw round-trip error {err3}"
    # yaw offset consistent with C: heading unit vector must rotate with C
    v = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=-1)
    v_sd = nav_to_sd_points(v)
    err4 = np.abs(
        wrap_angle(np.arctan2(v_sd[:, 1], v_sd[:, 0]) - yaw_sd)
    ).max()
    assert err4 < 1e-5, f"yaw/C consistency error {err4}"
    # 4x4 pose round-trip
    G = pose_to_matrix([1.0, 2.0, 0.5], Quaternion(axis=[0, 0, 1], radians=0.3))
    G_sd = G @ np.linalg.inv(C4)
    err5 = np.abs(G_sd @ C4 - G).max()
    assert err5 < 1e-5, f"pose round-trip error {err5}"


def load_scene_filter(filter_name):
    """Load a vendored NAVSIM scene-filter yaml (hydra keys ignored)."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "navsim_filters",
        f"{filter_name}.yaml",
    )
    with open(path) as f:
        cfg = yaml.safe_load(f)
    num_history = cfg.get("num_history_frames", 4)
    num_future = cfg.get("num_future_frames", 10)
    frame_interval = cfg.get("frame_interval") or (num_history + num_future)
    return dict(
        num_history_frames=num_history,
        num_future_frames=num_future,
        num_frames=num_history + num_future,
        frame_interval=frame_interval,
        has_route=cfg.get("has_route", True),
        max_scenes=cfg.get("max_scenes"),
        log_names=cfg.get("log_names"),
        tokens=set(cfg["tokens"]) if cfg.get("tokens") else None,
    )


def iter_log_paths(log_dir, log_names):
    names = sorted(os.listdir(log_dir))
    for name in names:
        if not name.endswith(".pkl"):
            continue
        stem = name[:-4]
        if log_names is not None and stem not in log_names:
            continue
        yield stem, os.path.join(log_dir, name)


def select_windows(frames, filt):
    """NAVSIM SceneLoader/filter_scenes window selection for ONE log's frames.

    Returns list of windows; each window is the list of num_frames raw frame
    dicts; current frame is window[num_history-1].
    """
    nf, fi = filt["num_frames"], filt["frame_interval"]
    cur = filt["num_history_frames"] - 1
    windows = []
    for start in range(0, len(frames), fi):
        window = frames[start : start + nf]
        if len(window) < nf:
            continue
        if filt["has_route"] and len(window[cur]["roadblock_ids"]) == 0:
            continue
        token = window[cur]["token"]
        if filt["tokens"] is not None and token not in filt["tokens"]:
            continue
        windows.append(window)
    return windows


def select_scenes(log_dir, filt):
    """Window selection over all logs; list of (log_name, window)."""
    scenes = []
    for log_name, path in iter_log_paths(log_dir, filt["log_names"]):
        with open(path, "rb") as f:
            frames = pickle.load(f)
        for window in select_windows(frames, filt):
            scenes.append((log_name, window))
            if filt["max_scenes"] and len(scenes) >= filt["max_scenes"]:
                return scenes
    return scenes


def assign_sequence_ids_one_log(frames, seq_id_start=0):
    """Connected-run grouping over ONE log's raw frames.

    A new run starts at the first frame, when prev/next sample links
    disagree, or the timestamp delta is outside [0.49, 0.51] s. Returns
    (token -> sequence_id, num_sequences, num_gap_boundaries).
    """
    seq_of = {}
    seq_id = seq_id_start - 1
    n_gap = 0
    prev = None
    for fr in frames:
        new_run = False
        if prev is None:
            new_run = True  # log change
        else:
            links_ok = (
                prev["sample_next"] == fr["token"]
                and fr["sample_prev"] == prev["token"]
            )
            dt = (fr["timestamp"] - prev["timestamp"]) / 1e6
            if not links_ok or not (DT_MIN <= dt <= DT_MAX):
                new_run = True
                if not (DT_MIN <= dt <= DT_MAX):
                    n_gap += 1
        if new_run:
            seq_id += 1
        seq_of[fr["token"]] = seq_id
        prev = fr
    return seq_of, seq_id + 1 - seq_id_start, n_gap


def assign_sequence_ids(log_dir, filt):
    """Connected-run grouping over all logs (serial path)."""
    seq_of = {}
    n_seq = 0
    n_gap = 0
    for _, path in iter_log_paths(log_dir, filt["log_names"]):
        with open(path, "rb") as f:
            frames = pickle.load(f)
        log_seq_of, log_n_seq, log_n_gap = assign_sequence_ids_one_log(
            frames, seq_id_start=n_seq
        )
        seq_of.update(log_seq_of)
        n_seq += log_n_seq
        n_gap += log_n_gap
    return seq_of, n_seq, n_gap


def build_track_token_table(scenes):
    """Deterministic integer instance ids from sorted global track_tokens."""
    tokens = set()
    for _, window in scenes:
        for fr in window:
            tokens.update(fr["anns"]["track_tokens"])
    ordered = sorted(tokens)
    return ordered, {t: i for i, t in enumerate(ordered)}


def convert_ego_status(fr):
    """10-value SparseDrive ego status from can_bus, with agreement gate."""
    cb = np.asarray(fr["can_bus"], dtype=np.float64)
    eds = np.asarray(fr["ego_dynamic_state"], dtype=np.float64)  # vx,vy,ax,ay
    assert np.abs(cb[10:12] - eds[:2]).max() < 1e-6, (
        f"can_bus velocity disagrees with ego_dynamic_state at {fr['token']}"
    )
    assert np.abs(cb[7:9] - eds[2:]).max() < 1e-6, (
        f"can_bus acceleration disagrees with ego_dynamic_state at {fr['token']}"
    )
    assert np.abs(cb[16:18]).max() == 0.0, (
        f"can_bus[16:18] padding is nonzero at {fr['token']}"
    )
    status = np.concatenate(
        [cb[7:10], cb[13:16], cb[10:13], np.zeros(1)]
    ).astype(np.float32)
    # Ego-axis scalars: NOT rotated (index 6 = forward velocity contract).
    return status


def convert_command(driving_command):
    """NAVSIM [left, straight, right, unknown] -> SD [right, left, straight]."""
    dc = np.asarray(driving_command)
    assert dc.shape == (4,), f"unexpected driving_command shape {dc.shape}"
    idx = int(np.argmax(dc))
    valid = 1.0
    if idx == 0:      # left
        cmd = [0, 1, 0]
    elif idx == 1:    # straight
        cmd = [0, 0, 1]
    elif idx == 2:    # right
        cmd = [1, 0, 0]
    else:             # unknown -> deterministic straight, loss masked
        cmd = [0, 0, 1]
        valid = 0.0
    return np.array(cmd, dtype=np.float32), np.float32(valid)


# Optional prefix that makes cam data_paths absolute in the output infos
# (set via --blob-root; used for exported sims whose blobs live outside the
# dataset root the training configs point at).
BLOB_ROOT = None


def convert_cams(fr):
    """Fixed-order camera dict with SparseDrive-frame extrinsics.

    lidar2ego is identity in NAVSIM logs (asserted), so sensor2lidar equals
    sensor2ego; T_camera_to_sd = C4 @ T_camera_to_nav.
    """
    l2e = np.asarray(fr["lidar2ego"], dtype=np.float64)
    assert np.abs(l2e - np.eye(4)).max() < 1e-9, "lidar2ego is not identity"
    cams = OrderedDict()
    if set(fr["cams"]) == set(CAMERA_ORDER):
        cam_order = CAMERA_ORDER
    else:
        # Non-navsim rig (e.g. native AlpaSim export): keep the exporter's
        # order, which must already put the front camera at index 0 (the
        # planning feature path uses cam 0).
        cam_order = list(fr["cams"])
        assert "front" in cam_order[0], f"cam 0 must be a front camera: {cam_order}"
    for cam in cam_order:
        c = fr["cams"][cam]
        rot_sd = C3 @ np.asarray(c["sensor2lidar_rotation"], dtype=np.float64)
        trans_sd = C3 @ np.asarray(
            c["sensor2lidar_translation"], dtype=np.float64
        )
        cams[cam] = dict(
            data_path=(os.path.join(BLOB_ROOT, c["data_path"])
                       if BLOB_ROOT else c["data_path"]),
            sensor2lidar_rotation=rot_sd,
            sensor2lidar_translation=trans_sd,
            cam_intrinsic=np.asarray(c["cam_intrinsic"], dtype=np.float64),
            distortion=np.asarray(c["distortion"], dtype=np.float64),
        )
    return cams


def ego2global_sd(fr):
    """Synthetic SparseDrive ego2global: G_sd = G_nav @ inv(C4)."""
    g_nav = pose_to_matrix(
        fr["ego2global_translation"], fr["ego2global_rotation"]
    )
    return g_nav @ np.linalg.inv(C4)


def convert_boxes(fr):
    """Boxes/velocities/names for one frame, rotated + range-filtered.

    Returns (gt_boxes(N,7), names(N,), velocity(N,2), track_tokens list).
    """
    anns = fr["anns"]
    boxes = np.asarray(anns["gt_boxes"], dtype=np.float64)
    names = np.asarray(anns["gt_names"])
    vel3 = np.asarray(anns["gt_velocity_3d"], dtype=np.float64)
    tracks = list(anns["track_tokens"])
    if boxes.size == 0:
        return (
            np.zeros((0, 7)), np.zeros((0,), dtype=names.dtype),
            np.zeros((0, 2)), [],
        )
    finite = np.isfinite(boxes).all(axis=1) & np.isfinite(vel3).all(axis=1)
    known = np.isin(names, NAVSIM_CLASSES)
    in_range = np.linalg.norm(boxes[:, :2], axis=1) <= BOX_RANGE_M
    keep = finite & known & in_range
    boxes, names, vel3 = boxes[keep], names[keep], vel3[keep]
    tracks = [t for t, k in zip(tracks, keep) if k]

    out = boxes.copy()
    out[:, :3] = nav_to_sd_points(boxes[:, :3])
    out[:, 6] = wrap_angle(boxes[:, 6] + YAW_OFFSET)
    vel_sd = nav_to_sd_points(vel3)[:, :2]
    return out, names, vel_sd, tracks


def transform_boxes_to_frame(boxes_sd, T_src2dst):
    """Transform (N,7) SD-frame boxes with a 4x4 SD-frame transform."""
    if len(boxes_sd) == 0:
        return boxes_sd.copy()
    out = boxes_sd.copy()
    out[:, :3] = boxes_sd[:, :3] @ T_src2dst[:3, :3].T + T_src2dst[:3, 3]
    cs = np.stack(
        [np.cos(boxes_sd[:, 6]), np.sin(boxes_sd[:, 6])], axis=-1
    )
    cs = cs @ T_src2dst[:2, :2].T
    out[:, 6] = np.arctan2(cs[:, 1], cs[:, 0])
    return out


def relative_se2(origin_xyyaw, poses_xyyaw):
    """nuPlan convert_absolute_to_relative_se2_array equivalent."""
    ox, oy, oyaw = origin_xyyaw
    rot = np.array(
        [[np.cos(oyaw), np.sin(oyaw)], [-np.sin(oyaw), np.cos(oyaw)]]
    )
    poses = np.asarray(poses_xyyaw, dtype=np.float64)
    rel = poses.copy()
    rel[:, :2] = (poses[:, :2] - np.array([ox, oy])) @ rot.T
    rel[:, 2] = wrap_angle(poses[:, 2] - oyaw)
    return rel


def geom2anno(map_geoms):
    """Shapely geometries -> serializable map_annos.

    Matches tools/data_converter/nuscenes_converter.py::geom2anno exactly:
    dict {label int -> list of float64 (N, 2) polylines in the local frame}.
    """
    vectors = {}
    for cls, geom_list in map_geoms.items():
        if cls not in MAP_CLASSES:
            continue  # drivable_area polygons are not vectorized
        label = MAP_CLASSES.index(cls)
        vectors[label] = [np.array(geom.coords) for geom in geom_list]
    return vectors


def embedded_map_annos(fr):
    """Local vector map from frame-embedded polylines (AlpaSim exports).

    Frames exported by tools/navsim_from_alpasim carry ``map_annos_nav``:
    {class name -> [(N, 2) polylines]} already in the NAVSIM ego frame
    (extracted from the scene artifact's trajdata VectorMap — simulated
    scenes have no nuPlan map API). Rotate into the SparseDrive frame and
    clip to the same MAP_ROI_SIZE patch the extractor path uses.
    """
    from shapely.geometry import LineString, box

    patch = box(
        -MAP_ROI_SIZE[0] / 2.0, -MAP_ROI_SIZE[1] / 2.0,
        MAP_ROI_SIZE[0] / 2.0, MAP_ROI_SIZE[1] / 2.0,
    )
    annos = {}
    for cls, lines in fr["map_annos_nav"].items():
        if cls not in MAP_CLASSES:
            continue
        label = MAP_CLASSES.index(cls)
        out = []
        for line in lines:
            pts_sd = np.asarray(line, dtype=np.float64) @ C3[:2, :2].T
            if len(pts_sd) < 2:
                continue
            clipped = LineString(pts_sd).intersection(patch)
            geoms = getattr(clipped, "geoms", [clipped])
            for g in geoms:
                if g.geom_type == "LineString" and len(g.coords) >= 2:
                    out.append(np.array(g.coords, dtype=np.float64))
        if out:
            annos[label] = out
    return annos


def extract_map_annos(map_extractor, fr, g_sd):
    """Local vector map for one frame, in the SparseDrive BEV frame.

    Uses the SD-frame ego2global pose (G_sd = G_nav @ inv(C4)): its SE(2)
    yaw already contains the NAVSIM->SparseDrive rotation contract, so the
    extractor's global->local mapping lands directly in the SD frame.

    Returns:
        (map_annos, map_tl_annos): map_annos is the label->polylines dict;
        map_tl_annos is a float64 (N, 3) array aligned with
        map_annos[STOP_LINE_LABEL]: columns (tl_x, tl_y, state) with the
        mean controlling-traffic-light position in the SD local frame (NaN
        when the map carries no bulb for the stop line) and state 1=red /
        0=green / -1=unknown resolved from the frame's per-lane-connector
        ``traffic_lights`` status.
    """
    yaw_sd = math.atan2(g_sd[1, 0], g_sd[0, 0])
    map_geoms = map_extractor.get_map_geom(
        fr["map_location"], g_sd[:2, 3], yaw_sd
    )
    extras = map_geoms.pop("stop_line_extras")
    annos = geom2anno(map_geoms)
    for label, lines in annos.items():
        for line in lines:
            assert line.ndim == 2 and line.shape[1] == 2, (
                f"map anno shape {line.shape} at {fr['token']}"
            )
            assert np.isfinite(line).all(), (
                f"non-finite map anno at {fr['token']}"
            )

    # per-frame red/green status keyed by lane connector fid
    tl_status = {
        int(cid): bool(is_red)
        for cid, is_red in fr.get("traffic_lights", [])
    }
    rows = []
    for ex in extras:
        if ex["tl_xy"] is None:
            tl_x, tl_y = float("nan"), float("nan")
        else:
            tl_x, tl_y = float(ex["tl_xy"][0]), float(ex["tl_xy"][1])
        states = [
            tl_status[c] for c in ex["connector_fids"] if c in tl_status
        ]
        # red wins over green when connectors sharing the bar disagree
        state = 1.0 if any(states) else (0.0 if states else -1.0)
        rows.append([tl_x, tl_y, state])
    tl_annos = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    assert len(tl_annos) == len(annos.get(STOP_LINE_LABEL, [])), (
        f"stop_line / map_tl_annos misalignment at {fr['token']}"
    )
    return annos, tl_annos


def frame_record(fr, seq_of):
    """Minimal deduplicated frame-table record (for temporal replay)."""
    g_sd = ego2global_sd(fr)
    return dict(
        token=fr["token"],
        log_name=fr["log_name"],
        scene_token=fr["scene_token"],
        frame_idx=fr["frame_idx"],
        timestamp=fr["timestamp"],
        sample_prev=fr["sample_prev"],
        sample_next=fr["sample_next"],
        sequence_id=seq_of[fr["token"]],
        ego2global_translation=g_sd[:3, 3].copy(),
        ego2global_rotation=list(Quaternion(matrix=g_sd[:3, :3], atol=1e-7).q),
        cams=convert_cams(fr),
        ego_status=convert_ego_status(fr),
        lidar_path=fr["lidar_path"],
    )


def convert_scene(log_name, window, filt, seq_of, track2ind,
                  map_extractor=None):
    """One SparseDrive sample row for a NAVSIM scenario window."""
    cur_idx = filt["num_history_frames"] - 1
    fr = window[cur_idx]

    g_sd_cur = ego2global_sd(fr)
    g_sd_cur_inv = np.linalg.inv(g_sd_cur)

    gt_boxes, names, velocity, tracks = convert_boxes(fr)
    n_box = len(gt_boxes)
    instance_inds = np.array(
        [track2ind[t] for t in tracks], dtype=np.int64
    )

    # ---- agent futures joined by track_token (masks stop at first absence)
    fut_frames = window[cur_idx + 1 :]
    fut_maps = []
    fut_T = []
    for ffr in fut_frames:
        fb, _, _, ftracks = convert_boxes(ffr)
        fut_maps.append({t: i for i, t in enumerate(ftracks)})
        fut_T.append((fb, g_sd_cur_inv @ ego2global_sd(ffr)))
    gt_fut_trajs = np.zeros((n_box, FUT_TS, 2), dtype=np.float64)
    gt_fut_masks = np.zeros((n_box, FUT_TS), dtype=np.float64)
    for i, track in enumerate(tracks):
        prev_xy = gt_boxes[i, :2]
        for step, (fmap, (fb, T)) in enumerate(zip(fut_maps, fut_T)):
            j = fmap.get(track)
            if j is None:
                break  # mask stops at first absence
            center = fb[j, :3] @ T[:3, :3].T + T[:3, 3]
            gt_fut_trajs[i, step] = center[:2] - prev_xy
            gt_fut_masks[i, step] = 1.0
            prev_xy = center[:2]

    # ---- future boxes in current SD frame (planning-eval collision boxes)
    fut_boxes = []
    for step in range(min(EGO_FUT_TS, len(fut_T))):
        fb, T = fut_T[step]
        fut_boxes.append(transform_boxes_to_frame(fb, T).astype(np.float64))

    # ---- ego future: 8 steps at 0.5 s, relative SE2 then rotated to SD
    ego_poses_nav = []
    for wfr in window[cur_idx:]:
        yaw = quat_yaw(wfr["ego2global_rotation"])
        t = wfr["ego2global_translation"]
        ego_poses_nav.append([t[0], t[1], yaw])
    rel = relative_se2(ego_poses_nav[0], np.asarray(ego_poses_nav[1:]))
    n_avail = min(EGO_FUT_TS, rel.shape[0])
    ego_fut_pos_sd = np.zeros((EGO_FUT_TS, 2), dtype=np.float64)
    ego_fut_masks = np.zeros((EGO_FUT_TS,), dtype=np.float32)
    ego_fut_pos_sd[:n_avail] = nav_to_sd_points(rel[:n_avail, :2])
    ego_fut_masks[:n_avail] = 1.0
    if n_avail < EGO_FUT_TS:
        ego_fut_pos_sd[n_avail:] = ego_fut_pos_sd[max(n_avail - 1, 0)]
    # cumulative positions -> per-step offsets
    ego_fut_trajs = np.diff(
        np.concatenate([np.zeros((1, 2)), ego_fut_pos_sd], axis=0), axis=0
    )

    cmd, cmd_valid = convert_command(fr["driving_command"])

    window_seq_ids = {seq_of[w["token"]] for w in window}
    window_scene_tokens = {w["scene_token"] for w in window}

    # ---- local vector map + traffic-light attributes for the stop lines
    if "map_annos_nav" in fr:
        # AlpaSim exports carry no stop-polygon / traffic-light layers
        map_annos = embedded_map_annos(fr)
        map_tl_annos = np.zeros((0, 3), dtype=np.float64)
    elif map_extractor is not None:
        map_annos, map_tl_annos = extract_map_annos(map_extractor, fr,
                                                    g_sd_cur)
    else:
        map_annos = {}
        map_tl_annos = np.zeros((0, 3), dtype=np.float64)

    info = dict(
        # identity / temporal linkage
        token=fr["token"],
        log_name=log_name,
        scene_token=fr["scene_token"],
        scene_name=fr["scene_name"],
        frame_idx=fr["frame_idx"],
        timestamp=fr["timestamp"],
        sample_prev=fr["sample_prev"],
        sample_next=fr["sample_next"],
        sequence_id=seq_of[fr["token"]],
        window_has_gap=len(window_seq_ids) > 1,
        window_crosses_scene=len(window_scene_tokens) > 1,
        history_tokens=[w["token"] for w in window[: cur_idx + 1]],
        history_masks=np.ones((cur_idx + 1,), dtype=np.float32),
        future_tokens=[w["token"] for w in fut_frames],
        future_masks=np.ones((len(fut_frames),), dtype=np.float32),
        map_location=fr["map_location"],
        vehicle_name=fr["vehicle_name"],
        roadblock_ids=list(fr["roadblock_ids"]),
        dataset="navsim",
        # poses (SparseDrive virtual frame); lidar2ego is identity in NAVSIM
        lidar_path=fr["lidar_path"],
        lidar2ego_translation=[0.0, 0.0, 0.0],
        lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
        ego2global_translation=g_sd_cur[:3, 3].copy(),
        ego2global_rotation=list(
            Quaternion(matrix=g_sd_cur[:3, :3], atol=1e-7).q
        ),
        sweeps=[],
        cams=convert_cams(fr),
        # detection / tracking
        gt_boxes=gt_boxes.astype(np.float64),
        gt_names=names,
        gt_velocity=velocity.astype(np.float64),
        valid_flag=np.ones((n_box,), dtype=bool),
        instance_inds=instance_inds,
        # motion
        gt_agent_fut_trajs=gt_fut_trajs.astype(np.float32),
        gt_agent_fut_masks=gt_fut_masks.astype(np.float32),
        fut_boxes=fut_boxes,
        # planning
        gt_ego_fut_trajs=ego_fut_trajs.astype(np.float32),
        gt_ego_fut_masks=ego_fut_masks,
        gt_ego_fut_cmd=cmd,
        gt_ego_fut_cmd_valid=cmd_valid,
        ego_status=convert_ego_status(fr),
        # maps: local ped_crossing/divider/boundary/stop_line vectors
        # (labels 0/1/2/3) in the SD frame, clipped to MAP_ROI_SIZE (empty
        # dict if disabled)
        map_annos=map_annos,
        # (N, 3) float64 aligned with map_annos[STOP_LINE_LABEL]: mean
        # traffic-light bulb position (SD local xy, NaN if unmapped) and
        # state 1=red / 0=green / -1=unknown for the current frame
        map_tl_annos=map_tl_annos,
    )
    _check_finite(info)
    return info


_FINITE_KEYS = (
    "gt_boxes", "gt_velocity", "gt_agent_fut_trajs", "gt_agent_fut_masks",
    "gt_ego_fut_trajs", "gt_ego_fut_masks", "gt_ego_fut_cmd", "ego_status",
    "ego2global_translation",
)


def _check_finite(info):
    for k in _FINITE_KEYS:
        v = np.asarray(info[k], dtype=np.float64)
        assert np.isfinite(v).all(), f"non-finite values in {k} at {info['token']}"
    for fb in info["fut_boxes"]:
        assert np.isfinite(fb).all(), f"non-finite fut_boxes at {info['token']}"


# --------------------------------------------------------------------------
# Parallel (per-log) conversion. Serial semantics are preserved exactly:
# logs are processed in sorted order, sequence_ids are per-log ids offset by
# the cumulative count over preceding logs (matching assign_sequence_ids),
# and instance_inds are remapped to the GLOBALLY sorted track_token table
# (matching build_track_token_table).
# --------------------------------------------------------------------------

_WORKER_STATE = {}


def _init_worker(filt, maps_root):
    _WORKER_STATE["filt"] = filt
    if maps_root is not None:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        from projects.mmdet3d_plugin.datasets.map_utils.navsim_map_extractor \
            import NavsimMapExtractor
        _WORKER_STATE["map_extractor"] = NavsimMapExtractor(
            maps_root, MAP_ROI_SIZE)
    else:
        _WORKER_STATE["map_extractor"] = None


def _process_log(item):
    """Convert one log. Returns log-local infos/frames/track table."""
    log_name, path = item
    filt = _WORKER_STATE["filt"]
    map_extractor = _WORKER_STATE["map_extractor"]
    with open(path, "rb") as f:
        raw_frames = pickle.load(f)
    windows = select_windows(raw_frames, filt)
    seq_of, n_seq, n_gap = assign_sequence_ids_one_log(raw_frames)
    tracks = sorted({
        t for w in windows for fr in w for t in fr["anns"]["track_tokens"]
    })
    track2ind = {t: i for i, t in enumerate(tracks)}
    infos = [
        convert_scene(log_name, w, filt, seq_of, track2ind,
                      map_extractor=map_extractor)
        for w in windows
    ]
    frames = {}
    for w in windows:
        for wfr in w:
            if wfr["token"] not in frames:
                frames[wfr["token"]] = frame_record(wfr, seq_of)
    return dict(log_name=log_name, infos=infos, frames=frames,
                tracks=tracks, n_seq=n_seq, n_gap=n_gap)


def _convert_parallel(log_dir, filt, maps_root, workers):
    """Per-log multiprocessing conversion; serial-equivalent output."""
    assert not filt["max_scenes"], "max_scenes unsupported with --workers"
    log_items = list(iter_log_paths(log_dir, filt["log_names"]))
    print(f"[info] converting {len(log_items)} logs with {workers} workers")
    results = []
    ctx = mp.get_context("spawn")  # fiona/pyproj are not fork-safe
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(filt, maps_root)) as pool:
        for i, res in enumerate(pool.imap(_process_log, log_items,
                                          chunksize=1)):
            results.append(res)
            if (i + 1) % 20 == 0 or i + 1 == len(log_items):
                n_sc = sum(len(r["infos"]) for r in results)
                print(f"  converted {i + 1}/{len(log_items)} logs "
                      f"({n_sc} scenarios)")

    # global track table: sorted union over all logs (serial-equivalent)
    track_tokens = sorted(set().union(*(set(r["tracks"]) for r in results))
                          if results else set())
    track2global = {t: i for i, t in enumerate(track_tokens)}

    infos, frames = [], {}
    seq_offset = 0
    n_seq_total = 0
    n_gap_total = 0
    for r in results:  # imap preserves sorted-log order
        remap = np.array([track2global[t] for t in r["tracks"]],
                         dtype=np.int64)
        for info in r["infos"]:
            if len(info["instance_inds"]):
                info["instance_inds"] = remap[info["instance_inds"]]
            info["sequence_id"] += seq_offset
            infos.append(info)
        for token, fr in r["frames"].items():
            fr["sequence_id"] += seq_offset
            frames[token] = fr
        seq_offset += r["n_seq"]
        n_seq_total += r["n_seq"]
        n_gap_total += r["n_gap"]
    return infos, frames, track_tokens, n_seq_total, n_gap_total


def load_split_logs(split_json, split_key):
    """Frozen log-split restriction (e.g. data/splits/navtrain_split_v1.json)."""
    with open(split_json) as f:
        d = json.load(f)
    logs = d[split_key]
    assert isinstance(logs, list) and logs, f"empty split {split_key}"
    return set(logs)


def convert(data_root, split, filter_name, output, count_only=False,
            maps_root=None, workers=0, split_json=None, split_key=None,
            expected_count=None):
    coordinate_roundtrip_selftest()
    print("[gate] coordinate round-trip self-test passed (<1e-5)")

    log_dir = os.path.join(data_root, "navsim_logs", split)
    filt = load_scene_filter(filter_name)
    if split_json is not None:
        split_logs = load_split_logs(split_json, split_key)
        if filt["log_names"] is None:
            filt["log_names"] = sorted(split_logs)
        else:
            before = len(filt["log_names"])
            filt["log_names"] = sorted(set(filt["log_names"]) & split_logs)
            print(f"[info] split restriction {split_key}: "
                  f"{before} filter logs -> {len(filt['log_names'])}")

    if count_only:
        scenes = select_scenes(log_dir, filt)
        print(f"[gate] selected {len(scenes)} scenario tokens for "
              f"{filter_name}")
        return None

    if workers and workers > 1:
        infos, frames, track_tokens, n_seq, n_gap = _convert_parallel(
            log_dir, filt, maps_root, workers)
    else:
        scenes = select_scenes(log_dir, filt)
        print(f"[gate] selected {len(scenes)} scenario tokens for "
              f"{filter_name}")

        seq_of, n_seq, n_gap = assign_sequence_ids(log_dir, filt)
        print(f"[info] {n_seq} connected runs, "
              f"{n_gap} timestamp-gap boundaries")

        track_tokens, track2ind = build_track_token_table(scenes)
        print(f"[info] {len(track_tokens)} unique track_tokens")

        map_extractor = None
        if maps_root is not None:
            # imported lazily: fiona/pyproj are converter-only dependencies
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", ".."))
            from projects.mmdet3d_plugin.datasets.map_utils \
                .navsim_map_extractor import NavsimMapExtractor
            map_extractor = NavsimMapExtractor(maps_root, MAP_ROI_SIZE)

        infos = []
        frames = {}
        for i, (log_name, window) in enumerate(scenes):
            infos.append(convert_scene(log_name, window, filt, seq_of,
                                       track2ind,
                                       map_extractor=map_extractor))
            for wfr in window:
                if wfr["token"] not in frames:
                    frames[wfr["token"]] = frame_record(wfr, seq_of)
            if (i + 1) % 50 == 0 or i + 1 == len(scenes):
                print(f"  converted {i + 1}/{len(scenes)} scenarios")

    print(f"[gate] converted {len(infos)} scenario tokens for {filter_name}")
    expected = {"navmini": 396, "navtest": 12146}.get(filter_name,
                                                      expected_count)
    if expected is not None:
        assert len(infos) == expected, (
            f"{filter_name} expected {expected} tokens, got {len(infos)}"
        )
    print(f"[info] {n_seq} connected runs, {n_gap} timestamp-gap boundaries; "
          f"{len(track_tokens)} unique track_tokens")

    n_gap_windows = sum(1 for x in infos if x["window_has_gap"])
    n_cross = sum(1 for x in infos if x["window_crosses_scene"])
    n_unknown_cmd = sum(1 for x in infos if x["gt_ego_fut_cmd_valid"] == 0)
    class_counts = Counter()
    for x in infos:
        class_counts.update(x["gt_names"].tolist())
    boxes_per_frame = np.array([len(x["gt_boxes"]) for x in infos])

    print(f"[info] windows with timestamp gap: {n_gap_windows}")
    print(f"[info] windows crossing raw scene_token boundary: {n_cross}")
    print(f"[info] unknown-command samples (loss-masked): {n_unknown_cmd}")
    print(f"[info] boxes/frame median {np.median(boxes_per_frame):.0f} "
          f"max {boxes_per_frame.max()}")
    print(f"[info] class counts: {dict(class_counts)}")

    # Embedded (AlpaSim) map annos are populated even without a maps_root.
    maps_enabled = maps_root is not None or any(x["map_annos"] for x in infos)
    map_stats = None
    if maps_enabled:
        per_class = {
            cls: np.array([len(x["map_annos"].get(li, [])) for x in infos])
            for li, cls in enumerate(MAP_CLASSES)
        }
        n_map_empty = int(sum(
            1 for x in infos
            if sum(len(v) for v in x["map_annos"].values()) == 0
        ))
        tl_states = np.concatenate(
            [np.asarray(x["map_tl_annos"]).reshape(-1, 3)[:, 2]
             for x in infos]
        ) if infos else np.zeros((0,))
        tl_pos_nan = np.concatenate(
            [np.isnan(np.asarray(x["map_tl_annos"]).reshape(-1, 3)[:, 0])
             for x in infos]
        ) if infos else np.zeros((0,), dtype=bool)
        stop_line_stats = dict(
            num_red=int((tl_states == 1).sum()),
            num_green=int((tl_states == 0).sum()),
            num_unknown_state=int((tl_states == -1).sum()),
            num_missing_tl_position=int(tl_pos_nan.sum()),
        )
        map_stats = dict(
            roi_size=list(MAP_ROI_SIZE),
            num_empty_map_samples=n_map_empty,
            vectors_per_sample={
                cls: dict(
                    total=int(v.sum()),
                    mean=float(v.mean()),
                    median=float(np.median(v)),
                    max=int(v.max()),
                    num_empty=int((v == 0).sum()),
                )
                for cls, v in per_class.items()
            },
            stop_line_stats=stop_line_stats,
        )
        print(f"[info] map vectors/sample: "
              + ", ".join(f"{c} mean {v.mean():.1f}"
                          for c, v in per_class.items()))
        print(f"[info] samples with fully empty maps: {n_map_empty}")
        print(f"[info] stop-line traffic-light states: {stop_line_stats}")

    metadata = dict(
        version=f"navsim-{split}-{filter_name}",
        converter_version=CONVERTER_VERSION,
        source_split=split,
        source_filter=filter_name,
        classes=list(NAVSIM_CLASSES),
        camera_order=list(CAMERA_ORDER),
        box_range_m=BOX_RANGE_M,
        fut_ts=FUT_TS,
        ego_fut_ts=EGO_FUT_TS,
        interval_s=0.5,
        num_history_frames=filt["num_history_frames"],
        num_future_frames=filt["num_future_frames"],
        num_samples=len(infos),
        num_frames=len(frames),
        num_sequences=n_seq,
        num_gap_boundaries=n_gap,
        num_gap_windows=n_gap_windows,
        num_scene_boundary_windows=n_cross,
        num_unknown_command=n_unknown_cmd,
        class_counts=dict(class_counts),
        num_track_tokens=len(track_tokens),
        coordinate_contract="x_sd=-y_nav, y_sd=x_nav, yaw_sd=wrap(yaw_nav+pi/2)",
        map_annos_populated=maps_enabled,
        map_classes=list(MAP_CLASSES),
        map_stats=map_stats,
        split_json=split_json,
        split_key=split_key,
        num_logs=len(filt["log_names"]) if filt["log_names"] else None,
        expected_count=expected,
    )

    data = dict(
        infos=infos,
        frames=frames,
        track_tokens=track_tokens,
        metadata=metadata,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    # stream the pickle to disk (navtrain payloads are multi-GB; avoid the
    # 2x memory spike of pickle.dumps) and hash the file in chunks
    with open(output, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    sha = hashlib.sha256()
    with open(output, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            sha.update(chunk)
    content_hash = sha.hexdigest()
    size_mb = os.path.getsize(output) / 1e6
    manifest = dict(metadata, content_sha256=content_hash, output=output)
    manifest_path = os.path.splitext(output)[0] + "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote {output} ({size_mb:.1f} MB)")
    print(f"[done] manifest {manifest_path} sha256={content_hash[:16]}...")
    return data


# --------------------------------------------------------------------------
# Calibration overlay artifact (Phase 0 acceptance gate)
# --------------------------------------------------------------------------

def _box_corners_sd(boxes):
    """(N,7) SD boxes -> (N,8,3) corners. Box x-axis = length = heading dir."""
    n = len(boxes)
    corners = np.zeros((n, 8, 3))
    x = boxes[:, 3] / 2.0  # length
    y = boxes[:, 4] / 2.0  # width
    z = boxes[:, 5] / 2.0  # height
    signs = np.array(
        [[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
         [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]],
        dtype=np.float64,
    )
    local = signs[None] * np.stack([x, y, z], axis=-1)[:, None, :]
    cy, sy = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    R = np.zeros((n, 3, 3))
    R[:, 0, 0], R[:, 0, 1] = cy, -sy
    R[:, 1, 0], R[:, 1, 1] = sy, cy
    R[:, 2, 2] = 1.0
    corners = np.einsum("nij,nkj->nki", R, local) + boxes[:, None, :3]
    return corners


_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]
_CLASS_COLORS = {
    "vehicle": (0, 200, 255), "pedestrian": (0, 255, 0),
    "bicycle": (255, 0, 255), "traffic_cone": (0, 128, 255),
    "barrier": (255, 255, 0), "czone_sign": (255, 128, 0),
    "generic_object": (200, 200, 200),
}


def render_overlays(data, data_root, split, out_dir, num=32):
    import cv2

    blob_root = os.path.join(data_root, "sensor_blobs", split)
    os.makedirs(out_dir, exist_ok=True)
    infos = sorted(data["infos"], key=lambda x: x["token"])[:num]
    files = []
    for info in infos:
        views = []
        boxes = info["gt_boxes"]
        corners = _box_corners_sd(boxes)
        for cam, c in info["cams"].items():
            img = cv2.imread(os.path.join(blob_root, c["data_path"]))
            assert img is not None, f"missing image {c['data_path']}"
            R_s2l = c["sensor2lidar_rotation"]
            t_s2l = c["sensor2lidar_translation"]
            R_l2c = R_s2l.T
            t_l2c = -R_s2l.T @ t_s2l
            rvec, _ = cv2.Rodrigues(R_l2c)
            K = c["cam_intrinsic"]
            dist = c["distortion"]
            h, w = img.shape[:2]
            n_seg = 16  # samples per box edge
            for bi in range(len(boxes)):
                pts_cam = corners[bi] @ R_l2c.T + t_l2c
                if pts_cam[:, 2].max() < 0.5:  # behind camera
                    continue
                color = _CLASS_COLORS.get(str(info["gt_names"][bi]),
                                          (255, 255, 255))
                # Draw each edge as sampled segments; a sample is valid only
                # if it is in front of the camera and its *pinhole*
                # projection is near the frame (the distortion polynomial
                # folds far-outside points back into the image).
                for e0, e1 in _EDGES:
                    ts = np.linspace(0.0, 1.0, n_seg + 1)[:, None]
                    pts3d = corners[bi, e0][None] * (1 - ts) \
                        + corners[bi, e1][None] * ts
                    pc = pts3d @ R_l2c.T + t_l2c
                    valid = pc[:, 2] > 0.1
                    if valid.sum() < 2:
                        continue
                    uv_pin = pc @ K.T
                    with np.errstate(divide="ignore", invalid="ignore"):
                        uv_pin = uv_pin[:, :2] / uv_pin[:, 2:3]
                    valid &= (
                        (uv_pin[:, 0] > -0.25 * w)
                        & (uv_pin[:, 0] < 1.25 * w)
                        & (uv_pin[:, 1] > -0.25 * h)
                        & (uv_pin[:, 1] < 1.25 * h)
                    )
                    if valid.sum() < 2:
                        continue
                    uv, _ = cv2.projectPoints(
                        pts3d.astype(np.float64), rvec,
                        t_l2c.astype(np.float64), K, dist,
                    )
                    uv = np.clip(np.round(uv.reshape(-1, 2)), -1e5, 1e5)
                    for si in range(n_seg):
                        if not (valid[si] and valid[si + 1]):
                            continue
                        p0 = (int(uv[si, 0]), int(uv[si, 1]))
                        p1 = (int(uv[si + 1, 0]), int(uv[si + 1, 1]))
                        cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)
            img = cv2.resize(img, (640, 360))
            cv2.putText(img, cam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
            views.append(img)
        top = np.concatenate(views[:4], axis=1)
        bot = np.concatenate(views[4:], axis=1)
        mosaic = np.concatenate([top, bot], axis=0)
        out_path = os.path.join(out_dir, f"{info['token']}.jpg")
        cv2.imwrite(out_path, mosaic)
        files.append(out_path)
    print(f"[done] wrote {len(files)} overlay mosaics to {out_dir}")
    return files


def main():
    parser = argparse.ArgumentParser(description="NAVSIM -> SparseDrive infos")
    parser.add_argument("--data-root", default="/media/applied/navsim")
    parser.add_argument(
        "--split", default="mini",
        help="navsim_logs/<split> subdirectory (mini, test, trainval, or an "
             "exported split like alpasim_pilot)")
    parser.add_argument("--filter", default="navmini",
                        help="vendored scene filter name "
                             "(navmini/navtest/navtrain)")
    parser.add_argument("--output", default="data/infos/navsim_infos_navmini.pkl")
    parser.add_argument("--count-only", action="store_true",
                        help="only verify the official token count")
    parser.add_argument("--workers", type=int, default=0,
                        help="per-log multiprocessing workers (0 = serial)")
    parser.add_argument("--split-json", default=None,
                        help="frozen log-split json (e.g. "
                             "data/splits/navtrain_split_v1.json); restricts "
                             "the filter's log_names to the chosen key")
    parser.add_argument("--split-key", default=None,
                        choices=[None, "train_logs", "val_logs"],
                        help="which split list to use from --split-json")
    parser.add_argument("--expected-count", type=int, default=None,
                        help="assert the converted scenario count "
                             "(navmini/navtest have built-in expectations)")
    parser.add_argument("--maps-root", default="/media/applied/navsim/maps",
                        help="nuPlan map GPKG root (<location>/<ver>/map.gpkg)")
    parser.add_argument("--no-maps", action="store_true",
                        help="skip local vector-map extraction (Phase-0 mode)")
    parser.add_argument("--overlay-dir", default=None,
                        help="also render calibration overlay mosaics here")
    parser.add_argument("--overlay-num", type=int, default=32)
    parser.add_argument("--blob-root", default=None,
                        help="prefix cam data_paths with this directory "
                             "(absolute paths in the output infos; for "
                             "exported sims outside the training data_root)")
    args = parser.parse_args()

    if (args.split_json is None) != (args.split_key is None):
        parser.error("--split-json and --split-key must be given together")
    if args.blob_root:
        global BLOB_ROOT
        BLOB_ROOT = os.path.abspath(args.blob_root)
    data = convert(args.data_root, args.split, args.filter, args.output,
                   count_only=args.count_only,
                   maps_root=None if args.no_maps else args.maps_root,
                   workers=args.workers,
                   split_json=args.split_json, split_key=args.split_key,
                   expected_count=args.expected_count)
    if data is not None and args.overlay_dir:
        render_overlays(data, args.data_root, args.split,
                        args.overlay_dir, args.overlay_num)


if __name__ == "__main__":
    main()
