"""NAVSIM ``AgentInput`` -> SparseDrive pipeline input dicts.

Duck-typed against ``navsim.common.dataclasses.AgentInput`` (no navsim
import), so it is usable in the SparseDrive env. Produces, per history
frame, the same pre-pipeline input dict as
``NavSim3DDataset.get_data_info`` builds from the converter's infos:

- fixed CAM_F0-first eight-camera order;
- camera extrinsics rotated into the SparseDrive BEV frame
  (``T_camera_to_sd = C4 @ T_camera_to_nav``, lidar2ego identity);
- synthetic SparseDrive ego2global from the relative SE(2) history poses
  (``G_sd = G_nav @ inv(C4)``; only relative transforms are consumed by
  the temporal banks);
- 10-value ego status ([acc xyz, ang-rate xyz, vel xyz, steering], ego
  axes, never rotated). ``AgentInput`` only exposes [vx, vy, ax, ay], so
  acc z / angular rates / vel z / steering are zero-filled — a known
  eval-time degradation vs the converter's full can_bus (documented in
  the runbooks);
- NAVSIM ``[left, straight, right, unknown]`` command -> SparseDrive
  ``[right, left, straight]`` with deterministic straight for unknown.
"""
import numpy as np

from .coord import C3, C4

# Fixed camera order: CAM_F0 must be index 0 (converter contract).
CAMERA_ORDER = (
    "cam_f0", "cam_l0", "cam_l1", "cam_l2",
    "cam_r0", "cam_r1", "cam_r2", "cam_b0",
)

FRAME_INTERVAL_S = 0.5


def convert_command(driving_command):
    """NAVSIM [left, straight, right, unknown] -> SD [right, left, straight].

    Identical mapping to tools/data_converter/navsim_converter.py
    (unknown -> deterministic straight; loss masking is a train-time
    concern only).
    """
    dc = np.asarray(driving_command).reshape(-1)
    assert dc.shape == (4,), f"unexpected driving_command shape {dc.shape}"
    idx = int(np.argmax(dc))
    if idx == 0:      # left
        cmd = [0, 1, 0]
    elif idx == 1:    # straight
        cmd = [0, 0, 1]
    elif idx == 2:    # right
        cmd = [1, 0, 0]
    else:             # unknown -> deterministic straight
        cmd = [0, 0, 1]
    return np.array(cmd, dtype=np.float32)


def ego_status_from_agent_input(ego_status):
    """[ax, ay, 0, 0, 0, 0, vx, vy, 0, 0] from a navsim ``EgoStatus``.

    Slot layout matches the converter ([acc(0:3), ang-rate(3:6), vel(6:9),
    steering(9)]); index 6 = forward velocity is the InstanceQueue contract.
    """
    vx, vy = np.asarray(ego_status.ego_velocity, dtype=np.float64)[:2]
    ax, ay = np.asarray(ego_status.ego_acceleration, dtype=np.float64)[:2]
    return np.array(
        [ax, ay, 0.0, 0.0, 0.0, 0.0, vx, vy, 0.0, 0.0], dtype=np.float32
    )


def se2_to_matrix(pose_xyyaw):
    """SE(2) [x, y, yaw] -> SE(3) homogeneous matrix (z/pitch/roll zero)."""
    x, y, yaw = [float(v) for v in np.asarray(pose_xyyaw, dtype=np.float64)]
    m = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    m[0, 3], m[1, 3] = x, y
    return m


def _camera_entries(cameras):
    """Ordered per-camera dicts in the SD frame from a navsim ``Cameras``."""
    entries = []
    for name in CAMERA_ORDER:
        cam = getattr(cameras, name)
        assert cam.image_path is not None or cam.image is not None, (
            f"camera {name} missing from AgentInput (the SparseDrive agent "
            f"needs all eight cameras on every history frame)"
        )
        rot_sd = C3 @ np.asarray(cam.sensor2lidar_rotation, dtype=np.float64)
        trans_sd = C3 @ np.asarray(
            cam.sensor2lidar_translation, dtype=np.float64
        )
        entries.append(
            dict(
                name=name,
                image_path=str(cam.image_path)
                if cam.image_path is not None else None,
                image=cam.image,
                sensor2lidar_rotation=rot_sd,
                sensor2lidar_translation=trans_sd,
                cam_intrinsic=np.asarray(cam.intrinsics, dtype=np.float64),
                distortion=np.asarray(cam.distortion, dtype=np.float64),
            )
        )
    return entries


def input_dict_from_frame(
    timestamp_s,
    lidar2global_sd,
    camera_entries,
    ego_status_10,
    gt_ego_fut_cmd,
):
    """One pre-pipeline SparseDrive input dict (mirrors
    ``NavSim3DDataset.get_data_info``)."""
    image_paths, images = [], []
    lidar2img_rts, lidar2cam_rts = [], []
    cam_intrinsic, cam_distortion = [], []
    for cam in camera_entries:
        image_paths.append(cam["image_path"])
        images.append(cam["image"])
        lidar2cam_r = np.linalg.inv(cam["sensor2lidar_rotation"])
        lidar2cam_t = cam["sensor2lidar_translation"] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam["cam_intrinsic"].copy()
        cam_intrinsic.append(intrinsic)
        viewpad = np.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        lidar2img_rts.append(viewpad @ lidar2cam_rt.T)
        lidar2cam_rts.append(lidar2cam_rt)
        cam_distortion.append(cam["distortion"].copy())

    input_dict = dict(
        timestamp=float(timestamp_s),
        lidar2global=np.asarray(lidar2global_sd, dtype=np.float64),
        img_filename=image_paths,
        lidar2img=lidar2img_rts,
        lidar2cam=lidar2cam_rts,
        cam_intrinsic=cam_intrinsic,
        cam_distortion=cam_distortion,
        ego_status=np.asarray(ego_status_10, dtype=np.float32),
        gt_ego_fut_cmd=np.asarray(gt_ego_fut_cmd, dtype=np.float32),
    )
    if any(im is not None for im in images):
        input_dict["_images_rgb"] = images  # optional in-memory RGB frames
    return input_dict


def frames_from_agent_input(agent_input):
    """NAVSIM ``AgentInput`` -> chronological (oldest..current) list of
    SparseDrive pre-pipeline input dicts.

    Uses the relative SE(2) history poses to synthesize per-frame
    ``lidar2global`` (``G_sd = G_nav_rel @ inv(C4)``, current frame at the
    origin); the temporal banks only ever consume relative products
    ``T_global_inv[cur] @ T_global[prev]``, so any shared global offset is
    irrelevant. Timestamps are synthesized on the nominal 0.5 s grid.
    """
    num_frames = len(agent_input.ego_statuses)
    assert len(agent_input.cameras) == num_frames
    # The command of the current frame drives the planner's branch select
    # on every replayed frame (history outputs are discarded).
    cmd = convert_command(agent_input.ego_statuses[-1].driving_command)

    frames = []
    for i in range(num_frames):
        status = agent_input.ego_statuses[i]
        g_nav = se2_to_matrix(status.ego_pose)  # frame i in current frame
        lidar2global_sd = g_nav @ np.linalg.inv(C4)
        frames.append(
            input_dict_from_frame(
                timestamp_s=i * FRAME_INTERVAL_S,
                lidar2global_sd=lidar2global_sd,
                camera_entries=_camera_entries(agent_input.cameras[i]),
                ego_status_10=ego_status_from_agent_input(status),
                gt_ego_fut_cmd=cmd,
            )
        )
    return frames
