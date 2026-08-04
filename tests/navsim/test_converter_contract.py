"""Phase 0 contract tests for the NAVSIM converter output.

Pure coordinate-contract tests always run; PKL-based tests are skipped when
``data/infos/navsim_infos_navmini.pkl`` has not been generated.

Run:  pytest -q tests/navsim
"""

import os
import pickle
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools", "data_converter"))

import navsim_converter as nc  # noqa: E402

PKL = os.path.join(REPO, "data", "infos", "navsim_infos_navmini.pkl")


def test_coordinate_roundtrip():
    nc.coordinate_roundtrip_selftest()


def test_command_mapping():
    cmd, valid = nc.convert_command([1, 0, 0, 0])  # left
    assert cmd.tolist() == [0, 1, 0] and valid == 1
    cmd, valid = nc.convert_command([0, 1, 0, 0])  # straight
    assert cmd.tolist() == [0, 0, 1] and valid == 1
    cmd, valid = nc.convert_command([0, 0, 1, 0])  # right
    assert cmd.tolist() == [1, 0, 0] and valid == 1
    cmd, valid = nc.convert_command([0, 0, 0, 1])  # unknown -> straight, masked
    assert cmd.tolist() == [0, 0, 1] and valid == 0


def test_relative_se2_identity():
    origin = [10.0, -5.0, 0.7]
    rel = nc.relative_se2(origin, np.array([origin]))
    assert np.abs(rel).max() < 1e-12


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(PKL):
        pytest.skip("navmini info pkl not generated")
    with open(PKL, "rb") as f:
        return pickle.load(f)


def test_token_count(data):
    assert len(data["infos"]) == 396
    assert data["metadata"]["num_samples"] == 396


def test_audited_temporal_counts(data):
    infos = data["infos"]
    assert sum(1 for x in infos if x["window_crosses_scene"]) == 130
    assert sum(1 for x in infos if x["window_has_gap"]) == 6


def test_camera_order_and_calibration(data):
    for info in data["infos"]:
        cams = list(info["cams"].keys())
        assert cams == list(nc.CAMERA_ORDER)
        assert cams[0] == "CAM_F0"
        for c in info["cams"].values():
            assert c["cam_intrinsic"].shape == (3, 3)
            assert c["distortion"].shape == (5,)
            assert c["sensor2lidar_rotation"].shape == (3, 3)
            # proper rotation
            R = c["sensor2lidar_rotation"]
            assert np.abs(R @ R.T - np.eye(3)).max() < 1e-6
            assert abs(np.linalg.det(R) - 1.0) < 1e-6
    # front camera looks along +y (forward) in the SparseDrive frame
    f0 = data["infos"][0]["cams"]["CAM_F0"]
    cam_forward_sd = f0["sensor2lidar_rotation"][:, 2]  # camera z-axis
    assert cam_forward_sd[1] > 0.99


def test_shapes_dtypes_finite(data):
    for info in data["infos"]:
        n = len(info["gt_boxes"])
        assert info["gt_boxes"].shape == (n, 7)
        assert info["gt_velocity"].shape == (n, 2)
        assert info["valid_flag"].shape == (n,) and info["valid_flag"].all()
        assert info["instance_inds"].shape == (n,)
        assert info["instance_inds"].dtype == np.int64
        assert info["gt_agent_fut_trajs"].shape == (n, 12, 2)
        assert info["gt_agent_fut_masks"].shape == (n, 12)
        # NAVSIM provides at most 10 future frames
        assert info["gt_agent_fut_masks"][:, 10:].max(initial=0) == 0
        assert info["gt_ego_fut_trajs"].shape == (8, 2)
        assert info["gt_ego_fut_masks"].shape == (8,)
        assert info["gt_ego_fut_cmd"].shape == (3,)
        assert info["gt_ego_fut_cmd"].sum() == 1
        assert info["ego_status"].shape == (10,)
        assert info["ego_status"][9] == 0.0  # steering unavailable
        assert len(info["fut_boxes"]) == 8
        for k in ("gt_boxes", "gt_velocity", "gt_agent_fut_trajs",
                  "gt_ego_fut_trajs", "ego_status"):
            assert np.isfinite(info[k]).all(), k
        assert np.linalg.norm(info["gt_boxes"][:, :2], axis=1).max(
            initial=0) <= nc.BOX_RANGE_M + 1e-9
        assert set(info["gt_names"]) <= set(nc.NAVSIM_CLASSES)


def test_agent_future_masks_monotone(data):
    """Masks stop at the first absence: no 0 -> 1 transitions."""
    for info in data["infos"]:
        m = info["gt_agent_fut_masks"]
        if len(m) == 0:
            continue
        assert (np.diff(m, axis=1) <= 0 + 1e-9).all() or (
            (m[:, 1:] <= m[:, :-1] + 1e-9).all()
        )


def test_instance_ids_deterministic_and_persistent(data):
    table = data["track_tokens"]
    assert table == sorted(table)
    lut = {t: i for i, t in enumerate(table)}
    # ids are table positions -> identical token, identical id everywhere,
    # including across linked raw scene_token boundaries.
    n_checked = 0
    for info in data["infos"]:
        assert info["instance_inds"].max(initial=-1) < len(table)
        n_checked += 1
    assert n_checked == 396
    # spot-check: recompute one row's ids from the raw log
    info = data["infos"][0]
    log = pickle.load(
        open(f"/media/applied/navsim/navsim_logs/mini/{info['log_name']}.pkl",
             "rb")
    )
    fr = next(f for f in log if f["token"] == info["token"])
    _, _, _, tracks = nc.convert_boxes(fr)
    assert info["instance_inds"].tolist() == [lut[t] for t in tracks]


def test_sequence_grouping(data):
    """Sequence ids never cross logs; window frames share the id unless the
    window is flagged as containing a gap."""
    frames = data["frames"]
    log_of_seq = {}
    for fr in frames.values():
        sid = fr["sequence_id"]
        assert log_of_seq.setdefault(sid, fr["log_name"]) == fr["log_name"]
    for info in data["infos"]:
        ids = {frames[t]["sequence_id"]
               for t in info["history_tokens"] + info["future_tokens"]}
        if not info["window_has_gap"]:
            assert ids == {info["sequence_id"]}
        else:
            assert len(ids) > 1


def test_ego_future_step_magnitude(data):
    """8 steps at 0.5 s: per-step offsets must be physically plausible."""
    for info in data["infos"]:
        step = np.linalg.norm(info["gt_ego_fut_trajs"], axis=1)
        assert step.max() < 25.0 * 0.5 + 1.0  # < ~13.5 m per 0.5 s
