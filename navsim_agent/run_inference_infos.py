"""Offline SparseDrive inference over converted NAVSIM infos (Phase 4a).

Runs in the SparseDrive env (sparsedrive310). For every scenario row in a
converted info PKL: resets all temporal banks, replays the four history
frames (from the converter's deduplicated frame table — full can_bus ego
status, SD-frame extrinsics), and writes the NAVSIM-frame trajectory:

    {token: (ego_fut_ts, 3) float32 [x, y, heading] NAVSIM local poses}

The pickle is then scored inside SparseDriveV2's environment with
``navsim_agent/score_pdm_v1.py`` (see docs/navsim_eval_openloop.md).

Usage (from the SparseDrive repo root):
    python -m navsim_agent.run_inference_infos \
        --config projects/configs/navsim/sparsedrive_navsim_stage2_geoinput.py \
        --checkpoint work_dirs/.../iter_xxx.pth \
        --infos data/infos/navsim_infos_navmini.pkl \
        --output work_dirs/navsim_eval/trajs_sparsedrive_navmini.pkl
"""
import argparse
import hashlib
import os
import pickle

import numpy as np
from tqdm import tqdm

from .agent_input_adapter import input_dict_from_frame
from .coord import sd_plan_to_navsim_poses
from .geometry_file import GeometryFileProducer
from .geometry_gate import GENERATED_GEOMETRY_SOURCES, PLANNER_GEOMETRY_SOURCES
from .runner import SparseDriveRunner


OCCURRENCE_GEOMETRY_SOURCE = "emperror_occurrence"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_eval_contract(infos, expected_scenarios, expected_replay_calls):
    if (
        not isinstance(expected_scenarios, int)
        or isinstance(expected_scenarios, bool)
        or expected_scenarios < 1
    ):
        raise ValueError("expected_scenarios must be a positive integer")
    if (
        not isinstance(expected_replay_calls, int)
        or isinstance(expected_replay_calls, bool)
        or expected_replay_calls < 1
    ):
        raise ValueError("expected_replay_calls must be a positive integer")
    history_lengths = [len(info.get("history_tokens", ())) for info in infos]
    if any(length != 4 for length in history_lengths):
        raise ValueError("every planner scenario must have exactly four history frames")
    if len(infos) != expected_scenarios:
        raise ValueError(
            f"planner scenario count mismatch: {len(infos)} != "
            f"{expected_scenarios}"
        )
    replay_calls = sum(history_lengths)
    if replay_calls != expected_replay_calls:
        raise ValueError(
            f"planner replay count mismatch: {replay_calls} != "
            f"{expected_replay_calls}"
        )
    return {"scenarios": len(infos), "replay_calls": replay_calls}


def _required_geometry_occurrences(infos):
    occurrences = {}
    for info in infos:
        scenario_token = info.get("token")
        history_tokens = tuple(info.get("history_tokens", ()))
        if (
            not isinstance(scenario_token, str)
            or not scenario_token
            or len(history_tokens) != 4
            or history_tokens[-1] != scenario_token
        ):
            raise ValueError(
                "external geometry requires four oldest-to-current history tokens"
            )
        if scenario_token in occurrences:
            raise ValueError("external geometry scenario tokens must be unique")
        occurrences[scenario_token] = history_tokens
    return occurrences


def frame_to_input_dict(frame_rec, data_root, cmd):
    """Converter frame-table record -> pre-pipeline SparseDrive input dict."""
    import pyquaternion

    ego2global = np.eye(4)
    ego2global[:3, :3] = pyquaternion.Quaternion(
        frame_rec["ego2global_rotation"]
    ).rotation_matrix
    ego2global[:3, 3] = np.asarray(frame_rec["ego2global_translation"])

    camera_entries = []
    for name, cam in frame_rec["cams"].items():
        camera_entries.append(
            dict(
                name=name,
                image_path=os.path.join(data_root, cam["data_path"])
                if data_root else cam["data_path"],
                image=None,
                # converter already stores SD-frame extrinsics
                sensor2lidar_rotation=np.asarray(
                    cam["sensor2lidar_rotation"], dtype=np.float64
                ),
                sensor2lidar_translation=np.asarray(
                    cam["sensor2lidar_translation"], dtype=np.float64
                ),
                cam_intrinsic=np.asarray(
                    cam["cam_intrinsic"], dtype=np.float64
                ),
                distortion=np.asarray(cam["distortion"], dtype=np.float64),
            )
        )
    input_dict = input_dict_from_frame(
        timestamp_s=frame_rec["timestamp"] / 1e6,
        lidar2global_sd=ego2global,
        camera_entries=camera_entries,
        ego_status_10=frame_rec["ego_status"],
        gt_ego_fut_cmd=cmd,
    )
    # The external producer consumes this before the image pipeline.
    input_dict["_frame_token"] = frame_rec["token"]
    return input_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256")
    parser.add_argument("--checkpoint", required=True,
                        help="'none' builds a random-init model "
                             "(plumbing check only)")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--infos", required=True)
    parser.add_argument("--infos-sha256")
    parser.add_argument("--expected-scenarios", type=int)
    parser.add_argument("--expected-replay-calls", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=os.environ.get(
        "NAVSIM_BLOBS_ROOT", "/media/applied/navsim/sensor_blobs/mini/"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--geometry-source",
        choices=("native", *PLANNER_GEOMETRY_SOURCES, OCCURRENCE_GEOMETRY_SOURCE),
        default="native",
        help=(
            "native features, re-encoded teacher geometry, token EMPERROR, "
            "or occurrence-indexed EMPERROR output"
        ),
    )
    parser.add_argument(
        "--geometry-file",
        help=(
            "schema-v1 token or schema-v2 occurrence geometry artifact; "
            "required for generated sources"
        ),
    )
    parser.add_argument(
        "--geometry-sha256",
        help="expected geometry artifact digest; required for generated sources",
    )
    parser.add_argument(
        "--geometry-teacher-config-sha256",
        help="expected training-teacher config digest for the EMPERROR artifact",
    )
    parser.add_argument(
        "--geometry-teacher-checkpoint-sha256",
        help="expected training-teacher checkpoint digest for the EMPERROR artifact",
    )
    parser.add_argument(
        "--geometry-producer-checkpoint-sha256",
        help="expected EMPERROR checkpoint digest that produced the artifact",
    )
    parser.add_argument("--geometry-uri", help="optional retrievable artifact URI")
    parser.add_argument(
        "--ego-query-only",
        action="store_true",
        help="run only the ego query while retaining agent/map geometry as K/V",
    )
    parser.add_argument(
        "--reset-perception-each-frame",
        action="store_true",
        help="reset detection/map temporal banks before every replay frame",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--shard", "--shard-idx", dest="shard_idx", type=int, default=0,
        help="strided shard index (with --num-shards)",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every", type=int, default=500,
        help="dump partial trajectories to <output>.partial every N "
             "scenarios; an existing partial is loaded on start and its "
             "tokens skipped (crash/kill resume)")
    args = parser.parse_args()

    contract_args = (
        args.infos_sha256,
        args.expected_scenarios,
        args.expected_replay_calls,
    )
    if args.geometry_source == OCCURRENCE_GEOMETRY_SOURCE and not all(
        contract_args
    ):
        parser.error(
            "emperror_occurrence requires --infos-sha256, "
            "--expected-scenarios, and --expected-replay-calls"
        )
    if any(value is not None for value in contract_args) and not all(
        value is not None for value in contract_args
    ):
        parser.error("infos/count contract arguments must be provided together")

    infos_sha256 = sha256_file(args.infos)
    if args.infos_sha256 and infos_sha256 != args.infos_sha256:
        raise ValueError("planner infos SHA256 mismatch")
    with open(args.infos, "rb") as f:
        data = pickle.load(f)
    infos, frame_table = data["infos"], data["frames"]
    required_tokens = {
        token for info in infos for token in info["history_tokens"]
    }
    geometry_producer = None
    teacher_identity = (
        args.geometry_teacher_config_sha256,
        args.geometry_teacher_checkpoint_sha256,
    )
    if args.geometry_source == OCCURRENCE_GEOMETRY_SOURCE:
        required_occurrences = _required_geometry_occurrences(infos)
        if not args.geometry_file or not args.geometry_sha256:
            parser.error(
                "--geometry-file and --geometry-sha256 are required "
                "for source=emperror_occurrence"
            )
        if any(teacher_identity) or args.geometry_producer_checkpoint_sha256:
            parser.error(
                "emperror_occurrence provenance is read from the immutable "
                "artifact, not command-line model SHA256 values"
            )
        geometry_producer = GeometryFileProducer(
            args.geometry_file,
            args.device,
            required_tokens=required_tokens,
            infos_sha256=infos_sha256,
            expected_sha256=args.geometry_sha256,
            required_occurrences=required_occurrences,
            require_occurrence_file=True,
        )
    elif args.geometry_source in GENERATED_GEOMETRY_SOURCES:
        required_occurrences = _required_geometry_occurrences(infos)
        if not args.geometry_file or not args.geometry_sha256:
            parser.error(
                "--geometry-file and --geometry-sha256 are required "
                "for generated geometry sources"
            )
        if not all(teacher_identity):
            parser.error(
                "both --geometry-teacher-*-sha256 values are required "
                "for generated geometry sources"
            )
        if not args.geometry_producer_checkpoint_sha256:
            parser.error(
                "--geometry-producer-checkpoint-sha256 is required "
                "for generated geometry sources"
            )
        geometry_producer = GeometryFileProducer(
            args.geometry_file,
            args.device,
            required_tokens=required_tokens,
            infos_sha256=infos_sha256,
            expected_sha256=args.geometry_sha256,
            expected_teacher_config_sha256=(
                args.geometry_teacher_config_sha256
            ),
            expected_teacher_checkpoint_sha256=(
                args.geometry_teacher_checkpoint_sha256
            ),
            expected_producer_checkpoint_sha256=(
                args.geometry_producer_checkpoint_sha256
            ),
            required_occurrences=required_occurrences,
        )
    elif (
        args.geometry_file
        or args.geometry_sha256
        or args.geometry_uri
        or any(teacher_identity)
        or args.geometry_producer_checkpoint_sha256
    ):
        parser.error("geometry artifact arguments require a generated geometry source")

    config_sha256 = sha256_file(args.config)
    if args.config_sha256 and config_sha256 != args.config_sha256:
        raise ValueError("planner config SHA256 mismatch")
    ckpt = None if args.checkpoint.lower() == "none" else args.checkpoint
    checkpoint_sha256 = None
    if args.checkpoint_sha256:
        if ckpt is None:
            parser.error("--checkpoint-sha256 requires a trained checkpoint")
        checkpoint_sha256 = sha256_file(ckpt)
        if checkpoint_sha256 != args.checkpoint_sha256:
            raise ValueError("planner checkpoint SHA256 mismatch")
    runner = SparseDriveRunner(
        args.config,
        ckpt,
        device=args.device,
        geometry_source=(
            None
            if args.geometry_source == "native"
            else "emperror"
            if args.geometry_source == OCCURRENCE_GEOMETRY_SOURCE
            else args.geometry_source
        ),
        geometry_producer=geometry_producer,
        ego_query_only=args.ego_query_only,
        reset_perception_each_frame=args.reset_perception_each_frame,
    )

    if args.limit:
        infos = infos[: args.limit]
    expected_contract = (
        _validate_eval_contract(
            infos, args.expected_scenarios, args.expected_replay_calls
        )
        if args.infos_sha256
        else None
    )
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    if args.num_shards > 1:
        infos = infos[args.shard_idx::args.num_shards]
        print(
            f"shard {args.shard_idx}/{args.num_shards}: "
            f"{len(infos)} scenarios"
        )

    partial_path = args.output + ".partial"
    trajectories = {}
    if args.checkpoint_every and os.path.exists(partial_path):
        with open(partial_path, "rb") as f:
            trajectories = pickle.load(f)
        print(f"resuming: {len(trajectories)} trajectories from "
              f"{partial_path}")

    def _dump_partial():
        tmp = partial_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(trajectories, f)
        os.replace(tmp, partial_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    since_dump = 0
    replay_calls = 0
    processed_scenarios = 0
    for info in tqdm(infos, desc="scenarios"):
        if info["token"] in trajectories:
            continue
        cmd = np.asarray(info["gt_ego_fut_cmd"], dtype=np.float32)
        frames = []
        for frame_index, token in enumerate(info["history_tokens"]):
            frame = frame_to_input_dict(frame_table[token], args.data_root, cmd)
            frame["_geometry_scenario_token"] = info["token"]
            frame["_geometry_frame_index"] = frame_index
            frames.append(frame)
        assert info["history_tokens"][-1] == info["token"]
        replay_calls += len(frames)
        plan_sd = runner.predict_scenario(frames)
        trajectories[info["token"]] = sd_plan_to_navsim_poses(plan_sd)
        processed_scenarios += 1
        since_dump += 1
        if args.checkpoint_every and since_dump >= args.checkpoint_every:
            _dump_partial()
            since_dump = 0
    if args.checkpoint_every:
        _dump_partial()

    if replay_calls != 4 * processed_scenarios:
        raise RuntimeError("local planner replay/trajectory count mismatch")
    if expected_contract is not None and args.num_shards == 1 and (
        processed_scenarios != expected_contract["scenarios"]
        or replay_calls != expected_contract["replay_calls"]
    ):
        raise RuntimeError("planner output count differs from pinned contract")

    gate_stats = runner.geometry_gate_stats()
    if args.geometry_source != "native" and gate_stats["calls"] != replay_calls:
        raise RuntimeError("planner geometry gate count mismatch")
    if (
        geometry_producer is not None
        and geometry_producer.calls != replay_calls
    ):
        raise RuntimeError("EMPERROR geometry producer count mismatch")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(
            dict(
                trajectories=trajectories,
                config=os.path.abspath(args.config),
                config_sha256=config_sha256,
                checkpoint=os.path.abspath(args.checkpoint)
                if ckpt else "RANDOM_INIT_PLUMBING_ONLY",
                checkpoint_sha256=checkpoint_sha256,
                infos=os.path.abspath(args.infos),
                infos_sha256=infos_sha256,
                expected_scenarios=args.expected_scenarios,
                expected_replay_calls=args.expected_replay_calls,
                interval_s=0.5,
                frame="navsim_local_se2",
                geometry_source=args.geometry_source,
                ego_query_only=args.ego_query_only,
                perception_temporal_reset=(
                    "every_frame"
                    if args.reset_perception_each_frame
                    else "sequence_boundary"
                ),
                geometry_gate=gate_stats,
                geometry_artifact=(
                    dict(
                        geometry_producer.metadata(),
                        **({"uri": args.geometry_uri} if args.geometry_uri else {}),
                    )
                    if geometry_producer is not None
                    else None
                ),
                replay_stats={
                    "scenarios": processed_scenarios,
                    "replay_calls": replay_calls,
                },
            ),
            f,
        )
    print(f"wrote {len(trajectories)} trajectories -> {args.output}")


if __name__ == "__main__":
    main()
