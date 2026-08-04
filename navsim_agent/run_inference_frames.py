"""SparseDrive inference over dumped two-stage frames (Phase 4b).

Runs in the SparseDrive env (sparsedrive310, GPU). Consumes the frames
pickle written by ``navsim_agent/dump_frames_two_stage.py`` (per-token
chronological pre-pipeline input dicts, path-only images) and, for every
token: resets all temporal banks, replays the history frames, and writes
the NAVSIM-frame trajectory — identical output contract to
``run_inference_infos.py``:

    {token: (ego_fut_ts, 3) float32 [x, y, heading] NAVSIM local poses}

Scored with ``navsim_agent/score_epdms_two_stage.py --agent traj:<pickle>``.

Usage (from the SparseDrive repo root):
    python -m navsim_agent.run_inference_frames \
        --config projects/configs/navsim/sparsedrive_navsim_stage2_geoinput_full.py \
        --checkpoint work_dirs/.../iter_xxx.pth \
        --frames work_dirs/navsim_eval/frames_navhard2s.pkl \
        --output work_dirs/navsim_eval/trajs_navhard2s.pkl
"""
import argparse
import hashlib
import os
import pickle

from tqdm import tqdm

from .coord import sd_plan_to_navsim_poses
from .geometry_file import GeometryFileProducer
from .geometry_gate import GENERATED_GEOMETRY_SOURCES, PLANNER_GEOMETRY_SOURCES
from .runner import SparseDriveRunner


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_geometry_tokens(frames_by_token):
    """Collect the real frame tokens embedded by the privileged-data dump."""
    required_tokens = set()
    for scenario_token, frames in frames_by_token.items():
        for history_index, frame in enumerate(frames):
            frame_token = frame.get("_frame_token")
            if not isinstance(frame_token, str) or not frame_token:
                raise ValueError(
                    "external geometry requires a real _frame_token for "
                    f"{scenario_token!r} history frame {history_index}"
                )
            required_tokens.add(frame_token)
    return required_tokens


def _required_geometry_occurrences(frames_by_token):
    occurrences = {}
    for scenario_token, frames in frames_by_token.items():
        if not isinstance(scenario_token, str) or not scenario_token:
            raise ValueError("external geometry requires a scenario token")
        if len(frames) != 4:
            raise ValueError("external geometry requires four history frames")
        history_tokens = tuple(
            frame.get("_frame_token") for frame in frames
        )
        if any(not isinstance(token, str) or not token for token in history_tokens):
            raise ValueError("external geometry requires real history frame tokens")
        if history_tokens[-1] != scenario_token:
            raise ValueError(
                "external geometry history must be ordered oldest-to-current"
            )
        occurrences[scenario_token] = history_tokens
    return occurrences


def _validate_eval_contract(frames_by_token, expected_scenarios, expected_replay_calls):
    if len(frames_by_token) != expected_scenarios:
        raise ValueError(
            f"planner scenario count mismatch: {len(frames_by_token)} != "
            f"{expected_scenarios}"
        )
    history_lengths = [len(frames) for frames in frames_by_token.values()]
    if any(length != 4 for length in history_lengths):
        raise ValueError("every planner scenario must have exactly four history frames")
    replay_calls = sum(history_lengths)
    if replay_calls != expected_replay_calls:
        raise ValueError(
            f"planner replay count mismatch: {replay_calls} != "
            f"{expected_replay_calls}"
        )
    return {"scenarios": len(frames_by_token), "replay_calls": replay_calls}


def _emperror_index_sha256(payload):
    value = payload.get("emperror_index_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("external geometry frames require emperror_index_sha256")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--frames", required=True)
    parser.add_argument("--frames-sha256")
    parser.add_argument("--expected-scenarios", type=int)
    parser.add_argument("--expected-replay-calls", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--geometry-source",
        choices=("native", *PLANNER_GEOMETRY_SOURCES),
        default="native",
    )
    parser.add_argument("--geometry-file")
    parser.add_argument("--geometry-sha256")
    parser.add_argument("--geometry-uri")
    parser.add_argument("--geometry-teacher-config-sha256")
    parser.add_argument("--geometry-teacher-checkpoint-sha256")
    parser.add_argument("--geometry-producer-checkpoint-sha256")
    parser.add_argument("--ego-query-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0,
                        help="strided shard index (with --num-shards)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every", type=int, default=500,
        help="dump partial trajectories to <output>.partial every N "
             "scenarios; an existing partial is loaded on start and its "
             "tokens skipped (crash/kill resume)")
    parser.add_argument(
        "--path-remap", default=None,
        help="'<old_prefix>:<new_prefix>' rewrite of dumped image paths "
             "(e.g. '/media/applied/navsim:/tmp/navsim_data' when the "
             "frames pickle was dumped on another machine)")
    args = parser.parse_args()

    contract_args = (
        args.frames_sha256,
        args.expected_scenarios,
        args.expected_replay_calls,
    )
    if any(value is not None for value in contract_args) and not all(
        value is not None for value in contract_args
    ):
        parser.error("frames/count contract arguments must be provided together")
    frames_sha256 = sha256_file(args.frames)
    if args.frames_sha256 and frames_sha256 != args.frames_sha256:
        raise ValueError("planner frames SHA256 mismatch")
    with open(args.frames, "rb") as f:
        payload = pickle.load(f)
    frames_by_token = payload["frames"]
    if args.frames_sha256:
        _validate_eval_contract(
            frames_by_token,
            args.expected_scenarios,
            args.expected_replay_calls,
        )

    teacher_identity = (
        args.geometry_teacher_config_sha256,
        args.geometry_teacher_checkpoint_sha256,
    )
    geometry_producer = None
    if args.geometry_source in GENERATED_GEOMETRY_SOURCES:
        if not all(
            (
                args.geometry_file,
                args.geometry_sha256,
                args.geometry_producer_checkpoint_sha256,
                *teacher_identity,
            )
        ):
            parser.error(
                "generated geometry sources require the artifact file/SHA256, "
                "producer checkpoint SHA256, and both teacher SHA256 values"
            )
        if not args.ego_query_only:
            parser.error("generated geometry sources require --ego-query-only")
        if not args.frames_sha256:
            parser.error(
                "generated geometry sources require a pinned frames/count contract"
            )
        emperror_index_sha256 = _emperror_index_sha256(payload)
        required_tokens = _required_geometry_tokens(frames_by_token)
        required_occurrences = _required_geometry_occurrences(frames_by_token)
        geometry_producer = GeometryFileProducer(
            args.geometry_file,
            args.device,
            required_tokens=required_tokens,
            infos_sha256=emperror_index_sha256,
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
    elif any(
        (
            args.geometry_file,
            args.geometry_sha256,
            args.geometry_uri,
            args.geometry_producer_checkpoint_sha256,
            *teacher_identity,
        )
    ):
        parser.error("geometry artifact arguments require a generated geometry source")

    config_sha256 = sha256_file(args.config)
    if args.config_sha256 and config_sha256 != args.config_sha256:
        raise ValueError("planner config SHA256 mismatch")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if (
        args.checkpoint_sha256
        and checkpoint_sha256 != args.checkpoint_sha256
    ):
        raise ValueError("planner checkpoint SHA256 mismatch")

    runner = SparseDriveRunner(
        args.config,
        args.checkpoint,
        device=args.device,
        geometry_source=(
            None if args.geometry_source == "native" else args.geometry_source
        ),
        geometry_producer=geometry_producer,
        ego_query_only=args.ego_query_only,
    )

    if args.path_remap:
        old, new = args.path_remap.split(":", 1)
        for token_frames in frames_by_token.values():
            for fr in token_frames:
                fr["img_filename"] = [
                    new + p[len(old):] if p.startswith(old) else p
                    for p in fr["img_filename"]
                ]
    tokens = sorted(frames_by_token)
    if args.limit:
        tokens = tokens[: args.limit]
    if args.num_shards > 1:
        tokens = tokens[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(tokens)} scenarios")

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
    for token in tqdm(tokens, desc="scenarios"):
        if token in trajectories:
            continue
        frames = frames_by_token[token]
        for frame_index, frame in enumerate(frames):
            frame["_geometry_scenario_token"] = token
            frame["_geometry_frame_index"] = frame_index
        replay_calls += len(frames)
        plan_sd = runner.predict_scenario(frames)
        trajectories[token] = sd_plan_to_navsim_poses(plan_sd)
        processed_scenarios += 1
        since_dump += 1
        if args.checkpoint_every and since_dump >= args.checkpoint_every:
            _dump_partial()
            since_dump = 0
    if args.checkpoint_every:
        _dump_partial()

    gate_stats = runner.geometry_gate_stats()
    if args.geometry_source != "native" and gate_stats["calls"] != replay_calls:
        raise RuntimeError("planner geometry gate count mismatch")
    if geometry_producer is not None and geometry_producer.calls != replay_calls:
        raise RuntimeError("EMPERROR geometry producer count mismatch")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(
            dict(
                trajectories=trajectories,
                config=os.path.abspath(args.config),
                config_sha256=config_sha256,
                checkpoint=os.path.abspath(args.checkpoint),
                checkpoint_sha256=checkpoint_sha256,
                frames=os.path.abspath(args.frames),
                frames_sha256=frames_sha256,
                emperror_index_sha256=payload.get("emperror_index_sha256"),
                expected_scenarios=args.expected_scenarios,
                expected_replay_calls=args.expected_replay_calls,
                split=payload.get("split"),
                interval_s=0.5,
                frame="navsim_local_se2",
                geometry_source=args.geometry_source,
                ego_query_only=args.ego_query_only,
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
