"""NAVSIM v2 two-stage EPDMS scoring (closed-loop protocol, Phase 4b).

Runs in SparseDriveV2's eval env (conda ``lilypad``). Reuses SparseDriveV2's
``run_pdm_score_navhard_fast`` scoring/aggregation functions verbatim, but
produces agent trajectories via ``compute_trajectory``-style evaluation
(the shipped harness only supports feature-cache agents), so pinned trivial
agents and precomputed SparseDrive trajectory pickles can be scored without
modifying the SparseDriveV2 checkout.

Differences from the shipped harness main():
- trajectories come from a pinned trivial agent (``constant_velocity``,
  ``human``) or a ``traj:<pickle>`` produced by the SparseDrive wrapper;
- ``worker=sequential`` by default (deterministic; navmini/subset scale);
- a missing ``reactive_all_mapping`` (e.g. train_test_split
  ``navmini_two_stage``) degrades gracefully to per-token stage-one PDMS
  (the shipped harness crashes on an unbound variable) — EPDMS aggregation
  then reports null and the run is labeled a stage-one smoke;
- only the bug-fixed scorer (``pdm_score_fix_bug``, the version the
  harness stores as ``navhard_bug_fix.csv``) is used.

Example:
    python /home/tejan/lwm-rl/SparseDrive/navsim_agent/score_epdms_two_stage.py \
        --agent constant_velocity --split navmini_two_stage \
        --metric-cache /home/tejan/lwm-rl/SparseDriveV2/exp/metric_cache_navminiv2_two_stage \
        --run-tag cv_navmini_run1
"""
import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

SPARSEDRIVEV2_ROOT = os.environ.get(
    "SPARSEDRIVEV2_ROOT", "/home/tejan/lwm-rl/SparseDriveV2"
)
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/media/applied/navsim")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/media/applied/navsim/maps")
os.environ.setdefault(
    "NAVSIM_EXP_ROOT",
    "/home/tejan/lwm-rl/SparseDrive/work_dirs/navsim_eval",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("score_epdms_two_stage")


def _validate_required_epdms(
    epdms, pseudo_closed_loop_valid, num_tokens, num_valid, expected_tokens
):
    import numpy as np

    if not pseudo_closed_loop_valid or any(
        not np.isfinite(epdms.get(name, np.nan))
        for name in ("combined", "stage_one", "stage_two")
    ):
        raise RuntimeError("required two-stage ePDMS aggregation is invalid")
    if expected_tokens is not None and (
        num_tokens != expected_tokens or num_valid != expected_tokens
    ):
        raise RuntimeError(
            "required two-stage ePDMS token/valid count differs from contract"
        )


def compose_cfg(split, worker="sequential", extra_overrides=()):
    from hydra import compose, initialize_config_dir

    config_dir = os.path.join(
        SPARSEDRIVEV2_ROOT, "navsim/planning/script/config/pdm_scoring"
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="default_run_pdm_score_fast",
            overrides=[
                f"train_test_split={split}",
                f"worker={worker}",
                "experiment_name=score_epdms_two_stage",
            ] + list(extra_overrides),
        )
    return cfg


def build_trajectories(agent_spec, scene_loader, tokens):
    """{token: navsim Trajectory} for every token to score."""
    import numpy as np
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
    from navsim.common.dataclasses import Trajectory

    sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    if agent_spec.startswith("traj:"):
        with open(agent_spec.split(":", 1)[1], "rb") as f:
            payload = pickle.load(f)
        trajs = {}
        for token in tokens:
            poses = np.asarray(payload["trajectories"][token],
                               dtype=np.float32)
            assert poses.shape == (8, 3), poses.shape
            trajs[token] = Trajectory(poses, sampling)
        return trajs

    if agent_spec == "constant_velocity":
        from navsim.agents.constant_velocity_agent import (
            ConstantVelocityAgent,
        )

        agent = ConstantVelocityAgent(trajectory_sampling=sampling)
        agent.initialize()
        trajs = {}
        for i, token in enumerate(tokens):
            agent_input = scene_loader.get_agent_input_from_token(token)
            trajs[token] = agent.compute_trajectory(agent_input)
            if (i + 1) % 50 == 0:
                logger.info("trajectories %d / %d", i + 1, len(tokens))
        return trajs

    if agent_spec == "human":
        from navsim.agents.human_agent import HumanAgent

        agent = HumanAgent(trajectory_sampling=sampling)
        agent.initialize()
        trajs = {}
        for token in tokens:
            scene = scene_loader.get_scene_from_token(token)
            trajs[token] = agent.compute_trajectory(
                scene.get_agent_input(), scene
            )
        return trajs

    raise ValueError(f"unknown agent spec {agent_spec}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", required=True,
        help="constant_velocity | human | traj:<pickle>")
    parser.add_argument("--split", default="navmini_two_stage")
    parser.add_argument("--metric-cache", required=True)
    parser.add_argument(
        "--logs", nargs="*", default=None,
        help="optional log_names subset (e.g. a navhard smoke subset)")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.environ["NAVSIM_EXP_ROOT"], "epdms_v2"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--require-epdms", action="store_true")
    parser.add_argument("--expected-tokens", type=int)
    parser.add_argument(
        "--worker", default="sequential",
        choices=[
            "sequential",
            "single_machine_thread_pool",
            "ray_distributed_no_torch",
        ],
        help="sequential (deterministic loop; navmini/subset scale) or "
             "single_machine_thread_pool (parallel scoring without nested "
             "Ray) or ray_distributed_no_torch (full navhard scale; the "
             "final CSV is token-sorted, so results are worker-order "
             "independent)")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from dataclasses import fields
    from hydra.utils import instantiate
    from nuplan.planning.utils.multithreading.worker_utils import worker_map

    from navsim.common.dataclasses import PDMResults, SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.common.enums import SceneFrameType
    from navsim.evaluate.pdm_score_fix_bug import (
        pdm_score as pdm_score_fix_bug,
    )
    from navsim.planning.script.builders.worker_pool_builder import (
        build_worker,
    )
    from navsim.planning.script.run_pdm_score_navhard_fast import (
        calculate_individual_mapping_scores,
        compute_final_scores,
        create_scene_aggregators,
        run_pdm_score,
    )

    from omegaconf import open_dict

    cfg = compose_cfg(args.split, worker=args.worker)
    cfg.metric_cache_path = os.path.abspath(args.metric_cache)
    # splits without a synthetic stage (e.g. navmini_two_stage) ship no
    # reactive_synthetic_initial_tokens; the harness worker iterates
    # reactive_tokens_stage_two, which must be [] rather than None
    with open_dict(cfg):
        if cfg.train_test_split.scene_filter.get(
            "reactive_synthetic_initial_tokens"
        ) is None:
            cfg.train_test_split.scene_filter.reactive_synthetic_initial_tokens = []
        # Metric caches serialize the producer's absolute map path.  Cluster
        # jobs stage the same pinned maps elsewhere, so force the reactive IDM
        # policy to use the runtime path instead of that stale cached path.
        map_root_override = os.environ.get("NUPLAN_MAPS_ROOT")
        if map_root_override:
            if not Path(map_root_override).is_dir():
                raise RuntimeError(
                    f"NUPLAN_MAPS_ROOT does not exist: {map_root_override}"
                )
            cfg.traffic_agents_policy.reactive.map_root_override = (
                map_root_override
            )
            logger.info("reactive map root override: %s", map_root_override)
    # cache metadata CSVs may record paths relative to the SparseDriveV2
    # root (cwd when the pinned caches were generated); the scoring workers
    # re-open them via those recorded paths
    os.chdir(SPARSEDRIVEV2_ROOT)
    if args.logs:
        cfg.train_test_split.scene_filter.log_names = list(args.logs)

    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = sorted(
        set(scene_loader.tokens) & set(metric_cache_loader.tokens)
    )
    logger.info(
        "tokens: scene_loader=%d cache=%d evaluating=%d "
        "(stage_one=%d synthetic_loaded=%d)",
        len(scene_loader.tokens), len(metric_cache_loader.tokens),
        len(tokens_to_evaluate), len(scene_loader.tokens_stage_one),
        len(scene_loader.synthetic_scenes),
    )
    assert tokens_to_evaluate, "no tokens to evaluate"

    trajectories = build_trajectories(
        args.agent, scene_loader, tokens_to_evaluate
    )

    worker = build_worker(cfg)
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": trajectories,
        }
        for log_file, tokens_list in
        sorted(scene_loader.get_tokens_list_per_log().items())
    ]
    score_rows = worker_map(
        worker, partial(run_pdm_score, pdm_score_fn=pdm_score_fix_bug),
        data_points,
    )
    pdm_score_df = pd.concat(score_rows)

    # ---- EPDMS aggregation (verbatim harness logic, guarded) ----------- #
    all_mappings = {}
    pseudo_closed_loop_valid = False
    try:
        raw_mapping = cfg.train_test_split.reactive_all_mapping
        for orig_token, prev_token, two_stage_pairs in raw_mapping:
            if (prev_token in set(scene_loader.tokens)
                    or orig_token in set(scene_loader.tokens)):
                all_mappings[(orig_token, prev_token)] = [
                    tuple(pair) for pair in two_stage_pairs
                ]
        pdm_score_df = create_scene_aggregators(
            all_mappings, pdm_score_df,
            instantiate(cfg.simulator.proposal_sampling),
        )
        pdm_score_df = compute_final_scores(pdm_score_df)
        pseudo_closed_loop_valid = True
    except Exception as error:
        if args.require_epdms:
            raise RuntimeError(
                f"required ePDMS aggregation failed for split {args.split}"
            ) from error
        logger.warning(
            "no usable reactive_all_mapping for split %s — reporting "
            "per-token stage-one PDMS only (stage-one smoke, not EPDMS)",
            args.split,
        )
        pdm_score_df["weight"] = 1.0
        if "ego_simulated_states" in pdm_score_df.columns:
            pdm_score_df = pdm_score_df.drop(
                columns=["ego_simulated_states"])
        pdm_score_df["two_frame_extended_comfort"] = np.nan

    score_cols = [
        c for c in pdm_score_df.columns
        if ((any(score.name in c for score in fields(PDMResults))
             or c in ("two_frame_extended_comfort", "score"))
            and c != "pdm_score")
    ]

    epdms = dict(combined=None, stage_one=None, stage_two=None)
    if pseudo_closed_loop_valid:
        pcl_group, pcl_s1, pcl_s2 = calculate_individual_mapping_scores(
            pdm_score_df[score_cols + ["token", "weight"]], all_mappings
        )
        epdms = dict(
            combined=float(pcl_group.get("score", np.nan)),
            stage_one=float(pcl_s1.get("score", np.nan)),
            stage_two=float(pcl_s2.get("score", np.nan)),
        )

    # per-stage columns for the CSV (mirrors the harness)
    for col in score_cols:
        s1 = pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL
        s2 = pdm_score_df["frame_type"] == SceneFrameType.SYNTHETIC
        pdm_score_df.loc[s1, f"{col}_stage_one"] = pdm_score_df.loc[s1, col]
        pdm_score_df.loc[s2, f"{col}_stage_two"] = pdm_score_df.loc[s2, col]

    # drop the set-iteration-order residue so two runs emit identical CSVs
    if "index" in pdm_score_df.columns:
        pdm_score_df = pdm_score_df.drop(columns=["index"])
    pdm_score_df = pdm_score_df.sort_values("token").reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.run_tag}.csv"
    keep_cols = [
        c for c in pdm_score_df.columns if c != "ego_simulated_states"
    ]
    pdm_score_df[keep_cols].to_csv(csv_path, index=False)

    harness_commit = subprocess.run(
        ["git", "-C", SPARSEDRIVEV2_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    num_valid = int(pdm_score_df["valid"].sum())
    if args.require_epdms:
        _validate_required_epdms(
            epdms,
            pseudo_closed_loop_valid,
            len(tokens_to_evaluate),
            num_valid,
            args.expected_tokens,
        )
    meta = dict(
        agent=args.agent,
        split=args.split,
        logs_subset=args.logs,
        metric_cache=os.path.abspath(args.metric_cache),
        num_tokens=len(tokens_to_evaluate),
        num_scored_rows=len(pdm_score_df),
        num_valid_rows=num_valid,
        epdms=epdms,
        pseudo_closed_loop_valid=pseudo_closed_loop_valid,
        stage_one_mean_pdms=float(
            pdm_score_df.loc[
                pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL,
                "score" if "score" in pdm_score_df.columns else "pdm_score",
            ].mean()
        ),
        scorer="pdm_score_fix_bug",
        worker=args.worker,
        harness=SPARSEDRIVEV2_ROOT,
        harness_commit=harness_commit,
        python=sys.executable,
        timestamp=datetime.now().isoformat(),
    )
    with open(out_dir / f"{args.run_tag}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"per-token CSV: {csv_path}")


if __name__ == "__main__":
    main()
