"""NAVSIM v2 EPDMS scoring on navtest (single-stage pseudo-closed-loop).

This is the protocol behind SparseDriveV2's published "90.3" (navtest-v2
EPDMS; reproduced 0.8702 pre-bugfix / 0.9037 bug-fix on this harness). It is
NOT the navhard two-stage EPDMS and NOT the navtest v1 PDMS — keep the three
tables separate.

Runs in SparseDriveV2's eval env (conda ``lilypad``). Mirrors the shipped
``run_pdm_score_navtest_v2_fast.py`` exactly — v2 scorer per original frame,
two-frame extended comfort via the start-time-inferred adjacent mapping
(``infer_start_adjacent_mapping`` + ``SceneAggregator(one_stage_only=True)``),
``compute_final_scores``, final score = mean over valid tokens — but consumes
trajectories from a ``traj:<pickle>`` (or pinned trivial agent) instead of the
feature-cache Lightning agent, like navsim_agent/score_epdms_two_stage.py.

Example:
    python navsim_agent/score_epdms_navtest_v2.py \
        --agent traj:work_dirs/navsim_eval/trajs_v6_full16g_iter8703_navtest.pkl \
        --metric-cache /home/tejan/lwm-rl/SparseDriveV2/exp/metric_cache_navtestv2 \
        --worker ray_distributed_no_torch \
        --run-tag a_full16g_iter8703_navtestv2
"""
import argparse
import json
import logging
import os
import pickle
import subprocess
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
logger = logging.getLogger("score_epdms_navtest_v2")


def compose_cfg(split, worker):
    from hydra import compose, initialize_config_dir

    config_dir = os.path.join(
        SPARSEDRIVEV2_ROOT, "navsim/planning/script/config/pdm_scoring"
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name="default_run_pdm_score_fast",
            overrides=[
                f"train_test_split={split}",
                f"worker={worker}",
                "experiment_name=score_epdms_navtest_v2",
            ],
        )


def build_trajectories(agent_spec, scene_loader, tokens):
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
        return {
            token: agent.compute_trajectory(
                scene_loader.get_agent_input_from_token(token))
            for token in tokens
        }

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
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--metric-cache", required=True)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.environ["NAVSIM_EXP_ROOT"],
                             "epdms_navtest_v2"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--scorer", default="fix_bug", choices=["fix_bug", "bug"],
        help="fix_bug = pdm_score_fix_bug (the variant behind V2's quoted "
             "0.9037); bug = the shipped pre-fix scorer (V2: 0.8702)")
    parser.add_argument(
        "--worker", default="ray_distributed_no_torch",
        choices=["sequential", "ray_distributed_no_torch"])
    args = parser.parse_args()

    import pandas as pd
    from dataclasses import fields
    from hydra.utils import instantiate
    from nuplan.planning.utils.multithreading.worker_utils import worker_map

    from navsim.common.dataclasses import PDMResults, SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.evaluate.pdm_score import pdm_score as pdm_score_bug
    from navsim.evaluate.pdm_score_fix_bug import (
        pdm_score as pdm_score_fix_bug,
    )
    from navsim.planning.script.builders.worker_pool_builder import (
        build_worker,
    )
    from navsim.planning.script.run_pdm_score_navtest_v2_fast import (
        compute_final_scores,
        create_scene_aggregators,
        infer_start_adjacent_mapping,
        run_pdm_score,
    )

    cfg = compose_cfg(args.split, worker=args.worker)
    cfg.metric_cache_path = os.path.abspath(args.metric_cache)
    # cache metadata CSVs record paths relative to the SparseDriveV2 root
    # (cwd when the pinned caches were generated)
    os.chdir(SPARSEDRIVEV2_ROOT)

    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=None,
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tokens_to_evaluate = sorted(
        set(scene_loader.tokens) & set(metric_cache_loader.tokens)
    )
    logger.info(
        "tokens: scene_loader=%d cache=%d evaluating=%d",
        len(scene_loader.tokens), len(metric_cache_loader.tokens),
        len(tokens_to_evaluate),
    )
    assert tokens_to_evaluate, "no tokens to evaluate"

    trajectories = build_trajectories(
        args.agent, scene_loader, tokens_to_evaluate
    )

    pdm_score_fn = (
        pdm_score_fix_bug if args.scorer == "fix_bug" else pdm_score_bug
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
        worker, partial(run_pdm_score, pdm_score_fn=pdm_score_fn),
        data_points,
    )
    pdm_score_df = pd.concat(score_rows)

    # ---- verbatim harness aggregation (navtest v2, one stage) ---------- #
    start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
    pdm_score_df = create_scene_aggregators(
        start_adjacent_mapping, pdm_score_df,
        instantiate(cfg.simulator.proposal_sampling),
    )
    pdm_score_df = compute_final_scores(pdm_score_df)

    score_cols = [
        c for c in pdm_score_df.columns
        if ((any(score.name in c for score in fields(PDMResults))
             or c in ("two_frame_extended_comfort", "score"))
            and c != "pdm_score")
    ]
    valid = pdm_score_df["valid"]
    epdms = float(pdm_score_df.loc[valid, "score"].mean())

    keep = ["token", "valid", "log_name", "start_time"] + score_cols
    out_df = (pdm_score_df[[c for c in keep if c in pdm_score_df.columns]]
              .sort_values("token").reset_index(drop=True))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / f"{args.run_tag}.csv", index=False)

    harness_commit = subprocess.run(
        ["git", "-C", SPARSEDRIVEV2_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    meta = dict(
        agent=args.agent,
        split=args.split,
        protocol="navtest_v2_epdms_one_stage",
        scorer=args.scorer,
        metric_cache=cfg.metric_cache_path,
        num_tokens=len(tokens_to_evaluate),
        num_valid=int(valid.sum()),
        num_two_frame_pairs=len(start_adjacent_mapping),
        harness=SPARSEDRIVEV2_ROOT,
        harness_commit=harness_commit,
        timestamp=datetime.now().isoformat(),
        epdms=epdms,
    )
    with open(out_dir / f"{args.run_tag}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"per-token CSV: {out_dir / f'{args.run_tag}.csv'}")


if __name__ == "__main__":
    main()
