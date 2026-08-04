"""NAVSIM v1 PDMS scoring (open-loop) inside SparseDriveV2's environment.

Runs in SparseDriveV2's eval env (conda ``lilypad``: ``navsim`` is installed
editable from /home/tejan/lwm-rl/SparseDriveV2, whose ``navsim.navsim_v1``
classes match the pinned v1 metric-cache pickles — do NOT score those caches
from any other code tree).

Agents:
- ``human``               pinned privileged agent (reproducibility gate);
- ``constant_velocity``   pinned trivial agent (reproducibility gate);
- ``traj:<pickle>``       trajectories from navsim_agent/run_inference_infos.py
                          ({token: (8,3) NAVSIM local poses}).

Scoring is a deterministic sequential loop over sorted tokens; two runs of
the same agent must produce identical per-token scores (gate: max abs diff
< 1e-6, checked by docs/navsim_eval_openloop.md's runbook commands).

Example (from anywhere; paths are absolute):
    python /home/tejan/lwm-rl/SparseDrive/navsim_agent/score_pdm_v1.py \
        --agent human --run-tag human_run1
"""
import argparse
import json
import logging
import lzma
import os
import pickle
import subprocess
import sys
from datetime import datetime
from dataclasses import asdict
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
logger = logging.getLogger("score_pdm_v1")


def _validate_complete_scores(num_tokens, num_valid, expected_tokens):
    if expected_tokens is None:
        raise ValueError("complete scoring requires --expected-tokens")
    if num_tokens != expected_tokens or num_valid != expected_tokens:
        raise RuntimeError("PDMS token/valid count differs from contract")


def compose_cfg(split):
    from hydra import compose, initialize_config_dir

    config_dir = os.path.join(
        SPARSEDRIVEV2_ROOT, "navsim/planning/script/config/pdm_scoring"
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="default_run_pdm_score_fast_v1",
            overrides=[
                f"train_test_split={split}",
                "experiment_name=score_pdm_v1",
            ],
        )
    return cfg


def absolutize_cache_paths(metric_cache_loader):
    """Cache metadata CSVs record paths relative to the SparseDriveV2 root
    (the cwd when the pinned caches were generated); resolve them so this
    script can run from anywhere."""
    for token, p in list(metric_cache_loader.metric_cache_paths.items()):
        if not os.path.isabs(str(p)):
            metric_cache_loader.metric_cache_paths[token] = os.path.join(
                SPARSEDRIVEV2_ROOT, str(p)
            )
    return metric_cache_loader


def build_agent(agent_spec):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    if agent_spec == "human":
        from navsim.agents.human_agent import HumanAgent

        return HumanAgent(trajectory_sampling=sampling), True, None
    if agent_spec == "constant_velocity":
        from navsim.agents.constant_velocity_agent import (
            ConstantVelocityAgent,
        )

        return ConstantVelocityAgent(trajectory_sampling=sampling), False, None
    if agent_spec.startswith("traj:"):
        path = agent_spec.split(":", 1)[1]
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return None, False, payload
    raise ValueError(f"unknown agent spec {agent_spec}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", required=True,
        help="human | constant_velocity | traj:<pickle>")
    parser.add_argument("--split", default="navmini")
    parser.add_argument(
        "--metric-cache",
        default=os.path.join(SPARSEDRIVEV2_ROOT, "exp/metric_cache_navminiv1"))
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.environ["NAVSIM_EXP_ROOT"], "pdm_v1"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--expected-tokens", type=int)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from hydra.utils import instantiate

    from navsim.common.dataclasses import Trajectory
    from navsim.navsim_v1.common.dataclasses import SensorConfig
    from navsim.navsim_v1.common.dataloader import (
        MetricCacheLoader,
        SceneLoader,
    )
    from navsim.navsim_v1.evaluate.pdm_score import pdm_score
    from navsim.navsim_v1.planning.metric_caching.metric_cache import (
        MetricCache,
    )

    cfg = compose_cfg(args.split)
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling

    agent, requires_scene, traj_payload = build_agent(args.agent)
    if agent is not None:
        agent.initialize()
        sensor_config = agent.get_sensor_config()
    else:
        sensor_config = SensorConfig.build_no_sensors()

    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=sensor_config,
    )
    metric_cache_loader = absolutize_cache_paths(
        MetricCacheLoader(Path(args.metric_cache))
    )

    tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    logger.info("scoring %d tokens (scene_loader=%d, cache=%d)",
                len(tokens), len(scene_loader.tokens),
                len(metric_cache_loader.tokens))

    if traj_payload is not None:
        trajs = traj_payload["trajectories"]
        missing = [t for t in tokens if t not in trajs]
        assert not missing, f"trajectory pickle missing {len(missing)} tokens"

    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    rows = []
    for i, token in enumerate(tokens):
        row = {"token": token, "valid": True}
        try:
            with lzma.open(
                metric_cache_loader.metric_cache_paths[token], "rb"
            ) as f:
                metric_cache: MetricCache = pickle.load(f)

            if traj_payload is not None:
                poses = np.asarray(
                    trajs[token], dtype=np.float32
                )
                assert poses.shape == (8, 3), poses.shape
                trajectory = Trajectory(poses, sampling)
            elif requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(
                    scene_loader.get_agent_input_from_token(token), scene
                )
            else:
                trajectory = agent.compute_trajectory(
                    scene_loader.get_agent_input_from_token(token)
                )

            result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            row.update(asdict(result))
        except Exception:
            logging.exception("agent failed for token %s", token)
            row["valid"] = False
        rows.append(row)
        if (i + 1) % 50 == 0:
            logger.info("scored %d / %d", i + 1, len(tokens))

    df = pd.DataFrame(rows)
    assert "score" in df.columns, "every token failed — see log for errors"
    num_valid = int(df["valid"].sum())
    if args.require_complete:
        _validate_complete_scores(len(tokens), num_valid, args.expected_tokens)
    avg = df.drop(columns=["token", "valid"]).mean(skipna=True)
    avg["token"] = "average"
    avg["valid"] = df["valid"].all()
    df.loc[len(df)] = avg

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.run_tag}.csv"
    df.to_csv(csv_path, index=False)

    harness_commit = subprocess.run(
        ["git", "-C", SPARSEDRIVEV2_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    meta = dict(
        agent=args.agent,
        split=args.split,
        metric_cache=os.path.abspath(args.metric_cache),
        num_tokens=len(tokens),
        num_valid=num_valid,
        harness=SPARSEDRIVEV2_ROOT,
        harness_commit=harness_commit,
        python=sys.executable,
        timestamp=datetime.now().isoformat(),
        pdms=float(df[df.token == "average"]["score"].iloc[0]),
    )
    if args.require_complete and not np.isfinite(meta["pdms"]):
        raise RuntimeError("required full PDMS is not finite")
    with open(out_dir / f"{args.run_tag}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"per-token CSV: {csv_path}")


if __name__ == "__main__":
    main()
