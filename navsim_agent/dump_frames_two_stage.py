"""Dump SparseDrive pre-pipeline input frames for a NAVSIM v2 two-stage split.

Runs in SparseDriveV2's eval env (conda ``lilypad``). For every token of a
two-stage split (original stage-one + synthetic stage-two), loads the NAVSIM
``AgentInput`` (path-only cameras via ``NAVSIM_CACHE_SKIP_IMAGES=1`` — the
image pixels are read later by the SparseDrive pipeline) and converts it with
the committed adapter (``navsim_agent.agent_input_adapter
.frames_from_agent_input``: SE(2)->SE(3) history poses, C4 frame change,
10-slot ego status with the documented zero-filled slots) into the exact
per-frame input dicts ``runner.predict_scenario`` consumes.

Output pickle: ``{"frames": {token: [input_dict x num_history_frames]},
"split": ..., "tokens_stage_one": [...], "tokens_stage_two": [...]}``.
Consumed by ``navsim_agent/run_inference_frames.py`` (sparsedrive310, GPU).

Example:
    python navsim_agent/dump_frames_two_stage.py \
        --split navhard_two_stage \
        --metric-cache work_dirs/navsim_eval/metric_cache_navhard2s_full \
        --output work_dirs/navsim_eval/frames_navhard2s.pkl
"""
import argparse
import logging
import os
import pickle
import sys
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
# path-only camera loading (pixels are read by the SparseDrive pipeline)
os.environ["NAVSIM_CACHE_SKIP_IMAGES"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dump_frames_two_stage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="navhard_two_stage")
    parser.add_argument("--metric-cache", required=True,
                        help="restrict to tokens present in this cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--logs", nargs="*", default=None)
    args = parser.parse_args()
    out_path = os.path.abspath(args.output)
    cache_path = os.path.abspath(args.metric_cache)

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader

    from navsim_agent.agent_input_adapter import frames_from_agent_input

    config_dir = os.path.join(
        SPARSEDRIVEV2_ROOT, "navsim/planning/script/config/pdm_scoring"
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="default_run_pdm_score_fast",
            overrides=[
                f"train_test_split={args.split}",
                "experiment_name=dump_frames_two_stage",
            ],
        )
    if args.logs:
        cfg.train_test_split.scene_filter.log_names = list(args.logs)

    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        # all eight cameras on every history frame (path-only); never load
        # LiDAR — SparseDrive is camera-only and some test logs miss
        # current-frame pcds (see the navtest LiDAR audit note)
        sensor_config=SensorConfig(
            cam_f0=True, cam_l0=True, cam_l1=True, cam_l2=True,
            cam_r0=True, cam_r1=True, cam_r2=True, cam_b0=True,
            lidar_pc=False,
        ),
    )
    cache_tokens = set(MetricCacheLoader(Path(cache_path)).tokens)
    tokens = sorted(set(scene_loader.tokens) & cache_tokens)
    stage_one = set(scene_loader.tokens_stage_one)
    logger.info(
        "tokens: scene_loader=%d cache=%d dumping=%d (stage_one=%d)",
        len(scene_loader.tokens), len(cache_tokens), len(tokens),
        len(set(tokens) & stage_one),
    )
    assert tokens, "no tokens to dump"

    frames = {}
    n_missing_img = 0
    for i, token in enumerate(tokens):
        agent_input = scene_loader.get_agent_input_from_token(token)
        token_frames = frames_from_agent_input(agent_input)
        for fr in token_frames:
            fr.pop("_images_rgb", None)  # paths only
            for p in fr["img_filename"]:
                if not os.path.exists(p):
                    n_missing_img += 1
        frames[token] = token_frames
        if (i + 1) % 200 == 0:
            logger.info("dumped %d / %d", i + 1, len(tokens))
    assert n_missing_img == 0, f"{n_missing_img} missing image files"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            dict(
                frames=frames,
                split=args.split,
                tokens_stage_one=sorted(set(tokens) & stage_one),
                tokens_stage_two=sorted(set(tokens) - stage_one),
                metric_cache=cache_path,
                num_history_frames=int(
                    cfg.train_test_split.scene_filter.num_history_frames),
            ),
            f,
        )
    logger.info("wrote %d tokens -> %s", len(frames), out_path)


if __name__ == "__main__":
    main()
