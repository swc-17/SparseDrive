"""Generate a NAVSIM v2 two-stage metric cache for a navhard log subset.

Runs in SparseDriveV2's eval env (conda ``lilypad``). Wraps SparseDriveV2's
own ``cache_data`` (navsim.planning.metric_caching.caching) with the
``navhard_two_stage`` split restricted to a small set of logs, writing a
fresh cache directory — the pinned full-split caches under
SparseDriveV2/exp are never touched or mixed with.

Needed because no navhard (synthetic-token) metric cache exists locally;
the closed-loop reproducibility gate scores a genuine two-stage subset.

Example:
    python /home/tejan/lwm-rl/SparseDrive/navsim_agent/cache_metrics_two_stage_subset.py \
        --logs 2021.05.25.14.16.10_veh-35_01100_01664 \
        --output /home/tejan/lwm-rl/SparseDrive/work_dirs/navsim_eval/metric_cache_navhard_subset
"""
import argparse
import logging
import os

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="navhard_two_stage")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir

    from navsim.planning.metric_caching.caching import cache_data
    from navsim.planning.script.builders.worker_pool_builder import (
        build_worker,
    )

    config_dir = os.path.join(
        SPARSEDRIVEV2_ROOT, "navsim/planning/script/config/metric_caching"
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="default_metric_caching",
            overrides=[
                f"train_test_split={args.split}",
                "worker=sequential",
                f"metric_cache_path={os.path.abspath(args.output)}",
            ],
        )
    cfg.train_test_split.scene_filter.log_names = list(args.logs)

    worker = build_worker(cfg)
    cache_data(cfg=cfg, worker=worker)


if __name__ == "__main__":
    main()
