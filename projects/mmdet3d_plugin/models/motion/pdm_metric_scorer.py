"""On-the-fly PDM sub-score supervision for the trajectory vocabulary.

Port of SparseDriveV2's navsim/agents/sparsedrive/scorer/get_pdm_score_v2.py.
Candidate trajectories are simulated and scored against the per-sample NAVSIM
metric cache (metric_cache.pkl) in a CPU process pool, exactly as V2 trains
its metric heads. Everything NAVSIM-related is imported lazily inside the
workers so that models without metric supervision never require the navsim
or nuplan packages.

Environment:
    SPARSEDRIVE_NAVSIM_DEVKIT_ROOT  path added to sys.path that contains the
                                    ``navsim`` package (the SparseDriveV2
                                    checkout). Unset if navsim is already
                                    importable.
    SPARSEDRIVE_PDM_SCORING_CFG     scoring-parameters yaml. Defaults to the
                                    V2 training config resolved inside the
                                    devkit root.
    SPARSEDRIVE_PDM_WORKERS         process-pool size per training process
                                    (default 8).
"""

import os
import sys
import lzma
import pickle
import multiprocessing as mp
import concurrent.futures as cf

_DEFAULT_CFG_RELPATH = (
    "navsim/planning/script/config/pdm_scoring/run_pdm_train.yaml"
)

_pool = None


def _devkit_root():
    return os.environ.get("SPARSEDRIVE_NAVSIM_DEVKIT_ROOT")


def _scoring_cfg_path():
    cfg = os.environ.get("SPARSEDRIVE_PDM_SCORING_CFG")
    if cfg:
        return cfg
    root = _devkit_root()
    if root:
        return os.path.join(root, _DEFAULT_CFG_RELPATH)
    return _DEFAULT_CFG_RELPATH


def _init_worker():
    """Build simulator/scorer/traffic policy once per pool worker."""
    global SIMULATOR, SCORER, TRAFFIC_AGENT_POLICY, PDM_SCORE_FN
    root = _devkit_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from navsim.agents.sparsedrive.scorer.pdm_score_v2 import pdm_score

    pdm_cfg = OmegaConf.load(_scoring_cfg_path())
    SIMULATOR = instantiate(pdm_cfg.simulator)
    SCORER = instantiate(pdm_cfg.scorer)
    SCORER.train_mode = True
    TRAFFIC_AGENT_POLICY = instantiate(
        pdm_cfg.non_reactive, SIMULATOR.proposal_sampling
    )
    PDM_SCORE_FN = pdm_score


def _score_one(args):
    cache_path, traj_np = args
    with lzma.open(cache_path, "rb") as f:
        metric_cache = pickle.load(f)
    return PDM_SCORE_FN(
        metric_cache=metric_cache,
        model_trajectory=traj_np,  # (G, T, 3) NAVSIM ego frame
        future_sampling=SIMULATOR.proposal_sampling,
        simulator=SIMULATOR,
        scorer=SCORER,
        traffic_agents_policy=TRAFFIC_AGENT_POLICY,
    )


def _get_pool():
    global _pool
    if _pool is None:
        _pool = cf.ProcessPoolExecutor(
            max_workers=int(os.environ.get("SPARSEDRIVE_PDM_WORKERS", "8")),
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
        )
    return _pool


def get_pdm_sub_scores(trajectories, metric_cache_paths):
    """Score candidate trajectories against their samples' metric caches.

    Args:
        trajectories: (B, G, T, 3) numpy array of candidate poses in the
            NAVSIM ego frame (x forward, y left, heading), T poses at 0.5 s.
        metric_cache_paths: length-B list of metric_cache.pkl paths; None or
            missing paths yield a None entry instead of sub-scores.

    Returns:
        Length-B list; each entry is a dict of per-metric (G,) arrays
        (the pdm_score_v2 output) or None when the cache was unavailable.
    """
    pool = _get_pool()
    futures = {}
    for idx, path in enumerate(metric_cache_paths):
        if path and os.path.exists(path):
            futures[idx] = pool.submit(_score_one, (path, trajectories[idx]))
    results = [None] * len(metric_cache_paths)
    for idx, future in futures.items():
        results[idx] = future.result()
    return results
