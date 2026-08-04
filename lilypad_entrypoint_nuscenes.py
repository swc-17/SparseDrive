"""Lilypad entrypoint for the combined NAVSIM+nuScenes SparseDrive plan.

Covers both job families of the "Combined NAVSIM + nuScenes training" plan
section:

- the nuScenes-only REPRODUCTION GATE (unified contract): stages the
  nuScenes keyframe images exactly like the declutter jobs
  (users/tejan/sparsedrive/declutter/data/sd_nuscenes.tar -> data/nuscenes)
  plus the unified infos / anchors from the NAVSIM prefix;
- the COMBINED training runs: same nuScenes staging PLUS the navtrain
  sharded frame tars (frames_prefix), like lilypad_entrypoint_navsim.

Everything else (packed conda env, CUDA-op build, resume/ckpt-sync,
torchrun via tools/train_pyfocal.py) mirrors lilypad_entrypoint_navsim.py.

S3 layout:
  users/tejan/sparsedrive/declutter/...       env, resnet ckpt, nuScenes tar
  users/tejan/navsim/sparsedrive/data/...     unified infos, combined kmeans
  users/tejan/navsim/sparsedrive/work_dirs/<run_name>/   outputs
"""

import logging
import os
import sys
import time
from typing import Any

# reuse the audited helpers from the declutter entrypoint (same repo root)
from lilypad_entrypoint import (  # noqa: E402
    DECLUTTER_PREFIX,
    ENV_LOCAL,
    USER_BUCKET,
    DATA_TAR_LOCAL,
    _download_file_s3,
    _find_cuda_home,
    _put_file_nonchunked,
    _run_cmd,
    _s3_client,
    _upload_dir,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

NAVSIM_PREFIX = "users/tejan/navsim/sparsedrive"

# nuScenes-unified gate defaults (combined jobs override via stage_files)
DEFAULT_STAGE_FILES = [
    "data/infos/nuscenes_infos_train_unified.pkl",
    "data/infos/nuscenes_infos_val_unified.pkl",
    "data/kmeans/nusc_unified/kmeans_motion_6_nusc_unified.npy",
    "data/kmeans/nusc_unified/kmeans_plan_6_nusc_unified.npy",
]
# the gate reuses the baseline nuScenes det/map anchors (class-agnostic
# k-means centers) from the declutter prefix
DECLUTTER_STAGE_FILES = [
    "data/kmeans/kmeans_det_900.npy",
    "data/kmeans/kmeans_map_100.npy",
]


def train_entrypoint_fn(config: dict[Any, Any]) -> None:
    """Train one gate/combined run on an 8-GPU node.

    config keys:
        config_file    : repo-relative mmcv config, e.g.
                         "projects/configs/combined/sparsedrive_nusc_unified_stage1.py"
        seed           : int random seed
        run_name       : wandb run name + S3 work_dir leaf
        load_from_s3   : S3 URI of an init checkpoint (null for stage 1)
        num_gpus       : default 8
        stage_files    : NAVSIM_PREFIX-relative data files (str or
                         [s3_rel, repo_rel]); default DEFAULT_STAGE_FILES
        declutter_files: DECLUTTER_PREFIX-relative extras; default
                         DECLUTTER_STAGE_FILES (baseline det/map anchors)
        stage_nuscenes : stage the nuScenes keyframe tar (default True)
        frames_prefix  : optional NAVSIM_PREFIX-relative prefix of sharded
                         per-log navtrain image tars (combined jobs)
        blobs_root     : where NAVSIM images are untarred; exported as
                         NAVSIM_BLOBS_ROOT (default "/tmp/navsim_blobs/trainval")
    """
    import ray

    num_gpus = config.get("num_gpus", 8)
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, runtime_env={"env_vars": cred_env})
    def _train(cfg, n):
        _run_combined_training(cfg, n)

    ray.get(_train.remote(config, num_gpus))


def _run_combined_training(config, num_gpus):
    import glob
    import threading

    repo_root = os.path.abspath(os.path.dirname(__file__))
    config_file = config["config_file"]
    seed = int(config.get("seed", 0))
    run_name = config["run_name"]
    load_from_s3 = config.get("load_from_s3")
    work_dir = f"/tmp/work_dirs/{run_name}"
    work_s3_prefix = f"{NAVSIM_PREFIX}/work_dirs/{run_name}"
    blobs_root = config.get("blobs_root", "/tmp/navsim_blobs/trainval")

    s3 = _s3_client()

    # ── GPU keepalive so Lilypad's idle-GPU detector doesn't kill setup ────
    stop_keepalive = threading.Event()
    stop_sync = threading.Event()

    def _gpu_keepalive():
        try:
            import torch as _t
            import time as _time
            if not _t.cuda.is_available():
                return
            x = _t.randn(4096, 4096, device="cuda")
            while not stop_keepalive.is_set():
                end = _time.monotonic() + 0.5
                while _time.monotonic() < end and not stop_keepalive.is_set():
                    with _t.no_grad():
                        x = _t.mm(x, x.fmod(100.0))
                _t.cuda.synchronize()
                _time.sleep(2.0)
            del x
            _t.cuda.empty_cache()
        except Exception as e:
            logger.warning("keepalive error: %s", e)

    threading.Thread(target=_gpu_keepalive, daemon=True).start()

    try:
        # ── 1. conda env (shared with the declutter/navsim runs) ───────────
        if not os.path.exists(os.path.join(ENV_LOCAL, "bin", "python")):
            logger.info("[env] downloading packed conda env ...")
            _download_file_s3(
                s3, USER_BUCKET,
                f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz",
                "/tmp/sparsedrive310_env.tar.gz",
            )
            os.makedirs(ENV_LOCAL, exist_ok=True)
            _run_cmd(["tar", "-xzf", "/tmp/sparsedrive310_env.tar.gz",
                      "-C", ENV_LOCAL], tag="env-untar")
            _run_cmd([os.path.join(ENV_LOCAL, "bin", "conda-unpack")],
                     tag="conda-unpack")
        env_python = os.path.join(ENV_LOCAL, "bin", "python")
        _run_cmd([env_python, "-c",
                  "import torch, mmdet, mmcv, mmseg, flash_attn, cv2, "
                  "shapely, pyquaternion; "
                  "print('env sanity OK', torch.__version__)"],
                 tag="env-sanity")
        logger.info("[env] ready: %s", env_python)

        # ── 2a. nuScenes keyframe images (declutter mechanism) ─────────────
        if config.get("stage_nuscenes", True):
            nusc_dir = os.path.join(repo_root, "data", "nuscenes")
            if not os.path.isdir(os.path.join(nusc_dir, "samples")):
                logger.info("[data] downloading nuScenes keyframes tar ...")
                _download_file_s3(
                    s3, USER_BUCKET,
                    f"{DECLUTTER_PREFIX}/data/sd_nuscenes.tar",
                    DATA_TAR_LOCAL,
                )
                os.makedirs(nusc_dir, exist_ok=True)
                _run_cmd(["tar", "-xf", DATA_TAR_LOCAL, "-C", nusc_dir],
                         tag="data-untar")
                os.remove(DATA_TAR_LOCAL)

        # ── 2b. infos + anchors + resnet ckpt ──────────────────────────────
        for entry in config.get("stage_files", DEFAULT_STAGE_FILES):
            if isinstance(entry, str):
                s3_rel, repo_rel = entry, entry
            else:
                s3_rel, repo_rel = entry
            _download_file_s3(s3, USER_BUCKET, f"{NAVSIM_PREFIX}/{s3_rel}",
                              os.path.join(repo_root, repo_rel))
        for entry in config.get("declutter_files", DECLUTTER_STAGE_FILES):
            if isinstance(entry, str):
                s3_rel, repo_rel = entry, entry
            else:
                s3_rel, repo_rel = entry
            _download_file_s3(s3, USER_BUCKET, f"{DECLUTTER_PREFIX}/{s3_rel}",
                              os.path.join(repo_root, repo_rel))
        _download_file_s3(
            s3, USER_BUCKET, f"{DECLUTTER_PREFIX}/ckpt/resnet50-19c8e357.pth",
            os.path.join(repo_root, "ckpt/resnet50-19c8e357.pth"))

        # ── 2c. optional NAVSIM navtrain frame tars (combined jobs) ────────
        frames_prefix = config.get("frames_prefix")
        if frames_prefix:
            staged_marker = os.path.join(blobs_root, ".staged_complete")
            if not os.path.exists(staged_marker):
                os.makedirs(blobs_root, exist_ok=True)
                from concurrent.futures import ThreadPoolExecutor
                prefix = f"{NAVSIM_PREFIX}/{frames_prefix}".rstrip("/") + "/"
                keys = []
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=USER_BUCKET,
                                               Prefix=prefix):
                    keys += [o["Key"] for o in page.get("Contents", [])
                             if o["Key"].endswith(".tar")]
                if not keys:
                    raise RuntimeError(
                        f"no frame tars under s3://{USER_BUCKET}/{prefix}")
                logger.info("[blobs] staging %d frame tars from %s",
                            len(keys), prefix)
                t0 = time.monotonic()
                n_done = [0]

                def _stage_one(key):
                    local_tar = os.path.join(
                        "/tmp", "frame_tars", os.path.basename(key))
                    os.makedirs(os.path.dirname(local_tar), exist_ok=True)
                    _s3_client().download_file(USER_BUCKET, key, local_tar)
                    _run_cmd(["tar", "-xf", local_tar, "-C", blobs_root],
                             tag=f"untar-{os.path.basename(key)}")
                    os.remove(local_tar)
                    n_done[0] += 1
                    if n_done[0] % 100 == 0:
                        logger.info("[blobs] %d/%d tars staged (%.0f s)",
                                    n_done[0], len(keys),
                                    time.monotonic() - t0)

                with ThreadPoolExecutor(max_workers=16) as pool:
                    list(pool.map(_stage_one, keys))
                logger.info("[blobs] staged %d tars in %.0f s", len(keys),
                            time.monotonic() - t0)
                with open(staged_marker, "w") as f:
                    f.write("ok\n")

        # ── 3. compile plugin CUDA ops for this node's arch ────────────────
        ops_dir = os.path.join(repo_root, "projects", "mmdet3d_plugin", "ops")
        for so in glob.glob(os.path.join(ops_dir, "*.so")):
            os.remove(so)
        cuda_home = _find_cuda_home()
        build_env = {
            **os.environ,
            "CUDA_HOME": cuda_home,
            "PATH": f"{os.path.join(cuda_home, 'bin')}:{os.environ.get('PATH', '')}",
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": config.get("torch_cuda_arch_list", "8.0"),
        }
        _run_cmd([env_python, "setup.py", "build_ext", "--inplace"],
                 tag="ops-build", cwd=ops_dir, env=build_env)
        if not glob.glob(os.path.join(ops_dir, "*.so")):
            raise RuntimeError("[ops-build] no .so produced — check nvcc/arch")
        _run_cmd([env_python, "-c",
                  "from projects.mmdet3d_plugin.ops import feature_maps_format; "
                  "print('ops import OK')"],
                 tag="ops-sanity", cwd=repo_root,
                 env={**os.environ, "PYTHONPATH": repo_root})
        # per-view valid-mask unit test must pass on the training node
        _run_cmd([env_python,
                  os.path.join(repo_root, "tools", "tests",
                               "test_cam_valid_mask.py")],
                 tag="mask-test", cwd=repo_root,
                 env={**os.environ, "PYTHONPATH": repo_root})

        # ── 4. init / resume checkpoints ────────────────────────────────────
        os.makedirs(work_dir, exist_ok=True)
        cfg_options = []
        if load_from_s3:
            local_init = "/tmp/init_ckpt.pth"
            bucket, key = load_from_s3.replace("s3://", "").split("/", 1)
            _download_file_s3(s3, bucket, key, local_init)
            cfg_options += [f"load_from={local_init}"]

        resume_args = []
        try:
            resp = s3.list_objects_v2(Bucket=USER_BUCKET,
                                      Prefix=f"{work_s3_prefix}/latest_iter_")
            cands = sorted(
                (o["Key"] for o in resp.get("Contents", [])),
                key=lambda k: int(k.rsplit("_", 1)[-1].split(".")[0]),
            )
            if cands:
                local_resume = os.path.join(work_dir, "resume.pth")
                _download_file_s3(s3, USER_BUCKET, cands[-1], local_resume)
                resume_args = ["--resume-from", local_resume]
                logger.info("[resume] resuming from %s", cands[-1])
        except Exception as e:
            logger.warning("[resume] check failed (fresh start): %s", e)

        # ── 5. periodic checkpoint upload ───────────────────────────────────
        def _upload_latest_forever():
            last_uploaded = None
            while not stop_sync.is_set():
                time.sleep(600)
                try:
                    latest = os.path.join(work_dir, "latest.pth")
                    if os.path.islink(latest):
                        target = os.path.realpath(latest)
                        if target != last_uploaded and os.path.exists(target):
                            iter_tag = os.path.basename(target).replace(
                                ".pth", "")
                            key = f"{work_s3_prefix}/latest_{iter_tag}.pth"
                            logger.info("[ckpt-sync] uploading %s", key)
                            s3_bg = _s3_client()
                            _put_file_nonchunked(s3_bg, target, USER_BUCKET,
                                                 key)
                            last_uploaded = target
                except Exception as e:
                    logger.warning("[ckpt-sync] %s", e)

        threading.Thread(target=_upload_latest_forever, daemon=True).start()

        # ── 6. train ────────────────────────────────────────────────────────
        train_env = {
            **os.environ,
            "PATH": f"{os.path.join(ENV_LOCAL, 'bin')}:{os.environ.get('PATH', '')}",
            "PYTHONPATH": repo_root,
            "WANDB_NAME": run_name,
            "PYTHONUNBUFFERED": "1",
            "NAVSIM_BLOBS_ROOT": blobs_root,
        }
        stop_keepalive.set()  # release GPUs for torchrun
        time.sleep(5)
        _run_cmd(
            [env_python, "-m", "torch.distributed.run",
             f"--nproc_per_node={num_gpus}", "--master_port=28651",
             os.path.join(repo_root, "tools", "train_pyfocal.py"),
             os.path.join(repo_root, config_file),
             "--launcher", "pytorch",
             "--seed", str(seed),
             "--work-dir", work_dir]
            + (["--cfg-options"] + cfg_options if cfg_options else [])
            + resume_args
            + ["--no-validate"],
            tag="train", cwd=repo_root, env=train_env,
        )

        # ── 7. final upload: full work_dir (ckpts, logs, dumped config) ────
        logger.info("[upload] final work_dir → s3://%s/%s", USER_BUCKET,
                    work_s3_prefix)
        _upload_dir(s3, work_dir, USER_BUCKET, work_s3_prefix)
        logger.info("[done] %s", run_name)
    finally:
        stop_keepalive.set()
        stop_sync.set()


# ══════════════════════════════════════════════════════════════════════════
# Eval worker: N independent tools/test.py evals on one node, one GPU each.
# Used for the unified-gate vs declutter-V6 nuScenes-val planning comparison.
# ══════════════════════════════════════════════════════════════════════════

def eval_entrypoint_fn(config: dict[Any, Any]) -> None:
    """Run a batch of checkpoint evals in parallel (one GPU each).

    config keys:
        evals          : list of {"name", "config_file", "ckpt_s3",
                         "cfg_options": [...extra --cfg-options...]}
        results_prefix : NAVSIM_PREFIX-relative S3 prefix for logs/metrics
                         (default "eval_logs/plan_compare")
        wandb_project  : W&B project for one summary run per eval
                         (default "sparsedrive-declutter")
        stage_files / declutter_files / stage_nuscenes / num_gpus: as in
        train_entrypoint_fn.
    """
    import ray

    num_gpus = config.get("num_gpus", 8)
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, runtime_env={"env_vars": cred_env})
    def _eval(cfg, n):
        _run_combined_eval(cfg, n)

    ray.get(_eval.remote(config, num_gpus))


def _parse_planning_table(text):
    """Parse the PlanningMetric PrettyTable printed by planning_eval into
    {metric}_{horizon} floats, e.g. plan_L2_1.0s, plan_obj_box_col_avg.
    Collision percentages are stored as printed (percent units)."""
    import re

    header = None
    out = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "metrics":
            header = cells[1:]
            continue
        if header and cells and cells[0] in ("L2", "obj_col", "obj_box_col"):
            for h, v in zip(header, cells[1:]):
                out[f"plan_{cells[0]}_{h}"] = float(v.rstrip("%"))
    return out


def _run_combined_eval(config, num_gpus):
    import glob
    import json
    import subprocess
    import threading

    repo_root = os.path.abspath(os.path.dirname(__file__))
    evals = config["evals"]
    results_prefix = config.get("results_prefix", "eval_logs/plan_compare")
    wandb_project = config.get("wandb_project", "sparsedrive-declutter")
    s3 = _s3_client()

    stop_keepalive = threading.Event()

    def _gpu_keepalive():
        try:
            import torch as _t
            import time as _time
            if not _t.cuda.is_available():
                return
            x = _t.randn(2048, 2048, device="cuda")
            while not stop_keepalive.is_set():
                with _t.no_grad():
                    x = _t.mm(x, x.fmod(100.0))
                _t.cuda.synchronize()
                _time.sleep(3.0)
        except Exception as e:
            logger.warning("keepalive error: %s", e)

    threading.Thread(target=_gpu_keepalive, daemon=True).start()

    try:
        # ── env + data staging (identical mechanism to training) ───────────
        if not os.path.exists(os.path.join(ENV_LOCAL, "bin", "python")):
            _download_file_s3(
                s3, USER_BUCKET,
                f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz",
                "/tmp/sparsedrive310_env.tar.gz",
            )
            os.makedirs(ENV_LOCAL, exist_ok=True)
            _run_cmd(["tar", "-xzf", "/tmp/sparsedrive310_env.tar.gz",
                      "-C", ENV_LOCAL], tag="env-untar")
            _run_cmd([os.path.join(ENV_LOCAL, "bin", "conda-unpack")],
                     tag="conda-unpack")
        env_python = os.path.join(ENV_LOCAL, "bin", "python")
        _run_cmd([env_python, "-c",
                  "import torch, mmdet, mmcv, wandb; print('env ok')"],
                 tag="env-sanity")

        if config.get("stage_nuscenes", True):
            nusc_dir = os.path.join(repo_root, "data", "nuscenes")
            if not os.path.isdir(os.path.join(nusc_dir, "samples")):
                _download_file_s3(
                    s3, USER_BUCKET,
                    f"{DECLUTTER_PREFIX}/data/sd_nuscenes.tar",
                    DATA_TAR_LOCAL,
                )
                os.makedirs(nusc_dir, exist_ok=True)
                _run_cmd(["tar", "-xf", DATA_TAR_LOCAL, "-C", nusc_dir],
                         tag="data-untar")
                os.remove(DATA_TAR_LOCAL)

        for entry in config.get("stage_files", DEFAULT_STAGE_FILES):
            s3_rel, repo_rel = (entry, entry) if isinstance(entry, str) else entry
            _download_file_s3(s3, USER_BUCKET, f"{NAVSIM_PREFIX}/{s3_rel}",
                              os.path.join(repo_root, repo_rel))
        for entry in config.get("declutter_files", DECLUTTER_STAGE_FILES):
            s3_rel, repo_rel = (entry, entry) if isinstance(entry, str) else entry
            _download_file_s3(s3, USER_BUCKET, f"{DECLUTTER_PREFIX}/{s3_rel}",
                              os.path.join(repo_root, repo_rel))
        _download_file_s3(
            s3, USER_BUCKET, f"{DECLUTTER_PREFIX}/ckpt/resnet50-19c8e357.pth",
            os.path.join(repo_root, "ckpt/resnet50-19c8e357.pth"))

        # ── plugin CUDA ops ─────────────────────────────────────────────────
        ops_dir = os.path.join(repo_root, "projects", "mmdet3d_plugin", "ops")
        for so in glob.glob(os.path.join(ops_dir, "*.so")):
            os.remove(so)
        cuda_home = _find_cuda_home()
        _run_cmd([env_python, "setup.py", "build_ext", "--inplace"],
                 tag="ops-build", cwd=ops_dir,
                 env={**os.environ, "CUDA_HOME": cuda_home,
                      "PATH": f"{os.path.join(cuda_home, 'bin')}:{os.environ.get('PATH', '')}",
                      "FORCE_CUDA": "1",
                      "TORCH_CUDA_ARCH_LIST": config.get(
                          "torch_cuda_arch_list", "8.0")})

        # ── run evals, one GPU each ─────────────────────────────────────────
        def run_one(gpu, ev):
            name = ev["name"]
            local_ckpt = f"/tmp/eval_ckpt_{name}.pth"
            bucket, key = ev["ckpt_s3"].replace("s3://", "").split("/", 1)
            _download_file_s3(_s3_client(), bucket, key, local_ckpt)
            log_path = f"/tmp/eval_{name}.log"
            cmd = [env_python, os.path.join(repo_root, "tools", "test.py"),
                   os.path.join(repo_root, ev["config_file"]), local_ckpt,
                   "--eval", "bbox", "--cfg-options",
                   f"work_dir=/tmp/eval_wd_{name}",
                   "data.workers_per_gpu=4",
                   ] + list(ev.get("cfg_options", []))
            env = {**os.environ, "PYTHONPATH": repo_root,
                   "CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": "disabled"}
            with open(log_path, "w") as f:
                p = subprocess.Popen(cmd, cwd=repo_root, stdout=f,
                                     stderr=subprocess.STDOUT, env=env)
            rc = p.wait()
            text = open(log_path, errors="replace").read()
            metrics = _parse_planning_table(text)
            logger.info("[eval] %s exit=%d metrics=%s", name, rc, metrics)
            s3w = _s3_client()
            s3w.put_object(Bucket=USER_BUCKET,
                           Key=f"{NAVSIM_PREFIX}/{results_prefix}/{name}.log",
                           Body=text.encode())
            s3w.put_object(Bucket=USER_BUCKET,
                           Key=f"{NAVSIM_PREFIX}/{results_prefix}/{name}.json",
                           Body=json.dumps(metrics, indent=1).encode())
            if rc != 0:
                raise RuntimeError(f"eval {name} failed rc={rc}, see log")
            if metrics:
                payload = json.dumps({"name": name, "metrics": metrics,
                                      "project": wandb_project})
                script = (
                    "import json, sys, wandb\n"
                    "d = json.loads(sys.argv[1])\n"
                    "w = wandb.init(project=d['project'],\n"
                    "               id='planval-' + d['name'],\n"
                    "               name='planval_' + d['name'],\n"
                    "               resume='allow')\n"
                    "w.summary.update(d['metrics'])\n"
                    "w.finish()\n"
                )
                _run_cmd([env_python, "-c", script, payload],
                         tag=f"wandb-{name}")
            try:
                os.remove(local_ckpt)
            except OSError:
                pass
            return metrics

        errors = []

        def worker(gpu, ev):
            try:
                run_one(gpu, ev)
            except Exception as e:  # surface after all finish
                logger.exception("[eval] %s failed", ev["name"])
                errors.append((ev["name"], str(e)))

        stop_keepalive.set()  # release GPUs for the eval processes
        time.sleep(3)
        threads = []
        for i, ev in enumerate(evals):
            th = threading.Thread(target=worker,
                                  args=(i % num_gpus, ev), daemon=False)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        if errors:
            raise RuntimeError(f"evals failed: {errors}")
        logger.info("[eval] all %d evals done", len(evals))
    finally:
        stop_keepalive.set()
