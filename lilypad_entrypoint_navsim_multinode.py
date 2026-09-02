"""Multi-node Lilypad entrypoint for NAVSIM SparseDrive training.

Multi-node counterpart of lilypad_entrypoint_navsim.py (which stays the
single-node path and is untouched — the running jobs use it). Lilypad
provisions one Ray cluster with num_gpus // 8 worker machines when the
workload config asks for a multiple of 8 GPUs on a100.8 (see
lilypad.public.schemas.workload_config: num_gpus may be any multiple of
GPUS_PER_MACHINE). The generic-workload entrypoint_fn runs on the Ray head
node; here it:

  1. reserves a STRICT_SPREAD placement group of num_nodes x {8 GPU}
     bundles (one bundle per machine),
  2. resolves the bundle-0 node IP as the torch.distributed master,
  3. launches one Ray task per node; every task stages the conda env,
     infos/anchors, per-log frame tars and checkpoints LOCALLY (same
     .staged_complete marker logic as the single-node entrypoint),
  4. synchronizes all nodes on a barrier actor after staging, then each
     node runs
        torchrun --nnodes=N --node_rank=r --master_addr=<bundle-0 ip>
                 --master_port=28651 --nproc_per_node=8
     so torchrun's static rendezvous never has to wait out staging skew.

Checkpoint sync-to-S3 and the final work_dir upload run on node 0 only
(mmcv rank 0 writes checkpoints there). W&B logging is mmcv's master-only
WandbLoggerHook, i.e. also node 0.
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
    _download_file_s3,
    _find_cuda_home,
    _put_file_nonchunked,
    _run_cmd,
    _s3_client,
    _upload_dir,
)
from lilypad_entrypoint_navsim import (
    DEFAULT_STAGE_FILES,
    NAVSIM_PREFIX,
    _stage_url_files,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

MASTER_PORT = 28651


def train_entrypoint_fn(config: dict[Any, Any]) -> None:
    """Train one NAVSIM run on num_gpus GPUs across num_gpus // 8 nodes.

    config keys: identical to lilypad_entrypoint_navsim.train_entrypoint_fn
    plus:
        num_gpus          : total GPUs (multiple of gpus_per_node), e.g. 16
        gpus_per_node     : default 8
        extra_cfg_options : optional list of mmcv --cfg-options strings,
                            e.g. ["runner.max_iters=200"] for smoke runs
    """
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    num_gpus = int(config.get("num_gpus", 16))
    gpus_per_node = int(config.get("gpus_per_node", 8))
    assert num_gpus % gpus_per_node == 0, (num_gpus, gpus_per_node)
    num_nodes = num_gpus // gpus_per_node

    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    logger.info("[topo] reserving %d x {%d GPU} STRICT_SPREAD bundles",
                num_nodes, gpus_per_node)
    pg = placement_group(
        [{"GPU": gpus_per_node, "CPU": 8}] * num_nodes,
        strategy="STRICT_SPREAD",
    )
    ray.get(pg.ready(), timeout=3600)

    @ray.remote(num_cpus=1)
    def _node_ip() -> str:
        return ray.util.get_node_ip_address()

    master_addr = ray.get(_node_ip.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0),
    ).remote())
    logger.info("[topo] master (bundle 0) at %s:%d", master_addr, MASTER_PORT)

    @ray.remote
    class _Barrier:
        def __init__(self, n: int) -> None:
            self.n = n
            self.ready = set()

        def arrive(self, rank: int) -> None:
            self.ready.add(rank)

        def all_arrived(self) -> bool:
            return len(self.ready) >= self.n

    barrier = _Barrier.remote(num_nodes)

    @ray.remote(num_gpus=gpus_per_node, num_cpus=4, max_retries=0,
                runtime_env={"env_vars": cred_env})
    def _train_node(cfg, node_rank):
        _run_navsim_training_node(
            cfg, node_rank, num_nodes, gpus_per_node, master_addr, barrier)

    futures = [
        _train_node.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i),
        ).remote(config, i)
        for i in range(num_nodes)
    ]
    ray.get(futures)


def _run_navsim_training_node(config, node_rank, num_nodes, gpus_per_node,
                              master_addr, barrier):
    """Stage data locally and run this node's torchrun shard.

    Staging (env, infos, anchors, frame tars, ops build, init/resume ckpts)
    is per-node: every node needs the full dataset and env on local disk.
    Only node 0 uploads checkpoints/work_dir (rank 0 lives there).
    """
    import glob
    import threading

    import ray

    repo_root = os.path.abspath(os.path.dirname(__file__))
    config_file = config["config_file"]
    seed = int(config.get("seed", 0))
    run_name = config["run_name"]
    load_from_s3 = config.get("load_from_s3")
    work_dir = f"/tmp/work_dirs/{run_name}"
    work_s3_prefix = f"{NAVSIM_PREFIX}/work_dirs/{run_name}"
    blobs_root = config.get("blobs_root", "/tmp/navsim_blobs/mini")
    is_chief = node_rank == 0

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
            logger.warning("[node %d] keepalive error: %s", node_rank, e)

    threading.Thread(target=_gpu_keepalive, daemon=True).start()

    try:
        # ── 1. conda env (shared with the declutter runs) ───────────────────
        env_s3_key = config.get(
            "env_s3_key", f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz"
        )
        env_local = ENV_LOCAL
        if env_s3_key != f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz":
            import hashlib
            env_local = ENV_LOCAL + "_" + hashlib.sha256(
                env_s3_key.encode()
            ).hexdigest()[:8]
        if not os.path.exists(os.path.join(env_local, "bin", "python")):
            logger.info("[node %d][env] downloading packed conda env %s ...",
                        node_rank, env_s3_key)
            _download_file_s3(
                s3, USER_BUCKET, env_s3_key, "/tmp/sparsedrive310_env.tar.gz",
            )
            os.makedirs(env_local, exist_ok=True)
            _run_cmd(["tar", "-xzf", "/tmp/sparsedrive310_env.tar.gz",
                      "-C", env_local], tag="env-untar")
            _run_cmd([os.path.join(env_local, "bin", "conda-unpack")],
                     tag="conda-unpack")
        env_python = os.path.join(env_local, "bin", "python")
        _run_cmd([env_python, "-c",
                  "import torch, mmdet, mmcv, mmseg, flash_attn, cv2, "
                  "shapely, pyquaternion; "
                  "print('env sanity OK', torch.__version__)"],
                 tag="env-sanity")
        logger.info("[node %d][env] ready: %s", node_rank, env_python)

        # ── 2. data: infos + anchors + current-frame images + resnet ckpt ──
        for entry in config.get("stage_files", DEFAULT_STAGE_FILES):
            if isinstance(entry, str):
                s3_rel, repo_rel = entry, entry
            else:
                s3_rel, repo_rel = entry
            _download_file_s3(s3, USER_BUCKET, f"{NAVSIM_PREFIX}/{s3_rel}",
                              os.path.join(repo_root, repo_rel))
        _stage_url_files(repo_root, config.get("url_files", []))
        _download_file_s3(s3, USER_BUCKET,
                          f"{DECLUTTER_PREFIX}/ckpt/resnet50-19c8e357.pth",
                          os.path.join(repo_root, "ckpt/resnet50-19c8e357.pth"))

        staged_marker = os.path.join(blobs_root, ".staged_complete")
        frames_prefix = config.get("frames_prefix")
        if not os.path.exists(staged_marker):
            os.makedirs(blobs_root, exist_ok=True)
            if frames_prefix:
                # sharded per-log tars (navtrain): parallel download + untar
                from concurrent.futures import ThreadPoolExecutor
                prefix = f"{NAVSIM_PREFIX}/{frames_prefix}".rstrip("/") + "/"
                keys = []
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=USER_BUCKET,
                                               Prefix=prefix):
                    keys += [o["Key"] for o in page.get("Contents", [])
                             if o["Key"].endswith(".tar")]
                if not keys:
                    raise RuntimeError(f"no frame tars under s3://"
                                       f"{USER_BUCKET}/{prefix}")
                logger.info("[node %d][blobs] staging %d frame tars from %s",
                            node_rank, len(keys), prefix)
                t0 = time.monotonic()
                n_done = [0]

                def _stage_one(key):
                    local_tar = os.path.join(
                        "/tmp", "frame_tars", os.path.basename(key))
                    os.makedirs(os.path.dirname(local_tar), exist_ok=True)
                    # thread-local client: boto3 clients are not thread-safe
                    _s3_client().download_file(USER_BUCKET, key, local_tar)
                    _run_cmd(["tar", "-xf", local_tar, "-C", blobs_root],
                             tag=f"untar-{os.path.basename(key)}")
                    os.remove(local_tar)
                    n_done[0] += 1
                    if n_done[0] % 100 == 0:
                        logger.info(
                            "[node %d][blobs] %d/%d tars staged (%.0f s)",
                            node_rank, n_done[0], len(keys),
                            time.monotonic() - t0)

                with ThreadPoolExecutor(max_workers=16) as pool:
                    list(pool.map(_stage_one, keys))
                logger.info("[node %d][blobs] staged %d tars in %.0f s",
                            node_rank, len(keys), time.monotonic() - t0)
            else:
                frames_tar = config.get(
                    "frames_tar", "data/navmini_current_frames.tar")
                local_tar = f"/tmp/{os.path.basename(frames_tar)}"
                _download_file_s3(s3, USER_BUCKET,
                                  f"{NAVSIM_PREFIX}/{frames_tar}", local_tar)
                _run_cmd(["tar", "-xf", local_tar, "-C", blobs_root],
                         tag="blobs-untar")
                os.remove(local_tar)
            with open(staged_marker, "w") as f:
                f.write("ok\n")

        # ── 2b. PDM metric supervision assets (per node, optional) ─────────
        from lilypad_entrypoint_navsim import _stage_metric_assets
        metric_cache_root, devkit_root = _stage_metric_assets(s3, config)

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

        # ── 4. init / resume checkpoints (every node: mmcv loads per rank) ──
        os.makedirs(work_dir, exist_ok=True)
        cfg_options = []
        if load_from_s3:
            local_init = "/tmp/init_ckpt.pth"
            bucket, key = load_from_s3.replace("s3://", "").split("/", 1)
            _download_file_s3(s3, bucket, key, local_init)
            cfg_options += [f"load_from={local_init}"]
        cfg_options += list(config.get("extra_cfg_options", []))

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
                logger.info("[node %d][resume] resuming from %s",
                            node_rank, cands[-1])
        except Exception as e:
            logger.warning("[node %d][resume] check failed (fresh start): %s",
                           node_rank, e)

        # ── 5. periodic checkpoint upload (node 0 only: rank 0 writes) ─────
        if is_chief:
            def _upload_latest_forever():
                last_uploaded = None
                while not stop_sync.is_set():
                    time.sleep(300)
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
                                _put_file_nonchunked(
                                    s3_bg, target, USER_BUCKET, key)
                                last_uploaded = target
                    except Exception as e:
                        logger.warning("[ckpt-sync] %s", e)

            threading.Thread(target=_upload_latest_forever, daemon=True).start()

        # ── 6. barrier: wait until every node finished staging ─────────────
        ray.get(barrier.arrive.remote(node_rank))
        t0 = time.monotonic()
        while not ray.get(barrier.all_arrived.remote()):
            if time.monotonic() - t0 > 4 * 3600:
                raise RuntimeError("timed out waiting for peer nodes to stage")
            time.sleep(10)
        logger.info("[node %d] all %d nodes staged — starting torchrun",
                    node_rank, num_nodes)

        # ── 7. train ────────────────────────────────────────────────────────
        train_env = {
            **os.environ,
            "PATH": f"{os.path.join(env_local, 'bin')}:{os.environ.get('PATH', '')}",
            "PYTHONPATH": repo_root,
            "WANDB_NAME": run_name,
            "PYTHONUNBUFFERED": "1",
            "NAVSIM_BLOBS_ROOT": blobs_root,
        }
        if metric_cache_root:
            train_env["SPARSEDRIVE_METRIC_CACHE_ROOT"] = metric_cache_root
            train_env["SPARSEDRIVE_PDM_WORKERS"] = str(
                config.get("pdm_workers", 8)
            )
        if devkit_root:
            train_env["SPARSEDRIVE_NAVSIM_DEVKIT_ROOT"] = devkit_root
        stop_keepalive.set()  # release GPUs for torchrun
        time.sleep(5)
        _run_cmd(
            [env_python, "-m", "torch.distributed.run",
             f"--nnodes={num_nodes}",
             f"--node_rank={node_rank}",
             f"--master_addr={master_addr}",
             f"--master_port={MASTER_PORT}",
             f"--nproc_per_node={gpus_per_node}",
             os.path.join(repo_root, "tools", "train_pyfocal.py"),
             os.path.join(repo_root, config_file),
             "--launcher", "pytorch",
             "--seed", str(seed),
             "--work-dir", work_dir]
            + (["--cfg-options"] + cfg_options if cfg_options else [])
            + resume_args
            + ["--no-validate"],
            tag=f"train-node{node_rank}", cwd=repo_root, env=train_env,
        )

        # ── 8. final upload: full work_dir (ckpts, logs, dumped config) ────
        if is_chief:
            logger.info("[upload] final work_dir → s3://%s/%s",
                        USER_BUCKET, work_s3_prefix)
            _upload_dir(s3, work_dir, USER_BUCKET, work_s3_prefix)
        logger.info("[done][node %d] %s", node_rank, run_name)
    finally:
        stop_keepalive.set()
        stop_sync.set()
