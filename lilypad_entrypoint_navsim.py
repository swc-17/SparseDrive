"""Lilypad entrypoint for NAVSIM SparseDrive training (Phase 1+).

Mirrors lilypad_entrypoint.train_entrypoint_fn (declutter matrix) but stages
NAVSIM artifacts instead of nuScenes:

  s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/
    data/infos/navsim_infos_navmini.pkl        converted navmini infos
                                               (v1.1+: map_annos populated)
    data/kmeans/kmeans_det_900_navsim.npy      navmini det anchors
    data/kmeans/kmeans_map_100_navsim.npy      navmini map anchors (Phase 2)
    data/kmeans/kmeans_motion_6_navsim.npy     navmini motion anchors (Phase 3)
    data/kmeans/kmeans_plan_6_navsim.npy       navmini plan anchors (Phase 3)
    data/navmini_current_frames.tar            current-frame jpgs (8 cams),
                                               paths relative to the blob root
    work_dirs/<run_name>/                      checkpoints + logs (output)

The packed conda env and the ResNet50 init checkpoint are reused from the
declutter prefix (users/tejan/sparsedrive/declutter/). Training runs through
tools/train_pyfocal.py (pure-python focal loss) exactly like the declutter
runs. The config reads the sensor-blob root from $NAVSIM_BLOBS_ROOT.
"""

import logging
import hashlib
import os
import sys
import time
import urllib.request
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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

NAVSIM_PREFIX = "users/tejan/navsim/sparsedrive"

TEACHER_DETECTIONS_EMPERROR_MAPS = "teacher_detections_emperror_maps"
EMPERROR_DETECTIONS_TEACHER_MAPS = "emperror_detections_teacher_maps"
OCCURRENCE_GEOMETRY_SOURCE = "emperror_occurrence"
GENERATED_GEOMETRY_SOURCES = (
    "emperror",
    TEACHER_DETECTIONS_EMPERROR_MAPS,
    EMPERROR_DETECTIONS_TEACHER_MAPS,
)
FILE_GEOMETRY_SOURCES = (*GENERATED_GEOMETRY_SOURCES, OCCURRENCE_GEOMETRY_SOURCE)
EXTERNAL_GEOMETRY_SOURCES = ("teacher", *FILE_GEOMETRY_SOURCES)

SD15_METRIC_CONFIG_FILE = (
    "projects/configs/navsim/sparsedrive_navsim_stage2_vocab_metric_full.py"
)
SD15_METRIC_CONFIG_SHA256 = (
    "ff0784e6c2dd59e067cb734a57acfcd02c993ea91245b880e0142623f00f44c6"
)
SD15_METRIC_CHECKPOINT_S3 = (
    "s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/"
    "work_dirs/navsim_stage2_vocab_metric_16g/iter_8703.pth"
)
SD15_METRIC_CHECKPOINT_SHA256 = (
    "2631b5ea4102bf4d53796dc236b9bbc25a7114bb8c5b6505814f45ebfa426987"
)
SD15_METRIC_STAGE_FILES = (
    (
        "data/kmeans/navtrain/kmeans_det_900_navsim.npy",
        "data/kmeans/kmeans_det_900_navsim.npy",
        "98bb7b616d71719033b4a30faaf17727665fe070827255c1005e178867f54462",
    ),
    (
        "data/kmeans/navtrain/kmeans_map_100_navsim.npy",
        "data/kmeans/kmeans_map_100_navsim.npy",
        "ff1443bb48f38fe6df78f57f597c5f04917630f95c2c55ee65824370edd6dd0c",
    ),
    (
        "data/kmeans/navtrain/kmeans_motion_6_navsim.npy",
        "data/kmeans/kmeans_motion_6_navsim.npy",
        "f0eed3f0636d69b00e07ccbc9222ded86b27c53eef126a280a0d292912c3302f",
    ),
    (
        "data/kmeans/sparsedrive_v2/path_1024.npy",
        "data/kmeans/sparsedrive_v2/path_1024.npy",
        "97ff2e843c264fa15a38a6a3920985fd4bdb8f387ef409b2a97d44eb5c88f9c1",
    ),
    (
        "data/kmeans/sparsedrive_v2/velocity_256.npy",
        "data/kmeans/sparsedrive_v2/velocity_256.npy",
        "acfaa8699401a20283c2744f7ea4beb4f8c734f3a3f145ca0366a3ba6abae2ee",
    ),
    (
        "data/kmeans/sparsedrive_v2/trajectory_1024_256.npz",
        "data/kmeans/sparsedrive_v2/trajectory_1024_256.npz",
        "94868f922fcd7ac3a027057476ddf630948c908053fec483d8936e84b2ad8683",
    ),
)


def _with_sd15_eval_defaults(config):
    """Use the matched SD1.5 perception/planner unless explicitly pinned."""

    resolved = dict(config)
    planner_paths = (resolved.get("config_file"), resolved.get("checkpoint_s3"))
    if not any(planner_paths):
        resolved.update(
            config_file=SD15_METRIC_CONFIG_FILE,
            config_sha256=SD15_METRIC_CONFIG_SHA256,
            checkpoint_s3=SD15_METRIC_CHECKPOINT_S3,
            checkpoint_sha256=SD15_METRIC_CHECKPOINT_SHA256,
            stage_files=[
                {"s3_rel": source, "repo_rel": target, "sha256": digest}
                for source, target, digest in SD15_METRIC_STAGE_FILES
            ],
        )
    elif not all(planner_paths):
        raise ValueError("config_file and checkpoint_s3 must be provided together")
    if resolved.get("geometry_source") in GENERATED_GEOMETRY_SOURCES:
        resolved.setdefault(
            "geometry_teacher_config_sha256", SD15_METRIC_CONFIG_SHA256
        )
        resolved.setdefault(
            "geometry_teacher_checkpoint_sha256",
            SD15_METRIC_CHECKPOINT_SHA256,
        )
    return resolved

# Default data staging (navmini overfit runs). Entries are either an s3 path
# relative to NAVSIM_PREFIX (downloaded to the same repo-relative path) or a
# [s3_rel, repo_rel] pair.
DEFAULT_STAGE_FILES = [
    "data/infos/navsim_infos_navmini.pkl",
    "data/kmeans/kmeans_det_900_navsim.npy",
    "data/kmeans/kmeans_map_100_navsim.npy",
    "data/kmeans/kmeans_motion_6_navsim.npy",   # stage 2
    "data/kmeans/kmeans_plan_6_navsim.npy",     # stage 2
]


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_url_files(repo_root, entries):
    """Download small public model assets with an optional SHA-256 gate."""
    for entry in entries:
        url, repo_rel = entry[:2]
        expected_sha256 = entry[2] if len(entry) > 2 else None
        destination = os.path.join(repo_root, repo_rel)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        temporary = destination + ".part"
        logger.info("[asset] downloading %s -> %s", url, repo_rel)
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "SparseDrive-stage2"}
                )
                with urllib.request.urlopen(
                    request, timeout=60
                ) as source, open(temporary, "wb") as sink:
                    while True:
                        chunk = source.read(1 << 20)
                        if not chunk:
                            break
                        sink.write(chunk)
                break
            except Exception:
                if attempt == 2:
                    raise
                logger.warning("[asset] retrying %s", url)
                time.sleep(2 ** attempt)
        if expected_sha256 is not None:
            digest = hashlib.sha256()
            with open(temporary, "rb") as asset:
                for chunk in iter(lambda: asset.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                os.remove(temporary)
                raise RuntimeError(f"SHA-256 mismatch for {url}")
        os.replace(temporary, destination)


def _stage_metric_assets(s3, config):
    """Stage PDM metric supervision assets onto the local node (optional).

    Returns (metric_cache_root, devkit_root); both None unless the config
    carries metric_cache_s3_prefix / navsim_devkit_s3_key. Idempotent per
    node (markers), so multinode nodes and warm restarts skip re-downloads.
    """
    metric_cache_root = None
    devkit_root = None
    metric_cache_prefix = config.get("metric_cache_s3_prefix")
    if metric_cache_prefix:
        metric_cache_root = "/tmp/metric_cache_navtrain"
        cache_marker = os.path.join(metric_cache_root, ".staged_complete")
        if not os.path.exists(cache_marker):
            from concurrent.futures import ThreadPoolExecutor
            os.makedirs(metric_cache_root, exist_ok=True)
            prefix = metric_cache_prefix.rstrip("/") + "/"
            keys = []
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=USER_BUCKET,
                                           Prefix=prefix):
                keys += [o["Key"] for o in page.get("Contents", [])
                         if o["Key"].endswith("metric_cache.pkl")]
            if not keys:
                raise RuntimeError(
                    f"no metric caches under s3://{USER_BUCKET}/{prefix}")
            logger.info("[metric-cache] staging %d caches from %s",
                        len(keys), prefix)
            t0 = time.monotonic()
            n_done = [0]

            def _stage_cache(key):
                # merge metric_cache_worker*/ shards into one tree:
                # keep only {log}/unknown/{token}/metric_cache.pkl
                rel = key[len(prefix):].split("/", 1)[1]
                dest = os.path.join(metric_cache_root, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                _s3_client().download_file(USER_BUCKET, key, dest)
                n_done[0] += 1
                if n_done[0] % 20000 == 0:
                    logger.info("[metric-cache] %d/%d (%.0f s)",
                                n_done[0], len(keys),
                                time.monotonic() - t0)

            with ThreadPoolExecutor(max_workers=32) as pool:
                list(pool.map(_stage_cache, keys))
            logger.info("[metric-cache] staged %d caches in %.0f s",
                        len(keys), time.monotonic() - t0)
            with open(cache_marker, "w") as f:
                f.write("ok\n")

    devkit_s3_key = config.get("navsim_devkit_s3_key")
    if devkit_s3_key:
        devkit_root = "/tmp/navsim_devkit"
        if not os.path.exists(os.path.join(devkit_root, "navsim")):
            local_tar = "/tmp/navsim_devkit.tar.gz"
            _download_file_s3(s3, USER_BUCKET, devkit_s3_key, local_tar)
            os.makedirs(devkit_root, exist_ok=True)
            _run_cmd(["tar", "-xzf", local_tar, "-C", devkit_root],
                     tag="devkit-untar")
            os.remove(local_tar)
    return metric_cache_root, devkit_root


def train_entrypoint_fn(config: dict[Any, Any]) -> None:
    """Train one NAVSIM run on an 8-GPU node.

    config keys:
        config_file   : repo-relative mmcv config, e.g.
                        "projects/configs/navsim/sparsedrive_navsim_stage1_overfit.py"
        seed          : int random seed
        run_name      : wandb run name + S3 work_dir leaf
        load_from_s3  : S3 URI of an init checkpoint (null for stage1)
        num_gpus      : default 8
        stage_files   : optional list of NAVSIM_PREFIX-relative data files to
                        download (str, or [s3_rel, repo_rel] pair); default
                        DEFAULT_STAGE_FILES (navmini artifacts)
        url_files     : optional [url, repo-relative destination, sha256]
                        entries for public assets such as the V2 vocabulary
        frames_prefix : optional NAVSIM_PREFIX-relative prefix of sharded
                        per-log image tars (e.g. "data/navtrain_frames/");
                        all *.tar under it are downloaded + untarred into
                        blobs_root in parallel (navtrain staging)
        frames_tar    : single-tar staging (default
                        "data/navmini_current_frames.tar"); ignored when
                        frames_prefix is set
        blobs_root    : where images are untarred; exported as
                        NAVSIM_BLOBS_ROOT (default "/tmp/navsim_blobs/mini")
        env_s3_key    : packed conda env key in USER_BUCKET (default the
                        shared declutter sparsedrive310 env). Runs needing
                        extra packages (nuplan for PDM metric heads) point
                        this at their own tarball.
        metric_cache_s3_prefix : optional USER_BUCKET prefix holding
                        metric_cache_worker*/ shards ({log}/unknown/{token}/
                        metric_cache.pkl). All shards are merged into one
                        local tree exported as SPARSEDRIVE_METRIC_CACHE_ROOT.
        navsim_devkit_s3_key : optional USER_BUCKET key of a tarball with the
                        navsim v2 devkit source; extracted and exported as
                        SPARSEDRIVE_NAVSIM_DEVKIT_ROOT for the PDM scoring
                        workers.
        pdm_workers   : scoring processes per training rank (default 8).
    """
    import ray

    num_gpus = config.get("num_gpus", 8)
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, runtime_env={"env_vars": cred_env})
    def _train(cfg, n):
        _run_navsim_training(cfg, n)

    ray.get(_train.remote(config, num_gpus))


def _run_navsim_training(config, num_gpus):
    import glob
    import threading

    repo_root = os.path.abspath(os.path.dirname(__file__))
    config_file = config["config_file"]
    seed = int(config.get("seed", 0))
    run_name = config["run_name"]
    load_from_s3 = config.get("load_from_s3")
    work_dir = f"/tmp/work_dirs/{run_name}"
    work_s3_prefix = f"{NAVSIM_PREFIX}/work_dirs/{run_name}"
    blobs_root = config.get("blobs_root", "/tmp/navsim_blobs/mini")

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
        # ── 1. conda env (shared with the declutter runs) ───────────────────
        env_s3_key = config.get(
            "env_s3_key", f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz"
        )
        # Custom envs unpack beside the default so a cached default env on a
        # reused node is never mistaken for one carrying extra packages.
        env_local = ENV_LOCAL
        if env_s3_key != f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz":
            env_local = ENV_LOCAL + "_" + hashlib.sha256(
                env_s3_key.encode()
            ).hexdigest()[:8]
        if not os.path.exists(os.path.join(env_local, "bin", "python")):
            logger.info("[env] downloading packed conda env %s ...", env_s3_key)
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
        logger.info("[env] ready: %s", env_python)

        # ── 2. data: infos + anchors + current-frame images + resnet ckpt ──
        for entry in config.get("stage_files", DEFAULT_STAGE_FILES):
            if isinstance(entry, str):
                s3_rel, repo_rel = entry, entry
            else:
                s3_rel, repo_rel = entry
            _download_file_s3(s3, USER_BUCKET, f"{NAVSIM_PREFIX}/{s3_rel}",
                              os.path.join(repo_root, repo_rel))
        _stage_url_files(repo_root, config.get("url_files", []))
        _download_file_s3(s3, USER_BUCKET, f"{DECLUTTER_PREFIX}/ckpt/resnet50-19c8e357.pth",
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
                logger.info("[blobs] staging %d frame tars from %s",
                            len(keys), prefix)
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
                        logger.info("[blobs] %d/%d tars staged (%.0f s)",
                                    n_done[0], len(keys),
                                    time.monotonic() - t0)

                with ThreadPoolExecutor(max_workers=16) as pool:
                    list(pool.map(_stage_one, keys))
                logger.info("[blobs] staged %d tars in %.0f s", len(keys),
                            time.monotonic() - t0)
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

        # ── 2b. PDM metric supervision assets (optional) ────────────────────
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
                time.sleep(300)
                try:
                    latest = os.path.join(work_dir, "latest.pth")
                    if os.path.islink(latest):
                        target = os.path.realpath(latest)
                        if target != last_uploaded and os.path.exists(target):
                            iter_tag = os.path.basename(target).replace(".pth", "")
                            key = f"{work_s3_prefix}/latest_{iter_tag}.pth"
                            logger.info("[ckpt-sync] uploading %s", key)
                            s3_bg = _s3_client()
                            _put_file_nonchunked(s3_bg, target, USER_BUCKET, key)
                            last_uploaded = target
                except Exception as e:
                    logger.warning("[ckpt-sync] %s", e)

        threading.Thread(target=_upload_latest_forever, daemon=True).start()

        # ── 6. train ────────────────────────────────────────────────────────
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
        logger.info("[upload] final work_dir → s3://%s/%s", USER_BUCKET, work_s3_prefix)
        _upload_dir(s3, work_dir, USER_BUCKET, work_s3_prefix)
        logger.info("[done] %s", run_name)
    finally:
        stop_keepalive.set()
        stop_sync.set()


# ═══════════════════════════════════════════════════════════════════════════
# NAVSIM evaluation entrypoint (Phase 4 on-cluster: v1 PDMS + v2 EPDMS)
# ═══════════════════════════════════════════════════════════════════════════

EVAL_ENV_LOCAL = "/tmp/navsim_eval_env"     # packed conda `lilypad` env
SDV2_LOCAL = "/tmp/SparseDriveV2"           # committed-tree snapshot
DATA_ROOT = "/tmp/navsim_data"              # OPENSCENE_DATA_ROOT on the node


def _parse_s3_uri(uri):
    assert uri.startswith("s3://"), uri
    bucket, key = uri[5:].split("/", 1)
    return bucket, key.rstrip("/")


def _sync_prefix(s3, uri, local_dir, workers=16):
    """Threaded download of every object under an s3:// prefix."""
    from concurrent.futures import ThreadPoolExecutor

    bucket, prefix = _parse_s3_uri(uri)
    prefix = prefix + "/"
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    assert keys, f"nothing under {uri}"
    logger.info("[sync] %d objects %s -> %s", len(keys), uri, local_dir)
    n_done = [0]
    t0 = time.monotonic()

    def _one(key):
        dst = os.path.join(local_dir, key[len(prefix):])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        _s3_client().download_file(bucket, key, dst)
        n_done[0] += 1
        if n_done[0] % 2000 == 0:
            logger.info("[sync] %d/%d (%.0f s)", n_done[0], len(keys),
                        time.monotonic() - t0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_one, keys))
    return len(keys)


def _stage_tars(s3, uri, target_dir, workers=12):
    """Download every *.tar under an s3:// prefix and untar into target_dir.

    Hardened after two silent staging hangs (workloads 9aydm9/hfrlhs):
    downloads run single-threaded inside each worker (use_threads=False —
    boto3's per-call multipart pools deadlocked under 12 concurrent
    download_file calls), every file gets bounded retries, and a heartbeat
    logs pool progress so a stall is visible in the workload logs.
    """
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from boto3.s3.transfer import TransferConfig

    bucket, prefix = _parse_s3_uri(uri)
    prefix = prefix + "/"
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])
                 if o["Key"].endswith(".tar")]
    assert keys, f"no tars under {uri}"
    os.makedirs(target_dir, exist_ok=True)
    logger.info("[tars] staging %d tars from %s", len(keys), uri)

    xfer_cfg = TransferConfig(use_threads=False)
    n_done = [0]
    done_lock = threading.Lock()
    stop_beat = threading.Event()

    def _heartbeat():
        while not stop_beat.wait(120):
            logger.info("[tars] progress %d/%d", n_done[0], len(keys))

    def _one(key):
        local_tar = os.path.join("/tmp/eval_tars", os.path.basename(key))
        os.makedirs(os.path.dirname(local_tar), exist_ok=True)
        last_err = None
        for attempt in range(3):
            try:
                _s3_client().download_file(bucket, key, local_tar,
                                           Config=xfer_cfg)
                break
            except Exception as e:  # noqa: BLE001 — retried, then re-raised
                last_err = e
                logger.warning("[tars] retry %d for %s: %s",
                               attempt + 1, key, e)
        else:
            raise RuntimeError(f"download failed after retries: {key}") \
                from last_err
        subprocess.run(["tar", "-xf", local_tar, "-C", target_dir],
                       check=True, timeout=600)
        os.remove(local_tar)
        with done_lock:
            n_done[0] += 1

    beat = threading.Thread(target=_heartbeat, daemon=True)
    beat.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_one, keys))
    finally:
        stop_beat.set()
    logger.info("[tars] staged %d/%d tars", n_done[0], len(keys))


def _rewrite_cache_metadata(cache_dir):
    """Regenerate the metric-cache metadata CSV with node-local paths
    (the generation-time CSV records paths from the producing machine)."""
    import csv
    import glob as _glob

    pkls = sorted(_glob.glob(os.path.join(cache_dir, "**", "metric_cache.pkl"),
                             recursive=True))
    assert pkls, f"no metric_cache.pkl under {cache_dir}"
    meta_dir = os.path.join(cache_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    for old in _glob.glob(os.path.join(meta_dir, "*.csv")):
        os.remove(old)
    with open(os.path.join(meta_dir, "metric_cache_metadata_node.csv"),
              "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name"])
        for p in pkls:
            writer.writerow([p])
    logger.info("[cache] rewrote metadata for %d entries", len(pkls))
    return len(pkls)


_ONCE_LOCKS: dict = {}
_ONCE_GUARD = None  # created lazily (threading.Lock)


def _once(key, fn):
    """Run fn() exactly once per key across concurrent eval threads
    (marker file + per-key lock); safe to call from a single eval too."""
    import hashlib
    import threading

    global _ONCE_GUARD
    if _ONCE_GUARD is None:
        _ONCE_GUARD = threading.Lock()
    with _ONCE_GUARD:
        lock = _ONCE_LOCKS.setdefault(key, threading.Lock())
    with lock:
        marker = os.path.join(
            "/tmp/stage_markers", hashlib.md5(key.encode()).hexdigest())
        if os.path.exists(marker):
            return
        fn()
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w") as f:
            f.write(key)


def _uri_dir(base, uri):
    import hashlib
    return os.path.join(base, hashlib.md5(uri.encode()).hexdigest()[:12])


def _start_persistent_keepalive(num_gpus, period_s=5.0, heartbeat_s=180.0):
    """Keep EVERY allocated GPU busy for the whole workload lifetime.

    The cluster's idle-resource reaper stops workloads whose GPUs idle
    (eval jobs have long CPU-only staging/scoring phases; training jobs
    never idle). A daemon thread per GPU runs a tiny matmul every
    ``period_s`` — negligible compute/memory, coexists with the inference
    subprocesses (separate CUDA context) — and logs a heartbeat every
    ``heartbeat_s``. Returns a stop Event; set it only at process exit.
    """
    import threading

    stop = threading.Event()

    def _one_gpu(idx):
        try:
            import torch as _t
            if not _t.cuda.is_available():
                logger.warning("[keepalive] no CUDA visible (gpu %d)", idx)
                return
            dev = f"cuda:{idx}"
            x = _t.randn(256, 256, device=dev)
            n = 0
            last_beat = time.monotonic()
            while not stop.is_set():
                with _t.no_grad():
                    y = _t.mm(x, x)
                _t.cuda.synchronize(dev)
                del y
                n += 1
                if time.monotonic() - last_beat >= heartbeat_s:
                    logger.info("[keepalive] gpu %d alive (%d ticks)",
                                idx, n)
                    last_beat = time.monotonic()
                stop.wait(period_s)
        except Exception as e:
            logger.warning("[keepalive] gpu %d error: %s", idx, e)

    for i in range(num_gpus):
        threading.Thread(target=_one_gpu, args=(i,), daemon=True).start()
    logger.info("[keepalive] started on %d GPU(s)", num_gpus)
    return stop


def eval_entrypoint_fn(config: dict[Any, Any]) -> None:
    """Run one NAVSIM eval (inference + scoring).

    config keys (all S3 references are full s3:// URIs):
        mode              : "pdm_v1" | "epdms_two_stage"
        split             : scorer --split (navmini | navtest |
                            navhard_two_stage | navmini_two_stage)
        config_file       : repo-relative SparseDrive eval config; omitted
                            with checkpoint_s3 to use SD1.5 metric defaults
        config_sha256     : optional pinned planner config digest
        checkpoint_s3     : model checkpoint .pth; see config_file default
        checkpoint_sha256 : optional pinned planner checkpoint digest
        run_tag           : output naming tag
        results_s3_output : upload prefix for trajs + CSV/JSON

        infos_s3          : converted infos pickle (pdm_v1 mode)
        infos_sha256      : optional pinned infos digest
        frames_sha256     : optional pinned frames digest (epdms mode)
        expected_scenarios: optional exact scenario count
        expected_replay_calls: optional exact four-frame replay count
        frames_pkl_s3     : dumped two-stage frames pickle (epdms mode)
        frame_tars_s3     : prefix of image tar shards (untarred into
                            blobs_root for pdm_v1, DATA_ROOT for epdms)
        extra_sync_s3     : optional [ [s3_prefix, DATA_ROOT-relative], ... ]
                            raw syncs (e.g. navhard synthetic sensor_blobs)
        blobs_root        : pdm_v1 image root (default /tmp/navsim_blobs)
        path_remap_from   : epdms: dumped path prefix to rewrite
                            (default /media/applied/navsim -> DATA_ROOT)

        metric_cache_s3   : metric cache prefix (REQUIRED)
        navsim_logs_s3    : navsim_logs/<split-dir> prefix (scene loader)
        maps_s3           : nuPlan maps prefix (optional; epdms only needs
                            the env var to exist)
        sdv2_snapshot_s3  : SparseDriveV2 committed-tree tar.gz
        eval_env_s3       : packed conda eval env tar.gz
        epdms_worker      : sequential | single_machine_thread_pool |
                            ray_distributed_no_torch
        require_epdms     : fail if two-stage aggregation is absent/invalid
        limit             : optional token limit (smoke runs)
        num_gpus          : GPUs reserved by this eval (default 1)
        num_shards        : concurrent inference shards (default 1)
        geometry_source   : native | teacher | emperror |
                            emperror_occurrence | hybrid source
        geometry_s3       : token/occurrence geometry artifact (file sources only)
        geometry_sha256   : pinned artifact digest (generated sources only)
        geometry_producer_checkpoint_sha256: pinned EMPERROR checkpoint digest
        geometry_teacher_config_sha256: pinned training-teacher config digest
        geometry_teacher_checkpoint_sha256: pinned training-teacher checkpoint
        ego_query_only    : keep only the ego query; agents/maps remain K/V
    """
    import ray

    num_gpus = int(config.get("num_gpus", 1))
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, max_retries=0,
                runtime_env={"env_vars": cred_env})
    def _eval(cfg, gpu_count):
        # persistent keepalive FIRST (before any staging): the idle-resource
        # reaper stops workloads whose GPUs idle during CPU-only phases
        stop = _start_persistent_keepalive(gpu_count)
        try:
            _run_navsim_eval(cfg)
        finally:
            stop.set()

    ray.get(_eval.remote(config, num_gpus))


def _validate_scoring_trajectories(path, expected_tokens):
    """Fail closed before scoring a previously exported trajectory file."""
    import pickle

    import numpy as np

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("trajectories"), dict
    ):
        raise RuntimeError("trajectory artifact has no trajectories mapping")
    trajectories = payload["trajectories"]
    if len(trajectories) != expected_tokens:
        raise RuntimeError("trajectory artifact token count differs from contract")
    for token, poses in trajectories.items():
        array = np.asarray(poses)
        if not isinstance(token, str) or array.shape != (8, 3):
            raise RuntimeError("trajectory artifact has an invalid token or shape")
        if not np.isfinite(array).all():
            raise RuntimeError("trajectory artifact contains non-finite poses")
    return payload


def score_epdms_entrypoint_fn(config: dict[Any, Any]) -> None:
    """Score an uploaded NAVHARD trajectory artifact without rerunning inference.

    This recovery path deliberately uses NAVSIM's built-in single-machine
    thread pool.  A nested Ray worker cannot start inside a Lilypad Ray job:
    Ray auto-discovers the parent cluster and rejects the worker's local
    ``num_cpus`` setting.
    """
    import glob

    required = (
        "trajectory_s3",
        "trajectory_sha256",
        "split",
        "run_tag",
        "results_s3_output",
        "metric_cache_s3",
        "navsim_logs_s3",
        "sdv2_snapshot_s3",
        "eval_env_s3",
        "expected_tokens",
    )
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f"missing score-only settings: {missing}")
    expected_tokens = config["expected_tokens"]
    if (
        not isinstance(expected_tokens, int)
        or isinstance(expected_tokens, bool)
        or expected_tokens < 1
    ):
        raise ValueError("expected_tokens must be a positive integer")
    trajectory_sha256 = config["trajectory_sha256"]
    if not isinstance(trajectory_sha256, str) or len(trajectory_sha256) != 64:
        raise ValueError("trajectory_sha256 must contain 64 characters")
    worker = config.get("epdms_worker", "single_machine_thread_pool")
    if worker not in ("sequential", "single_machine_thread_pool"):
        raise ValueError("score-only worker must not start nested Ray")

    repo_root = os.path.abspath(os.path.dirname(__file__))
    s3 = _s3_client()
    run_tag = config["run_tag"]
    out_dir = os.path.join("/tmp/epdms_score_only", run_tag)
    os.makedirs(out_dir, exist_ok=True)

    trajectory_local = os.path.join(out_dir, f"trajs_{run_tag}.pkl")
    bucket, key = _parse_s3_uri(config["trajectory_s3"])
    _download_file_s3(s3, bucket, key, trajectory_local)
    if _sha256_file(trajectory_local) != trajectory_sha256:
        raise RuntimeError("trajectory artifact SHA-256 mismatch")
    _validate_scoring_trajectories(trajectory_local, expected_tokens)
    logger.info("[score-only] accepted %d finite trajectories", expected_tokens)

    def _stage_archive(uri, local_archive, target):
        if os.path.exists(os.path.join(target, "bin", "python")) or (
            target == SDV2_LOCAL and os.path.exists(os.path.join(target, "navsim"))
        ):
            return
        archive_bucket, archive_key = _parse_s3_uri(uri)
        _download_file_s3(
            _s3_client(), archive_bucket, archive_key, local_archive
        )
        os.makedirs(target, exist_ok=True)
        _run_cmd(["tar", "-xzf", local_archive, "-C", target], tag="untar")

    _stage_archive(
        config["eval_env_s3"], "/tmp/navsim_eval_env.tar.gz", EVAL_ENV_LOCAL
    )
    eval_python = os.path.join(EVAL_ENV_LOCAL, "bin", "python")
    if os.path.exists(os.path.join(EVAL_ENV_LOCAL, "bin", "conda-unpack")):
        _run_cmd(
            [os.path.join(EVAL_ENV_LOCAL, "bin", "conda-unpack")],
            tag="eval-env-unpack",
        )
    _stage_archive(config["sdv2_snapshot_s3"], "/tmp/sdv2.tar.gz", SDV2_LOCAL)
    _run_cmd(
        [
            eval_python,
            "-c",
            "import sys; sys.path.insert(0, '"
            + SDV2_LOCAL
            + "'); import navsim, nuplan, hydra; print('eval env OK')",
        ],
        tag="eval-env-sanity",
    )

    cache_local = _uri_dir("/tmp/metric_cache", config["metric_cache_s3"])
    _sync_prefix(s3, config["metric_cache_s3"], cache_local)
    if _rewrite_cache_metadata(cache_local) != expected_tokens:
        raise RuntimeError("metric-cache token count differs from contract")
    logs_dirname = os.path.basename(_parse_s3_uri(config["navsim_logs_s3"])[1])
    _sync_prefix(
        s3,
        config["navsim_logs_s3"],
        os.path.join(DATA_ROOT, "navsim_logs", logs_dirname),
    )
    for source, relative_target in config.get("extra_sync_s3", []):
        _sync_prefix(s3, source, os.path.join(DATA_ROOT, relative_target))
    maps_root = os.path.join(DATA_ROOT, "maps")
    if config.get("maps_s3"):
        _sync_prefix(s3, config["maps_s3"], maps_root)
    os.makedirs(maps_root, exist_ok=True)

    score_env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("RAY_")},
        "PATH": f"{os.path.join(EVAL_ENV_LOCAL, 'bin')}:"
        f"{os.environ.get('PATH', '')}",
        "PYTHONPATH": f"{SDV2_LOCAL}:{repo_root}",
        "SPARSEDRIVEV2_ROOT": SDV2_LOCAL,
        "OPENSCENE_DATA_ROOT": DATA_ROOT,
        "NUPLAN_MAPS_ROOT": maps_root,
        "NAVSIM_EXP_ROOT": out_dir,
        "PYTHONUNBUFFERED": "1",
        "HYDRA_FULL_ERROR": "1",
    }
    _run_cmd(
        [
            eval_python,
            os.path.join(repo_root, "navsim_agent", "score_epdms_two_stage.py"),
            "--agent",
            f"traj:{trajectory_local}",
            "--split",
            config["split"],
            "--metric-cache",
            cache_local,
            "--worker",
            worker,
            "--output-dir",
            out_dir,
            "--run-tag",
            run_tag,
            "--require-epdms",
            "--expected-tokens",
            str(expected_tokens),
        ],
        tag="score-only",
        cwd=repo_root,
        env=score_env,
    )
    results_s3 = config["results_s3_output"].rstrip("/")
    for path in sorted(glob.glob(os.path.join(out_dir, "**", "*"), recursive=True)):
        if os.path.isfile(path) and not path.endswith(".pkl"):
            relative = os.path.relpath(path, out_dir)
            _put_file_nonchunked(
                s3, path, *_parse_s3_uri(f"{results_s3}/{relative}")
            )
    logger.info("[score-only] results at %s", results_s3)


def _run_navsim_eval(config, gpu_idx=None, inference_started=None):
    """One eval (staging + inference + scoring). Concurrency-safe: shared
    resources are staged via ``_once`` into URI-keyed directories, so
    several evals may run in threads (``eval_multi_entrypoint_fn``), one
    GPU each (``gpu_idx`` -> CUDA_VISIBLE_DEVICES for the inference
    subprocess). ``inference_started`` (threading.Event) is set right
    before the inference subprocess launches (keepalive handoff)."""
    import glob
    import threading

    config = _with_sd15_eval_defaults(config)
    repo_root = os.path.abspath(os.path.dirname(__file__))
    mode = config["mode"]
    assert mode in ("pdm_v1", "epdms_two_stage"), mode
    split = config["split"]
    run_tag = config["run_tag"]
    geometry_source = config.get("geometry_source", "native")
    if geometry_source not in ("native", *EXTERNAL_GEOMETRY_SOURCES):
        raise ValueError(f"unsupported geometry_source {geometry_source!r}")
    if geometry_source == OCCURRENCE_GEOMETRY_SOURCE and mode != "pdm_v1":
        raise ValueError("emperror_occurrence is only wired for pdm_v1 inference")
    reset_perception_each_frame = config.get(
        "reset_perception_each_frame", False
    )
    if not isinstance(reset_perception_each_frame, bool):
        raise TypeError("reset_perception_each_frame must be boolean")
    if reset_perception_each_frame and mode != "pdm_v1":
        raise ValueError("per-frame perception reset is only wired for pdm_v1")
    geometry_values = (
        config.get("geometry_s3"),
        config.get("geometry_sha256"),
        config.get("geometry_producer_checkpoint_sha256"),
        config.get("geometry_teacher_config_sha256"),
        config.get("geometry_teacher_checkpoint_sha256"),
    )
    if geometry_source == OCCURRENCE_GEOMETRY_SOURCE:
        if not all(geometry_values[:2]):
            raise ValueError(
                "geometry artifact path and SHA256 are required for "
                "emperror_occurrence"
            )
        if any(geometry_values[2:]):
            raise ValueError(
                "emperror_occurrence provenance comes from the immutable artifact"
            )
    elif geometry_source in GENERATED_GEOMETRY_SOURCES:
        if not all(geometry_values):
            raise ValueError(
                "geometry artifact, producer, and teacher SHA256 values "
                "are required for generated geometry sources"
            )
    elif any(geometry_values):
        raise ValueError(
            "geometry artifact settings require a generated geometry source"
        )
    ego_query_only = config.get("ego_query_only", False)
    if not isinstance(ego_query_only, bool):
        raise TypeError("ego_query_only must be boolean")
    if geometry_source in FILE_GEOMETRY_SOURCES and not ego_query_only:
        raise ValueError("emperror geometry requires ego_query_only=true")

    contract_sha_key = "infos_sha256" if mode == "pdm_v1" else "frames_sha256"
    contract_values = (
        config.get(contract_sha_key),
        config.get("expected_scenarios"),
        config.get("expected_replay_calls"),
    )
    has_contract = all(value is not None for value in contract_values)
    if any(value is not None for value in contract_values) and not has_contract:
        raise ValueError("planner input/count contract fields must be provided together")
    if geometry_source != "native" and not has_contract:
        raise ValueError("external geometry evaluation requires a pinned input contract")
    if geometry_source != "native" and not all(
        config.get(key) for key in ("config_sha256", "checkpoint_sha256")
    ):
        raise ValueError(
            "external geometry evaluation requires config/checkpoint SHA256"
        )
    if has_contract:
        input_sha256, expected_scenarios, expected_replay_calls = contract_values
        if not isinstance(input_sha256, str) or len(input_sha256) != 64:
            raise ValueError(f"{contract_sha_key} must contain 64 characters")
        for name, value in (
            ("expected_scenarios", expected_scenarios),
            ("expected_replay_calls", expected_replay_calls),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if expected_replay_calls != 4 * expected_scenarios:
            raise ValueError(
                "expected_replay_calls must equal four times expected_scenarios"
            )
    for key in (
        "config_sha256",
        "checkpoint_sha256",
        "frames_sha256",
        "geometry_sha256",
        "geometry_producer_checkpoint_sha256",
        "geometry_teacher_config_sha256",
        "geometry_teacher_checkpoint_sha256",
    ):
        value = config.get(key)
        if value is not None and (not isinstance(value, str) or len(value) != 64):
            raise ValueError(f"{key} must contain 64 characters")

    num_shards = int(config.get("num_shards", 1))
    num_gpus = int(config.get("num_gpus", 1))
    if num_shards < 1 or num_shards > num_gpus:
        raise ValueError("num_shards must be between one and num_gpus")
    if gpu_idx is not None and num_shards != 1:
        raise ValueError("multi-eval specs cannot fan out inference shards")

    results_s3 = config["results_s3_output"].rstrip("/")
    planner_config = os.path.join(repo_root, config["config_file"])
    if config.get("config_sha256") and _sha256_file(planner_config) != config[
        "config_sha256"
    ]:
        raise ValueError("planner config SHA256 mismatch")
    out_dir = f"/tmp/eval_out/{run_tag}"
    os.makedirs(out_dir, exist_ok=True)

    s3 = _s3_client()
    # GPU keepalive is owned by the entrypoint wrappers
    # (_start_persistent_keepalive) for the whole workload lifetime;
    # inference_started is kept as a coordination signal only.
    if inference_started is None:
        inference_started = threading.Event()

    try:
        # ── 1. SparseDrive inference env (same packed env as training) ─────
        def _stage_sd_env():
            if os.path.exists(os.path.join(ENV_LOCAL, "bin", "python")):
                return
            logger.info("[env] downloading packed sparsedrive env ...")
            _download_file_s3(
                _s3_client(), USER_BUCKET,
                f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz",
                "/tmp/sparsedrive310_env.tar.gz",
            )
            os.makedirs(ENV_LOCAL, exist_ok=True)
            _run_cmd(["tar", "-xzf", "/tmp/sparsedrive310_env.tar.gz",
                      "-C", ENV_LOCAL], tag="env-untar")
            _run_cmd([os.path.join(ENV_LOCAL, "bin", "conda-unpack")],
                     tag="conda-unpack")

        _once("sd_env", _stage_sd_env)
        sd_python = os.path.join(ENV_LOCAL, "bin", "python")

        # ── 2. compile plugin CUDA ops for this node's arch ────────────────
        def _build_ops():
            ops_dir = os.path.join(
                repo_root, "projects", "mmdet3d_plugin", "ops")
            for so in glob.glob(os.path.join(ops_dir, "*.so")):
                os.remove(so)
            cuda_home = _find_cuda_home()
            build_env = {
                **os.environ,
                "CUDA_HOME": cuda_home,
                "PATH": f"{os.path.join(cuda_home, 'bin')}:"
                        f"{os.environ.get('PATH', '')}",
                "FORCE_CUDA": "1",
                "TORCH_CUDA_ARCH_LIST": config.get(
                    "torch_cuda_arch_list", "8.0"),
            }
            _run_cmd([sd_python, "setup.py", "build_ext", "--inplace"],
                     tag="ops-build", cwd=ops_dir, env=build_env)

        _once("ops_build", _build_ops)

        # ── 3. stage model + eval inputs ────────────────────────────────────
        ckpt_local = f"/tmp/eval_ckpt_{run_tag}.pth"
        b, k = _parse_s3_uri(config["checkpoint_s3"])
        _once(f"ckpt:{config['checkpoint_s3']}:{run_tag}",
              lambda: _download_file_s3(_s3_client(), b, k, ckpt_local))
        if config.get("checkpoint_sha256") and _sha256_file(ckpt_local) != config[
            "checkpoint_sha256"
        ]:
            raise ValueError("planner checkpoint SHA256 mismatch")

        # anchors the eval config reads (same artifacts as training)
        for entry in config.get("stage_files", [
            ["data/kmeans/navtrain/kmeans_det_900_navsim.npy",
             "data/kmeans/kmeans_det_900_navsim.npy"],
            ["data/kmeans/navtrain/kmeans_map_100_navsim.npy",
             "data/kmeans/kmeans_map_100_navsim.npy"],
            ["data/kmeans/navtrain/kmeans_motion_6_navsim.npy",
             "data/kmeans/kmeans_motion_6_navsim.npy"],
            ["data/kmeans/navtrain/kmeans_plan_6_navsim.npy",
             "data/kmeans/kmeans_plan_6_navsim.npy"],
        ]):
            expected_sha256 = None
            if isinstance(entry, str):
                s3_rel = repo_rel = entry
            elif isinstance(entry, dict):
                s3_rel = entry["s3_rel"]
                repo_rel = entry["repo_rel"]
                expected_sha256 = entry["sha256"]
                if (
                    not isinstance(expected_sha256, str)
                    or len(expected_sha256) != 64
                ):
                    raise ValueError("stage file sha256 must contain 64 characters")
            else:
                s3_rel, repo_rel = entry
            destination = os.path.join(repo_root, repo_rel)
            _once(
                f"file:{s3_rel}->{repo_rel}",
                lambda s3_rel=s3_rel, repo_rel=repo_rel: _download_file_s3(
                    _s3_client(), USER_BUCKET, f"{NAVSIM_PREFIX}/{s3_rel}",
                    os.path.join(repo_root, repo_rel)))
            if (
                expected_sha256 is not None
                and _sha256_file(destination) != expected_sha256
            ):
                raise ValueError(f"stage file SHA256 mismatch: {repo_rel}")

        if mode == "pdm_v1":
            blobs_root = config.get(
                "blobs_root",
                _uri_dir("/tmp/navsim_blobs", config["frame_tars_s3"]))
        else:
            blobs_root = DATA_ROOT
        if config.get("frame_tars_s3"):
            uri = config["frame_tars_s3"]
            _once(f"tars:{uri}->{blobs_root}",
                  lambda: _stage_tars(_s3_client(), uri, blobs_root))
        for uri, rel in config.get("extra_sync_s3", []):
            _once(f"sync:{uri}->{rel}",
                  lambda uri=uri, rel=rel: _sync_prefix(
                      _s3_client(), uri, os.path.join(DATA_ROOT, rel)))

        # ── 4. inference (SparseDrive env, GPU) ────────────────────────────
        infer_env = {
            **os.environ,
            "PATH": f"{os.path.join(ENV_LOCAL, 'bin')}:"
                    f"{os.environ.get('PATH', '')}",
            "PYTHONPATH": repo_root,
            "PYTHONUNBUFFERED": "1",
        }
        if gpu_idx is not None:
            infer_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        trajs_local = os.path.join(out_dir, f"trajs_{run_tag}.pkl")
        limit_args = (["--limit", str(config["limit"])]
                      if config.get("limit") else [])
        geometry_args = []
        if geometry_source in FILE_GEOMETRY_SOURCES:
            geometry_local = _uri_dir(
                "/tmp/eval_geometry", config["geometry_s3"]
            ) + ".pkl"
            b, k = _parse_s3_uri(config["geometry_s3"])
            _once(
                f"geometry:{config['geometry_s3']}",
                lambda: _download_file_s3(
                    _s3_client(), b, k, geometry_local
                ),
            )
            if _sha256_file(geometry_local) != config["geometry_sha256"]:
                raise ValueError("EMPERROR geometry artifact SHA256 mismatch")
            geometry_args = [
                "--geometry-file", geometry_local,
                "--geometry-sha256", config["geometry_sha256"],
                "--geometry-uri", config["geometry_s3"],
            ]
            if geometry_source in GENERATED_GEOMETRY_SOURCES:
                geometry_args += [
                    "--geometry-producer-checkpoint-sha256",
                    config["geometry_producer_checkpoint_sha256"],
                    "--geometry-teacher-config-sha256",
                    config["geometry_teacher_config_sha256"],
                    "--geometry-teacher-checkpoint-sha256",
                    config["geometry_teacher_checkpoint_sha256"],
                ]
        inference_started.set()
        if mode == "pdm_v1":
            infos_local = _uri_dir("/tmp/eval_infos",
                                   config["infos_s3"]) + ".pkl"
            b, k = _parse_s3_uri(config["infos_s3"])
            _once(f"infos:{config['infos_s3']}",
                  lambda: _download_file_s3(_s3_client(), b, k, infos_local))
            if has_contract and _sha256_file(infos_local) != config["infos_sha256"]:
                raise ValueError("planner infos SHA256 mismatch")
            contract_args = (
                [
                    "--infos-sha256", config["infos_sha256"],
                    "--expected-scenarios", str(config["expected_scenarios"]),
                    "--expected-replay-calls",
                    str(config["expected_replay_calls"]),
                ]
                if has_contract
                else []
            )
            base_cmd = [
                sd_python, "-m", "navsim_agent.run_inference_infos",
                "--config", planner_config,
                "--checkpoint", ckpt_local,
                "--infos", infos_local,
                "--data-root", blobs_root,
                "--geometry-source", geometry_source,
                "--checkpoint-every", "0",
            ] + contract_args + geometry_args + (
                ["--ego-query-only"] if ego_query_only else []
            ) + (
                ["--reset-perception-each-frame"]
                if reset_perception_each_frame
                else []
            ) + (
                ["--checkpoint-sha256", config["checkpoint_sha256"]]
                if config.get("checkpoint_sha256")
                else []
            ) + (
                ["--config-sha256", config["config_sha256"]]
                if config.get("config_sha256")
                else []
            ) + limit_args
        else:
            frames_local = _uri_dir("/tmp/eval_frames",
                                    config["frames_pkl_s3"]) + ".pkl"
            b, k = _parse_s3_uri(config["frames_pkl_s3"])
            _once(f"frames:{config['frames_pkl_s3']}",
                  lambda: _download_file_s3(_s3_client(), b, k, frames_local))
            if has_contract and _sha256_file(frames_local) != config["frames_sha256"]:
                raise ValueError("planner frames SHA256 mismatch")
            remap_from = config.get("path_remap_from",
                                    "/media/applied/navsim")
            base_cmd = [
                sd_python, "-m", "navsim_agent.run_inference_frames",
                "--config", planner_config,
                "--checkpoint", ckpt_local,
                "--frames", frames_local,
                "--path-remap", f"{remap_from}:{DATA_ROOT}",
                "--geometry-source", geometry_source,
                "--checkpoint-every", "0",
            ] + (
                [
                    "--frames-sha256", config["frames_sha256"],
                    "--expected-scenarios", str(config["expected_scenarios"]),
                    "--expected-replay-calls",
                    str(config["expected_replay_calls"]),
                ]
                if has_contract
                else []
            ) + geometry_args + (
                ["--ego-query-only"] if ego_query_only else []
            ) + (
                ["--checkpoint-sha256", config["checkpoint_sha256"]]
                if config.get("checkpoint_sha256")
                else []
            ) + (
                ["--config-sha256", config["config_sha256"]]
                if config.get("config_sha256")
                else []
            ) + limit_args

        if num_shards == 1:
            _run_cmd(base_cmd + ["--output", trajs_local],
                     tag="inference", cwd=repo_root, env=infer_env)
        else:
            shard_errors = {}

            def _one_shard(index):
                env_i = dict(infer_env, CUDA_VISIBLE_DEVICES=str(index))
                try:
                    _run_cmd(
                        base_cmd + [
                            "--output", f"{trajs_local}.shard{index}",
                            "--shard", str(index),
                            "--num-shards", str(num_shards),
                        ],
                        tag=f"inference-shard{index}",
                        cwd=repo_root,
                        env=env_i,
                    )
                except Exception as error:
                    shard_errors[index] = repr(error)

            shard_threads = [
                threading.Thread(target=_one_shard, args=(index,), daemon=True)
                for index in range(num_shards)
            ]
            for thread in shard_threads:
                thread.start()
            for thread in shard_threads:
                thread.join()
            if shard_errors:
                raise RuntimeError(f"inference shards failed: {shard_errors}")

            import pickle as _pickle

            merged = {}
            meta = None
            gate_stats = []
            artifact_stats = []
            replay_stats = []
            for index in range(num_shards):
                with open(f"{trajs_local}.shard{index}", "rb") as handle:
                    payload = _pickle.load(handle)
                if meta is None:
                    meta = {
                        key: value
                        for key, value in payload.items()
                        if key != "trajectories"
                    }
                elif any(
                    payload.get(key) != meta.get(key)
                    for key in (
                        "config_sha256",
                        "checkpoint_sha256",
                        "checkpoint",
                        "infos_sha256",
                        "frames_sha256",
                        "emperror_index_sha256",
                        "expected_scenarios",
                        "expected_replay_calls",
                        "geometry_source",
                        "ego_query_only",
                        "perception_temporal_reset",
                    )
                ):
                    raise RuntimeError("inference shard provenance differs")
                overlap = set(merged) & set(payload["trajectories"])
                if overlap:
                    raise RuntimeError(
                        f"inference shard overlap: {sorted(overlap)[:5]}"
                    )
                merged.update(payload["trajectories"])
                if payload.get("geometry_gate") is not None:
                    gate_stats.append(payload["geometry_gate"])
                if payload.get("geometry_artifact") is not None:
                    artifact_stats.append(payload["geometry_artifact"])
                if payload.get("replay_stats") is not None:
                    replay_stats.append(payload["replay_stats"])

            if gate_stats:
                errors = [
                    value
                    for stats in gate_stats
                    if (value := stats.get("max_reencode_error")) is not None
                ]
                meta["geometry_gate"] = {
                    "calls": sum(stats["calls"] for stats in gate_stats),
                    "max_reencode_error": max(errors) if errors else None,
                    "max_source_delta": max(
                        (
                            stats["max_source_delta"]
                            for stats in gate_stats
                            if stats.get("max_source_delta") is not None
                        ),
                        default=None,
                    ),
                    "ego_query_only": all(
                        stats["ego_query_only"] for stats in gate_stats
                    ),
                    "perception_temporal_reset": (
                        "every_frame"
                        if reset_perception_each_frame
                        else "sequence_boundary"
                    ),
                }
            if artifact_stats:
                if len(artifact_stats) != num_shards or len(
                    {stats["sha256"] for stats in artifact_stats}
                ) != 1:
                    raise RuntimeError("inference shards used different geometry")
                used_tokens = {
                    token
                    for stats in artifact_stats
                    for token in stats.get("tokens_used", ())
                }
                meta["geometry_artifact"]["calls"] = sum(
                    stats["calls"] for stats in artifact_stats
                )
                meta["geometry_artifact"]["unique_tokens_used"] = len(
                    used_tokens
                )
                meta["geometry_artifact"]["tokens_used"] = sorted(used_tokens)
                meta["geometry_artifact"]["uri"] = config["geometry_s3"]
            scenarios = sum(stats["scenarios"] for stats in replay_stats)
            replay_calls = sum(stats["replay_calls"] for stats in replay_stats)
            if (
                len(replay_stats) != num_shards
                or scenarios != len(merged)
                or replay_calls != 4 * len(merged)
            ):
                raise RuntimeError("planner replay/trajectory count mismatch")
            if has_contract and (
                len(merged) != config["expected_scenarios"]
                or replay_calls != config["expected_replay_calls"]
            ):
                raise RuntimeError("planner replay count differs from pinned contract")
            meta["replay_stats"] = {
                "scenarios": scenarios,
                "replay_calls": replay_calls,
            }
            if geometry_source != "native" and (
                len(gate_stats) != num_shards
                or meta["geometry_gate"]["calls"] != replay_calls
            ):
                raise RuntimeError("planner geometry gate count mismatch")
            if geometry_source in FILE_GEOMETRY_SOURCES and (
                meta["geometry_artifact"]["calls"] != replay_calls
            ):
                raise RuntimeError("EMPERROR geometry producer count mismatch")
            with open(trajs_local, "wb") as handle:
                _pickle.dump(dict(meta, trajectories=merged), handle)
            logger.info(
                "[inference] merged %d shards -> %d trajectories",
                num_shards,
                len(merged),
            )
        _put_file_nonchunked(
            s3, trajs_local, *_parse_s3_uri(
                f"{results_s3}/trajs_{run_tag}.pkl"))
        logger.info("[trajs] uploaded")

        # ── 5. scoring env (packed conda `lilypad` env + SDV2 snapshot) ────
        def _stage_eval_env():
            if os.path.exists(os.path.join(EVAL_ENV_LOCAL, "bin", "python")):
                return
            b, k = _parse_s3_uri(config["eval_env_s3"])
            _download_file_s3(_s3_client(), b, k, "/tmp/navsim_eval_env.tar.gz")
            os.makedirs(EVAL_ENV_LOCAL, exist_ok=True)
            _run_cmd(["tar", "-xzf", "/tmp/navsim_eval_env.tar.gz",
                      "-C", EVAL_ENV_LOCAL], tag="eval-env-untar")
            _run_cmd([os.path.join(EVAL_ENV_LOCAL, "bin", "conda-unpack")],
                     tag="eval-env-unpack")

        def _stage_sdv2():
            if os.path.exists(os.path.join(SDV2_LOCAL, "navsim")):
                return
            b, k = _parse_s3_uri(config["sdv2_snapshot_s3"])
            _download_file_s3(_s3_client(), b, k, "/tmp/sdv2.tar.gz")
            os.makedirs(SDV2_LOCAL, exist_ok=True)
            _run_cmd(["tar", "-xzf", "/tmp/sdv2.tar.gz", "-C", SDV2_LOCAL],
                     tag="sdv2-untar")

        _once("eval_env", _stage_eval_env)
        _once("sdv2", _stage_sdv2)
        eval_python = os.path.join(EVAL_ENV_LOCAL, "bin", "python")
        _run_cmd([eval_python, "-c",
                  "import sys; sys.path.insert(0, '" + SDV2_LOCAL + "'); "
                  "import navsim, nuplan, hydra; print('eval env OK')"],
                 tag="eval-env-sanity")

        # ── 6. stage scoring data: metric cache + navsim logs (+ maps) ─────
        cache_local = _uri_dir("/tmp/metric_cache", config["metric_cache_s3"])

        def _stage_cache():
            _sync_prefix(_s3_client(), config["metric_cache_s3"], cache_local)
            _rewrite_cache_metadata(cache_local)

        _once(f"cache:{config['metric_cache_s3']}", _stage_cache)
        logs_dirname = os.path.basename(
            _parse_s3_uri(config["navsim_logs_s3"])[1])
        _once(f"logs:{config['navsim_logs_s3']}",
              lambda: _sync_prefix(
                  _s3_client(), config["navsim_logs_s3"],
                  os.path.join(DATA_ROOT, "navsim_logs", logs_dirname)))
        maps_root = os.path.join(DATA_ROOT, "maps")
        if config.get("maps_s3"):
            _once(f"maps:{config['maps_s3']}",
                  lambda: _sync_prefix(_s3_client(), config["maps_s3"],
                                       maps_root))
        os.makedirs(maps_root, exist_ok=True)

        # ── 7. score ────────────────────────────────────────────────────────
        # strip RAY_* so the scorer's own ray runtime (epdms
        # ray_distributed_no_torch worker) starts fresh instead of trying
        # to join the Lilypad driver's cluster
        score_base_env = {k: v for k, v in os.environ.items()
                          if not k.startswith("RAY_")}
        score_env = {
            **score_base_env,
            "PATH": f"{os.path.join(EVAL_ENV_LOCAL, 'bin')}:"
                    f"{os.environ.get('PATH', '')}",
            "PYTHONPATH": f"{SDV2_LOCAL}:{repo_root}",
            "SPARSEDRIVEV2_ROOT": SDV2_LOCAL,
            "OPENSCENE_DATA_ROOT": DATA_ROOT,
            "NUPLAN_MAPS_ROOT": maps_root,
            "NAVSIM_EXP_ROOT": out_dir,
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
        }
        if mode == "pdm_v1":
            _run_cmd(
                [eval_python,
                 os.path.join(repo_root, "navsim_agent", "score_pdm_v1.py"),
                 "--agent", f"traj:{trajs_local}",
                 "--split", split,
                 "--metric-cache", cache_local,
                 "--output-dir", out_dir,
                 "--run-tag", run_tag]
                + (["--require-complete"] if has_contract else [])
                + (
                    ["--expected-tokens", str(config["expected_scenarios"])]
                    if has_contract
                    else []
                ),
                tag="score", cwd=repo_root, env=score_env,
            )
        else:
            require_epdms = config.get(
                "require_epdms", split == "navhard_two_stage"
            )
            _run_cmd(
                [eval_python,
                 os.path.join(repo_root, "navsim_agent",
                              "score_epdms_two_stage.py"),
                 "--agent", f"traj:{trajs_local}",
                 "--split", split,
                 "--metric-cache", cache_local,
                 "--worker", config.get("epdms_worker",
                                        "ray_distributed_no_torch"),
                 "--output-dir", out_dir,
                 "--run-tag", run_tag]
                + (["--require-epdms"] if require_epdms else [])
                + (
                    ["--expected-tokens", str(config["expected_scenarios"])]
                    if require_epdms and has_contract
                    else []
                ),
                tag="score", cwd=repo_root, env=score_env,
            )

        # ── 8. upload results ───────────────────────────────────────────────
        for f in sorted(glob.glob(os.path.join(out_dir, "**", "*"),
                                  recursive=True)):
            if os.path.isfile(f) and not f.endswith(".pkl"):
                rel = os.path.relpath(f, out_dir)
                _put_file_nonchunked(
                    s3, f, *_parse_s3_uri(f"{results_s3}/{rel}"))
        logger.info("[done] results at %s", results_s3)
    finally:
        inference_started.set()


def eval_multi_entrypoint_fn(config: dict[Any, Any]) -> None:
    """Run several NAVSIM evals in parallel inside ONE full-node workload
    (fractional-node preemptible jobs get systematically reaped; full
    8-GPU workloads survive — mirrors lilypad_entrypoint_combined's
    parallel eval batch).

    config keys:
        evals    : list of per-eval specs (same schema as
                   eval_entrypoint_fn's entrypoint_fn_config)
        shared   : dict merged into every spec (eval_env_s3,
                   sdv2_snapshot_s3, maps_s3, ...); spec keys win
        num_gpus : node size to reserve (default 8); evals run one GPU
                   each (eval i -> CUDA_VISIBLE_DEVICES=i)
    """
    import ray

    num_gpus = config.get("num_gpus", 8)
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, max_retries=0,
                runtime_env={"env_vars": cred_env})
    def _eval(cfg, n):
        # persistent keepalive on EVERY GPU, started before any staging and
        # stopped only at workload exit (idle-resource reaper mitigation)
        stop = _start_persistent_keepalive(n)
        try:
            _run_navsim_eval_multi(cfg)
        finally:
            stop.set()

    ray.get(_eval.remote(config, num_gpus))


def _run_navsim_eval_multi(config):
    import threading

    shared = config.get("shared", {})
    evals = [dict(shared, **spec) for spec in config["evals"]]
    assert evals, "no eval specs"
    assert len(evals) <= config.get("num_gpus", 8), "more evals than GPUs"

    errors = {}

    def _one(i, spec):
        tag = spec.get("run_tag", f"eval{i}")
        try:
            logger.info("[multi] starting eval %d (%s) on GPU %d",
                        i, tag, i)
            _run_navsim_eval(spec, gpu_idx=i)
            logger.info("[multi] eval %d (%s) DONE", i, tag)
        except Exception as e:
            logger.exception("[multi] eval %d (%s) FAILED", i, tag)
            errors[tag] = repr(e)

    threads = [
        threading.Thread(target=_one, args=(i, spec), daemon=True)
        for i, spec in enumerate(evals)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"{len(errors)}/{len(evals)} evals failed: "
                           f"{errors}")
