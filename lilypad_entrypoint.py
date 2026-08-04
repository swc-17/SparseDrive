"""Lilypad entrypoint for SparseDrive data preparation on OCI."""

import logging
import os
import subprocess
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

# OCI S3 — research-datasets-chicago lives in the Chicago region
OCI_ENDPOINT   = "https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com"
OCI_REGION     = "us-chicago-1"
SHARED_BUCKET  = "research-datasets-chicago"
USER_BUCKET    = "research-datasets-chicago"
USER_PREFIX    = "users/tejan/sparsedrive"

# Local working dirs inside the Lilypad worker
NUSCENES_LOCAL = "/tmp/nuscenes"
OUTPUT_LOCAL   = "/tmp/sparsedrive_data"


def _s3_client(endpoint_url=None, region=None):
    import boto3, botocore.config
    # Match entrypoint_common._get_s3_client from SparseDriveV2 exactly.
    # The Chicago OCI key (AWS_ACCESS_KEY_ID) is cross-region and works with
    # both Chicago and Phoenix endpoints of the same tenancy.
    endpoint = endpoint_url or os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3") or OCI_ENDPOINT
    region   = region or os.environ.get("AWS_DEFAULT_REGION", OCI_REGION)
    cfg = botocore.config.Config(
        connect_timeout=30,
        read_timeout=300,
        signature_version="s3v4",
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    return boto3.client("s3", endpoint_url=endpoint, region_name=region, config=cfg)


def _sync_s3_prefix(s3, src_bucket, src_prefix, local_dir):
    """Download all objects under src_prefix to local_dir, skipping existing."""
    import botocore
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=src_bucket, Prefix=src_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel  = key[len(src_prefix):].lstrip("/")
            if not rel:
                continue
            local_path = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if os.path.exists(local_path) and os.path.getsize(local_path) == obj["Size"]:
                continue  # already downloaded
            s3.download_file(src_bucket, key, local_path)
            count += 1
    logger.info("Downloaded %d files from s3://%s/%s → %s", count, src_bucket, src_prefix, local_dir)


def _upload_dir(s3, local_dir, dst_bucket, dst_prefix):
    """Upload all files from local_dir to dst_bucket/dst_prefix, with retry."""
    count = skipped = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            full_path  = os.path.join(root, fname)
            rel        = os.path.relpath(full_path, local_dir)
            key        = f"{dst_prefix.rstrip('/')}/{rel}"
            local_size = os.path.getsize(full_path)

            # Skip if already on S3 with same size
            try:
                head = s3.head_object(Bucket=dst_bucket, Key=key)
                if head["ContentLength"] == local_size:
                    skipped += 1
                    continue
            except Exception:
                pass

            with open(full_path, "rb") as f:
                body = f.read()

            last_exc = None
            for attempt in range(8):
                try:
                    s3.put_object(Bucket=dst_bucket, Key=key, Body=body)
                    last_exc = None
                    break
                except Exception as exc:
                    msg = str(exc)
                    if "ConcurrentObjectUpdate" in msg or "FAILED_PRECONDITION" in msg:
                        wait = 2 ** attempt
                        logger.warning("ConcurrentObjectUpdate on %s, retry %d/8 in %ds", key, attempt + 1, wait)
                        time.sleep(wait)
                        last_exc = exc
                    else:
                        raise

            if last_exc is not None:
                raise RuntimeError(
                    f"Failed to upload {key} after 8 retries (ConcurrentObjectUpdate)."
                ) from last_exc

            count += 1

    logger.info("Uploaded %d files to s3://%s/%s (%d skipped)", count, dst_bucket, dst_prefix, skipped)


def _download_file_s3(s3, bucket: str, key: str, local_path: str) -> None:
    """Download a single S3 object (same pattern as SparseDriveV2 entrypoint_common)."""
    if os.path.exists(local_path):
        logger.info("Already present, skipping: %s", local_path)
        return
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    logger.info("Downloading s3://%s/%s → %s", bucket, key, local_path)
    s3.download_file(bucket, key, local_path)


def create_data_entrypoint_fn(config: dict[Any, Any]) -> None:
    """
    Download nuScenes archives from S3 (metadata + canbus + maps only — no sensor
    blobs needed since the converter only stores file paths as strings), run
    nuscenes_converter.py, and upload the resulting pkl files back to S3.

    config keys:
        nuscenes_s3_uri  : S3 URI of the nuScenes archives directory
                           default: s3://research-datasets-chicago/users/tejan/nuScenes
        output_s3_uri    : S3 URI to write infos pkl files
                           default: s3://research-datasets-chicago/users/tejan/sparsedrive/data/infos
        versions         : NuScenes versions to process  (default: ["v1.0-mini", "v1.0-trainval"])
        skip_download    : skip S3 download if archives already on disk (default: False)
    """
    nuscenes_uri  = config.get("nuscenes_s3_uri",
                               "s3://research-datasets-chicago/users/tejan/nuScenes")
    output_uri    = config.get("output_s3_uri",
                               "s3://research-datasets-chicago/users/tejan/sparsedrive/data/infos")
    versions      = config.get("versions", ["v1.0-mini", "v1.0-trainval"])
    skip_download = config.get("skip_download", False)

    src_bucket = nuscenes_uri.replace("s3://", "").split("/")[0]
    src_prefix = "/".join(nuscenes_uri.replace("s3://", "").split("/")[1:])
    dst_bucket = output_uri.replace("s3://", "").split("/")[0]
    dst_prefix = "/".join(output_uri.replace("s3://", "").split("/")[1:])

    # Install runtime deps not in requirements_lilypad.txt (avoid build-time conflicts)
    logger.info("Installing runtime pip packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "nuscenes-devkit==1.1.10",
         "mmcv==1.7.1",       # lite (no CUDA ops) — sufficient for converter
         "mmdet==2.28.2",
         "yapf==0.33.0",
         "-q", "--no-build-isolation"],
        check=True,
    )

    # research-datasets-chicago is in OCI Phoenix — use Phoenix endpoint explicitly
    s3 = _s3_client(
        endpoint_url="https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com",
        region="us-chicago-1",
    )
    os.makedirs(NUSCENES_LOCAL, exist_ok=True)

    # ── Download & extract the three lightweight archives ──────────────────
    # Sensor blobs (v1.0-trainval0N_blobs.tgz, ~300 GB) are intentionally skipped:
    # the converter only stores file paths in the pkl, it never opens sensor files.
    archives = [
        ("v1.0-trainval_meta.tgz",         "tgz"),  # ~461 MB — metadata JSON tables
        ("can_bus.zip",                     "zip"),  # ~780 MB — CAN bus messages
        ("nuScenes-map-expansion-v1.3.zip", "zip"),  # ~398 MB — HD map files
    ]

    if not skip_download:
        for archive_name, fmt in archives:
            key       = f"{src_prefix.rstrip('/')}/{archive_name}"
            local_arc = f"/tmp/{archive_name}"
            _download_file_s3(s3, src_bucket, key, local_arc)
            logger.info("Extracting %s → %s", archive_name, NUSCENES_LOCAL)
            if fmt == "tgz":
                subprocess.run(["tar", "-xzf", local_arc, "-C", NUSCENES_LOCAL], check=True)
            else:
                subprocess.run(["unzip", "-q", "-o", local_arc, "-d", NUSCENES_LOCAL], check=True)
            os.remove(local_arc)
    else:
        logger.info("Skipping download (skip_download=True)")

    # ── Run nuscenes_converter.py ──────────────────────────────────────────
    os.makedirs(os.path.join(OUTPUT_LOCAL, "infos"), exist_ok=True)

    # Anchor PYTHONPATH to the repo root (dirname of this file) so that
    # `projects.mmdet3d_plugin` is importable regardless of the Lilypad CWD.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{existing_pypath}" if existing_pypath else repo_root

    # Use absolute path to the converter to avoid CWD ambiguity.
    converter_script = os.path.join(repo_root, "tools", "data_converter", "nuscenes_converter.py")

    for version in versions:
        logger.info("Running converter for version=%s", version)
        subprocess.run(
            [
                sys.executable,
                converter_script,
                "nuscenes",
                "--root-path", NUSCENES_LOCAL,
                "--canbus",    NUSCENES_LOCAL,
                "--out-dir",   os.path.join(OUTPUT_LOCAL, "infos"),
                "--extra-tag", "nuscenes",
                "--version",   version,
            ],
            check=True,
            env=env,
            cwd=repo_root,   # ensure relative imports inside the converter resolve correctly
        )
        logger.info("Converter done for %s", version)

    # ── Upload pkl files to S3 ─────────────────────────────────────────────
    logger.info("Uploading infos → s3://%s/%s", dst_bucket, dst_prefix)
    _upload_dir(s3, os.path.join(OUTPUT_LOCAL, "infos"), dst_bucket, dst_prefix)
    logger.info("Done. Infos at s3://%s/%s", dst_bucket, dst_prefix)


# ══════════════════════════════════════════════════════════════════════════
# Declutter-matrix training (stage1/stage2 variants on a100.8 nodes)
# ══════════════════════════════════════════════════════════════════════════

DECLUTTER_PREFIX = "users/tejan/sparsedrive/declutter"
ENV_LOCAL        = "/tmp/sd_env"
DATA_TAR_LOCAL   = "/tmp/sd_nuscenes.tar"


def _find_cuda_home():
    """Find CUDA_HOME that has nvcc (PATH first, then pip nvidia/cuda_nvcc)."""
    import shutil, site
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        c = os.path.join(sp, "nvidia", "cuda_nvcc")
        if os.path.isfile(os.path.join(c, "bin", "nvcc")):
            return c
    return "/usr/local/cuda"


def _run_cmd(cmd, tag, cwd=None, env=None):
    from collections import deque

    logger.info("[%s] $ %s", tag, " ".join(map(str, cmd)))
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    tail = deque(maxlen=200)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    returncode = process.wait()
    if returncode != 0:
        detail = "\n".join(tail)
        raise RuntimeError(
            f"[{tag}] failed with returncode={returncode}\n"
            f"--- subprocess tail ---\n{detail}"
        )


def _put_file_nonchunked(client, path, bucket, key):
    """Upload a file without AWS-chunked transfer encoding (OCI-compatible).

    put_object/upload_part with a bytes body + explicit ContentLength never
    chunk-encodes; boto3's transfer manager (upload_file) does and OCI
    rejects it with NotImplemented.
    """
    size = os.path.getsize(path)
    part_size = 512 * 1024 * 1024
    if size <= part_size:
        with open(path, "rb") as f:
            body = f.read()
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentLength=size)
        return
    upload_id = client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
    try:
        parts = []
        num = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(part_size)
                if not chunk:
                    break
                num += 1
                r = client.upload_part(Bucket=bucket, Key=key, UploadId=upload_id,
                                       PartNumber=num, Body=chunk,
                                       ContentLength=len(chunk))
                parts.append({"ETag": r["ETag"], "PartNumber": num})
        client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id,
                                         MultipartUpload={"Parts": parts})
    except BaseException:
        client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise


def train_entrypoint_fn(config: dict[Any, Any]) -> None:
    """Train one declutter-matrix run on an 8-GPU node.

    config keys:
        config_file   : repo-relative mmcv config, e.g.
                        "projects/configs/declutter/stage2_v4_lean.py"
        seed          : int random seed (also picks the stage1 ckpt for stage2)
        run_name      : wandb run name + S3 work_dir leaf,
                        e.g. "stage2_v4_lean_seed0"
        load_from_s3  : S3 URI of the init checkpoint (stage2 runs; null for stage1)
        num_gpus      : default 8
    """
    import ray
    num_gpus = config.get("num_gpus", 8)
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, runtime_env={"env_vars": cred_env})
    def _train(cfg, n):
        _run_declutter_training(cfg, n)

    ray.get(_train.remote(config, num_gpus))


def _run_declutter_training(config, num_gpus):
    import glob
    import shutil
    import tarfile
    import threading

    repo_root = os.path.abspath(os.path.dirname(__file__))
    config_file = config["config_file"]
    seed = int(config.get("seed", 0))
    run_name = config["run_name"]
    load_from_s3 = config.get("load_from_s3")
    work_dir = f"/tmp/work_dirs/{run_name}"
    work_s3_prefix = f"{DECLUTTER_PREFIX}/work_dirs_v2/{run_name}"

    s3 = _s3_client()

    # ── GPU keepalive so Lilypad's idle-GPU detector doesn't kill setup ────
    stop_keepalive = threading.Event()
    stop_sync = threading.Event()  # checkpoint-sync runs through training

    def _gpu_keepalive():
        try:
            import torch as _t, time as _time
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
        # ── 1. conda env ────────────────────────────────────────────────────
        if not os.path.exists(os.path.join(ENV_LOCAL, "bin", "python")):
            logger.info("[env] downloading packed conda env ...")
            _download_file_s3(
                s3, USER_BUCKET, f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz",
                "/tmp/sparsedrive310_env.tar.gz",
            )
            os.makedirs(ENV_LOCAL, exist_ok=True)
            _run_cmd(["tar", "-xzf", "/tmp/sparsedrive310_env.tar.gz", "-C", ENV_LOCAL],
                     tag="env-untar")
            _run_cmd([os.path.join(ENV_LOCAL, "bin", "conda-unpack")], tag="conda-unpack")
        env_python = os.path.join(ENV_LOCAL, "bin", "python")
        # Fail fast if the packed env is unusable — clearer than a rank crash.
        _run_cmd([env_python, "-c",
                  "import torch, mmdet, mmcv, mmseg, flash_attn; "
                  "print('env sanity OK', torch.__version__)"],
                 tag="env-sanity")
        logger.info("[env] ready: %s", env_python)

        # ── 2. data ────────────────────────────────────────────────────────
        nusc_dir = os.path.join(repo_root, "data", "nuscenes")
        if not os.path.isdir(os.path.join(nusc_dir, "samples")):
            logger.info("[data] downloading nuScenes keyframes tar ...")
            _download_file_s3(
                s3, USER_BUCKET, f"{DECLUTTER_PREFIX}/data/sd_nuscenes.tar",
                DATA_TAR_LOCAL,
            )
            os.makedirs(nusc_dir, exist_ok=True)
            _run_cmd(["tar", "-xf", DATA_TAR_LOCAL, "-C", nusc_dir], tag="data-untar")
            os.remove(DATA_TAR_LOCAL)
        for rel in ["data/infos/nuscenes_infos_train_withmap.pkl",
                    "data/infos/nuscenes_infos_val_withmap.pkl",
                    "data/kmeans/kmeans_det_900.npy",
                    "data/kmeans/kmeans_map_100.npy",
                    "data/kmeans/kmeans_motion_6.npy",
                    "data/kmeans/kmeans_plan_6.npy",
                    "ckpt/resnet50-19c8e357.pth"]:
            _download_file_s3(s3, USER_BUCKET, f"{DECLUTTER_PREFIX}/{rel}",
                              os.path.join(repo_root, rel))

        # ── 3. compile plugin CUDA ops for this node's arch ────────────────
        ops_dir = os.path.join(repo_root, "projects", "mmdet3d_plugin", "ops")
        for so in glob.glob(os.path.join(ops_dir, "*.so")):
            os.remove(so)  # never trust shipped .so (built for another arch)
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
        built = glob.glob(os.path.join(ops_dir, "*.so"))
        if not built:
            raise RuntimeError("[ops-build] no .so produced — check nvcc/arch")
        logger.info("[ops-build] built: %s", built)
        # import check with the actual training interpreter
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
                time.sleep(600)
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
            # env bin first: conda-pack rewrites console-script shebangs to
            # `/usr/bin/env python`, which otherwise resolves to the driver
            # virtualenv (has torch but not mmdet) — see canary rxlx44 failure.
            "PATH": f"{os.path.join(ENV_LOCAL, 'bin')}:{os.environ.get('PATH', '')}",
            "PYTHONPATH": repo_root,
            "WANDB_NAME": run_name,
            "PYTHONUNBUFFERED": "1",
        }
        stop_keepalive.set()  # release GPUs for torchrun
        time.sleep(5)
        _run_cmd(
            # launch via env python -m so sys.executable (and therefore all
            # spawned ranks) is guaranteed to be the packed env's interpreter
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


# ══════════════════════════════════════════════════════════════════════════
# Declutter validation worker: poll S3 for new checkpoints, evaluate on the
# full nuScenes val split (devkit det eval requires the complete split), and
# log metrics to wandb (runs named val_<train_run>, step = training iter)
# ══════════════════════════════════════════════════════════════════════════

# run-name prefix -> (config path, planning eval on/off)
_VAL_CFG_MAP = [
    ("stage1_temporal",        ("projects/configs/declutter/stage1_temporal.py",        False)),
    ("stage1_singleframe",     ("projects/configs/declutter/stage1_singleframe.py",     False)),
    ("stage2_v1_baseline",     ("projects/configs/declutter/stage2_v1_baseline.py",     True)),
    ("stage2_v2_nomotion",     ("projects/configs/declutter/stage2_v2_nomotion.py",     True)),
    ("stage2_v3_singleframe",  ("projects/configs/declutter/stage2_v3_singleframe.py",  True)),
    ("stage2_v4_lean",         ("projects/configs/declutter/stage2_v4_lean.py",         True)),
    ("stage2_v5_lean_egostatus", ("projects/configs/declutter/stage2_v5_lean_egostatus.py", True)),
    ("stage2_v6_geoinput",     ("projects/configs/declutter/stage2_v6_geoinput.py",     True)),
    ("stage2_v6c_ctrl",        ("projects/configs/declutter/stage2_v6c_ctrl.py",        True)),
]

_METRIC_RES = {
    "det_mAP":  r"mAP:\s*([0-9.]+)",
    "det_NDS":  r"NDS:\s*([0-9.]+)",
    "det_mAVE": r"mAVE:\s*([0-9.]+)",
    "det_mATE": r"mATE:\s*([0-9.]+)",
}
_PLAN_RES = {
    # dataset evaluate() prints averages as `L2: 0.xxxx` and `obj_box_col: x.xxx%`
    "plan_L2_avg":  r"^L2:\s*([0-9.]+)",
    "plan_col_avg": r"obj_box_col:\s*([0-9.]+)%",
}


def validate_entrypoint_fn(config: dict[Any, Any]) -> None:
    """Long-running validation worker (see module header).

    config keys:
        poll_seconds (default 600), max_hours (default 48),
        num_gpus (default 8) — evals run one-per-GPU in parallel.
        Stop early by putting an object at
        s3://<bucket>/<DECLUTTER_PREFIX>/val_watch/STOP
    """
    import ray
    num_gpus = config.get("num_gpus", 1)
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, runtime_env={"env_vars": cred_env})
    def _watch(cfg, n):
        _run_val_watch(cfg, n)

    ray.get(_watch.remote(config, num_gpus))


def _run_val_watch(config, num_gpus):
    import glob
    import json
    import re
    import threading

    repo_root = os.path.abspath(os.path.dirname(__file__))
    poll_seconds = int(config.get("poll_seconds", 600))
    max_hours = float(config.get("max_hours", 48))
    # each entry is one experiment's dedicated monitor: its own GPU, wandb
    # curve, S3 state, and lifecycle. They share this node's dataset copy
    # (per-monitor nodes ran out of disk: 4 x 70 GB staging on one host).
    runs = config["runs"]  # list of {"run_name": ..., "final_iter": ...}
    eval_baseline = bool(config.get("eval_baseline", False))
    # use_rescore=True evaluates planning with the collision-aware rescore
    # (baseline default) instead of the declutter protocol's rescore-off;
    # run_suffix keeps its wandb curves and S3 state separate.
    use_rescore = bool(config.get("use_rescore", False))
    run_suffix = str(config.get("run_suffix", ""))
    s3 = _s3_client()
    stop_key = f"{DECLUTTER_PREFIX}/val_watch_v2/STOP"

    stop_keepalive = threading.Event()

    def _gpu_keepalive():
        try:
            import torch as _t, time as _time
            if not _t.cuda.is_available():
                return
            x = _t.randn(2048, 2048, device="cuda")
            while not stop_keepalive.is_set():
                with _t.no_grad():
                    x = _t.mm(x, x.fmod(100.0))
                _t.cuda.synchronize()
                _time.sleep(3.0)
        except Exception as e:
            logger.warning("keepalive: %s", e)

    threading.Thread(target=_gpu_keepalive, daemon=True).start()

    try:
        # ── stage env + data (same assets as training) ──────────────────────
        if not os.path.exists(os.path.join(ENV_LOCAL, "bin", "python")):
            _download_file_s3(s3, USER_BUCKET,
                              f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz",
                              "/tmp/sparsedrive310_env.tar.gz")
            os.makedirs(ENV_LOCAL, exist_ok=True)
            _run_cmd(["tar", "-xzf", "/tmp/sparsedrive310_env.tar.gz", "-C", ENV_LOCAL],
                     tag="env-untar")
            _run_cmd([os.path.join(ENV_LOCAL, "bin", "conda-unpack")], tag="conda-unpack")
            os.remove("/tmp/sparsedrive310_env.tar.gz")
        env_python = os.path.join(ENV_LOCAL, "bin", "python")
        _run_cmd([env_python, "-c", "import torch, mmdet, mmcv, wandb; print('env ok')"],
                 tag="env-sanity")

        nusc_dir = os.path.join(repo_root, "data", "nuscenes")
        if not os.path.isdir(os.path.join(nusc_dir, "samples")):
            _download_file_s3(s3, USER_BUCKET,
                              f"{DECLUTTER_PREFIX}/data/sd_nuscenes.tar", DATA_TAR_LOCAL)
            os.makedirs(nusc_dir, exist_ok=True)
            _run_cmd(["tar", "-xf", DATA_TAR_LOCAL, "-C", nusc_dir], tag="data-untar")
            os.remove(DATA_TAR_LOCAL)
        for rel in ["data/infos/nuscenes_infos_val_withmap.pkl",
                    "data/kmeans/kmeans_det_900.npy", "data/kmeans/kmeans_map_100.npy",
                    "data/kmeans/kmeans_motion_6.npy", "data/kmeans/kmeans_plan_6.npy",
                    "ckpt/resnet50-19c8e357.pth"]:
            _download_file_s3(s3, USER_BUCKET, f"{DECLUTTER_PREFIX}/{rel}",
                              os.path.join(repo_root, rel))

        ops_dir = os.path.join(repo_root, "projects", "mmdet3d_plugin", "ops")
        for so in glob.glob(os.path.join(ops_dir, "*.so")):
            os.remove(so)
        cuda_home = _find_cuda_home()
        _run_cmd([env_python, "setup.py", "build_ext", "--inplace"], tag="ops-build",
                 cwd=ops_dir, env={**os.environ, "CUDA_HOME": cuda_home,
                                   "PATH": f"{os.path.join(cuda_home, 'bin')}:{os.environ.get('PATH', '')}",
                                   "FORCE_CUDA": "1",
                                   "TORCH_CUDA_ARCH_LIST": config.get("torch_cuda_arch_list", "8.0")})

        # ── helpers ─────────────────────────────────────────────────────────
        def load_state(run):
            try:
                obj = s3.get_object(Bucket=USER_BUCKET,
                                    Key=f"{DECLUTTER_PREFIX}/val_watch_v2/state_{run}{run_suffix}.json")
                return json.loads(obj["Body"].read())
            except Exception:
                return {}

        def save_state(run, st):
            s3.put_object(Bucket=USER_BUCKET,
                          Key=f"{DECLUTTER_PREFIX}/val_watch_v2/state_{run}{run_suffix}.json",
                          Body=json.dumps(st).encode())

        def cfg_for(run):
            for prefix, v in _VAL_CFG_MAP:
                if run.startswith(prefix):
                    return v
            return None

        def newest_ckpt(run):
            resp = s3.list_objects_v2(Bucket=USER_BUCKET,
                                      Prefix=f"{DECLUTTER_PREFIX}/work_dirs_v2/{run}/")
            best = None
            for o in resp.get("Contents", []):
                m = re.search(r"(?:latest_)?iter_(\d+)\.pth$", o["Key"])
                if m:
                    it = int(m.group(1))
                    if best is None or it > best[0]:
                        best = (it, o["Key"])
            return best

        def stop_requested():
            try:
                s3.head_object(Bucket=USER_BUCKET, Key=stop_key)
                return True
            except Exception:
                return False

        def wandb_log(run, iter_num, metrics):
            payload = json.dumps({"run": run + run_suffix, "iter": iter_num,
                                  "metrics": metrics})
            script = (
                "import json, os, sys, wandb\n"
                "d = json.loads(sys.argv[1])\n"
                "w = wandb.init(project='sparsedrive-declutter',\n"
                "               id='valfull-' + d['run'], name='val_' + d['run'],\n"
                "               resume='allow')\n"
                "w.log({('val/' + k): float(v) for k, v in d['metrics'].items() if v},\n"
                "      step=d['iter'])\n"
                "w.finish()\n"
            )
            _run_cmd([env_python, "-c", script, payload], tag=f"wandb-{run}")

        def run_eval(run, iter_num, key, gpu):
            cfg_path, with_plan = cfg_for(run)
            local_ckpt = f"/tmp/val_{run}.pth"
            _download_file_s3(s3, USER_BUCKET, key, local_ckpt)
            log_path = f"/tmp/val_{run}_{iter_num}.log"
            cmd = [env_python, os.path.join(repo_root, "tools", "test.py"),
                   os.path.join(repo_root, cfg_path), local_ckpt, "--eval", "bbox",
                   "--cfg-options",
                   # isolated work_dir per eval: concurrent same-config evals
                   # otherwise race on ./work_dirs/<config>/results_nusc.json
                   f"work_dir=/tmp/eval_wd_{run}_{iter_num}",
                   "data.workers_per_gpu=4",
                   "evaluation.eval_mode.with_tracking=False",
                   "evaluation.eval_mode.with_motion=False",
                   f"evaluation.eval_mode.with_planning={with_plan}"]
            if with_plan and not use_rescore:
                cmd += ["model.head.motion_plan_head.planning_decoder.use_rescore=False"]
            env = {**os.environ, "PYTHONPATH": repo_root,
                   "CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": "disabled"}
            with open(log_path, "w") as f:
                p = subprocess.Popen(cmd, cwd=repo_root, stdout=f,
                                     stderr=subprocess.STDOUT, env=env)
            return p, log_path, local_ckpt

        def parse_metrics(log_path, with_plan):
            text = open(log_path, errors="replace").read()
            out = {}
            for name, pat in _METRIC_RES.items():
                m = re.search(pat, text)
                if m:
                    out[name] = m.group(1)
            m = re.search(r"mAP_normal\s*[=:]\s*([0-9.]+)", text)
            if m:
                out["map_mAP"] = m.group(1)
            if with_plan:
                for name, pat in _PLAN_RES.items():
                    m = re.search(pat, text, re.MULTILINE)
                    if m:
                        out[name] = m.group(1)
            return out

        # ── one-time baseline eval (released stage2 ckpt) ───────────────────
        if eval_baseline and "BASELINE" not in load_state("baseline_release"):
            logger.info("[val] baseline eval (released stage2 ckpt)")
            base_key = f"{DECLUTTER_PREFIX}/ckpt/sparsedrive_stage2.pth"
            p, log_path, ck = run_eval("stage2_v1_baseline_RELEASED", 0, base_key, 0)
            p.wait()
            metrics = parse_metrics(log_path, True)
            logger.info("[val] baseline metrics: %s", metrics)
            if metrics:
                wandb_log("baseline_release", 0, metrics)
                save_state("baseline_release", {"BASELINE": metrics})
            os.remove(ck)

        # ── one dedicated monitor thread per experiment ─────────────────────
        t0 = time.time()

        def monitor(gpu, run, final_iter):
            while (time.time() - t0) < max_hours * 3600 and not stop_requested():
                state = load_state(run)
                done_iter = state.get(run, -1)
                if done_iter >= final_iter:
                    logger.info("[val] %s: final iter %d evaluated — monitor done",
                                run, final_iter)
                    return
                best = newest_ckpt(run)
                if best and best[0] > done_iter:
                    it, key = best
                    logger.info("[val] %s: evaluating iter %d (gpu %d)", run, it, gpu)
                    p, log_path, ck = run_eval(run, it, key, gpu)
                    p.wait()
                    metrics = parse_metrics(log_path, cfg_for(run)[1])
                    logger.info("[val] %s iter=%d exit=%d metrics=%s",
                                run, it, p.returncode, metrics)
                    if p.returncode == 0 and metrics:
                        wandb_log(run, it, metrics)
                        state[run] = it
                        save_state(run, state)
                    try:
                        os.remove(ck)
                    except OSError:
                        pass
                time.sleep(poll_seconds)
            logger.info("[val] %s: monitor exiting (timeout/stop)", run)

        bad = [r["run_name"] for r in runs if cfg_for(r["run_name"]) is None]
        if bad:
            raise ValueError(f"no eval config mapping for runs: {bad}")
        threads = []
        for gpu, r in enumerate(runs):
            th = threading.Thread(
                target=monitor,
                args=(gpu % num_gpus, r["run_name"], int(r["final_iter"])),
                daemon=False,
            )
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        logger.info("[val] watcher done")
    finally:
        stop_keepalive.set()


# ══════════════════════════════════════════════════════════════════════════
# Geometric-planner training: N (variant, seed) runs per node, one per GPU.
# Stages only the packed env + geometry cache + infos — no nuScenes images.
# ══════════════════════════════════════════════════════════════════════════

def geo_train_entrypoint_fn(config: dict[Any, Any]) -> None:
    """config keys:
        runs: list of {"variant": ..., "seed": ...} (one GPU each, <= num_gpus)
        epochs (default 10), num_gpus (default 8)
    """
    import ray
    num_gpus = config.get("num_gpus", 8)
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, runtime_env={"env_vars": cred_env})
    def _train(cfg, n):
        _run_geo_training(cfg, n)

    ray.get(_train.remote(config, num_gpus))


def _run_geo_training(config, num_gpus):
    import glob

    repo_root = os.path.abspath(os.path.dirname(__file__))
    runs = config["runs"]
    epochs = int(config.get("epochs", 10))
    s3 = _s3_client()

    # ── env ──────────────────────────────────────────────────────────────
    if not os.path.exists(os.path.join(ENV_LOCAL, "bin", "python")):
        _download_file_s3(s3, USER_BUCKET,
                          f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz",
                          "/tmp/sparsedrive310_env.tar.gz")
        os.makedirs(ENV_LOCAL, exist_ok=True)
        _run_cmd(["tar", "-xzf", "/tmp/sparsedrive310_env.tar.gz", "-C", ENV_LOCAL],
                 tag="env-untar")
        _run_cmd([os.path.join(ENV_LOCAL, "bin", "conda-unpack")],
                 tag="conda-unpack")
        os.remove("/tmp/sparsedrive310_env.tar.gz")
    env_python = os.path.join(ENV_LOCAL, "bin", "python")
    _run_cmd([env_python, "-c", "import torch, mmdet, mmcv, wandb; print('env ok')"],
             tag="env-sanity")

    # ── data: geometry cache + infos + kmeans (NO images) ────────────────
    cache_dir = os.path.join(repo_root, "data", "geometry_cache")
    if not os.path.isdir(os.path.join(cache_dir, "train")):
        _download_file_s3(s3, USER_BUCKET,
                          f"{DECLUTTER_PREFIX}/data/geometry_cache.tar.gz",
                          "/tmp/geometry_cache.tar.gz")
        os.makedirs(os.path.join(repo_root, "data"), exist_ok=True)
        _run_cmd(["tar", "-xzf", "/tmp/geometry_cache.tar.gz",
                  "-C", os.path.join(repo_root, "data")], tag="cache-untar")
        os.remove("/tmp/geometry_cache.tar.gz")
    for rel in ["data/infos/nuscenes_infos_train.pkl",
                "data/infos/nuscenes_infos_val.pkl",
                "data/kmeans/kmeans_plan_6.npy",
                "data/kmeans/kmeans_det_900.npy",
                "data/kmeans/kmeans_map_100.npy",
                "data/kmeans/kmeans_motion_6.npy"]:
        _download_file_s3(s3, USER_BUCKET, f"{DECLUTTER_PREFIX}/{rel}",
                          os.path.join(repo_root, rel))

    # ── launch one training per GPU ───────────────────────────────────────
    procs = []
    for gpu, r in enumerate(runs[:num_gpus]):
        name = f"geo_{r['variant']}_seed{r['seed']}"
        log_path = f"/tmp/{name}.log"
        env = {**os.environ, "PYTHONPATH": repo_root,
               "CUDA_VISIBLE_DEVICES": str(gpu % num_gpus),
               "PYTHONUNBUFFERED": "1"}
        cmd = [env_python, os.path.join(repo_root, "geo_planner", "train.py"),
               "--variant", str(r["variant"]), "--seed", str(r["seed"]),
               "--epochs", str(epochs)]
        logger.info("[geo] launching %s on gpu %d", name, gpu)
        with open(log_path, "w") as f:
            p = subprocess.Popen(cmd, cwd=repo_root, stdout=f,
                                 stderr=subprocess.STDOUT, env=env)
        procs.append((name, p, log_path))

    failed = []
    for name, p, log_path in procs:
        rc = p.wait()
        tail = ""
        try:
            with open(log_path, errors="replace") as f:
                tail = "".join(f.readlines()[-15:])
        except OSError:
            pass
        logger.info("[geo] %s exit=%d\n%s", name, rc, tail)
        if rc != 0:
            failed.append(name)
        # upload artifacts (ckpts + metrics + results)
        out_dir = os.path.join(repo_root, "work_dirs", "geo_planner", name)
        if os.path.isdir(out_dir):
            _upload_dir(s3, out_dir, USER_BUCKET,
                        f"{DECLUTTER_PREFIX}/geo_runs/{name}")
    if failed:
        raise RuntimeError(f"[geo] failed runs: {failed}")
    logger.info("[geo] all %d runs done", len(procs))
