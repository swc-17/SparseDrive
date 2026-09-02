"""Lilypad entrypoint: standalone stage-1 perception on Bench2Drive.

Stages the packed sparsedrive310 conda env, extracts only the 6 rgb camera
streams from the mirrored Bench2Drive tars, downloads the stage-1 assets,
compiles the plugin CUDA ops, then runs training with checkpoint sync to S3.
"""

import glob
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

OCI_ENDPOINT = "https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com"
OCI_REGION = "us-chicago-1"
BUCKET = "research-datasets-chicago"
B2D_PREFIX = "bench2drive"                                # raw/, infos/, kmeans/, ckpt/
ENV_KEY = "users/tejan/sparsedrive/declutter/env/sparsedrive310_env.tar.gz"
WORK_PREFIX_DEFAULT = "users/tejan/sparsedrive_b2d/work_dirs"
ENV_LOCAL = "/tmp/sparsedrive310_env"

CAMERA_DIRS = ["rgb_front", "rgb_front_left", "rgb_front_right",
               "rgb_back", "rgb_back_left", "rgb_back_right"]
# HF repo has one truncated filename; the info pkls reference the full name.
RENAME_FIX = {
    "VehicleTurningRoutePedestrian_Town15_Route523_Weathe":
        "VehicleTurningRoutePedestrian_Town15_Route523_Weather2",
}
TRAIN_SAMPLES = 234769  # b2d_infos_train.pkl length (base split)


def _s3_client():
    import boto3
    import botocore.config

    cfg = botocore.config.Config(
        connect_timeout=30,
        read_timeout=300,
        retries={"max_attempts": 10, "mode": "adaptive"},
        signature_version="s3v4",
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    return boto3.client("s3", endpoint_url=OCI_ENDPOINT, region_name=OCI_REGION, config=cfg)


def _run_cmd(cmd, tag, cwd=None, env=None):
    logger.info("[%s] $ %s", tag, " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"[{tag}] failed with returncode={result.returncode}")


def _download_file_s3(s3, bucket, key, local_path):
    if os.path.exists(local_path):
        logger.info("already present: %s", local_path)
        return
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    tmp = local_path + ".part"
    s3.download_file(bucket, key, tmp)
    os.rename(tmp, local_path)
    logger.info("downloaded s3://%s/%s -> %s", bucket, key, local_path)


def _put_file_nonchunked(client, path, bucket, key):
    """put_object/upload_part with bytes bodies — OCI rejects aws-chunked."""
    size = os.path.getsize(path)
    part_size = 512 * 1024 * 1024
    if size <= part_size:
        with open(path, "rb") as f:
            body = f.read()
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentLength=size)
        return
    upload_id = client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
    try:
        parts, num = [], 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(part_size)
                if not chunk:
                    break
                num += 1
                r = client.upload_part(Bucket=bucket, Key=key, UploadId=upload_id,
                                       PartNumber=num, Body=chunk)
                parts.append({"ETag": r["ETag"], "PartNumber": num})
        client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id,
                                         MultipartUpload={"Parts": parts})
    except Exception:
        client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise


def _upload_dir(s3, local_dir, bucket, prefix):
    count = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            full = os.path.join(root, fname)
            if os.path.islink(full):
                continue
            key = f"{prefix.rstrip('/')}/{os.path.relpath(full, local_dir)}"
            _put_file_nonchunked(s3, full, bucket, key)
            count += 1
    logger.info("uploaded %d files to s3://%s/%s", count, bucket, prefix)


def _find_cuda_home():
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    env_nvcc = os.path.join(ENV_LOCAL, "bin", "nvcc")
    if os.path.isfile(env_nvcc):
        return ENV_LOCAL
    return "/usr/local/cuda"


def _stage_env(s3):
    if not os.path.exists(os.path.join(ENV_LOCAL, "bin", "python")):
        logger.info("[env] downloading packed conda env ...")
        _download_file_s3(s3, BUCKET, ENV_KEY, "/tmp/sparsedrive310_env.tar.gz")
        os.makedirs(ENV_LOCAL, exist_ok=True)
        _run_cmd(["tar", "-xzf", "/tmp/sparsedrive310_env.tar.gz", "-C", ENV_LOCAL],
                 tag="env-untar")
        _run_cmd([os.path.join(ENV_LOCAL, "bin", "conda-unpack")], tag="conda-unpack")
        os.remove("/tmp/sparsedrive310_env.tar.gz")
    env_python = os.path.join(ENV_LOCAL, "bin", "python")
    _run_cmd([env_python, "-c",
              "import torch, mmdet, mmcv, wandb; print('env sanity OK', torch.__version__)"],
             tag="env-sanity")
    return env_python


def _extract_one_clip(s3, key, v1_dir, tmp_dir):
    """Download one clip tar and extract only the 6 rgb camera dirs."""
    tar_name = os.path.basename(key)                     # <Clip>.tar.gz
    clip = tar_name[:-len(".tar.gz")]
    clip = RENAME_FIX.get(clip, clip)
    dst = os.path.join(v1_dir, clip)
    if os.path.isdir(os.path.join(dst, "camera", "rgb_front")):
        return clip, False
    local_tar = os.path.join(tmp_dir, tar_name)
    s3.download_file(BUCKET, key, local_tar)
    try:
        pats = [f"*/camera/{c}/*" for c in CAMERA_DIRS]
        _run_cmd(["tar", "-xzf", local_tar, "-C", v1_dir, "--wildcards"] + pats,
                 tag=f"untar:{clip}")
    finally:
        os.remove(local_tar)
    # normalize extracted dir name (handles ./<Clip>/ top dirs and the rename fix)
    for cand in glob.glob(os.path.join(v1_dir, "*")):
        base = os.path.basename(cand)
        fixed = RENAME_FIX.get(base, base)
        if fixed != base:
            os.rename(cand, os.path.join(v1_dir, fixed))
    if not os.path.isdir(os.path.join(dst, "camera", "rgb_front")):
        raise RuntimeError(f"extraction of {tar_name} produced no {dst}/camera/rgb_front")
    return clip, True


def _stage_data(s3, repo_root, num_threads=12):
    v1_dir = os.path.join(repo_root, "data", "bench2drive", "v1")
    tmp_dir = "/tmp/b2d_tars"
    os.makedirs(v1_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{B2D_PREFIX}/raw/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tar.gz"):
                keys.append(obj["Key"])
    logger.info("[data] %d clip tars to stage", len(keys))

    done = 0
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futs = {pool.submit(_extract_one_clip, _s3_client(), k, v1_dir, tmp_dir): k
                for k in keys}
        for fut in as_completed(futs):
            clip, fresh = fut.result()
            done += 1
            if done % 50 == 0:
                logger.info("[data] %d/%d clips staged", done, len(keys))
    logger.info("[data] staging complete: %d clips", done)

    for rel in ["infos/b2d_infos_train.pkl", "infos/b2d_infos_val.pkl",
                "kmeans/kmeans_det_900.npy", "kmeans/kmeans_map_100.npy"]:
        _download_file_s3(s3, BUCKET, f"{B2D_PREFIX}/{rel}",
                          os.path.join(repo_root, "data", rel))
    _download_file_s3(s3, BUCKET, f"{B2D_PREFIX}/ckpt/resnet50-19c8e357.pth",
                      os.path.join(repo_root, "ckpt", "resnet50-19c8e357.pth"))


def _compile_ops(env_python, repo_root, arch_list):
    ops_dir = os.path.join(repo_root, "projects", "mmdet3d_plugin", "ops")
    for so in glob.glob(os.path.join(ops_dir, "*.so")):
        os.remove(so)  # never trust shipped .so (built for another arch)
    cuda_home = _find_cuda_home()
    build_env = {
        **os.environ,
        "CUDA_HOME": cuda_home,
        "CPATH": os.path.join(ENV_LOCAL, "targets", "x86_64-linux", "include"),
        "LIBRARY_PATH": os.path.join(ENV_LOCAL, "targets", "x86_64-linux", "lib")
                        + ":" + os.path.join(ENV_LOCAL, "lib"),
        "PATH": f"{os.path.join(cuda_home, 'bin')}:{os.environ.get('PATH', '')}",
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": arch_list,
    }
    _run_cmd([env_python, "setup.py", "build_ext", "--inplace"],
             tag="ops-build", cwd=ops_dir, env=build_env)
    if not glob.glob(os.path.join(ops_dir, "*.so")):
        raise RuntimeError("[ops-build] no .so produced")
    _run_cmd([env_python, "-c",
              "from projects.mmdet3d_plugin.ops import feature_maps_format; print('ops OK')"],
             tag="ops-sanity", cwd=repo_root,
             env={**os.environ, "PYTHONPATH": repo_root})


def train_entrypoint_fn(config: dict[Any, Any]) -> None:
    """config keys:
        run_name (required), config_file (default stage1), num_gpus (default 8),
        batch_size (default 8), num_epochs (default 20), base_lr (default 5e-4),
        base_total_batch (default 256), seed (default 0),
        load_from_s3 (optional s3:// uri), work_s3_prefix (optional),
        torch_cuda_arch_list (default "8.0"), stage_threads (default 12)
    """
    import ray
    num_gpus = int(config.get("num_gpus", 8))
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, runtime_env={"env_vars": cred_env})
    def _train(cfg, n):
        _run_training(cfg, n)

    ray.get(_train.remote(config, num_gpus))


def _run_training(config, num_gpus):
    repo_root = os.path.abspath(os.path.dirname(__file__))
    run_name = config["run_name"]
    config_file = config.get(
        "config_file", "projects/configs/sparsedrive_b2d_stage1.py"
    )
    batch_size = int(config.get("batch_size", 8))
    num_epochs = int(config.get("num_epochs", 20))
    base_lr = float(config.get("base_lr", 5e-4))
    base_total_batch = int(config.get("base_total_batch", 256))
    seed = int(config.get("seed", 0))
    work_s3_prefix = config.get("work_s3_prefix", f"{WORK_PREFIX_DEFAULT}/{run_name}")
    work_dir = os.path.join(repo_root, "work_dirs", run_name)
    s3 = _s3_client()

    stop_sync = threading.Event()
    try:
        env_python = _stage_env(s3)
        _stage_data(s3, repo_root, num_threads=int(config.get("stage_threads", 12)))
        _compile_ops(env_python, repo_root,
                     config.get("torch_cuda_arch_list", "8.0"))

        # ── scaled schedule ──────────────────────────────────────────────
        total_batch = num_gpus * batch_size
        nipe = TRAIN_SAMPLES // total_batch
        lr = base_lr * total_batch / base_total_batch
        cfg_options = [
            f"optimizer.lr={lr}",
            f"data.samples_per_gpu={batch_size}",
            f"runner.max_iters={nipe * num_epochs}",
            f"checkpoint_config.interval={nipe}",
            "checkpoint_config.max_keep_ckpts=3",
            f"evaluation.interval={nipe * num_epochs}",
        ]
        load_from_s3 = config.get("load_from_s3")
        if load_from_s3:
            local_init = "/tmp/init_ckpt.pth"
            bucket, key = load_from_s3.replace("s3://", "").split("/", 1)
            _download_file_s3(s3, bucket, key, local_init)
            cfg_options.append(f"load_from={local_init}")
        logger.info("[schedule] total_batch=%d iters/epoch=%d lr=%g", total_batch, nipe, lr)

        # ── resume from latest synced ckpt if present ────────────────────
        os.makedirs(work_dir, exist_ok=True)
        resume_args = []
        try:
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{work_s3_prefix}/latest_iter_")
            cands = sorted((o["Key"] for o in resp.get("Contents", [])),
                           key=lambda k: int(k.rsplit("_", 1)[-1].split(".")[0]))
            if cands:
                local_resume = os.path.join(work_dir, "resume.pth")
                _download_file_s3(s3, BUCKET, cands[-1], local_resume)
                resume_args = ["--resume-from", local_resume]
                logger.info("[resume] resuming from %s", cands[-1])
        except Exception as e:
            logger.warning("[resume] check failed (fresh start): %s", e)

        # ── periodic checkpoint sync ─────────────────────────────────────
        def _upload_latest_forever():
            last = None
            while not stop_sync.is_set():
                time.sleep(600)
                try:
                    latest = os.path.join(work_dir, "latest.pth")
                    if os.path.islink(latest):
                        target = os.path.realpath(latest)
                        if target != last and os.path.exists(target):
                            tag = os.path.basename(target).replace(".pth", "")
                            key = f"{work_s3_prefix}/latest_{tag}.pth"
                            logger.info("[ckpt-sync] uploading %s", key)
                            _put_file_nonchunked(_s3_client(), target, BUCKET, key)
                            last = target
                except Exception as e:
                    logger.warning("[ckpt-sync] %s", e)

        threading.Thread(target=_upload_latest_forever, daemon=True).start()

        # ── train ────────────────────────────────────────────────────────
        train_env = {
            **os.environ,
            "PATH": f"{os.path.join(ENV_LOCAL, 'bin')}:{os.environ.get('PATH', '')}",
            "PYTHONPATH": repo_root,
            "WANDB_NAME": run_name,
            "PYTHONUNBUFFERED": "1",
        }
        _run_cmd(
            [env_python, "-m", "torch.distributed.run",
             f"--nproc_per_node={num_gpus}", "--master_port=28651",
             os.path.join(repo_root, "tools", "train_pyfocal.py"),
             os.path.join(repo_root, config_file),
             "--launcher", "pytorch",
             "--seed", str(seed),
             "--work-dir", work_dir,
             "--cfg-options"] + cfg_options
            + resume_args
            + ["--no-validate"],
            tag="train", cwd=repo_root, env=train_env,
        )

        logger.info("[upload] final work_dir -> s3://%s/%s", BUCKET, work_s3_prefix)
        _upload_dir(s3, work_dir, BUCKET, work_s3_prefix)
        logger.info("[done] %s", run_name)
    finally:
        stop_sync.set()


def _latest_iter_key(s3, work_s3_prefix):
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{work_s3_prefix}/latest_iter_")
    cands = sorted(
        (o["Key"] for o in resp.get("Contents", [])),
        key=lambda k: int(k.rsplit("_", 1)[-1].split(".")[0]),
    )
    if not cands:
        raise RuntimeError(f"no latest_iter_*.pth under s3://{BUCKET}/{work_s3_prefix}")
    return cands[-1]


def eval_entrypoint_fn(config: dict[Any, Any]) -> None:
    """Det+map eval on a B2D stage-1 checkpoint (latest synced iter unless pinned)."""
    import ray
    num_gpus = int(config.get("num_gpus", 8))
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, runtime_env={"env_vars": cred_env})
    def _eval(cfg, n):
        _run_eval(cfg, n)

    ray.get(_eval.remote(config, num_gpus))


def _run_eval(config, num_gpus):
    repo_root = os.path.abspath(os.path.dirname(__file__))
    run_name = config.get("run_name", "b2d_stage1_eval")
    config_file = config.get(
        "config_file", "projects/configs/sparsedrive_b2d_stage1.py"
    )
    train_run = config.get("train_run_name", "b2d_stage1_v1_a100x8")
    ckpt_prefix = config.get("work_s3_prefix", f"{WORK_PREFIX_DEFAULT}/{train_run}")
    out_prefix = config.get("eval_s3_prefix", f"{WORK_PREFIX_DEFAULT}/{run_name}")
    work_dir = os.path.join(repo_root, "work_dirs", run_name)
    s3 = _s3_client()

    env_python = _stage_env(s3)
    _stage_data(s3, repo_root, num_threads=int(config.get("stage_threads", 12)))
    _compile_ops(env_python, repo_root, config.get("torch_cuda_arch_list", "8.0"))

    ckpt_s3 = config.get("checkpoint_s3")
    if ckpt_s3:
        bucket, key = ckpt_s3.replace("s3://", "").split("/", 1)
    else:
        bucket, key = BUCKET, _latest_iter_key(s3, ckpt_prefix)
        ckpt_s3 = f"s3://{bucket}/{key}"
    local_ckpt = os.path.join(work_dir, os.path.basename(key))
    os.makedirs(work_dir, exist_ok=True)
    _download_file_s3(s3, bucket, key, local_ckpt)
    logger.info("[eval] checkpoint %s", ckpt_s3)

    eval_env = {
        **os.environ,
        "PATH": f"{os.path.join(ENV_LOCAL, 'bin')}:{os.environ.get('PATH', '')}",
        "PYTHONPATH": repo_root,
        "WANDB_NAME": run_name,
        "PYTHONUNBUFFERED": "1",
    }
    _run_cmd(
        [env_python, "-m", "torch.distributed.run",
         f"--nproc_per_node={num_gpus}", "--master_port=28652",
         os.path.join(repo_root, "tools", "test_pyfocal.py"),
         os.path.join(repo_root, config_file),
         local_ckpt,
         "--launcher", "pytorch",
         "--eval", "bbox",
         "--cfg-options",
         f"work_dir={work_dir}",
         "data.workers_per_gpu=4",
         "data.test.samples_per_gpu=4"],
        tag="eval", cwd=repo_root, env=eval_env,
    )
    logger.info("[upload] eval work_dir -> s3://%s/%s", BUCKET, out_prefix)
    _upload_dir(s3, work_dir, BUCKET, out_prefix)
    logger.info("[done] eval %s", run_name)
