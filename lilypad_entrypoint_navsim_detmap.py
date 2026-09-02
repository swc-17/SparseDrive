"""Det+map eval on NAVSIM navtest for SparseDrive stage-1."""
import glob
import logging
import os
import sys
from typing import Any

from lilypad_entrypoint import (
    DECLUTTER_PREFIX,
    ENV_LOCAL,
    USER_BUCKET,
    _download_file_s3,
    _find_cuda_home,
    _run_cmd,
    _s3_client,
    _upload_dir,
)
from lilypad_entrypoint_navsim import (
    NAVSIM_PREFIX,
    _once,
    _parse_s3_uri,
    _stage_tars,
    _start_persistent_keepalive,
    _uri_dir,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)


def _stage_sd_env(s3):
    python = os.path.join(ENV_LOCAL, "bin", "python")
    if os.path.exists(python):
        return python
    logger.info("[env] downloading packed sparsedrive env ...")
    _download_file_s3(
        s3, USER_BUCKET,
        f"{DECLUTTER_PREFIX}/env/sparsedrive310_env.tar.gz",
        "/tmp/sparsedrive310_env.tar.gz",
    )
    os.makedirs(ENV_LOCAL, exist_ok=True)
    _run_cmd(["tar", "-xzf", "/tmp/sparsedrive310_env.tar.gz", "-C", ENV_LOCAL],
             tag="env-untar")
    _run_cmd([os.path.join(ENV_LOCAL, "bin", "conda-unpack")], tag="conda-unpack")
    return python


def _compile_ops(env_python, repo_root, arch_list):
    ops_dir = os.path.join(repo_root, "projects", "mmdet3d_plugin", "ops")
    for so in glob.glob(os.path.join(ops_dir, "*.so")):
        os.remove(so)
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


def detmap_entrypoint_fn(config: dict[Any, Any]) -> None:
    import ray

    num_gpus = int(config.get("num_gpus", 8))
    if not ray.is_initialized():
        ray.init()
    cred_env = {k: v for k, v in os.environ.items()
                if k.startswith(("AWS_", "OCI_", "WANDB_")) and v}

    @ray.remote(num_gpus=num_gpus, max_retries=0,
                runtime_env={"env_vars": cred_env})
    def _eval(cfg, gpu_count):
        stop = _start_persistent_keepalive(gpu_count)
        try:
            _run_detmap_eval(cfg, gpu_count)
        finally:
            stop.set()

    ray.get(_eval.remote(config, num_gpus))


def _run_detmap_eval(config, num_gpus):
    repo_root = os.path.abspath(os.path.dirname(__file__))
    run_name = config.get("run_name", "navsim_stage1_navtest_detmap")
    config_file = config.get(
        "config_file",
        "projects/configs/navsim/sparsedrive_navsim_stage1_eval_navtest.py",
    )
    out_prefix = config.get(
        "eval_s3_prefix",
        f"{NAVSIM_PREFIX}/work_dirs/{run_name}",
    )
    work_dir = os.path.join(repo_root, "work_dirs", run_name)
    os.makedirs(work_dir, exist_ok=True)
    s3 = _s3_client()

    env_python = _stage_sd_env(s3)
    _compile_ops(env_python, repo_root,
                 config.get("torch_cuda_arch_list", "8.0"))

    for s3_rel, repo_rel in config.get("stage_files", [
        ["data/kmeans/navtrain/kmeans_det_900_navsim.npy",
         "data/kmeans/kmeans_det_900_navsim.npy"],
        ["data/kmeans/navtrain/kmeans_map_100_navsim.npy",
         "data/kmeans/kmeans_map_100_navsim.npy"],
    ]):
        dest = os.path.join(repo_root, repo_rel)
        _once(
            f"file:{s3_rel}->{repo_rel}",
            lambda s3_rel=s3_rel, dest=dest: _download_file_s3(
                _s3_client(), USER_BUCKET, f"{NAVSIM_PREFIX}/{s3_rel}", dest,
            ),
        )

    infos_s3 = config["infos_s3"]
    infos_local = os.path.join(repo_root, "data", "infos",
                               os.path.basename(infos_s3))
    b, k = _parse_s3_uri(infos_s3)
    _once(f"infos:{infos_s3}",
          lambda: _download_file_s3(_s3_client(), b, k, infos_local))

    blobs_root = config.get(
        "blobs_root",
        _uri_dir("/tmp/navsim_blobs", config["frame_tars_s3"]),
    )
    _once(
        f"tars:{config['frame_tars_s3']}->{blobs_root}",
        lambda: _stage_tars(_s3_client(), config["frame_tars_s3"], blobs_root),
    )

    ckpt_s3 = config["checkpoint_s3"]
    b, k = _parse_s3_uri(ckpt_s3)
    local_ckpt = os.path.join(work_dir, os.path.basename(k))
    _download_file_s3(s3, b, k, local_ckpt)
    logger.info("[eval] checkpoint %s", ckpt_s3)

    eval_env = {
        **os.environ,
        "PATH": f"{os.path.join(ENV_LOCAL, 'bin')}:{os.environ.get('PATH', '')}",
        "PYTHONPATH": repo_root,
        "PYTHONUNBUFFERED": "1",
        "NAVSIM_BLOBS_ROOT": blobs_root,
    }
    _run_cmd(
        [env_python, "-m", "torch.distributed.run",
         f"--nproc_per_node={num_gpus}", "--master_port=28653",
         os.path.join(repo_root, "tools", "test_pyfocal.py"),
         os.path.join(repo_root, config_file),
         local_ckpt,
         "--launcher", "pytorch",
         "--eval", "bbox", "map",
         "--cfg-options",
         f"work_dir={work_dir}",
         "data.workers_per_gpu=4",
         "data.test.samples_per_gpu=4"],
        tag="eval", cwd=repo_root, env=eval_env,
    )
    logger.info("[upload] eval work_dir -> s3://%s/%s", USER_BUCKET, out_prefix)
    _upload_dir(s3, work_dir, USER_BUCKET, out_prefix)
    logger.info("[done] eval %s", run_name)
