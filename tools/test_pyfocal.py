"""Eval entry for the packed sparsedrive310 env (mmcv-lite, torch>=2.6)."""
import os
import runpy

import torch
from mmcv.parallel.distributed import MMDistributedDataParallel

# mmcv 1.7 MMDDP predates torch DDP's _use_replicated_tensor_module.
# train_step never hits DDP.forward; tools/test.py does, so eval dies without this.
MMDistributedDataParallel._use_replicated_tensor_module = False

_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat

_here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_here, "test.py"), run_name="__main__")
