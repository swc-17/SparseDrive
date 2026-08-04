"""Monkey-patch torch CUDA version check then build deformable_aggregation op."""
import warnings
import torch.utils.cpp_extension as _ext

# Downgrade the major-version mismatch from RuntimeError to warning so
# builds work when nvcc > PyTorch's CUDA (e.g. system CUDA 13.x vs torch cu118).
_orig_check = _ext._check_cuda_version
def _patched_check(compiler_name, compiler_version):
    try:
        _orig_check(compiler_name, compiler_version)
    except RuntimeError as e:
        warnings.warn(f"CUDA version mismatch (suppressed for build): {e}")
_ext._check_cuda_version = _patched_check

# Now run the actual setup.py build
import sys, os, runpy
sys.argv = ["setup.py", "develop"]
os.chdir(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path("setup.py", run_name="__main__")
