"""10-epoch vocab schedule on 2 nodes / 16 GPUs, recipe unchanged.

Same total batch 32 (2/GPU x 16) so num_iters_per_epoch, max_iters, lr, and
the resume point are identical to the 8-GPU run — only wall-clock per iter
changes. Resumes seamlessly from the same work_dir checkpoints.
"""

_base_ = ["./sparsedrive_navsim_stage2_vocab_full_10ep.py"]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)
