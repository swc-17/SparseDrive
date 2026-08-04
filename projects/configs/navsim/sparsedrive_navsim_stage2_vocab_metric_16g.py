"""PDM metric-head fine-tune on 2 nodes / 16 GPUs, recipe unchanged.

Same total batch 32 (2/GPU x 16) as the single-node metric config; PDM
scoring pools run per rank, so two nodes double CPU scoring throughput.
"""

_base_ = ["./sparsedrive_navsim_stage2_vocab_metric_full.py"]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)
