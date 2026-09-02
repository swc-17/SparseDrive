# NAVSIM stage-1 offline det+map eval on official navtest
# (12,146 samples). Same model/metrics as
# sparsedrive_navsim_stage1_eval.py; only the info pickle and data_root
# wiring change. Images come from $NAVSIM_BLOBS_ROOT (eval-frame tars).
_base_ = ["./sparsedrive_navsim_stage1_eval.py"]

import os

data_root = os.environ.get("NAVSIM_BLOBS_ROOT", "data/navsim")
ann_file = "data/infos/navsim_infos_navtest.pkl"

eval_config = dict(
    data_root=data_root,
    ann_file=ann_file,
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val=dict(data_root=data_root, ann_file=ann_file, eval_config=eval_config),
    test=dict(
        data_root=data_root,
        ann_file=ann_file,
        samples_per_gpu=4,
        eval_config=eval_config,
    ),
)
