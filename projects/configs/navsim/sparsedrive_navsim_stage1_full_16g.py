# NAVSIM stage-1 FULL training, 2-node / 16-GPU variant of
# sparsedrive_navsim_stage1_full.py. Model is identical (gated Phase-1/2
# det+maps, sparsedrive_navsim_stage1_maps.py); only the schedule scales.
#
# Schedule (vs the 8-GPU full run: total batch 32, lr 2e-4, 87,030 iters):
#   - total_batch_size=128: 8/GPU on 16xA100-80G. The official nuScenes
#     recipe runs 8/GPU with 6 cams; NAVSIM is 8 cams + 4-frame history
#     replay, so 8/GPU is unproven on A100-80G — the 2-node smoke gates
#     this; fallback is 6/GPU (=96 global, rescale below).
#   - num_epochs=30 keeps the official run's total sample exposure:
#     nuScenes 100 ep x 28,130 = 2.81M ~= 30 ep x 92,853 = 2.79M samples,
#     giving 92,853 // 128 = 725 iters/epoch -> 21,750 iters total.
#   - lr=5.7e-4: official 4e-4 @ batch 64. Linear scaling to 128 gives
#     8e-4, which is aggressive for a DETR-style sparse detector with
#     AdamW (adaptive optimizers don't obey linear noise scaling; query
#     -based detection heads are warmup/lr sensitive). We use sqrt
#     scaling instead: 4e-4 * sqrt(128/64) = 5.66e-4 ~= 5.7e-4, with the
#     same 500-iter linear warmup (at batch 128 that is 64k warmup
#     samples, ~2x the official warmup exposure — extra margin).
# Anchors/data: identical to sparsedrive_navsim_stage1_full.py (navtrain
# kmeans anchors + frozen navtrain train/val splits).
_base_ = ["./sparsedrive_navsim_stage1_maps.py"]

import os

version = "navtrain_train"
length = {
    "navtrain_train": 92853,  # navsim_infos_navtrain_train.pkl (1067 logs)
    "navtrain_val": 10435,    # navsim_infos_navtrain_val.pkl  (125 logs)
}

total_batch_size = 128
num_gpus = 16
batch_size = total_batch_size // num_gpus  # 8 per GPU
num_iters_per_epoch = int(length[version] // total_batch_size)  # 725
num_epochs = 30
checkpoint_epoch_interval = 5

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval  # 3,625
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,  # 21,750
)
optimizer = dict(
    type="AdamW",
    lr=4e-4,  # was 5.7e-4 (sqrt-scaled): NaN'd det cls losses at iter ~2.4k on run tj8r39; kept at official base value
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

# navtrain sensor blobs (staged per node by
# lilypad_entrypoint_navsim_multinode under frames_prefix).
data_root = os.environ.get(
    "NAVSIM_BLOBS_ROOT", "/media/applied/navsim/sensor_blobs/trainval/"
)
anno_root = "data/infos/"

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        data_root=data_root,
        ann_file=anno_root + "navsim_infos_navtrain_train.pkl",
    ),
    val=dict(
        data_root=data_root,
        ann_file=anno_root + "navsim_infos_navtrain_val.pkl",
    ),
    test=dict(
        data_root=data_root,
        ann_file=anno_root + "navsim_infos_navtrain_val.pkl",
    ),
)

# no offline evaluator wired for stage 1; training runs with --no-validate.
evaluation = dict(interval=num_iters_per_epoch * num_epochs * 10)
