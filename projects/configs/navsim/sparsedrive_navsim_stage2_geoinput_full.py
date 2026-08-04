# NAVSIM stage-2 FULL training (Phase 5): joint detection + maps + motion +
# V6 geometric-input planner on the frozen navtrain train split, initialized
# from the stage-1 full checkpoint (injected on Lilypad via load_from_s3 ->
# --cfg-options load_from=...). Model is exactly the gated Phase-3 V6 config
# (sparsedrive_navsim_stage2_geoinput.py); only data + schedule change here.
#
# Schedule: mirrors the official nuScenes stage-2 recipe
# (projects/configs/sparsedrive_small_stage2.py: 10 epochs, total batch 48,
# lr 3e-4 cosine, warmup 500) scaled to navtrain:
#   - total_batch_size=32 (4/GPU on 8xA100), the memory-proven NAVSIM
#     setting; lr linearly rescaled 3e-4 * 32/48 = 2e-4;
#   - num_epochs=3 matches the official run's total sample exposure:
#     nuScenes 10 ep x 28,130 = 281k samples ~= 3 ep x 92,853 = 279k.
# Anchors: navtrain-regenerated kmeans (S3 data/kmeans/navtrain/, staged to
# the same local paths the base config reads; provenance in
# data/kmeans/kmeans_navsim_provenance.json).
_base_ = ["./sparsedrive_navsim_stage2_geoinput.py"]

import os

version = "navtrain_train"
length = {
    "navtrain_train": 92853,  # navsim_infos_navtrain_train.pkl (1067 logs)
    "navtrain_val": 10435,    # navsim_infos_navtrain_val.pkl  (125 logs)
}  # 92853 + 10435 = 103288 = the official navtrain scenario count

total_batch_size = 32
num_gpus = 8
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 3
checkpoint_epoch_interval = 1

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)
optimizer = dict(
    type="AdamW",
    lr=2e-4,  # official stage-2 3e-4 @ total batch 48, linear-scaled to 32
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
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

data_root = os.environ.get(
    "NAVSIM_BLOBS_ROOT", "/media/applied/navsim/sensor_blobs/trainval/"
)
anno_root = "data/infos/"

ego_fut_ts = 8
eval_pipeline = [
    dict(
        type="Collect",
        keys=[
            "gt_ego_fut_trajs",
            "gt_ego_fut_masks",
            "gt_ego_fut_cmd",
            "fut_boxes",
        ],
        meta_keys=["token", "timestamp"],
    ),
]
# Phase-4a open-loop L2/collision diagnostic on the held-out navtrain val
# split (base config's eval_config points at navmini).
eval_config = dict(
    type="NavSim3DDataset",
    ann_file=anno_root + "navsim_infos_navtrain_val.pkl",
    pipeline=eval_pipeline,
    test_mode=True,
    planning_metric=dict(
        n_future=ego_fut_ts,          # 8 x 0.5 s = 4 s
        ego_width=2.297,              # Pacifica
        ego_length=5.176,
        ego_height=1.777,
        ego_center_offset=1.461,
    ),
)

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=anno_root + "navsim_infos_navtrain_train.pkl",
    ),
    val=dict(
        data_root=data_root,
        ann_file=anno_root + "navsim_infos_navtrain_val.pkl",
        eval_config=eval_config,
    ),
    test=dict(
        data_root=data_root,
        ann_file=anno_root + "navsim_infos_navtrain_val.pkl",
        eval_config=eval_config,
    ),
)

# stage-2 initializes from the stage-1 full checkpoint; on Lilypad the
# entrypoint injects it via --cfg-options load_from=... (load_from_s3 knob in
# lilypad_config/navsim/stage2_geoinput_full.yaml).
load_from = None
