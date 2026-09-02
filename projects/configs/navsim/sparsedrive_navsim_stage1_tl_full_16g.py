# NAVSIM stage-1 FULL training with traffic lights, 2-node / 16-GPU
# variant. Model is sparsedrive_navsim_stage1_tl.py (det + 4-class maps
# incl. stop_line + TL attribute branch); schedule mirrors
# sparsedrive_navsim_stage1_full_16g.py (see rationale there).
#
# Requires the v1.3 navtrain infos + TL kmeans anchors:
#   python tools/data_converter/navsim_converter.py --split trainval \
#       --filter navtrain --split-json data/splits/navtrain_split_v1.json \
#       --split-key train_logs --workers 16 \
#       --output data/infos/navsim_infos_navtrain_train_tl.pkl
#   (same with --split-key val_logs -> navsim_infos_navtrain_val_tl.pkl)
#   python tools/kmeans/kmeans_map_navsim.py \
#       --info data/infos/navsim_infos_navtrain_train_tl.pkl \
#       --out data/kmeans/kmeans_map_100_navsim_tl.npy
_base_ = ["./sparsedrive_navsim_stage1_tl.py"]

import os

version = "navtrain_train"
length = {
    "navtrain_train": 92853,  # navsim_infos_navtrain_train_tl.pkl (1067 logs)
    "navtrain_val": 10435,    # navsim_infos_navtrain_val_tl.pkl  (125 logs)
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
    # maps-only survived at 4e-4; TL at 4e-4 NaN'd even with clip=1.
    # Halve LR (same trick that fixed maps 5.7e-4 -> 4e-4); restore
    # clip to maps-only default so this run isolates the LR change.
    lr=2e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
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
        ann_file=anno_root + "navsim_infos_navtrain_train_tl.pkl",
    ),
    val=dict(
        data_root=data_root,
        ann_file=anno_root + "navsim_infos_navtrain_val_tl.pkl",
    ),
    test=dict(
        data_root=data_root,
        ann_file=anno_root + "navsim_infos_navtrain_val_tl.pkl",
    ),
)

# no offline evaluator wired for stage 1; training runs with --no-validate.
evaluation = dict(interval=num_iters_per_epoch * num_epochs * 10)
