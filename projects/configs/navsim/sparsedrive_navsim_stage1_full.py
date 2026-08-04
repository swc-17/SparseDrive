# NAVSIM stage-1 FULL training (Phase 5): detection + online vector maps on
# the frozen navtrain train split (data/splits/navtrain_split_v1.json,
# navsim_converter --split-key train_logs), validated on the held-out
# navtrain val split. Model is exactly the gated Phase-1/2 det+maps config
# (sparsedrive_navsim_stage1_maps.py); only data + schedule change here.
#
# Schedule: mirrors the official nuScenes recipe
# (projects/configs/sparsedrive_small_stage1.py: 100 epochs, total batch 64,
# lr 4e-4 cosine, warmup 500) scaled to navtrain:
#   - total_batch_size=32 (4/GPU on 8xA100) — the memory-proven NAVSIM
#     setting (8 cams + 4-frame history replay is heavier than nuScenes
#     6-cam at 8/GPU); lr linearly rescaled 4e-4 * 32/64 = 2e-4;
#   - num_epochs=30 matches the official run's total sample exposure:
#     nuScenes 100 ep x 28,130 = 2.81M samples ~= 30 ep x 92,853 = 2.79M.
# Anchors: navtrain-regenerated kmeans (S3 data/kmeans/navtrain/, staged to
# the same local paths data/kmeans/kmeans_*_navsim.npy the base config
# reads; provenance in data/kmeans/kmeans_navsim_provenance.json).
_base_ = ["./sparsedrive_navsim_stage1_maps.py"]

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
num_epochs = 30
checkpoint_epoch_interval = 5

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)
optimizer = dict(
    type="AdamW",
    lr=2e-4,  # official 4e-4 @ total batch 64, linear-scaled to 32
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

# navtrain sensor blobs (per-log current-frame tars staged by
# lilypad_entrypoint_navsim under frames_prefix); local fallback is the
# full local blob tree.
data_root = os.environ.get(
    "NAVSIM_BLOBS_ROOT", "/media/applied/navsim/sensor_blobs/trainval/"
)
anno_root = "data/infos/"

data = dict(
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

# no offline evaluator wired for stage 1 (PDMS is stage-2/Phase-4);
# training runs with --no-validate.
evaluation = dict(interval=num_iters_per_epoch * num_epochs * 10)
