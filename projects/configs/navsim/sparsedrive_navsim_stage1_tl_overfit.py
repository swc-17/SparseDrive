# Traffic-light overfit gate: detection + map (incl. stop_line + TL
# attributes) on the same frozen 8-token navmini manifest as the Phase-1/2
# gates. Gate: total detection AND map losses (incl. map_loss_tl_*) fall
# by >= 95% from their first-20-step means.
#
# Mirrors sparsedrive_navsim_stage1_maps_overfit.py (no photometric jitter,
# no grid mask, no flips/rotation, deterministic resize/crop,
# sequences_split_num="all"), on the TL config.
_base_ = ["./sparsedrive_navsim_stage1_tl.py"]

# frozen overfit manifest: sorted(tokens)[:8] of navsim_infos_navmini
overfit_tokens = [
    "00b34c91088a5f04",
    "00f53a22cb3e5bd6",
    "0160a218dc9051bd",
    "01a303fd4e9d54d5",
    "01cbcd1439e05cbc",
    "01d3a49577c256d6",
    "02536b72a70250d3",
    "028125098bb45d66",
]

num_gpus = 8
batch_size = 1
num_iters = 1500

model = dict(use_grid_mask=False)

class_names = [
    "vehicle",
    "pedestrian",
    "bicycle",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
]
map_class_names = [
    "ped_crossing",
    "divider",
    "boundary",
    "stop_line",
]
stop_line_label = map_class_names.index("stop_line")
roi_size = (30, 60)
num_sample = 20
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="MultiViewUndistortImage"),
    dict(type="ResizeCropFlipImage"),
    dict(type="BBoxRotation"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type="VectorizeMap",
        roi_size=roi_size,
        simplify=False,
        normalize=False,
        sample_num=num_sample,
        permute=True,
        tl_label=stop_line_label,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_map_labels",
            "gt_map_pts",
            "gt_map_tl",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]

# deterministic augmentation: fixed minimal resize, no flip, no rotation
data_aug_conf = {
    "resize_lim": (704 / 1920, 704 / 1920),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 1080,
    "W": 1920,
    "rand_flip": False,
    "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        tokens=overfit_tokens,
        pipeline=train_pipeline,
        data_aug_conf=data_aug_conf,
        sequences_split_num="all",
    ),
)

optimizer = dict(lr=2e-4)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(type="IterBasedRunner", max_iters=num_iters)
checkpoint_config = dict(interval=500, max_keep_ckpts=3)

log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(project="sparsedrive-navsim"),
            interval=1,
            by_epoch=False,
        ),
    ],
)
evaluation = dict(interval=num_iters * 10)
