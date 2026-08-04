# V6c CONTROL: stock stage2 baseline reproduced under the exact V6 regime —
# same official-stage1 init, same 10-epoch schedule, same withmap pkls, same
# python-focal training entry — with the planner's IMAGE instance features
# left enabled (no geometric_inputs override). Establishes how low planning
# error goes when the planner keeps its image-derived inputs; the only
# difference vs stage2_v6_geoinput.py is the absence of the model override.
_base_ = ["../sparsedrive_small_stage2.py"]

_iters_per_epoch = 586

log_config = dict(
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(project="sparsedrive-declutter"),
            interval=51,
            by_epoch=False,
        ),
    ],
)

checkpoint_config = dict(interval=_iters_per_epoch, max_keep_ckpts=3)

# map-annotated infos (original pkls have empty map_annos)
data = dict(
    train=dict(ann_file="data/infos/nuscenes_infos_train_withmap.pkl"),
    val=dict(
        ann_file="data/infos/nuscenes_infos_val_withmap.pkl",
        eval_config=dict(ann_file="data/infos/nuscenes_infos_val_withmap.pkl"),
    ),
    test=dict(
        ann_file="data/infos/nuscenes_infos_val_withmap.pkl",
        eval_config=dict(ann_file="data/infos/nuscenes_infos_val_withmap.pkl"),
    ),
)
