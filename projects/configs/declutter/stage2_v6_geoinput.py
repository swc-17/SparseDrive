# V6: full baseline stage2 pipeline with GEOMETRIC-ONLY planner inputs.
# Everything intact — 901-query planner, temporal InstanceQueue, motion
# co-training, recursive predicted ego status, joint stage2 training from the
# official stage1 checkpoint — except every image-derived feature the planner
# consumes is replaced by an MLP encoding of the corresponding geometry
# (det anchors / map polylines / ego anchor). Isolates the input
# representation inside the baseline's own regime (no measured-ego shortcut).
# The ego CNN is not built (use_cam_ego_feature=False) so all params are
# used and DDP runs without find_unused_parameters.
_base_ = ["../sparsedrive_small_stage2.py"]

_iters_per_epoch = 586

model = dict(
    head=dict(
        motion_plan_head=dict(
            geometric_inputs=True,
            instance_queue=dict(use_cam_ego_feature=False),
        ),
    ),
)

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
