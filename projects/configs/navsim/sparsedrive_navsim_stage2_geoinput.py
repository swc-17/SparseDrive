# NAVSIM stage-2, built DIRECTLY as the V6 geometric-only-planner variant
# (NAVSIM_SPARSEDRIVE_PLAN.md Phase 3): detection + map perception (from the
# Phase-1/2 stage-1-maps config) joint-trained with motion + planning where
# every image-derived feature the planner consumes is replaced by an MLP
# encoding of the corresponding geometry (det anchors / map polylines / ego
# anchor): motion_plan_head.geometric_inputs=True and
# instance_queue.use_cam_ego_feature=False (the ego CNN is not built, so all
# params are used and DDP runs without find_unused_parameters).
#
# NAVSIM deltas vs projects/configs/sparsedrive_small_stage2.py:
#   - ego_fut_ts=8 (8 SE(2) poses at 0.5 s = 4 s horizon; nuScenes was 6x0.5s
#     = 3 s). All eight-step shapes are configuration-driven
#     (MotionPlanningRefinementModule, PlanningTarget,
#     HierarchicalPlanningDecoder take ego_fut_ts) — only the anchors change;
#   - fut_ts=12 for agents with masks (NAVSIM windows fill at most 10 steps);
#   - 7-class motion anchors kmeans_motion_6_navsim.npy (one row per NAVSIM
#     class; near-static classes cluster to near-zero motion);
#   - plan anchors kmeans_plan_6_navsim.npy (3 commands x 6 modes x 8 x 2),
#     generated EXCLUDING gt_ego_fut_cmd_valid=0 rows;
#   - command [right, left, straight] with gt_ego_fut_cmd_valid masking:
#     'unknown' rows contribute zero planner cls+reg loss (PlanningTarget);
#   - instance_queue ego anchor = Pacifica in the NAVSIM rear-axle frame
#     (not the nuScenes Zoe/lidar-frame anchor);
#   - use_rescore=False: the rescore path hardcodes nuScenes ego geometry and
#     reported planner results must not use heuristic rescoring (PORT doc);
#   - no LiDAR depth branch anywhere (inherited).
#
# TODO(navtrain): iteration counts and all four kmeans anchors below are
# navmini-scale; regenerate on the converted navtrain split before any real
# (non-overfit) training run.
_base_ = ["./sparsedrive_navsim_stage1_maps.py"]

class_names = [
    "vehicle",
    "pedestrian",
    "bicycle",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
]
roi_size = (30, 60)
num_sample = 20

fut_ts = 12
fut_mode = 6
ego_fut_ts = 8  # NAVSIM: 8 steps at 0.5 s (4 s); nuScenes stage2 uses 6
ego_fut_mode = 6
queue_length = 4  # history + current

embed_dims = 256
num_groups = 8
drop_out = 0.1
decouple_attn_motion = True
strides = [4, 8, 16, 32]
input_shape = (704, 256)

# NAVSIM ego vehicle: Chrysler Pacifica in the rear-axle frame
# (nuPlan get_pacifica_parameters: width 2.297, length 5.176 = 4.049 front +
# 1.127 rear, height 1.777; box center 5.176/2 - 1.127 = 1.461 m ahead of
# the rear axle). SD frame: forward=+Y, ground at z=-0.35. Slot layout
# matches the upstream nuScenes ego anchor: [X, Y, Z, logW, logL, logH,
# sin_yaw, cos_yaw, VX, VY, VZ] with yaw=+90 deg so the W slot holds the
# along-heading (length) extent — the same convention the converter's
# NAVSIM gt_boxes use (slot 3 = length).
import numpy as _np
ego_anchor_navsim = [
    0.0, 1.461, -0.35 + 1.777 / 2,
    float(_np.log(5.176)), float(_np.log(2.297)), float(_np.log(1.777)),
    1.0, 0.0, 0.0, 0.0, 0.0,
]

model = dict(
    head=dict(
        task_config=dict(
            with_det=True,
            with_map=True,
            with_motion_plan=True,
        ),
        motion_plan_head=dict(
            type="MotionPlanningHead",
            fut_ts=fut_ts,
            fut_mode=fut_mode,
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
            # navmini-only anchors — TODO(navtrain): regenerate
            motion_anchor=f"data/kmeans/kmeans_motion_{fut_mode}_navsim.npy",
            plan_anchor=f"data/kmeans/kmeans_plan_{ego_fut_mode}_navsim.npy",
            embed_dims=embed_dims,
            decouple_attn=decouple_attn_motion,
            # V6: geometric-only planner inputs
            geometric_inputs=True,
            instance_queue=dict(
                type="InstanceQueue",
                embed_dims=embed_dims,
                queue_length=queue_length,
                tracking_threshold=0.2,
                feature_map_scale=(
                    input_shape[1] / strides[-1],
                    input_shape[0] / strides[-1],
                ),
                use_cam_ego_feature=False,  # V6: no front-cam ego CNN
                ego_anchor=ego_anchor_navsim,
            ),
            operation_order=(
                [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "cross_gnn",
                    "norm",
                    "ffn",
                    "norm",
                ] * 3 +
                [
                    "refine",
                ]
            ),
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_motion
                else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_motion
                else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            cross_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 2,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            refine_layer=dict(
                type="MotionPlanningRefinementModule",
                embed_dims=embed_dims,
                fut_ts=fut_ts,
                fut_mode=fut_mode,
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),
            motion_sampler=dict(
                type="MotionTarget",
            ),
            motion_loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2,
            ),
            motion_loss_reg=dict(type="L1Loss", loss_weight=0.2),
            planning_sampler=dict(
                type="PlanningTarget",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),
            plan_loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.5,
            ),
            plan_loss_reg=dict(type="L1Loss", loss_weight=1.0),
            plan_loss_status=dict(type="L1Loss", loss_weight=1.0),
            motion_decoder=dict(type="SparseBox3DMotionDecoder"),
            planning_decoder=dict(
                type="HierarchicalPlanningDecoder",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
                use_rescore=False,  # rescore hardcodes nuScenes ego geometry
            ),
            num_det=50,
            num_map=10,
        ),
    ),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="MultiViewUndistortImage"),
    dict(type="ResizeCropFlipImage"),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
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
            "gt_agent_fut_trajs",
            "gt_agent_fut_masks",
            "gt_ego_fut_trajs",
            "gt_ego_fut_masks",
            "gt_ego_fut_cmd",
            "gt_ego_fut_cmd_valid",
            "ego_status",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="MultiViewUndistortImage"),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "ego_status",
            "gt_ego_fut_cmd",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

# ================== eval (Phase 4a open-loop L2/collision diagnostic) ======
# SparseDrive's PlanningMetric extended to the NAVSIM horizon: 8 steps at
# 0.5 s (4 s) with Pacifica ego geometry in the rear-axle frame. This is the
# fast tools/test.py diagnostic; official scores are NAVSIM v1 PDMS /
# v2 EPDMS via navsim_agent/ (see docs/navsim_eval_openloop.md).
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
eval_config = dict(
    type="NavSim3DDataset",
    ann_file="data/infos/navsim_infos_navmini.pkl",
    pipeline=eval_pipeline,
    test_mode=True,
    # consumed by NavSim3DDataset.evaluate -> planning_eval, not by the
    # dataset constructor
    planning_metric=dict(
        n_future=ego_fut_ts,          # 8 x 0.5 s = 4 s
        ego_width=2.297,              # Pacifica (nuPlan get_pacifica_parameters)
        ego_length=5.176,
        ego_height=1.777,
        ego_center_offset=1.461,      # box center ahead of the rear axle
    ),
)

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline, eval_config=eval_config),
    test=dict(pipeline=test_pipeline, eval_config=eval_config),
)

eval_mode = dict(
    with_det=False,
    with_tracking=False,
    with_map=False,
    with_motion=False,
    with_planning=True,
)
evaluation = dict(eval_mode=eval_mode)

# ================== training ========================
# Mirrors the nuScenes stage-2 recipe: lower lr than stage 1 and a
# 0.1 backbone lr multiplier, joint training from a stage-1 checkpoint.
optimizer = dict(
    type="AdamW",
    lr=3e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
)

# Stage-2 initializes from a stage-1 (det+map) checkpoint. Local runs:
# point this at the downloaded checkpoint; Lilypad jobs inject it via
# --cfg-options load_from=... (entrypoint load_from_s3 knob).
load_from = None
