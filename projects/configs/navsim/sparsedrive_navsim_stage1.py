# NAVSIM stage-1 (Phase 1: camera-only detection), modeled on
# projects/configs/sparsedrive_small_stage1.py with the PORT-doc deltas:
#   - 8 cameras (CAM_F0 first), num_cams=8 in the detection
#     DeformableFeatureAggregation;
#   - 7 NAVSIM classes (vehicle, pedestrian, bicycle, traffic_cone, barrier,
#     czone_sign, generic_object) and re-indexed class-dependent weights;
#   - source geometry 1920x1080, shared 704x256 network input;
#   - deterministic per-camera undistortion (plumb-bob coeffs from the infos)
#     before resize/crop, keeping the same camera matrix K;
#   - depth_branch=None, no LiDAR/point/depth pipeline (merged PCD unsupported
#     and LiDAR missing for some official samples);
#   - map head OFF (Phase 2: navmini infos have map_annos_populated=False);
#   - zero 3D rotation augmentation;
#   - use_valid_flag=True, 55 m range policy from the converter.
#
# TODO(navtrain): iteration counts/anchors below are navmini-scale. Before any
# real training run: regenerate kmeans anchors on the converted navtrain
# split (see tools/kmeans/kmeans_det_navsim.py) and update `length`.
import os

# ================ base config ===================
version = "navmini"
length = {"navmini": 396}

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None

total_batch_size = 32
num_gpus = 8
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 100
checkpoint_epoch_interval = 20

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(project="sparsedrive-navsim"),
            interval=10,
            by_epoch=False,
        ),
    ],
)
load_from = None
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)

# ================== model ========================
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
]
num_classes = len(class_names)
num_map_classes = len(map_class_names)
roi_size = (30, 60)

num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 8  # NAVSIM: 8 SE(2) poses at 0.5 s (4 s horizon)
ego_fut_mode = 6
queue_length = 4  # history + current

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
drop_out = 0.1
temporal = True
decouple_attn = True
with_quality_estimation = True
num_cams = 8  # fixed order: CAM_F0, L0, L1, L2, R0, R1, R2, B0

task_config = dict(
    with_det=True,
    with_map=False,  # Phase 2 (navmini infos carry no map annos yet)
    with_motion_plan=False,
)

model = dict(
    type="SparseDrive",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpt/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=None,  # no LiDAR depth supervision in the NAVSIM port (yet)
    head=dict(
        type="SparseDriveHead",
        task_config=task_config,
        det_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                # navmini-only anchors — TODO(navtrain): regenerate
                anchor="data/kmeans/kmeans_det_900_navsim.npy",
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal else -1,
                confidence_decay=0.6,
                feat_grad=False,
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder)
            )[2:],
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal
            else None,
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=num_cams,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_classes,
                refine_yaw=True,
                with_quality_estimation=with_quality_estimation,
            ),
            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={
                    class_names.index("traffic_cone"): [
                        2.0,
                        2.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                },
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=[class_names.index("barrier")],
            ),
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),
    ),
)

# ================== data ========================
dataset_type = "NavSim3DDataset"
# Root of the NAVSIM sensor blobs for this split; image paths in the infos
# are relative to it. Overridable so Lilypad jobs can stage blobs anywhere.
data_root = os.environ.get(
    "NAVSIM_BLOBS_ROOT", "/media/applied/navsim/sensor_blobs/mini/"
)
anno_root = "data/infos/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    # NAVSIM distortion is nonzero: rectify deterministically BEFORE any
    # resize/crop so all projection matrices (built from pinhole K) hold.
    dict(type="MultiViewUndistortImage"),
    dict(type="ResizeCropFlipImage"),
    dict(type="BBoxRotation"),  # rot3d_range is [0, 0] — identity, kept for parity
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
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
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    map_classes=map_class_names,
    modality=input_modality,
)
data_aug_conf = {
    # NAVSIM source geometry 1920x1080 -> shared 704x256 network input.
    # min resize 704/1920 = 0.3667 keeps width covered; bottom crop like
    # nuScenes (0.3667 * 1080 = 396 -> crop 140 px from the top).
    "resize_lim": (0.37, 0.42),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 1080,
    "W": 1920,
    "rand_flip": True,
    "rot3d_range": [0, 0],  # zero 3D rotation until geometry is proven
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "navsim_infos_navmini.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "navsim_infos_navmini.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "navsim_infos_navmini.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=4e-4,
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
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

# NAVSIM has no nuScenes-style offline evaluator (PDMS comes in Phase 4);
# train with --no-validate.
evaluation = dict(interval=num_iters_per_epoch * num_epochs * 10)
