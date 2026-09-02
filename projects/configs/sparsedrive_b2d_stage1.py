# ================ base config ===================
import os

version = os.environ.get("B2D_VERSION", "trainval")
# Computed from the corresponding info PKLs. Override when regenerating infos.
length = {
    "trainval": int(os.environ.get("B2D_TRAIN_SAMPLES", 234769)),
    "mini": int(os.environ.get("B2D_MINI_TRAIN_SAMPLES", 1932)),
}

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None

num_gpus = int(os.environ.get("B2D_NUM_GPUS", 8))
batch_size = int(
    os.environ.get("B2D_SAMPLES_PER_GPU", 8 if version == "trainval" else 4)
)
num_iters_per_epoch = max(1, length[version] // (num_gpus * batch_size))
num_epochs = 20
checkpoint_epoch_interval = 20

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
log_config = dict(
    interval=51 if "train" in version else 1,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ]
    + (
        [
            dict(
                type="WandbLoggerHook",
                by_epoch=False,
                init_kwargs=dict(
                    project="sparsedrive-b2d",
                    entity="research",
                    name=os.environ.get("WANDB_NAME"),
                ),
            )
        ]
        if os.environ.get("WANDB_API_KEY")
        else []
    ),
)
load_from = None
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 384)

data_root = os.environ.get("B2D_DATA_ROOT", "data/bench2drive/")
anno_root = os.environ.get("B2D_INFO_ROOT", "data/infos/")
kmeans_root = os.environ.get("B2D_KMEANS_ROOT", "data/kmeans/")
resnet50_checkpoint = os.environ.get(
    "B2D_RESNET50_CHECKPOINT", "ckpt/resnet50-19c8e357.pth"
)

# ================== model ========================
class_names = [
    'car',
    'van',
    'truck',
    'bicycle',
    'traffic_sign',
    'traffic_cone',
    'traffic_light',
    'pedestrian',
    'others',
]
map_class_names = [
    'Broken',
    'Solid',
    'SolidSolid',
    'Center',
    'TrafficLight',
    'StopSign',
]
NameMapping = {
    #=================vehicle=================
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.dodge.charger_police": 'car',
    "vehicle.dodge.charger_police_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.nissan.patrol_2021": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.ford.crown": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.tesla.model3": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": 'car',
    # bus
    # van
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    #=========================================

    #=================traffic sign============
    # traffic.speed_limit
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',
    
    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    #=========================================

    #===================Construction===========
    "static.prop.warningconstruction" : 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",

    #===================Construction===========
    "static.prop.constructioncone": 'traffic_cone',

    #=================pedestrian==============
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
    "walker.pedestrian.0017": 'pedestrian',
    "walker.pedestrian.0018": 'pedestrian',
    "walker.pedestrian.0019": 'pedestrian',
    "walker.pedestrian.0020": 'pedestrian',
    "walker.pedestrian.0022": 'pedestrian',
    "walker.pedestrian.0025": 'pedestrian',
    "walker.pedestrian.0035": 'pedestrian',
    "walker.pedestrian.0041": 'pedestrian',
    "walker.pedestrian.0046": 'pedestrian',
    "walker.pedestrian.0047": 'pedestrian',

    # ==========================================
    "static.prop.dirtdebris01": 'others',
    "static.prop.dirtdebris02": 'others',
}
num_classes = len(class_names)
num_map_classes = len(map_class_names)
roi_size = (30, 60)
num_cams = 6
# Computed from the invariant +1.84 m lidar2ego z translation in B2D infos.
ground_height = -float(os.environ.get("B2D_LIDAR_HEIGHT", 1.84))

num_sample = 20
fut_ts = 6
fut_mode = 6

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
use_deformable_func = os.environ.get("B2D_USE_DEFORMABLE_FUNC", "1") == "1"
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True

task_config = dict(
    with_det=True,
    with_map=True,
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
        pretrained=resnet50_checkpoint,
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
                anchor=os.path.join(kmeans_root, "kmeans_det_900.npy"),
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal else -1,
                confidence_decay=0.9,
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
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal
            else None,
            graph_model=dict(
                type="MultiheadAttention",
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
                # V1 Sparse4DHead DN with groups>0 OOB on cls/reg gather.
                # Match nuScenes/NAVSIM V1 stage-1: no denoising queries.
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
                # cls_allow_reverse=[class_names.index("barrier")],
            ),
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),
        map_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn_map,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor=os.path.join(kmeans_root, "kmeans_map_100.npy"),
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0 if temporal_map else -1,
                confidence_decay=0.9,
                feat_grad=True,
            ),
            anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=num_sample,
            ),
            num_single_frame_decoder=num_single_frame_decoder_map,
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder_map
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder_map)
            )[:],
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal_map
            else None,
            graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
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
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=num_sample,
                    num_learnable_pts=3,
                    fix_height=(0, 0.25, -0.25, 0.5, -0.5),
                    ground_height=ground_height,
                ),
            ),
            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,
                num_sample=num_sample,
                num_cls=num_map_classes,
            ),
            sampler=dict(
                type="SparsePoint3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[0.1] * 2,
                dn_trans_scale=[1.0] * 2,
                dn_rot_angle=2.5,
                dn_size_scale=0.05,
                max_dn_gt=16,
                noise_type=[1],
                dn_combination="all",
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes,
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(
                    type='LinesL1Loss',
                    loss_weight=10.0,
                    beta=0.01,
                ),
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            decoder=dict(
                type="SparsePoint3DDecoder",
                score_threshold=0.5,
            ),
            reg_weights=[1.0] * 40,
            gt_cls_key="gt_map_labels",
            gt_reg_key="gt_map_pts",
            gt_id_key="map_instance_id",
            with_instance_id=False,
            task_prefix='map',
        ),
        motion_plan_head=dict()
    ),
)

# ================== data ========================
dataset_type = "B2D3DDataset"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
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
        type='VectorizeMap',
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
            'gt_map_labels', 
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
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
            'ego_status',
            'gt_ego_fut_cmd',
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]
eval_pipeline = [
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=True,
        normalize=False,
    ),
    dict(
        type='Collect', 
        keys=[
            'vectors',
            "gt_bboxes_3d",
            "gt_labels_3d",
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks', 
            'gt_ego_fut_cmd',
            'fut_boxes'
        ],
        meta_keys=['token', 'timestamp']
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
    name_mapping=NameMapping,
    modality=input_modality,
    sample_interval=5,
    past_frames=2,
    future_frames=6,
)
eval_config = dict(
    **data_basic_config,
    ann_file=anno_root + 'b2d_infos_val.pkl',
    pipeline=eval_pipeline,
    test_mode=True,
)
data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "b2d_infos_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=5,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "b2d_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "b2d_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=5e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
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

# ================== eval ========================
eval_mode = dict(
    with_det=True,
    with_tracking=False,
    with_map=True,
    with_motion=False,
    with_planning=False,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
evaluation = dict(
    interval=num_iters_per_epoch*num_epochs,
    eval_mode=eval_mode,
)