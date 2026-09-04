# B2D stage-1 + traffic-light association on existing TrafficLight map class 4.
# Keeps SparseDriveV2's 6-class map taxonomy and kmeans_map_100.npy.
_base_ = ["../sparsedrive_b2d_stage1.py"]

traffic_light_label = 4
roi_size = (30, 60)

model = dict(
    head=dict(
        map_head=dict(
            type="SparseMapHead",
            roi_size=roi_size,
            gt_tl_key="gt_map_tl",
            refine_layer=dict(with_tl_branch=True),
            loss_tl_offset=dict(type="L1Loss", loss_weight=2.0),
            loss_tl_state=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
        ),
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
class_names = [
    "car", "van", "truck", "bicycle", "traffic_sign",
    "traffic_cone", "traffic_light", "pedestrian", "others",
]
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(class_names)),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type="VectorizeMap",
        roi_size=roi_size,
        simplify=False,
        normalize=False,
        sample_num=20,
        permute=True,
        tl_label=traffic_light_label,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img", "timestamp", "projection_mat", "image_wh", "focal",
            "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
            "gt_map_tl",
            "gt_agent_fut_trajs", "gt_agent_fut_masks",
            "gt_ego_fut_trajs", "gt_ego_fut_masks", "gt_ego_fut_cmd",
            "ego_status",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]
data = dict(train=dict(pipeline=train_pipeline))
