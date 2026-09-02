# NAVSIM stage-1 with maps + traffic lights: adds the 4th map class
# "stop_line" (traffic-light-controlled stop bars, label 3) and a
# per-instance traffic-light attribute branch on the map head:
#   - TL BEV position, regressed as an offset from the predicted polyline
#     mean (permutation-invariant; supervised only where the map carries a
#     bulb position);
#   - TL state (green/red), supervised per frame from the log's
#     per-lane-connector status (unknown states are loss-masked).
# Association TL <-> stop line is implicit: the attributes live on the
# stop-line instance matched by the standard Hungarian assignment.
#
# Requires infos with the v1.3 converter (stop_line class + map_tl_annos)
# and map anchors regenerated on 4-class polylines:
#   python tools/data_converter/navsim_converter.py --split mini \
#       --filter navmini --output data/infos/navsim_infos_navmini_tl.pkl
#   python tools/kmeans/kmeans_map_navsim.py \
#       --info data/infos/navsim_infos_navmini_tl.pkl \
#       --out data/kmeans/kmeans_map_100_navsim_tl.npy
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
map_class_names = [
    "ped_crossing",
    "divider",
    "boundary",
    "stop_line",
]
num_map_classes = len(map_class_names)
stop_line_label = map_class_names.index("stop_line")
roi_size = (30, 60)
num_sample = 20

model = dict(
    head=dict(
        map_head=dict(
            type="SparseMapHead",
            roi_size=roi_size,
            gt_tl_key="gt_map_tl",
            instance_bank=dict(
                # regenerated on the 4-class (incl. stop_line) polylines
                anchor="data/kmeans/kmeans_map_100_navsim_tl.npy",
            ),
            refine_layer=dict(
                num_cls=num_map_classes,
                with_tl_branch=True,
            ),
            sampler=dict(num_cls=num_map_classes),
            loss_tl_offset=dict(type="L1Loss", loss_weight=2.0),
            loss_tl_state=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
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

anno_root = "data/infos/"

data = dict(
    train=dict(
        ann_file=anno_root + "navsim_infos_navmini_tl.pkl",
        map_classes=map_class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        ann_file=anno_root + "navsim_infos_navmini_tl.pkl",
        map_classes=map_class_names,
    ),
    test=dict(
        ann_file=anno_root + "navsim_infos_navmini_tl.pkl",
        map_classes=map_class_names,
    ),
)
