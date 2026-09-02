# NAVSIM stage-1 traffic-light offline evaluation on the held-out navtrain
# val split: detection metrics + chamfer vector-map mAP over the 4 map
# classes (ped_crossing/divider/boundary/stop_line) + traffic-light
# association/attribute metrics (assoc P/R/F1/chamfer/success, state acc,
# TL pos L2) — see NavSim3DDataset._evaluate_tl and
# docs/pem/SPARSEDRIVE_TL_STOP_LINE.md.
#
# Model is exactly sparsedrive_navsim_stage1_tl_full_16g.py; this config
# only wires the eval_config/eval_mode plumbing.
#
# Usage (single GPU, no launcher):
#   python tools/test.py \
#       projects/configs/navsim/sparsedrive_navsim_stage1_tl_eval.py \
#       <ckpt.pth> --eval bbox map \
#       [--cfg-options data.test.max_samples=200 work_dir=...]
_base_ = ["./sparsedrive_navsim_stage1_tl_full_16g.py"]

import os

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
roi_size = (30, 60)

data_root = os.environ.get(
    "NAVSIM_BLOBS_ROOT", "/media/applied/navsim/sensor_blobs/trainval/"
)
anno_root = "data/infos/"

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

# GT-side pipeline for VectorEvaluate (simplified polylines, unnormalized).
eval_pipeline = [
    dict(
        type="VectorizeMap",
        roi_size=roi_size,
        simplify=True,
        normalize=False,
    ),
    dict(
        type="Collect",
        keys=["vectors"],
        meta_keys=["token", "timestamp"],
    ),
]

eval_config = dict(
    type="NavSim3DDataset",
    data_root=data_root,
    classes=class_names,
    map_classes=map_class_names,
    modality=input_modality,
    ann_file=anno_root + "navsim_infos_navtrain_val_tl.pkl",
    pipeline=eval_pipeline,
    test_mode=True,
)

data = dict(
    val=dict(eval_config=eval_config),
    test=dict(eval_config=eval_config),
)

# default task selection when tools/test.py gets no --eval metrics
eval_mode = dict(
    with_det=True,
    with_map=True,
    with_planning=False,
)
evaluation = dict(eval_mode=eval_mode)
