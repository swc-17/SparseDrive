# NAVSIM stage-1 offline evaluation on the held-out navtrain val split
# (10,435 samples / 125 logs): devkit-free nuScenes-style detection metrics
# (mAP over center-distance thresholds 0.5/1/2/4 m + ATE/ASE/AOE/AVE, 55 m
# range, 7 NAVSIM classes) and the repo's chamfer vector-map mAP
# (0.5/1.0/1.5 m over ped_crossing/divider/boundary).
#
# Model is exactly the stage-1 full det+maps model
# (sparsedrive_navsim_stage1_full_16g.py -> sparsedrive_navsim_stage1_maps.py);
# this config only wires the eval_config/eval_mode plumbing that
# NavSim3DDataset.evaluate needs.
#
# Usage (single GPU, no launcher):
#   python tools/test.py projects/configs/navsim/sparsedrive_navsim_stage1_eval.py \
#       <ckpt.pth> --eval bbox map \
#       [--cfg-options data.test.max_samples=200 work_dir=...]
# or the wrapper: tools/eval_navsim_stage1.sh <s3-or-local ckpt> [--subset N]
_base_ = ["./sparsedrive_navsim_stage1_full_16g.py"]

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

# GT-side pipeline for VectorEvaluate (mirrors the nuScenes stage-1
# eval_pipeline's map part: simplified polylines, unnormalized coords).
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
    ann_file=anno_root + "navsim_infos_navtrain_val.pkl",
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
