"""NAVSIM stage 2 vocab model + V2 PDM metric heads (3-epoch fine-tune).

Intended initialization is the finished 10-epoch vocab checkpoint
(work_dirs/navsim_stage2_vocab_full_8g/iter_29010.pth via load_from_s3);
only the eight metric heads start fresh.

Extends the vocabulary planner with released V2's per-candidate metric
supervision: eight EPDMS sub-metric BCE heads on the composed 200-candidate
embedding, ground truth computed live by simulating each candidate against
the sample's NAVSIM metric cache (metric_cache.pkl, v2 caching pipeline).
Final trajectory selection switches from the imitation score to V2's
combined metric score.

Requires:
  - metric caches laid out {metric_cache_root}/{log}/unknown/{token}/
    metric_cache.pkl (s3://.../navsim_cache/navtrain/metric_cache_worker*/)
  - the navsim v2 devkit importable in the scoring workers
    (SPARSEDRIVE_NAVSIM_DEVKIT_ROOT, see pdm_metric_scorer.py).
"""

_base_ = ["./sparsedrive_navsim_stage2_vocab_full.py"]

import os

metric_cache_root = os.environ.get(
    "SPARSEDRIVE_METRIC_CACHE_ROOT", "data/metric_cache_navtrain"
)

model = dict(
    head=dict(
        motion_plan_head=dict(
            trajectory_vocab=dict(
                metrics=(
                    "no_at_fault_collisions",
                    "drivable_area_compliance",
                    "driving_direction_compliance",
                    "traffic_light_compliance",
                    "time_to_collision_within_bound",
                    "ego_progress",
                    "lane_keeping",
                    "history_comfort",
                ),
                metric_loss_weight=5.0,
            ),
        )
    )
)

# The base train pipeline, with metric_cache_path added to Collect. mmcv
# merges lists wholesale, so the pipeline is restated in full (kept in sync
# with sparsedrive_navsim_stage2_geoinput.py).
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
class_names = {{_base_.class_names}}
roi_size = {{_base_.roi_size}}
num_sample = {{_base_.num_sample}}
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
            "metric_cache_path",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        metric_cache_root=metric_cache_root,
    ),
)
