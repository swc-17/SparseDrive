# NAVSIM stage-2 FULL training with the ORIGINAL image-feature planner
# (non-V6 ablation): identical to sparsedrive_navsim_stage2_geoinput_full.py
# — same data, schedule, anchors, stage-1 init (iter_21750 via load_from_s3)
# — except the planner consumes image/ego features again:
#   motion_plan_head.geometric_inputs=False (planner sees instance features)
#   instance_queue.use_cam_ego_feature=True (front-cam ego CNN is built;
#   its weights are fresh — stage-1 has no motion/planning head).
# Purpose: isolate what the V6 geometry-only restriction costs/buys on
# NAVSIM (compare against Model A = the V6 twin, iter_8703).
_base_ = ["./sparsedrive_navsim_stage2_geoinput_full.py"]

model = dict(
    head=dict(
        motion_plan_head=dict(
            geometric_inputs=False,
            instance_queue=dict(
                use_cam_ego_feature=True,
            ),
        ),
    ),
)
