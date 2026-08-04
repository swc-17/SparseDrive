"""NAVSIM stage 2: V1 V6 geometry fusion + V2 trajectory vocabulary.

The ResNet-50/FPN, detector, and map head initialize directly from the
completed NAVSIM stage-1 checkpoint. Planning never reads FPN features:
V6 encodes agent boxes, map polylines, ego/history, and route command, then
the V2 1024-path x 256-velocity vocabulary is pruned to 20 x 10 candidates.
All outputs are eight poses at 0.5 s in SparseDrive coordinates.
"""

_base_ = ["./sparsedrive_navsim_stage2_geoinput_full.py"]

import os


_vocab_root = os.environ.get(
    "SPARSEDRIVE_V2_VOCAB_ROOT", "data/kmeans/sparsedrive_v2"
)
_kmeans_root = os.environ.get("SPARSEDRIVE_NAVSIM_KMEANS_ROOT", "data/kmeans")
_info_root = os.environ.get("SPARSEDRIVE_NAVSIM_INFO_ROOT", "data/infos")

model = dict(
    head=dict(
        det_head=dict(
            instance_bank=dict(
                anchor=os.path.join(_kmeans_root, "kmeans_det_900_navsim.npy")
            )
        ),
        map_head=dict(
            instance_bank=dict(
                anchor=os.path.join(_kmeans_root, "kmeans_map_100_navsim.npy")
            )
        ),
        motion_plan_head=dict(
            motion_anchor=os.path.join(
                _kmeans_root, "kmeans_motion_6_navsim.npy"
            ),
            trajectory_vocab=dict(
                path_anchor=os.path.join(_vocab_root, "path_1024.npy"),
                velocity_anchor=os.path.join(_vocab_root, "velocity_256.npy"),
                trajectory_anchor=os.path.join(
                    _vocab_root, "trajectory_1024_256.npz"
                ),
                embed_dims=256,
                feedforward_dims=1024,
                num_heads=8,
                dropout=0.0,
                path_filter_num=(128, 20),
                velocity_filter_num=(64, 10),
                # Match released V2: the stored trajectory mask is carried
                # but not used to prune the composed vocabulary.
                require_full_horizon=False,
            ),
            refine_layer=dict(
                _delete_=True,
                type="MotionOnlyRefinementModule",
                embed_dims=256,
                fut_ts=12,
                fut_mode=6,
            ),
        )
    )
)

data = dict(
    train=dict(
        ann_file=os.path.join(_info_root, "navsim_infos_navtrain_train.pkl")
    ),
    val=dict(
        ann_file=os.path.join(_info_root, "navsim_infos_navtrain_val.pkl")
    ),
    test=dict(
        ann_file=os.path.join(_info_root, "navsim_infos_navtrain_val.pkl")
    ),
)

# Local default is the completed 16-GPU NAVSIM stage-1 model. Lilypad's
# load_from_s3 injection overrides this with /tmp/init_ckpt.pth.
load_from = os.environ.get(
    "SPARSEDRIVE_STAGE1_CHECKPOINT",
    "/home/tejan/lwm-rl/SparseDrive/work_dirs/navsim_stage1_eval/ckpts/"
    "navsim_stage1_full_16g_iter_21750.pth",
)
