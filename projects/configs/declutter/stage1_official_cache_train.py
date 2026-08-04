# Frozen official stage1 inference over TRAIN for the geometry cache.
# Test pipeline (no augmentation), sequential scene order.
_base_ = ["../sparsedrive_small_stage1.py"]

data = dict(
    test=dict(
        ann_file="data/infos/nuscenes_infos_train.pkl",
        eval_config=dict(ann_file="data/infos/nuscenes_infos_train.pkl"),
    ),
)
