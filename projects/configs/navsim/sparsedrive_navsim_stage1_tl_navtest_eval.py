# NAVSIM stage-1 TL offline evaluation on the **navtest** benchmark split
# (12,146 tokens / converter v1.3 with stop_line + map_tl_annos).
#
# Same perception metrics as sparsedrive_navsim_stage1_tl_eval.py (det +
# 4-class chamfer map mAP + TL attribute metrics), but on navtest with
# sensor blobs under sensor_blobs/test/ (not trainval/).
#
# Usage (single GPU, no launcher):
#   python tools/test.py \
#       projects/configs/navsim/sparsedrive_navsim_stage1_tl_navtest_eval.py \
#       <ckpt.pth> --eval bbox map \
#       [--cfg-options data.test.max_samples=200 work_dir=...]
_base_ = ["./sparsedrive_navsim_stage1_tl_eval.py"]

import os

navtest_ann_file = "data/infos/navsim_infos_navtest_tl.pkl"

# navtest sensor blobs: sensor_blobs/test/, not sensor_blobs/trainval/.
data_root = os.environ.get(
    "NAVSIM_BLOBS_ROOT", "/media/applied/navsim/sensor_blobs/test/"
)

_split = dict(data_root=data_root, ann_file=navtest_ann_file)

data = dict(
    val=dict(eval_config=_split, **_split),
    test=dict(eval_config=_split, **_split),
)
