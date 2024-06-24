## stage1
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage1.py \
   8 \
   --deterministic

## stage2
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage2.py \
   8 \
   --deterministic