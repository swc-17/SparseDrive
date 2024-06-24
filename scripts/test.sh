bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    ckpt/sparsedrive_stage2.pth \
    8 \
    --deterministic \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl