# Denoising arm: the only recipe delta from the released B2D stage-1 config,
# whose det/map heads use these same group counts. Requires the int64 widening
# in ops/src/deformable_aggregation_cuda.cu (num_kernels overflows int32 at
# batch_size 8 once DN anchors are added) and the out-of-range target handling
# in tools/train_pyfocal.py (DN marks negative queries with a -3 sentinel).
_base_ = ["../sparsedrive_b2d_stage1.py"]

model = dict(
    head=dict(
        det_head=dict(sampler=dict(num_dn_groups=5, num_temp_dn_groups=3)),
        map_head=dict(sampler=dict(num_dn_groups=5, num_temp_dn_groups=0)),
    )
)
