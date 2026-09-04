# Association + SparseDriveV2 DN: det 5/3, map 5/0.
_base_ = ["./sparsedrive_b2d_stage1_tl.py"]

model = dict(
    head=dict(
        det_head=dict(sampler=dict(num_dn_groups=5, num_temp_dn_groups=3)),
        map_head=dict(
            sampler=dict(
                num_dn_groups=5,
                num_temp_dn_groups=0,
                dn_noise_scale=[0.1] * 2,
                max_dn_gt=16,
                dn_combination="all",
            ),
        ),
    )
)
