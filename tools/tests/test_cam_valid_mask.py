"""Unit test: DeformableFeatureAggregation per-view valid masking.

Contract (combined NAVSIM + nuScenes training): padded camera views must
contribute EXACTLY zero to the aggregated instance features — enforced by
zeroing the softmaxed sampling weights of views with cam_valid_mask == 0,
not by learned rejection.

Checks:
1. weights of masked views are exactly zero;
2. corrupting the masked views' feature maps (and projection matrices) does
   not change the output bitwise — pure-torch sampling path;
3. same invariance through the fused CUDA deformable_aggregation path
   (skipped when CUDA/DAF unavailable);
4. with use_cam_valid_mask=False (default) the same corruption DOES change
   the output (test sensitivity / bit-identical default behavior);
5. PadMultiViewImage pads image/projection lists and emits the mask.

Run:  PYTHONPATH=. python tools/tests/test_cam_valid_mask.py
"""

import numpy as np
import torch

from projects.mmdet3d_plugin.models.blocks import DeformableFeatureAggregation
from projects.mmdet3d_plugin.datasets.pipelines.transform import (
    PadMultiViewImage,
)

BS, NUM_ANCHOR, NUM_CAMS, NUM_LEVELS, EMBED = 2, 12, 8, 4, 256
STRIDES = [4, 8, 16, 32]
W, H = 704, 256


def build_dfa(use_cam_valid_mask, use_deformable_func):
    torch.manual_seed(0)
    dfa = DeformableFeatureAggregation(
        embed_dims=EMBED,
        num_groups=8,
        num_levels=NUM_LEVELS,
        num_cams=NUM_CAMS,
        attn_drop=0.15,
        use_deformable_func=use_deformable_func,
        use_camera_embed=True,
        residual_mode="cat",
        use_cam_valid_mask=use_cam_valid_mask,
        kps_generator=dict(
            type="SparseBox3DKeyPointsGenerator",
            num_learnable_pts=6,
            fix_scale=[
                [0, 0, 0],
                [0.45, 0, 0],
                [-0.45, 0, 0],
                [0, 0.45, 0],
                [0, -0.45, 0],
                [0, 0, 0.45],
                [0, 0, -0.45],
            ],
        ),
    )
    dfa.eval()  # attn_drop off -> deterministic
    return dfa


def make_inputs(device):
    g = torch.Generator().manual_seed(1)
    instance_feature = torch.randn(BS, NUM_ANCHOR, EMBED, generator=g)
    anchor = torch.randn(BS, NUM_ANCHOR, 11, generator=g)
    anchor[..., :3] = anchor[..., :3] * 20
    anchor[..., 3:6] = 1.0
    anchor_embed = torch.randn(BS, NUM_ANCHOR, EMBED, generator=g)
    proj = torch.randn(BS, NUM_CAMS, 4, 4, generator=g)
    proj[:, :, 3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    image_wh = torch.tensor([[[W, H]]] * BS, dtype=torch.float32).repeat(
        1, NUM_CAMS, 1
    )
    feature_maps = [
        torch.randn(BS, NUM_CAMS, EMBED, H // s, W // s, generator=g)
        for s in STRIDES
    ]
    cam_valid_mask = torch.ones(BS, NUM_CAMS)
    cam_valid_mask[:, 6:] = 0.0  # last two views padded
    out = dict(
        instance_feature=instance_feature,
        anchor=anchor,
        anchor_embed=anchor_embed,
        feature_maps=feature_maps,
        proj=proj,
        image_wh=image_wh,
        cam_valid_mask=cam_valid_mask,
    )
    return {
        k: ([t.to(device) for t in v] if isinstance(v, list) else v.to(device))
        for k, v in out.items()
    }


def corrupt(feature_maps, proj):
    """Garbage in the padded views' features + projections."""
    fms = [fm.clone() for fm in feature_maps]
    for fm in fms:
        fm[:, 6:] = 1e6
    proj_bad = proj.clone()
    proj_bad[:, 6:] = proj_bad[:, 6:] * -3.7 + 11.0
    return fms, proj_bad


def run(dfa, x, feature_maps, proj, fused):
    metas = dict(
        projection_mat=proj,
        image_wh=x["image_wh"],
        cam_valid_mask=x["cam_valid_mask"],
    )
    if fused:
        from projects.mmdet3d_plugin.ops import feature_maps_format

        fms = feature_maps_format(feature_maps)
    else:
        fms = feature_maps
    with torch.no_grad():
        return dfa(
            x["instance_feature"], x["anchor"], x["anchor_embed"], fms, metas
        )


def main():
    device = "cpu"

    # ---- 1+2: pure-torch path, masked weights zero + output invariance
    dfa = build_dfa(use_cam_valid_mask=True, use_deformable_func=False)
    x = make_inputs(device)
    metas = dict(
        projection_mat=x["proj"],
        image_wh=x["image_wh"],
        cam_valid_mask=x["cam_valid_mask"],
    )
    w = dfa._get_weights(x["instance_feature"], x["anchor_embed"], metas)
    assert w.shape[2] == NUM_CAMS
    assert torch.all(w[:, :, 6:] == 0), "masked-view weights must be zero"
    assert torch.all(w[:, :, :6].sum() > 0)
    print("[1] masked-view sampling weights are exactly zero")

    out_clean = run(dfa, x, x["feature_maps"], x["proj"], fused=False)
    fms_bad, proj_bad = corrupt(x["feature_maps"], x["proj"])
    out_bad = run(dfa, x, fms_bad, proj_bad, fused=False)
    assert torch.equal(out_clean, out_bad), (
        "padded-view corruption changed the output (torch path)"
    )
    print("[2] torch path: output bitwise-invariant to padded-view content")

    # ---- 4: default config is sensitive to the same corruption
    dfa_off = build_dfa(use_cam_valid_mask=False, use_deformable_func=False)
    out_off_clean = run(dfa_off, x, x["feature_maps"], x["proj"], fused=False)
    out_off_bad = run(dfa_off, x, fms_bad, proj_bad, fused=False)
    assert not torch.equal(out_off_clean, out_off_bad), (
        "corruption should change the output when masking is off"
    )
    # and masking off == masking on with an all-ones mask (bit-identical
    # default behavior)
    x_ones = dict(x, cam_valid_mask=torch.ones_like(x["cam_valid_mask"]))
    out_on_ones = run(dfa, x_ones, x["feature_maps"], x["proj"], fused=False)
    assert torch.equal(out_off_clean, out_on_ones)
    print("[4] default path unchanged; all-ones mask == masking disabled")

    # ---- 3: fused CUDA deformable_aggregation path
    if torch.cuda.is_available():
        try:
            from projects.mmdet3d_plugin.ops import (  # noqa: F401
                deformable_aggregation_function,
            )

            dev = "cuda"
            dfa_c = build_dfa(
                use_cam_valid_mask=True, use_deformable_func=True
            ).to(dev)
            xc = make_inputs(dev)
            # the fused kernel is run-to-run nondeterministic (atomicAdd
            # accumulation order, ~3e-8): measure that floor first, then
            # require the corruption-induced diff to sit at the same floor
            out_a = run(dfa_c, xc, xc["feature_maps"], xc["proj"], fused=True)
            out_b = run(dfa_c, xc, xc["feature_maps"], xc["proj"], fused=True)
            rr = (out_a - out_b).abs().max().item()
            fms_bad_c, proj_bad_c = corrupt(xc["feature_maps"], xc["proj"])
            out_c_bad = run(dfa_c, xc, fms_bad_c, proj_bad_c, fused=True)
            diff = (out_a - out_c_bad).abs().max().item()
            tol = max(4 * rr, 1e-7)
            assert diff <= tol, (
                f"padded-view corruption changed the CUDA DAF output beyond "
                f"kernel nondeterminism: diff={diff} vs run-to-run={rr}"
            )
            assert torch.isfinite(out_a).all()
            assert torch.isfinite(out_c_bad).all()
            # corrupting with 1e6-scale garbage while diffs stay at the 1e-8
            # nondeterminism floor => contribution is exactly zero
            print(f"[3] CUDA DAF path: invariant to padded-view content "
                  f"(diff {diff:.1e} <= nondeterminism floor {rr:.1e}x4)")
        except ImportError:
            print("[3] SKIPPED (deformable_aggregation ext not built)")
    else:
        print("[3] SKIPPED (no CUDA)")

    # ---- 5: PadMultiViewImage
    pad = PadMultiViewImage(num_cams=8)
    results = dict(
        img=[np.full((900, 1600, 3), 7.0, dtype=np.float32)] * 6,
        lidar2img=[np.eye(4)] * 6,
        lidar2cam=[np.eye(4)] * 6,
        cam_intrinsic=[np.diag([1000.0, 1000.0, 1.0])] * 6,
    )
    results = pad(results)
    assert len(results["img"]) == 8
    assert np.all(results["img"][6] == 0) and np.all(results["img"][7] == 0)
    assert np.all(results["lidar2img"][6] == 0)
    assert np.array_equal(
        results["cam_valid_mask"],
        np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.float32),
    )
    # no-op case (already 8 views) emits an all-ones mask
    results8 = pad(
        dict(img=[np.zeros((2, 2, 3))] * 8, lidar2img=[np.eye(4)] * 8)
    )
    assert np.all(results8["cam_valid_mask"] == 1)
    print("[5] PadMultiViewImage pads views and emits cam_valid_mask")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
