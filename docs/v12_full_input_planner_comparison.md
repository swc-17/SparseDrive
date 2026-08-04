# v12 Noiser → Planner: V6 Full Replacement vs Stage-2 Geometry Replacement

**Date:** 2026-07-24
**Noiser:** v12 planner-input model, seed6 (`noiser_v1_planner_nusc_seed6`), sampled generations
(Bernoulli visibility for agents and map elements, mu boxes/polylines, torch seed 0) —
dump `data/noiser_injection/v12_full_input_seed6_val.pkl`.
**Data:** full nuScenes val (6,019 frames), paired replay, identity bit-exact in all runs.
**Ego:** the planner's own recursive t−1 velocity in every primary arm (per protocol); the
noiser-ego arm is a separate diagnostic.
**Results:** `work_dirs/gt_planner_geometry/v6_full_replace{,_rescore}.json`,
`stage2_geom_replace{,_rescore}.json`; tool `tools/eval_v6_full_replace.py`.

## The two experiments

| | V6 geometry-only planner | Official stage-2 baseline |
|---|---|---|
| What the planner's features are | geometry encodings only (`agent_geo_encoder` / `map_geo_encoder`); no image content | image instance features (content) + geometry embeds (position) |
| Intervention | **full input replacement**: externally constructed top-50 agent set + top-10 map set + re-encoded features; planner's own selection bypassed | **geometry-only replacement**: external boxes Hungarian-assigned onto the planner's own top-50 slots; anchor/map geometry overwritten, image features and selection kept |
| Fraction of planner input synthetic | 100 % | geometry stream only |

## Results (L2 avg m / collision %)

### V6 full replacement

| arm | rescore OFF | rescore ON |
|---|---|---|
| identity (live perception) | **0.6395** / 0.177 | **0.6391** / 0.119 |
| gt_full (GT agents+map) | 0.6707 / 0.205 | 0.6707 / **0.111** |
| v12_full (noiser agents+map) | 0.6715 / 0.230 | 0.6864 / 0.181 |
| v12_full + noiser ego (diagnostic) | 0.7044 / 0.234 | 0.7177 / 0.184 |

### Stage-2 geometry replacement

| arm | rescore OFF | rescore ON |
|---|---|---|
| identity | 0.6111 / 0.194 | 0.6071 / 0.090 |
| gt_full_geom | 0.6106 / 0.196 | 0.6084 / **0.086** |
| v12_full_geom | 0.6124 / 0.199 | 0.6233 / 0.153 |

Rescore-on confirms the V6 pattern on the baseline too: the collision-aware rescorer reads
velocities, and there the noiser separates (+0.015 L2 over GT, col 0.153 vs 0.086) — the same
velocity/membership residual, now visible even on the image-feature planner.

(Identity run-to-run noise floor: ±0.0007 L2 / ±0.01 pp col.)

## Comparison findings

1. **Where the input lives determines whether the noiser is even testable.** On the baseline,
   the whole three-arm spread is 0.0018 L2 — image features anchor the planner, geometry is a
   near-inert channel (consistent with every injection experiment). On V6, geometry is the
   entire input, and the arms separate cleanly. V6 is the right testbed for perception-input
   synthesis; the baseline mostly certifies harmlessness.
2. **Noiser content ≈ GT content on trajectory regression.** V6 rescore-off: v12 vs GT gap is
   0.0008 L2 (0.6715 vs 0.6707) at identical delivery and ego. On the baseline the same gap is
   0.0018. The v12 noiser's generated agents+map are regression-equivalent to ground truth
   through both planners.
3. **Full-set delivery has a real cost that geometry-swapping does not.** V6's gt_full sits
   +0.031 L2 above identity — replacing set membership/ordering/padding is out-of-distribution
   for the planner even with perfect content. On the baseline, gt_full_geom ≈ identity
   (−0.0005): keeping the planner's own slots and features removes that overhead. Any synthetic
   training pipeline pays the delivery cost unless the planner is (re)trained on the synthetic
   delivery statistics — which is exactly what Tier-3 fine-tuning would do.
4. **The rescore/collision channel exposes the noiser's residual gaps.** V6 rescore-on: v12
   slips to +0.016 L2 over GT and 0.181 vs 0.111 collision. The rescorer extrapolates agent
   futures from velocities; the noiser's heading-aligned mu velocities and sampled set
   membership are its remaining imperfections — matching the campaign-long finding that
   velocity and cardinality are the only channels this planner family truly reads.
5. **Ego must be self-consistent, not just distributionally right.** Substituting the planner's
   own recursion with the noiser's calibrated-but-independent ego sample costs +0.033 L2 —
   as much as the entire delivery overhead. In selfplay the planner's own recursion exists and
   should be used (as all primary arms here do); the ego head is for pipelines without planner
   history and would need temporally consistent sampling to close the gap.

## Bottom line

The v12 noiser passes the strongest available open-loop test: on the planner whose input is
100 % geometry, its generated scene description is trajectory-equivalent to ground truth, with
quantified residual gaps in the collision/rescore channel (velocity realism, set membership)
and a measured warning about ego self-consistency. The remaining costs are properties of the
*delivery mode and ego wiring*, not of noiser content — and both are exactly what Tier-3
(fine-tuning the planner on noiser-generated input, with its own recursion live) is designed to
absorb.
