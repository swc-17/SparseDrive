# Matched-GT planner geometry sanity check

Date: 2026-07-21

## Outcome

The matched predictions contain material geometry error, while replacing that
geometry with GT changes planning-L2 less than observed default run-to-run
variation. This remains true when Hungarian is restricted to the planner's
top-50 boxes/top-10 maps, which forces every feasible GT assignment into a
slot the planner consumes. This supports the hypothesis that the planner is
robust or insensitive to matched current-frame geometry error. It does not
cover unmatched false positives/misses or a stateful GT-injected rollout.

## Provenance and controls

- Checkpoint: official released SparseDrive-S stage-2
  `ckpt/sparsedrive_stage2.pth`
- Checkpoint SHA-256:
  `a9786bd3398907666ef436b287b465d6de8c424467413648e4614e0b884db7ad`
- Model: resolved model config is identical to
  `projects/configs/sparsedrive_small_stage2.py`.
- Runtime override: collision rescoring is disabled for every row.
- Data: `data/infos/nuscenes_infos_val_withmap.pkl`, 6,019 validation frames;
  official planning metrics use the same 5,119 complete-future frames in every
  row.
- Identity planner replay is bit-exact: maximum plan-logit, plan-regression,
  and final-trajectory deltas are all `0.0`.
- The latest default result, L2 `0.6099` / collision `0.186%`, agrees with the
  documented no-rescore reference, L2 `0.611` / collision `0.196%`.
- For comparison, the normal rescore-on released result is L2 `0.61` /
  collision `0.097%`; the existing local evaluation is L2 `0.6074` /
  collision `0.098%`.

## Intervention

The detector and map predictions are matched with the model's own training
Hungarian samplers over all 900 object queries and all 100 map queries. This is
not instance-ID matching or a hybrid. Predicted tracker IDs are used by the
temporal queue, but never by this GT assignment.

The object cost is focal classification (`weight=2`) plus encoded-box L1
(`weight=0.25`). Under the stage-2 config, the regression weights use XYZ and
log-size; yaw and velocity have zero assignment weight. The map cost is focal
classification (`weight=1`) plus normalized 20-point line Smooth-L1
(`weight=10`, `beta=0.01`), selecting the best GT direction/permutation. There
is no distance or IoU rejection threshold. The global assignments are then
intersected with the unchanged planner top-50/top-10 selections.

There are two object interventions. `GT bbox` replaces only XYZ, log-size, and
sin/cos-yaw (8 dimensions), retaining all predicted velocity. `GT bbox + vel`
also replaces GT VX/VY (10 supervised dimensions); predicted VZ is retained
because nuScenes does not supervise it. The corresponding selected K/V
position, all-query position, current temporal position, and trajectory-anchor
query are recomputed. For a matched map query, the sampler-selected GT polyline
permutation replaces the top-10 map position embedding. Predicted features,
logits, confidence ordering, slot count, and all unmatched geometry remain
fixed.

Only the default branch updates the temporal cache. Each oracle is therefore a
per-frame counterfactual conditioned on identical default model-predicted
history, not a stateful GT-injected rollout.

Coverage across all frames (assignment coverage, not thresholded precision):

- boxes: 102,919 / 300,950 selected slots matched (34.20%), covering 76.86%
  of processed object GT
- map: 42,140 / 60,190 selected slots matched (70.01%), covering 75.82% of
  map GT

## Matched geometry error

All errors are in the current LiDAR frame. Map error is the mean pointwise L2
over the 20 aligned points of each Hungarian-selected GT permutation.

| Matched geometry error | Mean | Median | P90 | Max |
|---|---:|---:|---:|---:|
| Bbox center XY L2 (m) | 0.641 | 0.394 | 1.484 | 19.713 |
| Bbox velocity XY L2 (m/s) | 0.300 | 0.074 | 0.792 | 58.512 |
| Map aligned mean-point L2 (m) | 1.676 | 0.953 | 3.580 | 27.061 |

## Full-validation results

All collision values are official `obj_box_col`, in percent.

| Condition | L2 1 s | L2 2 s | L2 3 s | L2 avg | Collision avg | Delta L2 | Delta collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default, no GT | 0.2967 | 0.5777 | 0.9551 | 0.6099 | 0.186% | — | — |
| Identity replay | 0.2967 | 0.5777 | 0.9551 | 0.6099 | 0.186% | 0.000000 | 0.000000 pp |
| Matched GT bbox geometry | 0.2970 | 0.5782 | 0.9558 | 0.6103 | 0.185% | +0.000454 | -0.001085 pp |
| Matched GT bbox geometry + velocity | 0.2968 | 0.5779 | 0.9553 | 0.6100 | 0.187% | +0.000138 | +0.000543 pp |
| Matched GT map | 0.2967 | 0.5780 | 0.9555 | 0.6101 | 0.187% | +0.000227 | +0.001085 pp |
| Matched GT bbox geometry + map | 0.2967 | 0.5778 | 0.9551 | 0.6099 | 0.185% | +0.000036 | -0.001085 pp |
| Matched GT bbox geometry + velocity + map | 0.2967 | 0.5778 | 0.9550 | 0.6098 | 0.186% | -0.000025 | -0.000543 pp |

## Selected-only Hungarian follow-up

The follow-up first selects exactly the 50 object and 10 map queries consumed
by the planner, then runs the same official Hungarian samplers on only those
queries. Hungarian is one-to-one and has no rejection gate, so it assigns
`min(K, number of GT)` pairs per frame, not all K slots when a frame has fewer
than K annotations. The full run exactly equals that deterministic ceiling.

| Input | Slots | Processed GT | Matches | Slot coverage | GT recall | Predicted/GT class agreement |
|---|---:|---:|---:|---:|---:|---:|
| Boxes, K=50 | 300,950 | 133,901 | 128,273 | 42.62% | 95.80% | 88.57% |
| Maps, K=10 | 60,190 | 55,582 | 46,659 | 77.52% | 83.95% | 95.42% |

There are at most 101 processed object GTs and 32 map GT polylines in a frame.
Only 364 frames exceed 50 boxes (5,628 GTs beyond capacity), while 1,919 frames
exceed 10 map lines (8,923 beyond capacity). Unmatched selected slots in all
other frames are background predictions; matching them many-to-one would no
longer be Hungarian.

Restricting the candidate pool exposes forced poor pairs that global matching
can assign to non-selected queries:

| Selected-only matched error | Mean | Median | P90 | Max |
|---|---:|---:|---:|---:|
| Bbox center XY L2 (m) | 2.821 | 0.610 | 7.477 | 93.290 |
| Bbox velocity XY L2 (m/s) | 0.445 | 0.080 | 1.156 | 67.931 |
| Map aligned mean-point L2 (m) | 3.045 | 1.094 | 7.507 | 61.747 |

The paired, no-rescore planning results are:

| Condition | L2 1 s | L2 2 s | L2 3 s | L2 avg | Collision avg | Delta L2 | Delta collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default, no GT | 0.2969 | 0.5781 | 0.9557 | 0.6102 | 0.192% | — | — |
| Identity replay | 0.2969 | 0.5781 | 0.9557 | 0.6102 | 0.192% | 0.000000 | 0.000000 pp |
| Top-50 GT bbox geometry | 0.2972 | 0.5786 | 0.9561 | 0.6106 | 0.195% | +0.000407 | +0.002713 pp |
| Top-50 GT bbox geometry + velocity | 0.2975 | 0.5788 | 0.9559 | 0.6108 | 0.196% | +0.000515 | +0.003798 pp |
| Top-10 GT map | 0.2967 | 0.5778 | 0.9552 | 0.6099 | 0.194% | -0.000348 | +0.002171 pp |
| Top-50 GT bbox geometry + top-10 GT map | 0.2972 | 0.5785 | 0.9556 | 0.6104 | 0.206% | +0.000191 | +0.014109 pp |
| Top-50 GT bbox geometry + velocity + top-10 GT map | 0.2974 | 0.5787 | 0.9558 | 0.6106 | 0.202% | +0.000379 | +0.010310 pp |

## Interpretation

The selected-only follow-up is the stronger box test: box center error rises to
`0.61 m` median / `7.48 m` p90, and 95.8% of processed GTs are injected into
the actual top-50 slots. Yet box-only planning L2 changes by at most
`0.000515 m`, below the observed repeated-default variation of about
`0.00111 m`; box-only collision changes by at most `0.00380` percentage points,
also below its `0.00814`-point default variation. This is evidence that the
planner is robust or insensitive to these matched current-frame box geometry
errors, potentially because predicted features, ego context, and history
compensate for them.

The combined selected-only box+map collision increase reaches `0.01411`
percentage points, above that observed default variation, so the broader claim
should not be extended to collision robustness under the combined forced
intervention.

It is not a general robustness proof. Hungarian has no rejection threshold, so
extreme maxima can be forced poor assignments; the median/p90 are more useful.
The global-match run corrects 34.20% of selected box slots and the selected-only
run corrects 42.62%; unmatched false positives and capacity-limited GT remain.
Predicted features are unchanged, and every oracle uses the default
model-predicted temporal cache. A stateful per-arm rollout and an oracle that
removes false positives/inserts missed GT are separate tests.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/tejan/miniconda3/envs/sparsedrive310/bin/python \
  tools/eval_gt_planner_geometry.py \
  projects/configs/sparsedrive_small_stage2_val_eval.py \
  ckpt/sparsedrive_stage2.pth \
  --workers 4 \
  --out work_dirs/gt_planner_geometry/val_full_6arms_matcherr.json
```

The machine-readable report is
`work_dirs/gt_planner_geometry/val_full_6arms_matcherr.json`.

For the selected-only follow-up, add `--match-topk` and write to
`work_dirs/gt_planner_geometry/val_full_6arms_top50match.json`.
