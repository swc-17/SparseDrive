# A/B/C data-regime matrix — metrics report

**Updated:** 2026-07-28. **Architecture (identical in every A/B/C row):**
original SparseDrive port with the V6 geometry-only planner
(`geometric_inputs=True`, `use_rescore=False`), 7-class/8-step unified
contract. The A/B/C experimental axis is training data only; the imgfeat row
varies exactly one thing vs A (the planner's input modality). Full
provenance: `docs/NAVSIM_PORT_REPORT.md`; per-metric values: W&B
`research/sparsedrive-navsim`, group `abc_matrix_eval`, and
`work_dirs/navsim_eval/abc_matrix_metrics.csv`.

| Training data | Model | Checkpoint |
| --- | --- | --- |
| NAVSIM navtrain only (1,067-log split) | A | `navsim_stage2_geoinput_full_16g/iter_8703` |
| NAVSIM navtrain only, image-feature planner (non-V6 twin of A) | imgfeat | `navsim_stage2_imgfeat_full_16g/iter_8703` |
| NAVSIM + nuScenes, naive 1:1 per-sample | B-v1 | `combined_stage2_geoinput_seed0/iter_8800` |
| NAVSIM + nuScenes, nuScenes 4x upweight + balanced anchors | B-v2 | `combined_stage2_v2_seed0/iter_8800` |
| nuScenes only (unified contract) | C | `nusc_unified_stage2_geoinput_seed0/iter_5860` |

Throughout: **bold** = best per column among comparable rows; † = zero-shot
(model never trained on the eval domain); *pending* = eval queued/running as
of this update.

---

## 1. Planning comparisons

### 1.1 NAVSIM (12,146 navtest tokens / 5,912 navhard tokens, pinned caches)

| Training data | Model | navtest v1 PDMS ↑ | navtest v2 EPDMS ↑ | navhard EPDMS ↑ (s1 / s2) |
| --- | --- | --- | --- | --- |
| NAVSIM only | A | **0.7620** | **0.7634** | **0.1917** (0.526 / 0.346) |
| NAVSIM only (image-feature planner) | imgfeat | 0.8060 | 0.8051 | 0.1977 (0.581 / 0.347) |
| NAVSIM + nuScenes (1:1) | B-v1 | 0.7493 | 0.7501 | 0.1712 (0.499 / 0.311) |
| NAVSIM + nuScenes (4x nusc) | B-v2 | 0.7239 | 0.7181 | 0.1426 (0.470 / 0.289) |
| nuScenes only | C † | 0.4569 | 0.4830 | — |
| NAVSIM navtrain (1,067-log split) | sd1.5 metric § | 0.8613 | 0.8581 | 0.3297 (0.687 / 0.471) |
| NAVSIM navtrain (all 1,192 logs) | SparseDriveV2 ‡ | 0.9222 | 0.9037 | 0.4046 |
| Reference: Human / const-velocity | — | 0.9455 / 0.2065 | — | — |

**Key message:** Specialists win at home — every step of added nuScenes
exposure costs NAVSIM planning (A 0.762 > B-v1 0.749 > B-v2 0.724), and the
cost compounds closed-loop on navhard. Trajectory-vocabulary planners
(sd1.5 0.861, SparseDriveV2 0.922) clearly outscore the full-stack V6 port
on open-loop NAVSIM benchmarks.

navtest v1 sub-metrics (per-token means, from the pinned CSVs):

| | NC | DAC | EP | TTC | comfort | DDC |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.950 | 0.878 | 0.724 | 0.873 | 1.000 | 0.961 |
| B-v1 | 0.945 | 0.866 | 0.715 | 0.868 | 1.000 | 0.950 |
| B-v2 | 0.931 | 0.853 | 0.697 | 0.844 | 1.000 | 0.939 |
| C † | 0.804 | 0.728 | 0.431 | 0.653 | 0.999 | 0.906 |

**Key message:** C's zero-shot collapse is perception-driven — ego progress
and TTC crater (0.43/0.65) while comfort and driving-direction stay normal:
the planner behaves sanely on inputs that are wrong for the domain.

Protocol notes. navtest v2 EPDMS is the single-stage pseudo-closed-loop
protocol behind SparseDriveV2's published "90.3" (bug-fix scorer,
`navsim_agent/score_epdms_navtest_v2.py`, pinned
`SparseDriveV2/exp/metric_cache_navtestv2`). Gate: re-scoring SparseDriveV2's
own saved predictions through our script reproduced its pinned
0.9037266761321409 exactly. The v1/v2 columns agree on every ordering; v2 is
slightly kinder to C (+0.026) and slightly harsher on B-v2 (−0.006). navhard
EPDMS is the two-stage protocol (stage two re-plans from the model's own
stage-one endpoints inside synthetic renders); keep all three protocols in
separate columns, never mixed.

### 1.2 nuScenes val (6-step comparable subset; collision = obj_box_col)

| Training data | Model | L2 avg (m) ↓ | collision (%) ↓ |
| --- | --- | --- | --- |
| NAVSIM only | A † | 1.125 | 1.042 |
| NAVSIM + nuScenes (1:1) | B-v1 | 0.731 | 0.202 |
| NAVSIM + nuScenes (4x nusc) | B-v2 | 0.722 | **0.157** |
| nuScenes only | C | **0.656** | 0.195 |

**Key message:** Combined training closes most of the 71% zero-shot L2 gap;
B-v2 even beats the nuScenes specialist on collision (0.157% vs 0.195%),
and its remaining ~10% L2 gap tracks its residual detection-precision gap.

Per-horizon values (0.5–4.0 s, 6-step + 8-step tails) are in the CSV/W&B.
C's numbers passed the pre-registered reproduction gate against baseline
seeds (0.639–0.660 m / 0.177–0.254%).

---

## 2. Perception comparisons

Both domains use the identical unified contract (7 classes, 8 cameras with
zero-attention padding on nuScenes, same three map semantics), so cells are
directly comparable within each column. All rows are **stage-2** checkpoints
— the same checkpoints as the planning tables. For reference, A's stage-1
checkpoint scored det mAP 0.3517 / NDS 0.4103 / map 0.8503 on navtrain-val:
stage-2 joint training mildly erodes perception (−0.010 det, −0.008 map).

### 2.1 Detection (mAP over center-distance thresholds; NDS-analog)

NAVSIM navtrain-val (10,435 samples, 125 held-out logs):

| Training data | Model | det mAP ↑ | NDS ↑ | mATE ↓ | mAVE ↓ |
| --- | --- | --- | --- | --- | --- |
| NAVSIM only | A | 0.3415 | 0.3981 | 0.590 | 0.718 |
| NAVSIM only (image-feature planner) | imgfeat | 0.3438 | 0.4100 | 0.574 | 0.682 |
| NAVSIM + nuScenes (1:1) | B-v1 | 0.3522 | 0.4264 | 0.557 | 0.587 |
| NAVSIM + nuScenes (4x nusc) | B-v2 | **0.3672** | **0.4423** | 0.550 | 0.575 |
| nuScenes only | C † | *pending* | | | |
| NAVSIM navtrain (1,067-log split) | sd1.5 metric § | 0.2503 | 0.2729 | 0.672 | 1.139 |

**Key message:** Mixing HELPS home-domain detection — both combined models
beat specialist A on NAVSIM, and the most exposure-balanced (B-v2) is best
(0.3672 vs 0.3415; velocity error 0.575 vs 0.718).

nuScenes val:

| Training data | Model | det mAP ↑ | NDS ↑ | mATE ↓ | mAVE ↓ |
| --- | --- | --- | --- | --- | --- |
| NAVSIM only | A † | *pending* | | | |
| NAVSIM + nuScenes (1:1) | B-v1 | 0.3498 | 0.4294 | 0.616 | 0.376 |
| NAVSIM + nuScenes (4x nusc) | B-v2 | 0.4232 | 0.4932 | 0.518 | 0.312 |
| nuScenes only | C | **0.4687** | **0.5438** | 0.463 | 0.288 |

**Key message:** The nuScenes-side detection gap is exposure-driven and
recoverable (B-v1 0.3498 → B-v2 0.4232, approaching C's 0.4687). Detection
transfers positively in both directions — the opposite of mapping (2.2).

### 2.2 Online mapping (chamfer mAP over ped_crossing / divider / boundary)

NAVSIM navtrain-val:

| Training data | Model | map mAP ↑ | ped_crossing | divider | boundary |
| --- | --- | --- | --- | --- | --- |
| NAVSIM only | A | 0.8425 | 0.822 | 0.890 | 0.816 |
| NAVSIM only (image-feature planner) | imgfeat | **0.8439** | 0.823 | 0.891 | 0.817 |
| NAVSIM + nuScenes (1:1) | B-v1 | 0.2113 | 0.099 | 0.386 | 0.149 |
| NAVSIM + nuScenes (4x nusc) | B-v2 | 0.2110 | 0.088 | 0.400 | 0.145 |
| nuScenes only | C † | *pending* | | | |
| NAVSIM navtrain (1,067-log split) | sd1.5 metric § | 0.7514 | 0.713 | 0.849 | 0.692 |

**Key message:** Combined models' NAVSIM maps collapse to ~0.21 (vs A's
0.84) regardless of sample weighting — exposure cannot explain this;
training one shared map head on two map conventions breaks the
majority-domain side too.

nuScenes val:

| Training data | Model | map mAP ↑ | ped_crossing | divider | boundary |
| --- | --- | --- | --- | --- | --- |
| NAVSIM only | A † | *pending* | | | |
| NAVSIM + nuScenes (1:1) | B-v1 | 0.1988 | 0.085 | 0.319 | 0.193 |
| NAVSIM + nuScenes (4x nusc) | B-v2 | **0.5679** | 0.531 | 0.600 | 0.572 |
| nuScenes only | C | 0.5521 | 0.495 | 0.582 | 0.579 |

**Key message:** Upweighting flips which convention the head serves —
nuScenes maps recover to above-specialist (0.568 > C's 0.552) while NAVSIM
maps stay exactly collapsed. One shared map head, one convention at a time.

Mapping read — **discriminator resolved 2026-07-28: hard trade / one
convention at a time.** B-v1's map head collapsed on **both** domains
(NAVSIM 0.2113 / nuScenes 0.1988) even though NAVSIM was ~85% of its
training samples — exposure deficit cannot explain the NAVSIM side. B-v2's
4x-nuScenes finetune restored nuScenes maps to above-C (0.568) while its
NAVSIM map stayed exactly collapsed (0.2110 ≈ B-v1's 0.2113, same per-class
signature). Conclusion: the shared map head serves only one map convention
well at a time. Mixing degrades both; upweighting one side recovers that
side only, without further damaging the other. The 2026-07-27 "exposure,
not label conflict" conclusion — drawn from nuScenes alone — was wrong as a
general claim: exposure governs *which* convention the head serves, but the
two-convention conflict itself is real. A per-domain map head (or a domain
token) is the obvious architectural fix to test.

Robustness note: B-v1 loses only 0.013 navtest PDMS vs A while its NAVSIM
map mAP is 4x worse (0.21 vs 0.84) and its DAC only drops 0.878 → 0.866.
The V6 planner is far more tolerant of degraded maps than the "map polylines
are direct planner inputs" framing suggested — detection quality appears to
carry more of the planning signal.

---

## 3. Findings (cross-cutting)

1. **Specialists win at home; zero-shot planning transfer fails
   symmetrically.** A leads every NAVSIM planning metric; C leads nuScenes
   L2 and detection. Across domains both collapse (A: 1.125 m / 1.04%
   collisions on nuScenes; C: 0.4569 PDMS, barely 2x the CV baseline). C's
   navtest sub-metrics locate the failure in perception-dependent terms —
   ego progress 0.431 and TTC 0.653 vs ~0.70/~0.85 for NAVSIM-trained models
   — while comfort (0.999) and driving direction (0.906) stay near-normal:
   the planner is sane, its inputs are wrong for the domain. The pending
   zero-shot perception cells (A-on-nuScenes, C-on-NAVSIM) will quantify how
   much of the collapse is perception.

2. **Naive mixing (B-v1): detection is robust, mapping is the fragile
   head.** NAVSIM detection at parity with A; nuScenes detection degraded by
   exposure (recoverable, see B-v2). Mapping collapsed on both domains
   (Section 2.2) — the one place the 1:1 mix broke something that more
   exposure on one side did not obviously cause.

3. **Mapping is convention-bound; detection is not (resolved 2026-07-28).**
   The 4x upweight restored nuScenes maps to above-C (0.199 → 0.568) while
   B-v2's NAVSIM map stayed collapsed at B-v1's level (0.211) — the shared
   map head serves one map convention at a time, and exposure picks which.
   Detection shows the opposite: mixing helps both domains (B-v2 best on
   NAVSIM, near-C on nuScenes). This dissociation also explains why the
   combined models plan competitively on NAVSIM despite 0.21 maps: the V6
   planner leans mostly on detections. Obvious next experiment: per-domain
   map heads or a domain-conditioning token.

4. **The rebalance is a Pareto trade on planning.** B-v2 pays ~0.025 navtest
   PDMS vs B-v1, spread uniformly across sub-metrics, and the cost compounds
   closed-loop (navhard 0.1712 → 0.1426, a bigger relative drop than
   open-loop). No variant dominates: A best if only NAVSIM matters, C if
   only nuScenes, B-v2 the best single checkpoint if both must work.

5. **Collision-vs-L2 dissociation in the combined models.** Both B variants
   beat or match C on collisions while trailing on L2 — the geometry-only
   planner trades trajectory-matching precision for obstacle avoidance when
   trained on more diverse geometry, consistent with the balanced anchors
   reducing worst-case nuScenes mode coverage from 17.3 m to 3.3 m.

---

## Footnotes

§ sd1.5 (2026-07-28), NOT part of the A/B/C data axis — a different
architecture: V1 NAVSIM stage-1 backbone/det/map + V2-style
trajectory-vocabulary planner (geometry attention, no image reads), worktree
`.wtc/worktrees/sd1.5/SparseDrive`, branch `research/sd15-v2-vocab`. Same
1,067-log navtrain split and pinned caches/harness as A. Checkpoint:
`navsim_stage2_vocab_metric_16g/iter_8703` — a 3-epoch fine-tune of the base
vocab planner (`navsim_stage2_vocab_full_8g/iter_29010`, 10 epochs, imitation
ranking: 0.8451 PDMS / 0.8402 EPDMS, in the CSV but not tabled) that adds
8 EPDMS BCE heads; decode re-ranks the vocabulary by predicted metric score,
worth +0.016 PDMS / +0.018 EPDMS over imitation ranking.
W&B runs `eval_sd15_{vocab_iter29010,metric_iter8703}` in the same
`abc_matrix_eval` group; sub-metrics in the CSV.

‡ External comparison, NOT part of the A/B/C matrix: SparseDriveV2's released
checkpoints (planning-only trajectory-vocabulary model, ResNet-34, no
detection/map/motion stack — a different architecture), evaluated by us on the
same pinned caches/harness. Its navhard EPDMS is 0.3753 with the shipped
scorer, 0.4046 with the bug-fix scorer (quoted above). Data asymmetry: it
trained on all 1,192 navtrain logs vs our 1,067 (~12% more), with no held-out
navtrain validation split. Its published "90.3" is a navtest-v2 EPDMS number
(reproduced: 0.9037), not a navhard result.

## Open items

- imgfeat row COMPLETE (2026-07-29): 0.8060 / 0.8051 / navhard 0.1977
  (s1 0.581 beats A's 0.526 on real frames; s2 0.347 == A's 0.346 on
  synthetic renders — the image-feature edge vanishes on renders);
  det 0.3438 / map 0.8439 (== A, as expected: same perception stack).
- Still open: C-zero-shot NAVSIM det/map; A-zero-shot nuScenes det/map
  (both were externally stopped 2026-07-28 and not re-queued).
- Untested: a stage-2-only retrain on B-v1's stage-1 (~4 h) to isolate the
  anchor contribution from the exposure contribution.
- An intermediate sample weight (e.g. [1,2]) would probe the PDMS↔nuScenes
  Pareto front between B-v1 and B-v2.
