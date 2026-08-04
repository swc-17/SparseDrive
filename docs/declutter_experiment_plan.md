# Decluttering SparseDrive, v2: a geometric-only planner over frozen perception

**Author:** Tejan Karmali · **Revised:** 2026-07-21 · **Status:** implementation complete, G-matrix training starting

> **Revision note.** v1 of this plan (full 5-variant × 3-seed retraining matrix,
> archived in git history) was superseded after a day of runs surfaced two
> insights: (a) what the planner plausibly needs from all the temporal/auxiliary
> machinery is **agent/ego kinematics**, and (b) with perception frozen, the
> planner question can be answered ~100× cheaper from a cached geometric
> abstraction. v1 side-findings retained: temporal stage1 recipe reproduces the
> paper; single-frame stage1 diverges at the temporal LR (needs lr/2) — evidence
> that temporal propagation also stabilizes detector optimization.

## 1. Reframed question

The planner needs to know **how agents are moving** (kinematics) and **where it
can drive** (geometry). SparseDrive feeds it image-derived instance features,
temporal feature queues, and a CNN ego feature — none of which are obviously
necessary if the geometric abstraction (boxes + velocities + map polylines +
ego kinematic state) carries the needed information. Tracking is one candidate
*mechanism* for supplying kinematics; forecasting supervision is another.

**Design rule:** the planner input is exclusively geometric — no image instance
features (agent, map) and no CNN ego feature. Image features appear only in the
explicit control (C1).

## 2. Setup

- **Perception:** official released stage1 checkpoint (temporal det+map;
  NDS 0.526 / map mAP 0.569), **frozen**. Its detector supplies boxes with
  velocity estimates and temporal tracking IDs.
- **Geometry cache:** one sequential inference pass over train (28,130 frames)
  and val (6,019) storing per frame: top-50 boxes (11-dim: position, size, yaw,
  velocity) + confidence + tracking ID (+ fp16 image features for C1 only),
  top-10 map polylines (40-dim), ego status (CAN), lidar→global transform.
  ~36 KB/frame, ~1.2 GB total. Image-feature leakage into geometric variants is
  impossible by construction: they never load those arrays.
- **Planner:** ~4 M-param attention decoder. Agent tokens = geometric content
  MLP + the same `SparseBox3DEncoder` positional encoding SparseDrive uses; map
  tokens = `SparsePoint3DEncoder`; ego token = kinematic anchor + status MLP;
  18 plan-mode queries from the k-means plan anchors. Output heads and the
  planning loss (PlanningTarget sampler, focal cls 0.5 / L1 reg 1.0 / L1
  status 1.0) mirror `MotionPlanningHead` exactly; evaluation reuses the repo's
  `planning_eval` (L2 @1/2/3 s + collision, rescore-free).
- Training: 10 epochs, bs 64, AdamW 5e-4 cosine, ~40 min/run on one local GPU.
  3 seeds per variant.

## 3. Hypotheses (all one factor from G0)

Noise floor σ = per-variant std over 3 seeds; a claimed effect requires
|Δ| > 2σ on planning L2/collision.

| ID | Variant | Claim | Falsified if |
|---|---|---|---|
| **C1** | image features as planner input (same frozen stage1, same everything else) | control — defines the image-feature reference | — |
| **G0** | geometric only | geometry+kinematics suffice: G0 ≈ C1 | G0 worse than C1 by > 2σ |
| **G1** | G0 − agent velocities | kinematic input is the load-bearing channel: large degradation | velocity removal ≈ free |
| **G2** | G0 + ID-matched 4-frame box history | tracking supplies kinematics beyond instantaneous velocity (acceleration/maneuver structure) | history adds nothing |
| **G3** | G0 + self-supervised forecast auxiliary (predict each tracked agent's future displacement from cache — tracking as *teacher*, no extra labels) | learning kinematics explicitly improves planning | no improvement |
| **G4** | G0 with ego velocity removed (`g4_none`) vs measured CAN velocity (G0 default) | ego kinematic state is critical (subject to open-loop-shortcut caveat → NavSim check on the winner) | equivalent |
| **G5** | G0 − map polylines | map geometry matters (more than in image planners — nothing dense to compensate) | dropping map ≈ free |

External reference row (not a controlled comparison — different training
regime): released stage2 planner, rescore-off, same eval: **L2 avg 0.611 /
collision 0.196 %** (and rescore-on collision benefit is a separable
inference-time effect, measured in v1 as the H5 bound).

## 4. Interpretation map

- **G0 ≈ C1** → appearance features are unnecessary for planning given good
  geometry: the decluttered SparseDrive is stage1 + a 4 M geometric planner.
- **G1 large, G2/G3 positive** → kinematics is the currency; tracking (input
  or teacher) is a justified component, everything else in
  InstanceQueue/InstanceBank-temporal is not needed *for planning*.
- **G0 ≪ C1** → image features carry planning-relevant signal beyond geometry
  (e.g. intent cues); quantify the gap and identify what C1 attends to.
- **G4 dominant** → ego kinematics dwarf scene understanding on open-loop
  nuScenes (known critique) — report with the closed-loop check, not as a
  positive result.

## 5. Deliverables

1. Table: {C1, G0–G5} × {L2 1/2/3s/avg, collision avg} mean ± std (3 seeds),
   with falsification verdicts.
2. Curves in wandb (`research/sparsedrive-declutter`, runs `geo_<variant>_seed<k>`).
3. If G0 holds: the lean model definition (frozen stage1 + geometric planner)
   and its total-parameter/latency comparison vs full SparseDrive.
4. NavSim closed-loop check on the headline pair (G0 winner vs C1).

## 6. Reproduction

```bash
# geometry cache (once, ~40 min local GPU)
PYTHONPATH=. python tools/create_geometry_cache.py --split val
PYTHONPATH=. python tools/create_geometry_cache.py --split train

# any variant
PYTHONPATH=. python geo_planner/train.py --variant g0 --seed 0
```

Code: `tools/create_geometry_cache.py`, `geo_planner/{dataset,model,train}.py`,
configs `projects/configs/declutter/stage1_official_cache_{train,val}.py`.
Official stage1 checkpoint: GitHub release v1.0 (`ckpt/sparsedrive_stage1.pth`).

---

# 7. RESULTS (2026-07-21, all 21 runs, 3 seeds/variant)

| Variant | L2 avg (m) | Collision avg (%) | Verdict (2σ) |
|---|---|---|---|
| C1 image features | 0.361 ± 0.010 | 0.048 ± 0.004 | control |
| G0 geometric only | 0.367 ± 0.004 | 0.053 ± 0.003 | **confirmed** — matches C1 |
| G1 − agent velocity | 0.367 ± 0.006 | 0.045 ± 0.009 | **falsified** — removal free |
| G2 + tracking history | 0.362 ± 0.005 | 0.056 ± 0.009 | **falsified** — ≤1σ |
| G3 + forecast auxiliary | 0.362 ± 0.010 | 0.056 ± 0.006 | **falsified** — ≤1σ |
| G4 − ego kinematics | 1.511 ± 0.015 | 0.426 ± 0.030 | **confirmed** — 4× L2, 8× col |
| G5 − map | 0.373 ± 0.005 | 0.099 ± 0.004 | **confirmed for collisions** — 2× col (≈17σ), L2 flat |
| released stage2 (rescore-off) | 0.611 | 0.196 | external reference |

## Conclusions

1. **Appearance features are unnecessary for the planner** (G0 ≈ C1): the
   geometric abstraction of frozen stage1 carries everything open-loop planning
   uses.
2. **Ego kinematic state dominates** (G4): an order of magnitude above every
   other factor. Known open-loop shortcut; reported as such.
3. **Map geometry is specifically a collision-avoidance channel** (G5): no L2
   effect, 2× collision effect — a clean metric dissociation.
4. **The agent channel is inert under open-loop evaluation** (G1/G2/G3 all
   ≤1σ): velocities, ID-matched history, and forecast supervision are all
   removable without measurable open-loop cost. Interpreted methodologically:
   open-loop nuScenes metrics cannot detect agent understanding — the
   original kinematics hypotheses are open-loop-undecidable, and the decisive
   experiment is closed-loop (NavSim), which is the recommended next step.
5. **Lean model** (open-loop): frozen stage1 + 4M geometric planner beats the
   released stage2 planner (0.37 vs 0.61 L2; 0.05% vs 0.20% collisions),
   with the measured-ego-velocity caveat.

Artifacts: checkpoints/results at s3://research-datasets-chicago/users/tejan/
sparsedrive/declutter/geo_runs/; wandb runs geo_<variant>_seed<k>.

---

# 8. V6: geometry-only planner inputs INSIDE the unchanged baseline (2026-07-21)

## 8.1 Motivation

Does the planner need image-derived instance features at all, or is the
geometric state of the scene (agent boxes + velocities, map polylines, ego
kinematics) sufficient? V6 asks the cleanest version of this question by
changing nothing but the planner's input representation:

> Take the stock stage2 baseline — 901-query planner, temporal InstanceQueue,
> motion co-training, recursive predicted ego status, joint training from the
> official stage1 checkpoint — and change **nothing except the planner's input
> features**: replace every image-derived instance feature the planner consumes
> with an MLP encoding of the corresponding geometry. Does planning quality
> survive?

## 8.2 What changed vs the baseline (and what did not)

| Planner input | Baseline | V6 |
|---|---|---|
| Agent features (GNN/temp-GNN K/V, motion queries) | image instance features | `agent_geo_encoder(det anchor)` — 11-d box (pos, size, yaw, vel) |
| Map features (cross-GNN K/V) | image map instance features | `map_geo_encoder(polyline)` — 40-d coords |
| Ego feature | CNN over front-cam feature map | encoded ego anchor only (kinematic state); the CNN is **not built** |
| Temporal queue contents | past image features | past *geometry encodings* (substitution happens before the queue caches) |
| Everything else | — | **identical**: predicted-ego-status recursion, motion co-training + rescore capability, losses, schedule, 901 queries, joint backbone training |

Implementation: `geometric_inputs=True` on `MotionPlanningHead` substitutes
`det_output/map_output["instance_feature"]` at the top of `forward`, before any
consumer (including `InstanceQueue.prepare_motion`, so agent/ego history is
geometry-derived too). `InstanceQueue(use_cam_ego_feature=False)` skips
*constructing* the ego CNN — this both removes the image path structurally and
avoids DDP unused-parameter failures (the first launch crashed with "Expected
to mark a variable ready only once" under `find_unused_parameters=True`; the
fix is parameter hygiene, not DDP workarounds). A constructor assert couples
the two flags. Selection nuance, stated for honesty: *which* top-50 agents /
top-10 polylines enter the planner is still chosen by detection/map confidence
scores, and the boxes/polylines themselves come from the camera-based
perception stack — "geometry-only" refers to the planner's feature inputs.

Runtime verification (from the actual seed0 training log): dumped config shows
`geometric_inputs=True`, `use_cam_ego_feature=False`; `agent_geo_encoder` /
`map_geo_encoder` weights registered fresh; zero occurrences of
`ego_feature_encoder` (the ego CNN did not exist in the trained model).

## 8.3 Protocol

- 3 seeds (0/1/2), init from the official stage1 checkpoint, 10 epochs
  (5860 iters, 8×A100 each), withmap pkls, `train_pyfocal.py`.
- Eval on FULL nuScenes val by per-experiment monitors (valgroup): every
  checkpoint → det/map/plan metrics → wandb `val_stage2_v6_geoinput_seed<k>`.
- Headline: **rescore-off** (baseline reference 0.611 L2 / 0.196% col).
  Rescore-ON also evaluated (V6 co-trains motion, so the collision-aware
  rescore is available) → wandb `..._seed<k>_rescore`.
- Configs `projects/configs/declutter/stage2_v6_geoinput.py`; workloads
  seed0-jlkt06 / seed1-3x4j8l / seed2-04je8n (+ valgroups 7lgedt rescore-off,
  f0j873 rescore-on).

## 8.4 Results (final checkpoints, epoch 10 / iter 5860, full val)

| | det mAP | map mAP | plan L2 avg (m) | plan col (%) |
|---|---|---|---|---|
| released stage2, rescore-off | 0.414 | 0.565 | 0.611 | 0.196 |
| **V6 geometry-only, rescore-off (mean, 3 seeds)** | 0.410 | 0.558 | **0.647 ± 0.011** | 0.207 ± 0.038 |
| V6 geometry-only, rescore-on (mean, 3 seeds) | 0.411 | 0.558 | 0.645 ± 0.011 | **0.140 ± 0.008** |

Per-seed, rescore-off: L2 0.641 / 0.640 / 0.660, col 0.186 / 0.183 / 0.251%.
Per-seed, rescore-on: L2 0.640 / 0.637 / 0.658, col 0.131 / 0.142 / 0.147%.

**Headline (rescore-off): geometry-only planner inputs cost +0.036 m L2
(0.647 vs 0.611, ~6% relative, ≈3σ of seed spread) and are neutral on
collisions (0.207 ± 0.038 vs 0.196).** Perception is at baseline parity
(det 0.410 vs 0.414, map 0.558 vs 0.565), confirming the intervention only
touched the planner. The collision-aware rescore does not change L2 (0.645
vs 0.647) but cuts collisions by a third (0.140 vs 0.207%) — the motion
head trained normally alongside the geometric planner inputs. Caveat: the
0.611 reference is the *released* checkpoint, trained by the authors; the
regime-matched comparison is the V6c control below.

## 8.5 Control (V6c)

To separate "input representation" from "our training regime", a control
reproduces the stock stage2 (planner keeps its image instance features) under
the exact same conditions as V6 — official stage1 init, 10 epochs, withmap
pkls, same trainer; the ONLY diff vs `stage2_v6_geoinput.py` is the absence of
the model override. Config `stage2_v6c_ctrl.py`, workload
`stage2_v6c_ctrl_seed0-i83pup` (single seed), rescore-off + rescore-on evals.
The honest comparison is then V6 (mean of 3 seeds) vs V6c under identical
training, with the released checkpoint as an external reference.

Results (final checkpoint, iter 5860, full val):

| | rescore | det mAP | map mAP | plan L2 avg (m) | plan col (%) |
|---|---|---|---|---|---|
| V6c control (image feats, seed0) | off | 0.411 | 0.557 | **0.601** | 0.173 |
| V6 geometry-only (mean, 3 seeds) | off | 0.410 | 0.558 | 0.647 ± 0.011 | 0.207 ± 0.038 |
| V6c control (image feats, seed0) | on | 0.411 | 0.557 | 0.601 | 0.095 |
| V6 geometry-only (mean, 3 seeds) | on | 0.411 | 0.558 | 0.645 ± 0.011 | 0.140 ± 0.008 |
| released stage2 | off | 0.414 | 0.565 | 0.611 | 0.196 |

The control reproduces the released baseline (0.601 vs 0.611 — our regime is
sound), so the regime-matched cost of geometry-only planner inputs is
**+0.046 m L2 (0.647 vs 0.601, ~8% relative)**. Collisions: rescore-off the
difference (0.207 vs 0.173%) is within V6's seed spread; rescore-on the
control is cleaner (0.095 vs 0.140%), suggesting image features help the
motion-based mode rescoring more than they help the trajectory itself. The
control is a single seed; V6's seed spread (±0.011 L2) bounds the noise
scale. Overall: inside the full unchanged baseline pipeline, replacing all
image-derived planner inputs with geometry encodings costs ~8% open-loop L2
and roughly nothing in perception — the planner's knowledge of the scene is
almost entirely geometric.
