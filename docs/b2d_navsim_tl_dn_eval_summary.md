# B2D + NAVSIM stage-1 TL/DN eval summary

Last updated: 2026-09-04.

Covers eval results gathered across one session: the B2D stage-1 TL/DN/TL+DN
training runs to completion, the NAVSIM stage-1 TL+DN training run to
completion, its navtest and B2D-zero-shot evals, a non-DN NAVSIM comparison,
and an SDv1 range-AP (mAP vs. distance cutoff) comparison. Background on the
zero-shot protocol and the original SDv1 baseline numbers: `docs/b2d_val_navsim_sparse4d_zero_shot.md`
and `docs/b2d_stage1_val_metrics.md`.

## 1. B2D stage-1, final checkpoints (bs=1)

All evaluated at `eval_samples_per_gpu=1` (required for correct temporal
`InstanceBank` continuity — see `feedback_eval_samples_per_gpu_1` memory),
official val (12,806 frames / 50 clips), `--eval bbox map`.

| Arm | Checkpoint | Job | mAP | NDS | mAP_normal |
|---|---|---|---:|---:|---:|
| SDv1 (baseline) | iter_73360 (final, 20 ep) | `sparsedrive_b2d_stage1_v1_rangeap-txymsh` ("all" pass) | **0.6941** | **0.7425** | — |
| TL | iter_18340 (final, 20 ep) | `sparsedrive_b2d_stage1_tl_eval-z0bawy` | **0.7074** | **0.7529** | 0.674 |
| DN | iter_18340 (final, 20 ep) | `sparsedrive_b2d_stage1_dn_eval-xp0d95` | **0.7566** | **0.8050** | 0.788 |
| TL+DN | iter_18340 (final, 20 ep) | `sparsedrive_b2d_stage1_tl_dn_eval-h3w3o2` | **0.7328** | **0.7861** | 0.726 |

**Correction to `docs/b2d_stage1_val_metrics.md`**: that doc's headline SDv1
number (mAP 0.4130 / NDS 0.5092) was computed at `eval_samples_per_gpu=4`
(hardcoded in the eval job that produced it, `sparsedrive_b2d_stage1_eval-iqf9b7`).
Batching 4 consecutive frames from one clip breaks the temporal `InstanceBank`,
which expects sequential single-frame calls — the same failure mode already
documented on a mini-val smoke test (mAP 0.204 at bs=1 → 0.057 at bs=4, a
~3.6x depression). The real bs=1 SDv1 number is **0.6941 mAP**, not 0.4130 —
roughly on par with the newer TL/DN/TL+DN arms, not far behind them as the
stale number implied.

DN is the dominant lever among the three new arms (TL 0.707 < TL+DN 0.733 <
DN 0.757 mAP); TL+DN does not simply add TL's and DN's individual gains.

**Caveat**: `tl/assoc_recall`, `tl/assoc_f1`, `tl/state_acc`, `tl/pos_l2` etc.
are not reported here — this branch (`research/navsim-sparse4d-on-b2d`) has no
such code under `projects/mmdet3d_plugin/datasets/evaluation/` (verified by
grep). The earlier jobs that did print those keys (`sparsedrive_b2d_stage1_tl_eval-azesy1`,
`..._dn_eval-73ug4e`, `..._tl_dn_eval-z9jjpe`, all at iter ~13-17k) must have
run from a different code snapshot/branch than this one. For reference, their
(non-final-checkpoint, mixed-config) numbers:

| Arm | Checkpoint | mAP | NDS | tl/assoc_f1 | tl/state_acc |
|---|---:|---:|---:|---:|---:|
| DN | iter 8253/18340 | 0.582 | 0.658 | 0.850 | NaN (no TL head) |
| TL | iter 13755/18340 | 0.665 | 0.720 | 0.780 | 0.935 |
| TL+DN | iter 8253/18340 | 0.571 | 0.654 | 0.775 | 0.866 |

## 2. B2D training recipe: SDv1 → TL / DN / TL+DN

Config inheritance (all under `projects/configs/b2d/`, copied into this
checkout from the `sd15-b2d` worktree this session — they weren't present in
`research/navsim-sparse4d-on-b2d` before):

- `sparsedrive_b2d_stage1.py` — thin `_base_` shim to `../sparsedrive_b2d_stage1.py` (SDv1's own config, unchanged)
- `sparsedrive_b2d_stage1_tl.py` (`_base_` → above) — adds only a map-branch TL offset+state head (`with_tl_branch=True`, `loss_tl_offset`, `loss_tl_state`)
- `sparsedrive_b2d_stage1_dn.py` (`_base_` → base, **not** tl.py) — adds `num_dn_groups=5/3` (det) and `5/0` (map) denoising sampler groups only; genuinely no TL head
- `sparsedrive_b2d_stage1_tl_dn.py` (`_base_` → tl.py) — same DN groups, layered on top of the TL head

Compute change from SDv1 to all three new arms (launch commands captured from
`lilypad workload logs sparsedrive_b2d_stage1_tl_dn-if3u9r`):

| | SDv1 | TL / DN / TL+DN |
|---|---|---|
| GPUs | 8×A100 | 16×A100 |
| samples/GPU | 8 | 16 |
| global batch | 64 | **256 (4x)** |
| epochs | 20 | 20 (unchanged) |
| max_iters | 73,360 | 18,340 (4x fewer steps, same epoch budget) |
| LR | 5e-4 | 5e-4 (unchanged — `--cfg-options optimizer.lr=0.0005`, same as SDv1's default) |

No LR scaling was applied for the 4x batch increase, and no comment in the
repo discusses whether that was considered.

## 3. NAVSIM stage-1 TL+DN navtest, checkpoint progression

Job `sd_navsim_stage1_tl_dn_navtest_eval-*`, checkpoint from
`s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/work_dirs/navsim_stage1_tl_16x16_dn_map5/`.

| iter | % of 10,860 | mAP | NDS | mAP_normal | stop_line_recall | tl/assoc_f1 | tl/state_acc | tl/pos_l2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1810 | 17% | 0.131 | 0.183 | 0.271 | 0.615 | 0.212 | 0.836 | 4.11 |
| 5430 | 50% | 0.244 | 0.275 | 0.472 | 0.824 | 0.362 | 0.854 | 2.20 |
| 10860 (final) | 100% | **0.300** | **0.362** | **0.553** | **0.873** | 0.440 | 0.863 | 1.64 |

Final checkpoint note: the true final save is `iter_10860.pth` (no
`latest_iter_` prefix), which the entrypoint's `_latest_iter_key` auto-detect
does **not** match (it only globs `latest_iter_*.pth`). Both navtest and the
B2D-zero-shot eval below were relaunched with `checkpoint_s3` pinned
explicitly after first landing on the stale 95%-checkpoint by accident.

## 4. NAVSIM DN ablation: TL-only vs. TL+DN

| | TL only (no DN) | TL+DN |
|---|---|---|
| Checkpoint | iter 21750 (final, 30 ep), run `navsim_stage1_tl_full_16g_lr2e4` | iter 10860 (final, 30 ep), run `navsim_stage1_tl_16x16_dn_map5` |
| Job | `sd_navsim_stage1_tl_nodn_navtest_eval-r7r2qj` | `sd_navsim_stage1_tl_dn_navtest_eval-v5d61s` |
| **Det mAP** | **0.055** | **0.300** (~5.5x) |
| Det NDS | 0.147 | 0.362 |
| mAP_normal | 0.574 | 0.553 |
| stop_line_recall | 0.883 | 0.873 |
| tl/assoc_f1 | 0.458 | 0.440 |
| tl/state_acc | 0.866 | 0.863 |

DN gives an ~5.5x detection-mAP boost with map/TL metrics essentially
unchanged (if anything marginally higher without DN) — consistent with DN
(denoising query training) doing its documented job of stabilizing bipartite
matching for the detection head specifically, not a general quality lever.

Recipe diff (config files: `projects/configs/navsim/sparsedrive_navsim_stage1_tl_full_16g.py`,
`..._dn.py`, `..._bs16_dn.py`):

| | TL only | TL+DN |
|---|---|---|
| GPUs × samples/GPU | 16 × 8 | 16 × 16 |
| global batch | 128 | 256 (2x) |
| epochs | 30 | 30 (unchanged) |
| LR | 2e-4 | 2e-4 (unchanged) |

Unlike B2D, the LR here was **explicitly investigated**: the DN base config's
comment records `"maps-only survived at 4e-4... TL at 4e-4 NaN'd even with
clip=1. Halve LR (same trick that fixed maps 5.7e-4 -> 4e-4)"`, settling on
2e-4 for both variants. B2D's LR was simply carried over from SDv1 with no
equivalent documented investigation, despite a larger (4x vs. 2x) batch jump.

## 5. NAVSIM → B2D zero-shot, day/dusk (35 clips, 2 Hz, virtual camera)

Same checkpoints as §3, job `sparsedrive_b2dval_navsim_tl_dn_vk_daydusk-*`,
reusing the round-2 protocol from `docs/b2d_val_navsim_sparse4d_zero_shot.md`.

| iter | mAP | NDS |
|---:|---:|---:|
| 1810 | 0.034 | 0.082 |
| 5430 | 0.074 | 0.112 |
| 10860 (final) | **0.100** | **0.149** |

Steady improvement with training, but still far below the in-domain B2D
DN/TL+DN numbers in §1 — expected for a zero-shot cross-domain transfer.

## 6. B2D SDv1 range-AP: all-scenes vs. day/dusk vs. zero-shot day/dusk

Computed via `tools_b2d/range_ap_b2d.py` (copied into this checkout this
session), run on-cluster through a new `rangeap_entrypoint_fn` in
`lilypad_entrypoint_b2d.py` (job `sparsedrive_b2d_sdv1_rangeap_compute-rv9hc6`)
to avoid downloading multi-GB `results.pkl` files locally.

| Range (m) | SDv1 all scenes (50 clips) | SDv1 day/dusk only (35 clips) | Zero-shot NAVSIM, day/dusk |
|---:|---:|---:|---:|
| 5 | 0.781 | 0.946 | 0.178 |
| 10 | 0.812 | 0.853 | 0.247 |
| 20 | 0.888 | 0.890 | 0.185 |
| 30 | 0.826 | 0.833 | 0.114 |
| 40 | 0.756 | 0.743 | — |
| 50 | 0.704 | 0.667 | 0.071 |
| 55 | — | — | 0.068 |

Day/dusk restriction consistently helps SDv1 (night is the hard subset
dragging down the all-scenes curve at long range), while the zero-shot model
sits 4-10x below SDv1 at every range cutoff. Interactive chart with hover/data
table: https://claude.ai/code/artifact/c49c4e89-35e7-49d0-9b51-dda6648e3bd7

## 7. Reproduction

| What | Job ID | S3 result path |
|---|---|---|
| SDv1 full-val (bs1) | `sparsedrive_b2d_stage1_v1_rangeap-txymsh` | `s3://research-datasets-chicago/users/tejan/sparsedrive_b2d/work_dirs/b2d_stage1_v1_rangeap_iter73360/all/` |
| SDv1 day/dusk (bs1) | `sparsedrive_b2d_stage1_v1_rangeap-aucr5g` | `.../b2d_stage1_v1_rangeap_iter73360/daydusk/` |
| SDv1 range-AP compute | `sparsedrive_b2d_sdv1_rangeap_compute-rv9hc6` | `.../b2d_stage1_v1_rangeap_iter73360/{all,daydusk}_rangeap.json` |
| TL final | `sparsedrive_b2d_stage1_tl_eval-z0bawy` | `.../b2d_stage1_tl_a100x16_bs16_eval_iter18340/` |
| DN final | `sparsedrive_b2d_stage1_dn_eval-xp0d95` | `.../b2d_stage1_dn_a100x16_bs16_eval_iter18340/` |
| TL+DN final | `sparsedrive_b2d_stage1_tl_dn_eval-h3w3o2` | `.../b2d_stage1_tl_dn_a100x16_bs16_eval_iter18340/` |
| NAVSIM navtest, TL+DN | `sd_navsim_stage1_tl_dn_navtest_eval-v5d61s` | `s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/work_dirs/navsim_stage1_tl_16x16_dn_map5_navtest_iter10860/` |
| NAVSIM navtest, TL only | `sd_navsim_stage1_tl_nodn_navtest_eval-r7r2qj` | `.../navsim_stage1_tl_full_16g_lr2e4_navtest_iter21750/` |
| NAVSIM→B2D zero-shot, TL+DN | `sparsedrive_b2dval_navsim_tl_dn_vk_daydusk-cfggt8` | `s3://research-datasets-chicago/users/tejan/sparsedrive_b2d/work_dirs/b2dval_navsim_tl_dn_vk_daydusk_iter10860/` |

Launch configs: `lilypad_config/eval_b2d_stage1_{tl_now,dn_final,tl_dn_final,rangeap}.yaml`,
`lilypad_config/rangeap_b2d_sdv1.yaml`, `lilypad_config/navsim_eval/eval_navtest_tl_{dn,nodn}.yaml`,
`lilypad_config/eval_b2dval_navsim_tl_dn_vk_daydusk.yaml`.

Two bugs hit and fixed along the way:
- **Auto-detect checkpoint bug**: `_latest_iter_key` in `lilypad_entrypoint_b2d.py`
  (and its NAVSIM equivalent) only globs `latest_iter_*.pth`, missing a
  training run's true final checkpoint when it's saved as plain `iter_N.pth`.
  Worked around by pinning `checkpoint_s3` explicitly once training completed.
- **OCI chunked-upload rejection**: `boto3>=1.36`'s `upload_file`/`upload_fileobj`
  default to `aws-chunked` transfer encoding, which OCI's S3-compat endpoint
  rejects (`NotImplemented: AWS chunked encoding not supported`). Fixed in the
  new `_run_rangeap` by using the repo's existing `_put_file_nonchunked`
  helper (already used elsewhere in this file for the same reason) instead of
  `s3.upload_file`.
