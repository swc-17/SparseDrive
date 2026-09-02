# B2D SparseDrive V1 stage-1 — official val metrics

**Updated:** 2026-09-02.

Standalone SparseDrive V1 perception (`with_planning=False`, DN off) trained
on Bench2Drive **base** (`b2d_infos_train.pkl`, 234,769 frames / 950 clips),
scored on official **val** (`b2d_infos_val.pkl`, 12,806 frames / 50 clips).
There is no separate test pickle; `data.test` is this val split.

| | |
|---|---|
| Config | `projects/configs/sparsedrive_b2d_stage1.py` |
| Train job | `sparsedrive_b2d_stage1-askfdq` (`b2d_stage1_v1_a100x8`) |
| Schedule | 8×A100, batch 64, 20 epochs, 73,360 iters |
| W&B | https://appliedintuition.wandb.io/research/sparsedrive-b2d/runs/sparsedrive_b2d_stage1-askfdq |
| Train ckpts | `s3://research-datasets-chicago/users/tejan/sparsedrive_b2d/work_dirs/b2d_stage1_v1_a100x8` |
| Eval artifacts (epoch 8) | `s3://research-datasets-chicago/users/tejan/sparsedrive_b2d/work_dirs/b2d_stage1_v1_eval_iter29344` |

Protocol: camera-only, 6 RGB views, 1600×900 → 704×384, ROI 30×60 m.
Detection AP is nuScenes-style center-distance (0.5/1/2/4 m) over 8 classes
(`others` is trained but not in `class_range`). Map AP is chamfer at
0.5/1.0/1.5 m over 6 polyline classes. Checkpoint for all numbers below:
`latest_iter_29344.pth` (iter 29,344 ≈ epoch 8 / 20), job
`sparsedrive_b2d_stage1_eval-hob6p7`.

| Metric | Value |
|---|---:|
| Det **mAP** | **0.3565** |
| Det **NDS** | **0.4194** |
| mATE | 0.7766 |
| mASE | 0.2024 |
| mAOE | 0.1599 |
| mAVE | 0.8692 |
| Map **mAP_normal** | **0.5251** |

---

## Classwise breakdown (epoch 8)

### Detection

Class AP is the mean of AP@0.5/1/2/4 m. Rows are config order.

| Class | AP | AP@0.5 | AP@1.0 | AP@2.0 | AP@4.0 | ATE | ASE | AOE | AVE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| car | 0.3242 | 0.0823 | 0.2215 | 0.4108 | 0.5822 | 0.665 | 0.106 | 0.060 | 1.325 |
| van | 0.1651 | 0.0037 | 0.0535 | 0.2045 | 0.3988 | 0.930 | 0.226 | 0.018 | 0.049 |
| truck | 0.2573 | 0.0081 | 0.1542 | 0.3989 | 0.4681 | 1.056 | 0.279 | 0.103 | 3.548 |
| bicycle | 0.4066 | 0.0412 | 0.1914 | 0.5863 | 0.8075 | 0.860 | 0.176 | 0.039 | 1.081 |
| traffic_sign | 0.3135 | 0.0685 | 0.2182 | 0.4195 | 0.5478 | 0.672 | 0.183 | 0.072 | 0.033 |
| traffic_cone | 0.4836 | 0.1397 | 0.3736 | 0.6245 | 0.7967 | 0.644 | 0.238 | 0.891 | 0.025 |
| traffic_light | 0.4116 | 0.0669 | 0.2734 | 0.5572 | 0.7489 | 0.721 | 0.226 | 0.056 | 0.044 |
| pedestrian | 0.4904 | 0.1349 | 0.4558 | 0.6569 | 0.7140 | 0.665 | 0.186 | 0.040 | 0.849 |
| **mean** | **0.3565** | | | | | **0.777** | **0.202** | **0.160** | **0.869** |

### Mapping

Class AP is the mean of chamfer AP@0.5/1.0/1.5 m. Rows are config order.
No StopLine class.

| Class | #GT | #pred | AP@0.5 | AP@1.0 | AP@1.5 | AP |
|---|---:|---:|---:|---:|---:|---:|
| Broken | 31548 | 41925 | 0.3468 | 0.5183 | 0.5811 | 0.4821 |
| Solid | 64225 | 74526 | 0.3840 | 0.5855 | 0.6649 | 0.5448 |
| SolidSolid | 4216 | 6588 | 0.4260 | 0.5540 | 0.5867 | 0.5222 |
| Center | 133551 | 149699 | 0.3898 | 0.6473 | 0.7657 | 0.6009 |
| TrafficLight | 11760 | 12416 | 0.3703 | 0.6265 | 0.7045 | 0.5671 |
| StopSign | 2396 | 2135 | 0.1221 | 0.4580 | 0.7198 | 0.4333 |
| **mAP_normal** | | | | | | **0.5251** |

---

## Epoch 20 (pending)

Checkpoint `iter_73360.pth`, job `sparsedrive_b2d_stage1_eval-iqf9b7`.
Same val protocol. Fill classwise tables when that eval finishes.

---

## Test plan execution (2026-09-02)

| Item | Result |
|---|---|
| Train yaml `run_name=b2d_stage1_v1_a100x8`; `iter_73360.pth` on S3 | **Pass.** Eval yaml pins that ckpt. Train yaml does not list the `.pth` (it writes that prefix). |
| Epoch-8 `hob6p7` vs this doc | **Pass.** Logged dict matches mAP 0.3565, NDS 0.4194, mAP_normal 0.5251 and classwise APs. Artifacts still on S3. |
| Epoch-20 `iqf9b7` classwise | **Open.** Job still staging B2D clips (~900/1000 at check). |
| Navtest protocol (8 cams, 7 det / 3 map) | **Config pass.** `sparsedrive_navsim_stage1_eval_navtest.py` inherits 8 cams and classes `vehicle…generic_object` / `ped_crossing, divider, boundary`. |
| Navtest job `sej3cn` | **Fail.** Inference ran; eval asserted 12144/12146 (`12146 % 8`). Sampler now gives leftover sequences to the last rank. Relaunch: `tppdxa` (running). |
