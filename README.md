# SparseDrive V1 / V1.5 — geometry-only planner fork

Fork of [swc-17/SparseDrive](https://github.com/swc-17/SparseDrive) carrying one
load-bearing design change plus a NAVSIM port. The upstream README is preserved
[below](#upstream-readme).

## The design change: the planner reads geometry, not pixels

Stock SparseDrive feeds the planner image-derived instance features. Both models
here replace every one of those with an MLP encoding of the corresponding
geometry, and build no ego CNN at all:

- `geometric_inputs=True` substitutes `det_output/map_output["instance_feature"]`
  with encodings of the 11-d agent boxes (position, size, yaw, velocity) and the
  40-d map polylines, at the top of `MotionPlanningHead.forward` — before every
  consumer, including `InstanceQueue.prepare_motion`, so temporal history is
  geometry-derived too.
- `use_cam_ego_feature=False` means the ego CNN is never constructed (not merely
  unused). A constructor assert couples the two flags.

**V1** keeps the stock regression planner. **V1.5** swaps in a trajectory
vocabulary planner (`trajectory_vocab_planner.py`) that scores a fixed
trajectory vocabulary and re-ranks it by predicted metric score
(`pdm_metric_scorer.py`); it is likewise geometry-attention with no image reads.

Each domain ships an image-feature twin as the control that isolates this one
change: V6c on nuScenes, `imgfeat` on NAVSIM.

Scope caveat, stated plainly: *which* top-50 agents and top-10 polylines reach
the planner is still chosen by detection/map confidence, and the boxes and
polylines themselves come from the camera-based perception stack.
"Geometry-only" describes the planner's **inputs**, not the whole pipeline.

Provenance: the decision rests on the declutter experiment in
[docs/declutter_experiment_plan.md](docs/declutter_experiment_plan.md), where a
standalone geometric planner over frozen stage-1 perception (`geo_planner/`)
matched its image-feature control (L2 0.367 +/- 0.004 vs 0.361 +/- 0.010).

## Model zoo

All paths are absolute on the research box. `$LWM = /home/tejan/lwm-rl`,
`$S3 = s3://research-datasets-chicago/users/tejan`.

### nuScenes val — stock 10-class, 6-step, rescore-off

| Model | Planner input | Config | Checkpoint |
| --- | --- | --- | --- |
| Official SparseDrive stage2 | image features | [stage2_v6c_ctrl.py](projects/configs/declutter/stage2_v6c_ctrl.py) (eval only) | `$LWM/SparseDrive/ckpt/sparsedrive_stage2.pth` |
| **V1 geometry-only** (V6, seeds 0/1/2) | geometry | [stage2_v6_geoinput.py](projects/configs/declutter/stage2_v6_geoinput.py) | `$LWM/SparseDrive/work_dirs/stage2_v6_geoinput_seed{0,1,2}/iter_5860.pth` |
| V1 image-feature control (V6c) | image features | [stage2_v6c_ctrl.py](projects/configs/declutter/stage2_v6c_ctrl.py) | `$LWM/SparseDrive/work_dirs/stage2_v6c_ctrl_seed0/iter_5860.pth` |

S3 mirror: `$S3/sparsedrive/declutter/work_dirs_v2/<run>/iter_5860.pth`,
official at `$S3/sparsedrive/declutter/ckpt/sparsedrive_stage2.pth`.

### NAVSIM — navtrain 1,067-log split

| Model | Planner input | Config | Checkpoint |
| --- | --- | --- | --- |
| Official SparseDriveV2 | vocabulary, planning-only | external | `$LWM/SparseDriveV2/weights/sparsedrive_navsimv1_92p2.ckpt`, `sparsedrive_navsimv2_90p3.ckpt` |
| **V1 geometry-only** (A) | geometry | [..._stage2_geoinput_full.py](projects/configs/navsim/sparsedrive_navsim_stage2_geoinput_full.py) | `$LWM/SparseDrive/work_dirs/navsim_stage2_geoinput_full_16g/iter_8703.pth` |
| V1 image-feature control | image features | [..._stage2_imgfeat_full.py](projects/configs/navsim/sparsedrive_navsim_stage2_imgfeat_full.py) | `$LWM/SparseDrive/work_dirs/navsim_stage2_imgfeat_full_16g/iter_8703.pth` |
| **V1.5 vocabulary planner** | geometry | [..._stage2_vocab_metric_full.py](projects/configs/navsim/sparsedrive_navsim_stage2_vocab_metric_full.py) | `$LWM/.wtc/worktrees/sd1.5/SparseDrive/work_dirs/navsim_stage2_vocab_metric_16g/iter_8703.pth` |

S3 mirror: `$S3/navsim/sparsedrive/work_dirs/<run>/iter_8703.pth`.

SparseDriveV2 is an external baseline, not an ablation of this stack: it is
planning-only (no detection/map/motion heads, so its perception cells are N/A)
and trained on all 1,192 navtrain logs against our 1,067, roughly 12% more data.

## Data paths

| What | Path |
| --- | --- |
| nuScenes root | `data/nuscenes` -> `/media/applied/nuScenes` (`samples`, `sweeps`, `maps`, `v1.0-trainval`, `can_bus`) |
| nuScenes infos (6-step, map-annotated) | `data/infos/nuscenes_infos_{train,val}_withmap.pkl` |
| NAVSIM raw | `/media/applied/navsim` (`navsim_logs/`, `sensor_blobs/{mini,test,trainval}`, `maps/`, `navhard_two_stage/`) |
| NAVSIM infos | `data/infos/navsim_infos_{navtrain_full,navtest,navmini}.pkl` |
| Detection / map / motion / plan anchors | `data/kmeans/*.npy` |
| V1.5 trajectory vocabulary | `data/kmeans/sparsedrive_v2/{path_1024.npy,velocity_256.npy,trajectory_1024_256.npz}` |
| navtest metric caches (v1, v2) | `$LWM/SparseDriveV2/exp/metric_cache_navtest_v1`, `metric_cache_navtestv2` |
| navhard two-stage metric cache | `work_dirs/navsim_eval/metric_cache_navhard2s_full` |
| Geometry caches (`geo_planner/`) | `data/geometry_cache{,_v6,_v6b}` |

S3 mirrors: nuScenes keyframes `$S3/sparsedrive/declutter/data/sd_nuscenes.tar`,
nuScenes infos/anchors under `$S3/sparsedrive/declutter/data/`, NAVSIM infos,
anchors, frame tars and metric caches under `$S3/navsim/sparsedrive/`.

`SparseDriveV2` is a hard runtime dependency, not just a baseline: the NAVSIM
scorers import its `navsim` package and read its pinned metric caches. It must
sit at commit `94d01a6`.

## Evaluating

NAVSIM has three protocols; never mix their caches, since the v1 pickles
deserialize `navsim.navsim_v1.*` classes.

| Protocol | Script | Cache | Tokens |
| --- | --- | --- | --- |
| navtest v1 PDMS | `navsim_agent/score_pdm_v1.py` | `metric_cache_navtest_v1` | 12,146 |
| navtest v2 EPDMS | `navsim_agent/score_epdms_navtest_v2.py` | `metric_cache_navtestv2` | 12,146 |
| navhard two-stage EPDMS | `navsim_agent/score_epdms_two_stage.py` | `metric_cache_navhard2s_full` | 5,912 |

Inference runs in the `sparsedrive310` env, scoring in `lilypad` (no single env
has both the mm-stack and the navsim/nuplan stack). On the cluster:

```bash
bash lilypad_config/navsim_eval/submit.sh evalclean_
```

Each job uploads the tree at `runtime_environment.code_assets.root_directory`,
so that key is what points the cluster at a given worktree. nuScenes evals go
through `lilypad_entrypoint_nuscenes.eval_entrypoint_fn` (N parallel
`tools/test.py` runs, one per GPU); NAVSIM through
`lilypad_entrypoint_navsim.eval_entrypoint_fn`.

---

# Upstream README

# SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation

https://github.com/swc-17/SparseDrive/assets/64842878/867276dc-7c19-4e01-9a8e-81c4ed844745

## News
* **`1 April, 2026`:** Our following work SparseDriveV2 is released. [arxiv](https://arxiv.org/abs/2603.29163), [Code](https://github.com/swc-17/SparseDriveV2).
* **`17 March, 2025`:** SparseDrive is accepted by ICRA 2025.
* **`24 June, 2024`:** We reorganize code for better readability. Code & Models are released.
* **`31 May, 2024`:** We release the SparseDrive paper on [arXiv](https://arxiv.org/abs/2405.19620). Code & Models will be released in June, 2024. Please stay tuned!


## Introduction
> SparseDrive is a Sparse-Centric paradigm for end-to-end autonomous driving.
- We explore the sparse scene representation for end-to-end autonomous driving and propose a Sparse-Centric paradigm named SparseDrive, which unifies multiple tasks with sparse instance representation.
- We revise the great similarity shared between motion prediction and planning, correspondingly leading to a parallel design for motion planner. We further propose a hierarchical planning selection strategy incorporating a collision-aware rescore module to boost the planning performance.
- On the challenging nuScenes benchmark, SparseDrive surpasses previous SOTA methods in terms of all metrics, especially the safety-critical metric collision rate, while keeping much higher training and inference efficiency.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="resources/overview.png" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Overview of SparseDrive. SparseDrive first encodes multi-view images into feature maps,
    then learns sparse scene representation through symmetric sparse perception, and finally perform
    motion prediction and planning in a parallel manner. An instance memory queue is devised for
    temporal modeling.</div>
</center>
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="resources/sparse_perception.png" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Model architecture of symmetric sparse perception, which unifies detection, tracking and
    online mapping in a symmetric structure.</div>
</center>
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="resources/motion_planner.png" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Model structure of parallel motion planner, which performs motion prediction and planning
    simultaneously and outputs safe planning trajectory.</div>
</center>

## Results in paper

- Comprehensive results for all tasks on [nuScenes](https://github.com/nutonomy/nuscenes-devkit).

| Method | NDS | AMOTA | minADE (m) | L2 (m) Avg | Col. (%) Avg | Training Time (h) | FPS |
| :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| UniAD | 0.498 | 0.359 | 0.71 | 0.73 | 0.61 | 144 | 1.8 |
| SparseDrive-S | 0.525 | 0.386 | 0.62 | 0.61 | 0.08 | **20** | **9.0** |
| SparseDrive-B | **0.588** | **0.501** | **0.60** | **0.58** | **0.06** | 30 | 7.3 |

- Open-loop planning results on [nuScenes](https://github.com/nutonomy/nuscenes-devkit).

| Method | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | L2 (m) Avg | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s | Col. (%) Avg | FPS |
| :---: | :---: | :---: | :---: | :---:| :---: | :---: | :---: | :---: | :---: |
| UniAD | 0.45 | 0.70 | 1.04 | 0.73 | 0.62 | 0.58 | 0.63 | 0.61 | 1.8 |
| VAD | 0.41 | 0.70 | 1.05 | 0.72 | 0.03 | 0.19 | 0.43 | 0.21 |4.5 |
| SparseDrive-S | **0.29** | 0.58 | 0.96 | 0.61 | 0.01 | 0.05 | 0.18 | 0.08 | **9.0** |
| SparseDrive-B | **0.29** | **0.55** | **0.91** | **0.58** | **0.01** | **0.02** | **0.13** | **0.06** | 7.3 |

## Results of released checkpoint
We found that some collision cases were not taken into consideration in our previous code, so we re-implement the evaluation metric for collision rate in released code and provide updated results.

## Main results
| Model | config | ckpt | log | det: NDS | mapping: mAP | track: AMOTA |track: AMOTP | motion: EPA_car |motion: minADE_car| motion: minFDE_car | motion: MissRate_car | planning: CR | planning: L2 |
| :---: | :---: | :---: | :---: | :---: | :---:|:---:|:---: | :---: | :----: | :----: | :----: | :----: | :----: |
| Stage1 |[cfg](projects/configs/sparsedrive_small_stage1.py)|[ckpt](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1.pth)|[log](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1_log.txt)|0.5260|0.5689|0.385|1.260| | | | | | |
| Stage2 |[cfg](projects/configs/sparsedrive_small_stage2.py)|[ckpt](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth)|[log](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2_log.txt)|0.5257|0.5656|0.372|1.248|0.492|0.61|0.95|0.133|0.097%|0.61|

## Detailed results for planning
| Method | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | L2 (m) Avg | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s | Col. (%) Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UniAD | 0.45 | 0.70 | 1.04 | 0.73 | 0.66 | 0.66 | 0.72 | 0.68 |
| UniAD-wo-post-optim | 0.32 | 0.58 | 0.94 | 0.61 | 0.17 | 0.27 | 0.42 | 0.29 |
| VAD | 0.41 | 0.70 | 1.05 | 0.72 | 0.03 | 0.21 | 0.49 | 0.24 | 
| SparseDrive-S | 0.30 | 0.58 | 0.95 | 0.61 | 0.01 | 0.05 | 0.23 | 0.10 | 


## Quick Start
[Quick Start](docs/quick_start.md)

## Citation
If you find SparseDrive useful in your research or applications, please consider giving us a star &#127775; and citing it by the following BibTeX entry.
```
@inproceedings{sun2025sparsedrive,
  title={Sparsedrive: End-to-end autonomous driving via sparse scene representation},
  author={Sun, Wenchao and Lin, Xuewu and Shi, Yining and Zhang, Chuang and Wu, Haoran and Zheng, Sifa},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={8795--8801},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgement
- [Sparse4D](https://github.com/HorizonRobotics/Sparse4D)
- [UniAD](https://github.com/OpenDriveLab/UniAD) 
- [VAD](https://github.com/hustvl/VAD)
- [StreamPETR](https://github.com/exiawsh/StreamPETR)
- [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)

