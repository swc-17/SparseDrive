# NAVSIM vs nuScenes: dataset comparison for the SparseDrive port

Extracted as a standalone deliverable from the full port audit
(`NAVSIM_SPARSEDRIVE_PORT.md`, `/home/tejan/lwm-rl/.claude/worktrees/sparsedrive-navsim/`).
Read that document for the coordinate-contract derivation and required
repository changes; this file only covers the dataset-level differences that
drive the converter design (Phase 0).

## What's on disk locally

| Item | Local location | State (2026-07-21) |
| --- | --- | --- |
| NAVSIM data | `/media/applied/navsim` | ~385 GB: `navsim_logs/{mini,test}`, `sensor_blobs`, `maps`, `navhard_two_stage` |
| nuScenes data | `/media/applied/nuScenes` | full trainval + mini, used by existing SparseDrive `data/nuscenes/` |
| NAVSIM source/API | `/home/tejan/lwm-rl/DrivoR` | NAVSIM 1.1 / nuPlan 1.2 schema, local source of truth |

There is **no `navsim_logs/trainval`** locally — `navtrain` (expected 103,288
scenario tokens / 1,192 logs) is not downloaded. Available splits:

| Filter | Scenario tokens | Logs | Cameras | Current-frame LiDAR |
| --- | ---: | ---: | --- | --- |
| `navmini` | 396 | 52 selected / 62 candidates / 64 raw | all 8 histories present | 340/396 present |
| `navtest` | 12,146 | 136/147 raw logs | all 8 histories present | 10,618/12,146 present |
| `navtrain` | 103,288 (expected) | 1,192 (expected) | not downloaded | not downloaded |

`navmini` supports conversion gates and deliberate overfit only, not a real
training result. `navtest` must never be used to fit the model.

## Annotation-contract differences (NAVSIM record vs. SparseDrive's nuScenes info contract)

| Concern | SparseDrive / nuScenes | NAVSIM | Decision |
| --- | --- | --- | --- |
| Local frame | BEV: x right, y forward | nuPlan/NAVSIM: x forward, y left (ego rear axle) | Rotate every geometric field at the adapter boundary (see coordinate contract below) |
| Boxes | `gt_boxes (N,7)`: x,y,z,l,w,h,yaw | same field layout in `anns.gt_boxes` | Rotate centers/yaw only; dimensions unchanged |
| Classes | 10: car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, cone | 7: vehicle, pedestrian, bicycle, cone, barrier, czone_sign, generic_object | Native 7-class head; NAVSIM can't recover vehicle subtypes |
| Velocity | `gt_velocity (N,2)` SparseDrive frame | `gt_velocity_3d (N,3)` NAVSIM frame | Rotate xy, keep first 2 components |
| Identity | stable integer `instance_inds` | per-observation `instance_tokens` / persistent `track_tokens` | Deterministic integer table from `track_tokens` (never Python `hash()`) |
| Visibility | `num_lidar_pts`, `num_radar_pts`, `valid_flag` | none | Define a class/range validity policy; initially mark all retained boxes valid |
| Agent futures | precomputed `(N,12,2)` deltas + masks | not stored | Join future frames by `track_token`, transform to current frame, delta-encode, mask absence |
| Ego future | 6 xy deltas / 3s + mask | future global ego poses; NAVSIM wants 8 SE(2) poses / 4s | Train/output 8 steps at 0.5s; regenerate planning head + anchors |
| Command | one-hot `[right,left,straight]` | `[left,straight,right,unknown]` | Map to `[right,left,straight]`; `unknown` → straight fallback but mask planner loss |
| Ego status | 10 values (accel xyz, ang. rate xyz, vel xyz, steering) | `can_bus (18)` + `[vx,vy,ax,ay]` | Construct matching 10-value order from `can_bus`; steering zero-filled (unavailable) |
| Maps | local vectors (crossing/divider/boundary) precomputed per info | global nuPlan GPKG + route roadblocks/traffic lights | Extract/clip local vectors offline; route/light tokens deferred |
| Cameras | 6 × 1600×900, pinhole | 8 × 1920×1080, 5 distortion coeffs | Fix 8-camera order; update both deformable heads; validate distortion/projection |
| LiDAR | single top-lidar float records | merged 5-lidar binary PCD `(6,N)` xyz/intensity/ring/lidar-id | Disable auxiliary depth first (deferred); new PCD loader + missing-depth mask is optional later |
| Temporal grouping | nuScenes `scene`, inferred partly from sweeps | connected frames in a log; valid windows cross raw `scene_token` boundaries | Group by linked runs, not `scene_token`/sweeps |
| Official eval | nuScenes det/track/map + local 3s planning metric | audited local NAVSIM v1 PDMS over 4s trajectory | Replace with NAVSIM `AbstractAgent` + pinned PDMS metric-cache |

NAVSIM mini has far denser annotation than nuScenes training frames typically
show: median 86 boxes/frame, max 467 before filtering, `generic_object` the
largest class. Silently dropping `generic_object` would remove real
obstacles — keep all 7 classes in this fixed order:

```text
vehicle, pedestrian, bicycle, traffic_cone, barrier, czone_sign, generic_object
```

`traffic_cone`, `barrier`, `czone_sign`, `generic_object` are static
(zero-motion targets) per NAVSIM/nuPlan's own classification.

## Coordinate contract (NAVSIM-local → SparseDrive BEV)

```text
C = [[ 0, -1, 0],
     [ 1,  0, 0],
     [ 0,  0, 1]]

p_sd   = C p_nav
x_sd   = -y_nav
y_sd   =  x_nav
yaw_sd =  wrap(yaw_nav + pi/2)
```

Applies to box centers/yaw, xy velocities, ego/agent trajectories, map
vectors, point clouds, camera extrinsics, temporal poses. Inverse for
predicted trajectories: `x_nav = y_sd, y_nav = -x_sd`. The planner predicts
xy only; NAVSIM needs SE(2) — derive heading from consecutive positions with
an explicit stationary-trajectory fallback.

Ego-status vector is **not** blindly rotated (its components are ego-axis
scalars, not BEV vectors):

```text
ego_status = concat(can_bus[7:10],   # acceleration xyz
                     can_bus[13:16], # angular rate xyz
                     can_bus[10:13], # velocity xyz
                     zeros(1))       # steering unavailable
```

## What this means for the converter (Phase 0)

- First executable milestone: **camera-only detection on navmini** — isolates
  transform/calibration bugs before they're buried in the full stage-2 loss.
- Vector maps, then motion/planning, follow after the detection gate passes.
- LiDAR depth is deferred entirely for the initial port.
- Full training is blocked on acquiring `navtrain` sensor blobs + log PKLs
  (not on disk yet). Once acquired, freeze a log-disjoint 90/10 split
  (`SHA256(log_name)` first 32 bits mod 10 == 0 → val) before any training —
  `navmini` stays overfit/smoke-only, `navtest` stays evaluation-only.
