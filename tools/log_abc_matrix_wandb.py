"""Log the A/B/C data-regime eval matrix to W&B (research/sparsedrive-navsim).

One run per model variant; metrics namespaced into two separate groups:
  detection/*  — nuScenes val det mAP/NDS/TP-errors + online-map chamfer mAP
                 (A additionally gets its navtrain-val stage-1 perception)
  planning/*   — ego planning: NAVSIM navtest v1 PDMS (+sub-metrics from the
                 per-token CSV), navhard two-stage EPDMS, and nuScenes val
                 L2/collision per horizon (6-step comparable + 8-step tail)

Sources (all pinned artifacts, not rounded report numbers):
  - PDMS:  work_dirs/navsim_eval/pdm_v1/<tag>.{csv,json}
  - EPDMS: work_dirs/navsim_eval/epdms_v2/<tag>.json
  - nuScenes planning: s3 eval_logs/{a,b,bv2}_on_nusc_v1/*_{6,8}step.json
  - nuScenes det/map:  s3 eval_logs/{bc,bv2}_detmap_v{1,2}/*.log (parsed values
    hardcoded below with provenance comments)

Idempotent-ish: re-running overwrites the same run names (resume="allow" via id).
Usage: source ~/.wandb/creds; python tools/log_abc_matrix_wandb.py
"""
import json
import os

import pandas as pd
import wandb

EV = "work_dirs/navsim_eval"
EVAL_JSON = os.environ.get(
    "EVALCMP_JSON_DIR",
    "/tmp/claude-1002/-home-tejan-lwm-rl/f7d81f7d-bb3d-472b-9bcf-a5cc91f400cd/scratchpad/evaljson",
)

SUBMETRICS = {
    "no_at_fault_collisions": "nc",
    "drivable_area_compliance": "dac",
    "ego_progress": "ep",
    "time_to_collision_within_bound": "ttc",
    "comfort": "comfort",
    "driving_direction_compliance": "ddc",
}


def pdms_metrics(tag):
    out = {}
    j = json.load(open(f"{EV}/pdm_v1/{tag}.json"))
    out["planning/navtest_pdms"] = j["pdms"]
    df = pd.read_csv(f"{EV}/pdm_v1/{tag}.csv")
    df = df[df["token"].str.len() == 16]  # drop any aggregate row
    assert len(df) == j["num_tokens"], (tag, len(df), j["num_tokens"])
    for col, short in SUBMETRICS.items():
        if col in df:
            out[f"planning/navtest_{short}"] = float(df[col].mean())
    return out


def navtest_v2_metrics(tag):
    """navtest-v2 single-stage EPDMS (SparseDriveV2-paper protocol,
    bug-fix scorer; gate: reproduces V2's pinned 0.9037 exactly)."""
    path = f"{EV}/epdms_navtest_v2/{tag}.json"
    if not os.path.exists(path):
        return {}
    j = json.load(open(path))
    return {"planning/navtest_v2_epdms": j["epdms"]}


def epdms_metrics(tag):
    j = json.load(open(f"{EV}/epdms_v2/{tag}.json"))
    e = j["epdms"]
    return {
        "planning/navhard_epdms": e["combined"],
        "planning/navhard_epdms_stage_one": e["stage_one"],
        "planning/navhard_epdms_stage_two": e["stage_two"],
    }


def nusc_plan_metrics(prefix, name):
    """6-step (comparable) + 8-step (tail) L2/collision from evalcmp JSONs."""
    out = {}
    for steps in (6, 8):
        d = json.load(open(f"{EVAL_JSON}/{prefix}/{name}_{steps}step.json"))
        for k, v in d.items():
            if k.startswith("plan_L2"):
                out[f"planning/nusc_{steps}step_{k[5:]}"] = v
            elif k.startswith("plan_obj_box_col"):
                out[f"planning/nusc_{steps}step_col{k[16:]}"] = v
    return out


# nuScenes det/map values parsed from eval_logs/*detmap*/ *.log (cluster jobs
# sd_evalcmp_bc_detmap 2026-07-24, bc_map2 2026-07-24, bv2_detmap 2026-07-25).
DETMAP = {
    "B-v1": {
        "detection/nusc_det_mAP": 0.3498, "detection/nusc_det_NDS": 0.4294,
        "detection/nusc_det_mATE": 0.6161, "detection/nusc_det_mASE": 0.2926,
        "detection/nusc_det_mAOE": 0.5997, "detection/nusc_det_mAVE": 0.3755,
        "detection/nusc_map_mAP": 0.1988,
        "detection/nusc_map_ped_crossing": 0.0851,
        "detection/nusc_map_divider": 0.3185,
        "detection/nusc_map_boundary": 0.1927,
    },
    "B-v2": {
        "detection/nusc_det_mAP": 0.4232, "detection/nusc_det_NDS": 0.4932,
        "detection/nusc_det_mATE": 0.5175, "detection/nusc_det_mASE": 0.2780,
        "detection/nusc_det_mAOE": 0.5700, "detection/nusc_det_mAVE": 0.3116,
        "detection/nusc_map_mAP": 0.5679,
        "detection/nusc_map_ped_crossing": 0.5309,
        "detection/nusc_map_divider": 0.6004,
        "detection/nusc_map_boundary": 0.5723,
    },
    "C": {
        "detection/nusc_det_mAP": 0.4687, "detection/nusc_det_NDS": 0.5438,
        "detection/nusc_det_mATE": 0.4630, "detection/nusc_det_mASE": 0.2669,
        "detection/nusc_det_mAOE": 0.4314, "detection/nusc_det_mAVE": 0.2877,
        "detection/nusc_map_mAP": 0.5521,
        "detection/nusc_map_ped_crossing": 0.4950,
        "detection/nusc_map_divider": 0.5822,
        "detection/nusc_map_boundary": 0.5789,
    },
    # A was never det/map-evaluated on nuScenes; its stage-1 perception is on
    # its own navtrain-val split (docs/NAVSIM_PORT_SESSION_STATUS.md).
    "imgfeat": {
        "detection/navsim_det_mAP": 0.3438, "detection/navsim_det_NDS": 0.4100,
        "detection/navsim_det_mATE": 0.5738, "detection/navsim_det_mAVE": 0.6822,
        "detection/navsim_map_mAP": 0.8439,
        "detection/navsim_map_ped_crossing": 0.8234,
        "detection/navsim_map_divider": 0.8913,
        "detection/navsim_map_boundary": 0.8171,
    },
    "A": {
        "detection/navtrainval_det_mAP": 0.3517,
        "detection/navtrainval_det_NDS_analog": 0.4103,
        "detection/navtrainval_map_chamfer_mAP": 0.8503,
    },
}


# NAVSIM navtrain-val det/map (stage-2 ckpts, local 5090 runs 2026-07-28;
# logs work_dirs/navsim_eval/logs/detmap_navsim_*.log). A's stage-1 ckpt
# reference: det 0.3517 / NDS 0.4103 / map 0.8503.
NAVSIM_DETMAP = {
    "A": {
        "detection/navsim_det_mAP": 0.3415, "detection/navsim_det_NDS": 0.3981,
        "detection/navsim_det_mATE": 0.5897, "detection/navsim_det_mAVE": 0.7180,
        "detection/navsim_map_mAP": 0.8425,
        "detection/navsim_map_ped_crossing": 0.8223,
        "detection/navsim_map_divider": 0.8895,
        "detection/navsim_map_boundary": 0.8156,
    },
    "B-v1": {
        "detection/navsim_det_mAP": 0.3522, "detection/navsim_det_NDS": 0.4264,
        "detection/navsim_det_mATE": 0.5571, "detection/navsim_det_mAVE": 0.5865,
        "detection/navsim_map_mAP": 0.2113,
        "detection/navsim_map_ped_crossing": 0.0993,
        "detection/navsim_map_divider": 0.3859,
        "detection/navsim_map_boundary": 0.1487,
    },
    "B-v2": {
        "detection/navsim_det_mAP": 0.3672, "detection/navsim_det_NDS": 0.4423,
        "detection/navsim_det_mATE": 0.5502, "detection/navsim_det_mAVE": 0.5753,
        "detection/navsim_map_mAP": 0.2110,
        "detection/navsim_map_ped_crossing": 0.0877,
        "detection/navsim_map_divider": 0.4005,
        "detection/navsim_map_boundary": 0.1447,
    },
    "C": {},
}

MODELS = {
    "imgfeat": dict(
        run="eval_imgfeat_navsim_only_iter8703",
        config=dict(
            regime="NAVSIM-only, image-feature planner (non-V6 twin of A: "
                   "geometric_inputs=False, use_cam_ego_feature=True)",
            checkpoint="navsim_stage2_imgfeat_full_16g/iter_8703",
            planner="original image-feature planner, use_rescore=False",
        ),
        pdms_tag="imgfeat_iter8703_navtest_local",
        navtest_v2_tag="imgfeat_iter8703_navtestv2",
        epdms_tag="imgfeat_iter8703_navhard2s",
        nusc_plan=None,
        extra={},
    ),
    "A": dict(
        run="eval_A_navsim_only_16g_iter8703",
        config=dict(
            regime="NAVSIM-only (navtrain 1,067-log split)",
            checkpoint="navsim_stage2_geoinput_full_16g/iter_8703",
            planner="V6 geometric_inputs, use_rescore=False",
        ),
        pdms_tag="v6_full16g_iter8703_navtest",
        navtest_v2_tag="a_full16g_iter8703_navtestv2",
        epdms_tag="v6_full16g_iter8703_navhard2s",
        nusc_plan=("a_on_nusc_v1", "a_navsim16g_on_nusc"),
        extra={"planning/navmini_pdms": 0.837118770073073,
               "planning/nusc_zero_shot": 1.0},
    ),
    "B-v1": dict(
        run="eval_Bv1_combined_naive_iter8800",
        config=dict(
            regime="combined navtrain+nuScenes, naive 1:1 per-sample",
            checkpoint="combined_stage2_geoinput_seed0/iter_8800",
            planner="V6 geometric_inputs, use_rescore=False",
        ),
        pdms_tag="b_combined_iter8800_navtest_local",
        navtest_v2_tag="bv1_combined_iter8800_navtestv2",
        epdms_tag="b_combined_iter8800_navhard2s",
        nusc_plan=("b_on_nusc_v1", "b_combined_on_nusc"),
        extra={},
    ),
    "B-v2": dict(
        run="eval_Bv2_combined_upweighted_iter8800",
        config=dict(
            regime="combined, nuScenes 4x upweight (sample_weights [1,4]) "
                   "+ balanced motion/plan anchors (combined_bal)",
            checkpoint="combined_stage2_v2_seed0/iter_8800",
            planner="V6 geometric_inputs, use_rescore=False",
        ),
        pdms_tag="bv2_iter8800_navtest_cluster",
        navtest_v2_tag="bv2_combined_iter8800_navtestv2",
        epdms_tag="bv2_combined_iter8800_navhard2s",  # pending; skipped if absent
        nusc_plan=("bv2_on_nusc_v1", "bv2_on_nusc"),
        extra={},
    ),
    "C": dict(
        run="eval_C_nusc_only_unified_iter5860",
        config=dict(
            regime="nuScenes-only (unified 7-class/8-step contract)",
            checkpoint="nusc_unified_stage2_geoinput_seed0/iter_5860",
            planner="V6 geometric_inputs, use_rescore=False",
        ),
        pdms_tag="zs_nusc_unified_iter5860_navtest_local",
        navtest_v2_tag="c_zs_nusc_iter5860_navtestv2",
        epdms_tag=None,
        nusc_plan=None,  # C's nuScenes numbers come from its own eval json below
        extra={"planning/navmini_pdms": 0.4901993634182138,
               "planning/navsim_zero_shot": 1.0,
               # C nuScenes val planning (reproduction-gate run, 6-step
               # comparable subset; docs/NAVSIM_PORT_SESSION_STATUS.md):
               "planning/nusc_6step_L2_1.0s": 0.3208,
               "planning/nusc_6step_L2_2.0s": 0.6227,
               "planning/nusc_6step_L2_3.0s": 1.0254,
               "planning/nusc_6step_L2_avg": 0.6563,
               "planning/nusc_6step_col_avg": 0.195,
               "planning/nusc_8step_L2_3.5s": 1.2686,
               "planning/nusc_8step_L2_4.0s": 1.5327},
    ),
}


def main():
    for name, m in MODELS.items():
        metrics = {}
        metrics.update(pdms_metrics(m["pdms_tag"]))
        metrics.update(navtest_v2_metrics(m["navtest_v2_tag"]))
        if m["epdms_tag"] and os.path.exists(f"{EV}/epdms_v2/{m['epdms_tag']}.json"):
            metrics.update(epdms_metrics(m["epdms_tag"]))
        if m["nusc_plan"]:
            metrics.update(nusc_plan_metrics(*m["nusc_plan"]))
        metrics.update(DETMAP[name])
        metrics.update(NAVSIM_DETMAP.get(name, {}))
        metrics.update(m["extra"])

        run = wandb.init(
            entity="research", project="sparsedrive-navsim",
            name=m["run"], id=m["run"], resume="allow",
            group="abc_matrix_eval", job_type="eval",
            config=dict(model=name, **m["config"]), reinit=True,
        )
        run.log(metrics)
        run.finish()
        print(f"[{name}] logged {len(metrics)} metrics -> {m['run']}")


if __name__ == "__main__":
    main()
