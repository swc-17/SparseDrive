"""Devkit-free nuScenes-style 3D detection evaluation.

A faithful re-implementation of the nuScenes detection eval
(nuscenes/eval/detection/algo.py: accumulate / calc_ap / calc_tp) that works
directly on gt/pred box arrays in a shared metric frame, with no dependency
on the (dataset-locked) nuScenes devkit. Written for the NAVSIM port where
gt comes from navsim_infos pkls, but dataset-agnostic.

Matching semantics (identical to detection_cvpr_2019 unless noted):
  - per-class greedy matching by 2D BEV center distance, thresholds
    0.5 / 1.0 / 2.0 / 4.0 m, predictions visited in descending score order;
  - AP = 101-point interpolated precision-recall area with min_recall=0.1
    and min_precision=0.1 clipping, averaged over the 4 thresholds;
  - TP metrics (ATE / ASE / AOE / AVE) accumulated at the 2.0 m threshold
    over true positives, cum-mean interpolated onto the recall grid and
    averaged from recall 0.1 to the max achieved recall;
  - single range cutoff for every class (NAVSIM converter policy: 55 m)
    instead of the nuScenes per-class class_range;
  - no attribute metric (mAAE): NAVSIM has no attributes. The composite
    score ("nds") therefore uses the 4 available TP metrics:
    (5 * mAP + sum(1 - min(1, tp_err))) / 9.

Per-class TP-metric exclusions mirror the nuScenes leaderboard treatment of
static / rotation-symmetric classes:
  - traffic_cone: no AOE, no AVE;
  - barrier: 180-deg orientation period, no AVE;
  - czone_sign: no AVE (static by construction).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

TP_METRICS = ("trans_err", "scale_err", "orient_err", "vel_err")
ERR_NAME_MAPPING = {
    "trans_err": "mATE",
    "scale_err": "mASE",
    "orient_err": "mAOE",
    "vel_err": "mAVE",
}
# metrics NOT computed for a class
DEFAULT_TP_EXCLUSIONS = {
    "traffic_cone": ("orient_err", "vel_err"),
    "barrier": ("vel_err",),
    "czone_sign": ("vel_err",),
}
# classes with a 180-deg orientation period (front/back symmetric)
DEFAULT_PERIOD_PI = ("barrier",)


def _cummean(x: np.ndarray) -> np.ndarray:
    """Cumulative mean that ignores NaNs (nuScenes utils.cummean)."""
    if np.all(np.isnan(x)):
        return np.ones(len(x))
    sum_vals = np.nancumsum(x.astype(float))
    count_vals = np.cumsum(~np.isnan(x))
    return np.divide(
        sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0
    )


def _scale_err(gt_wlh: np.ndarray, pred_wlh: np.ndarray) -> float:
    """1 - IoU of the two boxes after aligning translation and yaw."""
    min_wlh = np.minimum(gt_wlh, pred_wlh)
    inter = float(np.prod(min_wlh))
    union = float(np.prod(gt_wlh)) + float(np.prod(pred_wlh)) - inter
    return 1.0 - inter / max(union, 1e-9)


def _yaw_err(gt_yaw: float, pred_yaw: float, period: float) -> float:
    diff = (pred_yaw - gt_yaw + period / 2.0) % period - period / 2.0
    return abs(float(diff))


def _accumulate(
    gt_by_sample: List[Dict],
    preds: List[Dict],
    dist_th: float,
    npos: int,
    period: float,
    nelem: int = 101,
):
    """Match one class at one distance threshold.

    Args:
        gt_by_sample: per-sample dict(boxes (N,7), velocity (N,2)) of this
            class only.
        preds: flat list of dicts(sample_idx, box (7,), velocity (2,), score)
            of this class, any order.
        dist_th: center-distance match threshold in meters.
        npos: total number of gt boxes of this class.
        period: orientation period for AOE.

    Returns:
        dict with keys precision/confidence (nelem,), raw match errors, or
        None if npos == 0.
    """
    if npos == 0:
        return None

    order = np.argsort([-p["score"] for p in preds], kind="stable")
    tp, fp, conf = [], [], []
    match_data = {k: [] for k in TP_METRICS}
    match_data["conf"] = []
    taken = set()

    for idx in order:
        pred = preds[idx]
        s = pred["sample_idx"]
        gts = gt_by_sample[s]
        min_dist = np.inf
        match_gt_idx = None
        boxes = gts["boxes"]
        for gt_idx in range(len(boxes)):
            if (s, gt_idx) in taken:
                continue
            d = float(np.linalg.norm(boxes[gt_idx, :2] - pred["box"][:2]))
            if d < min_dist:
                min_dist = d
                match_gt_idx = gt_idx

        if min_dist < dist_th:
            taken.add((s, match_gt_idx))
            tp.append(1)
            fp.append(0)
            conf.append(pred["score"])
            gt_box = boxes[match_gt_idx]
            gt_vel = gts["velocity"][match_gt_idx]
            match_data["trans_err"].append(min_dist)
            match_data["scale_err"].append(
                _scale_err(gt_box[3:6], pred["box"][3:6])
            )
            match_data["orient_err"].append(
                _yaw_err(gt_box[6], pred["box"][6], period)
            )
            match_data["vel_err"].append(
                float(np.linalg.norm(pred["velocity"] - gt_vel))
            )
            match_data["conf"].append(pred["score"])
        else:
            tp.append(0)
            fp.append(1)
            conf.append(pred["score"])

    if len(match_data["trans_err"]) == 0:
        # no matches at all: worst-case TP curves
        return dict(
            precision=np.zeros(nelem),
            recall=np.linspace(0, 1, nelem),
            confidence=np.zeros(nelem),
            **{k: np.ones(nelem) for k in TP_METRICS},
        )

    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    rec_interp = np.linspace(0, 1, nelem)
    precision = np.interp(rec_interp, rec, prec, right=0)
    confidence = np.interp(rec_interp, rec, conf, right=0)

    out = dict(precision=precision, recall=rec_interp, confidence=confidence)
    for key in TP_METRICS:
        tmp = _cummean(np.array(match_data[key]))
        out[key] = np.interp(
            confidence[::-1], np.array(match_data["conf"])[::-1], tmp[::-1]
        )[::-1]
    return out


def _calc_ap(md: Dict, min_recall: float, min_precision: float) -> float:
    prec = np.copy(md["precision"])
    prec = prec[round(100 * min_recall) + 1 :]  # exclude operating points < min recall
    prec -= min_precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def _calc_tp(md: Dict, min_recall: float, metric_name: str) -> float:
    first_ind = round(100 * min_recall) + 1
    nz = np.nonzero(md["confidence"])[0]
    last_ind = nz[-1] if len(nz) > 0 else 0
    if last_ind < first_ind:
        return 1.0  # assign worst value when no predictions in valid range
    return float(np.mean(md[metric_name][first_ind : last_ind + 1]))


def evaluate_detection(
    gt_annos: List[Dict],
    pred_annos: List[Dict],
    class_names: Sequence[str],
    dist_ths: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    dist_th_tp: float = 2.0,
    class_range: float = 55.0,
    min_recall: float = 0.1,
    min_precision: float = 0.1,
    tp_exclusions: Optional[Dict] = None,
    verbose: bool = True,
    logger=None,
) -> Dict:
    """Evaluate 3D detections nuScenes-style, devkit-free.

    Args:
        gt_annos: per sample dict(boxes (N,>=7) [x,y,z,w,l,h,yaw,...],
            names (N,) str, velocity (N,2)). Boxes in the metric BEV frame
            shared with the predictions.
        pred_annos: per sample dict(boxes (M,>=7), scores (M,),
            labels (M,) int index into class_names, velocity (M,2)).
        class_names: ordered class names.
        class_range: single BEV-range validity cutoff applied to both gt
            and predictions.

    Returns:
        metrics dict: label_aps, label_tp_errors, tp_errors, mean_ap,
        nd_score, per-class gt/pred counts.
    """
    assert len(gt_annos) == len(pred_annos), (
        f"gt/pred sample count mismatch: {len(gt_annos)} vs {len(pred_annos)}"
    )
    if tp_exclusions is None:
        tp_exclusions = DEFAULT_TP_EXCLUSIONS
    n_samples = len(gt_annos)

    # pre-split by class with range filtering
    gt_cls = {c: [] for c in class_names}
    npos = {c: 0 for c in class_names}
    for s in range(n_samples):
        g = gt_annos[s]
        boxes = np.asarray(g["boxes"], dtype=float)
        names = np.asarray(g["names"])
        vel = np.asarray(g["velocity"], dtype=float)
        if len(vel):
            vel = np.nan_to_num(vel)
        if len(boxes):
            in_range = np.linalg.norm(boxes[:, :2], axis=1) <= class_range
        else:
            in_range = np.zeros(0, dtype=bool)
        for c in class_names:
            m = (names == c) & in_range if len(boxes) else in_range
            gt_cls[c].append(
                dict(
                    boxes=boxes[m] if len(boxes) else boxes.reshape(0, 7),
                    velocity=vel[m] if len(boxes) else vel.reshape(0, 2),
                )
            )
            npos[c] += int(m.sum())

    pred_cls = {c: [] for c in class_names}
    npred = {c: 0 for c in class_names}
    for s in range(n_samples):
        p = pred_annos[s]
        boxes = np.asarray(p["boxes"], dtype=float)
        if len(boxes) == 0:
            continue
        scores = np.asarray(p["scores"], dtype=float)
        labels = np.asarray(p["labels"], dtype=int)
        vel = np.asarray(p["velocity"], dtype=float)
        in_range = np.linalg.norm(boxes[:, :2], axis=1) <= class_range
        for i in np.nonzero(in_range)[0]:
            li = labels[i]
            if li < 0 or li >= len(class_names):
                continue
            c = class_names[li]
            pred_cls[c].append(
                dict(
                    sample_idx=s,
                    box=boxes[i, :7],
                    velocity=vel[i, :2],
                    score=float(scores[i]),
                )
            )
            npred[c] += 1

    label_aps = {c: {} for c in class_names}
    label_tp_errors = {c: {} for c in class_names}
    for c in class_names:
        period = np.pi if c in DEFAULT_PERIOD_PI else 2 * np.pi
        md_tp = None
        for dist_th in dist_ths:
            md = _accumulate(gt_cls[c], pred_cls[c], dist_th, npos[c], period)
            if md is None:  # no gt of this class in the split
                label_aps[c][dist_th] = float("nan")
                continue
            label_aps[c][dist_th] = _calc_ap(md, min_recall, min_precision)
            if dist_th == dist_th_tp:
                md_tp = md
        for metric in TP_METRICS:
            if metric in tp_exclusions.get(c, ()):
                label_tp_errors[c][metric] = float("nan")
            elif md_tp is None:
                label_tp_errors[c][metric] = float("nan")
            else:
                label_tp_errors[c][metric] = _calc_tp(md_tp, min_recall, metric)

    # means over classes that actually have gt / defined metrics
    ap_vals = [
        v for c in class_names for v in label_aps[c].values() if not np.isnan(v)
    ]
    mean_ap = float(np.mean(ap_vals)) if ap_vals else 0.0
    tp_errors = {}
    for metric in TP_METRICS:
        vals = [
            label_tp_errors[c][metric]
            for c in class_names
            if not np.isnan(label_tp_errors[c][metric])
        ]
        tp_errors[metric] = float(np.mean(vals)) if vals else float("nan")

    # composite score (nuScenes NDS formula, 4 available TP metrics)
    total = 5.0 * mean_ap
    n_tp = 0
    for metric in TP_METRICS:
        if not np.isnan(tp_errors[metric]):
            total += 1.0 - min(1.0, tp_errors[metric])
            n_tp += 1
    nd_score = total / (5.0 + n_tp) if n_tp else mean_ap

    metrics = dict(
        label_aps={c: {str(k): v for k, v in label_aps[c].items()} for c in class_names},
        label_tp_errors=label_tp_errors,
        tp_errors=tp_errors,
        mean_dist_aps={
            c: float(np.nanmean(list(label_aps[c].values())))
            if len(label_aps[c])
            else float("nan")
            for c in class_names
        },
        mean_ap=mean_ap,
        nd_score=nd_score,
        num_gts=npos,
        num_preds=npred,
        cfg=dict(
            dist_ths=list(dist_ths),
            dist_th_tp=dist_th_tp,
            class_range=class_range,
            min_recall=min_recall,
            min_precision=min_precision,
        ),
    )

    if verbose:
        try:
            import prettytable

            from mmcv.utils import print_log

            table = prettytable.PrettyTable(
                ["class", "n_gt", "n_pred"]
                + [f"AP@{th}" for th in dist_ths]
                + ["AP", "ATE", "ASE", "AOE", "AVE"]
            )
            for c in class_names:
                row = [c, npos[c], npred[c]]
                for th in dist_ths:
                    v = label_aps[c].get(th, float("nan"))
                    row.append("-" if np.isnan(v) else f"{v:.4f}")
                m = metrics["mean_dist_aps"][c]
                row.append("-" if np.isnan(m) else f"{m:.4f}")
                for metric in TP_METRICS:
                    v = label_tp_errors[c][metric]
                    row.append("-" if np.isnan(v) else f"{v:.4f}")
                table.add_row(row)
            summary = (
                f"\n{table}\n"
                f"mAP: {mean_ap:.4f}  "
                + "  ".join(
                    f"{ERR_NAME_MAPPING[m]}: {tp_errors[m]:.4f}"
                    for m in TP_METRICS
                    if not np.isnan(tp_errors[m])
                )
                + f"  NDS: {nd_score:.4f}"
            )
            print_log(summary, logger=logger)
        except Exception as e:  # printing must never kill an eval
            print(f"[det_eval] summary print failed: {e}")

    return metrics
