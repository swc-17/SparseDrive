"""Train/evaluate a geometric-planner variant on the frozen-stage1 cache.

Usage:
    PYTHONPATH=. python geo_planner/train.py --variant g0 --seed 0
    PYTHONPATH=. python geo_planner/train.py --variant c1 --seed 0
Variants: c1 g0 g1 g2 g3 g4_none g5

Loss/eval mirror SparseDrive's planning branch exactly:
  - PlanningTarget sampler (cmd-select + best-mode targets)
  - FocalLoss(0.5) cls + L1(1.0) reg + L1(1.0) status
  - final metrics via the repo's planning_eval (L2 + collision, rescore-free)
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(repo_root)
sys.path.insert(0, repo_root)

# pure-python focal loss (local mmcv lacks CUDA ops; bit-identical to reference)
import mmdet.models.losses.focal_loss as _fl


def _py_sigmoid_focal_loss_ext(pred, target, gamma=2.0, alpha=0.25,
                               weight=None, reduction='none'):
    num_classes = pred.size(1)
    one_hot = F.one_hot(target.long(), num_classes + 1)[:, :num_classes]
    one_hot = one_hot.to(pred.dtype)
    pred_sigmoid = pred.sigmoid()
    pt = (1 - pred_sigmoid) * one_hot + pred_sigmoid * (1 - one_hot)
    focal_weight = (alpha * one_hot + (1 - alpha) * (1 - one_hot)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, one_hot, reduction='none') * focal_weight
    if weight is not None:
        loss = loss * weight
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


_fl._sigmoid_focal_loss = _py_sigmoid_focal_loss_ext

import projects.mmdet3d_plugin  # noqa: registries
from mmdet.models import build_loss
from projects.mmdet3d_plugin.models.motion.target import PlanningTarget

from geo_planner.dataset import GeoCacheDataset
from geo_planner.model import GeoPlanner

CACHE = "data/geometry_cache"
INFOS = dict(train="data/infos/nuscenes_infos_train.pkl",
             val="data/infos/nuscenes_infos_val.pkl")


def build_losses():
    cls = build_loss(dict(type="FocalLoss", use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=0.5))
    reg = build_loss(dict(type="L1Loss", loss_weight=1.0))
    status = build_loss(dict(type="L1Loss", loss_weight=1.0))
    return cls, reg, status


def planning_loss(out, batch, sampler, loss_cls, loss_reg, loss_status):
    data = {"gt_ego_fut_cmd": batch["gt_ego_fut_cmd"]}
    cls, cls_target, cls_weight, reg_pred, reg_target, reg_weight = \
        sampler.sample(out["plan_cls"], out["plan_reg"],
                       batch["gt_ego_fut_trajs"], batch["gt_ego_fut_masks"],
                       data)
    cls = cls.flatten(end_dim=1)
    cls_target = cls_target.flatten(end_dim=1)
    cls_weight = cls_weight.flatten(end_dim=1)
    l_cls = loss_cls(cls, cls_target, weight=cls_weight)

    reg_weight = reg_weight.flatten(end_dim=1).unsqueeze(-1)
    reg_pred = reg_pred.flatten(end_dim=1)
    reg_target = reg_target.flatten(end_dim=1)
    l_reg = loss_reg(reg_pred, reg_target, weight=reg_weight)

    l_status = loss_status(out["plan_status"], batch["ego_status"])
    return l_cls, l_reg, l_status


def forecast_loss(out, batch):
    pred = out["forecast"]                       # (B, N, F, 2) offsets
    # target = displacement from the agent's CURRENT position (well-conditioned)
    tgt = batch["agent_fut"] - batch["agent_geo"][..., None, :2]
    mask = batch["agent_fut_mask"][..., None]
    denom = mask.sum().clamp(min=1.0)
    return (torch.abs(pred - tgt) * mask).sum() / denom


@torch.no_grad()
def select_final(plan_cls, plan_reg, cmd):
    """cmd-select + argmax mode -> cumulative trajectory (B, 6, 2)."""
    B = plan_cls.shape[0]
    idx = torch.arange(B, device=plan_cls.device)
    cls = plan_cls.reshape(B, 3, -1)[idx, cmd]                    # (B, 6)
    reg = plan_reg.reshape(B, 3, -1, plan_reg.shape[-2], 2)[idx, cmd]
    best = cls.argmax(dim=-1)
    return reg[idx, best].cumsum(dim=-2)                          # (B, 6, 2)


@torch.no_grad()
def quick_l2(model, loader, device):
    """planning_eval-style cumulative-mean L2 at 1/2/3 s (no collision)."""
    per_step = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if not batch["gt_ego_fut_masks"].all():
            keep = batch["gt_ego_fut_masks"].all(dim=1)
            if not keep.any():
                continue
            batch = {k: v[keep] for k, v in batch.items()}
        out = model(batch)
        cmd = batch["gt_ego_fut_cmd"].argmax(dim=-1)
        traj = select_final(out["plan_cls"], out["plan_reg"], cmd)
        gt = batch["gt_ego_fut_trajs"].cumsum(dim=-2)
        per_step.append(torch.linalg.norm(traj - gt, dim=-1))     # (B, 6)
    per_step = torch.cat(per_step).mean(dim=0)                    # (6,)
    cummean = torch.cumsum(per_step, 0) / torch.arange(
        1, 7, device=per_step.device)
    return dict(l2_1s=cummean[1].item(), l2_2s=cummean[3].item(),
                l2_3s=cummean[5].item(),
                l2_avg=(cummean[1] + cummean[3] + cummean[5]).item() / 3)


def full_eval(model, loader, device, out_pkl):
    """Emit final_planning per val sample (dataset order) for planning_eval."""
    import pickle
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            cmd = batch["gt_ego_fut_cmd"].argmax(dim=-1)
            traj = select_final(out["plan_cls"], out["plan_reg"], cmd)
            for b in range(traj.shape[0]):
                results.append(
                    {"img_bbox": {"final_planning": traj[b].cpu()}})
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["c1", "g0", "g1", "g2", "g3", "g4_none", "g5"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--forecast-weight", type=float, default=0.5)
    ap.add_argument("--out-dir", default="work_dirs/geo_planner")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    run_name = f"geo_{args.variant}_seed{args.seed}"
    if args.epochs != 10:
        run_name += f"_e{args.epochs}"  # schedule-robustness runs get own names
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    wandb = None
    if not args.no_wandb:
        import netrc
        host = "appliedintuition.wandb.io"
        os.environ.setdefault("WANDB_BASE_URL", f"https://{host}")
        try:
            os.environ.setdefault(
                "WANDB_API_KEY", netrc.netrc().authenticators(host)[2])
        except Exception:
            pass
        import wandb as _wandb
        wandb = _wandb
        # the Lilypad harness injects WANDB_RUN_ID (= workload id); with
        # several trainings per node they'd all collide on one run id
        for k in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RUN_GROUP"):
            os.environ.pop(k, None)
        try:
            wandb.init(project="sparsedrive-declutter", entity="research",
                       name=run_name, config=vars(args))
        except Exception:
            wandb.init(project="sparsedrive-declutter", name=run_name,
                       config=vars(args))

    load_img = args.variant == "c1"
    train_ds = GeoCacheDataset(f"{CACHE}/train", INFOS["train"],
                               load_image_feats=load_img)
    val_ds = GeoCacheDataset(f"{CACHE}/val", INFOS["val"],
                             load_image_feats=load_img)
    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=8, drop_last=True, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                        num_workers=8, pin_memory=True)

    model = GeoPlanner(variant=args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{run_name}] params={n_params/1e6:.2f}M "
          f"train={len(train_ds)} val={len(val_ds)}")

    sampler = PlanningTarget(ego_fut_ts=6, ego_fut_mode=6)
    loss_cls, loss_reg, loss_status = build_losses()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * len(train_ld)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    best = float("inf")
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_ld:
            batch = {k: v.to(device, non_blocking=True)
                     for k, v in batch.items()}
            out = model(batch)
            l_cls, l_reg, l_status = planning_loss(
                out, batch, sampler, loss_cls, loss_reg, loss_status)
            loss = l_cls + l_reg + l_status
            log = dict(loss_cls=l_cls.item(), loss_reg=l_reg.item(),
                       loss_status=l_status.item())
            if model.forecast_aux:
                l_fc = forecast_loss(out, batch) * args.forecast_weight
                loss = loss + l_fc
                log["loss_forecast"] = l_fc.item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            step += 1
            if wandb and step % 50 == 0:
                wandb.log({f"train/{k}": v for k, v in log.items()}
                          | {"train/loss": loss.item(),
                             "lr": sched.get_last_lr()[0]}, step=step)

        model.eval()
        metrics = quick_l2(model, val_ld, device)
        print(f"[{run_name}] epoch {epoch+1}/{args.epochs} "
              + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if wandb:
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
        if metrics["l2_avg"] < best:
            best = metrics["l2_avg"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pth"))

    # full eval (L2 + collision) with the repo's planning_eval on best ckpt
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pth")))
    results = full_eval(model, val_ld, device,
                        os.path.join(out_dir, "results_val.pkl"))
    from mmcv import Config
    from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval \
        import planning_eval
    cfg = Config.fromfile("projects/configs/declutter/stage1_official_cache_val.py")
    metric_str = planning_eval(results, cfg.eval_config, logger=None)
    print(metric_str)
    with open(os.path.join(out_dir, "final_metrics.txt"), "w") as f:
        f.write(str(metric_str))
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
