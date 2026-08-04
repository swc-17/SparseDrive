"""Geometric planner: plans from the geometric abstraction of the scene.

Inputs are agent boxes (+kinematics), map polylines, and the ego kinematic
anchor. Image features enter ONLY when variant='c1' (the control).

Mirrors the output contract of SparseDrive's MotionPlanningHead planning
branch so losses/decoding/eval are reused verbatim:
    plan_cls    (B, 18)            cmd(3) x mode(6)
    plan_reg    (B, 1, 18, 6, 2)   per-step offsets (cumsum at decode)
    plan_status (B, 10)
"""
import numpy as np
import torch
import torch.nn as nn

from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import (
    SparseBox3DEncoder,
)
from projects.mmdet3d_plugin.models.map.map_blocks import SparsePoint3DEncoder
from projects.mmdet3d_plugin.models.attention import gen_sineembed_for_position
from projects.mmdet3d_plugin.models.blocks import linear_relu_ln

VY = 9
D = 256


def _mlp(din, dout, hidden=256):
    return nn.Sequential(
        nn.Linear(din, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, dout),
    )


class DecoderLayer(nn.Module):
    """Pre-LN: self-attn over queries, cross-attn to agents, cross to map, FFN."""

    def __init__(self, d=D, heads=8, ffn=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
        self.agent_attn = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
        self.map_attn = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(ffn, d),
        )
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(d) for _ in range(4))

    def forward(self, q, agent_kv, agent_pos, map_kv, map_pos):
        x = self.n1(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        x = self.n2(q)
        q = q + self.agent_attn(x, agent_kv + agent_pos, agent_kv,
                                need_weights=False)[0]
        x = self.n3(q)
        q = q + self.map_attn(x, map_kv + map_pos, map_kv,
                              need_weights=False)[0]
        q = q + self.ffn(self.n4(q))
        return q


class GeoPlanner(nn.Module):
    def __init__(self, variant="g0", plan_anchor_path="data/kmeans/kmeans_plan_6.npy",
                 ego_fut_ts=6, ego_fut_mode=6, num_layers=3, hist_len=4,
                 fut_len=3):
        super().__init__()
        assert variant in {"c1", "g0", "g1", "g2", "g3", "g4_none", "g5"}
        self.variant = variant
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.use_image = variant == "c1"
        self.no_velocity = variant == "g1"
        self.use_history = variant == "g2"
        self.forecast_aux = variant == "g3"
        self.ego_vel_none = variant == "g4_none"
        self.no_map = variant == "g5"

        # geometric positional encoders (identical modules to SparseDrive)
        self.box_pos_enc = SparseBox3DEncoder(
            vel_dims=3, embed_dims=[128, 32, 32, 64], mode="cat",
            output_fc=False, in_loops=1, out_loops=4)
        self.map_pos_enc = SparsePoint3DEncoder(embed_dims=D, num_sample=20)

        # content encoders
        if self.use_image:
            self.agent_content = nn.Linear(D, D)
            self.map_content = nn.Linear(D, D)
        else:
            self.agent_content = _mlp(12, D)   # geometry(11)+conf
            self.map_content = _mlp(41, D)     # polyline(40)+conf
        if self.use_history:
            self.hist_enc = _mlp(hist_len * 12, D)  # (11+mask) per step
        if self.forecast_aux:
            self.forecast_head = _mlp(D, fut_len * 2)

        # ego: kinematic anchor + status
        self.ego_status_mlp = _mlp(10, D)

        # plan mode queries from kmeans anchors (as in MotionPlanningHead)
        pa = np.load(plan_anchor_path)  # (3, 6, 6, 2)
        self.plan_anchor = nn.Parameter(
            torch.tensor(pa, dtype=torch.float32), requires_grad=False)
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(D, 1, 1), nn.Linear(D, D))

        self.layers = nn.ModuleList(DecoderLayer() for _ in range(num_layers))

        # heads mirroring MotionPlanningRefinementModule
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(D, 1, 2), nn.Linear(D, 1))
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, ego_fut_ts * 2))
        self.plan_status_branch = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, 10))
        nn.init.constant_(self.plan_cls_branch[-1].bias, -4.595)  # p=0.01

    def forward(self, batch):
        agent_geo = batch["agent_geo"]            # (B, 50, 12)
        B, N, _ = agent_geo.shape
        if self.no_velocity:
            agent_geo = agent_geo.clone()
            agent_geo[..., 8:11] = 0.0

        # positions from raw geometry (encoders never see image features)
        agent_pos = self.box_pos_enc(agent_geo[..., :11])
        if self.use_image:
            agent_kv = self.agent_content(batch["agent_feat"])
        else:
            agent_kv = self.agent_content(agent_geo)
        if self.use_history:
            h = torch.cat([batch["agent_hist"],
                           batch["agent_hist_mask"][..., None]], dim=-1)
            agent_kv = agent_kv + self.hist_enc(h.flatten(2))

        map_geo = batch["map_geo"]                # (B, 10, 41)
        map_pos = self.map_pos_enc(map_geo[..., :40])
        if self.use_image:
            map_kv = self.map_content(batch["map_feat"])
        else:
            map_kv = self.map_content(map_geo)
        if self.no_map:
            map_kv = torch.zeros_like(map_kv)
            map_pos = torch.zeros_like(map_pos)

        # ego token: kinematic anchor (VY = measured velocity unless g4_none)
        ego_anchor = batch["ego_anchor"].clone()  # (B, 11)
        if not self.ego_vel_none:
            ego_anchor[:, VY] = batch["ego_status"][:, 6]
        ego_tok = self.box_pos_enc(ego_anchor[:, None, :])[:, 0]  # (B, D)
        ego_tok = ego_tok + self.ego_status_mlp(
            batch["ego_status"] if not self.ego_vel_none
            else torch.zeros_like(batch["ego_status"]))

        # plan queries: 18 = cmd(3) x mode(6)
        pa = self.plan_anchor[None].expand(B, -1, -1, -1, -1)   # (B,3,6,6,2)
        plan_pos = gen_sineembed_for_position(pa[..., -1, :])   # (B,3,6,D)
        plan_q = self.plan_anchor_encoder(plan_pos).flatten(1, 2)  # (B,18,D)
        plan_q = plan_q + ego_tok[:, None, :]

        q = torch.cat([ego_tok[:, None, :], plan_q], dim=1)     # (B,19,D)
        for layer in self.layers:
            q = layer(q, agent_kv, agent_pos, map_kv, map_pos)
        ego_out, plan_out = q[:, 0], q[:, 1:]                   # (B,D),(B,18,D)

        plan_cls = self.plan_cls_branch(plan_out).squeeze(-1)   # (B,18)
        plan_reg = self.plan_reg_branch(plan_out).reshape(
            B, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)
        plan_status = self.plan_status_branch(ego_out)          # (B,10)

        out = dict(plan_cls=plan_cls, plan_reg=plan_reg,
                   plan_status=plan_status)
        if self.forecast_aux:
            out["forecast"] = self.forecast_head(agent_kv + agent_pos).reshape(
                B, N, -1, 2)
        return out
