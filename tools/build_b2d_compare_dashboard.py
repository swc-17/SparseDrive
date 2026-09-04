"""Merge a second model's B2D predictions into the existing zero-shot dashboard.

Reads the existing viz_4clip_2hz_vk/preds.json (model A: NAVSIM-trained
Sparse4D, zero-shot on B2D) and a second model's native-B2D results.pkl +
matching infos pkl (model B, e.g. the SDv1 B2D-trained checkpoint), joins them
by (clip, frame-index-within-clip), and writes a new preds_compare.json +
index_compare.html with a model A/B/both toggle next to the existing
confidence sliders.

Usage:
    python tools/build_b2d_compare_dashboard.py \
        --results-b work_dirs/b2dval_sdv1/results.pkl \
        --infos-b work_dirs/b2dval_sdv1/b2d_4clip_infos_val_2hz.pkl \
        --label-b "SDv1 (B2D-trained)" \
        --dashboard-dir work_dirs/b2dval_navsim/viz_4clip_2hz_vk
"""
import argparse
import io
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch

# B2D3DDataset class_names / map_class_names (projects/configs/sparsedrive_b2d_stage1.py)
DET_CLASSES_B = ["car", "van", "truck", "bicycle", "traffic_sign",
                  "traffic_cone", "traffic_light", "pedestrian", "others"]
MAP_CLASSES_B = ["Broken", "Solid", "SolidSolid", "Center", "TrafficLight", "StopSign"]


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def build_frame_key_map(infos_b):
    """(clip, native frame_idx) -> dataset index, matching preds.json's 'frame' field
    (which is the raw frame_idx, not a re-enumerated 0..N-1 position)."""
    key_to_idx = {}
    for idx, info in enumerate(infos_b):
        clip = info["folder"].split("/", 1)[-1] if "/" in info["folder"] else info["folder"]
        key_to_idx[(clip, info["frame_idx"])] = idx
    return key_to_idx


def convert_frame(det, det_classes, score_thr_keep=0.05):
    boxes_3d = _to_list(det["boxes_3d"])  # (N, 10): x,y,z,l,w,h,yaw,vx,vy,+1
    scores = _to_list(det["scores_3d"])
    labels = _to_list(det["labels_3d"])
    boxes = []
    for box, score, label in zip(boxes_3d, scores, labels):
        if score < score_thr_keep:
            continue
        x, y, z, l, w, h, yaw = box[:7]
        boxes.append([x, y, z, l, w, h, yaw, float(score), int(label)])

    map_out = []
    if "vectors" in det and det["vectors"] is not None:
        vec_scores = _to_list(det["scores"])
        vec_labels = _to_list(det["labels"])
        for pts, score, label in zip(det["vectors"], vec_scores, vec_labels):
            if score < score_thr_keep:
                continue
            pts = _to_list(pts) if not isinstance(pts, list) else pts
            map_out.append([float(score), int(label), pts])
    return boxes, map_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard-dir",
                     default="work_dirs/b2dval_navsim/viz_4clip_2hz_vk")
    ap.add_argument("--results-b", required=True,
                     help="native-B2D results.pkl for model B")
    ap.add_argument("--infos-b", required=True,
                     help="infos pkl (native B2D schema) matching results-b's order")
    ap.add_argument("--label-b", default="model B")
    ap.add_argument("--score-thr-keep", type=float, default=0.05,
                     help="drop boxes/vectors below this score before writing JSON "
                          "(the dashboard slider only ever raises the threshold further)")
    args = ap.parse_args()

    base_path = os.path.join(args.dashboard_dir, "preds.json")
    with open(base_path) as f:
        meta_a = json.load(f)

    results_b = _load_pkl(args.results_b)
    infos_b = _load_pkl(args.infos_b)
    key_to_idx = build_frame_key_map(infos_b)

    # (clip, native frame_idx) -> (boxes, map) for model B
    b_lookup = {}
    for key, dataset_idx in key_to_idx.items():
        det = results_b[dataset_idx]["img_bbox"]
        boxes, map_out = convert_frame(det, DET_CLASSES_B, args.score_thr_keep)
        b_lookup[key] = (boxes, map_out)

    matched = 0
    for row in meta_a:
        key = (row["clip"], row["frame"])
        if key in b_lookup:
            boxes2, map2 = b_lookup[key]
            row["boxes2"] = boxes2
            row["map2"] = map2
            matched += 1
        else:
            row["boxes2"] = []
            row["map2"] = []
    print(f"[build_b2d_compare_dashboard] matched {matched}/{len(meta_a)} frames "
          f"against {len(results_b)} model-B results")

    out_json = os.path.join(args.dashboard_dir, "preds_compare.json")
    with open(out_json, "w") as f:
        json.dump(meta_a, f)
    print(f"[build_b2d_compare_dashboard] wrote {out_json}")

    write_index_html(args.dashboard_dir, args.label_b)


def write_index_html(dashboard_dir, label_b):
    html = INDEX_TEMPLATE.replace("__LABEL_B__", json.dumps(label_b)) \
                          .replace("__DET_CLASSES_B__", json.dumps(DET_CLASSES_B)) \
                          .replace("__MAP_CLASSES_B__", json.dumps(MAP_CLASSES_B))
    out_path = os.path.join(dashboard_dir, "index_compare.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[build_b2d_compare_dashboard] wrote {out_path}")


INDEX_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>B2D zero-shot vs SDv1</title>
<style>
body{margin:0;background:#111;color:#ddd;font:14px system-ui}
#bar{display:flex;gap:14px;align-items:center;padding:8px 12px;background:#1c1c1c;flex-wrap:wrap}
#bar label{display:flex;gap:6px;align-items:center}
select,button{background:#2a2a2a;color:#ddd;border:1px solid #444;padding:4px 8px}
input[type=range]{accent-color:#fa5}
#frame{width:320px}#det,#map{width:160px}
#wrap{position:relative;width:100%}
#img{width:100%;display:block}
#ov{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
#info{padding:6px 12px;color:#aaa}.k{color:#8f8}.o{color:#fa5}.m{color:#f5a}.o2{color:#5cf}.m2{color:#c9f}b{color:#fff}
</style></head><body>
<div id="bar"><b>B2D val: zero-shot NAVSIM vs __LABEL_B__</b>
<label>clip <select id="clip"><option value="">all</option></select></label>
<label>model <select id="model"><option value="a">A: zero-shot (orange)</option><option value="b">B: __LABEL_B__ (blue)</option><option value="both" selected>both</option></select></label>
<label>det score &ge; <input type="range" id="det" min="0.05" max="0.95" step="0.01" value="0.30"><span id="detv">0.30</span></label>
<label>map score &ge; <input type="range" id="map" min="0.05" max="0.95" step="0.01" value="0.30"><span id="mapv">0.30</span></label>
<label><input type="checkbox" id="lbl" checked> labels</label>
<label><input type="checkbox" id="showBoxes" checked> boxes</label>
<label><input type="checkbox" id="showMap" checked> map</label>
<button id="prev">&#9664;</button><button id="play">&#9654; play</button><button id="next">&#9654;</button>
<label>frame <input type="range" id="frame" min="0" value="0"><span id="pos"></span></label></div>
<div id="info"></div>
<div id="wrap"><img id="img"><canvas id="ov"></canvas></div>
<script>
const CLASSES=["vehicle","pedestrian","bicycle","traffic_cone","barrier","czone_sign","generic_object"];
const CLASSES_B=__DET_CLASSES_B__;
const MAPC={0:"#f0f",1:"#fa5",2:"#ff5"};
const MAP_CLASSES_B=__MAP_CLASSES_B__;
const MAPC_B={0:"#8cf",1:"#48f",2:"#04f",3:"#0cf",4:"#0ff",5:"#c9f"};
const $=id=>document.getElementById(id);
let meta=[],view=[],idx=0,timer=null;
fetch('preds_compare.json').then(r=>r.json()).then(m=>{meta=m;[...new Set(m.map(r=>r.clip))].forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;$('clip').appendChild(o)});rebuild()});
function rebuild(){const c=$('clip').value;view=meta.filter(r=>!c||r.clip===c);idx=Math.min(idx,Math.max(view.length-1,0));$('frame').max=Math.max(view.length-1,0);show()}
function corners(b){const [x,y,z,l,w,h,yaw]=b,c=Math.cos(yaw),s=Math.sin(yaw),out=[];for(const sx of[1,-1])for(const sy of[1,-1])for(const sz of[1,-1]){const dx=sx*l/2,dy=sy*w/2;out.push([x+c*dx-s*dy,y+s*dx+c*dy,z+sz*h/2])}return out}
const EDGES=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
const TOP=[0,2,6,4];
function color(s){const t=Math.max(0,Math.min(1,(s-0.3)/0.7));return `rgb(255,${Math.round(80+175*t)},0)`}
function colorB(s){const t=Math.max(0,Math.min(1,(s-0.3)/0.7));return `rgb(0,${Math.round(120+100*t)},255)`}
function proj(P,p){const x=P[0]*p[0]+P[1]*p[1]+P[2]*p[2]+P[3],y=P[4]*p[0]+P[5]*p[1]+P[6]*p[2]+P[7],z=P[8]*p[0]+P[9]*p[1]+P[10]*p[2]+P[11];return [x/Math.max(z,1e-3),y/Math.max(z,1e-3),z]}
function drawBoxesCam(ctx,t,boxes,colorFn,classes,showLbl){
  for(const b of boxes){const cs=corners(b).map(p=>proj(t.P,p));if(cs.every(p=>p[2]<0.1))continue;
    const uv=cs.map(p=>[t.ox+p[0]*t.sx,t.oy+p[1]*t.sy]);
    if(!uv.some((p,i)=>cs[i][2]>0.1&&p[0]>t.ox-t.w&&p[0]<t.ox+2*t.w&&p[1]>t.oy-t.h&&p[1]<t.oy+2*t.h))continue;
    ctx.strokeStyle=ctx.fillStyle=colorFn(b[7]);ctx.beginPath();
    for(const [a,c] of EDGES){if(cs[a][2]<0.1||cs[c][2]<0.1)continue;ctx.moveTo(uv[a][0],uv[a][1]);ctx.lineTo(uv[c][0],uv[c][1])}ctx.stroke();
    if(showLbl){let top=null;uv.forEach((p,i)=>{if(cs[i][2]>0.1&&(!top||p[1]<top[1]))top=p});if(top)ctx.fillText(`${classes[b[8]].slice(0,4)} ${b[7].toFixed(2)}`,top[0],top[1]-4)}}
}
function drawBoxesBev(ctx,px,boxes,colorFn){
  ctx.lineWidth=1.5;
  for(const b of boxes){const cs=corners(b);ctx.strokeStyle=colorFn(b[7]);ctx.beginPath();TOP.forEach((i,k)=>{const q=px(cs[i]);k?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});ctx.closePath();
    const c0=px(b),c1=px([b[0]+2*Math.cos(b[6]),b[1]+2*Math.sin(b[6])]);ctx.moveTo(c0[0],c0[1]);ctx.lineTo(c1[0],c1[1]);ctx.stroke()}
}
function draw(){
  const r=view[idx];if(!r)return;const img=$('img'),cv=$('ov');
  const W=img.naturalWidth,H=img.naturalHeight;if(!W)return;cv.width=W;cv.height=H;
  const ctx=cv.getContext('2d');ctx.clearRect(0,0,W,H);ctx.lineWidth=2;ctx.font="15px system-ui";
  const dthr=+$('det').value,mthr=+$('map').value,showLbl=$('lbl').checked,mdl=$('model').value;
  const showA=mdl==='a'||mdl==='both', showB=mdl==='b'||mdl==='both';
  const showBoxes=$('showBoxes').checked, showMap=$('showMap').checked;
  const boxesA=showA?r.boxes.filter(b=>b[7]>=dthr):[];
  const boxesB=showB?(r.boxes2||[]).filter(b=>b[7]>=dthr):[];
  if(showBoxes)for(const t of r.tiles){
    ctx.save();ctx.beginPath();ctx.rect(t.ox,t.oy,t.w,t.h);ctx.clip();
    drawBoxesCam(ctx,t,boxesA,color,CLASSES,showLbl);
    drawBoxesCam(ctx,t,boxesB,colorB,CLASSES_B,showLbl);
    ctx.restore()}
  const bv=r.bev,sc=bv.size/(2*bv.rng),cx=bv.ox+bv.size/2,cy=bv.size/2,px=p=>[cx+p[0]*sc,cy-p[1]*sc];
  ctx.save();ctx.beginPath();ctx.rect(bv.ox,0,bv.size,bv.size);ctx.clip();
  if(showMap&&showA)for(const m of r.map){if(m[0]<mthr)continue;ctx.strokeStyle=MAPC[m[1]]||'#ccc';ctx.lineWidth=2;ctx.beginPath();m[2].forEach((p,i)=>{const q=px(p);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});ctx.stroke()}
  if(showMap&&showB)for(const m of (r.map2||[])){if(m[0]<mthr)continue;ctx.strokeStyle=MAPC_B[m[1]]||'#8cf';ctx.lineWidth=2;ctx.setLineDash([4,3]);ctx.beginPath();m[2].forEach((p,i)=>{const q=px(p);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});ctx.stroke();ctx.setLineDash([])}
  if(showBoxes){drawBoxesBev(ctx,px,boxesA,color);drawBoxesBev(ctx,px,boxesB,colorB)}
  ctx.restore();
  let hitA=0;for(const g of r.gt){if(boxesA.some(b=>Math.hypot(b[0]-g[0],b[1]-g[1])<2))hitA++}
  let hitB=0;for(const g of r.gt){if(boxesB.some(b=>Math.hypot(b[0]-g[0],b[1]-g[1])<2))hitB++}
  const nm=r.map.filter(m=>m[0]>=mthr).length, nm2=(r.map2||[]).filter(m=>m[0]>=mthr).length;
  $('info').innerHTML=`<b>${r.clip}</b> frame ${r.frame} · GT <span class=k>${r.gt.length}</span> · `+
    `A det&ge;${dthr.toFixed(2)}: <span class=o>${boxesA.length}</span> recall@2m <span class=o>${hitA}/${r.gt.length}</span> map <span class=m>${nm}</span> · `+
    `B det&ge;${dthr.toFixed(2)}: <span class=o2>${boxesB.length}</span> recall@2m <span class=o2>${hitB}/${r.gt.length}</span> map <span class=m2>${nm2}</span> · `+
    `<span style="color:#666">${r.img}</span>`;
}
function show(){const r=view[idx];if(!r){$('info').textContent='no frames';return}$('frame').value=idx;$('pos').textContent=`${idx+1}/${view.length}`;
  const img=$('img');if(img.dataset.src!==r.img){img.dataset.src=r.img;img.onload=draw;img.src=r.img}else draw()}
function step(d){if(!view.length)return;idx=(idx+d+view.length)%view.length;show()}
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('frame').oninput=e=>{idx=+e.target.value;show()};
$('clip').onchange=()=>{idx=0;rebuild()};
$('model').onchange=draw;
$('det').oninput=e=>{$('detv').textContent=(+e.target.value).toFixed(2);draw()};
$('map').oninput=e=>{$('mapv').textContent=(+e.target.value).toFixed(2);draw()};
$('lbl').onchange=draw;
$('showBoxes').onchange=draw;
$('showMap').onchange=draw;
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='&#9654; play'}else{timer=setInterval(()=>step(1),400);$('play').textContent='&#9208; pause'}};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'&&e.target.type==='range'&&e.target.id!=='frame')return;
  if(e.key==='ArrowRight')step(1);else if(e.key==='ArrowLeft')step(-1);else if(e.key===' '){e.preventDefault();$('play').click()}
  else if(e.key==='ArrowUp'){e.preventDefault();$('det').value=Math.min(0.95,+$('det').value+0.05);$('det').oninput({target:$('det')})}
  else if(e.key==='ArrowDown'){e.preventDefault();$('det').value=Math.max(0.05,+$('det').value-0.05);$('det').oninput({target:$('det')})}});
window.addEventListener('resize',draw);
</script></body></html>
"""


if __name__ == "__main__":
    main()
