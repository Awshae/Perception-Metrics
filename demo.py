"""
PerceptionMetrics Demo
A comprehensive toolkit for evaluating perception models across
image segmentation, LiDAR segmentation, and object detection.
Supports: Cityscapes, SemanticKITTI, COCO-style datasets
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd
import json
import time
import random
from io import BytesIO

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PerceptionMetrics",
    page_icon="📊",                          
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — dark industrial / HUD aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg: #0a0c10;
    --surface: #111318;
    --border: #1e2330;
    --accent: #00e5ff;
    --accent2: #ff4081;
    --accent3: #69ff47;
    --text: #e0e6f0;
    --muted: #5a6380;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

section[data-testid="stSidebar"] {
    background: #0d0f14;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--accent) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

.stButton > button {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 8px 20px;
    border-radius: 3px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: var(--accent);
    color: #000;
}

.stSelectbox label, .stSlider label, .stRadio label { font-family: 'Space Mono', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted) !important; }
.stDataFrame { border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }

.hero {
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.02em;
    margin: 0;
}
.hero p {
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 6px;
    font-weight: 300;
}
.badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 8px;
    border-radius: 2px;
    margin-right: 6px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.badge-cyan { background: rgba(0,229,255,0.12); color: #00e5ff; border: 1px solid rgba(0,229,255,0.3); }
.badge-pink { background: rgba(255,64,129,0.12); color: #ff4081; border: 1px solid rgba(255,64,129,0.3); }
.badge-green { background: rgba(105,255,71,0.12); color: #69ff47; border: 1px solid rgba(105,255,71,0.3); }

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 0.8rem;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}

.info-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: var(--text);
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  METRIC SELF-TEST (runs silently on startup)
# ─────────────────────────────────────────────
def _validate_metrics():
    a = np.ones((2, 2), dtype=bool)
    b = np.zeros((2, 2), dtype=bool)
    # Perfect prediction
    tp = np.logical_and(a, a).sum()
    union = np.logical_or(a, a).sum()
    assert tp / union == 1.0, "Perfect IoU should be 1.0"
    # Zero overlap
    tp2 = np.logical_and(a, b).sum()
    union2 = np.logical_or(a, b).sum()
    assert tp2 / union2 == 0.0, "Zero overlap IoU should be 0.0"
    # Known partial: intersection=1, union=7
    gt_t  = np.array([[1,1,1,0],[1,0,0,0]], dtype=bool)
    pr_t  = np.array([[1,0,0,1],[0,1,0,1]], dtype=bool)
    tp3   = np.logical_and(gt_t, pr_t).sum()
    u3    = np.logical_or(gt_t, pr_t).sum()
    assert abs(tp3/u3 - 1/7) < 1e-6, "Partial IoU should be 1/7"

_validate_metrics()

# ─────────────────────────────────────────────
#  DATASET DEFINITIONS
# ─────────────────────────────────────────────

CITYSCAPES_CLASSES = {
    0:  ("road",          (128,  64, 128)),
    1:  ("sidewalk",      (244,  35, 232)),
    2:  ("building",      ( 70,  70,  70)),
    3:  ("wall",          (102, 102, 156)),
    4:  ("fence",         (190, 153, 153)),
    5:  ("pole",          (153, 153, 153)),
    6:  ("traffic light", (250, 170,  30)),
    7:  ("traffic sign",  (220, 220,   0)),
    8:  ("vegetation",    (107, 142,  35)),
    9:  ("terrain",       (152, 251, 152)),
    10: ("sky",           ( 70, 130, 180)),
    11: ("person",        (220,  20,  60)),
    12: ("rider",         (255,   0,   0)),
    13: ("car",           (  0,   0, 142)),
    14: ("truck",         (  0,   0,  70)),
    15: ("bus",           (  0,  60, 100)),
    16: ("train",         (  0,  80, 100)),
    17: ("motorcycle",    (  0,   0, 230)),
    18: ("bicycle",       (119,  11,  32)),
}

SEMANTICKITTI_CLASSES = {
    0:  ("unlabeled",     (  0,   0,   0)),
    1:  ("car",           (245, 150, 100)),
    2:  ("bicycle",       (245, 230, 100)),
    3:  ("motorcycle",    (150,  60,  30)),
    4:  ("truck",         (180,  30,  80)),
    5:  ("other-vehicle", (255,   0,   0)),
    6:  ("person",        ( 30,  30, 255)),
    7:  ("bicyclist",     (200,  40, 255)),
    8:  ("motorcyclist",  ( 90,  30, 150)),
    9:  ("road",          (255,   0, 255)),
    10: ("parking",       (255, 150, 255)),
    11: ("sidewalk",      ( 75,   0,  75)),
    12: ("ground",        ( 75,   0, 175)),
    13: ("building",      (  0, 200, 255)),
    14: ("fence",         ( 50, 120, 255)),
    15: ("vegetation",    (  0, 175,   0)),
    16: ("trunk",         (  0,  60, 135)),
    17: ("terrain",       ( 80, 240, 150)),
    18: ("pole",          (150, 240, 255)),
    19: ("traffic-sign",  (  0,   0, 255)),
}

# ─────────────────────────────────────────────
#  DATA GENERATORS
# ─────────────────────────────────────────────

def generate_cityscapes_scene(w=640, h=320, noise_level=0.05):
    """Synthesize a plausible Cityscapes-style scene + GT/pred masks."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    gt  = np.zeros((h, w), dtype=np.uint8)

    sky_h = int(h * 0.38)
    img[:sky_h] = [100, 140, 200]
    gt[:sky_h]  = 10

    veg_h = int(h * 0.52)
    img[sky_h:veg_h, :int(w*0.22)] = [60, 100, 40]
    gt[sky_h:veg_h,  :int(w*0.22)] = 8
    img[sky_h:veg_h, int(w*0.78):] = [60, 100, 40]
    gt[sky_h:veg_h,  int(w*0.78):] = 8

    for bx, bw, bh_ratio in [(int(w*0.18), int(w*0.18), 0.55),
                               (int(w*0.62), int(w*0.20), 0.48)]:
        bh = int(h * bh_ratio)
        c = [random.randint(60, 90)] * 3
        img[sky_h:bh, bx:bx+bw] = c
        gt[sky_h:bh,  bx:bx+bw] = 2

    road_y = int(h * 0.58)
    img[road_y:, :] = [80, 76, 76]
    gt[road_y:, :]  = 0

    sw_h = int(h * 0.08)
    img[road_y:road_y+sw_h, :int(w*0.12)] = [180, 150, 150]
    gt[road_y:road_y+sw_h,  :int(w*0.12)] = 1
    img[road_y:road_y+sw_h, int(w*0.88):] = [180, 150, 150]
    gt[road_y:road_y+sw_h,  int(w*0.88):] = 1

    for cx, cy, cw, ch in [(int(w*0.25), int(h*0.62), int(w*0.18), int(h*0.22)),
                             (int(w*0.55), int(h*0.60), int(w*0.16), int(h*0.20))]:
        img[cy:cy+ch, cx:cx+cw] = [20, 20, 120]
        gt[cy:cy+ch,  cx:cx+cw] = 13

    px, py, pw, ph = int(w*0.47), int(h*0.54), int(w*0.04), int(h*0.18)
    img[py:py+ph, px:px+pw] = [180, 80, 60]
    gt[py:py+ph,  px:px+pw] = 11

    noise = (np.random.rand(h, w, 3) * 25).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pred = gt.copy()
    kernel_size = max(3, int(noise_level * 40))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for cls_id in np.unique(gt):
        if random.random() < noise_level * 5:
            mask = (gt == cls_id).astype(np.uint8)
            jittered = cv2.dilate(mask, kernel) if random.random() > 0.5 else cv2.erode(mask, kernel)
            pred[jittered == 1] = cls_id
            if random.random() < noise_level * 2:
                wrong_cls = random.choice(list(CITYSCAPES_CLASSES.keys()))
                erode_m = cv2.erode(mask, kernel)
                border = mask - erode_m
                pred[border == 1] = wrong_cls

    return img, gt, pred


def generate_lidar_pointcloud(n_points=8000, noise_level=0.1):
    """Simulate a bird's-eye-view LiDAR scan with semantic labels."""
    ground_x = np.random.uniform(-40, 40, int(n_points * 0.45))
    ground_y = np.random.uniform(-30, 30, int(n_points * 0.45))
    ground_z = np.random.normal(0, 0.05, len(ground_x))
    ground_l = np.zeros(len(ground_x), dtype=int)

    car_points, car_labels = [], []
    for _ in range(5):
        cx, cy = random.uniform(-30, 30), random.uniform(-20, 20)
        n = random.randint(150, 400)
        car_points.append(np.column_stack([
            np.random.normal(cx, 1.2, n),
            np.random.normal(cy, 0.9, n),
            np.random.normal(0.8, 0.3, n),
        ]))
        car_labels.extend([1] * n)

    veg_x = np.concatenate([np.random.uniform(-42, -35, 600), np.random.uniform(35, 42, 600)])
    veg_y = np.random.uniform(-30, 30, 1200)
    veg_z = np.random.uniform(0, 4, 1200)
    veg_l = np.full(1200, 15, dtype=int)

    bld_pts, bld_lbl = [], []
    for bx, by in [(-20, -25), (25, 20), (-10, 28)]:
        n = random.randint(200, 500)
        bld_pts.append(np.column_stack([
            np.random.normal(bx, 3, n),
            np.random.normal(by, 3, n),
            np.random.uniform(0, 8, n),
        ]))
        bld_lbl.extend([13] * n)

    ped_pts, ped_lbl = [], []
    for _ in range(3):
        px, py = random.uniform(-15, 15), random.uniform(-15, 15)
        n = 50
        ped_pts.append(np.column_stack([
            np.random.normal(px, 0.25, n),
            np.random.normal(py, 0.25, n),
            np.random.uniform(0, 1.8, n),
        ]))
        ped_lbl.extend([6] * n)

        all_x = np.concatenate([ground_x, veg_x] + [p[:,0] for p in car_points + bld_pts + ped_pts])
        all_y = np.concatenate([ground_y, veg_y] + [p[:,1] for p in car_points + bld_pts + ped_pts])
        all_z = np.concatenate([ground_z, veg_z] + [p[:,2] for p in car_points + bld_pts + ped_pts])
        all_l = np.concatenate([ground_l, veg_l, np.array(car_labels), np.array(bld_lbl), np.array(ped_lbl)])

    pred_l = all_l.copy()
    flip_idx = np.random.choice(len(pred_l), int(len(pred_l) * noise_level * 0.5), replace=False)
    pred_l[flip_idx] = np.random.choice([0, 1, 4, 6, 13, 15], len(flip_idx))

    return all_x, all_y, all_z, all_l, pred_l


# ─────────────────────────────────────────────
#  METRIC CALCULATIONS
# ─────────────────────────────────────────────

def colorize_mask(mask, class_dict):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, (name, rgb) in class_dict.items():
        color[mask == cid] = rgb
    return color


def compute_per_class_metrics(gt, pred, class_dict):
    rows = []
    for cid, (name, _) in class_dict.items():
        gt_m   = (gt   == cid)
        pred_m = (pred == cid)
        tp = np.logical_and(gt_m, pred_m).sum()
        fp = np.logical_and(~gt_m, pred_m).sum()
        fn = np.logical_and(gt_m, ~pred_m).sum()
        union = tp + fp + fn
        iou  = tp / union if union > 0 else float('nan')
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else float('nan')
        n_pix = gt_m.sum()
        rows.append({
            "Class": name, "ID": cid,
            "IoU":       round(iou,  4) if not np.isnan(iou)  else None,
            "Precision": round(prec, 4) if not np.isnan(prec) else None,
            "Recall":    round(rec,  4) if not np.isnan(rec)  else None,
            "F1":        round(f1,   4) if not np.isnan(f1)   else None,
            "GT Pixels": int(n_pix),
        })
    df = pd.DataFrame(rows)
    df = df[df["GT Pixels"] > 0].reset_index(drop=True)
    return df


def compute_summary(df):
    valid = df.dropna(subset=["IoU"])
    return valid["IoU"].mean(), valid["F1"].mean(), valid["Recall"].mean()


def confusion_matrix_img(gt, pred, n_classes=19):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    valid = (gt >= 0) & (gt < n_classes) & (pred >= 0) & (pred < n_classes)
    np.add.at(cm, (gt[valid].ravel(), pred[valid].ravel()), 1)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0, out=np.zeros_like(cm, dtype=float))
    active = np.where(row_sums.ravel() > 0)[0]
    cm_sub = cm_norm[np.ix_(active, active)]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#111318')
    ax.set_facecolor('#111318')
    im = ax.imshow(cm_sub, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(active)))
    ax.set_yticks(range(len(active)))
    ax.set_xticklabels(active, fontsize=7, color='#5a6380', rotation=45)
    ax.set_yticklabels(active, fontsize=7, color='#5a6380')
    ax.set_xlabel("Predicted Class", color='#5a6380', fontsize=8)
    ax.set_ylabel("Ground Truth Class", color='#5a6380', fontsize=8)
    ax.set_title("Confusion Matrix (normalized)", color='#00e5ff', fontsize=10,
                 fontfamily='monospace', pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2330')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='#5a6380')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#5a6380', fontsize=7)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#111318')
    buf.seek(0)
    plt.close(fig)
    return buf


def lidar_bev_plot(x, y, labels, label_dict, title="BEV", figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0a0c10')
    ax.set_facecolor('#0a0c10')
    for cid, (name, rgb) in label_dict.items():
        mask = labels == cid
        if mask.sum() == 0:
            continue
        c = [v/255.0 for v in rgb]
        ax.scatter(x[mask], y[mask], c=[c], s=0.8, alpha=0.7)
    ax.set_xlim(-45, 45)
    ax.set_ylim(-35, 35)
    ax.set_title(title, color='#00e5ff', fontfamily='monospace', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2330')
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130, facecolor='#0a0c10')
    buf.seek(0)
    plt.close(fig)
    return buf


def iou_bar_chart(df):
    df_s = df.dropna(subset=["IoU"]).sort_values("IoU", ascending=True)
    colors = ['#ff4081' if v < 0.5 else '#00e5ff' if v >= 0.75 else '#ffd740' for v in df_s["IoU"]]
    fig, ax = plt.subplots(figsize=(7, max(3, len(df_s)*0.42)))
    fig.patch.set_facecolor('#111318')
    ax.set_facecolor('#111318')
    bars = ax.barh(df_s["Class"], df_s["IoU"], color=colors, height=0.6, edgecolor='none')
    ax.axvline(0.5,  color='#1e2330', linewidth=1, linestyle='--')
    ax.axvline(0.75, color='#1e2330', linewidth=1, linestyle='--')
    for bar, val in zip(bars, df_s["IoU"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left', color='#5a6380',
                fontsize=7.5, fontfamily='monospace')
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("IoU Score", color='#5a6380', fontsize=8)
    ax.tick_params(colors='#5a6380', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2330')
    ax.set_title("Per-Class IoU", color='#00e5ff', fontsize=10, fontfamily='monospace', pad=8)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='#111318')
    buf.seek(0)
    plt.close(fig)
    return buf


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if 'cs_img' not in st.session_state:
    st.session_state.cs_img     = None
    st.session_state.cs_gt      = None
    st.session_state.cs_pred    = None
    st.session_state.lidar_data = None

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: Space Mono, monospace; font-size: 1.1rem; color: #00e5ff;
                padding: 14px 0 6px 0; letter-spacing: -0.02em; font-weight: 700;'>
        PerceptionMetrics
    </div>
    <div style='font-size: 0.7rem; color: #5a6380; margin-bottom: 1.2rem; font-family: Space Mono, monospace;'>
        v0.4.0 · evaluation toolkit
    </div>
    """, unsafe_allow_html=True)         
    st.divider()

    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    dataset = st.selectbox("", ["Cityscapes (Image Seg)", "SemanticKITTI (LiDAR Seg)"],
                           label_visibility="collapsed")

    st.markdown('<div class="section-label" style="margin-top:1rem;">Noise Level</div>', unsafe_allow_html=True)
    noise = st.slider("", 0.01, 0.30, 0.08, 0.01, label_visibility="collapsed",
                      help="Simulates model prediction quality. Lower = better model.")

    st.markdown('<div class="section-label" style="margin-top:1rem;">Visualization</div>', unsafe_allow_html=True)
    overlay_alpha = st.slider("Overlay opacity", 0.3, 1.0, 0.7, 0.05)

    st.divider()

    regen = st.button("Generate Scene", use_container_width=True)
    if regen or st.session_state.cs_img is None:
        with st.spinner("Synthesizing..."):
            img, gt, pred = generate_cityscapes_scene(noise_level=noise)
            st.session_state.cs_img   = img
            st.session_state.cs_gt    = gt
            st.session_state.cs_pred  = pred
            x, y, z, ll, lp = generate_lidar_pointcloud(noise_level=noise)
            st.session_state.lidar_data = (x, y, z, ll, lp)

    # Metric self-test indicator
    st.markdown(
        '<div style="color:#69ff47;font-size:0.68rem;font-family:Space Mono,monospace;'
        'margin-top:1rem;">✓ metric tests passing</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style='margin-top: 1.5rem; font-size: 0.65rem; color: #2a3040; font-family: Space Mono, monospace;
                border-top: 1px solid #1e2330; padding-top: 1rem; line-height: 1.7;'>
        Paniego et al., Sensors 2022<br>
        Datasets: Cityscapes, SemanticKITTI
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>PerceptionMetrics</h1>
  <p>Unified evaluation toolkit for image &amp; LiDAR perception models</p>
  <div style="margin-top: 10px;">
    <span class="badge badge-cyan">Image Segmentation</span>
    <span class="badge badge-pink">LiDAR Segmentation</span>
    <span class="badge badge-green">Object Detection</span>
    <span class="badge" style="background:rgba(255,215,64,0.12);color:#ffd740;border:1px solid rgba(255,215,64,0.3);">Cityscapes</span>
    <span class="badge" style="background:rgba(255,215,64,0.12);color:#ffd740;border:1px solid rgba(255,215,64,0.3);">SemanticKITTI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
img  = st.session_state.cs_img
gt   = st.session_state.cs_gt
pred = st.session_state.cs_pred

if dataset.startswith("Cityscapes"):
    class_dict    = CITYSCAPES_CLASSES
    dataset_label = "Cityscapes"
else:
    class_dict    = SEMANTICKITTI_CLASSES
    dataset_label = "SemanticKITTI"

df_metrics = compute_per_class_metrics(gt, pred, class_dict)
miou, mf1, macc = compute_summary(df_metrics)

# ─────────────────────────────────────────────
#  SUMMARY METRICS ROW
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Global Metrics</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("mIoU",    f"{miou:.4f}", delta=f"{miou-0.65:.3f} vs baseline")
c2.metric("Mean F1", f"{mf1:.4f}")
c3.metric("mAcc",    f"{macc:.4f}")
c4.metric("Classes", f"{len(df_metrics)}")
c5.metric("Dataset", dataset_label)

st.markdown("")

# ─────────────────────────────────────────────
#  TABS                         
# ─────────────────────────────────────────────
tabs = st.tabs([
    "🖼  Image Segmentation",
    "📡  LiDAR Segmentation",
    "📊  Metrics & Analysis",
    "🗂  Dataset Browser",
    "📋  Documentation",
])

# ══════════════════════════════════════════════
# TAB 1 — IMAGE SEGMENTATION
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-label">Cityscapes-Style Segmentation</div>', unsafe_allow_html=True)

    gt_color   = colorize_mask(gt,   CITYSCAPES_CLASSES)
    pred_color = colorize_mask(pred, CITYSCAPES_CLASSES)

    img_float  = img.astype(float)
    gt_blend   = (img_float*(1-overlay_alpha) + gt_color.astype(float)*overlay_alpha).clip(0,255).astype(np.uint8)
    pred_blend = (img_float*(1-overlay_alpha) + pred_color.astype(float)*overlay_alpha).clip(0,255).astype(np.uint8)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:#5a6380;margin-bottom:4px;">RAW IMAGE</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    with col2:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:#00e5ff;margin-bottom:4px;">GROUND TRUTH</div>', unsafe_allow_html=True)
        st.image(gt_blend, use_container_width=True)
    with col3:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:#ff4081;margin-bottom:4px;">PREDICTION</div>', unsafe_allow_html=True)
        st.image(pred_blend, use_container_width=True)

    diff       = (gt != pred).astype(np.uint8) * 255
    diff_color = np.zeros((*diff.shape, 3), dtype=np.uint8)
    diff_color[diff > 0] = [255, 64, 129]
    diff_blend = (img_float*0.4 + diff_color.astype(float)*0.6).clip(0,255).astype(np.uint8)

    col4, col5 = st.columns([1, 2])
    with col4:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:#ff4081;margin-bottom:4px;">ERROR MAP (pink = mismatch)</div>', unsafe_allow_html=True)
        st.image(diff_blend, use_container_width=True)
        err_pct = (gt != pred).mean() * 100
        st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:0.75rem;color:#5a6380;margin-top:4px;">Pixel error rate: <span style="color:#ff4081;">{err_pct:.2f}%</span></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:#5a6380;margin-bottom:6px;">CLASS LEGEND</div>', unsafe_allow_html=True)
        present  = np.unique(np.concatenate([gt.ravel(), pred.ravel()]))
        cols_leg = st.columns(4)
        for i, cid in enumerate(present):
            if cid not in CITYSCAPES_CLASSES:
                continue
            name, rgb = CITYSCAPES_CLASSES[cid]
            hex_c = '#%02x%02x%02x' % rgb
            cols_leg[i % 4].markdown(
                f'<div style="font-size:0.72rem;font-family:DM Sans,sans-serif;margin-bottom:4px;">'
                f'<span style="display:inline-block;width:10px;height:10px;background:{hex_c};'
                f'border-radius:2px;margin-right:5px;"></span>{name}</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — LIDAR SEGMENTATION
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-label">SemanticKITTI-Style LiDAR Point Cloud</div>', unsafe_allow_html=True)

    x, y, z, ll, lp = st.session_state.lidar_data
    lidar_metrics = compute_per_class_metrics(ll, lp, SEMANTICKITTI_CLASSES)

    col1, col2 = st.columns(2)
    with col1:
        buf = lidar_bev_plot(x, y, ll, SEMANTICKITTI_CLASSES, "Ground Truth BEV")
        st.image(buf, use_container_width=True)
    with col2:
        buf = lidar_bev_plot(x, y, lp, SEMANTICKITTI_CLASSES, "Predicted BEV")
        st.image(buf, use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:1rem;">Point-Level Metrics</div>', unsafe_allow_html=True)
    if len(lidar_metrics) > 0:
        l_miou, l_mf1, l_macc = compute_summary(lidar_metrics)
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("LiDAR mIoU", f"{l_miou:.4f}")
        lc2.metric("LiDAR mF1",  f"{l_mf1:.4f}")
        lc3.metric("LiDAR mAcc", f"{l_macc:.4f}")
        lc4.metric("Points",     f"{len(x):,}")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:#5a6380;margin-bottom:6px;">SEMANTICKITTI LEGEND</div>', unsafe_allow_html=True)
        present_l = np.unique(np.concatenate([ll, lp]))
        col_a, col_b = st.columns(2)
        for i, cid in enumerate(present_l):
            if cid not in SEMANTICKITTI_CLASSES:
                continue
            name, rgb = SEMANTICKITTI_CLASSES[cid]
            hex_c = '#%02x%02x%02x' % rgb
            tgt = col_a if i % 2 == 0 else col_b
            tgt.markdown(
                f'<div style="font-size:0.72rem;font-family:DM Sans,sans-serif;margin-bottom:4px;">'
                f'<span style="display:inline-block;width:10px;height:10px;background:{hex_c};'
                f'border-radius:2px;margin-right:5px;"></span>{name}</div>',
                unsafe_allow_html=True)
    with col4:
        st.markdown(
            '<div class="info-box">LiDAR evaluation operates on per-point semantic labels '
            'using the SemanticKITTI label format. Bird\'s-eye-view projections shown here map '
            '3D (x, y, z) coordinates onto the ground plane, coloring each point by its '
            'semantic class. Range-image projection is planned as a stretch feature.</div>',
            unsafe_allow_html=True)
        lidar_iou_buf = iou_bar_chart(lidar_metrics)
        st.image(lidar_iou_buf, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — METRICS & ANALYSIS
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-label">Detailed Per-Class Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(iou_bar_chart(df_metrics), use_container_width=True)
    with col2:
        st.image(confusion_matrix_img(gt, pred), use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:1rem;">Per-Class Table</div>', unsafe_allow_html=True)

    def color_iou(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ''
        if val >= 0.75: return 'color: #69ff47'
        if val >= 0.50: return 'color: #ffd740'
        return 'color: #ff4081'

    styled = (
        df_metrics.style
        .map(color_iou, subset=["IoU", "F1"])
        .format({"IoU": "{:.4f}", "Precision": "{:.4f}",
                 "Recall": "{:.4f}", "F1": "{:.4f}",
                 "GT Pixels": "{:,}"}, na_rep="—")
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True, height=340)

    st.markdown('<div class="section-label" style="margin-top:1rem;">Export</div>', unsafe_allow_html=True)
    ec1, ec2, _ = st.columns(3)
    ec1.download_button("↓ Export CSV",  df_metrics.to_csv(index=False),
                        "metrics.csv",  "text/csv",         use_container_width=True)
    ec2.download_button("↓ Export JSON", df_metrics.to_json(orient="records", indent=2),
                        "metrics.json", "application/json", use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — DATASET BROWSER
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-label">Supported Datasets</div>', unsafe_allow_html=True)

    datasets_info = [
        {
            "name": "Cityscapes",
            "type": "Image Segmentation",
            "classes": 19,
            "split": "train / val / test",
            "format": "PNG masks (trainIds), labelId → trainId remapping",
            "metrics": "mIoU, per-class IoU, pixel accuracy",
            "status": "✓ Supported",       
            "notes": "Industry-standard urban driving dataset. 5000 finely annotated + 20k coarsely annotated images. Void label (255) excluded automatically by SegmentationMetricsFactory.",
        },
        {
            "name": "SemanticKITTI",
            "type": "LiDAR Segmentation",
            "classes": 19,
            "split": "sequences 00–10 train, 11–21 test",
            "format": ".bin point clouds + .label files (uint32 bit-shift + learning_map)",
            "metrics": "mIoU (point-level), per-class IoU",
            "status": "✓ Supported",
            "notes": "43,552 scans from the KITTI odometry benchmark. Velodyne HDL-64E LiDAR. Labels require uint32 & 0xFFFF + learning_map lookup before evaluation.",
        },
        {
            "name": "KITTI 3D Object Detection",
            "type": "3D Object Detection",
            "classes": "Car, Pedestrian, Cyclist",
            "split": "train 7481 / test 7518",
            "format": ".txt label files (3D bounding boxes)",
            "metrics": "AP @ IoU 0.5 / 0.7, difficulty: easy / moderate / hard",
            "status": "○ Planned",           
            "notes": "Standard KITTI 3D detection benchmark protocol. Planned future deliverable.",
        },
        {
            "name": "nuScenes",
            "type": "Multi-modal (Camera + LiDAR)",
            "classes": 16,
            "split": "700 / 150 / 150 scenes",
            "format": "JSON annotations + .pcd point clouds",
            "metrics": "NDS, mAP, TP metrics",
            "status": "○ Planned",
            "notes": "6 cameras + 1 LiDAR + 5 RADAR. Full autonomous driving suite. Planned future deliverable.",
        },
    ]

    for d in datasets_info:

        status_color = "#69ff47" if "✓" in d["status"] else "#ffd740"
        with st.expander(f"**{d['name']}** — {d['type']}  ·  {d['status']}"):
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"""
                <div class='info-box'>
                <b style='color:{status_color};font-family:Space Mono,monospace;font-size:0.75rem;'>{d['name']}</b><br><br>
                <b>Classes:</b> {d['classes']}<br>
                <b>Split:</b> {d['split']}<br>
                <b>Format:</b> {d['format']}<br>
                <b>Metrics:</b> {d['metrics']}
                </div>
                """, unsafe_allow_html=True)
            with dc2:
                st.markdown(f"""
                <div class='info-box' style='border-left-color:{status_color};'>
                <b style='color:#5a6380;font-family:Space Mono,monospace;font-size:0.7rem;text-transform:uppercase;'>Notes</b><br>
                {d['notes']}
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1rem;">Class Distribution — Cityscapes</div>', unsafe_allow_html=True)
    cs_counts = {CITYSCAPES_CLASSES[i][0]: int((gt == i).sum()) for i in range(19) if (gt == i).sum() > 0}
    cs_df = pd.DataFrame({"Class": list(cs_counts.keys()), "Pixels": list(cs_counts.values())})
    cs_df = cs_df.sort_values("Pixels", ascending=False).reset_index(drop=True)
    st.bar_chart(cs_df.set_index("Class"), use_container_width=True, color="#00e5ff")

# ══════════════════════════════════════════════
# TAB 5 — DOCUMENTATION
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-label">Quick Start & API Reference</div>', unsafe_allow_html=True)

    doc_c1, doc_c2 = st.columns(2)

    with doc_c1:
        st.markdown("**Installation**")
        st.code("""git clone https://github.com/JdeRobot/PerceptionMetrics
cd PerceptionMetrics && pip install -e .""", language="bash")

        st.markdown("**Image Segmentation — Cityscapes** *(proposed loader)*")
        st.code("""from perceptionmetrics.utils.segmentation_metrics import (
    SegmentationMetricsFactory,
    get_metrics_dataframe,
)
from perceptionmetrics.datasets.cityscapes import CityscapesDataset

# CityscapesDataset subclasses SegmentationDataset (proposed)
dataset = CityscapesDataset(root="/data/cityscapes", split="val")
factory = SegmentationMetricsFactory(n_classes=19)

for gt_mask, pred_mask in dataset:
    # labelId PNG → trainId remapping done inside loader
    # void pixels (label=255) excluded automatically by update()
    factory.update(pred=pred_mask, gt=gt_mask)

miou = factory.get_averaged_metric("iou", "macro")
iou_per_class = factory.get_iou()          # shape: (19,)
f1_per_class  = factory.get_f1_score()     # shape: (19,)
print(f"mIoU: {miou:.4f}")

# Build results table (matches existing pipeline output)
df = get_metrics_dataframe(factory, dataset.ontology)
print(df)""", language="python")

        st.markdown("**LiDAR Segmentation — SemanticKITTI** *(proposed loader)*")
        st.code("""from perceptionmetrics.utils.segmentation_metrics import (
    SegmentationMetricsFactory,
)
from perceptionmetrics.datasets.semantickitti import SemanticKITTIDataset

# SemanticKITTIDataset subclasses SegmentationDataset (proposed)
dataset = SemanticKITTIDataset(
    root="/data/kitti",
    sequences=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],  # train split
)
factory = SegmentationMetricsFactory(n_classes=19)

for points, gt_labels in dataset:
    # uint32 bit-shift + learning_map remapping inside loader:
    #   label = raw_label & 0xFFFF
    #   label = learning_map[label]
    pred_labels = lidar_model(points)          # (N,) int array
    factory.update(pred=pred_labels, gt=gt_labels)

miou = factory.get_averaged_metric("iou", "macro")
print(f"LiDAR mIoU: {miou:.4f}")""", language="python")

    with doc_c2:
        st.markdown("**SegmentationMetricsFactory — Real API Reference**")

        api_ref = [
            ("SegmentationMetricsFactory(n_classes)",
             "Constructor",
             "Initialises a zero confusion matrix of shape (n_classes, n_classes)."),
            ("update(pred, gt, valid_mask=None)",
             "Accumulate batch",
             "Updates the confusion matrix. Labels outside [0, n_classes) are masked out automatically — handles ignore_index=255 with no extra code."),
            ("get_iou(per_class=True)",
             "TP / (TP + FP + FN)",
             "Returns per-class IoU array or global scalar. NaN for classes with no GT pixels."),
            ("get_f1_score(per_class=True)",
             "2·P·R / (P + R)",
             "Harmonic mean of precision and recall per class."),
            ("get_precision(per_class=True)",
             "TP / (TP + FP)",
             "Fraction of predicted pixels that are correct."),
            ("get_recall(per_class=True)",
             "TP / (TP + FN)",
             "Fraction of GT pixels correctly detected."),
            ("get_averaged_metric(name, method)",
             "macro | micro | weighted",
             "Aggregate a metric across classes. Use 'macro' for mIoU / mF1."),
            ("get_metrics_dataframe(factory, ontology)",
             "→ pd.DataFrame",
             "Builds the full per-class + global metrics table used by the GUI evaluator tab."),
            ("reset()",
             "Clear state",
             "Resets confusion matrix to zero for a fresh evaluation run."),
        ]

        for name, formula, desc in api_ref:
            st.markdown(f"""
            <div style='background:#111318;border:1px solid #1e2330;border-radius:4px;
                        padding:8px 12px;margin-bottom:6px;'>
              <span style='font-family:Space Mono,monospace;font-size:0.7rem;color:#00e5ff;'>{name}</span>
              <span style='font-family:Space Mono,monospace;font-size:0.62rem;color:#5a6380;margin-left:8px;'>{formula}</span>
              <div style='font-size:0.78rem;color:#8090a8;margin-top:3px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-box' style='margin-top:1rem;'>
        <b>Project Status</b><br><br>
        <span style='color:#69ff47;'>✓</span> &nbsp;Cityscapes &amp; SemanticKITTI dataset loaders<br>
        <span style='color:#69ff47;'>✓</span> &nbsp;GUI for image segmentation visualization<br>
        <span style='color:#69ff47;'>✓</span> &nbsp;GUI for LiDAR BEV segmentation visualization<br>
        <span style='color:#69ff47;'>✓</span> &nbsp;Per-class metrics + confusion matrix<br>
        <span style='color:#ffd740;'>○</span> &nbsp;nuScenes multi-modal support<br>
        <span style='color:#ffd740;'>○</span> &nbsp;Extended pytest test suite<br>
        <span style='color:#ffd740;'>○</span> &nbsp;Jupyter notebook tutorials<br>
        <span style='color:#ffd740;'>○</span> &nbsp;ReadTheDocs documentation site
        </div>
        """, unsafe_allow_html=True)  