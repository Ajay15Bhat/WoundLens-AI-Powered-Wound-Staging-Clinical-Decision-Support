"""
core/visualization.py
======================
WoundLens — Visualization Module
Member 2: CV Engineer — Phase 4 Deliverable

PURPOSE:
    Generate polished, demo-ready visualizations for the Streamlit UI:
        1. Color-coded tissue overlay (red=granulation, yellow=slough, black=necrotic)
        2. Tissue composition pie chart
        3. Wound analysis dashboard (combined image for judges)
        4. Wound contour with severity badge

These visuals are what the judges SEE. They must look clinical yet beautiful.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works in Streamlit/headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple, Optional
from io import BytesIO
from PIL import Image


# =============================================================================
#  COLOR PALETTE (consistent across all visuals)
# =============================================================================

# BGR colors (OpenCV convention)
COLORS_BGR = {
    "granulation": (60,  80,  220),   # Coral red
    "slough":      (0,   210, 230),   # Warm yellow
    "necrotic":    (40,  40,  50),    # Near-black
    "unclassified":(180, 180, 180),   # Light gray
    "wound_border":(0,   255, 130),   # Bright green
}

# RGB colors (Matplotlib convention)
COLORS_RGB = {
    "granulation": "#DC5040",    # Coral red
    "slough":      "#E6D200",    # Warm yellow
    "necrotic":    "#323228",    # Near-black
    "unclassified":"#B4B4B4",   # Light gray
}

# Severity badge colors
SEVERITY_COLORS = {
    "LOW":      ((76, 175, 80),   "LOW RISK"),       # Green
    "MODERATE": ((0, 193, 255),   "MODERATE RISK"),   # Amber
    "HIGH":     ((0, 152, 255),   "HIGH RISK"),       # Orange
    "CRITICAL": ((60, 60, 230),   "CRITICAL"),        # Red
}


# =============================================================================
#  1. TISSUE OVERLAY — Color-coded wound map
# =============================================================================

def create_tissue_overlay(image_bgr: np.ndarray,
                           masks: Dict[str, np.ndarray],
                           wound_mask: Optional[np.ndarray] = None,
                           alpha: float = 0.5) -> np.ndarray:
    """
    Create a smooth, color-coded tissue overlay on the wound image.

    Colors:
        Red/coral  = Granulation (healing)
        Yellow     = Slough (infection risk)
        Dark       = Necrotic (dead tissue)

    Args:
        image_bgr:  Original BGR wound image
        masks:      Tissue masks from tissue_classifier
        wound_mask: Optional wound boundary mask (for contour drawing)
        alpha:      Overlay opacity (0.0–1.0)

    Returns:
        BGR image with tissue overlay
    """
    overlay = image_bgr.copy()

    # Apply tissue colors with Gaussian blur for smooth edges
    for tissue in ["necrotic", "slough", "granulation"]:  # Paint order: dark → light
        if tissue in masks:
            mask = masks[tissue]
            if np.sum(mask > 0) == 0:
                continue

            # Smooth the mask edges for a more polished look
            smooth_mask = cv2.GaussianBlur(mask, (7, 7), 0)
            color = COLORS_BGR[tissue]

            # Blend smoothly using the blurred mask as weight
            mask_3ch = cv2.merge([smooth_mask, smooth_mask, smooth_mask]).astype(np.float32) / 255.0
            color_layer = np.full_like(overlay, color, dtype=np.uint8)
            overlay = (overlay.astype(np.float32) * (1 - mask_3ch * alpha) +
                       color_layer.astype(np.float32) * mask_3ch * alpha).astype(np.uint8)

    # Draw wound boundary contour
    if wound_mask is not None:
        contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLORS_BGR["wound_border"], 2, cv2.LINE_AA)

    return overlay


# =============================================================================
#  2. TISSUE PIE CHART — Composition breakdown
# =============================================================================

def create_pie_chart(ratios: Dict[str, float],
                      size: Tuple[int, int] = (400, 400)) -> np.ndarray:
    """
    Generate a clean tissue composition pie chart.

    Args:
        ratios:  {"granulation": %, "slough": %, "necrotic": %, "unclassified": %}
        size:    Output image size (width, height)

    Returns:
        BGR image of the pie chart
    """
    # Filter out zero-value tissues
    labels_map = {
        "granulation":  "Granulation\n(Healing)",
        "slough":       "Slough\n(Infection Risk)",
        "necrotic":     "Necrotic\n(Dead Tissue)",
        "unclassified": "Unclassified",
    }

    labels = []
    sizes = []
    colors = []
    explode = []

    for tissue in ["granulation", "slough", "necrotic", "unclassified"]:
        pct = ratios.get(tissue, 0)
        if pct > 0.5:  # Only show tissues with >0.5%
            labels.append(labels_map[tissue])
            sizes.append(pct)
            colors.append(COLORS_RGB[tissue])
            # Explode the dominant tissue slightly
            explode.append(0.05 if pct == max(ratios.values()) else 0)

    if not sizes:
        sizes = [100]
        labels = ["No tissue\ndetected"]
        colors = ["#B4B4B4"]
        explode = [0]

    # Create figure with dark background
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
    )

    # Style text
    for text in texts:
        text.set_color("white")
        text.set_fontsize(9)
        text.set_fontweight("bold")
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax.set_title("Tissue Composition", color="white", fontsize=13,
                 fontweight="bold", pad=15)

    plt.tight_layout()

    # Convert matplotlib figure to OpenCV BGR image
    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    chart_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Resize to exact target size
    chart_bgr = cv2.resize(chart_bgr, size, interpolation=cv2.INTER_AREA)

    return chart_bgr


# =============================================================================
#  3. SEVERITY BADGE — Stage indicator
# =============================================================================

def create_severity_badge(stage: str,
                           severity: str,
                           description: str,
                           size: Tuple[int, int] = (400, 120)) -> np.ndarray:
    """
    Create a styled severity badge showing the wound stage.

    Args:
        stage:       e.g., "Stage 3"
        severity:    "LOW" / "MODERATE" / "HIGH" / "CRITICAL"
        description: Stage description text
        size:        Output image size (width, height)

    Returns:
        BGR image of the badge
    """
    w, h = size
    badge = np.full((h, w, 3), (30, 26, 26), dtype=np.uint8)  # Dark background

    color_bgr, label = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["MODERATE"])

    # Color bar on left
    cv2.rectangle(badge, (0, 0), (8, h), color_bgr, -1)

    # Stage text (large)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(badge, stage, (20, 45), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # Severity label
    cv2.putText(badge, label, (20, 75), font, 0.55, color_bgr, 1, cv2.LINE_AA)

    # Description (smaller)
    # Truncate if too long
    max_chars = w // 8
    desc_text = description[:max_chars] + "..." if len(description) > max_chars else description
    cv2.putText(badge, desc_text, (20, 105), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    return badge


# =============================================================================
#  4. TREATMENT CARD — Quick recommendation display
# =============================================================================

def create_treatment_card(treatment: dict,
                           size: Tuple[int, int] = (400, 280)) -> np.ndarray:
    """
    Create a styled treatment recommendation card.

    Args:
        treatment: Output from treatment_engine.get_treatment()
        size:      Output image size (width, height)

    Returns:
        BGR image of the treatment card
    """
    w, h = size
    card = np.full((h, w, 3), (40, 35, 30), dtype=np.uint8)  # Dark background

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30

    # Title
    cv2.putText(card, "TREATMENT PLAN", (15, y), font, 0.6, (100, 220, 255), 1, cv2.LINE_AA)
    y += 10
    cv2.line(card, (15, y), (w - 15, y), (70, 70, 70), 1)
    y += 25

    # Top dressing recommendation
    if treatment["dressings"]:
        top_dressing = treatment["dressings"][0]
        cv2.putText(card, "Primary Dressing:", (15, y), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        y += 22
        name = top_dressing["name"]
        if len(name) > 40:
            name = name[:37] + "..."
        cv2.putText(card, name, (25, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 25

    # Top 3 actions
    cv2.putText(card, "Key Actions:", (15, y), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    y += 22

    for i, action in enumerate(treatment["primary_actions"][:3]):
        max_chars = (w - 40) // 7
        text = action[:max_chars] + "..." if len(action) > max_chars else action
        cv2.putText(card, f"{i+1}. {text}", (25, y), font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        y += 18

    # Follow-up
    y += 10
    cv2.putText(card, f"Follow-up: Every {treatment['follow_up_days']} day(s)",
                (15, y), font, 0.45, (100, 220, 255), 1, cv2.LINE_AA)

    # Referral
    y += 22
    ref = treatment["referral"]
    if len(ref) > 50:
        ref = ref[:47] + "..."
    cv2.putText(card, f"Referral: {ref}", (15, y), font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    return card


# =============================================================================
#  5. FULL ANALYSIS DASHBOARD — Combined view for demo
# =============================================================================

def create_analysis_dashboard(image_bgr: np.ndarray,
                                tissue_result: dict,
                                seg_result: dict,
                                stage_result: dict,
                                treatment: dict) -> np.ndarray:
    """
    Create the complete WoundLens analysis dashboard — a single image
    combining all visuals for maximum demo impact.

    Layout (2x2 grid + header):
    ┌─────────────────────────────────────────┐
    │           WoundLens Analysis             │  (header)
    ├────────────────────┬────────────────────┤
    │                    │   Tissue Overlay    │
    │  Original Image    │   (color-coded)     │
    │                    │                     │
    ├────────────────────┼────────────────────┤
    │   Pie Chart        │  Stage Badge +      │
    │   (composition)    │  Treatment Card     │
    └────────────────────┴────────────────────┘

    Args:
        image_bgr:     Pre-processed wound image
        tissue_result: Output from classify_wound_tissue()
        seg_result:    Output from segment_wound()
        stage_result:  Output from classify_stage()
        treatment:     Output from get_treatment()

    Returns:
        Dashboard BGR image (ready for display or saving)
    """
    # Target cell size
    cell_w, cell_h = 400, 350
    header_h = 60
    total_w = cell_w * 2
    total_h = header_h + cell_h * 2

    dashboard = np.full((total_h, total_w, 3), (26, 26, 30), dtype=np.uint8)  # Dark bg

    # --- HEADER ---
    cv2.putText(dashboard, "WoundLens Analysis Report", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(dashboard, (0, header_h - 2), (total_w, header_h - 2), (70, 70, 70), 2)

    # --- TOP LEFT: Original image with wound contour ---
    from core.segmentation import draw_wound_contour
    img_display = draw_wound_contour(image_bgr, seg_result["wound_mask"],
                                      color=COLORS_BGR["wound_border"], thickness=2)
    img_resized = _resize_to_cell(img_display, cell_w, cell_h)
    _paste(dashboard, img_resized, 0, header_h)

    # Label
    cv2.putText(dashboard, "Original + Wound Boundary", (10, header_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # --- TOP RIGHT: Tissue overlay ---
    overlay = create_tissue_overlay(image_bgr, tissue_result["masks"],
                                     wound_mask=seg_result["wound_mask"], alpha=0.5)
    overlay_resized = _resize_to_cell(overlay, cell_w, cell_h)
    _paste(dashboard, overlay_resized, cell_w, header_h)

    # Label
    cv2.putText(dashboard, "Tissue Classification Map", (cell_w + 10, header_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # --- BOTTOM LEFT: Pie chart ---
    pie = create_pie_chart(tissue_result["ratios"], size=(cell_w, cell_h))
    _paste(dashboard, pie, 0, header_h + cell_h)

    # --- BOTTOM RIGHT: Stage badge + Treatment card ---
    badge = create_severity_badge(
        stage_result["stage"],
        stage_result["severity"],
        stage_result["description"],
        size=(cell_w, 120)
    )
    _paste(dashboard, badge, cell_w, header_h + cell_h)

    treatment_card = create_treatment_card(treatment, size=(cell_w, cell_h - 120))
    _paste(dashboard, treatment_card, cell_w, header_h + cell_h + 120)

    return dashboard


# =============================================================================
#  HELPERS
# =============================================================================

def _resize_to_cell(image: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    """Resize image to fit within a cell, preserving aspect ratio, with letterboxing."""
    h, w = image.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center within cell (letterbox)
    canvas = np.full((cell_h, cell_w, 3), (26, 26, 30), dtype=np.uint8)
    x_off = (cell_w - new_w) // 2
    y_off = (cell_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Paste a patch onto a canvas at position (x, y). Modifies canvas in-place."""
    ph, pw = patch.shape[:2]
    ch, cw = canvas.shape[:2]
    # Clip to canvas bounds
    pw = min(pw, cw - x)
    ph = min(ph, ch - y)
    if pw > 0 and ph > 0:
        canvas[y:y + ph, x:x + pw] = patch[:ph, :pw]


def get_tissue_color_legend() -> list:
    """
    Return tissue legend info for Streamlit UI display.

    Returns:
        List of dicts with name, color_hex, description
    """
    return [
        {
            "name": "Granulation",
            "color_hex": COLORS_RGB["granulation"],
            "description": "Healthy healing tissue (red/pink)",
            "clinical": "Good sign - active blood vessel formation",
        },
        {
            "name": "Slough",
            "color_hex": COLORS_RGB["slough"],
            "description": "Devitalized tissue (yellow/cream)",
            "clinical": "Infection risk - may need debridement",
        },
        {
            "name": "Necrotic",
            "color_hex": COLORS_RGB["necrotic"],
            "description": "Dead tissue / eschar (black/brown)",
            "clinical": "Urgent - blocks healing, requires intervention",
        },
    ]


# =============================================================================
#  CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    import io
    import argparse

    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(description="WoundLens Visualization — Phase 4 Test")
    parser.add_argument("--image", type=str, required=True, help="Path to wound image")
    parser.add_argument("--out", type=str, default=None, help="Output path (default: <image>_dashboard.jpg)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not load: {args.image}")
        sys.exit(1)

    from segmentation import segment_wound
    from tissue_classifier import classify_wound_tissue
    from staging_engine import classify_stage
    from treatment_engine import get_treatment

    print("Running full pipeline...")

    seg = segment_wound(img)
    tissue = classify_wound_tissue(seg["preprocessed"], wound_mask=seg["wound_mask"])
    ratios = tissue["ratios"]
    stage = classify_stage(ratios["granulation"], ratios["slough"], ratios["necrotic"])
    treatment = get_treatment(stage["stage"])

    print(f"  Stage: {stage['stage']} ({stage['severity']})")
    print(f"  Gran={ratios['granulation']:.1f}% Slough={ratios['slough']:.1f}% Necrotic={ratios['necrotic']:.1f}%")

    dashboard = create_analysis_dashboard(
        seg["preprocessed"], tissue, seg, stage, treatment
    )

    out_path = args.out or args.image.rsplit(".", 1)[0] + "_dashboard.jpg"
    cv2.imwrite(out_path, dashboard)
    print(f"  Dashboard saved to: {out_path}")
