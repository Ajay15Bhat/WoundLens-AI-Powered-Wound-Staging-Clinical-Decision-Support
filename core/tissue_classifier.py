"""
core/tissue_classifier.py
==========================
WoundLens — Tissue Classification Module
Member 2: CV Engineer

PURPOSE:
    Classify wound tissue into three clinical categories based on color-space analysis:
        - Granulation Tissue  (red/pink)  → Healing
        - Slough Tissue       (yellow)    → Infection risk
        - Necrotic Tissue     (black/dark brown) → Dead tissue, urgent

RESEARCH BASIS:
    HSV thresholds derived from:
    - Wannous et al. (2010): "Supervised tissue classification from color images
      for a remote wound monitoring system" — Journal of Medical Engineering
    - Chakraborty et al. (2021): "Wound tissue segmentation using HSV color model"
    - DFUC2022 Challenge dataset color analysis
    - Clinical wound photography standards (standardized lighting assumed)

COORDINATE SYSTEM (OpenCV HSV):
    H: 0–179  (OpenCV halves the standard 0–360 Hue range)
    S: 0–255
    V: 0–255

PHASE 1 STATUS: ✅ HSV thresholds researched & documented
PHASE 2 STATUS: ✅ Full segmentation pipeline with HSV masking, morphological
                   cleanup, LAB confirmation & tissue ratio calculator
PHASE 3 TODO:   Apply tissue analysis only within detected wound region (crop to bbox)
PHASE 4 TODO:   Generate beautiful color-coded tissue overlay visualization
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

# =============================================================================
# 🎨  HSV COLOR THRESHOLDS  (OpenCV scale: H 0-179, S 0-255, V 0-255)
# =============================================================================

# ── GRANULATION TISSUE — Red / Pink / Salmon ──────────────────────────────────
#   Appearance:  Beefy red, moist, granular (cobblestone texture)
#   Clinical:    GOOD sign — active healing, new blood vessel formation
#   Challenge:   Red wraps around H=0/179; needs two HSV ranges

GRANULATION_LOWER_1    = np.array([0,   60,  40],  dtype=np.uint8)   # lower red band
GRANULATION_UPPER_1    = np.array([10,  255, 255],  dtype=np.uint8)
GRANULATION_LOWER_2    = np.array([160, 60,  40],  dtype=np.uint8)   # upper red band
GRANULATION_UPPER_2    = np.array([179, 255, 255],  dtype=np.uint8)
GRANULATION_PINK_LOWER = np.array([0,   30,  120], dtype=np.uint8)   # pink/salmon extension
GRANULATION_PINK_UPPER = np.array([15,  100, 255], dtype=np.uint8)

# ── SLOUGH TISSUE — Yellow / Cream / Tan ─────────────────────────────────────
#   Appearance:  Stringy, mucinous, moist yellow-white coating
#   Clinical:    BAD sign — devitalized tissue, infection risk

SLOUGH_LOWER       = np.array([18,  40,  80],  dtype=np.uint8)       # vivid yellow
SLOUGH_UPPER       = np.array([40,  255, 255], dtype=np.uint8)
SLOUGH_CREAM_LOWER = np.array([18,  20,  160], dtype=np.uint8)       # cream/off-white variant
SLOUGH_CREAM_UPPER = np.array([35,  80,  255], dtype=np.uint8)

# ── NECROTIC TISSUE — Black / Dark Brown / Eschar ────────────────────────────
#   Appearance:  Dry leathery eschar or wet gangrenous tissue
#   Clinical:    URGENT — dead tissue, blocks healing

NECROTIC_LOWER       = np.array([0,  0,   0],  dtype=np.uint8)       # dark/black
NECROTIC_UPPER       = np.array([30, 80,  60], dtype=np.uint8)
NECROTIC_BROWN_LOWER = np.array([5,  40,  20], dtype=np.uint8)       # brown eschar variant
NECROTIC_BROWN_UPPER = np.array([25, 180, 80], dtype=np.uint8)

# ── HEALTHY SKIN — Excluded from wound analysis (reference only) ──────────────
HEALTHY_SKIN_LOWER = np.array([0,  10,  120], dtype=np.uint8)
HEALTHY_SKIN_UPPER = np.array([20, 80,  255], dtype=np.uint8)

# =============================================================================
# 📐  LAB COLOR SPACE — SUPPLEMENTARY THRESHOLDS
# =============================================================================
# OpenCV LAB: L 0–255, A 0–255 (128=neutral), B 0–255 (128=neutral)

LAB_THRESHOLDS = {
    "granulation": {"L_min": 50,  "L_max": 220, "A_min": 135, "A_max": 255, "B_min": 100, "B_max": 200},
    "slough":      {"L_min": 100, "L_max": 255, "A_min": 110, "A_max": 145, "B_min": 140, "B_max": 255},
    "necrotic":    {"L_min": 0,   "L_max": 80,  "A_min": 0,   "A_max": 200, "B_min": 0,   "B_max": 200},
}

# =============================================================================
# 📦  VISUALIZATION COLORS  (BGR, OpenCV convention)
# =============================================================================

TISSUE_OVERLAY_COLORS = {
    "granulation": (0,   80,  220),   # Red/coral   → healing
    "slough":      (0,   220, 220),   # Yellow      → infection risk
    "necrotic":    (30,  30,  30),    # Near-black  → dead tissue
}
TISSUE_OVERLAY_ALPHA = 0.45

# =============================================================================
# ⚙️  MORPHOLOGICAL CLEANUP CONFIG
# =============================================================================

# Kernel sizes for mask cleanup
MORPH_CLOSE_KSIZE = 9    # Fill small holes within tissue regions
MORPH_OPEN_KSIZE  = 5    # Remove speckle noise
MIN_BLOB_AREA     = 150  # Minimum pixel area to retain a tissue blob


# =============================================================================
# 🔧  STEP 1 — RAW HSV MASKS
# =============================================================================

def create_tissue_masks(hsv_image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create raw binary masks for each tissue type from an HSV image.

    Args:
        hsv_image: Image in HSV color space (OpenCV H 0-179, S 0-255, V 0-255)

    Returns:
        {
            "granulation": np.ndarray (uint8, 0 or 255),
            "slough":      np.ndarray,
            "necrotic":    np.ndarray,
        }
    """
    # Granulation: lower red + upper red + pink
    m1 = cv2.inRange(hsv_image, GRANULATION_LOWER_1, GRANULATION_UPPER_1)
    m2 = cv2.inRange(hsv_image, GRANULATION_LOWER_2, GRANULATION_UPPER_2)
    mp = cv2.inRange(hsv_image, GRANULATION_PINK_LOWER, GRANULATION_PINK_UPPER)
    mask_gran = cv2.bitwise_or(cv2.bitwise_or(m1, m2), mp)

    # Slough: vivid yellow + cream
    ms = cv2.inRange(hsv_image, SLOUGH_LOWER, SLOUGH_UPPER)
    mc = cv2.inRange(hsv_image, SLOUGH_CREAM_LOWER, SLOUGH_CREAM_UPPER)
    mask_slough = cv2.bitwise_or(ms, mc)

    # Necrotic: dark + brown eschar
    mn  = cv2.inRange(hsv_image, NECROTIC_LOWER, NECROTIC_UPPER)
    mnb = cv2.inRange(hsv_image, NECROTIC_BROWN_LOWER, NECROTIC_BROWN_UPPER)
    mask_necrotic = cv2.bitwise_or(mn, mnb)

    return {
        "granulation": mask_gran,
        "slough":      mask_slough,
        "necrotic":    mask_necrotic,
    }


# =============================================================================
# 🔧  STEP 2 — LAB CONFIRMATION (secondary filter)
# =============================================================================

def create_lab_confirmation_masks(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate tissue masks using LAB color space thresholds.
    Used as a secondary confirmation layer to reduce HSV false positives.

    Args:
        image_bgr: BGR image

    Returns:
        {"granulation": mask, "slough": mask, "necrotic": mask}
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    result = {}
    for tissue, t in LAB_THRESHOLDS.items():
        mask = (
            (L >= t["L_min"]) & (L <= t["L_max"]) &
            (A >= t["A_min"]) & (A <= t["A_max"]) &
            (B >= t["B_min"]) & (B <= t["B_max"])
        ).astype(np.uint8) * 255
        result[tissue] = mask
    return result


# =============================================================================
# 🔧  STEP 3 — MORPHOLOGICAL CLEANUP
# =============================================================================

def clean_tissue_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean a raw tissue binary mask:
        1. Close  — fill small intra-region holes
        2. Open   — remove speckle noise
        3. Remove blobs smaller than MIN_BLOB_AREA pixels

    Args:
        mask: Raw binary uint8 mask (0 or 255)

    Returns:
        Cleaned binary mask (uint8, 0 or 255)
    """
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KSIZE, MORPH_CLOSE_KSIZE))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_KSIZE,  MORPH_OPEN_KSIZE))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

    # Remove small blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for label in range(1, num_labels):                   # 0 = background
        if stats[label, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA:
            clean[labels == label] = 255

    return clean


# =============================================================================
# 🔧  STEP 4 — COMBINE HSV + LAB
# =============================================================================

def combine_masks(hsv_masks: Dict[str, np.ndarray],
                  lab_masks: Dict[str, np.ndarray],
                  use_lab_confirmation: bool = True) -> Dict[str, np.ndarray]:
    """
    Merge HSV and LAB masks per tissue type.

    Strategy:
        - If use_lab_confirmation=True:  use AND (HSV ∩ LAB) — high precision
        - If use_lab_confirmation=False: use HSV only — higher recall (fallback)

    The LAB confirmation is especially important for differentiating:
        - Granulation vs. healthy skin (both red-ish in HSV)
        - Slough vs. fat/adipose tissue (both yellowish)

    Args:
        hsv_masks:            Masks from create_tissue_masks()
        lab_masks:            Masks from create_lab_confirmation_masks()
        use_lab_confirmation: If True, intersect HSV and LAB masks

    Returns:
        Combined masks dict {"granulation": ..., "slough": ..., "necrotic": ...}
    """
    combined = {}
    for tissue in ["granulation", "slough", "necrotic"]:
        hsv = hsv_masks[tissue]
        lab = lab_masks[tissue]
        if use_lab_confirmation:
            combined[tissue] = cv2.bitwise_and(hsv, lab)
        else:
            combined[tissue] = hsv
    return combined


# =============================================================================
# 🔧  STEP 5 — RESOLVE PIXEL CONFLICTS
# =============================================================================

def resolve_conflicts(masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Resolve pixels that are assigned to multiple tissue types (overlap).

    Priority order (clinically motivated):
        1. Necrotic   — most distinctive/critical; low-V pixels are unambiguous
        2. Slough     — yellow is fairly distinctive
        3. Granulation — red overlaps most with healthy skin; lowest priority

    This ensures a pixel is assigned to at most one tissue type.

    Args:
        masks: Dict of (possibly overlapping) binary tissue masks

    Returns:
        Dict of non-overlapping masks
    """
    necrotic    = masks["necrotic"].copy()
    slough      = masks["slough"].copy()
    granulation = masks["granulation"].copy()

    # Remove necrotic pixels from slough and granulation
    slough      = cv2.bitwise_and(slough,      cv2.bitwise_not(necrotic))
    granulation = cv2.bitwise_and(granulation, cv2.bitwise_not(necrotic))

    # Remove slough pixels from granulation
    granulation = cv2.bitwise_and(granulation, cv2.bitwise_not(slough))

    return {
        "granulation": granulation,
        "slough":      slough,
        "necrotic":    necrotic,
    }


# =============================================================================
# 🔧  STEP 6 — TISSUE RATIO CALCULATOR
# =============================================================================

def calculate_tissue_ratios(masks: Dict[str, np.ndarray],
                              wound_mask: Optional[np.ndarray] = None
                              ) -> Dict[str, float]:
    """
    Calculate the percentage of each tissue type within the wound area.

    Args:
        masks:       Dict of non-overlapping binary masks (from resolve_conflicts)
        wound_mask:  Optional binary mask restricting analysis to wound ROI.
                     Pass this from segmentation.py in Phase 3.
                     If None, uses the full image (Phase 2 fallback).

    Returns:
        {
            "granulation":  float (0.0–100.0),
            "slough":       float,
            "necrotic":     float,
            "unclassified": float,   # wound pixels not matching any type
        }
    """
    if wound_mask is not None:
        # Gate all masks to within the wound boundary
        gated = {k: cv2.bitwise_and(v, v, mask=wound_mask) for k, v in masks.items()}
        total_pixels = int(np.sum(wound_mask > 0))
    else:
        gated = masks
        h, w  = next(iter(masks.values())).shape[:2]
        total_pixels = h * w

    if total_pixels == 0:
        return {"granulation": 0.0, "slough": 0.0, "necrotic": 0.0, "unclassified": 100.0}

    counts = {k: int(np.sum(v > 0)) for k, v in gated.items()}
    classified   = sum(counts.values())
    unclassified = max(0, total_pixels - classified)

    ratios = {k: round((n / total_pixels) * 100, 2) for k, n in counts.items()}
    ratios["unclassified"] = round((unclassified / total_pixels) * 100, 2)
    return ratios


# =============================================================================
# 🚀  MAIN ENTRY POINT
# =============================================================================

def classify_wound_tissue(image_bgr: np.ndarray,
                            wound_mask: Optional[np.ndarray] = None,
                            use_lab_confirmation: bool = True
                            ) -> Dict:
    """
    Full Phase 2 tissue classification pipeline.

    Steps:
        1. Convert to HSV → create raw tissue masks
        2. Convert to LAB → create confirmation masks
        3. Combine HSV + LAB (intersection for precision)
        4. Morphological cleanup (close holes, remove noise)
        5. Resolve pixel conflicts (priority: necrotic > slough > granulation)
        6. Calculate tissue ratios (within wound_mask if provided)

    Args:
        image_bgr:            BGR image (uint8)
        wound_mask:           Optional binary wound boundary mask from segmentation.py.
                              Restricts analysis to wound ROI. Pass in Phase 3.
        use_lab_confirmation: If True (default), intersect HSV and LAB masks for
                              higher precision. Set False for recall-first mode.

    Returns:
        {
            "ratios": {
                "granulation":  float,
                "slough":       float,
                "necrotic":     float,
                "unclassified": float,
            },
            "masks": {
                "granulation": np.ndarray (binary),
                "slough":      np.ndarray,
                "necrotic":    np.ndarray,
            },
            "dominant_tissue": str,   # tissue with highest ratio
            "wound_severity":  str,   # "healing" | "moderate" | "severe"
        }
    """
    # Step 1: HSV masks
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv_masks = create_tissue_masks(hsv)

    # Step 2: LAB confirmation masks
    lab_masks = create_lab_confirmation_masks(image_bgr)

    # Step 3: Combine
    combined = combine_masks(hsv_masks, lab_masks, use_lab_confirmation)

    # Step 4: Morphological cleanup per tissue type
    cleaned = {k: clean_tissue_mask(v) for k, v in combined.items()}

    # Step 5: Resolve conflicts
    final_masks = resolve_conflicts(cleaned)

    # Step 6: Calculate ratios
    ratios = calculate_tissue_ratios(final_masks, wound_mask)

    # Derive dominant tissue and severity label
    tissue_types   = ["granulation", "slough", "necrotic"]
    dominant       = max(tissue_types, key=lambda t: ratios.get(t, 0))
    wound_severity = _severity_label(ratios)

    return {
        "ratios":          ratios,
        "masks":           final_masks,
        "dominant_tissue": dominant,
        "wound_severity":  wound_severity,
    }


# Backwards-compatible alias used by Phase 1 code
def classify_image(image_bgr: np.ndarray,
                   wound_mask: Optional[np.ndarray] = None) -> Dict:
    """Alias for classify_wound_tissue() — keeps Phase 1 callers working."""
    return classify_wound_tissue(image_bgr, wound_mask)


# =============================================================================
# 🩺  SEVERITY HELPER
# =============================================================================

def _severity_label(ratios: Dict[str, float]) -> str:
    """
    Derive a simple severity label from tissue ratios.
    (This feeds into Member 4's staging engine as a soft signal.)

    Rules:
        "healing"  — granulation dominant (≥ 50%)
        "severe"   — necrotic ≥ 30%  OR  (necrotic + slough) ≥ 60%
        "moderate" — everything else
    """
    gran     = ratios.get("granulation", 0)
    slough   = ratios.get("slough",      0)
    necrotic = ratios.get("necrotic",    0)

    if gran >= 50:
        return "healing"
    if necrotic >= 30 or (necrotic + slough) >= 60:
        return "severe"
    return "moderate"


# =============================================================================
# 🎨  VISUALIZATION
# =============================================================================

def apply_tissue_overlay(image_bgr: np.ndarray,
                          masks: Dict[str, np.ndarray],
                          alpha: float = TISSUE_OVERLAY_ALPHA) -> np.ndarray:
    """
    Overlay color-coded tissue masks on the original image.

    Color coding (BGR):
        Granulation → Red/coral  (0, 80, 220)
        Slough      → Yellow     (0, 220, 220)
        Necrotic    → Dark       (30, 30, 30)

    Args:
        image_bgr: Original wound image (BGR)
        masks:     Final tissue masks from classify_wound_tissue()
        alpha:     Overlay opacity (0 = invisible, 1 = opaque). Default 0.45.

    Returns:
        Image with color overlay (BGR uint8)
    """
    overlay = image_bgr.copy()
    for tissue, color in TISSUE_OVERLAY_COLORS.items():
        if tissue in masks:
            overlay[masks[tissue] > 0] = color
    return cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)


def draw_tissue_legend(image_bgr: np.ndarray,
                        ratios: Dict[str, float]) -> np.ndarray:
    """
    Draw a text legend showing tissue percentages on the image.

    Args:
        image_bgr: BGR image to annotate
        ratios:    Output of calculate_tissue_ratios()

    Returns:
        Annotated image (BGR uint8)
    """
    vis = image_bgr.copy()
    labels = [
        ("Granulation", ratios.get("granulation", 0),  (0, 80, 220)),
        ("Slough",      ratios.get("slough", 0),        (0, 210, 210)),
        ("Necrotic",    ratios.get("necrotic", 0),      (80, 80, 80)),
    ]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    y_start    = 25
    y_step     = 22

    for i, (name, pct, color) in enumerate(labels):
        text = f"{name}: {pct:.1f}%"
        y    = y_start + i * y_step
        # Shadow
        cv2.putText(vis, text, (11, y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # Colored label
        cv2.putText(vis, text, (10, y),     font, font_scale, color,     thickness,     cv2.LINE_AA)

    return vis


# =============================================================================
# 🛠️  DEBUG HELPERS
# =============================================================================

def print_threshold_summary():
    """Print a human-readable summary of all HSV thresholds (for team reference)."""
    print("=" * 60)
    print("  WoundLens HSV Threshold Summary (OpenCV scale)")
    print("=" * 60)
    thresholds = {
        "Granulation (red band 1)":  (GRANULATION_LOWER_1,    GRANULATION_UPPER_1),
        "Granulation (red band 2)":  (GRANULATION_LOWER_2,    GRANULATION_UPPER_2),
        "Granulation (pink/salmon)": (GRANULATION_PINK_LOWER,  GRANULATION_PINK_UPPER),
        "Slough (yellow)":           (SLOUGH_LOWER,            SLOUGH_UPPER),
        "Slough (cream)":            (SLOUGH_CREAM_LOWER,      SLOUGH_CREAM_UPPER),
        "Necrotic (dark)":           (NECROTIC_LOWER,          NECROTIC_UPPER),
        "Necrotic (brown eschar)":   (NECROTIC_BROWN_LOWER,    NECROTIC_BROWN_UPPER),
    }
    for name, (lo, hi) in thresholds.items():
        print(f"\n  {name}")
        print(f"    Lower: H={lo[0]:3d}  S={lo[1]:3d}  V={lo[2]:3d}")
        print(f"    Upper: H={hi[0]:3d}  S={hi[1]:3d}  V={hi[2]:3d}")
    print("=" * 60)


# =============================================================================
# 🚀  CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="WoundLens Tissue Classifier — Phase 2")
    parser.add_argument("--image",      type=str,  default=None, help="Path to wound image")
    parser.add_argument("--thresholds", action="store_true",     help="Print HSV thresholds")
    parser.add_argument("--no-lab",     action="store_true",     help="Disable LAB confirmation")
    args = parser.parse_args()

    if args.thresholds:
        print_threshold_summary()
        sys.exit(0)

    if not args.image:
        print("ℹ️  Use --image <path> to classify a wound image.")
        print("   Use --thresholds to see all HSV ranges.")
        sys.exit(0)

    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ Could not load: {args.image}")
        sys.exit(1)

    result = classify_wound_tissue(img, use_lab_confirmation=not args.no_lab)
    r      = result["ratios"]

    print("\n🔬 WoundLens Tissue Analysis (Phase 2)")
    print("─" * 40)
    print(f"  🔴 Granulation : {r['granulation']:6.2f}%")
    print(f"  🟡 Slough      : {r['slough']:6.2f}%")
    print(f"  ⚫ Necrotic    : {r['necrotic']:6.2f}%")
    print(f"  ⬜ Unclassified: {r['unclassified']:6.2f}%")
    print(f"\n  Dominant tissue : {result['dominant_tissue'].upper()}")
    print(f"  Wound severity  : {result['wound_severity'].upper()}")
    print("─" * 40)

    # Save annotated output
    overlay     = apply_tissue_overlay(img, result["masks"])
    annotated   = draw_tissue_legend(overlay, r)
    out_path    = args.image.rsplit(".", 1)[0] + "_tissue_analysis.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\n  ✅ Annotated overlay saved to: {out_path}")