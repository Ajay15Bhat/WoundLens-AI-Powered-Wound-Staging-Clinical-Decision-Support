"""
core/segmentation.py
=====================
WoundLens — Wound Bed Segmentation Module
Member 2: CV Engineer — Phase 2 Deliverable

PURPOSE:
    Given a raw wound image (or a YOLO-detected bounding box crop),
    segment the wound bed from surrounding healthy skin and produce a
    binary wound_mask that gates all downstream tissue classification.

PIPELINE:
    Raw image
        → Pre-process  (resize, denoise, CLAHE contrast enhance)
        → Wound mask   (GrabCut seeded from center or from YOLO bbox)
        → Morphological cleanup  (close holes, remove small noise islands)
        → Return mask + cropped wound ROI

PHASE 2 STATUS: ✅ Full segmentation pipeline
PHASE 3 TODO:   Wire bbox from predict.py directly into segment_from_bbox()
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict

# =============================================================================
#  CONFIG
# =============================================================================

# GrabCut iterations — higher = more accurate but slower
GRABCUT_ITERATIONS = 5

# Minimum wound area as fraction of image — smaller blobs are noise
MIN_WOUND_AREA_FRACTION = 0.01

# CLAHE parameters for contrast enhancement
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)

# Morphological kernel sizes
MORPH_CLOSE_KSIZE = 15   # Close small holes inside wound mask
MORPH_OPEN_KSIZE  = 7    # Remove small noise islands outside wound


# =============================================================================
#  PRE-PROCESSING
# =============================================================================

def preprocess(image_bgr: np.ndarray,
               target_long_side: int = 640) -> np.ndarray:
    """
    Resize (preserving aspect ratio) + denoise + CLAHE contrast enhancement.

    Args:
        image_bgr:       Raw BGR image from camera / upload
        target_long_side: Resize so the longer dimension = this value

    Returns:
        Pre-processed BGR image (uint8)
    """
    h, w = image_bgr.shape[:2]
    scale = target_long_side / max(h, w)
    if scale < 1.0:                               # Only downscale, never upscale
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Mild denoising to smooth HSV noise without blurring wound edges
    denoised = cv2.fastNlMeansDenoisingColored(image_bgr, None, 6, 6, 7, 21)

    # CLAHE on L channel of LAB — improves contrast for dark necrotic tissue
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced


# =============================================================================
#  WOUND MASK GENERATION
# =============================================================================

def _grabcut_from_rect(image_bgr: np.ndarray,
                        rect: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Run GrabCut using a rectangle seed (used when YOLO bbox is available).

    Args:
        image_bgr: Pre-processed BGR image
        rect:      (x, y, w, h) rectangle — YOLO bbox or auto-center rect

    Returns:
        Binary mask (uint8): 255 = foreground (wound), 0 = background
    """
    mask   = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model,
                GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)

    # GrabCut labels: 0=sure_bg, 1=sure_fg, 2=probable_bg, 3=probable_fg
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                      255, 0).astype(np.uint8)
    return binary


def _center_rect(h: int, w: int, margin_frac: float = 0.15) -> Tuple[int, int, int, int]:
    """
    Auto-generate a center seed rectangle when no YOLO bbox is available.
    Assumes the wound is roughly centered (common in clinical wound photography).

    Args:
        h, w:        Image height and width
        margin_frac: Fraction of dimensions to exclude from edges

    Returns:
        (x, y, rect_w, rect_h) GrabCut-style rectangle
    """
    mx = int(w * margin_frac)
    my = int(h * margin_frac)
    return (mx, my, w - 2 * mx, h - 2 * my)


def _color_distance_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Fallback segmentation using color distance from the image center pixel.
    Used when GrabCut produces a degenerate mask (all 0 or all 255).

    Strategy: assume center pixels are wound tissue, use floodFill-style
    color clustering to find similar-colored region.

    Returns:
        Binary mask (uint8): 255 = wound-like region
    """
    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Sample a 20x20 patch from center as the "wound color"
    cy, cx = h // 2, w // 2
    patch = lab[cy-10:cy+10, cx-10:cx+10]
    wound_color = patch.mean(axis=(0, 1))          # (L, A, B) mean

    # Euclidean distance in LAB space
    diff = lab - wound_color                        # (H, W, 3)
    dist = np.sqrt((diff ** 2).sum(axis=2))         # (H, W)

    # Threshold — pixels within distance 35 LAB units are "wound-like"
    mask = (dist < 35).astype(np.uint8) * 255
    return mask


def generate_wound_mask(image_bgr: np.ndarray,
                         bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
                         ) -> np.ndarray:
    """
    Generate a binary wound mask using GrabCut.

    Args:
        image_bgr:  Pre-processed BGR image
        bbox_xyxy:  Optional YOLO bounding box in (x1, y1, x2, y2) format.
                    If None, uses center-rectangle heuristic.

    Returns:
        wound_mask: Binary uint8 ndarray (255 = wound, 0 = background)
                    Same spatial dimensions as image_bgr.
    """
    h, w = image_bgr.shape[:2]

    # Convert YOLO xyxy → GrabCut xywh, with a small inset margin
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        margin = 10
        x1 = max(0, x1 + margin)
        y1 = max(0, y1 + margin)
        x2 = min(w - 1, x2 - margin)
        y2 = min(h - 1, y2 - margin)
        rect = (x1, y1, x2 - x1, y2 - y1)
    else:
        rect = _center_rect(h, w)

    # Sanity check — rect must have positive area
    if rect[2] <= 0 or rect[3] <= 0:
        rect = _center_rect(h, w)

    mask = _grabcut_from_rect(image_bgr, rect)

    # Fallback: if GrabCut gives a degenerate result, use color distance
    wound_pixels = int(np.sum(mask > 0))
    total_pixels = h * w
    if wound_pixels < total_pixels * MIN_WOUND_AREA_FRACTION or wound_pixels > total_pixels * 0.95:
        mask = _color_distance_mask(image_bgr)

    return mask


# =============================================================================
#  MORPHOLOGICAL CLEANUP
# =============================================================================

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean up the raw wound mask:
        1. CLOSE  — fill small holes and gaps inside the wound region
        2. OPEN   — remove small noise islands outside the wound
        3. Keep only the largest connected component (the actual wound)

    Args:
        mask: Raw binary mask (uint8, 0 or 255)

    Returns:
        Cleaned binary mask (uint8, 0 or 255)
    """
    # Step 1: Close holes
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (MORPH_CLOSE_KSIZE, MORPH_CLOSE_KSIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # Step 2: Remove noise
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_OPEN_KSIZE, MORPH_OPEN_KSIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # Step 3: Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask                                # No foreground at all

    # stats[0] is background — find largest foreground component
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1     # +1 because we skipped background
    clean = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return clean


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

def segment_wound(image_bgr: np.ndarray,
                   bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
                   ) -> Dict:
    """
    Full segmentation pipeline: raw image → wound mask + ROI crop.

    Args:
        image_bgr:  Raw BGR image (from cv2.imread or camera frame)
        bbox_xyxy:  Optional YOLO bounding box (x1, y1, x2, y2).
                    Pass this in Phase 3 when predict.py is integrated.

    Returns:
        {
            "preprocessed":  np.ndarray — pre-processed BGR image
            "wound_mask":    np.ndarray — binary mask (255=wound, 0=bg)
            "wound_crop":    np.ndarray — BGR image cropped to wound bbox
            "mask_crop":     np.ndarray — wound_mask cropped to same bbox
            "wound_bbox":    (x, y, w, h) — tight bounding box around wound mask
            "wound_area_px": int — number of wound pixels
            "coverage_pct":  float — wound as % of image
        }
    """
    # 1. Pre-process
    processed = preprocess(image_bgr)

    # 2. Generate raw wound mask
    raw_mask = generate_wound_mask(processed, bbox_xyxy)

    # 3. Morphological cleanup
    wound_mask = clean_mask(raw_mask)

    # 4. Compute tight bounding box around the wound mask
    contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    else:
        h, w = processed.shape[:2]
        x, y, bw, bh = 0, 0, w, h

    wound_crop = processed[y:y+bh, x:x+bw]
    mask_crop  = wound_mask[y:y+bh, x:x+bw]

    wound_area_px = int(np.sum(wound_mask > 0))
    h, w = processed.shape[:2]
    coverage_pct = round((wound_area_px / (h * w)) * 100, 2)

    return {
        "preprocessed":  processed,
        "wound_mask":    wound_mask,
        "wound_crop":    wound_crop,
        "mask_crop":     mask_crop,
        "wound_bbox":    (x, y, bw, bh),
        "wound_area_px": wound_area_px,
        "coverage_pct":  coverage_pct,
    }


def segment_from_bbox(image_bgr: np.ndarray,
                       bbox_xyxy: Tuple[int, int, int, int]) -> Dict:
    """
    Convenience wrapper for Phase 3: called directly with YOLO bbox.

    Args:
        image_bgr:  Raw BGR image
        bbox_xyxy:  YOLO prediction in (x1, y1, x2, y2) pixel coords

    Returns:
        Same dict as segment_wound()
    """
    return segment_wound(image_bgr, bbox_xyxy=bbox_xyxy)


# =============================================================================
#  VISUALIZATION HELPERS
# =============================================================================

def draw_wound_contour(image_bgr: np.ndarray,
                        wound_mask: np.ndarray,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    """
    Draw the wound boundary contour on the image.

    Args:
        image_bgr:  BGR image to draw on
        wound_mask: Binary wound mask
        color:      BGR color of the contour line (default: green)
        thickness:  Line thickness in pixels

    Returns:
        Image with contour drawn (BGR, uint8)
    """
    vis = image_bgr.copy()
    contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


def blend_mask_overlay(image_bgr: np.ndarray,
                        wound_mask: np.ndarray,
                        color: Tuple[int, int, int] = (0, 200, 100),
                        alpha: float = 0.3) -> np.ndarray:
    """
    Blend a semi-transparent colored overlay of the wound mask onto the image.

    Args:
        image_bgr:  Original BGR image
        wound_mask: Binary wound mask (255 = wound)
        color:      BGR highlight color
        alpha:      Overlay transparency (0 = invisible, 1 = opaque)

    Returns:
        Blended BGR image (uint8)
    """
    overlay = image_bgr.copy()
    overlay[wound_mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)


# =============================================================================
#  CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="WoundLens Segmentation — Phase 2 Test")
    parser.add_argument("--image", type=str, required=True, help="Path to wound image")
    parser.add_argument("--bbox", type=str, default=None,
                        help="Optional YOLO bbox as x1,y1,x2,y2 (e.g. 50,80,300,400)")
    parser.add_argument("--out", type=str, default="segmentation_result.jpg",
                        help="Output visualization path")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ Could not load: {args.image}")
        sys.exit(1)

    bbox = None
    if args.bbox:
        bbox = tuple(int(x) for x in args.bbox.split(","))

    result = segment_wound(img, bbox_xyxy=bbox)

    print("\n🔬 WoundLens Segmentation Result")
    print("─" * 40)
    print(f"  Wound area : {result['wound_area_px']:,} px")
    print(f"  Coverage   : {result['coverage_pct']:.1f}% of image")
    print(f"  Wound bbox : {result['wound_bbox']}")
    print("─" * 40)

    # Save a side-by-side: original | mask overlay | contour
    vis_overlay  = blend_mask_overlay(result["preprocessed"], result["wound_mask"])
    vis_contour  = draw_wound_contour(result["preprocessed"], result["wound_mask"])
    side_by_side = np.hstack([result["preprocessed"], vis_overlay, vis_contour])
    cv2.imwrite(args.out, side_by_side)
    print(f"\n  ✅ Result saved to: {args.out}")
    print("     [Original | Mask Overlay | Contour]")