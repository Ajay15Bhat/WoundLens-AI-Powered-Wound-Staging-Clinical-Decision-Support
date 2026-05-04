"""
test_pipeline.py
================
Member 2 — End-to-End Pipeline Test Script

Tests the full flow WITHOUT needing Member 1's YOLO model:
    Sample image → Segmentation → Tissue Classification → Staging → Treatment

USAGE:
    python test_pipeline.py --image data/sample_wounds/test_wound.jpg
    python test_pipeline.py --image data/sample_wounds/test_wound.jpg --no-lab
    python test_pipeline.py --all   (runs on ALL images in sample_wounds/)

OUTPUT:
    Saves annotated results to data/sample_wounds/results/
"""

import sys
import os
import io
import argparse
import glob

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.segmentation import segment_wound, draw_wound_contour, blend_mask_overlay
from core.tissue_classifier import classify_wound_tissue, apply_tissue_overlay, draw_tissue_legend
from core.staging_engine import classify_stage, get_severity_emoji
from core.treatment_engine import get_treatment, format_treatment_summary
from core.visualization import create_analysis_dashboard


def run_pipeline(image_path: str, use_lab: bool = True, save_dir: str = None) -> dict:
    """
    Run the full WoundLens pipeline on a single image.

    Args:
        image_path: Path to wound image
        use_lab:    Whether to use LAB confirmation (default True)
        save_dir:   Directory to save output visualizations

    Returns:
        Complete analysis result dict
    """
    # --- Load image ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [ERROR] Could not load: {image_path}")
        return None

    filename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(filename)[0]
    print(f"\n{'='*60}")
    print(f"  ANALYZING: {filename}")
    print(f"  Image size: {img.shape[1]}x{img.shape[0]}")
    print(f"{'='*60}")

    # --- Step 1: Segmentation (no YOLO bbox — uses center heuristic) ---
    print("\n  [1/4] Segmenting wound region...")
    seg_result = segment_wound(img, bbox_xyxy=None)
    print(f"         Wound area: {seg_result['wound_area_px']:,} px ({seg_result['coverage_pct']}% of image)")
    print(f"         Wound bbox: {seg_result['wound_bbox']}")

    # --- Step 2: Tissue Classification ---
    print("\n  [2/4] Classifying tissue types...")
    tissue_result = classify_wound_tissue(
        seg_result["preprocessed"],
        wound_mask=seg_result["wound_mask"],
        use_lab_confirmation=use_lab
    )
    ratios = tissue_result["ratios"]
    print(f"         Granulation : {ratios['granulation']:6.2f}%  (healing)")
    print(f"         Slough      : {ratios['slough']:6.2f}%  (infection risk)")
    print(f"         Necrotic    : {ratios['necrotic']:6.2f}%  (dead tissue)")
    print(f"         Unclassified: {ratios['unclassified']:6.2f}%")
    print(f"         Dominant    : {tissue_result['dominant_tissue'].upper()}")
    print(f"         Severity    : {tissue_result['wound_severity'].upper()}")

    # --- Step 3: Staging ---
    print("\n  [3/4] Determining wound stage...")
    stage_result = classify_stage(
        ratios["granulation"],
        ratios["slough"],
        ratios["necrotic"]
    )
    print(f"         Stage       : {stage_result['stage']}")
    print(f"         Severity    : {stage_result['severity']}")
    print(f"         Description : {stage_result['description']}")

    # --- Step 4: Treatment ---
    print("\n  [4/4] Generating treatment recommendations...")
    treatment = get_treatment(stage_result["stage"])
    print(f"         Urgency     : {treatment['urgency']}")
    print(f"         Follow-up   : Every {treatment['follow_up_days']} day(s)")
    print(f"         Actions     : {len(treatment['primary_actions'])} steps")
    print(f"         Dressings   : {len(treatment['dressings'])} options")
    print(f"         Top action  : {treatment['primary_actions'][0]}")
    print(f"         Top dressing: {treatment['dressings'][0]['name']}")

    # --- Save Visualizations ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        preprocessed = seg_result["preprocessed"]

        # 1. Wound contour overlay
        vis_contour = draw_wound_contour(preprocessed, seg_result["wound_mask"])
        contour_path = os.path.join(save_dir, f"{name_no_ext}_1_contour.jpg")
        cv2.imwrite(contour_path, vis_contour)

        # 2. Wound mask highlight
        vis_mask = blend_mask_overlay(preprocessed, seg_result["wound_mask"])
        mask_path = os.path.join(save_dir, f"{name_no_ext}_2_mask.jpg")
        cv2.imwrite(mask_path, vis_mask)

        # 3. Tissue overlay (color-coded)
        vis_tissue = apply_tissue_overlay(preprocessed, tissue_result["masks"])
        vis_tissue = draw_tissue_legend(vis_tissue, ratios)
        tissue_path = os.path.join(save_dir, f"{name_no_ext}_3_tissue.jpg")
        cv2.imwrite(tissue_path, vis_tissue)

        # 4. Side-by-side comparison
        side_by_side = np.hstack([preprocessed, vis_contour, vis_tissue])
        combined_path = os.path.join(save_dir, f"{name_no_ext}_4_combined.jpg")
        cv2.imwrite(combined_path, side_by_side)

        # 5. Full analysis dashboard (the WOW image for judges)
        dashboard = create_analysis_dashboard(
            preprocessed, tissue_result, seg_result, stage_result, treatment
        )
        dashboard_path = os.path.join(save_dir, f"{name_no_ext}_5_dashboard.jpg")
        cv2.imwrite(dashboard_path, dashboard)

        print(f"\n  [SAVED] Results in: {save_dir}/")
        print(f"          {name_no_ext}_1_contour.jpg   — wound boundary")
        print(f"          {name_no_ext}_2_mask.jpg      — wound region highlight")
        print(f"          {name_no_ext}_3_tissue.jpg    — tissue classification overlay")
        print(f"          {name_no_ext}_4_combined.jpg   — side-by-side comparison")
        print(f"          {name_no_ext}_5_dashboard.jpg  — full analysis dashboard")

    return {
        "segmentation": seg_result,
        "tissue": tissue_result,
        "staging": stage_result,
        "treatment": treatment,
    }


def main():
    parser = argparse.ArgumentParser(
        description="WoundLens Pipeline Test — runs full analysis without YOLO"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single wound image")
    parser.add_argument("--all", action="store_true",
                        help="Run on ALL images in data/sample_wounds/")
    parser.add_argument("--no-lab", action="store_true",
                        help="Disable LAB confirmation (HSV-only mode)")
    parser.add_argument("--outdir", type=str, default="data/sample_wounds/results",
                        help="Output directory for visualizations")
    args = parser.parse_args()

    if not args.image and not args.all:
        print("WoundLens Pipeline Test")
        print("-" * 40)
        print("Usage:")
        print("  python test_pipeline.py --image <path_to_wound_image>")
        print("  python test_pipeline.py --all")
        print("")
        print("Place test images in: data/sample_wounds/")
        return

    use_lab = not args.no_lab
    results = []

    if args.image:
        # Single image mode
        result = run_pipeline(args.image, use_lab=use_lab, save_dir=args.outdir)
        if result:
            results.append(result)

    elif args.all:
        # Batch mode — all images in sample_wounds/
        image_dir = "data/sample_wounds"
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))

        if not image_files:
            print(f"No images found in {image_dir}/")
            print("Place wound images there and try again.")
            return

        print(f"Found {len(image_files)} images in {image_dir}/")
        for img_path in sorted(image_files):
            result = run_pipeline(img_path, use_lab=use_lab, save_dir=args.outdir)
            if result:
                results.append(result)

    # --- Summary ---
    if results:
        print(f"\n\n{'='*60}")
        print(f"  SUMMARY — {len(results)} image(s) analyzed")
        print(f"{'='*60}")
        for i, r in enumerate(results):
            ratios = r["tissue"]["ratios"]
            stage = r["staging"]["stage"]
            severity = r["staging"]["severity"]
            print(f"\n  Image {i+1}:")
            print(f"    Gran={ratios['granulation']:.1f}% | "
                  f"Slough={ratios['slough']:.1f}% | "
                  f"Necrotic={ratios['necrotic']:.1f}%")
            print(f"    Stage: {stage} ({severity})")
            print(f"    Treatment: {r['treatment']['dressings'][0]['name']}")

        print(f"\n{'='*60}")
        print(f"  Pipeline test complete. Check {args.outdir}/ for visuals.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
