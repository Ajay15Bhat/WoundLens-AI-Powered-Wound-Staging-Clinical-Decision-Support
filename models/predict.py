"""
models/predict.py
=================
WoundLens -- Wound Region Detection
Member 1: ML Engineer

Two detection strategies:
  1. Roboflow API (if available and working)
  2. Color-based wound detection fallback (works offline, no API needed)
"""

import cv2
import requests
import numpy as np
from typing import Optional, Tuple

API_KEY = "tnehPf1LOMfUyfJAv4dn"
MODEL_ID = "wound-qr9o-e2jqw/1"


def _detect_roboflow(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Try Roboflow API. Returns bbox or None on any failure."""
    try:
        url = f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}"
        success, buffer = cv2.imencode('.jpg', image_bgr)
        if not success:
            return None
        img_bytes = buffer.tobytes()
        response = requests.post(
            url,
            files={"file": ("image.jpg", img_bytes, "image/jpeg")},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        if "predictions" not in data or len(data["predictions"]) == 0:
            return None
        pred = max(data["predictions"], key=lambda x: x.get("confidence", 0))
        if pred.get("confidence", 0) < 0.4:
            return None
        x_center, y_center = pred["x"], pred["y"]
        width, height = pred["width"], pred["height"]
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        h, w = image_bgr.shape[:2]
        return (max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))
    except Exception:
        return None


def _detect_color_based(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect wound region using color analysis in HSV and LAB spaces.
    Wounds typically appear as red/pink/dark regions that differ from
    surrounding healthy skin. This works offline without any API.
    """
    h, w = image_bgr.shape[:2]

    # Convert to HSV and LAB
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # --- Wound-like color masks ---
    # Red/pink tones (granulation tissue, fresh wounds, blood)
    red_low1 = cv2.inRange(hsv, (0, 50, 40), (12, 255, 255))
    red_low2 = cv2.inRange(hsv, (160, 50, 40), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_low1, red_low2)

    # Dark tissue (necrotic, scabs, eschar)
    dark_mask = cv2.inRange(hsv, (0, 0, 10), (180, 120, 80))

    # Yellow/cream (slough, pus)
    yellow_mask = cv2.inRange(hsv, (15, 40, 80), (40, 255, 255))

    # High A-channel in LAB (reddish tones are strong wound indicators)
    a_channel = lab[:, :, 1]
    lab_red = (a_channel > 140).astype(np.uint8) * 255

    # Combine all wound-like regions
    wound_combined = cv2.bitwise_or(red_mask, dark_mask)
    wound_combined = cv2.bitwise_or(wound_combined, yellow_mask)
    wound_combined = cv2.bitwise_or(wound_combined, lab_red)

    # Morphological cleanup to merge nearby regions and remove noise
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    wound_combined = cv2.morphologyEx(wound_combined, cv2.MORPH_CLOSE, kernel_close)
    wound_combined = cv2.morphologyEx(wound_combined, cv2.MORPH_OPEN, kernel_open)

    # Find contours and pick the largest wound-like blob
    contours, _ = cv2.findContours(wound_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter: wound must be at least 1% of image area
    min_area = h * w * 0.01
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid:
        return None

    # Take the largest valid contour
    largest = max(valid, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # Add a small padding around the detected region (10% of bbox dims)
    pad_x = int(bw * 0.10)
    pad_y = int(bh * 0.10)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w - 1, x + bw + pad_x)
    y2 = min(h - 1, y + bh + pad_y)

    return (x1, y1, x2, y2)


def detect_wound(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the wound in the image and return its bounding box.

    Strategy:
      1. Try Roboflow API first (most accurate, needs internet)
      2. Fall back to color-based detection (works offline, still good)

    Returns:
        Bounding box as (x1, y1, x2, y2) in pixel coordinates, or None.
    """
    # Try Roboflow first
    bbox = _detect_roboflow(image_bgr)
    if bbox is not None:
        return bbox

    # Fall back to color-based detection
    bbox = _detect_color_based(image_bgr)
    return bbox
