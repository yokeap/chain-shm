"""
horizontal_correction.py
========================
Perspective correction for horizontal chain camera angle.

Problem:
  Camera is not perpendicular to the horizontal chain axis —
  it views the chain from an angle (e.g., from the right).
  This causes the wire to appear wider on the near side and
  narrower on the far side, inflating wear% on one side.

Solution:
  Use the longest horizontal wire segment's top/bottom edge lines
  to compute a homography that rectifies the wire to be parallel
  and uniform width across the image.

  Source: 4 corners of the wire trapezoid (from RANSAC-fit edges)
  Dest:   4 corners of a rectangle (same average width)

Usage:
  from horizontal_correction import correct_horizontal_perspective

  img_corr, H_matrix = correct_horizontal_perspective(image, hw_result)
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def correct_horizontal_perspective(
    image: np.ndarray,
    hw_result: Dict,
    mask: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    """
    Rectify the image so the horizontal wire appears with uniform width.

    Parameters
    ----------
    image     : input image (BGR)
    hw_result : dict from model_horizontal_wire()
    mask      : optional mask to warp alongside

    Returns
    -------
    img_out   : rectified image
    mask_out  : rectified mask (or None)
    H         : 3x3 homography matrix
    info      : dict with correction details
    """
    h_img, w_img = image.shape[:2]

    segments = hw_result.get("segments", [])
    if not segments:
        logger.warning("No wire segments for horizontal correction")
        return image, np.eye(3), mask, {}

    # Pick the longest wire segment (most reliable edges)
    seg = max(segments, key=lambda s: s["x2"] - s["x1"])

    s_top, i_top = seg["slope_top"], seg["intercept_top"]
    s_bot, i_bot = seg["slope_bot"], seg["intercept_bot"]
    x1, x2 = seg["x1"], seg["x2"]

    # Source points: 4 corners of the wire trapezoid
    src = np.float32([
        [x1, s_top * x1 + i_top],   # top-left
        [x2, s_top * x2 + i_top],   # top-right
        [x2, s_bot * x2 + i_bot],   # bot-right
        [x1, s_bot * x1 + i_bot],   # bot-left
    ])

    # Compute average wire width (b) at the midpoint
    x_mid = (x1 + x2) / 2.0
    y_top_mid = s_top * x_mid + i_top
    y_bot_mid = s_bot * x_mid + i_bot
    b_avg = y_bot_mid - y_top_mid

    # Destination: rectangle with uniform width = b_avg
    # Keep the same x range, center the wire at the average y position
    y_center = (y_top_mid + y_bot_mid) / 2.0
    dst = np.float32([
        [x1, y_center - b_avg / 2],   # top-left
        [x2, y_center - b_avg / 2],   # top-right
        [x2, y_center + b_avg / 2],   # bot-right
        [x1, y_center + b_avg / 2],   # bot-left
    ])

    # Check if correction is needed (if slopes are already ~0 and parallel)
    slope_diff = abs(s_top - s_bot)
    max_slope = max(abs(s_top), abs(s_bot))

    # Compute width difference between left and right ends
    b_left = abs((s_bot * x1 + i_bot) - (s_top * x1 + i_top))
    b_right = abs((s_bot * x2 + i_bot) - (s_top * x2 + i_top))
    b_diff = abs(b_right - b_left)
    b_diff_pct = b_diff / max(b_avg, 1) * 100

    logger.info(
        f"  H-correction: b_left={b_left:.1f}  b_right={b_right:.1f}  "
        f"diff={b_diff:.1f}px ({b_diff_pct:.1f}%)  "
        f"slope_top={s_top:.4f}  slope_bot={s_bot:.4f}"
    )

    # Compute homography
    H = cv2.getPerspectiveTransform(src, dst)

    # Warp image
    img_out = cv2.warpPerspective(image, H, (w_img, h_img),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

    # Warp mask if provided
    mask_out = None
    if mask is not None:
        mask_out = cv2.warpPerspective(mask, H, (w_img, h_img),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT)

    info = {
        "b_left": b_left,
        "b_right": b_right,
        "b_avg": b_avg,
        "b_diff": b_diff,
        "b_diff_pct": b_diff_pct,
        "slope_top": s_top,
        "slope_bot": s_bot,
        "H": H,
    }

    return img_out, H, mask_out, info
