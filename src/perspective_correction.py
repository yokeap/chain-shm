"""
perspective_correction.py
=========================
Correct camera tilt using symmetry constraint ของ vertical chain link.

โซ่กลม → d_top == d_bot (ถ้าไม่มี wear)
ถ้า d_top ≠ d_bot → camera tilt รอบแกนนอน

วิธี:
  1. หา y_sym = midpoint ระหว่าง top-inner และ bot-inner ที่ apex
  2. คำนวณ scale_top = d_target/d_top, scale_bot = d_target/d_bot
  3. สร้าง remap: แต่ละ pixel ใน dst → map ไป src_y โดย scale รอบ y_sym
  4. Warp image + mask ด้วย cv2.remap

Result: d_top = d_bot = d_target = (d_top_orig + d_bot_orig) / 2

Usage
-----
    from perspective_correction import correct_tilt
    img_rect, mask_rect, info = correct_tilt(image, vert_mask)
    # แล้วส่ง img_rect, mask_rect ไปให้ reconstruct_links
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Helpers (dup เล็กน้อยจาก reconstruction.py เพื่อ independence)
# ══════════════════════════════════════════════════════════════════

def _scan_edges(comp, x1, x2):
    xs, tops, bots = [], [], []
    W = comp.shape[1]
    for x in range(max(0, x1), min(W, x2 + 1)):
        px = np.where(comp[:, x] > 0)[0]
        if len(px):
            xs.append(x); tops.append(int(px[0])); bots.append(int(px[-1]))
    return np.array(xs), np.array(tops, float), np.array(bots, float)


def _find_tip(xs, tops, bots, side, tip_height=6):
    heights = bots - tops
    if side == "left":
        idx = int(np.argmax(heights >= tip_height))
        return max(0, idx - 1)
    arr   = heights[::-1]
    idx_r = int(np.argmax(arr >= tip_height))
    return len(heights) - 1 - max(0, idx_r - 1)


def _nearest_idx(xs, x_target):
    return int(np.argmin(np.abs(xs - x_target)))


def _pair_blobs(mask):
    lab, n = scipy_label(mask > 0)
    blobs  = []
    for i in range(1, n + 1):
        comp = (lab == i).astype(np.uint8) * 255
        if comp.sum() // 255 < 2000: continue
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        blobs.append({"comp": comp,
                      "y1": int(rows[0]), "y2": int(rows[-1]),
                      "x1": int(cols[0]), "x2": int(cols[-1]),
                      "cy": float(rows.mean()), "cx": float(cols.mean())})
    if not blobs: return None
    med_cy = np.median([b["cy"] for b in blobs])
    tops_b = sorted([b for b in blobs if b["cy"] < med_cy],  key=lambda b: b["cx"])
    bots_b = sorted([b for b in blobs if b["cy"] >= med_cy], key=lambda b: b["cx"])
    if not tops_b or not bots_b: return None
    t, b = tops_b[0], bots_b[0]
    return {"top": t["comp"], "bot": b["comp"],
            "top_x1": t["x1"], "top_x2": t["x2"],
            "x1": max(t["x1"], b["x1"]), "x2": min(t["x2"], b["x2"])}


# ══════════════════════════════════════════════════════════════════
# Core analysis
# ══════════════════════════════════════════════════════════════════

def compute_tilt_info(vert_mask: np.ndarray) -> Dict:
    """
    วิเคราะห์ tilt จาก mask
    คืน dict: d_top, d_bot, d_target, y_sym, scale_top, scale_bot, theta_deg
    """
    import math
    _, bw = cv2.threshold(vert_mask, 127, 255, cv2.THRESH_BINARY)
    pair  = _pair_blobs(bw)
    if pair is None:
        raise ValueError("Cannot find link pair in mask")

    # top blob
    xs_t, tops_t, bots_t = _scan_edges(pair["top"], pair["top_x1"], pair["top_x2"])
    li_t = _find_tip(xs_t, tops_t, bots_t, "left")
    ri_t = _find_tip(xs_t, tops_t, bots_t, "right")
    ai_t = _nearest_idx(xs_t, (int(xs_t[li_t]) + int(xs_t[ri_t])) // 2)

    # bot blob
    xs_b, tops_b, bots_b = _scan_edges(pair["bot"], pair["x1"], pair["x2"])
    sm_b  = uniform_filter1d(tops_b.astype(float), size=30)
    half  = len(xs_b) // 2
    li_b  = _find_tip(xs_b, tops_b, bots_b, "left")
    ri_b  = _find_tip(xs_b, tops_b, bots_b, "right")
    ai_b  = _nearest_idx(xs_b, (int(xs_b[li_b]) + int(xs_b[ri_b])) // 2)

    d_top    = float(bots_t[ai_t] - tops_t[ai_t])
    d_bot    = float(bots_b[ai_b] - tops_b[ai_b])
    d_target = (d_top + d_bot) / 2.0

    y_top_inner = float(bots_t[ai_t])
    y_bot_inner = float(tops_b[ai_b])
    y_sym       = (y_top_inner + y_bot_inner) / 2.0

    theta_rad = math.acos(min(1.0, d_bot / d_top)) if d_top > 0 else 0.0

    return {
        "d_top"    : d_top,
        "d_bot"    : d_bot,
        "d_target" : d_target,
        "y_sym"    : y_sym,
        "scale_top": d_target / d_top,   # < 1: compress
        "scale_bot": d_target / d_bot,   # > 1: expand
        "theta_deg": math.degrees(theta_rad),
        "apex_x"   : int(xs_t[ai_t]),
    }


# ══════════════════════════════════════════════════════════════════
# Remap correction
# ══════════════════════════════════════════════════════════════════

def build_remap(info: Dict, img_shape: Tuple[int, int]):
    """
    สร้าง map_x, map_y สำหรับ cv2.remap

    สำหรับแต่ละ dst pixel (x, y):
      ถ้า y ≤ y_sym → src_y = y_sym + (y - y_sym) / scale_top
      ถ้า y > y_sym → src_y = y_sym + (y - y_sym) / scale_bot
      src_x = x  (ไม่แตะแกน x)
    """
    h, w  = img_shape
    y_sym = info["y_sym"]
    s_top = info["scale_top"]
    s_bot = info["scale_bot"]

    ys_col = np.arange(h, dtype=np.float32).reshape(h, 1)
    xs_row = np.arange(w, dtype=np.float32).reshape(1, w)

    map_x = np.broadcast_to(xs_row, (h, w)).copy()
    map_y = np.where(
        ys_col <= y_sym,
        y_sym + (ys_col - y_sym) / s_top,
        y_sym + (ys_col - y_sym) / s_bot,
    ).astype(np.float32)
    map_y = np.broadcast_to(map_y, (h, w)).copy()

    return map_x, map_y


def correct_tilt(
    image    : np.ndarray,
    vert_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Rectify camera tilt so d_top ≈ d_bot

    Returns
    -------
    img_rect  : rectified BGR image
    mask_rect : rectified binary mask
    info      : tilt analysis dict (includes d_top, d_bot, theta_deg, etc.)
    """
    h, w = image.shape[:2]

    info = compute_tilt_info(vert_mask)
    logger.info(
        f"Tilt: d_top={info['d_top']:.1f}px  d_bot={info['d_bot']:.1f}px  "
        f"θ={info['theta_deg']:.2f}°  y_sym={info['y_sym']:.1f}  "
        f"scale_top={info['scale_top']:.4f}  scale_bot={info['scale_bot']:.4f}"
    )

    map_x, map_y = build_remap(info, (h, w))

    img_rect  = cv2.remap(image,     map_x, map_y,
                           cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REPLICATE)
    mask_rect = cv2.remap(vert_mask, map_x, map_y,
                           cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)

    logger.info(f"Rectification done → d_target={info['d_target']:.1f}px")
    return img_rect, mask_rect, info


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    import argparse, json
    from pathlib import Path
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask",     required=True)
    parser.add_argument("--image",    required=True)
    parser.add_argument("--save-dir", default="debug_seg")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(args.image)
    mask  = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    img_rect, mask_rect, info = correct_tilt(image, mask)

    stem = Path(args.image).stem
    cv2.imwrite(str(save_dir / f"rect_{stem}.jpg"),      img_rect)
    cv2.imwrite(str(save_dir / f"rect_mask_{stem}.jpg"), mask_rect)

    print(f"\n{'─'*52}")
    print(f"  Before: d_top={info['d_top']:.1f}px  d_bot={info['d_bot']:.1f}px")
    print(f"  θ     : {info['theta_deg']:.2f}°")
    print(f"  After : d_top = d_bot = {info['d_target']:.1f}px")
    print(f"  Saved : {save_dir}/rect_{stem}.jpg")
    print(f"{'─'*52}\n")


if __name__ == "__main__":
    main()