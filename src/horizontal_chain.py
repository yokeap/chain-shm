"""
horizontal_chain.py
===================
Model the horizontal chain link (wire) from its binary mask.

Outputs
-------
For each horizontal wire segment visible in the mask:
  - Top/bottom edge lines (red) — fitted via RANSAC or linear regression
  - Wire thickness **b** (px) — perpendicular distance between the two edges
  - Tangent circle at tip point (1) — diameter = b, tangent to the wire's
    end-cap arc.  This circle represents the top-view cross-section of the
    horizontal link at the point where it contacts the vertical link.

Context
-------
b is later combined with d (vertical-link wire thickness from
reconstruction.py) to compute wear:
    wear% = 1 − (b + d) / (b_nom + d_nom)
This module only computes b; wear calculation is done downstream.

Usage
-----
    from horizontal_chain import model_horizontal_wire

    hw = model_horizontal_wire(wire_mask)
    # hw["b_px"]         → float
    # hw["top_line"]      → ((x1,y1),(x2,y2))  — top edge
    # hw["bot_line"]      → ((x1,y1),(x2,y2))  — bottom edge
    # hw["segments"]      → list of per-segment dicts
"""

from __future__ import annotations
import logging, math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# A.  Segment extraction — find individual horizontal wire blobs
# ══════════════════════════════════════════════════════════════════

def _extract_wire_segments(wire_mask: np.ndarray, min_area: int = 3000) -> List[Dict]:
    """
    Label connected components in wire_mask and return blobs
    sorted left-to-right.
    """
    _, bw = cv2.threshold(wire_mask, 127, 255, cv2.THRESH_BINARY)
    lbl, n = scipy_label(bw > 0)
    segments = []
    for i in range(1, n + 1):
        comp = (lbl == i).astype(np.uint8) * 255
        area = int(comp.sum() // 255)
        if area < min_area:
            continue
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        segments.append({
            "comp": comp,
            "area": area,
            "x1": int(cols[0]),  "x2": int(cols[-1]),
            "y1": int(rows[0]),  "y2": int(rows[-1]),
            "cx": float(cols.mean()),
            "cy": float(rows.mean()),
        })
    segments.sort(key=lambda s: s["cx"])
    return segments


# ══════════════════════════════════════════════════════════════════
# B.  Edge extraction — top / bottom boundary of the wire
# ══════════════════════════════════════════════════════════════════

def _scan_top_bot(comp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each x-column in comp, find the topmost and bottommost white pixel.
    Returns (xs, tops, bots) arrays.
    """
    cols = np.where(comp.any(axis=0))[0]
    xs, tops, bots = [], [], []
    for x in cols:
        px = np.where(comp[:, x] > 0)[0]
        if len(px):
            xs.append(x)
            tops.append(int(px[0]))
            bots.append(int(px[-1]))
    return np.array(xs), np.array(tops, dtype=float), np.array(bots, dtype=float)


# ══════════════════════════════════════════════════════════════════
# C.  Line fitting — RANSAC for robust top/bottom edges
# ══════════════════════════════════════════════════════════════════

def _fit_line_ransac(
    xs: np.ndarray,
    ys: np.ndarray,
    n_iter: int = 500,
    thresh: float = 3.0,
) -> Tuple[float, float]:
    """
    RANSAC line fit  y = slope * x + intercept.
    Returns (slope, intercept).
    """
    best_inliers = 0
    best_s, best_i = 0.0, float(np.mean(ys))
    n = len(xs)
    if n < 2:
        return best_s, best_i

    for _ in range(n_iter):
        idx = np.random.choice(n, 2, replace=False)
        x0, x1 = xs[idx]
        y0, y1 = ys[idx]
        dx = x1 - x0
        if abs(dx) < 1e-6:
            continue
        s = (y1 - y0) / dx
        ic = y0 - s * x0
        residuals = np.abs(ys - (s * xs + ic))
        inliers = int((residuals < thresh).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            # Re-fit on inliers
            mask = residuals < thresh
            if mask.sum() >= 2:
                A = np.column_stack([xs[mask], np.ones(mask.sum())])
                coef, _, _, _ = np.linalg.lstsq(A, ys[mask], rcond=None)
                best_s, best_i = float(coef[0]), float(coef[1])

    return best_s, best_i


def _line_endpoints(slope, intercept, x1, x2):
    """Return two (x,y) endpoints of a line segment."""
    return ((int(x1), int(slope * x1 + intercept)),
            (int(x2), int(slope * x2 + intercept)))


def _perpendicular_distance(s1, i1, s2, i2, x_mid):
    """
    Perpendicular distance between two parallel-ish lines at x_mid.
    Lines: y = s*x + i.
    """
    y1 = s1 * x_mid + i1
    y2 = s2 * x_mid + i2
    dy = abs(y2 - y1)
    # True perpendicular distance (accounting for slope)
    avg_slope = (s1 + s2) / 2.0
    return dy * math.cos(math.atan(avg_slope))


# ══════════════════════════════════════════════════════════════════
# D.  Tip detection — find the arc end-cap of the wire
# ══════════════════════════════════════════════════════════════════

def _find_wire_tips(
    comp: np.ndarray,
    xs: np.ndarray,
    tops: np.ndarray,
    bots: np.ndarray,
) -> List[Dict]:
    """
    Detect the rounded end-caps (tips) of the wire.

    A tip is where the wire narrows rapidly — the height (bots-tops)
    drops from the full wire width to near zero within a short span.

    Tips that sit at the image edge (within edge_margin pixels) are
    excluded — they are partial wire segments cut off by the frame,
    not true end-caps.

    Returns list of dicts with:
      side:    "left" or "right"
      x_tip:   x at the tip apex (point 1)
      y_tip:   y at the tip apex
      x_start: x where the wire begins narrowing
    """
    heights = bots - tops
    if len(heights) < 10:
        return []

    h, w = comp.shape[:2]
    edge_margin = max(15, int(w * 0.02))   # ~2% of image width

    h_max = np.percentile(heights, 90)
    h_thresh = h_max * 0.5

    tips = []

    # Left tip
    for i in range(len(heights)):
        if heights[i] >= h_thresh:
            x_start = int(xs[i])
            x_tip = int(xs[0])
            y_tip = int((tops[0] + bots[0]) / 2)
            # Reject if tip is at the image edge (wire continues beyond frame)
            if x_tip > edge_margin:
                tips.append({
                    "side": "left",
                    "x_tip": x_tip, "y_tip": y_tip,
                    "x_start": x_start,
                })
            break

    # Right tip
    for i in range(len(heights) - 1, -1, -1):
        if heights[i] >= h_thresh:
            x_start = int(xs[i])
            x_tip = int(xs[-1])
            y_tip = int((tops[-1] + bots[-1]) / 2)
            if x_tip < w - edge_margin:
                tips.append({
                    "side": "right",
                    "x_tip": x_tip, "y_tip": y_tip,
                    "x_start": x_start,
                })
            break

    return tips


# ══════════════════════════════════════════════════════════════════
# E.  Tangent circle at tip
# ══════════════════════════════════════════════════════════════════

def _tangent_circle_at_tip(
    tip: Dict,
    b_px: float,
    slope_top: float,
    intercept_top: float,
    slope_bot: float,
    intercept_bot: float,
) -> Dict:
    """
    Place a circle of diameter b tangent to the wire end-cap arc.

    The circle must be INSCRIBED — its edge touches the tip point,
    and its center sits one radius INWARD along the wire centerline.
    
    For a left tip:  center_x = x_tip + r  (shifted right = inward)
    For a right tip: center_x = x_tip - r  (shifted left  = inward)
    """
    x_tip = tip["x_tip"]
    r  = b_px / 2.0

    # Shift center inward along the wire axis
    if tip["side"] == "left":
        cx = float(x_tip + r)
    else:
        cx = float(x_tip - r)

    # Center y = midpoint of top/bottom edges at the circle center x
    y_top = slope_top * cx + intercept_top
    y_bot = slope_bot * cx + intercept_bot
    cy = (y_top + y_bot) / 2.0

    return {
        "center": (cx, float(cy)),
        "radius": float(r),
        "diameter_px": float(b_px),
    }


# ══════════════════════════════════════════════════════════════════
# F.  Public API
# ══════════════════════════════════════════════════════════════════

def model_horizontal_wire(
    wire_mask: np.ndarray,
    vert_cp_top: Optional[Dict] = None,
    vert_cp_bot: Optional[Dict] = None,
) -> Dict:
    """
    Model horizontal wire(s) from the wire binary mask.

    Parameters
    ----------
    wire_mask    : uint8 binary mask of horizontal wire(s)
    vert_cp_top  : (optional) control points from vertical link top blob —
                   used to identify which wire tip is adjacent to pt(1)
    vert_cp_bot  : (optional) control points from vertical link bottom blob

    Returns
    -------
    dict with:
      segments   : list of per-segment analysis dicts
      b_mean_px  : mean wire thickness across all segments
    """
    segments = _extract_wire_segments(wire_mask)
    logger.info(f"Found {len(segments)} horizontal wire segment(s)")

    results = []
    for si, seg in enumerate(segments):
        comp = seg["comp"]
        xs, tops, bots = _scan_top_bot(comp)
        if len(xs) < 20:
            logger.warning(f"  Segment {si}: too few columns ({len(xs)}), skipping")
            continue

        # ── Exclude tip regions for edge fitting ──
        # Use the central 60% of the wire (exclude rounded tips)
        n = len(xs)
        i_start = int(n * 0.20)
        i_end   = int(n * 0.80)
        xs_mid   = xs[i_start:i_end]
        tops_mid = tops[i_start:i_end]
        bots_mid = bots[i_start:i_end]

        if len(xs_mid) < 10:
            logger.warning(f"  Segment {si}: central region too small, skipping")
            continue

        # ── Fit top and bottom edge lines ──
        s_top, i_top = _fit_line_ransac(xs_mid, tops_mid)
        s_bot, i_bot = _fit_line_ransac(xs_mid, bots_mid)

        # ── Measure b (perpendicular wire thickness) ──
        x_mid = float(xs_mid[len(xs_mid) // 2])
        b_px = _perpendicular_distance(s_top, i_top, s_bot, i_bot, x_mid)

        # Also measure b at several x positions for robustness
        b_samples = []
        for frac in (0.25, 0.4, 0.5, 0.6, 0.75):
            xi = float(xs_mid[int(len(xs_mid) * frac)])
            bi = _perpendicular_distance(s_top, i_top, s_bot, i_bot, xi)
            b_samples.append(bi)
        b_median = float(np.median(b_samples))

        logger.info(
            f"  Segment {si}: b={b_median:.1f}px  "
            f"slope_top={s_top:.4f}  slope_bot={s_bot:.4f}  "
            f"x=[{seg['x1']},{seg['x2']}]"
        )

        # ── Edge line endpoints (for visualization) ──
        top_line = _line_endpoints(s_top, i_top, seg["x1"], seg["x2"])
        bot_line = _line_endpoints(s_bot, i_bot, seg["x1"], seg["x2"])

        # ── Detect tips ──
        tips = _find_wire_tips(comp, xs, tops, bots)

        # ── Tangent circles at tips ──
        circles = []
        for tip in tips:
            circ = _tangent_circle_at_tip(tip, b_median, s_top, i_top, s_bot, i_bot)
            circles.append({**tip, **circ})
            logger.info(
                f"    Tip {tip['side']}: ({tip['x_tip']},{tip['y_tip']})  "
                f"circle center=({circ['center'][0]:.0f},{circ['center'][1]:.0f})  "
                f"r={circ['radius']:.1f}px"
            )

        results.append({
            "seg_index"  : si,
            "x1": seg["x1"], "x2": seg["x2"],
            "y1": seg["y1"], "y2": seg["y2"],
            "b_px"       : b_median,
            "b_samples"  : b_samples,
            "slope_top"  : s_top,
            "intercept_top": i_top,
            "slope_bot"  : s_bot,
            "intercept_bot": i_bot,
            "top_line"   : top_line,
            "bot_line"   : bot_line,
            "tips"       : circles,
        })

    b_all = [r["b_px"] for r in results]
    b_mean = float(np.mean(b_all)) if b_all else 0.0

    return {
        "segments" : results,
        "b_mean_px": b_mean,
    }


# ══════════════════════════════════════════════════════════════════
# G.  Visualization
# ══════════════════════════════════════════════════════════════════

def draw_horizontal_wire(
    image: np.ndarray,
    hw_result: Dict,
    wire_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw edge lines, b measurements, tips, and tangent circles."""
    vis = image.copy()

    RED    = (0,   50, 255)
    YELLOW = (0,  220, 255)
    GREEN  = (50, 200,  50)
    WHITE  = (255, 255, 255)
    CYAN   = (255, 200,   0)

    # Light overlay of wire mask
    if wire_mask is not None:
        ov = np.zeros_like(vis)
        ov[wire_mask > 127] = (80, 60, 40)
        vis = cv2.addWeighted(vis, 0.7, ov, 0.3, 0)

    for seg in hw_result["segments"]:
        # Top / bottom edge lines (red)
        cv2.line(vis, seg["top_line"][0], seg["top_line"][1], RED, 2, cv2.LINE_AA)
        cv2.line(vis, seg["bot_line"][0], seg["bot_line"][1], RED, 2, cv2.LINE_AA)

        # b measurement at midpoint
        x_mid = (seg["x1"] + seg["x2"]) // 2
        y_top = int(seg["slope_top"] * x_mid + seg["intercept_top"])
        y_bot = int(seg["slope_bot"] * x_mid + seg["intercept_bot"])
        cv2.line(vis, (x_mid, y_top), (x_mid, y_bot), YELLOW, 3, cv2.LINE_AA)
        cv2.putText(vis, f"b={seg['b_px']:.0f}px",
                    (x_mid + 8, (y_top + y_bot) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2, cv2.LINE_AA)

        # Tips and tangent circles
        for tip in seg["tips"]:
            # Tip point
            cv2.circle(vis, (tip["x_tip"], tip["y_tip"]), 8, YELLOW, -1, cv2.LINE_AA)
            cv2.circle(vis, (tip["x_tip"], tip["y_tip"]), 8, WHITE, 1, cv2.LINE_AA)
            cv2.putText(vis, "1", (tip["x_tip"] - 18, tip["y_tip"] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)

            # Tangent circle (green)
            cx, cy = int(tip["center"][0]), int(tip["center"][1])
            r = int(tip["radius"])
            cv2.circle(vis, (cx, cy), r, GREEN, 2, cv2.LINE_AA)

            # b diameter line inside circle
            cv2.line(vis, (cx, cy - r), (cx, cy + r), YELLOW, 2, cv2.LINE_AA)
            cv2.putText(vis, "b", (cx + 6, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)

    return vis


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    from pathlib import Path

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(
        description="Model horizontal wire from mask → measure b"
    )
    parser.add_argument("--wire-mask", required=True, help="Wire mask image")
    parser.add_argument("--image",     required=True, help="Original image (for overlay)")
    parser.add_argument("--save-dir",  default="debug_seg")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wire_mask = cv2.imread(args.wire_mask, cv2.IMREAD_GRAYSCALE)
    image     = cv2.imread(args.image)
    if wire_mask is None: raise FileNotFoundError(args.wire_mask)
    if image is None:     raise FileNotFoundError(args.image)

    hw = model_horizontal_wire(wire_mask)

    vis = draw_horizontal_wire(image, hw, wire_mask)
    stem = Path(args.image).stem
    out = save_dir / f"hwire_{stem}.jpg"
    cv2.imwrite(str(out), vis)
    logger.info(f"Saved: {out}")

    # JSON export
    export = {
        "b_mean_px": hw["b_mean_px"],
        "segments": [],
    }
    for seg in hw["segments"]:
        s = {k: v for k, v in seg.items()
             if k not in ("comp",)}
        # Convert tuple endpoints to lists
        s["top_line"] = [list(p) for p in seg["top_line"]]
        s["bot_line"] = [list(p) for p in seg["bot_line"]]
        s["tips"] = []
        for tip in seg["tips"]:
            t = {k: v for k, v in tip.items()}
            t["center"] = list(tip["center"])
            s["tips"].append(t)
        export["segments"].append(s)
    with open(save_dir / f"hwire_{stem}.json", "w") as f:
        json.dump(export, f, indent=2)

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Horizontal wire model")
    print(sep)
    print(f"  Segments : {len(hw['segments'])}")
    print(f"  b_mean   : {hw['b_mean_px']:.1f} px")
    for seg in hw["segments"]:
        print(f"  Seg {seg['seg_index']}: b={seg['b_px']:.1f}px  "
              f"x=[{seg['x1']},{seg['x2']}]  tips={len(seg['tips'])}")
        for tip in seg["tips"]:
            print(f"    {tip['side']}: tip=({tip['x_tip']},{tip['y_tip']})  "
                  f"circle=({tip['center'][0]:.0f},{tip['center'][1]:.0f}) r={tip['radius']:.0f}")
    print(sep)


if __name__ == "__main__":
    import json
    main()