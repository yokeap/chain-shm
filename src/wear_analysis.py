"""
wear_analysis.py
================
Area-based chain wear measurement.

Cross-section (crescent) — region between outer and inner side arcs:
  RIGHT boundary : outer side-arc (orange dashed)
  LEFT  boundary : inner side-arc (yellow dashed)
  TOP/BOTTOM     : horizontal wire edges (red lines)

Polygon:
  outer_pts (top→bot) + inner_pts reversed (bot→top), clipped to wire band.

Interference:
  interf_mask = crescent ∩ circle_mask
  wear%       = interf_area / cs_area × 100
"""

from __future__ import annotations
import logging, math
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# A.  Build crescent mask
# ══════════════════════════════════════════════════════════════════

def _wire_y_at_x(slope: float, intercept: float, x: float) -> float:
    return slope * x + intercept


def _build_crescent_mask(
    side_arc : Dict,
    wire_seg : Dict,
    shape    : Tuple[int, int],
    full_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Crescent = region between outer_pts and inner_pts, clipped to wire band.

    Polygon:
      outer_pts (top→bot) + inner_pts[::-1] (bot→top)
    Then AND with wire_band and optionally full_mask.
    """
    h, w = shape
    s_top = wire_seg["slope_top"];  i_top = wire_seg["intercept_top"]
    s_bot = wire_seg["slope_bot"];  i_bot = wire_seg["intercept_bot"]

    outer_pts = side_arc["outer_pts"].copy()
    inner_pts = side_arc["inner_pts"].copy()

    # Sort both top→bottom
    if outer_pts[0, 1] > outer_pts[-1, 1]:
        outer_pts = outer_pts[::-1]
    if inner_pts[0, 1] > inner_pts[-1, 1]:
        inner_pts = inner_pts[::-1]

    # Find wire band y at the midpoint x of each curve
    x_mid_o = float(np.median(outer_pts[:, 0]))
    x_mid_i = float(np.median(inner_pts[:, 0]))
    x_mid   = (x_mid_o + x_mid_i) / 2.0
    y_band_top = _wire_y_at_x(s_top, i_top, x_mid)
    y_band_bot = _wire_y_at_x(s_bot, i_bot, x_mid)
    if y_band_top > y_band_bot:
        y_band_top, y_band_bot = y_band_bot, y_band_top

    tol = 6
    oc = outer_pts[(outer_pts[:, 1] >= y_band_top - tol) &
                   (outer_pts[:, 1] <= y_band_bot + tol)]
    ic = inner_pts[(inner_pts[:, 1] >= y_band_top - tol) &
                   (inner_pts[:, 1] <= y_band_bot + tol)]

    if len(oc) < 2 or len(ic) < 2:
        logger.warning(f"  crescent: not enough pts after clip "
                       f"(outer={len(oc)}, inner={len(ic)})")
        return np.zeros((h, w), dtype=np.uint8)

    # Polygon: outer top→bot, then inner bot→top (closing the banana shape)
    polygon = np.vstack([oc, ic[::-1]]).astype(np.int32)

    mask_out = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_out, [polygon], 255)

    # Clip to wire band column-by-column
    x_lo = int(max(0,   min(oc[:, 0].min(), ic[:, 0].min()) - 2))
    x_hi = int(min(w-1, max(oc[:, 0].max(), ic[:, 0].max()) + 2))
    wire_band = np.zeros((h, w), dtype=np.uint8)
    for x in range(x_lo, x_hi + 1):
        yt = int(max(0,   math.floor(_wire_y_at_x(s_top, i_top, x))))
        yb = int(min(h-1, math.ceil (_wire_y_at_x(s_bot, i_bot, x))))
        if yt < yb:
            wire_band[yt:yb + 1, x] = 255

    mask_out = cv2.bitwise_and(mask_out, wire_band)

    # Optionally clip to chain material
    if full_mask is not None:
        _, bw = cv2.threshold(full_mask, 127, 255, cv2.THRESH_BINARY)
        if bw.shape[:2] != (h, w):
            bw = cv2.resize(bw, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_out = cv2.bitwise_and(mask_out, bw)

    return mask_out


# ══════════════════════════════════════════════════════════════════
# B.  Circle mask
# ══════════════════════════════════════════════════════════════════

def _build_circle_mask(tip: Dict, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask,
               (int(tip["center"][0]), int(tip["center"][1])),
               int(tip["radius"]), 255, -1)
    return mask


# ══════════════════════════════════════════════════════════════════
# C.  Compute wear
# ══════════════════════════════════════════════════════════════════

def _collect_all_tips(hw_result: Dict) -> List[Dict]:
    tips = []
    for seg in hw_result.get("segments", []):
        for tip in seg.get("tips", []):
            tips.append({
                **tip,
                "seg_index"     : seg["seg_index"],
                "b_px"          : seg["b_px"],
                "slope_top"     : seg["slope_top"],
                "intercept_top" : seg["intercept_top"],
                "slope_bot"     : seg["slope_bot"],
                "intercept_bot" : seg["intercept_bot"],
            })
    return tips


def compute_wear(
    recon_results : List[Dict],
    hw_result     : Dict,
    image_shape   : Tuple[int, int] = None,
    vert_mask     : np.ndarray = None,   # used as full_mask for clipping
) -> Dict:
    empty = {"pairs": [], "d_mean_px": 0, "b_mean_px": 0,
             "wear_pct_left": 0, "wear_pct_right": 0}

    if not recon_results or not hw_result.get("segments"):
        return empty

    res       = recon_results[0]
    cp_t      = res["cp_top"]
    cp_b      = res["cp_bot"]
    side_arcs = res.get("side_arcs", [])
    if not side_arcs:
        return empty

    if image_shape is None:
        image_shape = (1200, 1600)
    h, w = image_shape

    x_center  = cp_t["x_apex"]
    all_tips  = _collect_all_tips(hw_result)
    pairs     = []
    used_tips = set()

    for sa in side_arcs:
        side = "left" if sa["pt7"][0] < x_center else "right"

        # Match tip: closest circle center x to pt7 x (outer arc midpoint)
        ref_x = float(sa["pt7"][0])
        best_tip, best_dist, best_idx = None, float("inf"), -1
        for i, tip in enumerate(all_tips):
            if i in used_tips:
                continue
            d = abs(tip["center"][0] - ref_x)
            if d < best_dist:
                best_dist, best_tip, best_idx = d, tip, i
        if best_tip is None:
            continue
        used_tips.add(best_idx)

        wire_seg = {
            "slope_top"     : best_tip["slope_top"],
            "intercept_top" : best_tip["intercept_top"],
            "slope_bot"     : best_tip["slope_bot"],
            "intercept_bot" : best_tip["intercept_bot"],
        }

        cs_mask  = _build_crescent_mask(sa, wire_seg, (h, w), full_mask=vert_mask)
        cs_area  = int(cs_mask.sum() // 255)

        if cs_area < 50:
            logger.warning(f"  {side}: crescent too small ({cs_area} px²)")
            continue

        circ_mask   = _build_circle_mask(best_tip, (h, w))
        circ_area   = int(circ_mask.sum() // 255)
        interf_mask = cv2.bitwise_and(circ_mask, cs_mask)
        interf_area = int(interf_mask.sum() // 255)
        wear_pct    = interf_area / cs_area * 100.0 if cs_area > 0 else 0.0

        logger.info(
            f"  Wear {side}: crescent={cs_area}px²  "
            f"circ={circ_area}px²  interf={interf_area}px²  wear={wear_pct:.1f}%"
        )

        pairs.append({
            "side"        : side,
            "cs_area"     : cs_area,
            "circ_area"   : circ_area,
            "interf_area" : interf_area,
            "wear_pct"    : wear_pct,
            "tip"         : best_tip,
            "cs_mask"     : cs_mask,
            "circ_mask"   : circ_mask,
            "interf_mask" : interf_mask,
        })

    return {
        "pairs"          : pairs,
        "d_mean_px"      : res.get("d_mean_px", 0),
        "b_mean_px"      : hw_result.get("b_mean_px", 0),
        "wear_pct_left"  : next((p["wear_pct"] for p in pairs if p["side"] == "left"),  0),
        "wear_pct_right" : next((p["wear_pct"] for p in pairs if p["side"] == "right"), 0),
    }


# ══════════════════════════════════════════════════════════════════
# D.  Drawing helpers
# ══════════════════════════════════════════════════════════════════

RED     = (  0,  50, 255);  BLUE    = (220,  90,  20)
YELLOW  = (  0, 220, 255);  GREEN   = ( 50, 200,  50)
ORANGE  = (  0, 165, 255);  WHITE   = (255, 255, 255)
MAGENTA = (255,  50, 255)
BLUE_FILL    = (255, 160,  40)
MAGENTA_FILL = (200,  50, 200)


def _pv(c, x):   return np.polyval(c, x)

def _draw_curve(v, c, x1, x2, col, th=3):
    if c is None: return
    h  = v.shape[0]
    xs = np.arange(x1, x2 + 1, 2, dtype=np.float64)
    ys = np.clip(_pv(c, xs).astype(int), 0, h - 1)
    p  = np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)
    if len(p) >= 2: cv2.polylines(v, [p], False, col, th, cv2.LINE_AA)

def _draw_dashed(v, pts, col, th=3, seg=14, gap=10):
    p = pts.astype(np.int32).reshape(-1, 1, 2); n = len(p); i = 0; on = True
    while i < n:
        e = min(i + (seg if on else gap), n)
        if on and e - i >= 2: cv2.polylines(v, [p[i:e]], False, col, th, cv2.LINE_AA)
        i = e; on = not on

def _dot(v, pt, col, r=8, lbl=None):
    if pt is None: return
    cx, cy = int(pt[0]), int(pt[1])
    cv2.circle(v, (cx, cy), r, col,  -1, cv2.LINE_AA)
    cv2.circle(v, (cx, cy), r, WHITE,  1, cv2.LINE_AA)
    if lbl:
        cv2.putText(v, lbl, (cx + r + 2, cy - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 220, 30), 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════
# E.  Main drawing function
# ══════════════════════════════════════════════════════════════════

def draw_full_recon(
    image        : np.ndarray,
    recon_results: List[Dict],
    hw_result    : Dict,
    wear_result  : Dict,
    vert_mask    : np.ndarray = None,
) -> np.ndarray:
    vis = image.copy()
    if not recon_results: return vis

    res  = recon_results[0]
    cp_t = res["cp_top"]
    cp_b = res["cp_bot"]

    # 1. Area overlays
    for p in wear_result.get("pairs", []):
        cs = p.get("cs_mask")
        if cs is not None:
            ov = np.zeros_like(vis); ov[cs > 0] = BLUE_FILL
            vis = cv2.addWeighted(vis, 1.0, ov, 0.40, 0)
        im = p.get("interf_mask")
        if im is not None:
            ov = np.zeros_like(vis); ov[im > 0] = MAGENTA_FILL
            vis = cv2.addWeighted(vis, 1.0, ov, 0.55, 0)

    # 2. Outer arcs (red)
    _draw_curve(vis, res["coef_outer_top"], res["top_x1"], res["top_x2"], RED, 3)
    _draw_curve(vis, res["coef_outer_bot"], res["x1"],     res["x2"],     RED, 3)

    # 3. Inner arcs (blue)
    _draw_curve(vis, res["coef_inner_top"], cp_t["x2_left"], cp_t["x2_right"], BLUE, 3)
    _draw_curve(vis, res["coef_inner_bot"], cp_b["x2_left"], cp_b["x2_right"], BLUE, 3)

    # 4. Side arcs
    for sa in res.get("side_arcs", []):
        _draw_dashed(vis, sa["outer_pts"], ORANGE, 3)
        _draw_dashed(vis, sa["inner_pts"], YELLOW, 2, seg=10, gap=8)
        _dot(vis, sa["pt7"], ORANGE, 10, "7")
        _dot(vis, sa["pt8"], ORANGE, 10, "8")
        p7, p8 = sa["pt7"], sa["pt8"]
        cv2.line(vis, (int(p7[0]), int(p7[1])), (int(p8[0]), int(p8[1])),
                 YELLOW, 2, cv2.LINE_AA)
        ds = sa["d_side_px"]; mx, my = (p7[0]+p8[0])/2, (p7[1]+p8[1])/2
        cv2.putText(vis, f"d={ds:.0f}", (int(mx)+8, int(my)+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)

    # 5. Key control points
    for cp in (cp_t, cp_b):
        _dot(vis, cp.get("pt3"), RED,  8, "3")
        _dot(vis, cp.get("pt4"), BLUE, 8, "4")
        for s in ("left", "right"):
            _dot(vis, cp.get(f"pt1_{s}"), RED,  8, "1")
            _dot(vis, cp.get(f"pt2_{s}"), BLUE, 8, "2")

    # 6. d at apex
    for co, ci, cp in [
        (res["coef_outer_top"], res["coef_inner_top"], cp_t),
        (res["coef_outer_bot"], res["coef_inner_bot"], cp_b),
    ]:
        if co is not None and ci is not None:
            xap = cp["x_apex"]
            yo = int(_pv(co, xap)); yi = int(_pv(ci, xap)); d = abs(yi - yo)
            cv2.line(vis, (xap, yo), (xap, yi), YELLOW, 3, cv2.LINE_AA)
            cv2.putText(vis, f"d={d:.0f}px", (xap+10, (yo+yi)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2, cv2.LINE_AA)

    # 7. Horizontal wire
    if hw_result:
        for seg in hw_result.get("segments", []):
            cv2.line(vis, seg["top_line"][0], seg["top_line"][1], RED, 2, cv2.LINE_AA)
            cv2.line(vis, seg["bot_line"][0], seg["bot_line"][1], RED, 2, cv2.LINE_AA)
            xm = (seg["x1"] + seg["x2"]) // 2
            yt = int(seg["slope_top"]*xm + seg["intercept_top"])
            yb = int(seg["slope_bot"]*xm + seg["intercept_bot"])
            cv2.line(vis, (xm, yt), (xm, yb), YELLOW, 2, cv2.LINE_AA)
            cv2.putText(vis, f"b={seg['b_px']:.0f}px", (xm+8, (yt+yb)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
            for tip in seg["tips"]:
                cx, cy = int(tip["center"][0]), int(tip["center"][1])
                r = int(tip["radius"])
                cv2.circle(vis, (cx, cy), r, GREEN, 2, cv2.LINE_AA)
                cv2.line(vis, (cx, cy-r), (cx, cy+r), YELLOW, 2, cv2.LINE_AA)

    # 8. Wear labels
    for p in wear_result.get("pairs", []):
        tip = p["tip"]; cx, cy = int(tip["center"][0]), int(tip["center"][1])
        r = int(tip["radius"]); wp = p["wear_pct"]; side = p["side"]
        lx = (cx - r - 170) if side == "left" else (cx + r + 10)
        cv2.putText(vis, f"wear={wp:.1f}%", (lx, cy-r-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(vis, f"({p['interf_area']}/{p['cs_area']}px\u00B2)",
                    (lx, cy-r+14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MAGENTA, 1, cv2.LINE_AA)

    # 9. Summary
    dm = wear_result.get("d_mean_px", 0); bm = wear_result.get("b_mean_px", 0)
    wl = wear_result.get("wear_pct_left", 0); wr = wear_result.get("wear_pct_right", 0)
    py = max(20, res.get("top_y1", 40) - 50)
    for i, line in enumerate([f"d={dm:.0f}px  b={bm:.0f}px",
                               f"wear L={wl:.1f}%  R={wr:.1f}%"]):
        y = py + i*24; cv2.rectangle(vis, (8, y-16), (330, y+6), (20,20,20), -1)
        cv2.putText(vis, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)

    return vis