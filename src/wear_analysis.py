"""
wear_analysis.py
================
Measure chain wear via the **gap** between:
  - Yellow b-line  (vertical diameter at circle center, from horizontal_chain)
  - Red ref line   (pt(2)_top ↔ pt(2)_bot, from reconstruction)

Sign convention — always  gap = x_b_line − x_ref :
  Left side  : gap < 0 → b-line shifted left  (outward) → wear
  Right side : gap > 0 → b-line shifted right (outward) → wear

wear_pct = |gap| / (b/2) × 100
  0 %  = no wear (b-line sits on ref line)
  100% = maximum  (b-line shifted by a full radius)

Usage
-----
    from wear_analysis import compute_wear, draw_full_recon
"""

from __future__ import annotations
import logging, math
from typing import Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# A.  Match tips → pt(2) refs (closest b-line to each ref)
# ══════════════════════════════════════════════════════════════════

def _collect_all_tips(hw_result: Dict) -> List[Dict]:
    tips = []
    for seg in hw_result.get("segments", []):
        for tip in seg.get("tips", []):
            tips.append({**tip, "seg_index": seg["seg_index"],
                         "b_px": seg["b_px"]})
    return tips


def _match_tips_to_refs(recon_result: Dict, hw_result: Dict) -> List[Dict]:
    cp_t = recon_result["cp_top"]
    cp_b = recon_result["cp_bot"]
    all_tips = _collect_all_tips(hw_result)
    if not all_tips:
        return []

    sides = {
        "left":  {"ref_top": cp_t["pt2_left"],  "ref_bot": cp_b["pt2_left"]},
        "right": {"ref_top": cp_t["pt2_right"], "ref_bot": cp_b["pt2_right"]},
    }

    matches, used = [], set()
    for side, refs in sides.items():
        ref_top, ref_bot = refs["ref_top"], refs["ref_bot"]
        x_ref = (ref_top[0] + ref_bot[0]) / 2.0
        best_i, best_d = -1, float("inf")
        for i, tip in enumerate(all_tips):
            if i in used:
                continue
            d = abs(tip["center"][0] - x_ref)
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            used.add(best_i)
            t = all_tips[best_i]
            matches.append({
                "side": side, "x_b_line": t["center"][0],
                "y_b_center": t["center"][1], "b_radius": t["radius"],
                "x_ref": x_ref, "ref_top": ref_top, "ref_bot": ref_bot,
                "tip": t, "b_px": t["b_px"],
            })
    return matches


# ══════════════════════════════════════════════════════════════════
# B.  Compute wear
# ══════════════════════════════════════════════════════════════════

def compute_wear(recon_results: List[Dict], hw_result: Dict) -> Dict:
    if not recon_results or not hw_result.get("segments"):
        return {"pairs": [], "d_mean_px": 0, "b_mean_px": 0,
                "gap_left": 0, "gap_right": 0,
                "wear_pct_left": 0, "wear_pct_right": 0}

    recon = recon_results[0]
    matches = _match_tips_to_refs(recon, hw_result)

    pairs = []
    for m in matches:
        x_b, x_ref = m["x_b_line"], m["x_ref"]

        # gap = x_b − x_ref  (always this formula for both sides)
        gap = x_b - x_ref

        b_half = m["b_px"] / 2.0
        wear_pct = abs(gap) / b_half * 100.0 if b_half > 0 else 0.0

        pairs.append({**m, "gap_px": gap, "wear_pct": wear_pct})

        logger.info(
            f"  Wear {m['side']}: x_b={x_b:.0f}  x_ref={x_ref:.0f}  "
            f"gap={gap:.1f}px  wear={wear_pct:.1f}%"
        )

    d_mean = recon.get("d_mean_px", 0)
    b_mean = hw_result.get("b_mean_px", 0)

    return {
        "pairs"     : pairs,
        "d_mean_px" : d_mean,
        "b_mean_px" : b_mean,
        "gap_left"  : next((p["gap_px"]   for p in pairs if p["side"]=="left"),  0),
        "gap_right" : next((p["gap_px"]   for p in pairs if p["side"]=="right"), 0),
        "wear_pct_left" : next((p["wear_pct"] for p in pairs if p["side"]=="left"),  0),
        "wear_pct_right": next((p["wear_pct"] for p in pairs if p["side"]=="right"), 0),
    }


# ══════════════════════════════════════════════════════════════════
# C.  Clean full-reconstruction drawing
# ══════════════════════════════════════════════════════════════════

RED    = (0,   50, 255)
BLUE   = (220,  90,  20)
YELLOW = (0,  220, 255)
GREEN  = (50, 200,  50)
ORANGE = (0,  165, 255)
WHITE  = (255, 255, 255)
MAGENTA = (255, 50, 255)


def _poly_eval(coef, x):
    return np.polyval(coef, x)


def _draw_curve_clipped(vis, coef, x1, x2, color, thickness=3):
    """Draw polynomial curve clipped to [x1, x2]."""
    if coef is None:
        return
    h = vis.shape[0]
    xs = np.arange(x1, x2 + 1, 2, dtype=np.float64)
    ys = np.clip(_poly_eval(coef, xs).astype(int), 0, h - 1)
    pts = np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(vis, [pts], False, color, thickness, cv2.LINE_AA)


def _draw_dashed_polyline(vis, pts, color, thickness=3, seg=14, gap_len=10):
    pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
    n, i, on = len(pts_i), 0, True
    while i < n:
        end = min(i + (seg if on else gap_len), n)
        if on and end - i >= 2:
            cv2.polylines(vis, [pts_i[i:end]], False, color, thickness, cv2.LINE_AA)
        i, on = end, not on


def _dot(vis, pt, color, r=8, label=None, label_color=(30,220,30)):
    if pt is None:
        return
    cx, cy = int(pt[0]), int(pt[1])
    cv2.circle(vis, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), r, WHITE, 1, cv2.LINE_AA)
    if label:
        cv2.putText(vis, label, (cx + r + 2, cy - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2, cv2.LINE_AA)


def draw_full_recon(
    image: np.ndarray,
    recon_results: List[Dict],
    hw_result: Dict,
    wear_result: Dict,
) -> np.ndarray:
    """
    Clean full reconstruction overlay:
      - Outer arc (red), inner arc clipped at pt(2) (blue)
      - Side arcs (orange/yellow dashed)
      - d measurement at apex
      - Horizontal wire edges + tangent circles
      - Wear gap arrows + labels
      - Summary panel
    """
    vis = image.copy()

    if not recon_results:
        return vis

    res = recon_results[0]
    cp_t, cp_b = res["cp_top"], res["cp_bot"]

    # ── Outer arcs (red, full range) ──
    _draw_curve_clipped(vis, res["coef_outer_top"],
                        res["top_x1"], res["top_x2"], RED, 3)
    _draw_curve_clipped(vis, res["coef_outer_bot"],
                        res["x1"], res["x2"], RED, 3)

    # ── Inner arcs (blue, CLIPPED at pt(2) boundaries) ──
    x2l_t, x2r_t = cp_t["x2_left"], cp_t["x2_right"]
    x2l_b, x2r_b = cp_b["x2_left"], cp_b["x2_right"]
    _draw_curve_clipped(vis, res["coef_inner_top"], x2l_t, x2r_t, BLUE, 3)
    _draw_curve_clipped(vis, res["coef_inner_bot"], x2l_b, x2r_b, BLUE, 3)

    # ── Key control points only ──
    for cp in (cp_t, cp_b):
        for k, lbl in (("pt3","3"),):
            _dot(vis, cp.get(k), RED, 8, lbl)
        for k, lbl in (("pt2_left","2"), ("pt4","4"), ("pt2_right","2")):
            _dot(vis, cp.get(k), BLUE, 8, lbl)

    # ── d measurement at apex (yellow vertical line) ──
    for label, coef_o, coef_i, cp in [
        ("top", res["coef_outer_top"], res["coef_inner_top"], cp_t),
        ("bot", res["coef_outer_bot"], res["coef_inner_bot"], cp_b),
    ]:
        if coef_o is not None and coef_i is not None:
            xap = cp["x_apex"]
            y_o = int(_poly_eval(coef_o, xap))
            y_i = int(_poly_eval(coef_i, xap))
            d_px = abs(y_i - y_o)
            cv2.line(vis, (xap, y_o), (xap, y_i), YELLOW, 3, cv2.LINE_AA)
            cv2.putText(vis, f"d={d_px:.0f}px",
                        (xap + 10, (y_o + y_i) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2, cv2.LINE_AA)

    # ── Side arcs (occluded zone) ──
    for sa in res.get("side_arcs", []):
        _draw_dashed_polyline(vis, sa["outer_pts"], ORANGE, 3)
        _draw_dashed_polyline(vis, sa["inner_pts"], YELLOW, 2, seg=10, gap_len=8)
        _dot(vis, sa["pt7"], ORANGE, 10, "7")
        _dot(vis, sa["pt8"], ORANGE, 10, "8")
        p7, p8 = sa["pt7"], sa["pt8"]
        cv2.line(vis, (int(p7[0]), int(p7[1])), (int(p8[0]), int(p8[1])),
                 YELLOW, 2, cv2.LINE_AA)
        d_s = sa["d_side_px"]
        mx, my = (p7[0]+p8[0])/2, (p7[1]+p8[1])/2
        cv2.putText(vis, f"d={d_s:.0f}", (int(mx)+8, int(my)+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)

    # ── Horizontal wire: edges + circles ──
    if hw_result:
        for seg in hw_result.get("segments", []):
            cv2.line(vis, seg["top_line"][0], seg["top_line"][1], RED, 2, cv2.LINE_AA)
            cv2.line(vis, seg["bot_line"][0], seg["bot_line"][1], RED, 2, cv2.LINE_AA)
            for tip in seg["tips"]:
                cx, cy = int(tip["center"][0]), int(tip["center"][1])
                r = int(tip["radius"])
                cv2.circle(vis, (cx, cy), r, GREEN, 2, cv2.LINE_AA)
                cv2.line(vis, (cx, cy-r), (cx, cy+r), YELLOW, 2, cv2.LINE_AA)

    # ── Wear gap arrows + labels ──
    for pair in wear_result.get("pairs", []):
        ref_top, ref_bot = pair["ref_top"], pair["ref_bot"]
        x_b = pair["x_b_line"]
        gap = pair["gap_px"]
        wear_pct = pair["wear_pct"]
        side = pair["side"]

        # Reference line (red, extended)
        x_rt, y_rt = int(ref_top[0]), int(ref_top[1])
        x_rb, y_rb = int(ref_bot[0]), int(ref_bot[1])
        dx, dy = x_rb - x_rt, y_rb - y_rt
        ln = math.sqrt(dx*dx + dy*dy) + 1e-6
        ux, uy = dx/ln, dy/ln
        ext = 35
        cv2.line(vis,
                 (int(x_rt - ux*ext), int(y_rt - uy*ext)),
                 (int(x_rb + ux*ext), int(y_rb + uy*ext)),
                 RED, 2, cv2.LINE_AA)

        # Gap arrow (magenta)
        x_ref_mid = (x_rt + x_rb) // 2
        y_mid = int(pair["y_b_center"])

        if abs(gap) > 2:
            cv2.arrowedLine(vis, (x_ref_mid, y_mid), (int(x_b), y_mid),
                            MAGENTA, 3, cv2.LINE_AA, tipLength=0.2)

        # Label: position above the arrow, offset outward for readability
        if side == "left":
            lx = min(x_ref_mid, int(x_b)) - 10
            anchor = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis, f"gap={gap:.0f}px", (lx - 130, y_mid - 8),
                        anchor, 0.6, MAGENTA, 2, cv2.LINE_AA)
            cv2.putText(vis, f"wear={wear_pct:.1f}%", (lx - 130, y_mid + 16),
                        anchor, 0.6, MAGENTA, 2, cv2.LINE_AA)
        else:
            lx = max(x_ref_mid, int(x_b)) + 10
            cv2.putText(vis, f"gap={gap:.0f}px", (lx, y_mid - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MAGENTA, 2, cv2.LINE_AA)
            cv2.putText(vis, f"wear={wear_pct:.1f}%", (lx, y_mid + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MAGENTA, 2, cv2.LINE_AA)

    # ── Summary panel (top-left) ──
    d_mean = wear_result.get("d_mean_px", 0)
    b_mean = wear_result.get("b_mean_px", 0)
    wl = wear_result.get("wear_pct_left", 0)
    wr = wear_result.get("wear_pct_right", 0)

    panel_lines = [
        f"d={d_mean:.0f}px   b={b_mean:.0f}px",
        f"wear L={wl:.1f}%   R={wr:.1f}%",
    ]
    py = max(20, res.get("top_y1", 40) - 40)
    for i, line in enumerate(panel_lines):
        y = py + i * 24
        cv2.rectangle(vis, (8, y - 16), (340, y + 6), (20,20,20), -1)
        cv2.putText(vis, line, (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)

    return vis