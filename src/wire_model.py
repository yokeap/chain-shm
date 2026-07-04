"""
wire_model.py  (v2)
===================
Horizontal-wire model following the annotation in
``resource/horizontal_mask_annotated.png``:

  point 1     — wire tip (apex of each rounded end-cap)
  points 2    — the FOUR tangent corners where the straight top/bottom edges
                meet the rounded caps (top-left, bot-left, top-right, bot-right)
  red lines   — top / bottom straight edges (RANSAC fit on the central span)
  b (blue)    — wire thickness = perpendicular gap between the two edge lines
  circle B    — reconstructed rounded end, Ø = b, inscribed tangent at the tip.
                Its area is **area B**, intersected downstream with the vertical
                link's crescent **area A** to give wear = area(A∩B)/area(A).

The low-level primitives (edge scan, RANSAC line fit, tangent circle) are
reused from ``horizontal_chain``; this module adds

  * explicit MAIN-component selection (ignore frame-cut partial links),
  * the pt2 tangent-corner detector (persistence / hysteresis based), and
  * a draw function that mirrors the reference annotation.

Usage
-----
    from wire_model import model_wire, draw_wire
    wire = model_wire(wire_mask)
    wire["b_px"]        # wire thickness
    wire["circles"]     # list of circle-B dicts (one per detected tip) == area B
    wire["corners"]     # list of pt2 tangent corners
"""

from __future__ import annotations
import logging, math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from horizontal_chain import (
    _extract_wire_segments,
    _scan_top_bot,
    _fit_line_ransac,
    _tangent_circle_at_tip,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# A.  Main-component selection
# ══════════════════════════════════════════════════════════════════

def _select_main_segment(segments: List[Dict], img_w: int,
                         edge_margin: int = 15) -> Optional[Dict]:
    """
    Choose the horizontal link under inspection: the largest-area segment whose
    LEFT edge does not touch the frame (``x1 > edge_margin``).  A partial link
    cut off by the frame starts at x≈0 and is rejected — its cap is missing so
    it cannot yield a valid circle B.
    """
    candidates = [s for s in segments if s["x1"] > edge_margin]
    if not candidates:
        candidates = segments          # fall back to whatever we have
    if not candidates:
        return None
    return max(candidates, key=lambda s: s["area"])


# ══════════════════════════════════════════════════════════════════
# B.  pt2 tangent-corner detector  (persistence / hysteresis)
# ══════════════════════════════════════════════════════════════════

def _tangent_corner(xs: np.ndarray, edge: np.ndarray,
                    slope: float, intercept: float,
                    center_idx: int, direction: int,
                    thr: float = 6.0, persist: int = 5) -> Tuple[int, int]:
    """
    Walk OUTWARD from ``center_idx`` (``direction`` = -1 toward xs[0], +1 toward
    xs[-1]) along one edge and find the tangent corner — the column at which the
    measured edge departs the fitted straight line and the cap begins to curve.

    A simple "first column beyond ``thr``" break is fragile: a single noisy jag
    near the middle terminates the scan early.  Instead we require ``persist``
    CONSECUTIVE out-of-tolerance columns to confirm we have entered the cap, and
    report the FIRST column of that confirmed run as the corner.  The corner's
    y is taken ON the fitted line (the tangent point lies on the straight edge
    by definition), which suppresses per-column mask noise.
    """
    line = slope * xs + intercept
    dev = np.abs(edge - line)
    n = len(xs)

    run = 0
    run_start = center_idx
    corner_idx = center_idx
    idx = center_idx
    while 0 <= idx < n:
        if dev[idx] > thr:
            if run == 0:
                run_start = idx
            run += 1
            if run >= persist:
                corner_idx = run_start
                break
        else:
            run = 0
            corner_idx = idx          # last in-tolerance column so far
        idx += direction

    x = int(xs[corner_idx])
    y = int(round(slope * x + intercept))
    return x, y


# ══════════════════════════════════════════════════════════════════
# C.  Public API
# ══════════════════════════════════════════════════════════════════

def _model_one_segment(seg: Dict, w: int, px_per_mm: float,
                       edge_margin: int) -> Optional[Dict]:
    """Model one wire segment → edges, thickness b, pt1 tips, pt2 corners, circle(s) B."""
    xs, tops, bots = _scan_top_bot(seg["comp"])
    n = len(xs)
    if n < 20:
        return None

    # straight edges from the central 60 % (exclude the caps)
    i0, i1 = int(n * 0.20), int(n * 0.80)
    s_top, i_top = _fit_line_ransac(xs[i0:i1], tops[i0:i1])
    s_bot, i_bot = _fit_line_ransac(xs[i0:i1], bots[i0:i1])

    # thickness b = perpendicular gap of the two edges
    x_mid = float(xs[n // 2])
    dy = abs((s_bot * x_mid + i_bot) - (s_top * x_mid + i_top))
    b_px = dy * math.cos(math.atan((s_top + s_bot) / 2.0))

    # pt1 tips (extreme columns), reject frame-cut ends
    tips: List[Dict] = []
    if int(xs[0]) > edge_margin:
        tips.append({"side": "left",
                     "tip": (int(xs[0]), int(round((tops[0] + bots[0]) / 2)))})
    if int(xs[-1]) < w - edge_margin:
        tips.append({"side": "right",
                     "tip": (int(xs[-1]), int(round((tops[-1] + bots[-1]) / 2)))})

    # pt2 tangent corners (scan outward from centre on the relevant side)
    c_idx = n // 2
    corners: List[Dict] = []
    for t in tips:
        d = -1 if t["side"] == "left" else +1
        corners.append({"side": t["side"], "edge": "top",
                        "pt": _tangent_corner(xs, tops, s_top, i_top, c_idx, d)})
        corners.append({"side": t["side"], "edge": "bot",
                        "pt": _tangent_corner(xs, bots, s_bot, i_bot, c_idx, d)})

    # circle B per tip (Ø = b, inscribed tangent at the tip)
    r = b_px / 2.0
    area_B_px = math.pi * r * r
    circles: List[Dict] = []
    for t in tips:
        tip_d = {"x_tip": t["tip"][0], "y_tip": t["tip"][1], "side": t["side"]}
        circ = _tangent_circle_at_tip(tip_d, b_px, s_top, i_top, s_bot, i_bot)
        circles.append({
            "side": t["side"], "tip": t["tip"],
            "center": circ["center"], "radius": circ["radius"],
            "b_px": b_px, "area_B_px": area_B_px,
            "area_B_mm2": area_B_px / (px_per_mm ** 2),
            "slope_top": s_top, "intercept_top": i_top,
            "slope_bot": s_bot, "intercept_bot": i_bot,
        })

    top_line = ((seg["x1"], int(round(s_top * seg["x1"] + i_top))),
                (seg["x2"], int(round(s_top * seg["x2"] + i_top))))
    bot_line = ((seg["x1"], int(round(s_bot * seg["x1"] + i_bot))),
                (seg["x2"], int(round(s_bot * seg["x2"] + i_bot))))

    return {
        "b_px": b_px, "slopes": (s_top, i_top, s_bot, i_bot),
        "tips": tips, "corners": corners, "circles": circles,
        "top_line": top_line, "bot_line": bot_line,
        "x1": seg["x1"], "x2": seg["x2"], "y1": seg["y1"], "y2": seg["y2"],
        "area": seg["area"], "n_tips": len(tips),
    }


def model_wire(wire_mask: np.ndarray, px_per_mm: float = 1.0) -> Dict:
    """
    Model **every** horizontal wire segment in the mask, so *all* caps (area B),
    hence all interlocks, are captured — not just the largest wire link.

    Returns a dict with the aggregate ``tips`` / ``corners`` / ``circles`` (one
    circle B per non-frame-cut cap across all segments), plus the main (largest)
    segment's ``b_px`` / ``slopes`` / edge lines for the thickness annotation, and
    ``full`` (main wire link has both caps).  ``segments`` holds each per-segment
    model.
    """
    _, bw = cv2.threshold(wire_mask, 127, 255, cv2.THRESH_BINARY)
    segments = _extract_wire_segments(bw)
    w = wire_mask.shape[1]
    edge_margin = max(15, int(w * 0.02))

    mods = [m for s in segments
            if (m := _model_one_segment(s, w, px_per_mm, edge_margin)) is not None]
    if not mods:
        logger.warning("model_wire: no valid horizontal segment found")
        return {"b_px": 0.0, "b_mm": 0.0, "tips": [], "corners": [],
                "circles": [], "full": False, "segments": []}

    tips    = [t for m in mods for t in m["tips"]]
    corners = [c for m in mods for c in m["corners"]]
    circles = [c for m in mods for c in m["circles"]]

    primary = max(mods, key=lambda m: m["area"])   # main link → b, edge lines
    b_px = primary["b_px"]
    wire_full = (primary["n_tips"] == 2)

    logger.info(f"model_wire: {len(mods)} segment(s)  b={b_px:.1f}px  "
                f"caps(B)={len(circles)}  full={wire_full}")

    return {
        "b_px": float(b_px), "b_mm": float(b_px / px_per_mm),
        "top_line": primary["top_line"], "bot_line": primary["bot_line"],
        "slopes": primary["slopes"],
        "tips": tips, "corners": corners, "circles": circles,
        "full": wire_full,
        "x1": primary["x1"], "x2": primary["x2"],
        "y1": primary["y1"], "y2": primary["y2"],
        "segments": mods,
    }


# ══════════════════════════════════════════════════════════════════
# D.  Visualization — mirrors resource/horizontal_mask_annotated.png
# ══════════════════════════════════════════════════════════════════

def draw_wire(image: np.ndarray, wire: Dict,
              wire_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Overlay: red edge lines, blue b line, yellow pt1/pt2 dots, green circle B."""
    vis = image.copy()
    RED    = (0,   50, 255)
    BLUE   = (255, 120,  0)
    YELLOW = (0,  220, 255)
    GREEN  = (60, 200,  60)
    WHITE  = (255, 255, 255)
    BLACK  = (0, 0, 0)

    if wire_mask is not None:
        ov = np.zeros_like(vis)
        ov[wire_mask > 127] = (70, 55, 40)
        vis = cv2.addWeighted(vis, 0.75, ov, 0.25, 0)

    if not wire.get("tips"):
        return vis

    # Straight edges (red) + thickness b (blue) for every wire segment
    for m in wire.get("segments", [{"top_line": wire["top_line"],
                                    "bot_line": wire["bot_line"],
                                    "slopes": wire["slopes"],
                                    "x1": wire["x1"], "x2": wire["x2"]}]):
        cv2.line(vis, m["top_line"][0], m["top_line"][1], RED, 2, cv2.LINE_AA)
        cv2.line(vis, m["bot_line"][0], m["bot_line"][1], RED, 2, cv2.LINE_AA)
        x_mid = (m["x1"] + m["x2"]) // 2
        s_top, i_top, s_bot, i_bot = m["slopes"]
        cv2.line(vis, (x_mid, int(round(s_top * x_mid + i_top))),
                 (x_mid, int(round(s_bot * x_mid + i_bot))), BLUE, 3, cv2.LINE_AA)

    # Circle B (green) + area label
    for c in wire["circles"]:
        cx, cy = int(round(c["center"][0])), int(round(c["center"][1]))
        r = int(round(c["radius"]))
        cv2.circle(vis, (cx, cy), r, GREEN, 2, cv2.LINE_AA)
        cv2.putText(vis, "B", (cx - 14, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3, cv2.LINE_AA)
        # area B, printed just below the circle
        area_txt = f"area B={c['area_B_px']:.0f}px2"
        cv2.putText(vis, area_txt, (cx - r, cy + r + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)

    # pt2 tangent corners (yellow "2")
    for c in wire["corners"]:
        x, y = c["pt"]
        cv2.circle(vis, (x, y), 12, YELLOW, -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 12, BLACK, 1, cv2.LINE_AA)
        cv2.putText(vis, "2", (x - 6, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, cv2.LINE_AA)

    # pt1 tips (yellow "1")
    for t in wire["tips"]:
        x, y = t["tip"]
        cv2.circle(vis, (x, y), 12, YELLOW, -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 12, BLACK, 1, cv2.LINE_AA)
        cv2.putText(vis, "1", (x - 5, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, cv2.LINE_AA)

    return vis
