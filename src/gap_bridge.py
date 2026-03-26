"""
gap_bridge.py  v2
=================
Reconstruct the vertical hidden segments (occluded by horizontal wire)
between top blob and bot blob.

Two bridges per side (left / right):

  outer bridge (green dashed)
    control points: pt5_top → pt1_top → pt1_bot → pt5_bot

  inner bridge (red dashed)
    control points: pt6_top → pt2_top → pt2_bot → pt6_bot

Why Catmull-Rom?
  - Passes EXACTLY through all control points (unlike Bezier)
  - Tangent at each point auto-computed from neighbours
    → smooth join where bridge meets existing arcs
  - No manual tangent estimation needed

n_pts: proportional to gap height  max(8, gap_px // STEP_PX)
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np

STEP_PX: int = 3


# ══════════════════════════════════════════════════════════════════
# Catmull-Rom
# ══════════════════════════════════════════════════════════════════

def _catmull_rom_segment(p0, p1, p2, p3, n: int) -> np.ndarray:
    """One C-R segment from p1 → p2 (p0, p3 for tangents). Returns (n, 2)."""
    t  = np.linspace(0.0, 1.0, n, endpoint=False)
    t2, t3 = t*t, t*t*t
    b0 = -0.5*t3 + 1.0*t2 - 0.5*t
    b1 =  1.5*t3 - 2.5*t2 + 1.0
    b2 = -1.5*t3 + 2.0*t2 + 0.5*t
    b3 =  0.5*t3 - 0.5*t2
    return b0[:,None]*p0 + b1[:,None]*p1 + b2[:,None]*p2 + b3[:,None]*p3


def _catmull_rom_4pts(p0, p1, p2, p3, n_total: int) -> np.ndarray:
    """
    Catmull-Rom through 4 control points p0..p3.
    Renders all three segments (p0→p1, p1→p2, p2→p3).
    Phantom points are reflected from the ends to get natural tangents.
    """
    pts = [np.array(p, dtype=float) for p in (p0, p1, p2, p3)]
    ph0 = 2.0*pts[0] - pts[1]   # phantom before p0
    ph3 = 2.0*pts[3] - pts[2]   # phantom after  p3
    chain = [ph0] + pts + [ph3]  # 6 points total

    # Chord lengths for segments 1→2, 2→3, 3→4  (i.e. p0→p1, p1→p2, p2→p3)
    chords = [max(1.0, float(np.linalg.norm(chain[i+1] - chain[i])))
              for i in range(1, 4)]
    total = sum(chords)
    ns    = [max(2, int(round(n_total * c / total))) for c in chords]

    segs = []
    for i, n_seg in enumerate(ns):
        segs.append(_catmull_rom_segment(
            chain[i], chain[i+1], chain[i+2], chain[i+3], n_seg))
    segs.append(pts[3][None, :])   # close the last point
    return np.vstack(segs)


def _n_pts(pa, pb) -> int:
    return max(8, int(round(abs(float(pb[1]) - float(pa[1])) / STEP_PX)))


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def build_vertical_bridges(
    cp_top        : Dict,
    cp_bot        : Dict,
    coef_outer_top = None,   # kept for API compat, unused in v2
    coef_inner_top = None,
    coef_outer_bot = None,
    coef_inner_bot = None,
    step_px       : int = STEP_PX,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Build four vertical bridge polylines:

      outer_left  : pt5_left_top  → pt1_left_top  → pt1_left_bot  → pt5_left_bot
      outer_right : pt5_right_top → pt1_right_top → pt1_right_bot → pt5_right_bot
      inner_left  : pt6_left_top  → pt2_left_top  → pt2_left_bot  → pt6_left_bot
      inner_right : pt6_right_top → pt2_right_top → pt2_right_bot → pt6_right_bot

    Returns dict key → np.ndarray (n, 2)  or None on failure.
    """
    global STEP_PX
    STEP_PX = step_px

    out: Dict[str, Optional[np.ndarray]] = {
        "outer_left": None, "outer_right": None,
        "inner_left": None, "inner_right": None,
    }

    def _safe(key, fn):
        try:
            out[key] = fn()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"bridge {key} failed: {e}")

    # outer left:  pt5_left_top → pt1_left_top → pt1_left_bot → pt5_left_bot
    _safe("outer_left", lambda: _catmull_rom_4pts(
        cp_top["pt5_left"], cp_top["pt1_left"],
        cp_bot["pt1_left"], cp_bot["pt5_left"],
        _n_pts(cp_top["pt1_left"], cp_bot["pt1_left"]),
    ))

    # outer right: pt5_right_top → pt1_right_top → pt1_right_bot → pt5_right_bot
    _safe("outer_right", lambda: _catmull_rom_4pts(
        cp_top["pt5_right"], cp_top["pt1_right"],
        cp_bot["pt1_right"], cp_bot["pt5_right"],
        _n_pts(cp_top["pt1_right"], cp_bot["pt1_right"]),
    ))

    # inner left:  pt6_left_top → pt2_left_top → pt2_left_bot → pt6_left_bot
    _safe("inner_left", lambda: _catmull_rom_4pts(
        cp_top["pt6_left"], cp_top["pt2_left"],
        cp_bot["pt2_left"], cp_bot["pt6_left"],
        _n_pts(cp_top["pt2_left"], cp_bot["pt2_left"]),
    ))

    # inner right: pt6_right_top → pt2_right_top → pt2_right_bot → pt6_right_bot
    _safe("inner_right", lambda: _catmull_rom_4pts(
        cp_top["pt6_right"], cp_top["pt2_right"],
        cp_bot["pt2_right"], cp_bot["pt6_right"],
        _n_pts(cp_top["pt2_right"], cp_bot["pt2_right"]),
    ))

    return out


# ══════════════════════════════════════════════════════════════════
# Draw helper
# ══════════════════════════════════════════════════════════════════

def draw_vertical_bridges(
    vis         : "np.ndarray",
    bridges     : Dict[str, Optional[np.ndarray]],
    color_outer  = (50,  200,  50),   # BGR green
    color_inner  = (0,    50, 255),   # BGR red
    thickness   : int  = 2,
    dashed      : bool = True,
    dash_len    : int  = 10,
    gap_len     : int  = 5,
) -> None:
    """Draw all four bridge polylines on vis in-place."""
    import cv2

    def _draw(pts, color):
        if pts is None or len(pts) < 2:
            return
        pi = pts.astype(np.int32)
        if not dashed:
            cv2.polylines(vis, [pi.reshape(-1, 1, 2)],
                          False, color, thickness, cv2.LINE_AA)
            return
        draw_seg, count = True, 0
        for i in range(1, len(pi)):
            if draw_seg:
                cv2.line(vis, tuple(pi[i-1]), tuple(pi[i]),
                         color, thickness, cv2.LINE_AA)
            count += 1
            if count >= (dash_len if draw_seg else gap_len):
                draw_seg = not draw_seg
                count    = 0

    _draw(bridges.get("outer_left"),  color_outer)
    _draw(bridges.get("outer_right"), color_outer)
    _draw(bridges.get("inner_left"),  color_inner)
    _draw(bridges.get("inner_right"), color_inner)