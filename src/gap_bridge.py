"""
gap_bridge.py
=============
Reconstruct the vertical hidden segments between top blob and bot blob
that are occluded by the horizontal wire.

Two bridges per side (left / right):
  outer bridge : pt1_top  →  pt1_bot   (tip corner, nearly vertical)
  inner bridge : pt2_top  →  pt2_bot   (wire-contact boundary)

Strategy: Cubic Hermite spline  (B + C)
  - p0, p1  : known endpoint positions
  - m0, m1  : tangent vectors at each endpoint
               outer tip  → from local blob-edge finite-difference
               inner pt2  → from polynomial coef  (np.polyder)
  n_pts     : proportional to gap height  n = max(4, gap_height // STEP_PX)

Coordinate convention
---------------------
  All points are (x, y) image coordinates.
  Parametric form x(t), y(t)  handles near-vertical curves without singularity.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np

STEP_PX: int = 4   # one sample point per N pixels of gap height


# ══════════════════════════════════════════════════════════════════
# Cubic Hermite (parametric)
# ══════════════════════════════════════════════════════════════════

def _hermite(p0, p1, m0, m1, n):
    """
    Parametric cubic Hermite.
      p0, p1 : (2,) endpoints
      m0, m1 : (2,) scaled tangent vectors  (chord-length units)
      n      : number of sample points
    Returns (n, 2) array.
    """
    t   = np.linspace(0.0, 1.0, n)
    h00 =  2*t**3 - 3*t**2 + 1
    h10 =    t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 =    t**3 -   t**2
    return (h00[:,None]*p0 + h10[:,None]*m0 +
            h01[:,None]*p1 + h11[:,None]*m1)


def _n_pts(p0, p1):
    return max(4, int(round(abs(float(p1[1]) - float(p0[1])) / STEP_PX)))


# ══════════════════════════════════════════════════════════════════
# Tangent helpers
# ══════════════════════════════════════════════════════════════════

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0.0, 1.0])


def _tangent_from_coef(coef: np.ndarray, x: float, going_down: bool) -> np.ndarray:
    """
    Tangent direction from polynomial coef (y = poly(x)).
    going_down=True  → leaving top blob downward
    going_down=False → entering bot blob (arriving from above)
    The x-component sign follows the local slope direction on the arc.
    """
    dcoef = np.polyder(coef)
    dydx  = float(np.polyval(dcoef, x))
    # on the arc dy/dx gives the horizontal tangent; we want the
    # *vertical* departure direction at the gap boundary.
    # At pt2 the arc is mostly horizontal, so the tangent into the gap
    # is dominated by the vertical component.  We use (0,1) rotated by
    # a small horizontal offset proportional to dydx.
    sign  = +1.0 if going_down else -1.0
    vec   = np.array([dydx * 0.3, sign])   # small x-lean, strong y
    return _unit(vec)


def _tangent_outer_tip(cp_top_pt1, cp_bot_pt1,
                       cp_top_pt5, cp_bot_pt5,
                       side: str, going_down: bool) -> np.ndarray:
    """
    Tangent at outer tip (pt1).  Estimated from the direction
    pt5 → pt1 on each blob (the local arc heading toward the tip).

    side : 'left' or 'right'
    going_down : True for top-blob departure, False for bot-blob arrival
    """
    if going_down:
        # direction of arc arriving at the tip FROM the blob centre
        vec = np.array(cp_top_pt1, dtype=float) - np.array(cp_top_pt5, dtype=float)
    else:
        # direction of arc leaving the tip INTO the blob centre (reversed)
        vec = np.array(cp_bot_pt5, dtype=float) - np.array(cp_bot_pt1, dtype=float)

    # Force downward component to be positive (into the gap)
    u = _unit(vec)
    if going_down and u[1] < 0:
        u[1] = abs(u[1])
    if not going_down and u[1] > 0:
        u[1] = -abs(u[1])
    return _unit(u)


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def build_vertical_bridges(
    cp_top        : Dict,
    cp_bot        : Dict,
    coef_outer_top: Optional[np.ndarray] = None,
    coef_inner_top: Optional[np.ndarray] = None,
    coef_outer_bot: Optional[np.ndarray] = None,
    coef_inner_bot: Optional[np.ndarray] = None,
    step_px       : int = STEP_PX,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Build four vertical bridge polylines across the hidden gap:
      outer_left, outer_right  — pt1_top → pt1_bot  (red)
      inner_left, inner_right  — pt2_top → pt2_bot  (blue)

    Returns dict with each key → np.ndarray (n, 2) or None.
    """
    global STEP_PX
    STEP_PX = step_px

    out: Dict[str, Optional[np.ndarray]] = {
        "outer_left": None, "outer_right": None,
        "inner_left": None, "inner_right": None,
    }

    # ── helpers ───────────────────────────────────────────────────
    def _bridge_outer(side):
        k    = f"pt1_{side}"
        k5   = f"pt5_{side}"
        p0   = np.array(cp_top[k], dtype=float)
        p1   = np.array(cp_bot[k], dtype=float)
        m0_u = _tangent_outer_tip(cp_top[k], cp_bot[k],
                                   cp_top[k5], cp_bot[k5],
                                   side, going_down=True)
        m1_u = _tangent_outer_tip(cp_top[k], cp_bot[k],
                                   cp_top[k5], cp_bot[k5],
                                   side, going_down=False)
        chord = float(np.linalg.norm(p1 - p0))
        n     = _n_pts(p0, p1)
        return _hermite(p0, p1, m0_u * chord, m1_u * chord, n)

    def _bridge_inner(side, coef_top, coef_bot):
        k  = f"pt2_{side}"
        p0 = np.array(cp_top[k], dtype=float)
        p1 = np.array(cp_bot[k], dtype=float)

        m0_u = (_tangent_from_coef(coef_top, p0[0], going_down=True)
                if coef_top is not None
                else np.array([0.0, 1.0]))
        m1_u = (_tangent_from_coef(coef_bot, p1[0], going_down=False)
                if coef_bot is not None
                else np.array([0.0, -1.0]))

        chord = float(np.linalg.norm(p1 - p0))
        n     = _n_pts(p0, p1)
        return _hermite(p0, p1, m0_u * chord, m1_u * chord, n)

    # ── outer ─────────────────────────────────────────────────────
    for side, key in (("left", "outer_left"), ("right", "outer_right")):
        try:
            out[key] = _bridge_outer(side)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"outer_{side} bridge failed: {e}")

    # ── inner ─────────────────────────────────────────────────────
    for side, key in (("left", "inner_left"), ("right", "inner_right")):
        try:
            out[key] = _bridge_inner(side, coef_inner_top, coef_inner_bot)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"inner_{side} bridge failed: {e}")

    return out


# ══════════════════════════════════════════════════════════════════
# Draw helper
# ══════════════════════════════════════════════════════════════════

def draw_vertical_bridges(
    vis         : "np.ndarray",
    bridges     : Dict[str, Optional[np.ndarray]],
    color_outer  = (0,  50, 255),   # BGR red
    color_inner  = (220, 90,  20),  # BGR blue
    thickness   : int  = 2,
    dashed      : bool = True,
    dash_len    : int  = 10,
    gap_len     : int  = 5,
) -> None:
    """Draw bridge polylines on vis in-place."""
    import cv2

    def _draw(pts, color):
        if pts is None or len(pts) < 2:
            return
        pi = pts.astype(np.int32)
        if not dashed:
            cv2.polylines(vis, [pi.reshape(-1,1,2)],
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
