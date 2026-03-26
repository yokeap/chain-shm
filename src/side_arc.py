"""
side_arc.py
===========
Reconstruct the occluded side arcs of a vertical chain link
(the portions hidden behind the horizontal wire).

Control-point strategy
----------------------
Outer side arc (e.g. left side):
    Interpolate through: (5)L_top → (1)L_top → (1)L_bot → (5)L_bot
    (7) = midpoint along the arc length of this curve

Inner side arc:
    (8) = (7) offset inward by d_mean pixels
    Interpolate through: (6)L_top → (2)L_top → (8) → (2)L_bot → (6)L_bot

Right side is the mirror of left.

Usage
-----
    from side_arc import compute_side_arcs

    sa = compute_side_arcs(cp_top, cp_bot, d_mean, side="left")
    # sa["pt7"]       → (x, y)  outer midpoint
    # sa["pt8"]       → (x, y)  inner midpoint (= pt7 + d inward)
    # sa["d_side_px"] → float   distance between pt7 and pt8
    # sa["outer_pts"] → (N, 2)  sampled outer curve
    # sa["inner_pts"] → (N, 2)  sampled inner curve
"""

from __future__ import annotations
import math
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


# ──────────────────────────────────────────────────────────────────
# Parametric cubic-spline helpers
# ──────────────────────────────────────────────────────────────────

def _arc_length_cumulative(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc length along a polyline of (x, y) points."""
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _interpolate_parametric(ctrl_pts: list, n_samples: int = 200) -> np.ndarray:
    """
    Fit a parametric cubic spline through control points
    (parameter = cumulative chord length) and sample it.

    Parameters
    ----------
    ctrl_pts  : list of (x, y) tuples — at least 2 points.
    n_samples : number of output samples.

    Returns
    -------
    (n_samples, 2) ndarray of interpolated points.
    """
    pts = np.array(ctrl_pts, dtype=float)
    if len(pts) < 2:
        return pts

    t = _arc_length_cumulative(pts)
    if t[-1] < 1e-6:
        return pts

    # Ensure strictly increasing (remove near-duplicate parameter values)
    mask = np.ones(len(t), dtype=bool)
    for i in range(1, len(t)):
        if t[i] - t[i - 1] < 1e-9:
            mask[i] = False
    t   = t[mask]
    pts = pts[mask]
    if len(pts) < 2:
        return pts

    cs_x = CubicSpline(t, pts[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, pts[:, 1], bc_type="natural")

    t_new = np.linspace(0, t[-1], n_samples)
    return np.column_stack([cs_x(t_new), cs_y(t_new)])


def _arc_midpoint(pts: np.ndarray) -> Tuple[float, float]:
    """Point at 50 % of total arc length (linear interp between samples)."""
    cum  = _arc_length_cumulative(pts)
    half = cum[-1] / 2.0
    idx  = int(np.searchsorted(cum, half))
    idx  = min(idx, len(pts) - 1)
    if idx == 0:
        return float(pts[0, 0]), float(pts[0, 1])
    frac = (half - cum[idx - 1]) / (cum[idx] - cum[idx - 1] + 1e-12)
    x = pts[idx - 1, 0] + frac * (pts[idx, 0] - pts[idx - 1, 0])
    y = pts[idx - 1, 1] + frac * (pts[idx, 1] - pts[idx - 1, 1])
    return float(x), float(y)


def _arc_tangent_at_midpoint(pts: np.ndarray) -> Tuple[float, float]:
    """Unit tangent vector at the arc midpoint."""
    cum  = _arc_length_cumulative(pts)
    half = cum[-1] / 2.0
    idx  = int(np.searchsorted(cum, half))
    idx  = max(1, min(idx, len(pts) - 1))
    dx = pts[idx, 0] - pts[idx - 1, 0]
    dy = pts[idx, 1] - pts[idx - 1, 1]
    norm = math.sqrt(dx * dx + dy * dy) + 1e-12
    return dx / norm, dy / norm


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def compute_side_arcs(
    cp_t   : Dict,
    cp_b   : Dict,
    d_mean : float,
    side   : str,
    n_samples: int = 300,
) -> Dict:
    """
    Compute the occluded side-arc control points (7) and (8)
    and the interpolated outer / inner curves for one side.

    Parameters
    ----------
    cp_t, cp_b : control-point dicts for top and bottom blobs
                 (must contain pt1, pt2, pt5, pt6 for *side*)
    d_mean     : mean wire thickness d (px) from apex measurements
    side       : ``"left"`` or ``"right"``
    n_samples  : curve sampling density

    Returns
    -------
    dict with keys:
        pt7         – (x, y)  midpoint of outer side arc
        pt8         – (x, y)  pt7 offset inward by d_mean
        d_side_px   – float   Euclidean distance pt7→pt8 (≈ d_mean)
        outer_pts   – (N, 2)  interpolated outer side curve
        inner_pts   – (N, 2)  interpolated inner side curve
        outer_ctrl  – list of (x, y) control points used for outer
        inner_ctrl  – list of (x, y) control points used for inner
    """
    s = side   # shorthand

    # ── Outer:  (5)_top → (1)_top → (1)_bot → (5)_bot ──────────
    outer_ctrl = [
        cp_t[f"pt5_{s}"],
        cp_t[f"pt1_{s}"],
        cp_b[f"pt1_{s}"],
        cp_b[f"pt5_{s}"],
    ]

    outer_curve = _interpolate_parametric(outer_ctrl, n_samples)
    pt7 = _arc_midpoint(outer_curve)
    tx, ty = _arc_tangent_at_midpoint(outer_curve)

    # Inward normal (perpendicular to tangent, pointing toward link centre)
    if side == "left":
        nx, ny =  ty, -tx        # 90° CW
    else:
        nx, ny = -ty,  tx        # 90° CCW

    # Verify normal points inward (toward link centre)
    cx_link = (cp_t["x_apex"] + cp_b.get("x_apex", cp_t["x_apex"])) / 2.0
    cy_link = (cp_t["pt3"][1] + cp_b["pt3"][1]) / 2.0
    if nx * (cx_link - pt7[0]) + ny * (cy_link - pt7[1]) < 0:
        nx, ny = -nx, -ny

    pt8 = (pt7[0] + nx * d_mean, pt7[1] + ny * d_mean)

    # ── Inner:  (6)_top → (2)_top → (8) → (2)_bot → (6)_bot ───
    inner_ctrl = [
        cp_t[f"pt6_{s}"],
        cp_t[f"pt2_{s}"],
        pt8,
        cp_b[f"pt2_{s}"],
        cp_b[f"pt6_{s}"],
    ]

    inner_curve = _interpolate_parametric(inner_ctrl, n_samples)

    d_side = math.sqrt((pt7[0] - pt8[0]) ** 2 + (pt7[1] - pt8[1]) ** 2)

    return {
        "pt7"       : pt7,
        "pt8"       : pt8,
        "d_side_px" : d_side,
        "outer_pts" : outer_curve,
        "inner_pts" : inner_curve,
        "outer_ctrl": outer_ctrl,
        "inner_ctrl": inner_ctrl,
    }
