"""
link_model.py  (v2)
===================
12-control-point model of the vertical chain link + reconstructed **area A**,
following ``resource/vertical_mask_annotated.png``.

The vertical link under inspection is the leftmost oval in ``mask_vert_chain``.
Both of its rounded end-caps are occluded by the crossing horizontal links, so
they are *reconstructed*:

  Rails (visible)           per-column edges of the two white bands
    top rail : top_outer / top_inner   (top_inner  = hole top boundary)
    bot rail : bot_inner / bot_outer   (bot_inner  = hole bottom boundary)

  Points 9,10,11,12         bar thickness ``d`` at the apex (oval centre)
    9  = top outer   10 = top inner   11 = bot inner   12 = bot outer
    d_top = |9-10|   d_bot = |11-12|   d = mean

  Points 3,4,5,6 (inner)    hole-boundary corners where the bar reaches full
                            thickness (arc endpoints, red arcs)
    left  : 3 = inner top, 4 = inner bot
    right : 5 = inner top, 6 = inner bot

  Points 1,2,7,8 (outer)    inner corners offset OUTWARD by ``d`` (blue arcs)
    left  : 1 = 3 - d,  2 = 4 - d        (cap opens left)
    right : 7 = 5 + d,  8 = 6 + d        (cap opens right)

  Area A                    crescent between the inner (red) arc and outer
                            (blue) arc at each cap; width ≈ d.

Usage
-----
    from link_model import model_link, draw_link
    link = model_link(img_rect, vert_mask_rect)
    link["sides"]["right"]["area_A_poly"]   # (N,2) closed crescent
    vis  = draw_link(vert_mask_rect, link)  # annotation-style overlay
"""

from __future__ import annotations
import logging
from typing import Dict, Optional

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────

def _pair_left_oval(bw: np.ndarray):
    """Return (top_comp, bot_comp) bool arrays for the leftmost rail pair."""
    lab, n = scipy_label(bw > 0)
    blobs = []
    for i in range(1, n + 1):
        comp = (lab == i)
        if comp.sum() < 2000:
            continue
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        blobs.append({"comp": comp, "cy": rows.mean(), "cx": cols.mean()})
    if len(blobs) < 2:
        return None, None
    med = np.median([b["cy"] for b in blobs])
    tops = sorted([b for b in blobs if b["cy"] < med],  key=lambda b: b["cx"])
    bots = sorted([b for b in blobs if b["cy"] >= med], key=lambda b: b["cx"])
    if not tops or not bots:
        return None, None
    return tops[0]["comp"], bots[0]["comp"]


def _scan(comp: np.ndarray):
    """Per-column first/last white row over the columns the comp occupies."""
    xs, e0, e1 = [], [], []
    cols = np.where(comp.any(axis=0))[0]
    for x in range(int(cols[0]), int(cols[-1]) + 1):
        ys = np.where(comp[:, x])[0]
        if len(ys):
            xs.append(x); e0.append(ys[0]); e1.append(ys[-1])
    return np.array(xs), np.array(e0, float), np.array(e1, float)


def _band_edges(wire_bw: np.ndarray, x_lo: int, x_hi: int):
    """Median top/bottom edge of the horizontal band over columns [x_lo,x_hi]."""
    yt, yb = [], []
    x_lo = max(0, x_lo); x_hi = min(wire_bw.shape[1] - 1, x_hi)
    for x in range(x_lo, x_hi + 1):
        ys = np.where(wire_bw[:, x])[0]
        if len(ys):
            yt.append(ys[0]); yb.append(ys[-1])
    if not yt:
        return None, None
    return float(np.median(yt)), float(np.median(yb))


def _cap_arc(inner_top, inner_bot, outward: int, n: int = 60) -> np.ndarray:
    """Shallow rounded arc from inner_top → inner_bot, bulging outward (±1)."""
    ax, ay = inner_top; bx, by = inner_bot
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    bulge = abs(by - ay) * 0.35
    tip = (mx + outward * bulge, my)
    c = np.array([inner_top, tip, inner_bot], float)
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * c[0] + 2 * (1 - t) * t * c[1] + t ** 2 * c[2]


def _close_crescent(outer_pts: np.ndarray, inner_pts: np.ndarray) -> np.ndarray:
    """Closed polygon: outer arc then inner arc reversed (endpoints matched)."""
    o = np.asarray(outer_pts, float)
    i = np.asarray(inner_pts, float)
    if len(o) < 2 or len(i) < 2:
        return np.vstack([o, i]) if len(o) or len(i) else np.empty((0, 2))
    if np.linalg.norm(i[0] - o[-1]) > np.linalg.norm(i[-1] - o[-1]):
        i = i[::-1]
    return np.vstack([o, i])


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def model_link(
    image: np.ndarray,
    vert_mask: np.ndarray,
    wire_mask: Optional[np.ndarray] = None,
    px_per_mm: float = 1.0,
) -> Optional[Dict]:
    """
    Build the 12-point vertical-link model for the (leftmost) link.

    Points 1,2,7,8 (outer corners) are placed at the **intersection of the
    vertical link with the crossing horizontal chain**: the x is the vertical
    link's outer extent (its junction/neck with the horizontal band) and the y
    is the horizontal band's top/bottom edge (from ``wire_mask``).  Points
    3,4,5,6 (inner corners) sit on the link hole boundary, x-aligned per side.
    If ``wire_mask`` is None the outer corners fall back to the inner corners
    offset outward by the bar thickness ``d``.

    Returns a dict (or None) with:
        d_top_px, d_bot_px, d_mean_px, (+ _mm)
        p1..p12                    – annotation points (x, y)
        apex_x
        sides = {"left": {...}, "right": {...}}
    """
    _, bw = cv2.threshold(vert_mask, 127, 255, cv2.THRESH_BINARY)
    top, bot = _pair_left_oval(bw)
    if top is None:
        logger.warning("model_link: no link pair found in vert_mask")
        return None

    xt, t_out, t_in = _scan(top)   # top rail: outer(top edge), inner(bottom edge)
    xb, b_in, b_out = _scan(bot)   # bot rail: inner(top edge),  outer(bottom edge)
    if len(xt) < 10 or len(xb) < 10:
        logger.warning("model_link: rail scan too short")
        return None

    # oval outer extent (includes the neck junction with the horizontal band)
    x_oL = min(int(xt[0]), int(xb[0]))
    x_oR = max(int(xt[-1]), int(xb[-1]))
    apex = (x_oL + x_oR) // 2

    def at(xs, arr, x):
        return float(arr[int(np.argmin(np.abs(xs - x)))])

    p9  = (apex, at(xt, t_out, apex))   # top outer
    p10 = (apex, at(xt, t_in,  apex))   # top inner
    p11 = (apex, at(xb, b_in,  apex))   # bot inner
    p12 = (apex, at(xb, b_out, apex))   # bot outer
    d_top = p10[1] - p9[1]
    d_bot = p12[1] - p11[1]
    d_mean = 0.5 * (d_top + d_bot)

    # ── inner corners: where the link hole opens, x-aligned per side ──
    # Detect the hole-opening column from the TOP rail full-thickness span, then
    # sample BOTH rails' inner edges at that same x (so 3↔4 and 5↔6 are aligned).
    th_t = t_in - t_out
    kt = np.where(th_t >= 0.9 * th_t.max())[0]
    x_hL, x_hR = int(xt[kt[0]]), int(xt[kt[-1]])
    p3 = (float(x_hL), at(xt, t_in, x_hL))   # left  inner top
    p4 = (float(x_hL), at(xb, b_in, x_hL))   # left  inner bot
    p5 = (float(x_hR), at(xt, t_in, x_hR))   # right inner top
    p6 = (float(x_hR), at(xb, b_in, x_hR))   # right inner bot

    # ── outer corners: vertical∩horizontal intersection ──
    if wire_mask is not None:
        _, wbw = cv2.threshold(wire_mask, 127, 255, cv2.THRESH_BINARY)
        span = x_oR - x_oL
        yTl, yBl = _band_edges(wbw, x_oL - span // 6, x_oL + span // 6)
        yTr, yBr = _band_edges(wbw, x_oR - span // 6, x_oR + span // 6)
    else:
        yTl = yBl = yTr = yBr = None

    if yTl is not None:
        p1 = (float(x_oL), yTl); p2 = (float(x_oL), yBl)
    else:                                        # fallback: offset inner by d
        p1 = (p3[0] - d_mean, p3[1]); p2 = (p4[0] - d_mean, p4[1])
    if yTr is not None:
        p7 = (float(x_oR), yTr); p8 = (float(x_oR), yBr)
    else:
        p7 = (p5[0] + d_mean, p5[1]); p8 = (p6[0] + d_mean, p6[1])

    # crescent boundaries as rounded caps (circular-arc / semicircle, bulging out):
    #   inner = hole rounded end (3→4 / 5→6),  outer = link outer end (1→2 / 7→8)
    inL = _cap_arc(p3, p4, -1); outL = _cap_arc(p1, p2, -1)   # left cap opens left
    inR = _cap_arc(p5, p6, +1); outR = _cap_arc(p7, p8, +1)   # right cap opens right

    polyL = _close_crescent(outL, inL)
    polyR = _close_crescent(outR, inR)
    areaL = float(cv2.contourArea(polyL.astype(np.float32))) if len(polyL) >= 3 else 0.0
    areaR = float(cv2.contourArea(polyR.astype(np.float32))) if len(polyR) >= 3 else 0.0
    scale2 = px_per_mm * px_per_mm

    sides = {
        "left": {
            "labels": {"outer_top": "1", "outer_bot": "2",
                       "inner_top": "3", "inner_bot": "4"},
            "outer_top": p1, "outer_bot": p2, "inner_top": p3, "inner_bot": p4,
            "outer_pts": outL, "inner_pts": inL,
            "d_side_px": float(d_mean),
            "area_A_poly": polyL,
            "area_A_px": areaL, "area_A_mm2": areaL / scale2,
        },
        "right": {
            "labels": {"inner_top": "5", "inner_bot": "6",
                       "outer_top": "7", "outer_bot": "8"},
            "outer_top": p7, "outer_bot": p8, "inner_top": p5, "inner_bot": p6,
            "outer_pts": outR, "inner_pts": inR,
            "d_side_px": float(d_mean),
            "area_A_poly": polyR,
            "area_A_px": areaR, "area_A_mm2": areaR / scale2,
        },
    }
    logger.info(f"model_link: area_A left={areaL:.0f}px²  right={areaR:.0f}px²")

    logger.info(f"model_link: apex={apex}  d_top={d_top:.1f} d_bot={d_bot:.1f} "
                f"d_mean={d_mean:.1f}px")

    return {
        "d_top_px": float(d_top), "d_top_mm": float(d_top) / px_per_mm,
        "d_bot_px": float(d_bot), "d_bot_mm": float(d_bot) / px_per_mm,
        "d_mean_px": float(d_mean), "d_mean_mm": float(d_mean) / px_per_mm,
        "apex_x": apex, "top_y1": int(min(p9[1], p10[1])),
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6,
        "p7": p7, "p8": p8, "p9": p9, "p10": p10, "p11": p11, "p12": p12,
        "sides": sides,
    }


# ──────────────────────────────────────────────────────────────────
# Annotation-style overlay (vertical mask only)
# ──────────────────────────────────────────────────────────────────

_RED    = (0,   50, 255)
_GREEN  = (50, 200,  50)
_YELLOW = (0,  220, 255)
_BLUE   = (230, 120,  20)
_WHITE  = (255, 255, 255)


def _dashed(vis, pts, color, th=3, seg=14, gap=9):
    p = np.asarray(pts, np.int32).reshape(-1, 1, 2)
    i, on = 0, True
    while i < len(p):
        e = min(i + (seg if on else gap), len(p))
        if on and e - i >= 2:
            cv2.polylines(vis, [p[i:e]], False, color, th, cv2.LINE_AA)
        i, on = e, not on


def _dot(vis, pt, color, lbl):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(vis, (x, y), 16, color, -1, cv2.LINE_AA)
    cv2.circle(vis, (x, y), 16, _WHITE, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(vis, lbl, (x - tw // 2, y + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


_FILL_L = (200, 150, 40)     # left crescent fill  (teal-ish)
_FILL_R = (60, 160, 240)     # right crescent fill (orange-ish)


def _fill_poly(vis, poly, color, alpha=0.45):
    if poly is None or len(poly) < 3:
        return
    ov = vis.copy()
    cv2.fillPoly(ov, [poly.astype(np.int32)], color)
    cv2.addWeighted(ov, alpha, vis, 1 - alpha, 0, vis)


def _area_label(vis, side, color):
    """Print 'A = NNNN px²' at the crescent centroid."""
    poly = side["area_A_poly"]
    if poly is None or len(poly) < 3:
        return
    cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
    txt = f"A={side['area_A_px']:.0f}px2"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    org = (int(cx - tw / 2), int(cy + th / 2))
    cv2.rectangle(vis, (org[0] - 6, org[1] - th - 6), (org[0] + tw + 6, org[1] + 6),
                  (25, 25, 25), -1)
    cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_link(vert_mask: np.ndarray, link: Dict) -> np.ndarray:
    """Render the 12-point model + filled/labelled crescent areas on the mask."""
    vis = cv2.cvtColor(vert_mask, cv2.COLOR_GRAY2BGR)
    if link is None:
        return vis
    L, R = link["sides"]["left"], link["sides"]["right"]

    # filled crescent areas: left (1,2,3,4) and right (5,6,7,8)
    _fill_poly(vis, L["area_A_poly"], _FILL_L)
    _fill_poly(vis, R["area_A_poly"], _FILL_R)

    _dashed(vis, L["inner_pts"], _RED);  _dashed(vis, R["inner_pts"], _RED)
    _dashed(vis, L["outer_pts"], _BLUE); _dashed(vis, R["outer_pts"], _BLUE)

    P = {k: link[k] for k in ("p9", "p10", "p11", "p12")}
    cv2.line(vis, tuple(map(int, P["p9"])),  tuple(map(int, P["p10"])), _YELLOW, 3, cv2.LINE_AA)
    cv2.line(vis, tuple(map(int, P["p11"])), tuple(map(int, P["p12"])), _YELLOW, 3, cv2.LINE_AA)

    _area_label(vis, L, _WHITE)
    _area_label(vis, R, _WHITE)

    colmap = {"p1": _YELLOW, "p2": _YELLOW, "p7": _YELLOW, "p8": _YELLOW,
              "p3": _GREEN, "p4": _GREEN, "p5": _GREEN, "p6": _GREEN,
              "p9": _RED, "p10": _RED, "p11": _RED, "p12": _RED}
    for k in range(1, 13):
        _dot(vis, link[f"p{k}"], colmap[f"p{k}"], str(k))
    return vis
