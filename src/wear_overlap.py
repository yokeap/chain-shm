"""
wear_overlap.py  (v2)
=====================
Area-overlap wear metric, per ``resource/full_mask_annotated.png``:

    wear% = area(A ∩ B) / area(A) × 100

where
    A = reconstructed end-crescent of the vertical link  (link_model.py)
    B = reconstructed round end of the horizontal wire   (wire_model.py)

A and B are the *nominal* (extrapolated, unworn) reconstructions; where the chain
has worn, the two reconstructions interpenetrate, and that overlap — as a fraction
of the link crescent A — is the wear proxy.  A larger circle B pushing deeper into
crescent A ⇒ larger overlap ⇒ higher wear.

Each vertical-link side (left/right) is paired with the nearest wire circle B (by x).

Usage
-----
    from wear_overlap import compute_wear, draw_overlay
    wear = compute_wear(link, wire, image.shape[:2])
    vis  = draw_overlay(image, link, wire, wear)
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR palette (matches annotation intent)
RED    = (0,   50, 255)
BLUE   = (220,  90,  20)
GREEN  = (50, 200,  50)
YELLOW = (0,  220, 255)
ORANGE = (0,  165, 255)
WHITE  = (255, 255, 255)
MAGENTA= (255, 50, 255)
A_FILL = (180, 120,  40)    # translucent area-A fill
OVER   = (0,   0,  255)     # overlap highlight (red)


# ══════════════════════════════════════════════════════════════════
# Mask builders
# ══════════════════════════════════════════════════════════════════

def _poly_mask(poly: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    if poly is None or len(poly) < 3:
        return m
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m


def _circle_mask(center, radius, shape) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(center[0]), int(center[1])), int(round(radius)), 255, -1)
    return m


def _poly_ref_x(poly: np.ndarray) -> float:
    return float(np.mean(poly[:, 0])) if poly is not None and len(poly) else 0.0


# ══════════════════════════════════════════════════════════════════
# Wear computation
# ══════════════════════════════════════════════════════════════════

def compute_wear(link: Optional[Dict], wire: Dict, shape) -> Dict:
    """
    Compute per-side wear = area(A∩B)/area(A)×100.

    Parameters
    ----------
    link  : dict from link_model.model_link (or None)
    wire  : dict from wire_model.model_wire
    shape : (h, w) of the image the masks live in

    Returns
    -------
    dict: pairs[], wear_pct_left, wear_pct_right, d_mean_px, b_px
    """
    empty = {"pairs": [], "wear_pct_left": None, "wear_pct_right": None,
             "d_mean_px": 0.0, "b_px": wire.get("b_px", 0.0)}
    if link is None or not link.get("sides"):
        return empty

    circles = list(wire.get("circles", []))
    pairs: List[Dict] = []
    used = set()

    for side, sd in link["sides"].items():
        A_poly = sd["area_A_poly"]
        A_mask = _poly_mask(A_poly, shape)
        area_A = int(A_mask.sum() // 255)
        if area_A == 0:
            logger.warning(f"  wear {side}: area A is empty, skipping")
            continue

        # Pair with nearest unused circle B by x.
        ref_x = _poly_ref_x(A_poly)
        best, bi, bd = None, -1, float("inf")
        for i, c in enumerate(circles):
            if i in used:
                continue
            dd = abs(float(c["center"][0]) - ref_x)
            if dd < bd:
                bd, best, bi = dd, c, i
        if best is None:
            logger.warning(f"  wear {side}: no wire circle B to pair with")
            continue
        used.add(bi)

        B_mask = _circle_mask(best["center"], best["radius"], shape)
        area_B = int(B_mask.sum() // 255)
        over   = cv2.bitwise_and(A_mask, B_mask)
        area_over = int(over.sum() // 255)
        wear_pct  = 100.0 * area_over / max(1, area_A)

        logger.info(
            f"  wear {side}: A={area_A}px²  B={area_B}px²  "
            f"A∩B={area_over}px²  wear={wear_pct:.1f}%"
        )

        pairs.append({
            "side": side,
            "area_A": area_A, "area_B": area_B, "area_overlap": area_over,
            "wear_pct": wear_pct,
            "A_poly": A_poly, "circle": best,
            "overlap_mask": over,
        })

    def _by_side(s):
        return next((p["wear_pct"] for p in pairs if p["side"] == s), None)

    return {
        "pairs": pairs,
        "wear_pct_left":  _by_side("left"),
        "wear_pct_right": _by_side("right"),
        "d_mean_px": link.get("d_mean_px", 0.0),
        "b_px": wire.get("b_px", 0.0),
    }


# ══════════════════════════════════════════════════════════════════
# Overlay drawing
# ══════════════════════════════════════════════════════════════════

def _dot(vis, pt, color, r=8, label=None):
    if pt is None:
        return
    cx, cy = int(pt[0]), int(pt[1])
    cv2.circle(vis, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), r, WHITE, 1, cv2.LINE_AA)
    if label:
        cv2.putText(vis, label, (cx + r + 1, cy - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 220, 30), 2, cv2.LINE_AA)


def _dashed(vis, pts, color, th=2, seg=12, gap=8):
    p = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    n = len(p); i = 0; on = True
    while i < n:
        e = min(i + (seg if on else gap), n)
        if on and e - i >= 2:
            cv2.polylines(vis, [p[i:e]], False, color, th, cv2.LINE_AA)
        i = e; on = not on


def draw_overlay(image: np.ndarray, link: Optional[Dict], wire: Dict,
                 wear: Dict) -> np.ndarray:
    """Render area A, circle B, A∩B overlap, points 1-12, and wear labels."""
    vis = image.copy()
    if link is None:
        return vis

    # 1. Area-A fill (translucent) + overlap highlight
    for p in wear.get("pairs", []):
        A_mask = _poly_mask(p["A_poly"], vis.shape[:2])
        ov = np.zeros_like(vis); ov[A_mask > 0] = A_FILL
        vis = cv2.addWeighted(vis, 1.0, ov, 0.35, 0)
        om = p.get("overlap_mask")
        if om is not None:
            ov2 = np.zeros_like(vis); ov2[om > 0] = OVER
            vis = cv2.addWeighted(vis, 1.0, ov2, 0.55, 0)

    # 2. Per-side arcs (blue outer, red inner) + labelled points 1-8
    for side, sd in link["sides"].items():
        _dashed(vis, sd["outer_pts"], BLUE, 2, seg=12, gap=8)
        _dashed(vis, sd["inner_pts"], RED,  2, seg=10, gap=8)
        lm = sd["labels"]
        _dot(vis, sd["outer_top"], BLUE, 8, lm["outer_top"])
        _dot(vis, sd["outer_bot"], BLUE, 8, lm["outer_bot"])
        _dot(vis, sd["inner_top"], RED,  8, lm["inner_top"])
        _dot(vis, sd["inner_bot"], RED,  8, lm["inner_bot"])

    # 3. Points 9-12 + thickness d lines
    p9, p10, p11, p12 = link.get("p9"), link.get("p10"), link.get("p11"), link.get("p12")
    if p9 and p10:
        cv2.line(vis, (int(p9[0]), int(p9[1])), (int(p10[0]), int(p10[1])), YELLOW, 3, cv2.LINE_AA)
        cv2.putText(vis, f"d_top={link['d_top_px']:.0f}", (int(p9[0]) + 8, int(p9[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)
    if p11 and p12:
        cv2.line(vis, (int(p11[0]), int(p11[1])), (int(p12[0]), int(p12[1])), YELLOW, 3, cv2.LINE_AA)
    _dot(vis, p9, YELLOW, 7, "9"); _dot(vis, p10, YELLOW, 7, "10")
    _dot(vis, p11, YELLOW, 7, "11"); _dot(vis, p12, YELLOW, 7, "12")

    # 4. Wire: circle B, tip (point 1), thickness b
    for seg in wire.get("segments", []):
        cv2.line(vis, seg["top_line"][0], seg["top_line"][1], RED, 2, cv2.LINE_AA)
        cv2.line(vis, seg["bot_line"][0], seg["bot_line"][1], RED, 2, cv2.LINE_AA)
    for c in wire.get("circles", []):
        cx, cy = int(c["center"][0]), int(c["center"][1]); r = int(round(c["radius"]))
        cv2.circle(vis, (cx, cy), r, GREEN, 2, cv2.LINE_AA)
        cv2.line(vis, (cx, cy - r), (cx, cy + r), YELLOW, 2, cv2.LINE_AA)
        _dot(vis, c["tip"], YELLOW, 7, "1")
        cv2.putText(vis, "B", (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAGENTA, 2, cv2.LINE_AA)

    # 5. Per-side wear labels near the overlap
    for p in wear.get("pairs", []):
        c = p["circle"]; cx, cy = int(c["center"][0]), int(c["center"][1])
        cv2.putText(vis, f"{p['side']} wear={p['wear_pct']:.1f}%", (cx - 40, cy - int(c['radius']) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, MAGENTA, 2, cv2.LINE_AA)

    # 6. Summary banner
    wl = wear.get("wear_pct_left"); wr = wear.get("wear_pct_right")
    lines = [
        f"d={wear.get('d_mean_px', 0):.0f}px  b={wear.get('b_px', 0):.0f}px",
        f"wear  L={_fmt(wl)}  R={_fmt(wr)}",
    ]
    py = max(20, link.get("top_y1", 40) - 50)
    for i, ln in enumerate(lines):
        y = py + i * 24
        cv2.rectangle(vis, (8, y - 16), (360, y + 6), (20, 20, 20), -1)
        cv2.putText(vis, ln, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
    return vis


def _fmt(v):
    return "n/a" if v is None else f"{v:.1f}%"
