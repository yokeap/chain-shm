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

Each vertical-link crescent A is paired with the wire cap B *inserted into it* —
the cap whose x-span overlaps the crescent's (same interlock) — assigned greedily,
closest first.  A crescent whose interlock partner is off-frame reports N/A.

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

def _normalize_parallax(pairs: List[Dict], frame_w: int) -> Optional[float]:
    """
    Remove the systematic left→right wear ramp caused by **wire↔link parallax**.

    The horizontal wire and the vertical links sit at different depths (the wire
    threads *through* the link, one bar-thickness apart in z).  A camera viewing
    at an angle therefore sees the wire shifted relative to the links by an amount
    that grows ~linearly across the frame — so the measured A∩B overlap ramps with
    the interlock's x-position even though the physical wear along one chain is
    roughly uniform.  A single homography cannot remove this (it is non-planar).

    We model it as ``wear(x) = wear_center + k·(x − x_c)`` (x_c = frame centre,
    where parallax ≈ 0), fit ``k`` from the frame's interlocks by least squares,
    and report each interlock's wear **referenced back to the centre**
    (``wear_corr``).  Needs ≥2 interlocks; with one, no correction is possible.

    Returns the fitted slope ``k`` (%/px) or ``None``.
    """
    if len(pairs) < 2:
        for p in pairs:
            p["wear_corr"] = p["wear_pct"]
        return None
    xc = frame_w / 2.0
    xs = np.array([float(p["circle"]["center"][0]) - xc for p in pairs])
    ws = np.array([float(p["wear_pct"]) for p in pairs])
    k = float(np.polyfit(xs, ws, 1)[0])           # slope of wear vs (x − centre)
    for p, dx in zip(pairs, xs):
        p["wear_corr"] = float(p["wear_pct"] - k * dx)   # wear at frame centre
    return k


def compute_wear(link: Optional[Dict], wire: Dict, shape) -> Dict:
    """
    Compute wear = area(A∩B)/area(A)×100, **anchored on the horizontal wire**.

    The wire link is the reference: each of its caps (circle **B**) presses into
    the flat-link crescent (**area A**) it is inserted into.  We therefore iterate
    over the wire's caps and, for each, pick the crescent whose x-span the cap
    overlaps — pooling crescents from *every* flat link (``sides`` +
    ``extra_sides``) so both the left cap (→ left flat link) and the right cap
    (→ right flat link) are measured symmetrically.  Results are labelled by the
    **cap** side (left/right of the wire).

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
    if link is None:
        return empty

    circles = list(wire.get("circles", []))

    # ── Pool crescents A from every flat link (primary + extras) ──────────
    crescents: Dict[str, Dict] = {}
    crescents.update(link.get("sides", {}))
    crescents.update(link.get("extra_sides", {}))

    cres: Dict[str, Dict] = {}
    for key, sd in crescents.items():
        A_poly = sd.get("area_A_poly")
        if A_poly is None or len(A_poly) < 3:
            continue
        A_mask = _poly_mask(A_poly, shape)
        area_A = int(A_mask.sum() // 255)
        if area_A == 0:
            continue
        cres[key] = {"A_poly": A_poly, "A_mask": A_mask, "area_A": area_A,
                     "ref_x": _poly_ref_x(A_poly),
                     "ax0": float(A_poly[:, 0].min()), "ax1": float(A_poly[:, 0].max())}

    # ── For each cap B, find the crescent A it is inserted into (x-span overlap),
    #    assign greedily closest-first, one crescent per cap ──
    cand = []   # (center-distance, circle_index, crescent_key)
    for i, c in enumerate(circles):
        cx, r = float(c["center"][0]), float(c["radius"])
        for key, a in cres.items():
            if cx + r < a["ax0"] or cx - r > a["ax1"]:
                continue                      # cap not inserted into this crescent
            cand.append((abs(cx - a["ref_x"]), i, key))
    cand.sort()

    pairs: List[Dict] = []
    used_circ, used_cres = set(), set()
    for _, i, key in cand:
        if i in used_circ or key in used_cres:
            continue
        used_circ.add(i); used_cres.add(key)
        c = circles[i]; a = cres[key]
        side = c.get("side", "?")            # label by the CAP side (wire anchor)

        B_mask = _circle_mask(c["center"], c["radius"], shape)
        area_B = int(B_mask.sum() // 255)
        over   = cv2.bitwise_and(a["A_mask"], B_mask)
        area_over = int(over.sum() // 255)
        wear_pct  = 100.0 * area_over / max(1, a["area_A"])   # raw geometric ratio

        logger.info(
            f"  wear {side} cap (crescent {key}): cap_x={float(c['center'][0]):.0f}  "
            f"A={a['area_A']}px²  B={area_B}px²  A∩B={area_over}px²  wear={wear_pct:.1f}%"
        )

        pairs.append({
            "side": side, "crescent": key,
            "area_A": a["area_A"], "area_B": area_B, "area_overlap": area_over,
            "wear_pct": wear_pct,
            "A_poly": a["A_poly"], "circle": c,
            "overlap_mask": over,
        })

    for i, c in enumerate(circles):
        if i not in used_circ:
            logger.warning(f"  wear {c.get('side','?')} cap: no crescent A to press into (N/A)")

    # ── parallax normalization: flatten the left→right ramp (see helper) ──
    h, w = shape
    k_par = _normalize_parallax(pairs, w)
    if k_par is not None:
        logger.info(f"  parallax slope k={k_par*1000:.2f}%/1000px  → "
                    f"wear referenced to frame centre (x={w//2})")

    def _by_side(s):
        return next((p["wear_pct"] for p in pairs if p["side"] == s), None)

    # sample-level headline = worst (deepest) interlock, mirroring the caliper
    wear_raw_max = max((p["wear_pct"] for p in pairs), default=None)
    # parallax-corrected: interlocks now agree, so the frame-centre value is the
    # single representative wear (mean of the corrected per-interlock values)
    corr = [p["wear_corr"] for p in pairs if "wear_corr" in p]
    wear_corr_mean = float(np.mean(corr)) if corr else None

    return {
        "pairs": pairs,
        "wear_pct_left":  _by_side("left"),
        "wear_pct_right": _by_side("right"),
        "wear_raw_max": wear_raw_max,
        "wear_corr_mean": wear_corr_mean,
        "parallax_k": k_par,
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
    #    (primary link + every extra flat link's crescents)
    for side, sd in {**link.get("sides", {}), **link.get("extra_sides", {})}.items():
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

    # 4. Wire: straight edges, circle B, tip (point 1), pt2 corners, thickness b
    if wire.get("top_line") and wire.get("bot_line"):
        cv2.line(vis, wire["top_line"][0], wire["top_line"][1], RED, 2, cv2.LINE_AA)
        cv2.line(vis, wire["bot_line"][0], wire["bot_line"][1], RED, 2, cv2.LINE_AA)
    for cn in wire.get("corners", []):
        _dot(vis, cn["pt"], YELLOW, 6, "2")
    for c in wire.get("circles", []):
        cx, cy = int(c["center"][0]), int(c["center"][1]); r = int(round(c["radius"]))
        cv2.circle(vis, (cx, cy), r, GREEN, 2, cv2.LINE_AA)
        cv2.line(vis, (cx, cy - r), (cx, cy + r), YELLOW, 2, cv2.LINE_AA)
        _dot(vis, c["tip"], YELLOW, 7, "1")
        cv2.putText(vis, "B", (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(vis, f"area B={c.get('area_B_px', 0):.0f}px2", (cx - r, cy + r + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)

    # 5. Per-side wear labels near the overlap
    for p in wear.get("pairs", []):
        c = p["circle"]; cx, cy = int(c["center"][0]), int(c["center"][1])
        cv2.putText(vis, f"{p['side']} wear={p['wear_pct']:.1f}%", (cx - 40, cy - int(c['radius']) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, MAGENTA, 2, cv2.LINE_AA)

    # 6. Summary banner
    wl = wear.get("wear_pct_left"); wr = wear.get("wear_pct_right")
    trig = wear.get("trigger") or {}
    tflag = "TRIGGERED" if trig.get("triggered") else "NOT TRIGGERED"
    lines = [
        f"d={wear.get('d_mean_px', 0):.0f}px  b={wear.get('b_px', 0):.0f}px",
        f"wear  L={_fmt(wl)}  R={_fmt(wr)}",
        f"[{tflag}] {trig.get('reason', '')}",
    ]
    py = max(20, link.get("top_y1", 40) - 74)
    for i, ln in enumerate(lines):
        y = py + i * 24
        col = (GREEN if trig.get("triggered") else ORANGE) if i == 2 else WHITE
        cv2.rectangle(vis, (8, y - 16), (8 + 11 * len(ln), y + 6), (20, 20, 20), -1)
        cv2.putText(vis, ln, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
    return vis


def _fmt(v):
    return "n/a" if v is None else f"{v:.1f}%"


# ══════════════════════════════════════════════════════════════════
# Trigger gate
# ══════════════════════════════════════════════════════════════════

def evaluate_trigger(link: Optional[Dict], wire: Dict, wear: Dict) -> Dict:
    """
    Decide whether this frame yields a measurement.

    A single **interlock** is all that is required: a real wire cap (area B,
    already frame-cut-rejected by ``model_wire``) inserted into a reconstructable
    crescent (area A).  That joint is self-contained, so it can be measured
    **regardless of whether the whole wire link or the whole flat link is fully
    in-frame** — the surrounding links may run off the frame edge.  We therefore
    fire on *any* complete interlock.

    ``wire_full`` and ``n_links`` are still reported as **confidence context**
    (a fully-visible wire link gives both interlocks; a cut one gives a single
    one) but they no longer gate the measurement.

    Returns
    -------
    dict: triggered(bool), wire_full(bool), has_interlock(bool),
          n_interlocks(int), reason(str)
    """
    n = len(wear.get("pairs", []))
    triggered = n > 0
    wire_full = bool(wire and wire.get("full"))
    n_caps = len(wire.get("circles", [])) if wire else 0

    if triggered:
        reason = f"{n} interlock(s) measured" + ("" if wire_full else " (wire cut - partial view)")
    elif n_caps == 0:
        reason = "no horizontal wire cap detected"
    elif link is None:
        reason = "no vertical crescent detected"
    else:
        reason = "wire cap present but not inserted into any crescent"

    return {"triggered": triggered, "wire_full": wire_full,
            "has_interlock": triggered, "n_interlocks": n, "reason": reason}


# ══════════════════════════════════════════════════════════════════
# Full reconstruction mask — the clean A / B / A∩B schematic
# ══════════════════════════════════════════════════════════════════

def draw_full_recon_mask(link: Optional[Dict], wire: Dict, wear: Dict,
                         shape) -> np.ndarray:
    """
    Render the *reconstruction* on a dark canvas (not the photo): every
    vertical-link crescent **area A** (blue), every wire cap **area B** (green),
    and the measured **A∩B** overlap (red) with the wear % per interlock.

    This is the ``full_recon_mask`` — the geometric result of combining the
    vertical and horizontal reconstructions, independent of the source texture.
    """
    h, w = shape
    vis = np.full((h, w, 3), 18, dtype=np.uint8)   # near-black canvas
    if link is None:
        return vis

    # ── all reconstructed area A crescents (blue, filled + outlined) ──
    all_sides = {**link.get("sides", {}), **link.get("extra_sides", {})}
    for side, sd in all_sides.items():
        poly = sd.get("area_A_poly")
        if poly is None or len(poly) < 3:
            continue
        A = _poly_mask(poly, shape)
        vis[A > 0] = (150, 90, 30)
        cv2.polylines(vis, [poly.astype(np.int32)], True, BLUE, 2, cv2.LINE_AA)

    # ── all reconstructed circle B caps (green, filled + outlined) ──
    for c in wire.get("circles", []):
        B = _circle_mask(c["center"], c["radius"], shape)
        g = np.zeros_like(vis); g[B > 0] = (30, 130, 30)
        vis = cv2.addWeighted(vis, 1.0, g, 0.6, 0)
        cx, cy, r = int(c["center"][0]), int(c["center"][1]), int(round(c["radius"]))
        cv2.circle(vis, (cx, cy), r, GREEN, 2, cv2.LINE_AA)

    # ── measured overlaps (red) + per-interlock wear ──
    for p in wear.get("pairs", []):
        om = p.get("overlap_mask")
        if om is not None:
            vis[om > 0] = OVER
        c = p["circle"]; cx, cy = int(c["center"][0]), int(c["center"][1])
        cv2.putText(vis, f"A^B={p['area_overlap']}px2", (cx - 70, cy - int(c["radius"]) - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, OVER, 2, cv2.LINE_AA)
        cv2.putText(vis, f"wear={p['wear_pct']:.1f}%", (cx - 70, cy - int(c["radius"]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

    # ── legend + banner ──
    cv2.putText(vis, "A (link crescent)", (14, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2, cv2.LINE_AA)
    cv2.putText(vis, "B (wire cap)",      (14, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    cv2.putText(vis, "A n B (overlap)",   (14, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, OVER, 2, cv2.LINE_AA)
    wl = wear.get("wear_pct_left"); wr = wear.get("wear_pct_right")
    trig = wear.get("trigger") or {}
    tflag = "TRIGGERED" if trig.get("triggered") else "NOT TRIGGERED"
    banner = (f"d={wear.get('d_mean_px',0):.0f}px  b={wear.get('b_px',0):.0f}px   "
              f"wear L={_fmt(wl)}  R={_fmt(wr)}")
    cv2.rectangle(vis, (8, 12), (8 + 12 * len(banner), 44), (20, 20, 20), -1)
    cv2.putText(vis, banner, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
    tline = f"[{tflag}] {trig.get('reason', '')}"
    cv2.rectangle(vis, (8, 48), (8 + 11 * len(tline), 78), (20, 20, 20), -1)
    cv2.putText(vis, tline, (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                GREEN if trig.get("triggered") else ORANGE, 2, cv2.LINE_AA)
    return vis


def draw_recon_overlay(image: np.ndarray, link: Optional[Dict], wire: Dict,
                       wear: Dict, alpha: float = 0.45) -> np.ndarray:
    """
    Reconstruction composited with **opacity over the real (corrected) image**:
    filled area A (blue) and every cap area B (green) at ``alpha`` transparency,
    each measured A∩B overlap (red, stronger) with its wear %, plus a banner.
    Same information as ``full_recon_mask`` but on the actual photo so the
    geometry can be checked against the chain texture.
    """
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    shape = vis.shape[:2]
    all_sides = {**link.get("sides", {}), **link.get("extra_sides", {})} if link else {}

    # translucent fills: A (blue), then B (green) on top
    layer = vis.copy()
    A = np.zeros(shape, np.uint8)
    for sd in all_sides.values():
        poly = sd.get("area_A_poly")
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(A, [poly.astype(np.int32)], 255)
    B = np.zeros(shape, np.uint8)
    for c in wire.get("circles", []):
        cv2.circle(B, (int(c["center"][0]), int(c["center"][1])), int(round(c["radius"])), 255, -1)
    layer[A > 0] = BLUE
    layer[B > 0] = GREEN
    reg = (A > 0) | (B > 0)
    blended = cv2.addWeighted(vis, 1 - alpha, layer, alpha, 0)
    vis[reg] = blended[reg]

    # measured overlaps (red, higher opacity)
    for p in wear.get("pairs", []):
        om = p.get("overlap_mask")
        if om is None:
            continue
        oc = vis.copy(); oc[om > 0] = OVER
        b2 = cv2.addWeighted(vis, 0.30, oc, 0.70, 0)
        vis[om > 0] = b2[om > 0]

    # outlines for crispness
    for sd in all_sides.values():
        poly = sd.get("area_A_poly")
        if poly is not None and len(poly) >= 3:
            cv2.polylines(vis, [poly.astype(np.int32)], True, BLUE, 2, cv2.LINE_AA)
    for c in wire.get("circles", []):
        cv2.circle(vis, (int(c["center"][0]), int(c["center"][1])),
                   int(round(c["radius"])), GREEN, 2, cv2.LINE_AA)

    # per-interlock wear labels
    for p in wear.get("pairs", []):
        c = p["circle"]; cx, cy = int(c["center"][0]), int(c["center"][1])
        cv2.putText(vis, f"{p['wear_pct']:.1f}%", (cx - 42, cy - int(c["radius"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 3, cv2.LINE_AA)
        cv2.putText(vis, f"{p['wear_pct']:.1f}%", (cx - 42, cy - int(c["radius"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, OVER, 2, cv2.LINE_AA)

    # banner
    trig = wear.get("trigger") or {}
    tflag = "TRIGGERED" if trig.get("triggered") else "NOT TRIGGERED"
    n = trig.get("n_interlocks", len(wear.get("pairs", [])))
    banner = f"d={wear.get('d_mean_px',0):.0f}px  b={wear.get('b_px',0):.0f}px   interlocks={n}"
    cv2.rectangle(vis, (8, 10), (8 + 12 * len(banner), 42), (20, 20, 20), -1)
    cv2.putText(vis, banner, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
    tline = f"[{tflag}] {trig.get('reason', '')}"
    cv2.rectangle(vis, (8, 46), (8 + 11 * len(tline), 76), (20, 20, 20), -1)
    cv2.putText(vis, tline, (14, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                GREEN if trig.get("triggered") else ORANGE, 2, cv2.LINE_AA)
    return vis
