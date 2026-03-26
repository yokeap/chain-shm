"""
reconstruction.py  v8
=====================
Link #1 (left link) only — top arc + bottom arc

Control points (top arc):
  Outer RED  5 pts: (1)L, (5)L, (3), (5)R, (1)R
  Inner BLUE 5 pts: (2)L, (6)L, (4), (6)R, (2)R

Control points (bottom arc) — mirror of top:
  Outer RED  5 pts: (1)L, (5)L, (3), (5)R, (1)R  (outer = bottommost edge)
  Inner BLUE 5 pts: (2)L, (6)L, (4), (6)R, (2)R  (inner = topmost edge)

  (1) tip corner  (blob height ≈ 0 at edge)
  (2) wire-contact boundary  (inner edge flat-zone boundary)
  (3) outer apex  (x_mid of (1)L.x → (1)R.x)
  (4) inner mid   (same x as (3), opposite edge)
  (5) outer at x of (2)
  (6) inner at x = midpoint((2).x, (3).x)

  d = |inner(x_apex) − outer(x_apex)|  (top arc)
"""

from __future__ import annotations
import argparse, json, logging, math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# A.  Blob pairing — return only the FIRST (leftmost) pair
# ══════════════════════════════════════════════════════════════════

def _pair_blobs(mask: np.ndarray) -> List[Dict]:
    lbl, n = scipy_label(mask > 0)
    blobs = []
    for i in range(1, n + 1):
        comp = (lbl == i).astype(np.uint8) * 255
        if comp.sum() // 255 < 2000:
            continue
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        blobs.append({
            "comp": comp,
            "y1": int(rows[0]),  "y2": int(rows[-1]),
            "x1": int(cols[0]),  "x2": int(cols[-1]),
            "cy": float(rows.mean()),
            "cx": float(cols.mean()),
        })
    if not blobs:
        return []
    med_cy = np.median([b["cy"] for b in blobs])
    tops = sorted([b for b in blobs if b["cy"] < med_cy],  key=lambda b: b["cx"])
    bots = sorted([b for b in blobs if b["cy"] >= med_cy], key=lambda b: b["cx"])

    pairs, used_bot = [], set()
    for t in tops:
        best, best_ov = None, 0.0
        for bi, b in enumerate(bots):
            if bi in used_bot:
                continue
            ov    = max(0, min(t["x2"], b["x2"]) - max(t["x1"], b["x1"]))
            span  = min(t["x2"] - t["x1"], b["x2"] - b["x1"])
            ratio = ov / span if span > 0 else 0.0
            if ratio > best_ov:
                best_ov, best = ratio, bi
        if best is not None and best_ov > 0.3:
            b = bots[best]
            used_bot.add(best)
            pairs.append({
                "top": t["comp"], "bot": b["comp"],
                "x1" : max(t["x1"], b["x1"]),
                "x2" : min(t["x2"], b["x2"]),
                "top_x1": t["x1"], "top_x2": t["x2"],
                "top_y1": t["y1"], "bot_y2": b["y2"],
                "cx": t["cx"],
            })
    # sort by cx → leftmost first
    pairs.sort(key=lambda p: p["cx"])
    return pairs[:1]   # ← ONLY Link #1


# ══════════════════════════════════════════════════════════════════
# B.  Edge scan
# ══════════════════════════════════════════════════════════════════

def _scan_edges(comp, x1, x2):
    W = comp.shape[1]
    xs, tops, bots = [], [], []
    for x in range(max(0, x1), min(W, x2 + 1)):
        px = np.where(comp[:, x] > 0)[0]
        if len(px):
            xs.append(x)
            tops.append(int(px[0]))
            bots.append(int(px[-1]))
    return (np.array(xs),
            np.array(tops, dtype=float),
            np.array(bots, dtype=float))


def _find_tip(xs, tops, bots, side, tip_height=6):
    heights = bots - tops
    if side == "left":
        idx = int(np.argmax(heights >= tip_height))
        return max(0, idx - 1)
    else:
        arr = heights[::-1]
        idx_r = int(np.argmax(arr >= tip_height))
        return len(heights) - 1 - max(0, idx_r - 1)


def _nearest_idx(xs, x_target):
    return int(np.argmin(np.abs(xs - x_target)))


# ══════════════════════════════════════════════════════════════════
# C.  Control points — top blob (outer=top edge, inner=bot edge)
# ══════════════════════════════════════════════════════════════════

def _ctrl_top(comp, x1, x2) -> Dict:
    xs, tops, bots = _scan_edges(comp, x1, x2)
    if len(xs) < 10:
        return {}

    li = _find_tip(xs, tops, bots, "left")
    ri = _find_tip(xs, tops, bots, "right")

    # wire contact: inner (bots) drops from max
    sm  = uniform_filter1d(bots, size=min(80, len(bots)//4*2+1))
    contact = np.where(sm <= sm.max() - 8)[0]
    wli = int(contact[0])  if len(contact) else len(xs)//4
    wri = int(contact[-1]) if len(contact) else 3*len(xs)//4

    x_apex = (int(xs[li]) + int(xs[ri])) // 2
    ai  = _nearest_idx(xs, x_apex)
    i6l = _nearest_idx(xs, (int(xs[wli]) + x_apex) // 2)
    i6r = _nearest_idx(xs, (int(xs[wri]) + x_apex) // 2)

    def pt(i, e): return (int(xs[i]), int(tops[i] if e=="top" else bots[i]))

    return {
        "pt1_left" : pt(li,  "top"),  "pt1_right": pt(ri,  "top"),
        "pt5_left" : pt(wli, "top"),  "pt5_right": pt(wri, "top"),
        "pt3"      : pt(ai,  "top"),
        "pt2_left" : pt(wli, "bot"),  "pt2_right": pt(wri, "bot"),
        "pt6_left" : pt(i6l, "bot"),  "pt6_right": pt(i6r, "bot"),
        "pt4"      : pt(ai,  "bot"),
        "x_apex": x_apex, "x2_left": int(xs[wli]), "x2_right": int(xs[wri]),
    }


# ══════════════════════════════════════════════════════════════════
# D.  Control points — bottom blob  (mirror of top, outer↔inner swapped)
# ══════════════════════════════════════════════════════════════════

def _ctrl_bot(comp, x1, x2) -> Dict:
    xs, tops, bots = _scan_edges(comp, x1, x2)
    if len(xs) < 10:
        return {}

    # same tip logic as top
    li = _find_tip(xs, tops, bots, "left")
    ri = _find_tip(xs, tops, bots, "right")

    # wire contact: inner = tops = U-shape
    # (2) = local minimum ของ tops ในแต่ละ half = มุมที่ inner edge โค้งเข้า
    sm   = uniform_filter1d(tops.astype(float), size=30)
    n    = len(xs)
    half = n // 2
    wli  = int(np.argmin(sm[:half]))
    wri  = half + int(np.argmin(sm[half:]))

    x_apex = (int(xs[li]) + int(xs[ri])) // 2
    ai  = _nearest_idx(xs, x_apex)
    i6l = _nearest_idx(xs, (int(xs[wli]) + x_apex) // 2)
    i6r = _nearest_idx(xs, (int(xs[wri]) + x_apex) // 2)

    # outer = bots (bottom edge),  inner = tops (top edge)  — exact mirror of _ctrl_top
    def pt(i, e): return (int(xs[i]), int(tops[i] if e == "top" else bots[i]))

    return {
        "pt1_left" : pt(li,  "bot"),  "pt1_right": pt(ri,  "bot"),
        "pt5_left" : pt(wli, "bot"),  "pt5_right": pt(wri, "bot"),
        "pt3"      : pt(ai,  "bot"),
        "pt2_left" : pt(wli, "top"),  "pt2_right": pt(wri, "top"),
        "pt6_left" : pt(i6l, "top"),  "pt6_right": pt(i6r, "top"),
        "pt4"      : pt(ai,  "top"),
        "x_apex": x_apex, "x2_left": int(xs[wli]), "x2_right": int(xs[wri]),
    }


# ══════════════════════════════════════════════════════════════════
# E.  Polynomial fit (exact, degree = n_pts - 1)
# ══════════════════════════════════════════════════════════════════

def _fit_exact_poly(pts: list) -> Optional[np.ndarray]:
    valid = [p for p in pts if p is not None]
    if len(valid) < 2:
        return None
    xs = np.array([p[0] for p in valid], dtype=float)
    ys = np.array([p[1] for p in valid], dtype=float)
    deg = len(valid) - 1
    try:
        A    = np.vstack([xs**d for d in range(deg, -1, -1)]).T
        coef, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        return coef
    except Exception as e:
        logger.warning(f"poly fit failed: {e}")
        return np.polyfit(xs, ys, deg=min(deg, 4))


def _poly_eval(coef, x):
    return np.polyval(coef, x)


def _measure_d(coef_outer, coef_inner, x_eval):
    if coef_outer is None or coef_inner is None:
        return float("nan")
    return abs(float(_poly_eval(coef_inner, x_eval))
             - float(_poly_eval(coef_outer, x_eval)))


# ══════════════════════════════════════════════════════════════════
# F.  Visualise
# ══════════════════════════════════════════════════════════════════

RED    = (0,   50, 255)
BLUE   = (220,  90,  20)
YELLOW = (0,  220, 255)
PURPLE = (200,  60, 180)
GREEN  = (50,  200,  50)
WHITE  = (255, 255, 255)


def _draw_curve(vis, coef, x1, x2, color, thickness=3):
    if coef is None: return
    h = vis.shape[0]
    xs = np.arange(x1, x2 + 1, 2, dtype=np.float64)
    ys = np.clip(_poly_eval(coef, xs).astype(int), 0, h - 1)
    pts = np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(vis, [pts], False, color, thickness, cv2.LINE_AA)


def _dot(vis, pt, color, r=10, label=None):
    if pt is None: return
    cx, cy = int(pt[0]), int(pt[1])
    cv2.circle(vis, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), r, WHITE,  1, cv2.LINE_AA)
    if label:
        cv2.putText(vis, label, (cx + r + 2, cy - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 220, 30), 2, cv2.LINE_AA)


def _draw_arc(vis, cp, coef_outer, coef_inner, x1, x2):
    """Draw one arc (top or bottom) with all decorations."""
    xap = cp["x_apex"]

    _draw_curve(vis, coef_outer, x1, x2, RED,  3)
    _draw_curve(vis, coef_inner, x1, x2, BLUE, 3)

    # purple baseline at inner-mid y
    if cp.get("pt4"):
        py = cp["pt4"][1]
        cv2.line(vis, (x1, py), (x2, py), PURPLE, 2, cv2.LINE_AA)

    # outer dots (RED)
    for k, lbl in (("pt1_left","1"),("pt5_left","5"),("pt3","3"),
                   ("pt5_right","5"),("pt1_right","1")):
        _dot(vis, cp.get(k), RED, 10, lbl)

    # inner dots (BLUE)
    for k, lbl in (("pt2_left","2"),("pt6_left","6"),("pt4","4"),
                   ("pt6_right","6"),("pt2_right","2")):
        _dot(vis, cp.get(k), BLUE, 10, lbl)

    # green verticals: (5)→(2) and (3)→(4)
    for rk, bk in (("pt5_left","pt2_left"),("pt5_right","pt2_right"),("pt3","pt4")):
        rp, bp = cp.get(rk), cp.get(bk)
        if rp and bp:
            cv2.line(vis, rp, bp, GREEN, 2, cv2.LINE_AA)

    # d line (yellow) at apex
    if coef_outer is not None and coef_inner is not None:
        y_o = int(_poly_eval(coef_outer, xap))
        y_i = int(_poly_eval(coef_inner, xap))
        d_px = abs(y_i - y_o)
        cv2.line(vis, (xap, y_o), (xap, y_i), YELLOW, 3, cv2.LINE_AA)
        cv2.putText(vis, f"d={d_px:.0f}px",
                    (xap + 10, (y_o + y_i) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2, cv2.LINE_AA)


def _draw_results(image, results):
    vis = image.copy()
    for res in results:
        x1t, x2t = res["top_x1"], res["top_x2"]
        x1b, x2b = res["x1"],     res["x2"]

        _draw_arc(vis, res["cp_top"], res["coef_outer_top"],
                  res["coef_inner_top"], x1t, x2t)
        _draw_arc(vis, res["cp_bot"], res["coef_outer_bot"],
                  res["coef_inner_bot"], x1b, x2b)

        # overall info panel
        py = max(30, res["top_y1"] - 14)
        cv2.rectangle(vis, (x1t-4, py-26), (x1t+310, py+8), (20,20,20), -1)
        cv2.putText(vis,
            f"Link #1  top={res['d_top_px']:.0f}  bot={res['d_bot_px']:.0f}  mean={res['d_mean_px']:.0f}  min={res['d_min_px']:.0f}px  diff={res['diff_pct']:.1f}%",
            (x1t+4, py), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
    return vis


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def reconstruct_links(
    image    : np.ndarray,
    vert_mask: np.ndarray,
    px_per_mm: float = 1.0,
) -> Tuple[List[Dict], np.ndarray]:
    _, bw = cv2.threshold(vert_mask, 127, 255, cv2.THRESH_BINARY)
    pairs = _pair_blobs(bw)
    logger.info(f"Processing {len(pairs)} link(s) (left only)")

    results = []
    for lid, pair in enumerate(pairs):
        cp_t = _ctrl_top(pair["top"], pair["top_x1"], pair["top_x2"])
        cp_b = _ctrl_bot(pair["bot"], pair["x1"],     pair["x2"])
        if not cp_t or not cp_b:
            continue

        coef_ot = _fit_exact_poly([cp_t["pt1_left"], cp_t["pt5_left"], cp_t["pt3"],
                                    cp_t["pt5_right"], cp_t["pt1_right"]])
        coef_it = _fit_exact_poly([cp_t["pt2_left"], cp_t["pt6_left"], cp_t["pt4"],
                                    cp_t["pt6_right"], cp_t["pt2_right"]])
        coef_ob = _fit_exact_poly([cp_b["pt1_left"], cp_b["pt5_left"], cp_b["pt3"],
                                    cp_b["pt5_right"], cp_b["pt1_right"]])
        coef_ib = _fit_exact_poly([cp_b["pt2_left"], cp_b["pt6_left"], cp_b["pt4"],
                                    cp_b["pt6_right"], cp_b["pt2_right"]])

        d_top  = _measure_d(coef_ot, coef_it, cp_t["x_apex"])
        d_bot  = _measure_d(coef_ob, coef_ib, cp_b["x_apex"])
        d_mean = (d_top + d_bot) / 2
        d_min  = min(d_top, d_bot)
        diff_pct = abs(d_top - d_bot) / d_mean * 100 if d_mean > 0 else 0

        logger.info(
            f"  Link {lid+1}: d_top={d_top:.1f}px  d_bot={d_bot:.1f}px  "
            f"d_mean={d_mean:.1f}px  d_min={d_min:.1f}px  "
            f"diff={abs(d_top-d_bot):.1f}px ({diff_pct:.1f}%)"
        )

        results.append({
            "link_id"       : lid,
            "x1": pair["x1"], "x2": pair["x2"],
            "top_x1": pair["top_x1"], "top_x2": pair["top_x2"],
            "top_y1": pair["top_y1"],
            "d_top_px" : d_top,   "d_top_mm" : d_top  / px_per_mm,
            "d_bot_px" : d_bot,   "d_bot_mm" : d_bot  / px_per_mm,
            "d_mean_px": d_mean,  "d_mean_mm": d_mean / px_per_mm,
            "d_min_px" : d_min,   "d_min_mm" : d_min  / px_per_mm,
            "diff_pct" : diff_pct,
            # legacy keys
            "d_px": d_mean, "d_mm": d_mean / px_per_mm,
            "cp_top": cp_t, "cp_bot": cp_b,
            "coef_outer_top": coef_ot, "coef_inner_top": coef_it,
            "coef_outer_bot": coef_ob, "coef_inner_bot": coef_ib,
        })

    vis = _draw_results(image, results)
    return results, vis


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask",      required=True)
    parser.add_argument("--image",     required=True)
    parser.add_argument("--px-per-mm", type=float, default=1.0)
    parser.add_argument("--save-dir",  default="debug_seg")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    image     = cv2.imread(args.image)
    vert_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if image is None:     raise FileNotFoundError(args.image)
    if vert_mask is None: raise FileNotFoundError(args.mask)

    results, vis = reconstruct_links(image, vert_mask, px_per_mm=args.px_per_mm)

    stem = Path(args.image).stem
    out  = save_dir / f"recon_{stem}.jpg"
    cv2.imwrite(str(out), vis)
    logger.info(f"Saved: {out}")

    export = []
    for r in results:
        row = {k: v for k, v in r.items()
               if not k.startswith("coef_") and k not in ("cp_top","cp_bot")}
        for key in ("coef_outer_top","coef_inner_top","coef_outer_bot","coef_inner_bot"):
            row[key] = r[key].tolist() if r[key] is not None else None
        export.append(row)
    with open(save_dir / f"recon_{stem}.json", "w") as f:
        json.dump(export, f, indent=2)

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  {'Link':>4}  {'d_top':>8}  {'d_bot':>8}  {'d_mean':>8}  {'d_min':>8}  {'diff%':>7}  (px)")
    print(sep)
    for r in results:
        print(f"  {r['link_id']+1:>4}  {r['d_top_px']:>8.1f}  {r['d_bot_px']:>8.1f}"
              f"  {r['d_mean_px']:>8.1f}  {r['d_min_px']:>8.1f}  {r['diff_pct']:>6.1f}%")
    print(sep)
    print(f"  Note: diff > 5% → check camera tilt or uneven wear")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()