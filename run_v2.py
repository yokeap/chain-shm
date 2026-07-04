"""
run_v2.py
=========
Chain wear inspection — **v2** (12-point A∩B overlap model).

Pipeline:
  seg → V-correction (tilt) → H-correction (horizontal perspective) →
  vertical link model (area A) → wire model (circle B) → wear = A∩B / A

v2 keeps the v9 pipeline (run_offline_test.py) untouched for comparison.

Usage:
  python3 run_v2.py --image ./sample/chain.png --model sam_b.pt
  python3 run_v2.py --image ./sample/chain.png --skip-sam      # no GPU
"""

import argparse, logging, sys, time
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from vertical_link_seg       import get_full_mask, get_wire_mask_sam, get_vertical_link_mask
from perspective_correction  import compute_tilt_info, build_remap
from horizontal_chain        import model_horizontal_wire
from horizontal_correction   import correct_horizontal_perspective
from link_model              import model_link, draw_link
from wire_model              import model_wire, draw_wire
from wear_overlap            import (compute_wear, draw_overlay, draw_full_recon_mask,
                                     draw_recon_overlay, evaluate_trigger)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def get_wire_mask_simple(gray, full_mask):
    """Brightness heuristic wire mask (no GPU) — same as run_offline_test.py."""
    h, w = gray.shape
    rm = [float(gray[y, full_mask[y] > 0].mean()) if (full_mask[y] > 0).any() else 255.0
          for y in range(h)]
    rm = np.array(rm)
    thr = np.percentile(rm, 15)
    rows = np.where(rm < thr)[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(rows):
        y1, y2 = max(0, int(rows[0]) - 5), min(h - 1, int(rows[-1]) + 5)
        mask[y1:y2 + 1] = full_mask[y1:y2 + 1]
    return mask


def _remap_v(mask, map_x, map_y, nearest=True):
    return cv2.remap(mask, map_x, map_y,
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _warp_h(mask, H, size):
    return cv2.warpPerspective(mask, H, size, flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _save_fig(base, folder, stem, img, title):
    """Save one captioned stage figure into base/<folder>/<stem>.jpg (report-ready)."""
    d = base / folder
    d.mkdir(parents=True, exist_ok=True)
    if img is None:
        return
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    fig = cv2.copyMakeBorder(img, 46, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(fig, title, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(d / f"{stem}.jpg"), fig)


def main():
    ap = argparse.ArgumentParser(description="Chain wear analysis v2 (A∩B overlap)")
    ap.add_argument("--image",     required=True)
    ap.add_argument("--model",     default="sam_b.pt")
    ap.add_argument("--skip-sam",  action="store_true")
    ap.add_argument("--px-per-mm", type=float, default=1.0)
    ap.add_argument("--save-dir",  default="debug_v2")
    args = ap.parse_args()

    t0 = time.time()
    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"Cannot open: {args.image}"); sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stem = Path(args.image).stem
    h, w = img.shape[:2]

    # Each pipeline stage writes ONE captioned result picture into its own
    # process folder (00_input, 01_segmentation, …) so figures can be pulled
    # straight into the research report.
    _save_fig(save, "00_input", stem, img, f"Input  [{stem}]")

    # ── Step 1: segmentation ──
    logger.info("── Step 1: full / wire / vertical masks")
    full_mask = get_full_mask(gray)
    if args.skip_sam:
        wire_mask = get_wire_mask_simple(gray, full_mask)
    else:
        wire_mask = get_wire_mask_sam(img, full_mask, args.model)
    vert_mask = get_vertical_link_mask(full_mask, wire_mask)
    _save_fig(save, "01_segmentation", f"{stem}_1_full", full_mask, f"1. Segmentation - full chain mask  [{stem}]")
    _save_fig(save, "01_segmentation", f"{stem}_2_wire", wire_mask, f"1. Segmentation - horizontal wire mask  [{stem}]")
    _save_fig(save, "01_segmentation", f"{stem}_3_vert", vert_mask, f"1. Segmentation - vertical link mask  [{stem}]")

    # ── Step 2: V-correction (tilt → d_top≈d_bot), applied to every layer ──
    logger.info("── Step 2: vertical perspective correction")
    vtitle = "2. V-correction (tilt)"
    try:
        info = compute_tilt_info(vert_mask)
        map_x, map_y = build_remap(info, (h, w))
        img_v   = _remap_v(img,       map_x, map_y, nearest=False)
        vmask_v = _remap_v(vert_mask,  map_x, map_y)
        wmask_v = _remap_v(wire_mask,  map_x, map_y)
        vtitle = f"2. V-correction (tilt)  theta={info['theta_deg']:.1f}deg"
        logger.info(f"  V-correction: d_top={info['d_top']:.1f} d_bot={info['d_bot']:.1f} "
                    f"θ={info['theta_deg']:.2f}°")
    except Exception as e:
        logger.warning(f"  V-correction skipped ({e})")
        img_v, vmask_v, wmask_v = img, vert_mask, wire_mask
    _save_fig(save, "02_v_correction", stem, img_v, f"{vtitle}  [{stem}]")

    # ── Step 3: H-correction (uniform wire width), same H on every layer ──
    logger.info("── Step 3: horizontal perspective correction")
    htitle = "3. H-correction (horizontal perspective)"
    try:
        hw_v = model_horizontal_wire(wmask_v)
        img_h, H, _, hinfo = correct_horizontal_perspective(img_v, hw_v, mask=None)
        vmask_h = _warp_h(vmask_v, H, (w, h))
        wmask_h = _warp_h(wmask_v, H, (w, h))
        if hinfo:
            htitle = f"3. H-correction  b_diff={hinfo.get('b_diff', 0):.1f}px ({hinfo.get('b_diff_pct', 0):.1f}%)"
            logger.info(f"  H-correction: b_diff={hinfo.get('b_diff', 0):.1f}px "
                        f"({hinfo.get('b_diff_pct', 0):.1f}%)")
    except Exception as e:
        logger.warning(f"  H-correction skipped ({e})")
        img_h, vmask_h, wmask_h = img_v, vmask_v, wmask_v
    _save_fig(save, "03_h_correction", stem, img_h, f"{htitle}  [{stem}]")

    # ── Step 4: vertical link model (area A) ──
    logger.info("── Step 4: vertical link model (area A)")
    link = model_link(img_h, vmask_h, wire_mask=wmask_h, px_per_mm=args.px_per_mm)
    _save_fig(save, "04_vertical_link_model", stem, draw_link(vmask_h, link),
              f"4. Vertical link model - 12 points + area A  [{stem}]")

    # ── Step 5: horizontal wire model (circle B) ──
    logger.info("── Step 5: horizontal wire model (circle B)")
    wire = model_wire(wmask_h)
    _save_fig(save, "05_wire_model", stem, draw_wire(img_h, wire, wmask_h),
              f"5. Wire model - circle B + area B  [{stem}]")

    # ── Step 6: wear = A∩B / A ──
    logger.info("── Step 6: wear (A∩B / A)")
    wear = compute_wear(link, wire, img_h.shape[:2])

    # Trigger gate: only a full HORIZONTAL wire link + an inserted cap counts as
    # a valid inspection frame (mirrors the moving-chain capture logic).
    trig = evaluate_trigger(link, wire, wear)
    wear["trigger"] = trig
    if trig["triggered"]:
        logger.info(f"  TRIGGER: measured — {trig['reason']}")
    else:
        logger.warning(f"  TRIGGER: skipped — {trig['reason']}")

    # ── Step 7: wear overlay + opacity overlay + full reconstruction mask ──
    vis = draw_overlay(img_h, link, wire, wear)
    _save_fig(save, "06_wear_overlay", stem, vis,
              f"6. Wear overlay  A n B / A  [{stem}]")
    # opacity overlay of the reconstruction on the real (corrected) image
    recon_ov = draw_recon_overlay(img_h, link, wire, wear)
    _save_fig(save, "07_recon_overlay", stem, recon_ov,
              f"7. Reconstruction overlay (opacity) on real image  [{stem}]")
    recon_mask = draw_full_recon_mask(link, wire, wear, img_h.shape[:2])
    _save_fig(save, "08_full_recon_mask", stem, recon_mask,
              f"8. Full reconstruction mask  [{stem}]")

    elapsed = time.time() - t0
    sep = "─" * 56
    d = wear.get("d_mean_px", 0); b = wear.get("b_px", 0)
    lines = [
        sep,
        f"  Image   : {args.image}",
        f"  Elapsed : {elapsed:.1f}s",
        sep,
        f"  d (vertical, mean) = {d:.1f} px",
        f"  b (horizontal)     = {b:.1f} px",
        sep,
        f"  Trigger : {'TRIGGERED' if trig['triggered'] else 'NOT TRIGGERED'}  "
        f"(wire_full={trig['wire_full']}, interlock={trig['has_interlock']})",
        f"            {trig['reason']}",
        sep,
        f"  Wear = area(A∩B) / area(A)   [raw | parallax-corrected to frame centre]"
        + ("" if trig["triggered"] else "   [advisory — frame not triggered]"),
    ]
    if wear.get("pairs"):
        for p in wear["pairs"]:
            wc = p.get("wear_corr")
            wc_s = f"{wc:5.1f}%" if wc is not None else "  n/a"
            lines.append(
                f"    {p['side']:>5} @x={int(p['circle']['center'][0]):>4}: "
                f"raw={p['wear_pct']:5.1f}%  corr={wc_s}   "
                f"(A={p['area_A']}  B={p['area_B']}  A∩B={p['area_overlap']})")
        km = wear.get("parallax_k"); wcm = wear.get("wear_corr_mean")
        lines.append(sep)
        if wcm is not None:
            lines.append(f"  Parallax slope  : {km*1000:+.2f}%/1000px  (left→right ramp removed)")
            lines.append(f"  WEAR (corrected): {wcm:.1f}%   [frame-centre, parallax-free]")
        else:
            lines.append("  WEAR: single interlock — parallax not separable (raw reported)")
    else:
        lines.append("    (no A/B pairs measured)")
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report + "\n")
    rdir = save / "09_report"; rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"{stem}.txt").write_text(report + "\n")
    logger.info(f"Done → {save}/ (per-stage folders)")


if __name__ == "__main__":
    main()
