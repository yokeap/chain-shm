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
from link_model              import model_link
from wire_model              import model_wire
from wear_overlap            import compute_wear, draw_overlay

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

    # ── Step 1: segmentation ──
    logger.info("── Step 1: full / wire / vertical masks")
    full_mask = get_full_mask(gray)
    cv2.imwrite(str(save / f"mask_full_{stem}.jpg"), full_mask)
    if args.skip_sam:
        wire_mask = get_wire_mask_simple(gray, full_mask)
    else:
        wire_mask = get_wire_mask_sam(img, full_mask, args.model)
    cv2.imwrite(str(save / f"mask_wire_{stem}.jpg"), wire_mask)
    vert_mask = get_vertical_link_mask(full_mask, wire_mask)
    cv2.imwrite(str(save / f"mask_vert_{stem}.jpg"), vert_mask)

    # ── Step 2: V-correction (tilt → d_top≈d_bot), applied to every layer ──
    logger.info("── Step 2: vertical perspective correction")
    try:
        info = compute_tilt_info(vert_mask)
        map_x, map_y = build_remap(info, (h, w))
        img_v   = _remap_v(img,       map_x, map_y, nearest=False)
        vmask_v = _remap_v(vert_mask,  map_x, map_y)
        wmask_v = _remap_v(wire_mask,  map_x, map_y)
        logger.info(f"  V-correction: d_top={info['d_top']:.1f} d_bot={info['d_bot']:.1f} "
                    f"θ={info['theta_deg']:.2f}°")
    except Exception as e:
        logger.warning(f"  V-correction skipped ({e})")
        img_v, vmask_v, wmask_v = img, vert_mask, wire_mask
    cv2.imwrite(str(save / f"rect_v_{stem}.jpg"), img_v)

    # ── Step 3: H-correction (uniform wire width), same H on every layer ──
    logger.info("── Step 3: horizontal perspective correction")
    try:
        hw_v = model_horizontal_wire(wmask_v)
        img_h, H, _, hinfo = correct_horizontal_perspective(img_v, hw_v, mask=None)
        vmask_h = _warp_h(vmask_v, H, (w, h))
        wmask_h = _warp_h(wmask_v, H, (w, h))
        if hinfo:
            logger.info(f"  H-correction: b_diff={hinfo.get('b_diff', 0):.1f}px "
                        f"({hinfo.get('b_diff_pct', 0):.1f}%)")
    except Exception as e:
        logger.warning(f"  H-correction skipped ({e})")
        img_h, vmask_h, wmask_h = img_v, vmask_v, wmask_v
    cv2.imwrite(str(save / f"rect_h_{stem}.jpg"), img_h)

    # ── Step 4: models on the fully-corrected frame ──
    logger.info("── Step 4: vertical link model (area A)")
    link = model_link(img_h, vmask_h, wire_mask=wmask_h, px_per_mm=args.px_per_mm)

    logger.info("── Step 5: horizontal wire model (circle B)")
    wire = model_wire(wmask_h)

    # ── Step 6: wear = A∩B / A ──
    logger.info("── Step 6: wear (A∩B / A)")
    wear = compute_wear(link, wire, img_h.shape[:2])

    # ── Step 7: overlay + report ──
    vis = draw_overlay(img_h, link, wire, wear)
    cv2.imwrite(str(save / f"wear_overlay_{stem}.jpg"), vis)

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
        f"  Wear = area(A∩B) / area(A)",
    ]
    if wear.get("pairs"):
        for p in wear["pairs"]:
            lines.append(
                f"    {p['side']:>5}: {p['wear_pct']:5.1f}%   "
                f"(A={p['area_A']}px²  B={p['area_B']}px²  A∩B={p['area_overlap']}px²)")
    else:
        lines.append("    (no A/B pairs measured)")
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report + "\n")
    (save / "report.txt").write_text(report + "\n")
    logger.info(f"Done → {save}/")


if __name__ == "__main__":
    main()
