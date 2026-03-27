"""
run_offline_test.py
===================
Pipeline: seg → reconstruct → horizontal wire → area-based wear

Usage:  python3 run_offline_test.py --image chain.png --model sam_b.pt
"""

import argparse, logging, sys, time
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from vertical_link_seg     import get_full_mask, get_wire_mask_sam, get_vertical_link_mask, draw_result
from perspective_correction import correct_tilt
from reconstruction         import reconstruct_links
from horizontal_chain       import model_horizontal_wire, draw_horizontal_wire
from wear_analysis          import compute_wear, draw_full_recon

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def get_wire_mask_simple(gray, full_mask):
    h, w = gray.shape
    rm = [float(gray[y, full_mask[y]>0].mean()) if (full_mask[y]>0).any() else 255.0
          for y in range(h)]
    rm = np.array(rm)
    thr = np.percentile(rm, 15)
    rows = np.where(rm < thr)[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(rows):
        y1, y2 = max(0, int(rows[0])-5), min(h-1, int(rows[-1])+5)
        mask[y1:y2+1] = full_mask[y1:y2+1]
    return mask


def main():
    ap = argparse.ArgumentParser(description="Chain wear analysis")
    ap.add_argument("--image",     required=True)
    ap.add_argument("--model",     default="sam_b.pt")
    ap.add_argument("--skip-sam",  action="store_true")
    ap.add_argument("--px-per-mm", type=float, default=1.0)
    ap.add_argument("--save-dir",  default="debug_seg")
    args = ap.parse_args()

    t0 = time.time()
    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"Cannot open: {args.image}"); sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stem = Path(args.image).stem

    # Step 1
    logger.info("── Step 1: full chain mask")
    full_mask = get_full_mask(gray)
    cv2.imwrite(str(save / f"mask_full_{stem}.jpg"), full_mask)

    # Step 2
    if args.skip_sam:
        logger.info("── Step 2: simple wire mask")
        wire_mask = get_wire_mask_simple(gray, full_mask)
    else:
        logger.info("── Step 2: SAM → wire mask")
        wire_mask = get_wire_mask_sam(img, full_mask, args.model)
    cv2.imwrite(str(save / f"mask_wire_{stem}.jpg"), wire_mask)

    # Step 3
    logger.info("── Step 3: vertical link mask")
    vert_mask = get_vertical_link_mask(full_mask, wire_mask)
    cv2.imwrite(str(save / f"mask_vert_{stem}.jpg"), vert_mask)

    # Step 4
    logger.info("── Step 4: perspective correction")
    try:
        img_rect, mask_rect, tilt_info = correct_tilt(img, vert_mask)
        cv2.imwrite(str(save / f"rect_{stem}.jpg"), img_rect)
    except Exception as e:
        logger.warning(f"Tilt failed ({e})")
        img_rect, mask_rect, tilt_info = img, vert_mask, {}

    # Step 5
    logger.info("── Step 5: reconstruction → d")
    results, vis_recon = reconstruct_links(img_rect, mask_rect)
    cv2.imwrite(str(save / f"recon_{stem}.jpg"), vis_recon)

    # Step 6
    logger.info("── Step 6: horizontal wire → b")
    try:
        hw = model_horizontal_wire(wire_mask)
        vis_hw = draw_horizontal_wire(img, hw, wire_mask)
        cv2.imwrite(str(save / f"hwire_{stem}.jpg"), vis_hw)
    except Exception as e:
        logger.warning(f"H-wire failed ({e})")
        hw = {"segments": [], "b_mean_px": 0}

    # Step 7
    logger.info("── Step 7: wear analysis (area-based)")
    try:
        wear = compute_wear(results, hw, image_shape=img.shape[:2],
                            vert_mask=full_mask)
    except Exception as e:
        logger.warning(f"Wear failed ({e})")
        wear = {"pairs": [], "d_mean_px": 0, "b_mean_px": 0,
                "wear_pct_left": 0, "wear_pct_right": 0}

    # Step 8
    logger.info("── Step 8: full_recon overlay")
    vis_full = draw_full_recon(img, results, hw, wear, vert_mask=full_mask)
    cv2.imwrite(str(save / f"full_recon_{stem}.jpg"), vis_full)

    elapsed = time.time() - t0

    # ── Report ──
    sep = "─" * 52
    d = results[0].get("d_mean_px", 0) if results else 0
    b = hw.get("b_mean_px", 0)
    wl = wear.get("wear_pct_left", 0)
    wr = wear.get("wear_pct_right", 0)

    lines = [
        sep,
        f"  Image   : {args.image}",
        f"  Elapsed : {elapsed:.1f}s",
        sep,
        f"  d (vertical)   = {d:.1f} px",
        f"  b (horizontal) = {b:.1f} px",
        sep,
        f"  Wear (area-based)",
    ]
    for p in wear.get("pairs", []):
        lines.append(
            f"    {p['side']:>5}: {p['wear_pct']:.1f}%  "
            f"({p['interf_area']}px²/{p['cs_area']}px²)"
        )
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report + "\n")
    (save / "report.txt").write_text(report + "\n")
    logger.info(f"Done → {save}/")


if __name__ == "__main__":
    main()