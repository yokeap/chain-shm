"""
run_offline_test.py
===================
Pipeline: seg → reconstruct → horizontal wire → wear analysis

Usage
-----
    python3 run_offline_test.py --image chain.png --model sam_b.pt
    python3 run_offline_test.py --image chain.png --skip-sam

Outputs (debug_seg/)
--------------------
    mask_full_<s>.jpg   full chain mask
    mask_wire_<s>.jpg   horizontal wire mask
    mask_vert_<s>.jpg   vertical link mask
    recon_<s>.jpg       d measurement + arcs (detail)
    hwire_<s>.jpg       horizontal wire model (detail)
    full_recon_<s>.jpg  clean combined overlay
    report.txt          summary
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


# ────────────────────────────────────────────────────────────────
def get_wire_mask_simple(gray, full_mask):
    h, w = gray.shape
    row_means = []
    for y in range(h):
        px = gray[y, full_mask[y, :] > 0]
        row_means.append(float(px.mean()) if len(px) else 255.0)
    row_means = np.array(row_means)
    thresh = np.percentile(row_means, 15)
    wire_rows = np.where(row_means < thresh)[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(wire_rows):
        y1 = max(0, int(wire_rows[0]) - 5)
        y2 = min(h-1, int(wire_rows[-1]) + 5)
        mask[y1:y2+1, :] = full_mask[y1:y2+1, :]
    return mask


# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Chain wear analysis pipeline")
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

    # Step 1 — full mask
    logger.info("── Step 1: full chain mask")
    full_mask = get_full_mask(gray)
    cv2.imwrite(str(save / f"mask_full_{stem}.jpg"), full_mask)

    # Step 2 — wire mask
    if args.skip_sam:
        logger.info("── Step 2: simple wire mask (no SAM)")
        wire_mask = get_wire_mask_simple(gray, full_mask)
    else:
        logger.info("── Step 2: SAM → wire mask")
        wire_mask = get_wire_mask_sam(img, full_mask, args.model)
    cv2.imwrite(str(save / f"mask_wire_{stem}.jpg"), wire_mask)

    # Step 3 — vertical link mask
    logger.info("── Step 3: vertical link mask")
    vert_mask = get_vertical_link_mask(full_mask, wire_mask)
    cv2.imwrite(str(save / f"mask_vert_{stem}.jpg"), vert_mask)

    # Step 4 — perspective correction
    logger.info("── Step 4: perspective correction")
    try:
        img_rect, mask_rect, tilt_info = correct_tilt(img, vert_mask)
        cv2.imwrite(str(save / f"rect_{stem}.jpg"), img_rect)
    except Exception as e:
        logger.warning(f"Tilt correction failed ({e})")
        img_rect, mask_rect, tilt_info = img, vert_mask, {}

    # Step 5 — reconstruction (d measurement)
    logger.info("── Step 5: reconstruction → d")
    results, vis_recon = reconstruct_links(img_rect, mask_rect)
    cv2.imwrite(str(save / f"recon_{stem}.jpg"), vis_recon)

    # Step 6 — horizontal wire model (b measurement)
    logger.info("── Step 6: horizontal wire → b")
    try:
        hw = model_horizontal_wire(wire_mask)
        vis_hw = draw_horizontal_wire(img, hw, wire_mask)
        cv2.imwrite(str(save / f"hwire_{stem}.jpg"), vis_hw)
    except Exception as e:
        logger.warning(f"Horizontal wire failed ({e})")
        hw = {"segments": [], "b_mean_px": 0.0}

    # Step 7 — wear analysis (gap measurement)
    logger.info("── Step 7: wear analysis → gap")
    try:
        wear = compute_wear(results, hw)
    except Exception as e:
        logger.warning(f"Wear analysis failed ({e})")
        wear = {"pairs": [], "d_mean_px": 0, "b_mean_px": 0,
                "gap_left": 0, "gap_right": 0,
                "wear_pct_left": 0, "wear_pct_right": 0}

    # Step 8 — full reconstruction overlay (clean)
    logger.info("── Step 8: full_recon overlay")
    vis_full = draw_full_recon(img, results, hw, wear)
    cv2.imwrite(str(save / f"full_recon_{stem}.jpg"), vis_full)

    elapsed = time.time() - t0

    # ── Report ──
    sep = "─" * 52
    d_mean = results[0].get("d_mean_px", 0) if results else 0
    b_mean = hw.get("b_mean_px", 0)
    gl = wear.get("gap_left", 0)
    gr = wear.get("gap_right", 0)
    wl = wear.get("wear_pct_left", 0)
    wr = wear.get("wear_pct_right", 0)

    lines = [
        sep,
        f"  Image   : {args.image}",
        f"  Elapsed : {elapsed:.1f}s",
        sep,
        f"  d (vertical link)    = {d_mean:.1f} px",
        f"  b (horizontal wire)  = {b_mean:.1f} px",
        sep,
        f"  gap left  = {gl:+.1f} px   wear = {wl:.1f}%",
        f"  gap right = {gr:+.1f} px   wear = {wr:.1f}%",
        sep,
    ]

    if tilt_info:
        lines.insert(3, f"  Tilt     : θ = {tilt_info.get('theta_deg',0):.2f}°")

    report = "\n".join(lines)
    print("\n" + report + "\n")
    (save / "report.txt").write_text(report + "\n")
    logger.info(f"Done → {save}/")


if __name__ == "__main__":
    main()