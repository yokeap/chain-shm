"""
offline_test.py
===============
Pipeline test: รัน vertical_link_seg → reconstruction → แสดงผล

Usage
-----
    python3 offline_test.py --image chain.png
    python3 offline_test.py --image chain.png --model sam_b.pt --px-per-mm 12.5
    python3 offline_test.py --image chain.png --skip-sam   # ใช้ simple wire mask แทน SAM

Outputs (ใน debug_seg/)
-----------------------
    mask_full_<stem>.jpg    - full chain mask  (Step 1)
    mask_wire_<stem>.jpg    - horizontal wire mask (Step 2 / fallback)
    mask_vert_<stem>.jpg    - vertical link mask   (Step 3)
    vert_<stem>.jpg         - vertical link overlay (green)
    reconstruction.jpg      - d measurement + arc reconstruction (blue/red)
    report.txt              - text summary
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── ensure src/ is importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from vertical_link_seg import get_full_mask, get_wire_mask_sam, get_vertical_link_mask, draw_result
from reconstruction    import reconstruct_links

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Fallback wire mask (ไม่ใช้ SAM)
# ────────────────────────────────────────────────────────────────────────────

def get_wire_mask_simple(
    gray     : np.ndarray,
    full_mask: np.ndarray,
) -> np.ndarray:
    """
    Simple heuristic wire mask (ไม่ต้องใช้ SAM):
    - หา horizontal band ที่ค่า intensity เฉลี่ยต่ำที่สุด (เหล็กมืด)
      ภายใน full_mask zone เท่านั้น
    - ใช้ Sobel horizontal + threshold เพื่อจำกัดให้แคบ
    """
    h, w = gray.shape

    # mask-weighted row mean (เฉพาะ pixel ที่อยู่บนเนื้อโซ่)
    row_means = []
    for y in range(h):
        row_px = gray[y, full_mask[y, :] > 0]
        row_means.append(float(row_px.mean()) if len(row_px) > 0 else 255.0)
    row_means = np.array(row_means)

    # ช่วง 10% มืดที่สุด → นั่นคือแนว horizontal wire
    thresh = np.percentile(row_means, 15)
    wire_rows = np.where(row_means < thresh)[0]

    wire_mask = np.zeros((h, w), dtype=np.uint8)
    if len(wire_rows) > 0:
        y_min = max(0,     int(wire_rows[0])  - 5)
        y_max = min(h - 1, int(wire_rows[-1]) + 5)
        # เฉพาะ pixel ที่อยู่บน full_mask เท่านั้น
        wire_mask[y_min:y_max + 1, :] = full_mask[y_min:y_max + 1, :]

    return wire_mask


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chain wear analysis: seg → reconstruct → wear %"
    )
    parser.add_argument("--image",      required=True,       help="Input image")
    parser.add_argument("--model",      default="sam_b.pt",  help="SAM model path")
    parser.add_argument("--skip-sam",   action="store_true", help="Use simple wire mask (no SAM)")
    parser.add_argument("--px-per-mm",  type=float, default=1.0, help="Calibration px/mm")
    parser.add_argument("--save-dir",   default="debug_seg", help="Output directory")
    args = parser.parse_args()

    t0       = time.time()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img  = cv2.imread(args.image)
    if img is None:
        logger.error(f"Cannot open image: {args.image}")
        sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stem = Path(args.image).stem

    # ── Step 1: Full chain mask ─────────────────────────────────────────────
    logger.info("─── Step 1: Wiener + Otsu → full chain mask")
    full_mask = get_full_mask(gray)
    cv2.imwrite(str(save_dir / f"mask_full_{stem}.jpg"), full_mask)

    # ── Step 2: Wire mask ───────────────────────────────────────────────────
    if args.skip_sam:
        logger.info("─── Step 2: Simple heuristic wire mask (SAM skipped)")
        wire_mask = get_wire_mask_simple(gray, full_mask)
    else:
        logger.info("─── Step 2: SAM → horizontal wire mask")
        wire_mask = get_wire_mask_sam(img, full_mask, args.model)
    cv2.imwrite(str(save_dir / f"mask_wire_{stem}.jpg"), wire_mask)

    # ── Step 3: Vertical link mask ──────────────────────────────────────────
    logger.info("─── Step 3: Subtract wire → vertical link mask")
    vert_mask = get_vertical_link_mask(full_mask, wire_mask)
    cv2.imwrite(str(save_dir / f"mask_vert_{stem}.jpg"), vert_mask)

    # Green overlay
    vis_seg = draw_result(img, vert_mask, wire_mask)
    cv2.imwrite(str(save_dir / f"vert_{stem}.jpg"), vis_seg)

    # ── Step 4: Reconstruction + wear measurement ───────────────────────────
    logger.info("─── Step 4: Reconstruction → d measurement + arc fit")
    results, vis_recon = reconstruct_links(
        image     = img,
        vert_mask = vert_mask,
        wire_mask = wire_mask,
        px_per_mm = args.px_per_mm,
        debug_dir = str(save_dir),
    )
    cv2.imwrite(str(save_dir / f"recon_{stem}.jpg"), vis_recon)

    elapsed = time.time() - t0

    # ── Report ──────────────────────────────────────────────────────────────
    sep = "─" * 65
    lines = [
        sep,
        f"  Image      : {args.image}",
        f"  px / mm    : {args.px_per_mm}",
        f"  SAM used   : {not args.skip_sam}",
        f"  Elapsed    : {elapsed:.1f} s",
        sep,
        f"  {'Link':>4}  {'d (px)':>10}  {'d0 (px)':>10}"
        f"  {'d (mm)':>9}  {'d0 (mm)':>9}  {'Wear %':>8}",
        sep,
    ]
    for r in results:
        lines.append(
            f"  {r['link_id']+1:>4}  {r['d_px']:>10.1f}  {r['d0_px']:>10.1f}"
            f"  {r['d_mm']:>9.3f}  {r['d0_mm']:>9.3f}  {r['wear_pct']:>7.2f}%"
        )
    lines.append(sep)
    report = "\n".join(lines)
    print("\n" + report + "\n")

    report_path = save_dir / "report.txt"
    report_path.write_text(report + "\n")
    logger.info(f"Report saved: {report_path}")
    logger.info(f"All outputs in: {save_dir}/")


if __name__ == "__main__":
    main()
