"""
scripts/run_offline_test.py

Quick offline test that:
  1. Copies the sample image into data/samples/
  2. Runs the adaptive segmenter (no GPU / FastSAM required)
  3. Saves annotated output + thickness plot

Run from project root:
    uv run python run_offline_test.py --image path/to/chain.jpg
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── make sure src/ is importable when running directly ──────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inspector import (
    ChainSegmenter,
    ThicknessAnalyser,
    ResultAnnotator,
)
from loguru import logger


def run(image_path: Path, reject_threshold: float = 0.10) -> None:
    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    # --- Segmentation ---------------------------------------------------
    logger.info("Segmenting (adaptive threshold mode)…")
    seg   = ChainSegmenter()
    mask  = seg.segment(image)

    # --- Thickness analysis ----------------------------------------------
    logger.info("Analysing thickness…")
    analyser = ThicknessAnalyser(reject_threshold=reject_threshold)
    result   = analyser.analyse(mask)

    logger.info(
        f"Reference={result.reference_thickness_px:.1f}px  "
        f"MinThick={result.min_local_thickness_px:.1f}px  "
        f"Wear={result.wear_percent:.2f}%  "
        f"{'REJECT' if result.is_rejected else 'PASS'}"
    )

    # --- Annotate --------------------------------------------------------
    annotator  = ResultAnnotator()
    annotated  = annotator.draw(image, mask, result)

    # --- Save outputs ----------------------------------------------------
    out_dir = Path("data/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = out_dir / f"annotated_{image_path.stem}.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    logger.info(f"Saved annotated image → {annotated_path}")

    # --- Thickness profile plot ------------------------------------------
    if len(result.thickness_profile) > 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        x = np.arange(len(result.thickness_profile))
        ax.plot(x, result.thickness_profile, color="#2196F3", lw=1.5, label="Local thickness (px)")
        ax.axhline(result.reference_thickness_px,
                   color="#4CAF50", ls="--", lw=1.2, label=f"Reference ({result.reference_thickness_px:.1f} px)")
        ax.axhline(result.min_local_thickness_px,
                   color="#F44336", ls=":",  lw=1.2, label=f"Min ({result.min_local_thickness_px:.1f} px)")
        reject_line = result.reference_thickness_px * (1 - reject_threshold)
        ax.axhline(reject_line,
                   color="#FF9800", ls="-.", lw=1.2, label=f"Reject limit ({reject_line:.1f} px)")
        ax.fill_between(x, 0, result.thickness_profile,
                        where=(result.thickness_profile < reject_line),
                        alpha=0.25, color="#F44336", label="Worn zone")
        ax.set_xlabel("Skeleton position")
        ax.set_ylabel("Thickness (pixels)")
        ax.set_title(f"Chain wear profile – {image_path.name}  |  Wear {result.wear_percent:.2f}%")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plot_path = out_dir / f"thickness_profile_{image_path.stem}.png"
        fig.tight_layout()
        fig.savefig(str(plot_path), dpi=120)
        plt.close(fig)
        logger.info(f"Saved thickness plot → {plot_path}")

    # --- Mask visualisation ----------------------------------------------
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_path = out_dir / f"mask_{image_path.stem}.jpg"
    cv2.imwrite(str(mask_path), mask_rgb)

    print("\n" + "─" * 50)
    print(f"  Image   : {image_path.name}")
    print(f"  Ref     : {result.reference_thickness_px:.1f} px")
    print(f"  Min     : {result.min_local_thickness_px:.1f} px")
    print(f"  Wear    : {result.wear_percent:.2f} %")
    print(f"  Decision: {'❌  REJECT' if result.is_rejected else '✅  PASS'}")
    print("─" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline chain inspection test")
    parser.add_argument("--image", required=True, help="Path to chain image")
    parser.add_argument("--reject-threshold", type=float, default=0.10)
    args = parser.parse_args()
    run(Path(args.image), args.reject_threshold)
