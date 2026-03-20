"""
chain_inspector/cli.py
Command-line interface for offline testing.

Usage:
    chain-inspect image.jpg
    chain-inspect image.jpg --no-fastsam --show
    chain-inspect image.jpg --reject-threshold 0.08 --save-dir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.inspector import ChainInspector

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chain wear inspection – offline mode"
    )
    parser.add_argument("images", nargs="+", help="Input image path(s)")
    parser.add_argument(
        "--no-fastsam", action="store_true",
        help="Use adaptive threshold segmentation instead of FastSAM"
    )
    parser.add_argument(
        "--fastsam-variant", choices=["small", "x"], default="small",
        help="FastSAM model variant (default: small)"
    )
    parser.add_argument(
        "--reject-threshold", type=float, default=0.10,
        help="Wear ratio threshold for rejection (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display annotated result window"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save annotated images"
    )
    args = parser.parse_args()

    inspector = ChainInspector(
        use_fastsam      = not args.no_fastsam,
        fastsam_variant  = args.fastsam_variant,
        reject_threshold = args.reject_threshold,
    )

    table = Table(title="Chain Inspection Results")
    table.add_column("File",           style="cyan")
    table.add_column("Wear %",         justify="right")
    table.add_column("Ref (px)",       justify="right")
    table.add_column("Min thick (px)", justify="right")
    table.add_column("Decision",       justify="center")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for img_path in args.images:
        try:
            result = inspector.inspect(img_path)
        except Exception as e:
            logger.error(f"{img_path}: {e}")
            continue

        wr = result.wear_results[0]
        decision = "[bold red]REJECT[/]" if wr.is_rejected else "[bold green]PASS[/]"
        table.add_row(
            Path(img_path).name,
            f"{wr.wear_percent:.2f}",
            f"{wr.reference_thickness_px:.1f}",
            f"{wr.min_local_thickness_px:.1f}",
            decision,
        )

        if wr.is_rejected:
            exit_code = 1

        if args.show and result.annotated_image is not None:
            cv2.imshow(f"Chain Inspection – {Path(img_path).name}", result.annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_dir and result.annotated_image is not None:
            out_path = save_dir / f"annotated_{Path(img_path).name}"
            cv2.imwrite(str(out_path), result.annotated_image)
            logger.info(f"Saved → {out_path}")

    console.print(table)
    sys.exit(exit_code)
