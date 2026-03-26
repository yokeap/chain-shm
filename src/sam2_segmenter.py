"""
sam2_segmenter.py
=================
Chain segmentation using SAM2 (Segment Anything Model 2).

Strategy: point prompt at image centre → SAM2 segments the chain wire
that occupies the centre of frame (wire is always centred by design
of the test station).

Two-pass approach:
  Pass 1: positive point at image centre → gets horizontal wire mask
  Pass 2: positive point at left portion → gets vertical link mask

Usage
-----
    python3 sam2_segmenter.py --image chain.png
    python3 sam2_segmenter.py --image chain.png --save-dir debug_sam2/
    python3 sam2_segmenter.py --image chain.png --benchmark
"""

from __future__ import annotations

import argparse
from email.mime import image
import logging
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM2Segmenter
# ---------------------------------------------------------------------------

class SAM2Segmenter:
    """
    Segments chain components using SAM2 with point prompts.

    Model variants (ultralytics):
      sam2_t.pt  : tiny  (~38MB,  fastest)
      sam2_s.pt  : small (~46MB)
      sam2_b.pt  : base  (~80MB)
      sam2_l.pt  : large (~224MB, most accurate)

    For Jetson Orin NX: sam2_s.pt recommended (speed/accuracy balance).

    Cite
    ----
    Ravi et al., "SAM 2: Segment Anything in Images and Videos",
    Meta AI Research, arXiv:2408.00714, 2024.
    """

    def __init__(
        self,
        model_variant : str   = "sam_b.pt",
        device        : str   = "cuda",
        warmup        : bool  = True,
    ):
        from ultralytics import SAM

        logger.info(f"Loading SAM2 model: {model_variant}")
        t0 = time.perf_counter()
        self.model   = SAM(model_variant)
        self.device  = device
        self.variant = model_variant
        logger.info(f"Loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        logger.info("Warming up SAM2...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(
            dummy,
            points=[[320, 320]],
            labels=[1],
            device=self.device,
            verbose=False,
        )
        logger.info("Warm-up done")

    # ------------------------------------------------------------------
    def segment_wire(self, image: np.ndarray) -> SAM2Result:
        """
        Segment horizontal wire using centre point prompt.
        The wire always passes through the image centre by test station design.
        """
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2

        return self._segment_with_points(
            image,
            pos_points = [[cx, cy]],
            neg_points = [],
            label      = "wire",
        )

    def segment_vertical_link(self, image: np.ndarray) -> SAM2Result:
        h, w = image.shape[:2]

        pos_points = [
            [int(w * 0.10), int(h * 0.22)],   # top arc
            [int(w * 0.02), int(h * 0.50)],   # left side
            [int(w * 0.10), int(h * 0.78)],   # bottom arc
        ]
        neg_points = [
            [int(w * 0.50), int(h * 0.50)],   # horizontal wire
            [int(w * 0.30), int(h * 0.50)],   # background ในรู
        ]

        return self._segment_with_points(
            image,
            pos_points = pos_points,
            neg_points = neg_points,
            label      = "vertical_link",
        )

    def segment_both(
        self, image: np.ndarray
    ) -> tuple["SAM2Result", "SAM2Result"]:
        """
        Segment both horizontal wire and vertical link in one call.
        Returns (wire_result, link_result).
        """
        wire = self.segment_wire(image)
        link = self.segment_vertical_link(image)
        return wire, link

    # ------------------------------------------------------------------
    def _segment_with_points(
        self,
        image      : np.ndarray,
        pos_points : list,
        neg_points : list,
        label      : str,
    ) -> "SAM2Result":
        h, w = image.shape[:2]
        t0   = time.perf_counter()

        all_points = pos_points + neg_points
        all_labels = [1] * len(pos_points) + [0] * len(neg_points)

        results = self.model(
            image,
            points  = all_points,
            labels  = all_labels,
            device  = self.device,
            verbose = False,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if results[0].masks is None or len(results[0].masks) == 0:
            logger.warning(f"SAM2: no mask for {label}")
            return SAM2Result(
                mask       = np.zeros((h, w), dtype=np.uint8),
                score      = 0.0,
                label      = label,
                elapsed_ms = elapsed_ms,
                prompt_pts = pos_points,
            )

        # SAM2 returns masks sorted by score descending — take best
        masks  = results[0].masks.data.cpu().numpy()   # (N, H', W')
        scores = results[0].boxes.conf.cpu().numpy() \
                 if results[0].boxes is not None \
                 else np.ones(len(masks))

        best_idx  = int(np.argmax(scores))
        raw_mask  = masks[best_idx]
        mask_full = cv2.resize(
            (raw_mask * 255).astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        logger.info(
            f"SAM2 [{label}] | score={scores[best_idx]:.3f} | "
            f"cover={np.count_nonzero(mask_full)/mask_full.size*100:.1f}% | "
            f"{elapsed_ms:.0f}ms"
        )

        return SAM2Result(
            mask       = mask_full,
            score      = float(scores[best_idx]),
            label      = label,
            elapsed_ms = elapsed_ms,
            prompt_pts = pos_points,
        )

    # ------------------------------------------------------------------
    def benchmark(self, image: np.ndarray, n: int = 10) -> dict:
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.segment_wire(image)
            times.append((time.perf_counter() - t0) * 1000)
        stats = {
            "mean_ms": float(np.mean(times)),
            "std_ms" : float(np.std(times)),
            "fps"    : float(1000 / np.mean(times)),
        }
        logger.info(
            f"Benchmark | {stats['mean_ms']:.1f}±{stats['std_ms']:.1f}ms "
            f"| {stats['fps']:.1f} FPS"
        )
        return stats


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

class SAM2Result:
    def __init__(
        self,
        mask       : np.ndarray,
        score      : float,
        label      : str,
        elapsed_ms : float,
        prompt_pts : list,
    ):
        self.mask       = mask          # uint8 (H,W) 0/255
        self.score      = score
        self.label      = label
        self.elapsed_ms = elapsed_ms
        self.prompt_pts = prompt_pts

    @property
    def is_valid(self) -> bool:
        return self.score > 0 and np.count_nonzero(self.mask) > 100

    @property
    def coverage(self) -> float:
        return float(np.count_nonzero(self.mask)) / max(self.mask.size, 1)

    def contours(self):
        cnts, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        return cnts


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class SAM2Visualiser:

    COLORS = {
        "wire"          : (0,  180,  60),   # green
        "vertical_link" : (220, 100,   0),  # blue
        "unknown"       : (150, 150, 150),
    }
    ALPHA = 0.35
    FONT  = cv2.FONT_HERSHEY_SIMPLEX

    def draw_single(
        self, image: np.ndarray, result: SAM2Result
    ) -> np.ndarray:
        vis     = image.copy()
        color   = self.COLORS.get(result.label, self.COLORS["unknown"])
        overlay = vis.copy()
        overlay[result.mask > 0] = color
        vis = cv2.addWeighted(vis, 1 - self.ALPHA, overlay, self.ALPHA, 0)

        # Contour
        cv2.drawContours(vis, result.contours(), -1, color, 2, cv2.LINE_AA)

        # Prompt points
        for px, py in result.prompt_pts:
            cv2.drawMarker(vis, (int(px), int(py)),
                           (0, 255, 255), cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)

        # Banner
        txt = (
            f"[{result.label}] score={result.score:.2f}  "
            f"cover={result.coverage*100:.1f}%  {result.elapsed_ms:.0f}ms"
        )
        cv2.rectangle(vis, (0, 0), (680, 44), (30, 30, 30), -1)
        cv2.putText(vis, txt, (10, 30), self.FONT, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return vis

    def draw_both(
        self,
        image       : np.ndarray,
        wire_result : SAM2Result,
        link_result : SAM2Result,
    ) -> np.ndarray:
        vis = image.copy()

        for result in [link_result, wire_result]:
            color   = self.COLORS.get(result.label, self.COLORS["unknown"])
            overlay = vis.copy()
            overlay[result.mask > 0] = color
            vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)
            cv2.drawContours(vis, result.contours(), -1, color, 2, cv2.LINE_AA)
            for px, py in result.prompt_pts:
                cv2.drawMarker(vis, (int(px), int(py)),
                               (0, 255, 255), cv2.MARKER_CROSS, 20, 3)

        total_ms = wire_result.elapsed_ms + link_result.elapsed_ms
        txt = (
            f"wire score={wire_result.score:.2f}  "
            f"link score={link_result.score:.2f}  "
            f"total={total_ms:.0f}ms"
        )
        cv2.rectangle(vis, (0, 0), (720, 44), (30, 30, 30), -1)
        cv2.putText(vis, txt, (10, 30), self.FONT, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2 chain segmentation")
    parser.add_argument("--image",     nargs="+", required=True)
    parser.add_argument("--model", default="sam_b.pt",
                    choices=["sam_b.pt", "sam_l.pt", "mobile_sam.pt"])
    parser.add_argument("--save-dir",  default="debug_sam2")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    seg = SAM2Segmenter(model_variant=args.model)
    vis = SAM2Visualiser()

    for img_path in args.image:
        img_path = Path(img_path)
        image    = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Cannot read: {img_path}")
            continue

        stem = img_path.stem

        # Segment both
        wire_r, link_r = seg.segment_both(image)

        h, w = image.shape[:2]
        debug = image.copy()
        for px, py in [[int(w*0.10),int(h*0.22)],[int(w*0.02),int(h*0.50)],[int(w*0.10),int(h*0.78)]]:
            cv2.drawMarker(debug,(px,py),(0,255,255),cv2.MARKER_CROSS,40,4)
            cv2.putText(debug,f"pos({px},{py})",(px+10,py),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        for px, py in [[int(w*0.50),int(h*0.50)],[int(w*0.30),int(h*0.50)]]:
            cv2.drawMarker(debug,(px,py),(0,0,255),cv2.MARKER_CROSS,40,4)
            cv2.putText(debug,f"neg({px},{py})",(px+10,py),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.imwrite(str(save_dir/f"prompt_points.jpg"), debug)

        # Save outputs
        cv2.imwrite(str(save_dir / f"wire_{stem}.jpg"),
                    vis.draw_single(image, wire_r))
        cv2.imwrite(str(save_dir / f"link_{stem}.jpg"),
                    vis.draw_single(image, link_r))
        cv2.imwrite(str(save_dir / f"both_{stem}.jpg"),
                    vis.draw_both(image, wire_r, link_r))
        cv2.imwrite(str(save_dir / f"mask_wire_{stem}.jpg"), wire_r.mask)
        cv2.imwrite(str(save_dir / f"mask_link_{stem}.jpg"), link_r.mask)

        print(
            f"\n{'─'*54}\n"
            f"  Image   : {img_path.name}\n"
            f"  Wire    : score={wire_r.score:.3f}  "
            f"cover={wire_r.coverage*100:.1f}%  {wire_r.elapsed_ms:.0f}ms\n"
            f"  Link    : score={link_r.score:.3f}  "
            f"cover={link_r.coverage*100:.1f}%  {link_r.elapsed_ms:.0f}ms\n"
            f"{'─'*54}"
        )

    if args.benchmark:
        img   = cv2.imread(args.image[0])
        stats = seg.benchmark(img)
        print(
            f"\n{'─'*54}\n"
            f"  Benchmark ({10} frames)\n"
            f"  Mean : {stats['mean_ms']:.1f}ms\n"
            f"  FPS  : {stats['fps']:.1f}\n"
            f"{'─'*54}"
        )


if __name__ == "__main__":
    main()
