"""
segmentation.py
===============
Chain segmentation pipeline for wear inspection.

Two-stage approach:
  Stage 1 — BackgroundRemover  : Otsu threshold + morphology
                                  separates chain from white background
  Stage 2 — ChainSegmenter     : YOLOv8n-seg (TensorRT) detects
                                  individual link objects within the mask

Usage
-----
    python3 segmentation.py --image chain.png
    python3 segmentation.py --image chain.png --model chain_seg_best.engine
    python3 segmentation.py --image chain.png --no-yolo   # threshold only
    python3 segmentation.py --image chain.png --benchmark
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label, regionprops

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — Background Removal (Otsu + Morphology)
# ---------------------------------------------------------------------------

class BackgroundRemover:
    """
    Separates chain (dark) from background (white/light) using
    Otsu's global threshold followed by morphological refinement.

    Method
    ------
    1. Gaussian blur  → suppress specular noise before thresholding
    2. Otsu threshold → automatic, no manual tuning required
       Justified by bimodal histogram: bright background vs dark chain
       Cite: Otsu, N. IEEE Trans. SMC, 1979.
    3. MORPH_CLOSE    → fill small holes (specular reflections)
    4. MORPH_OPEN     → remove small noise blobs
    5. Area filter    → keep only blobs > min_area_ratio of image

    Parameters
    ----------
    blur_ksize      : Gaussian blur kernel size (odd, default 5)
    close_ksize     : Closing kernel size — controls hole filling
    open_ksize      : Opening kernel size — controls noise removal
    close_iterations: Morphological closing iterations
    open_iterations : Morphological opening iterations
    min_area_ratio  : Minimum blob area as fraction of image (default 0.005)
    """

    def __init__(
        self,
        blur_ksize       : int   = 5,
        close_ksize      : int   = 15,
        open_ksize       : int   = 5,
        close_iterations : int   = 3,
        open_iterations  : int   = 1,
        min_area_ratio   : float = 0.005,
    ):
        self.blur_ksize        = blur_ksize
        self.close_ksize       = close_ksize
        self.open_ksize        = open_ksize
        self.close_iterations  = close_iterations
        self.open_iterations   = open_iterations
        self.min_area_ratio    = min_area_ratio

    # ------------------------------------------------------------------
    def remove(self, image: np.ndarray) -> BackgroundResult:
        """
        Parameters
        ----------
        image : np.ndarray  BGR image (H, W, 3) or grayscale (H, W)

        Returns
        -------
        BackgroundResult
        """
        t0 = time.perf_counter()

        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image.copy()
        )
        h, w = gray.shape

        # Step 1: Gaussian blur
        blur = cv2.GaussianBlur(
            gray,
            (self.blur_ksize, self.blur_ksize),
            0,
        )

        # Step 2: Otsu threshold — automatic bimodal separation
        otsu_val, mask_raw = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        # Step 3: Morphological closing — fill specular reflection holes
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.close_ksize, self.close_ksize),
        )
        mask_closed = cv2.morphologyEx(
            mask_raw, cv2.MORPH_CLOSE, k_close,
            iterations=self.close_iterations,
        )

        # Step 4: Morphological opening — remove small noise blobs
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.open_ksize, self.open_ksize),
        )
        mask_clean = cv2.morphologyEx(
            mask_closed, cv2.MORPH_OPEN, k_open,
            iterations=self.open_iterations,
        )

        # Step 5: Area filter — discard blobs smaller than threshold
        min_area = h * w * self.min_area_ratio
        lbl      = label(mask_clean > 0)
        props    = regionprops(lbl)

        mask_final = np.zeros_like(mask_clean)
        blobs_kept = []
        for p in props:
            if p.area > min_area:
                mask_final[lbl == p.label] = 255
                blobs_kept.append(p)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"BackgroundRemover | otsu={otsu_val:.0f} | "
            f"blobs={len(blobs_kept)} | "
            f"coverage={np.count_nonzero(mask_final)/mask_final.size*100:.1f}% | "
            f"{elapsed_ms:.1f}ms"
        )

        return BackgroundResult(
            mask         = mask_final,
            otsu_val     = float(otsu_val),
            blobs        = blobs_kept,
            elapsed_ms   = elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Stage 2 — YOLOv8-seg (link object detection within mask)
# ---------------------------------------------------------------------------

class ChainSegmenter:
    """
    YOLOv8n-seg fine-tuned on chain dataset.
    Detects individual chain link objects within the background-removed image.

    Specs
    -----
    Model   : chain_seg_best.engine (TensorRT FP16)
    Dataset : chain-detection, Roboflow (210 images)
    Metrics : mAP50-mask = 0.978, mAP50-95 = 0.862
    Speed   : ~34ms per frame on Jetson Orin NX (29 FPS)
    """

    def __init__(
        self,
        model_path : str   = "chain_seg_best.engine",
        conf       : float = 0.50,
        warmup     : bool  = True,
    ):
        from ultralytics import YOLO

        logger.info(f"Loading YOLO model: {model_path}")
        t0 = time.perf_counter()
        self.model      = YOLO(model_path, task="segment")
        self.conf       = conf
        self.model_path = model_path
        logger.info(f"Model loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

        if warmup:
            self._warmup()

    def _warmup(self, n: int = 3) -> None:
        logger.info("Warming up YOLO model...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self.model(dummy, verbose=False)
        logger.info("Warm-up complete")

    def detect(self, image: np.ndarray) -> list[YOLODetection]:
        """
        Detect all chain link objects and return their masks + metadata.
        Input image should be the original BGR — YOLO handles its own resize.
        """
        h, w = image.shape[:2]
        t0   = time.perf_counter()

        results = self.model(image, imgsz=640, conf=self.conf, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000

        detections = []
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            xyxys = boxes.xyxy.cpu().numpy()

            for i, (raw_mask, conf, xyxy) in enumerate(
                zip(masks, confs, xyxys)
            ):
                mask_resized = cv2.resize(
                    (raw_mask * 255).astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                detections.append(YOLODetection(
                    mask       = mask_resized,
                    confidence = float(conf),
                    bbox_xyxy  = xyxy.tolist(),
                    index      = i,
                ))

        logger.info(
            f"YOLO | detected={len(detections)} | {elapsed:.1f}ms"
        )
        return detections

    def benchmark(self, image: np.ndarray, n: int = 20) -> dict:
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.model(image, imgsz=640, conf=self.conf, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        stats = {
            "mean_ms": float(np.mean(times)),
            "std_ms" : float(np.std(times)),
            "min_ms" : float(np.min(times)),
            "max_ms" : float(np.max(times)),
            "fps"    : float(1000 / np.mean(times)),
        }
        logger.info(
            f"Benchmark | {stats['mean_ms']:.1f}±{stats['std_ms']:.1f}ms "
            f"| {stats['fps']:.1f} FPS"
        )
        return stats


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class BackgroundResult:
    """Output of BackgroundRemover.remove()"""

    def __init__(
        self,
        mask       : np.ndarray,
        otsu_val   : float,
        blobs      : list,
        elapsed_ms : float,
    ):
        self.mask       = mask        # uint8 (H,W) 0/255
        self.otsu_val   = otsu_val
        self.blobs      = blobs
        self.elapsed_ms = elapsed_ms

    @property
    def coverage(self) -> float:
        return float(np.count_nonzero(self.mask)) / self.mask.size

    @property
    def is_valid(self) -> bool:
        return len(self.blobs) > 0


class YOLODetection:
    """Single detection from ChainSegmenter"""

    def __init__(
        self,
        mask       : np.ndarray,
        confidence : float,
        bbox_xyxy  : list,
        index      : int,
    ):
        self.mask       = mask        # uint8 (H,W) 0/255
        self.confidence = confidence
        self.bbox_xyxy  = bbox_xyxy   # [x1, y1, x2, y2]
        self.index      = index

    @property
    def bbox_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bbox_width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def bbox_height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def aspect_ratio(self) -> float:
        return self.bbox_width / (self.bbox_height + 1e-6)


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class SegmentVisualiser:
    """Draw segmentation results onto image."""

    COLORS = [
        (0, 200, 80),    # green
        (0, 120, 255),   # blue
        (255, 160, 0),   # orange
        (200, 0, 200),   # purple
        (0, 200, 200),   # cyan
    ]
    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    ALPHA = 0.35

    def draw_background(
        self,
        image  : np.ndarray,
        result : BackgroundResult,
    ) -> np.ndarray:
        """Visualise background removal result."""
        vis     = image.copy()
        overlay = vis.copy()
        overlay[result.mask > 0] = (0, 200, 80)
        vis = cv2.addWeighted(vis, 1 - self.ALPHA, overlay, self.ALPHA, 0)

        contours, _ = cv2.findContours(
            result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 200, 80), 2)

        label = (
            f"Otsu={result.otsu_val:.0f}  "
            f"cover={result.coverage*100:.1f}%  "
            f"blobs={len(result.blobs)}  "
            f"{result.elapsed_ms:.0f}ms"
        )
        cv2.rectangle(vis, (0, 0), (620, 44), (0, 140, 60), -1)
        cv2.putText(vis, label, (10, 30), self.FONT, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return vis

    def draw_yolo(
        self,
        image      : np.ndarray,
        detections : list[YOLODetection],
    ) -> np.ndarray:
        """Visualise all YOLO detections with distinct colours."""
        vis = image.copy()
        for det in detections:
            color   = self.COLORS[det.index % len(self.COLORS)]
            overlay = vis.copy()
            overlay[det.mask > 0] = color
            vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)

            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                f"#{det.index} {det.confidence:.2f} "
                f"ar={det.aspect_ratio:.1f}",
                (x1, y1 - 8),
                self.FONT, 0.65, color, 2, cv2.LINE_AA,
            )

        status = (
            f"{len(detections)} detection(s)"
            if detections
            else "NO DETECTION"
        )
        cv2.rectangle(vis, (0, 0), (360, 44), (30, 30, 30), -1)
        cv2.putText(vis, status, (10, 30), self.FONT, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chain segmentation — background removal + YOLO"
    )
    parser.add_argument("--image",     nargs="+", required=True)
    parser.add_argument("--model",     default="chain_seg_best.engine")
    parser.add_argument("--conf",      type=float, default=0.50)
    parser.add_argument("--no-yolo",   action="store_true",
                        help="Run background removal only (no YOLO)")
    parser.add_argument("--save-dir",  default="debug_seg")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialise models
    remover = BackgroundRemover()
    segmenter = None
    if not args.no_yolo:
        try:
            segmenter = ChainSegmenter(
                model_path=args.model,
                conf=args.conf,
            )
        except Exception as e:
            logger.warning(f"YOLO unavailable ({e}) — running threshold only")

    vis = SegmentVisualiser()

    for img_path in args.image:
        img_path = Path(img_path)
        image    = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Cannot read: {img_path}")
            continue

        stem = img_path.stem

        # ── Stage 1: Background removal ──────────────────────────────
        bg_result = remover.remove(image)

        out_bg   = vis.draw_background(image, bg_result)
        out_mask = bg_result.mask

        cv2.imwrite(str(save_dir / f"bg_{stem}.jpg"),   out_bg)
        cv2.imwrite(str(save_dir / f"mask_{stem}.jpg"), out_mask)

        print(
            f"\n{'─'*52}\n"
            f"  Image      : {img_path.name}\n"
            f"  Otsu val   : {bg_result.otsu_val:.0f}\n"
            f"  Coverage   : {bg_result.coverage*100:.1f}%\n"
            f"  Blobs      : {len(bg_result.blobs)}\n"
            f"  Time       : {bg_result.elapsed_ms:.1f}ms\n"
        )

        # ── Stage 2: YOLO detection ───────────────────────────────────
        if segmenter is not None:
            detections = segmenter.detect(image)
            out_yolo   = vis.draw_yolo(image, detections)
            cv2.imwrite(str(save_dir / f"yolo_{stem}.jpg"), out_yolo)

            print(f"  Detections : {len(detections)}")
            for d in detections:
                print(
                    f"    #{d.index} conf={d.confidence:.3f} "
                    f"aspect={d.aspect_ratio:.1f} "
                    f"bbox=[{','.join(f'{v:.0f}' for v in d.bbox_xyxy)}]"
                )

        print(f"{'─'*52}")

    # ── Benchmark ─────────────────────────────────────────────────────
    if args.benchmark and segmenter is not None:
        img    = cv2.imread(args.image[0])
        stats  = segmenter.benchmark(img)
        print(
            f"\n{'─'*52}\n"
            f"  Benchmark (20 frames)\n"
            f"  Mean  : {stats['mean_ms']:.1f}ms\n"
            f"  Std   : {stats['std_ms']:.1f}ms\n"
            f"  FPS   : {stats['fps']:.1f}\n"
            f"{'─'*52}"
        )


if __name__ == "__main__":
    main()