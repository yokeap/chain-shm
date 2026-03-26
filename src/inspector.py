"""
chain_inspector/inspector.py
Core pipeline: FastSAM → Skeleton/Distance Transform → Local Thickness → Wear Ratio
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from loguru import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WearResult:
    """Result for a single chain region (link or pin zone)."""
    region_id: int
    reference_thickness_px: float          # expected / un-worn thickness
    min_local_thickness_px: float          # thinnest measured point
    wear_ratio: float                      # (ref - min) / ref  → 0..1
    is_rejected: bool
    thickness_profile: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    skeleton_points: np.ndarray  = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def wear_percent(self) -> float:
        return self.wear_ratio * 100.0


@dataclass
class InspectionResult:
    """Aggregated result for one frame."""
    image_path: str
    overall_pass: bool
    wear_results: list[WearResult]
    annotated_image: Optional[np.ndarray] = field(repr=False, default=None)

    @property
    def max_wear_percent(self) -> float:
        if not self.wear_results:
            return 0.0
        return max(r.wear_percent for r in self.wear_results)


# ---------------------------------------------------------------------------
# Segmentation wrapper (FastSAM via ultralytics)
# ---------------------------------------------------------------------------

class ChainSegmenter:
    """
    YOLOv8n-seg fine-tuned on chain dataset.
    Inference: ~34ms on Jetson Orin NX via TensorRT engine.
    
    Model: chain_seg_best.engine (TensorRT FP16)
    Dataset: chain-detection Roboflow (210 images, mAP50-mask=0.978)
    """

    def __init__(
        self,
        model_path: str = "chain_seg_best.engine",
        conf: float = 0.5,
        task: str = "segment",
    ):
        from ultralytics import YOLO
        import logging
        logger.info(f"Loading segmentation model: {model_path}")
        self.model = YOLO(model_path, task=task)
        self.conf  = conf

        # Warmup — eliminates 49s first-inference JIT penalty
        logger.info("Warming up...")
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.model(dummy, verbose=False)
        logger.info("Ready")

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Returns binary uint8 mask (255=chain, 0=background).
        Selects highest-confidence mask if multiple detected.
        """
        h, w = image.shape[:2]

        results = self.model(
            image,
            imgsz   = 640,
            conf    = self.conf,
            verbose = False,
        )

        if results[0].masks is None:
            logger.warning("No mask detected — returning empty mask")
            return np.zeros((h, w), dtype=np.uint8)

        # เลือก mask ที่มี confidence สูงสุด
        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        
        mask_raw = results[0].masks.data[best_idx].cpu().numpy()
        mask_resized = cv2.resize(
            mask_raw.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        return (mask_resized * 255).astype(np.uint8)

    @staticmethod
    def _keep_largest_blob(mask: np.ndarray) -> np.ndarray:
        from skimage.measure import label, regionprops
        lbl = label(mask > 0)
        if lbl.max() == 0:
            return mask
        props   = regionprops(lbl)
        largest = max(props, key=lambda p: p.area)
        return ((lbl == largest.label) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dynamic thresholding fallback (when FastSAM is not available / for testing)
# ---------------------------------------------------------------------------

class AdaptiveSegmenter:
    """
    Fallback segmenter using Otsu + morphology + flood-fill hole closing.
    Robust against rusty/spotted chain surfaces that fool adaptive threshold.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape

        # Step 1: Gentle blur เพื่อ suppress rust spots
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Step 2: Otsu global threshold (stable กว่า adaptive สำหรับ texture แบบนี้)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Step 3: Large closing เพื่อ bridge รูโหว่จาก rust spots
        k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        closed  = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k_large, iterations=4)

        # Step 4: Flood-fill จาก border เพื่อหา true background
        #         แล้ว invert → รูโหว่ภายในโซ่จะถูก fill
        flood   = closed.copy()
        mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
        for seed in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
            if flood[seed] == 0:
                cv2.floodFill(flood, mask_ff, (seed[1], seed[0]), 128)
        flood[flood == 0] = 255           # interior holes → foreground
        filled = np.where(flood == 128, 0, 255).astype(np.uint8)  # bg → 0

        # Step 5: Open เพื่อกำจัด noise blob เล็กๆ
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k_small, iterations=2)

        return ChainSegmenter._keep_largest_blob(cleaned)

# ---------------------------------------------------------------------------
# Thickness analyser
# ---------------------------------------------------------------------------

class ThicknessAnalyser:
    """
    Computes local thickness along the skeleton using Distance Transform.

    Algorithm:
      1. Distance Transform on mask → each foreground pixel = radius to nearest bg
      2. Skeletonize → centre-line
      3. Sample distance transform along skeleton → thickness = 2 * distance
      4. Smooth with Savitzky-Golay filter
      5. Reference = 95th-percentile of thickness (presumed un-worn sections)
      6. Wear ratio = (reference - min_in_window) / reference
    """

    def __init__(
        self,
        sg_window: int   = 21,    # S-G filter window (must be odd)
        sg_poly:   int   = 3,     # S-G polynomial order
        reject_threshold: float = 0.10,  # 10 % wear
    ):
        self.sg_window = sg_window
        self.sg_poly   = sg_poly
        self.reject_threshold = reject_threshold

    # ------------------------------------------------------------------
    def analyse(self, mask: np.ndarray) -> WearResult:
        """Full analysis on a single binary mask."""
        binary = (mask > 127).astype(np.uint8)

        # 1. Distance transform
        dist = distance_transform_edt(binary).astype(np.float32)

        # 2. Skeletonize
        skel  = skeletonize(binary.astype(bool))
        skel  = remove_small_objects(skel, min_size=20)

        # 3. Extract skeleton points ordered along the chain axis
        pts = self._order_skeleton(skel)
        if len(pts) < self.sg_window + 2:
            logger.warning("Skeleton too short – skipping smoothing")
            raw_profile = dist[skel]
            smooth_profile = raw_profile
        else:
            raw_profile    = dist[pts[:, 0], pts[:, 1]]
            smooth_profile = savgol_filter(raw_profile, self.sg_window, self.sg_poly)
            smooth_profile = np.clip(smooth_profile, 0, None)

        # Thickness = 2 × radius (distance transform gives radius)
        thickness = smooth_profile * 2.0

        # 4. Reference = robust upper percentile (un-worn region)
        reference = float(np.percentile(thickness, 95))
        if reference < 1e-3:
            logger.error("Reference thickness near zero – mask may be empty")
            reference = 1.0

        # 5. Wear
        min_thick  = float(np.min(thickness)) if len(thickness) else reference
        wear_ratio = max(0.0, (reference - min_thick) / reference)
        rejected   = wear_ratio > self.reject_threshold

        return WearResult(
            region_id=0,
            reference_thickness_px=reference,
            min_local_thickness_px=min_thick,
            wear_ratio=wear_ratio,
            is_rejected=rejected,
            thickness_profile=thickness,
            skeleton_points=pts,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _order_skeleton(skel: np.ndarray) -> np.ndarray:
        """
        Order skeleton pixels using 8-connectivity graph walk.
        Finds an endpoint (pixel with exactly 1 neighbour) as start,
        then does a depth-first walk staying on connected pixels.
        Returns (N, 2) array of [row, col].
        """
        pts = np.argwhere(skel)
        if len(pts) == 0:
            return pts

        # Build lookup set for O(1) membership test
        pt_set = set(map(tuple, pts.tolist()))

        def neighbours(r: int, c: int):
            return [
                (r + dr, c + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in pt_set
            ]

        # Find an endpoint (degree == 1), or fall back to left-most point
        start = None
        for p in pts:
            if len(neighbours(p[0], p[1])) == 1:
                start = tuple(p)
                break
        if start is None:
            start = tuple(pts[int(np.argmin(pts[:, 1]))])

        # BFS/DFS walk
        ordered = []
        visited = set()
        stack   = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            ordered.append(cur)
            for nb in neighbours(cur[0], cur[1]):
                if nb not in visited:
                    stack.append(nb)

        return np.array(ordered, dtype=np.intp)


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class ResultAnnotator:
    PASS_COLOR   = (0, 220, 60)
    REJECT_COLOR = (0, 60, 220)
    SKEL_COLOR   = (0, 200, 255)

    def draw(self, image: np.ndarray, mask: np.ndarray, result: WearResult) -> np.ndarray:
        vis = image.copy()

        # Overlay mask
        color_mask = np.zeros_like(vis)
        c = self.REJECT_COLOR if result.is_rejected else self.PASS_COLOR
        color_mask[mask > 0] = c
        vis = cv2.addWeighted(vis, 0.7, color_mask, 0.3, 0)

        # Draw skeleton
        if len(result.skeleton_points):
            for r, c_ in result.skeleton_points:
                cv2.circle(vis, (int(c_), int(r)), 1, self.SKEL_COLOR, -1)

        # Status banner
        label_text = (
            f"REJECT  Wear={result.wear_percent:.1f}%"
            if result.is_rejected
            else f"PASS    Wear={result.wear_percent:.1f}%"
        )
        banner_color = self.REJECT_COLOR if result.is_rejected else self.PASS_COLOR
        cv2.rectangle(vis, (0, 0), (420, 48), banner_color, -1)
        cv2.putText(vis, label_text, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        ref_txt = (
            f"Ref={result.reference_thickness_px:.1f}px  "
            f"Min={result.min_local_thickness_px:.1f}px"
        )
        cv2.putText(vis, ref_txt, (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        return vis


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

class ChainInspector:
    """
    Orchestrates: load image → segment → analyse → annotate → return result.
    """

    def __init__(
        self,
        use_fastsam:      bool  = True,
        fastsam_variant:  str   = "small",
        reject_threshold: float = 0.10,
    ):
        if use_fastsam:
            try:
                self.segmenter = ChainSegmenter(variant=fastsam_variant)
            except Exception as e:
                logger.warning(f"FastSAM unavailable ({e}), falling back to adaptive segmenter")
                self.segmenter = AdaptiveSegmenter()
        else:
            self.segmenter = AdaptiveSegmenter()

        self.analyser   = ThicknessAnalyser(reject_threshold=reject_threshold)
        self.annotator  = ResultAnnotator()

    def inspect(self, image_path: str | Path) -> InspectionResult:
        image_path = Path(image_path)
        logger.info(f"Inspecting: {image_path.name}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Segment
        mask   = self.segmenter.segment(image)

        # Analyse
        result = self.analyser.analyse(mask)

        # Annotate
        annotated = self.annotator.draw(image, mask, result)

        return InspectionResult(
            image_path   = str(image_path),
            overall_pass = not result.is_rejected,
            wear_results = [result],
            annotated_image = annotated,
        )
