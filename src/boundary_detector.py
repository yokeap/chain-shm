"""
boundary_detector.py
====================
Step 2-4: Detect and fit geometric boundaries for chain wear measurement.

  Step 2 — Blue ellipse  : outer boundary of vertical link
  Step 3 — Red ellipse   : inner void boundary of vertical link
  Step 4 — Green box     : bounding rectangle of horizontal wire

All three boundaries are fitted from the binary mask produced by
BackgroundRemover (segmentation.py Step 1).

Key insight from image analysis
--------------------------------
Column-wise fill analysis reveals three regions:
  - Left  (x=0..~950)    : vertical link (tall fill ~36–56%)
  - Center(x=~950..1400) : horizontal wire (small fill ~19–24%)
  - Right (x=~1400..end) : right vertical link (partial)

The horizontal wire x-range is found automatically by scanning for
columns where fill < wire_fill_threshold.

Usage
-----
    python3 boundary_detector.py --image chain.png
    python3 boundary_detector.py --image chain.png --save-dir debug_boundary/
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EllipseParams:
    """OpenCV ellipse: ((cx, cy), (width, height), angle_deg)"""
    center : tuple[float, float]
    axes   : tuple[float, float]   # (minor, major) full diameters
    angle  : float                 # degrees

    def to_cv2(self) -> tuple:
        return (self.center, self.axes, self.angle)

    def point_at_y(self, y: float) -> float | None:
        """
        Return the RIGHT-MOST x on the ellipse at a given y.
        Used to find where the blue outer ellipse intersects
        the horizontal centre of the wire (the × measurement point).

        Derivation: axis-aligned ellipse equation, then rotate.
        For near-vertical ellipses (angle ≈ 90°) this gives the
        correct right-hand intersection robustly.
        """
        cx, cy   = self.center
        a, b     = self.axes[1] / 2, self.axes[0] / 2   # semi-major, semi-minor
        theta    = np.deg2rad(self.angle)

        # Parametric search: t ∈ [0, 2π], find t where ellipse_y(t) = y
        # ellipse_x(t) = cx + a*cos(t)*cos(θ) - b*sin(t)*sin(θ)
        # ellipse_y(t) = cy + a*cos(t)*sin(θ) + b*sin(t)*cos(θ)
        t_vals = np.linspace(0, 2 * np.pi, 3600)
        ex = cx + a * np.cos(t_vals) * np.cos(theta) - b * np.sin(t_vals) * np.sin(theta)
        ey = cy + a * np.cos(t_vals) * np.sin(theta) + b * np.sin(t_vals) * np.cos(theta)

        # Find where ey crosses y (sign change)
        diff   = ey - y
        signs  = np.sign(diff)
        cross  = np.where(np.diff(signs) != 0)[0]

        if len(cross) == 0:
            return None

        # Interpolate x at each crossing, return rightmost
        x_crossings = []
        for idx in cross:
            t0, t1 = t_vals[idx], t_vals[idx + 1]
            y0, y1 = ey[idx],     ey[idx + 1]
            x0, x1 = ex[idx],     ex[idx + 1]
            frac = (y - y0) / (y1 - y0 + 1e-9)
            x_crossings.append(x0 + frac * (x1 - x0))

        return float(max(x_crossings))


@dataclass
class WireBox:
    """Axis-aligned bounding box of horizontal wire segment."""
    x1     : int
    y1     : int
    x2     : int
    y2     : int
    cy     : float   # vertical centre (subpixel)

    @property
    def width(self)  -> int:   return self.x2 - self.x1
    @property
    def height(self) -> int:   return self.y2 - self.y1
    @property
    def cx(self)     -> float: return (self.x1 + self.x2) / 2


@dataclass
class BoundaryResult:
    """Output of BoundaryDetector.detect()"""
    outer_ellipse  : EllipseParams | None   # blue
    inner_ellipse  : EllipseParams | None   # red
    wire_box       : WireBox       | None   # green
    elapsed_ms     : float         = 0.0

    # Measurement points (purple ×)
    x_green_left   : float | None  = None   # left edge of wire box
    x_blue_right   : float | None  = None   # blue ellipse at wire cy

    @property
    def d_wear_px(self) -> float | None:
        """
        Wear distance in pixels.
        d_wear = x_blue_right − x_green_left
        Positive = worn (blue boundary overlaps into wire)
        Zero     = no wear (boundaries coincide)
        """
        if self.x_green_left is None or self.x_blue_right is None:
            return None
        return float(self.x_blue_right - self.x_green_left)

    @property
    def is_valid(self) -> bool:
        return (
            self.outer_ellipse is not None and
            self.inner_ellipse is not None and
            self.wire_box      is not None
        )


# ---------------------------------------------------------------------------
# BoundaryDetector
# ---------------------------------------------------------------------------

class BoundaryDetector:
    """
    Fits blue/red ellipses (vertical link) and green box (horizontal wire)
    from the binary chain mask.

    Parameters
    ----------
    wire_fill_threshold : float
        Columns with fill < this fraction of image height are classified
        as wire region (not vertical link). Default 0.35.
    wire_min_width : int
        Minimum width of a candidate wire region in pixels. Default 100.
    wire_cx_range  : tuple[float, float]
        Wire centre must lie within this fractional horizontal range
        of the image. Default (0.2, 0.8) — centre 60% of frame.
    smooth_edge_window : int
        Savitzky-Golay window for smoothing wire top/bottom edges.
    """

    def __init__(
        self,
        wire_fill_threshold : float = 0.35,
        wire_min_width      : int   = 100,
        wire_cx_range       : tuple = (0.20, 0.80),
        smooth_edge_window  : int   = 31,
    ):
        self.wire_fill_threshold = wire_fill_threshold
        self.wire_min_width      = wire_min_width
        self.wire_cx_range       = wire_cx_range
        self.smooth_edge_window  = smooth_edge_window

    # ------------------------------------------------------------------
    def detect(self, mask: np.ndarray) -> BoundaryResult:
        """
        Parameters
        ----------
        mask : np.ndarray  uint8 binary mask (255=chain) from BackgroundRemover

        Returns
        -------
        BoundaryResult
        """
        t0   = time.perf_counter()
        h, w = mask.shape

        # Step A: locate horizontal wire region
        wire_box = self._find_wire_box(mask, h, w)

        # Step B: isolate left vertical link
        if wire_box is not None:
            left_mask = mask.copy()
            left_mask[:, wire_box.x1:] = 0
        else:
            # fallback: use left 55% of image
            left_mask = mask.copy()
            left_mask[:, int(w * 0.55):] = 0

        # Step C: fit blue outer ellipse
        outer_ellipse = self._fit_outer_ellipse(left_mask)

        # Step D: fit red inner ellipse (void)
        inner_ellipse = self._fit_inner_ellipse(left_mask)

        # Step E: compute measurement points
        x_green_left = float(wire_box.x1)  if wire_box      else None
        x_blue_right = None
        if outer_ellipse is not None and wire_box is not None:
            x_blue_right = outer_ellipse.point_at_y(wire_box.cy)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"BoundaryDetector | "
            f"outer={'OK' if outer_ellipse else 'FAIL'} | "
            f"inner={'OK' if inner_ellipse else 'FAIL'} | "
            f"wire={'OK' if wire_box else 'FAIL'} | "
            f"{elapsed:.1f}ms"
        )
        if x_green_left is not None and x_blue_right is not None:
            logger.info(
                f"  x_green_left={x_green_left:.1f}  "
                f"x_blue_right={x_blue_right:.1f}  "
                f"d_wear={x_blue_right - x_green_left:.1f}px"
            )

        return BoundaryResult(
            outer_ellipse = outer_ellipse,
            inner_ellipse = inner_ellipse,
            wire_box      = wire_box,
            elapsed_ms    = elapsed,
            x_green_left  = x_green_left,
            x_blue_right  = x_blue_right,
        )

    # ------------------------------------------------------------------
    def _find_wire_box(
        self, mask: np.ndarray, h: int, w: int
    ) -> WireBox | None:
        """
        Locate horizontal wire: columns where vertical fill < threshold.
        Returns axis-aligned bounding box with subpixel top/bottom edges.
        """
        col_fill = np.array(
            [np.count_nonzero(mask[:, x]) for x in range(w)],
            dtype=np.float32,
        )
        threshold_px = h * self.wire_fill_threshold
        is_wire_col  = col_fill < threshold_px

        # Find contiguous wire regions
        regions = []
        in_region, start = False, 0
        for x in range(w):
            if is_wire_col[x] and not in_region:
                in_region, start = True, x
            elif not is_wire_col[x] and in_region:
                in_region = False
                regions.append((start, x - 1))
        if in_region:
            regions.append((start, w - 1))

        # Filter by width and horizontal position
        cx_lo = w * self.wire_cx_range[0]
        cx_hi = w * self.wire_cx_range[1]
        candidates = [
            (s, e) for s, e in regions
            if (e - s) >= self.wire_min_width
            and cx_lo < (s + e) / 2 < cx_hi
        ]
        if not candidates:
            logger.warning("Wire region not found")
            return None

        # Largest candidate = horizontal wire
        s, e = max(candidates, key=lambda r: r[1] - r[0])

        # Subpixel top/bottom edge per column using S-G smoothing
        tops, bots = [], []
        for x in range(s, e + 1):
            col = mask[:, x]
            fg  = np.where(col > 0)[0]
            if len(fg) > 0:
                tops.append(float(fg[0]))
                bots.append(float(fg[-1]))

        if not tops:
            return None

        win = min(self.smooth_edge_window, len(tops) - 1)
        win = win if win % 2 == 1 else win - 1
        if win >= 3:
            tops_s = savgol_filter(tops, win, 3)
            bots_s = savgol_filter(bots, win, 3)
        else:
            tops_s = np.array(tops)
            bots_s = np.array(bots)

        y1 = int(np.min(tops_s))
        y2 = int(np.max(bots_s))
        cy = float(np.median((tops_s + bots_s) / 2))

        logger.info(
            f"  Wire box: x={s}..{e} y={y1}..{y2} "
            f"({e-s}×{y2-y1}px) cy={cy:.1f}"
        )
        return WireBox(x1=s, y1=y1, x2=e, y2=y2, cy=cy)

    # ------------------------------------------------------------------
    def _fit_outer_ellipse(
        self, left_mask: np.ndarray
    ) -> EllipseParams | None:
        """Fit blue ellipse to outer contour of vertical link."""
        contours, _ = cv2.findContours(
            left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            logger.warning("No outer contour found")
            return None

        outer = max(contours, key=cv2.contourArea)
        if len(outer) < 5:
            logger.warning("Outer contour too small for ellipse fit")
            return None

        ell = cv2.fitEllipse(outer)
        return EllipseParams(
            center = ell[0],
            axes   = ell[1],
            angle  = ell[2],
        )

    # ------------------------------------------------------------------
    def _fit_inner_ellipse(
        self, left_mask: np.ndarray
    ) -> EllipseParams | None:
        """
        Fit red ellipse to inner void (hole) of vertical link.
        Uses RETR_CCOMP to find child contours (holes).
        """
        contours, hierarchy = cv2.findContours(
            left_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if hierarchy is None or not contours:
            logger.warning("No inner contour found")
            return None

        hier = hierarchy[0]
        # Inner void = contour that has a parent (hierarchy[i][3] >= 0)
        inner_candidates = [
            contours[i]
            for i in range(len(contours))
            if hier[i][3] >= 0
            and cv2.contourArea(contours[i]) > 5000
        ]
        if not inner_candidates:
            logger.warning("No inner void contour found")
            return None

        inner = max(inner_candidates, key=cv2.contourArea)
        if len(inner) < 5:
            logger.warning("Inner contour too small for ellipse fit")
            return None

        ell = cv2.fitEllipse(inner)
        return EllipseParams(
            center = ell[0],
            axes   = ell[1],
            angle  = ell[2],
        )


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class BoundaryVisualiser:
    """Draw all three boundaries + measurement points onto image."""

    BLUE_COLOR   = (220, 100, 0)    # BGR blue
    RED_COLOR    = (0,   60, 220)   # BGR red
    GREEN_COLOR  = (0,  180,  60)   # BGR green
    YELLOW_COLOR = (0,  200, 220)   # BGR yellow
    PURPLE_COLOR = (200,  0, 200)   # BGR purple
    FONT         = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        image  : np.ndarray,
        result : BoundaryResult,
    ) -> np.ndarray:
        vis = image.copy()

        # ── Blue outer ellipse ────────────────────────────────────────
        if result.outer_ellipse is not None:
            cv2.ellipse(vis, result.outer_ellipse.to_cv2(),
                        self.BLUE_COLOR, 2, cv2.LINE_AA)

        # ── Red inner ellipse ─────────────────────────────────────────
        if result.inner_ellipse is not None:
            cv2.ellipse(vis, result.inner_ellipse.to_cv2(),
                        self.RED_COLOR, 2, cv2.LINE_AA)

        # ── Green wire box ────────────────────────────────────────────
        if result.wire_box is not None:
            wb = result.wire_box
            cv2.rectangle(vis, (wb.x1, wb.y1), (wb.x2, wb.y2),
                          self.GREEN_COLOR, 2, cv2.LINE_AA)
            # Centre line (for reference)
            cv2.line(vis,
                     (wb.x1, int(wb.cy)),
                     (wb.x2, int(wb.cy)),
                     self.GREEN_COLOR, 1, cv2.LINE_AA)

        # ── Purple × marks ────────────────────────────────────────────
        def draw_cross(img, x, y, size=18, color=(200, 0, 200), thick=3):
            x, y = int(x), int(y)
            cv2.line(img, (x-size,y), (x+size,y), color, thick, cv2.LINE_AA)
            cv2.line(img, (x,y-size), (x,y+size), color, thick, cv2.LINE_AA)

        if result.x_green_left is not None and result.wire_box is not None:
            draw_cross(vis, result.x_green_left, result.wire_box.cy,
                       color=self.PURPLE_COLOR)

        if result.x_blue_right is not None and result.wire_box is not None:
            draw_cross(vis, result.x_blue_right, result.wire_box.cy,
                       color=self.PURPLE_COLOR)

        # ── Yellow measurement arrow ──────────────────────────────────
        if (result.x_green_left is not None and
                result.x_blue_right is not None and
                result.wire_box is not None):
            cy = int(result.wire_box.cy)
            x1 = int(result.x_green_left)
            x2 = int(result.x_blue_right)
            cv2.arrowedLine(vis, (x2, cy), (x1, cy),
                            self.YELLOW_COLOR, 3, cv2.LINE_AA, tipLength=0.15)
            cv2.arrowedLine(vis, (x1, cy), (x2, cy),
                            self.YELLOW_COLOR, 3, cv2.LINE_AA, tipLength=0.15)
            # d_wear label
            mid_x = (x1 + x2) // 2
            d = result.d_wear_px
            cv2.putText(vis, f"d_wear={d:.1f}px",
                        (mid_x - 80, cy - 20),
                        self.FONT, 0.8, self.YELLOW_COLOR, 2, cv2.LINE_AA)

        # ── Info banner ───────────────────────────────────────────────
        lines = [
            f"outer={'OK' if result.outer_ellipse else 'FAIL'}  "
            f"inner={'OK' if result.inner_ellipse else 'FAIL'}  "
            f"wire={'OK' if result.wire_box else 'FAIL'}  "
            f"{result.elapsed_ms:.0f}ms",
        ]
        if result.d_wear_px is not None:
            lines.append(f"d_wear = {result.d_wear_px:.1f} px")

        cv2.rectangle(vis, (0, 0), (700, 30 + 30 * len(lines)), (30, 30, 30), -1)
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (10, 26 + 30 * i),
                        self.FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        return vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chain boundary detection — Steps 2, 3, 4"
    )
    parser.add_argument("--image",    nargs="+", required=True)
    parser.add_argument("--save-dir", default="debug_boundary")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from segmentation import BackgroundRemover
    remover  = BackgroundRemover()
    detector = BoundaryDetector()
    vis      = BoundaryVisualiser()

    for img_path in args.image:
        img_path = Path(img_path)
        image    = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Cannot read: {img_path}")
            continue

        # Step 1: background removal
        bg_result = remover.remove(image)

        # Steps 2-4: boundary detection
        result = detector.detect(bg_result.mask)

        # Visualise
        out = vis.draw(image, result)
        cv2.imwrite(str(save_dir / f"boundary_{img_path.stem}.jpg"), out)

        print(
            f"\n{'─'*54}\n"
            f"  Image   : {img_path.name}\n"
        )
        if result.outer_ellipse:
            e = result.outer_ellipse
            print(f"  Blue  (outer): center=({e.center[0]:.0f},{e.center[1]:.0f})"
                  f"  axes=({e.axes[0]:.0f},{e.axes[1]:.0f})  angle={e.angle:.1f}°")
        if result.inner_ellipse:
            e = result.inner_ellipse
            print(f"  Red   (inner): center=({e.center[0]:.0f},{e.center[1]:.0f})"
                  f"  axes=({e.axes[0]:.0f},{e.axes[1]:.0f})  angle={e.angle:.1f}°")
        if result.wire_box:
            wb = result.wire_box
            print(f"  Green (wire) : x={wb.x1}..{wb.x2}  y={wb.y1}..{wb.y2}"
                  f"  cy={wb.cy:.1f}")
        if result.d_wear_px is not None:
            print(f"\n  d_wear = {result.d_wear_px:.1f} px")
        print(f"{'─'*54}")


if __name__ == "__main__":
    main()
