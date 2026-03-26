"""
wire_analyser.py
================
Column-wise wire diameter measurement for chain wear inspection.

Physical rationale
------------------
Chain wear occurs primarily at the pin/contact zones where links interlock.
The worn cross-section appears as a **local minimum in wire diameter** along
the horizontal axis of the straight (horizontal) wire segment.

Algorithm
---------
1. Crop a horizontal strip around the image centre (wire is always here).
2. For each column x, find the topmost and bottommost foreground pixel
   within the strip → diameter(x) = bottom_y − top_y  [pixels].
3. Apply Savitzky-Golay smoothing to suppress edge noise.
4. Identify "straight segment" columns: those where the wire is visible
   across at least `min_fill_ratio` of the strip height.
5. Reference diameter  = 95th-percentile of straight-segment diameters
   (robust estimator of un-worn wire cross-section).
6. Pin/contact zone    = region where diameter < reject_limit.
7. Wear index WI       = (d_ref − d_min) / d_ref  ∈ [0, 1].
8. Reject if WI > reject_threshold (default 0.10 → 10 %).

Citation anchor
---------------
The perpendicular cross-section approach follows ISO 4347:2021 §6 which
defines chain wire diameter as measured perpendicular to the wire axis.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import savgol_filter
from loguru import logger


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class WireResult:
    """Measurement result for one frame."""

    # Core measurements
    ref_diameter_px: float        # robust reference (un-worn) diameter
    min_diameter_px: float        # minimum measured diameter (pin zone)
    wear_index: float             # (ref - min) / ref  ∈ [0, 1]
    is_rejected: bool

    # Profiles (for plotting / paper figures)
    x_coords: np.ndarray          = field(repr=False)
    raw_diameters: np.ndarray     = field(repr=False)
    smooth_diameters: np.ndarray  = field(repr=False)

    # Strip geometry (for annotation)
    strip_top: int  = 0
    strip_bot: int  = 0

    @property
    def wear_percent(self) -> float:
        return self.wear_index * 100.0

    @property
    def reject_limit_px(self) -> float:
        return self.ref_diameter_px * (1.0 - 0.10)   # fixed 10% for display


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class WireAnalyser:
    """
    Measures wire diameter column-by-column across the horizontal wire
    segment in the centre of the frame.

    Parameters
    ----------
    strip_center   : float  Fractional row position of strip centre (0–1).
    strip_height   : float  Fractional height of strip relative to image.
    min_fill_ratio : float  Min fraction of strip that must be foreground
                            for a column to be included in the profile.
    sg_window      : int    Savitzky-Golay window length (must be odd).
    sg_poly        : int    Savitzky-Golay polynomial order.
    reject_threshold: float Wear index threshold for rejection.
    ref_percentile : float  Percentile used as reference diameter.
    """

    def __init__(
        self,
        strip_center:     float = 0.50,
        strip_height:     float = 0.40,
        min_fill_ratio:   float = 0.20,
        sg_window:        int   = 31,
        sg_poly:          int   = 3,
        reject_threshold: float = 0.10,
        ref_percentile:   float = 95.0,
    ):
        self.strip_center     = strip_center
        self.strip_height     = strip_height
        self.min_fill_ratio   = min_fill_ratio
        self.sg_window        = sg_window
        self.sg_poly          = sg_poly
        self.reject_threshold = reject_threshold
        self.ref_percentile   = ref_percentile

    # ------------------------------------------------------------------
    def analyse(self, mask: np.ndarray) -> WireResult:
        """
        Parameters
        ----------
        mask : np.ndarray  Binary mask, uint8, 255 = chain wire.

        Returns
        -------
        WireResult
        """
        h, w = mask.shape
        binary = (mask > 127).astype(np.uint8)

        # 1. Define horizontal strip around image centre
        half = int(h * self.strip_height / 2)
        cy   = int(h * self.strip_center)
        strip_top = max(0,     cy - half)
        strip_bot = min(h - 1, cy + half)
        strip     = binary[strip_top:strip_bot, :]
        strip_h   = strip_bot - strip_top

        logger.debug(f"Strip rows {strip_top}–{strip_bot}  ({strip_h}px tall)")

        # 2. Column-wise diameter
        x_list, d_list = [], []
        for x in range(w):
            col = strip[:, x]
            fg  = np.where(col > 0)[0]

            if len(fg) < int(strip_h * self.min_fill_ratio):
                continue

            # เพิ่ม: ตรวจสอบว่า wire เป็น single continuous blob
            # ถ้ามีช่องว่างกลาง (เช่น link gap) ให้ข้ามไป
            gaps = np.diff(fg)
            if np.any(gaps > 5):      # มี gap > 5px = ไม่ใช่ solid wire
                continue

            diameter = float(fg[-1] - fg[0])
            if diameter < 2:
                continue

            x_list.append(x)
            d_list.append(diameter)

        if len(d_list) < self.sg_window + 2:
            logger.warning(
                f"Only {len(d_list)} valid columns found – "
                "check strip_center / strip_height settings"
            )
            # Return a degenerate result rather than crash
            x_arr = np.array(x_list, dtype=np.float32)
            d_arr = np.array(d_list, dtype=np.float32)
            ref   = float(np.percentile(d_arr, self.ref_percentile)) if len(d_arr) else 1.0
            mn    = float(np.min(d_arr)) if len(d_arr) else 0.0
            wi    = max(0.0, (ref - mn) / ref)
            return WireResult(
                ref_diameter_px=ref, min_diameter_px=mn,
                wear_index=wi, is_rejected=wi > self.reject_threshold,
                x_coords=x_arr, raw_diameters=d_arr, smooth_diameters=d_arr,
                strip_top=strip_top, strip_bot=strip_bot,
            )

        x_arr = np.array(x_list, dtype=np.float32)
        d_arr = np.array(d_list, dtype=np.float32)

        # 3. Savitzky-Golay smoothing
        win = self.sg_window if self.sg_window % 2 == 1 else self.sg_window + 1
        win = min(win, len(d_arr) - 1 if len(d_arr) % 2 == 0 else len(d_arr))
        win = win if win % 2 == 1 else win - 1
        smooth = savgol_filter(d_arr, window_length=win, polyorder=self.sg_poly)
        smooth = np.clip(smooth, 0, None)

        # 4. Reference = robust upper percentile (un-worn sections)
        ref_diameter = float(np.percentile(smooth, self.ref_percentile))

        # 5. Minimum diameter (pin/contact zone)
        min_diameter = float(np.min(smooth))

        # 6. Wear index
        wear_index = max(0.0, (ref_diameter - min_diameter) / ref_diameter)
        is_rejected = wear_index > self.reject_threshold

        logger.info(
            f"Ref={ref_diameter:.1f}px  "
            f"Min={min_diameter:.1f}px  "
            f"Wear={wear_index*100:.2f}%  "
            f"{'REJECT' if is_rejected else 'PASS'}"
        )

        return WireResult(
            ref_diameter_px  = ref_diameter,
            min_diameter_px  = min_diameter,
            wear_index       = wear_index,
            is_rejected      = is_rejected,
            x_coords         = x_arr,
            raw_diameters    = d_arr,
            smooth_diameters = smooth,
            strip_top        = strip_top,
            strip_bot        = strip_bot,
        )


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class WireAnnotator:

    PASS_COLOR   = (0, 220, 60)
    REJECT_COLOR = (0, 60, 220)
    STRIP_COLOR  = (0, 200, 255)
    PROFILE_COLOR= (255, 200, 0)

    def draw(self, image: np.ndarray, result: WireResult) -> np.ndarray:
        vis = image.copy()
        h, w = vis.shape[:2]

        # Draw measurement strip boundary
        cv2.line(vis, (0, result.strip_top), (w, result.strip_top),
                 self.STRIP_COLOR, 1, cv2.LINE_AA)
        cv2.line(vis, (0, result.strip_bot), (w, result.strip_bot),
                 self.STRIP_COLOR, 1, cv2.LINE_AA)

        # Draw diameter profile as polyline inside the strip
        strip_mid = (result.strip_top + result.strip_bot) // 2
        if len(result.x_coords) > 1:
            pts = []
            for x, d in zip(result.x_coords, result.smooth_diameters):
                y = int(strip_mid - d / 2)   # top of wire at this column
                pts.append((int(x), y))
            pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts_arr], False, self.PROFILE_COLOR, 2, cv2.LINE_AA)

        # Status banner
        c = self.REJECT_COLOR if result.is_rejected else self.PASS_COLOR
        label = (
            f"REJECT  Wear={result.wear_percent:.1f}%"
            if result.is_rejected
            else f"PASS    Wear={result.wear_percent:.1f}%"
        )
        cv2.rectangle(vis, (0, 0), (460, 48), c, -1)
        cv2.putText(vis, label, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        info = (f"Ref={result.ref_diameter_px:.1f}px  "
                f"Min={result.min_diameter_px:.1f}px")
        cv2.putText(vis, info, (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        return vis
