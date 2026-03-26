"""
component_splitter.py
=====================
Splits the full chain mask into individual components using
column-wise fill analysis — no deep learning required.

Method
------
Otsu mask covers all chain metal as one connected blob.
Scanning column-by-column reveals three fill-level zones:

  High fill  (>35% of height) → vertical link body
  Low fill   (<35% of height) → horizontal wire segment
  Transition                  → occlusion boundary

This gives clean separation between:
  • vertical_link : left oval link (blue/red ellipse target)
  • wire          : horizontal straight segment (green box target)

Speed: ~8ms on Jetson (pure NumPy, no DL inference)

Usage
-----
    python3 component_splitter.py --image chain.png
    python3 component_splitter.py --image chain.png --save-dir debug_split/
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter
from skimage.measure import label, regionprops

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ChainComponents:
    """
    Result of ComponentSplitter.split()

    Attributes
    ----------
    vertical_link_mask : uint8 (H,W) — outer ring of chain link (blue target)
    wire_mask          : uint8 (H,W) — horizontal straight segment (green target)
    wire_x1, wire_x2   : pixel column boundaries of wire region
    wire_y_top         : smoothed top edge of wire (subpixel, array over x)
    wire_y_bot         : smoothed bottom edge of wire (subpixel, array over x)
    wire_cy            : median vertical centre of wire
    elapsed_ms         : processing time
    """

    def __init__(
        self,
        vertical_link_mask : np.ndarray,
        wire_mask          : np.ndarray,
        wire_x1            : int,
        wire_x2            : int,
        wire_y_top         : np.ndarray,
        wire_y_bot         : np.ndarray,
        wire_cy            : float,
        elapsed_ms         : float,
    ):
        self.vertical_link_mask = vertical_link_mask
        self.wire_mask          = wire_mask
        self.wire_x1            = wire_x1
        self.wire_x2            = wire_x2
        self.wire_y_top         = wire_y_top
        self.wire_y_bot         = wire_y_bot
        self.wire_cy            = wire_cy
        self.elapsed_ms         = elapsed_ms

    @property
    def is_valid(self) -> bool:
        return (
            np.count_nonzero(self.vertical_link_mask) > 1000 and
            np.count_nonzero(self.wire_mask)          > 1000
        )

    @property
    def link_contours(self):
        cnts, _ = cv2.findContours(
            self.vertical_link_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        return cnts

    @property
    def wire_contour(self):
        cnts, _ = cv2.findContours(
            self.wire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        return cnts


# ---------------------------------------------------------------------------
# ComponentSplitter
# ---------------------------------------------------------------------------

class ComponentSplitter:
    """
    Splits full chain mask into vertical link and horizontal wire components.

    Parameters
    ----------
    wire_fill_threshold : float
        Columns where vertical fill fraction < this value are classified
        as wire region. Default 0.35 (35% of image height).
    wire_min_width : int
        Minimum width in pixels for a valid wire region. Default 100.
    wire_cx_range : tuple[float, float]
        Wire centre must lie within this fractional range of image width.
        Default (0.20, 0.80).
    sg_window : int
        Savitzky-Golay smoothing window for wire edge profile. Default 31.
    """

    def __init__(
        self,
        wire_fill_threshold : float = 0.35,
        wire_min_width      : int   = 100,
        wire_cx_range       : tuple = (0.20, 0.80),
        sg_window           : int   = 31,
    ):
        self.wire_fill_threshold = wire_fill_threshold
        self.wire_min_width      = wire_min_width
        self.wire_cx_range       = wire_cx_range
        self.sg_window           = sg_window

    # ------------------------------------------------------------------
    def split(self, mask: np.ndarray) -> ChainComponents:
        """
        Parameters
        ----------
        mask : uint8 (H,W) binary mask — 255=chain, 0=background
               from BackgroundRemover

        Returns
        -------
        ChainComponents
        """
        t0   = time.perf_counter()
        h, w = mask.shape

        # ── Step 1: Column-wise fill → find wire region ───────────────
        col_fill  = np.array(
            [np.count_nonzero(mask[:, x]) for x in range(w)],
            dtype=np.float32,
        )
        thr_px   = h * self.wire_fill_threshold
        is_wire  = col_fill < thr_px

        # Find contiguous wire-like regions
        regions, in_r, s = [], False, 0
        for x in range(w):
            if is_wire[x] and not in_r:
                in_r, s = True, x
            elif not is_wire[x] and in_r:
                in_r = False
                regions.append((s, x - 1))
        if in_r:
            regions.append((s, w - 1))

        # Select wire region: centre of image, minimum width
        cx_lo = w * self.wire_cx_range[0]
        cx_hi = w * self.wire_cx_range[1]
        candidates = [
            (s, e) for s, e in regions
            if (e - s) >= self.wire_min_width
            and cx_lo < (s + e) / 2 < cx_hi
        ]

        if not candidates:
            logger.warning("Wire region not found — returning empty components")
            empty = np.zeros_like(mask)
            return ChainComponents(
                vertical_link_mask = empty,
                wire_mask          = empty,
                wire_x1=0, wire_x2=0,
                wire_y_top=np.array([]),
                wire_y_bot=np.array([]),
                wire_cy=float(h/2),
                elapsed_ms=(time.perf_counter()-t0)*1000,
            )

        wire_x1, wire_x2 = max(candidates, key=lambda r: r[1] - r[0])

        # ── Step 2: Split mask ────────────────────────────────────────
        # Vertical link: left of wire region
        link_mask = mask.copy()
        link_mask[:, wire_x1:] = 0

        # Wire: within wire region only
        wire_mask = mask.copy()
        wire_mask[:, :wire_x1]  = 0
        wire_mask[:, wire_x2:]  = 0

        # ── Step 3: Clean up small blobs ──────────────────────────────
        link_mask = self._keep_largest(link_mask)
        wire_mask = self._keep_largest(wire_mask)

        # ── Step 4: Subpixel wire edge profile ───────────────────────
        tops, bots = [], []
        for x in range(wire_x1, wire_x2):
            col = wire_mask[:, x]
            fg  = np.where(col > 0)[0]
            if len(fg) > 5:
                tops.append(float(fg[0]))
                bots.append(float(fg[-1]))
            else:
                tops.append(np.nan)
                bots.append(np.nan)

        tops_arr = np.array(tops)
        bots_arr = np.array(bots)

        # Fill NaN gaps with linear interpolation
        def fill_nan(arr):
            nans = np.isnan(arr)
            if nans.all():
                return arr
            idx = np.arange(len(arr))
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
            return arr

        tops_arr = fill_nan(tops_arr)
        bots_arr = fill_nan(bots_arr)

        # Savitzky-Golay smoothing
        win = min(self.sg_window, len(tops_arr) - 1)
        win = win if win % 2 == 1 else win - 1
        if win >= 5 and not np.isnan(tops_arr).any():
            tops_arr = savgol_filter(tops_arr, win, 3)
            bots_arr = savgol_filter(bots_arr, win, 3)

        wire_cy = float(np.nanmedian((tops_arr + bots_arr) / 2))

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"ComponentSplitter | "
            f"wire=x{wire_x1}..{wire_x2} ({wire_x2-wire_x1}px) | "
            f"link_area={np.count_nonzero(link_mask)} | "
            f"wire_area={np.count_nonzero(wire_mask)} | "
            f"wire_cy={wire_cy:.1f} | "
            f"{elapsed:.1f}ms"
        )

        return ChainComponents(
            vertical_link_mask = link_mask,
            wire_mask          = wire_mask,
            wire_x1            = wire_x1,
            wire_x2            = wire_x2,
            wire_y_top         = tops_arr,
            wire_y_bot         = bots_arr,
            wire_cy            = wire_cy,
            elapsed_ms         = elapsed,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _keep_largest(mask: np.ndarray) -> np.ndarray:
        lbl   = label(mask > 0)
        if lbl.max() == 0:
            return mask
        props   = regionprops(lbl)
        largest = max(props, key=lambda p: p.area)
        return ((lbl == largest.label) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class ComponentVisualiser:

    LINK_COLOR = (220, 80,   0)    # blue — vertical link
    WIRE_COLOR = (0,   180, 60)    # green — horizontal wire
    EDGE_COLOR = (0,   210, 230)   # cyan — wire edges
    FONT       = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        image      : np.ndarray,
        components : ChainComponents,
    ) -> np.ndarray:
        vis = image.copy()

        # Vertical link overlay
        if np.count_nonzero(components.vertical_link_mask) > 0:
            overlay = vis.copy()
            overlay[components.vertical_link_mask > 0] = self.LINK_COLOR
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            cnts, _ = cv2.findContours(
                components.vertical_link_mask,
                cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(vis, cnts, -1, self.LINK_COLOR, 2, cv2.LINE_AA)

        # Wire overlay
        if np.count_nonzero(components.wire_mask) > 0:
            overlay = vis.copy()
            overlay[components.wire_mask > 0] = self.WIRE_COLOR
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            cnts, _ = cv2.findContours(
                components.wire_mask,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(vis, cnts, -1, self.WIRE_COLOR, 2, cv2.LINE_AA)

        # Wire boundary lines (split lines)
        h = vis.shape[0]
        cv2.line(vis, (components.wire_x1, 0),
                 (components.wire_x1, h), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(vis, (components.wire_x2, 0),
                 (components.wire_x2, h), (0, 255, 255), 1, cv2.LINE_AA)

        # Wire top/bottom edge profile
        if len(components.wire_y_top) > 1:
            xs = np.arange(components.wire_x1, components.wire_x2)
            pts_top = np.array(
                [[int(x), int(y)] for x, y in zip(xs, components.wire_y_top)],
                dtype=np.int32,
            )
            pts_bot = np.array(
                [[int(x), int(y)] for x, y in zip(xs, components.wire_y_bot)],
                dtype=np.int32,
            )
            cv2.polylines(vis, [pts_top], False, self.EDGE_COLOR, 2, cv2.LINE_AA)
            cv2.polylines(vis, [pts_bot], False, self.EDGE_COLOR, 2, cv2.LINE_AA)

            # Wire centre line
            cy = int(components.wire_cy)
            cv2.line(vis, (components.wire_x1, cy),
                     (components.wire_x2, cy),
                     self.EDGE_COLOR, 1, cv2.LINE_AA)

        # Banner
        lines = [
            f"link={np.count_nonzero(components.vertical_link_mask)//1000}k px  "
            f"wire={np.count_nonzero(components.wire_mask)//1000}k px  "
            f"{components.elapsed_ms:.0f}ms",
            f"wire_x: {components.wire_x1}..{components.wire_x2}  "
            f"wire_cy: {components.wire_cy:.1f}px",
        ]
        cv2.rectangle(vis, (0, 0), (700, 56), (30, 30, 30), -1)
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (10, 24 + 28*i),
                        self.FONT, 0.78, (255,255,255), 2, cv2.LINE_AA)
        return vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split chain mask into vertical link + horizontal wire"
    )
    parser.add_argument("--image",    nargs="+", required=True)
    parser.add_argument("--save-dir", default="debug_split")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Import BackgroundRemover from segmentation.py
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from segmentation import BackgroundRemover

    remover  = BackgroundRemover()
    splitter = ComponentSplitter()
    vis      = ComponentVisualiser()

    for img_path in args.image:
        img_path = Path(img_path)
        image    = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Cannot read: {img_path}")
            continue

        stem = img_path.stem

        # Step 1: background removal
        bg = remover.remove(image)

        # Step 2: split components
        comps = splitter.split(bg.mask)

        # Save outputs
        out = vis.draw(image, comps)
        cv2.imwrite(str(save_dir / f"split_{stem}.jpg"),        out)
        cv2.imwrite(str(save_dir / f"mask_link_{stem}.jpg"),    comps.vertical_link_mask)
        cv2.imwrite(str(save_dir / f"mask_wire_{stem}.jpg"),    comps.wire_mask)

        print(
            f"\n{'─'*56}\n"
            f"  Image     : {img_path.name}\n"
            f"  Link mask : {np.count_nonzero(comps.vertical_link_mask):,} px\n"
            f"  Wire mask : {np.count_nonzero(comps.wire_mask):,} px\n"
            f"  Wire x    : {comps.wire_x1} → {comps.wire_x2} "
            f"({comps.wire_x2-comps.wire_x1}px)\n"
            f"  Wire cy   : {comps.wire_cy:.1f} px\n"
            f"  Time      : {comps.elapsed_ms:.1f}ms\n"
            f"  Valid     : {comps.is_valid}\n"
            f"{'─'*56}"
        )


if __name__ == "__main__":
    main()
