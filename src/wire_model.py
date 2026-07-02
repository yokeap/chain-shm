"""
wire_model.py  (v2)
===================
Circle-**B** model of the horizontal wire, following the annotation in
``resource/horizontal_mask_annotated.png``:

  point 1     — wire tip (apex of the rounded end)
  points 2L/2R— top / bottom straight edges (red lines), fit left→right
  b           — wire thickness (blue line) = perpendicular gap of the two edges
  circle B    — reconstructed rounded end, Ø = b, tangent at the tip

This is a thin, named adapter over the proven ``horizontal_chain.model_horizontal_wire``
so that all v2 naming lives in one seam; the heavy lifting (RANSAC edges, tip
detection, tangent circle) is reused unchanged.

Usage
-----
    from wire_model import model_wire
    wire = model_wire(wire_mask)
    wire["b_px"]
    wire["circles"]   # list of {side, tip, center, radius, ...}  == area B(s)
"""

from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np

from horizontal_chain import model_horizontal_wire

logger = logging.getLogger(__name__)


def model_wire(wire_mask: np.ndarray) -> Dict:
    """
    Model the horizontal wire → thickness b + circle(s) B.

    Returns
    -------
    dict with:
        b_px      : mean wire thickness
        segments  : raw per-segment dicts from horizontal_chain (edges, tips)
        circles   : flat list of circle-B dicts, one per detected tip:
                    {side, tip(=point 1), center, radius, b_px,
                     slope_top, intercept_top, slope_bot, intercept_bot}
    """
    hw = model_horizontal_wire(wire_mask)

    circles: List[Dict] = []
    for seg in hw.get("segments", []):
        for tip in seg.get("tips", []):
            circles.append({
                "side"         : tip["side"],
                "tip"          : (tip["x_tip"], tip["y_tip"]),   # point 1
                "center"       : tip["center"],
                "radius"       : tip["radius"],
                "b_px"         : seg["b_px"],
                "slope_top"    : seg["slope_top"],
                "intercept_top": seg["intercept_top"],
                "slope_bot"    : seg["slope_bot"],
                "intercept_bot": seg["intercept_bot"],
                "seg_index"    : seg["seg_index"],
            })

    logger.info(f"model_wire: b={hw.get('b_mean_px', 0):.1f}px  circles(B)={len(circles)}")

    return {
        "b_px"    : hw.get("b_mean_px", 0.0),
        "segments": hw.get("segments", []),
        "circles" : circles,
    }
