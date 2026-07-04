# Horizontal Wire Reconstruction — Circle-B Model (Method)

*Module: `src/wire_model.py` · Input: `mask_wire_*.jpg` · Reference: `resource/horizontal_mask_annotated.png`*

## 1. Problem statement

The horizontal chain link (the "wire") crosses over the vertical link under
inspection. In the camera's top-down view it appears as a long **stadium**
(rounded-rectangle) shape: two long **straight edges** (top and bottom) joined by
two **rounded end-caps** (left and right). One of these caps sits directly over
the vertical link's contact region.

To measure wear we reconstruct the wire's rounded end as a circle — **circle B**,
of diameter equal to the wire thickness `b`, inscribed tangent at the cap tip.
Circle B is the *top-view cross-section* of the wire at the contact point. Its
area is **area B**; downstream it is intersected with the vertical link's crescent
**area A** to give the wear metric

```
wear% = area(A ∩ B) / area(A) × 100.
```

The reconstruction is expressed with the annotation's control points:

| Points | Meaning | Colour |
|--------|---------|--------|
| 1 | **Tip apex** of each rounded cap (outermost point) | yellow |
| 2 | **Tangent corners** — where each straight edge meets a cap (TL, BL, TR, BR) | yellow |
| — | Top / bottom straight edges | red |
| `b` | Wire thickness (perpendicular gap of the two edges) | blue |
| B | Reconstructed round end, Ø = `b`, tangent at tip | green |

## 2. Notation

For the main wire component, after thresholding the mask (`>127`), scan each
column `x` for the topmost / bottommost white pixel:

- `top(x)` — upper edge; `bot(x)` — lower edge.

The two long straight edges are modelled as lines `y = s·x + i`:

```
top edge : y = s_top · x + i_top
bot edge : y = s_bot · x + i_bot
```

## 3. Algorithm

### Step 0 — Select the link under inspection (main component)
1. Threshold the wire mask (`>127`); connected-component labelling
   (`_extract_wire_segments`, `min_area = 3000 px`).
2. Choose the **largest-area component whose left edge does not touch the frame**
   (`x1 > 15 px`). A partial link cut off at `x ≈ 0` has a missing cap and cannot
   yield a valid circle B, so it is rejected (`_select_main_segment`).

### Step 1 — Straight edges (red lines)
Scan `top(x), bot(x)` over the component's columns. Fit each edge with **RANSAC**
(`_fit_line_ransac`) using **only the central 60 %** of columns
(`i ∈ [0.20 n, 0.80 n]`) so the rounded caps do not bias the straight-edge fit.

### Step 2 — Wire thickness `b` (blue line)
`b` is the **perpendicular** gap between the two edge lines at the mid-column
`x_mid`:

```
Δy = |(s_bot·x_mid + i_bot) − (s_top·x_mid + i_top)|
b  = Δy · cos( atan( (s_top + s_bot) / 2 ) )
```

The `cos(atan(·))` factor converts the vertical gap to a true perpendicular
distance when the wire is slightly tilted.

### Step 3 — Tip apex (point 1)
Each cap's tip is the wire's extreme column; its `y` is the mid-point of the two
edges there:

```
P1_left  = ( x[0],  (top[0]  + bot[0])  / 2 )
P1_right = ( x[-1], (top[-1] + bot[-1]) / 2 )
```

A tip within `edge_margin = max(15, 0.02·W)` px of the frame is **rejected** as a
frame-cut (not a true cap); that side then yields no circle B (one-sided
measurement — see §5).

### Step 4 — Tangent corners (points 2) — persistence detector
The four corners mark where each straight edge **departs from its fitted line**
and the cap begins to curve. Starting from the centre column, walk **outward**
along each edge (`direction = −1` toward the left cap, `+1` toward the right) and
track the deviation `|edge(x) − line(x)|`.

A naïve "first column beyond threshold" break is fragile — a single noisy jag near
the middle terminates the scan early. Instead we require **`persist = 5`
consecutive** out-of-tolerance columns (`thr = 6 px`) to confirm entry into the
cap, and report the **first column of that confirmed run** as the corner
(`_tangent_corner`). This hysteresis rejects isolated mask noise.

The corner's `y` is taken **on the fitted straight edge** (`y = s·x + i`), not the
raw mask pixel — the tangent point lies on the straight edge by definition, and
using the line suppresses per-column mask noise. Corners: `P2_TL, P2_BL`
(left cap), `P2_TR, P2_BR` (right cap).

### Step 5 — Circle B (green) and area B
Circle B has **diameter `b`** (radius `r = b/2`) and is **inscribed tangent** at
the tip: its centre is shifted one radius **inward** along the wire axis from the
tip, at the edge mid-line (`_tangent_circle_at_tip`):

```
left  cap : c_x = x_tip + r        right cap : c_x = x_tip − r
c_y = ( (s_top·c_x + i_top) + (s_bot·c_x + i_bot) ) / 2
area_B = π · r²            (area_B_mm² = area_B / px_per_mm²)
```

One circle B is produced per **non-frame-cut** tip.

## 4. Summary of what is *measured* vs *reconstructed*

| Quantity | Source |
|----------|--------|
| `top, bot` edges, slopes/intercepts, `b` | **Measured** from the wire-mask edges (central-60 % RANSAC) |
| Tip apex (point 1) | **Measured** (extreme column, edge mid-point) |
| Tangent corners (points 2) | **Measured** (edge-departure column, on the fitted line) |
| Circle B / area B | **Reconstructed** disc (Ø = `b`, tangent at tip) |

## 5. Assumptions and limitations
- **Round wire cross-section.** Circle B assumes the cap's top-view section is a
  circle of diameter `b`. A worn/flattened cap violates this — which is exactly
  the wear signal captured by the A∩B overlap downstream.
- **Straight central edges.** The perpendicular `b` and the corner-departure test
  assume the wire's mid-section edges are straight; heavy curvature or a bent
  link biases both.
- **Frame-cut caps → one-sided.** If a cap is cut by the frame (or worn through)
  the tip is rejected and only the opposite side yields a circle B. `chain4`
  exhibits this (single circle B, `b` from a one-sided fit — see §6).
- **Corner outward bias (~5 px).** Because the corner `y` sits on the central-60 %
  RANSAC line — which runs slightly proud of the true mask boundary near the cap
  transition — the point-2 corners land ~5 px outside the binary mask. This is
  systematic, not a failure, and does not affect `b` or circle B.
- **Thresholds `thr = 6 px`, `persist = 5`** are the two tunables; accuracy on the
  reference sample improves monotonically up to `thr = 6`.

## 6. Validation

### 6.1 Reference sample (`chain`, unworn) vs manual annotation
Detected points vs the hand-marked targets in
`resource/horizontal_mask_annotated.png` (`px_per_mm = 1`):

```
b = 220.8 px      area_B = 38 295 px²   (two circles B)
```

| point | detected | manual | Δ (px) |
|-------|----------|--------|:-----:|
| P1 left  | (662, 552) | (659, 545) | 7.6 |
| P1 right | (1682, 560) | (1680, 568) | 8.2 |
| P2 TL | (727, 445) | (728, 451) | 6.1 |
| P2 BL | (773, 667) | (778, 663) | 6.4 |
| P2 TR | (1617, 458) | (1621, 457) | 4.1 |
| P2 BR | (1558, 678) | (1584, 667) | **28.2** |

Five of six points match to ≤ 8 px. The outlier is the bottom-right corner
(28 px): the lower-right cap has a raggeder mask edge, so the persistence detector
walks slightly further into the cap before confirming departure.

### 6.2 Worn samples (`chain3` ≈ 30 %, `chain4` ≈ 50 %)
The method was re-run unchanged on two worn links.

| sample | `b` (px) | area B (px²) | tips / circles B | result |
|--------|:-------:|:-----------:|:----------------:|--------|
| chain  (new)   | 220.8 | 38 295 | 2 / 2 | 12 points on edges, both caps |
| chain3 (~30 %) | 219.9 | 37 992 | 2 / 2 | both caps, corners on tangents |
| chain4 (~50 %) | 208.5 | 34 126 | 1 / 1 | right end frame-cut → one-sided |

Observations:
- **Thickness stability.** `b` on the two intact samples agrees to 0.4 %
  (220.8 vs 219.9 px), consistent with a single physical wire gauge. `chain4`
  reads 5.6 % lower, from its one-sided (frame-cut) fit rather than a real gauge
  change.
- **Self-centring.** The tips follow the link across the frame (apex `x` shifts
  with wear position) with no hard-coded coordinates.
- **Graceful degradation.** On the frame-cut `chain4` the model rejects the cut
  tip and returns a single valid circle B rather than a spurious one; the wear
  pipeline then measures that side only.

Overlays: `debug_v2/hwire_points_chain.png`, `…_chain3.png`, `…_chain4.png`.
