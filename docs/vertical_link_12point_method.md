# Vertical Link Reconstruction — 12-Point Control Model (Method)

*Module: `src/link_model.py` · Input: `mask_vert_chain.jpg` · Reference: `resource/vertical_mask_annotated.png`*

## 1. Problem statement

The vertical chain link under inspection is a closed loop (a rounded rectangle /
"stadium" shape). In the camera view its two **rounded end-caps are occluded** by
the two crossing horizontal links. The segmentation therefore returns only the
two long **rails** (top and bottom bands) of the link; the caps are missing.

To measure wear at the contact region we must **reconstruct the occluded caps** as
two boundary curves — an **inner** curve (the hole edge) and an **outer** curve
(the link's outer edge) — whose enclosed material forms the crescent **area A**.

The reconstruction is expressed with **12 control points**, matching the reference
annotation:

| Points | Meaning | Colour |
|--------|---------|--------|
| 9, 10, 11, 12 | Bar thickness `d` at the apex (top-outer, top-inner, bot-inner, bot-outer) | red |
| 3, 4, 5, 6 | **Inner** cap corners (hole boundary) — left top/bot, right top/bot | green |
| 1, 2, 7, 8 | **Outer** cap corners — left top/bot, right top/bot | yellow |

Inner arcs (3–4, 5–6) are drawn red; outer arcs (1–2, 7–8) blue; the crescent
between them is **area A**.

## 2. Notation

For each image column `x`, after thresholding the mask to a binary image `B(x,y)`:

- **Top rail** contributes two edges: `t_out(x)` = first (smallest-`y`) white pixel
  = outer edge; `t_in(x)` = last white pixel = inner edge (upper boundary of the hole).
- **Bottom rail** contributes: `b_in(x)` = first white pixel = inner edge (lower
  boundary of the hole); `b_out(x)` = last white pixel = outer edge.

Local bar thickness of each rail:

```
th_top(x) = t_in(x) − t_out(x)
th_bot(x) = b_out(x) − b_in(x)
```

## 3. Algorithm

### Step 0 — Isolate the link (blob pairing)
1. Threshold the mask (`>127`) to binary.
2. Connected-component labelling; discard components with area `< 2000 px`
   (noise / partial neighbours).
3. Split components by the **median centroid-`y`** into a *top* group and a
   *bottom* group.
4. Take the **left-most** component of each group → the top rail and bottom rail
   of the link under inspection. (The link nearest the frame centre can be chosen
   instead; here we follow the annotation and take the left oval.)

### Step 1 — Edge scan
For every column spanned by each rail, record the first/last white pixel to build
the four discrete edge arrays `t_out, t_in, b_in, b_out` over their column indices
`x_t` (top rail) and `x_b` (bottom rail).

### Step 2 — Apex and thickness `d` (points 9–12)
The apex is the horizontal centre of the whole oval:

```
x_apex = ( min(x_t[0], x_b[0]) + max(x_t[-1], x_b[-1]) ) / 2
```

Evaluate the four edges at `x_apex` (nearest-column lookup):

```
P9  = (x_apex, t_out(x_apex))     # top outer
P10 = (x_apex, t_in (x_apex))     # top inner
P11 = (x_apex, b_in (x_apex))     # bottom inner
P12 = (x_apex, b_out(x_apex))     # bottom outer
```

Bar thicknesses and the working thickness `d`:

```
d_top = P10.y − P9.y
d_bot = P12.y − P11.y
d     = (d_top + d_bot) / 2
```

`d` is the **yellow-line thickness** of §3.2 of the spec and the offset used to
build the outer curve. (If the link is unworn and untilted, `d_top ≈ d_bot`; a
difference is a tilt/perspective cue, corrected upstream.)

### Step 3 — Inner cap corners (points 3, 4, 5, 6)
The inner corners mark where the **link hole opens** — the last hole-boundary
column before the rail thins into the occluded cap. The opening column is found
from the **top rail** at (near) full thickness (90 % of its maximum), and **both**
rail inner edges are then sampled at that **same column per side** so that the top
and bottom inner corners are vertically aligned (`P3∥P4`, `P5∥P6`):

```
K_top = { i : th_top(x_t[i]) ≥ 0.9 · max th_top }
x_hL  = x_t[min K_top]        x_hR = x_t[max K_top]     (left / right opening)

P3 = (x_hL, t_in(x_hL))   P4 = (x_hL, b_in(x_hL))       # left  inner top / bot
P5 = (x_hR, t_in(x_hR))   P6 = (x_hR, b_in(x_hR))       # right inner top / bot
```

**Rationale.** The rail thickness is ≈ constant along the straight section and
drops as the segmentation fades into the occluded cap; the 90 %-of-max crossing is
a stable, scale-free estimate of "where the straight rail ends". Using one column
per side keeps `3↔4` and `5↔6` aligned, as in the annotation.

### Step 4 — Outer cap corners (points 1, 2, 7, 8) — vertical ∩ horizontal
The outer corners are the **intersection of the vertical link with the crossing
horizontal chain**, not a geometric offset. Two ingredients:

1. **x (junction column)** = the vertical link's outer extent `x_oL` / `x_oR`. The
   segmented vertical-link component runs out to its *neck*, where it joins the
   horizontal band, so `x_oL = min(x_t[0], x_b[0])`, `x_oR = max(x_t[-1], x_b[-1])`
   already give that junction column.
2. **y (band edge)** = the horizontal band's top/bottom edge from the **wire mask**,
   taken as the median first/last white row over a window around each junction:

```
(yT_L, yB_L) = band_edges(wire, around x_oL)
(yT_R, yB_R) = band_edges(wire, around x_oR)

P1 = (x_oL, yT_L)   P2 = (x_oL, yB_L)     # left  outer (junction with left band)
P7 = (x_oR, yT_R)   P8 = (x_oR, yB_R)     # right outer (junction with right band)
```

**Fallback.** If no `wire_mask` is supplied, the outer corners revert to the inner
corners offset outward by the bar thickness `d` (`P1 = P3 − (d,0)`, etc.). The
intersection form is preferred because the true junction moves with wear/geometry,
whereas a fixed `d`-offset only coincides with it on the unworn reference link.

### Step 5 — Arc reconstruction (inner red, outer blue) — Bézier sagitta
Each cap boundary is reconstructed as a **quadratic Bézier** curve between its two
corner points, bulging outward with a shallow sagitta so the curve rounds the
occluded tip without over-inflating it. For corners `A = (top)`, `B = (bot)`:

```
M      = (A + B) / 2                       # chord midpoint
bulge  = 0.35 · |B.y − A.y|                # sagitta (shallow, empirical)
Tip    = M + s · (bulge, 0)                # s = −1 left cap, +1 right cap
C(t)   = (1−t)² A + 2(1−t)t · Tip + t² B,   t ∈ [0,1]
```

Sampling `C(t)` at 60 points gives:
- **Inner arcs** (red): `P3→P4` (left), `P5→P6` (right) — the hole's rounded end.
- **Outer arcs** (blue): `P1→P2` (left), `P7→P8` (right) — the link's outer end.

A full semicircle (radius = ½·chord) was tried but over-bulges the cap; the
shallow `0.35·chord` sagitta matches the observed cap curvature and is the single
tunable constant in the reconstruction.

### Step 6 — Area A (per cap, with value)
Per cap, the crescent polygon is the closed loop
`outer_arc ++ reverse(inner_arc)` (endpoints matched):

```
area_A_poly = [ outer arc samples ] + [ inner arc samples reversed ]
area_A_px   = |contourArea(area_A_poly)|          # shoelace
area_A_mm2  = area_A_px / (px_per_mm)²
```

The value is reported per side and rendered as a filled, labelled region
(`A = N px²`) in the overlay. Downstream, `area_A` is intersected with the
horizontal-wire circle **B** to give wear = area(A ∩ B) / area(A).

Measured areas (reference run, `px_per_mm = 1`):

| sample | left A (px²) | right A (px²) |
|--------|:-----------:|:------------:|
| chain  (new)   | 45 548 | 47 077 |
| chain3 (~30 %) | 54 526 | 51 624 |
| chain4 (~50 %) | 44 329 | 44 374 |

Note: `area_A` is the *nominal reconstructed cap* (bounded by the hole and the
horizontal-chain junction), not the wear metric itself — the wear signal is the
overlap of A with the wire circle B, computed in the next stage.

## 4. Summary of what is *measured* vs *reconstructed*

| Quantity | Source |
|----------|--------|
| `t_out, t_in, b_in, b_out`, apex, `d_top`, `d_bot`, `d` | **Measured** from the vertical-mask edges |
| Inner corners 3,4,5,6 | **Measured** (hole-opening column, both rails, x-aligned) |
| Outer corners 1,2,7,8 | **Measured** — intersection of vertical link (junction column) and horizontal chain (band edge, wire mask) |
| Area A | **Reconstructed** crescent between inner edge (3–4 / 5–6) and outer edge (1–2 / 7–8) |

## 5. Assumptions and limitations
- **Constant bar thickness** across the cap (`outer = inner ⊕ d`). Valid for a
  round-stock link; a worn/flattened cap violates it — which is precisely the wear
  signal captured downstream by the A∩B overlap.
- **Horizontal cap axis** near the tip (offset applied along `x`). The link must be
  centred / perspective-corrected first (pipeline steps V + H) for this to hold;
  off-frame or tilted links bias the lower corners (2,4,6,8).
- **Sagitta constant `0.35`** is empirical, tuned to the reference annotation.
- The full-thickness threshold (`0.9`) trades robustness vs. how far the corners
  sit from the true cap start; it is stable for clean masks but sensitive to
  ragged segmentation at the occlusion boundary.

## 6. Validation

### 6.1 Reference sample (`chain.png`, unworn)
The detected points reproduce the manual annotation to within a few pixels (max
drift on the lower outer corners 2/4/6/8, from asymmetric rail full-thickness spans):

```
apex = 518,  d_top = 220,  d_bot = 202,  d = 211 px
pt1 (32,438)  pt2 (20,673)  pt3 (243,438)  pt4 (231,673)
pt5 (793,438) pt6 (802,674) pt7 (1004,438) pt8 (1013,674)
pt9 (518,200) pt10(518,420) pt11(518,694)  pt12(513,896)
```

### 6.2 Worn samples (`chain3` ≈ 30 %, `chain4` ≈ 50 %)
The method was re-run unchanged on two worn links to confirm it generalises to
different link positions, scales and wear states.

| sample | apex `x` | `d` (px) | outer corners 1,7 (x) | result |
|--------|:-------:|:-----:|:---------------------:|--------|
| chain  (new)   | 518 | 211 | 28 → 1008 | 12 points on edges |
| chain3 (~30 %) | 578 | 210 | 70 → 1086 | ≤14 px vs manual annotation |
| chain4 (~50 %) | 792 | 196 | 296 → 1288 | 12 points on edges |

For `chain3` (the sample with a manual full-mask annotation,
`resource/mask_full_chain3_annotated.png`) every one of points 1–8 lands within
**≤14 px** of the hand-marked target — outer corners 1,2,7,8 within ±8 px at the
vertical/horizontal junction, inner corners 3,4,5,6 x-aligned on the hole.

Observations:
- **Self-centring.** The apex is derived from the link's own horizontal extent, so
  detection follows the link across the frame (518 → 578 → 792 px) with no
  hard-coded coordinates.
- **Thickness stability.** `d` is recovered independently on each mask and stays in
  a narrow band (196–211 px), consistent with a single physical chain gauge.
- **Corner rule generalises.** The 0.9·max-thickness endpoint rule located the
  correct arc endpoints on all three masks, including the raggeder `chain4`
  segmentation.

Overlays: `debug_v2/vert_points_chain.png`, `…_chain3.png`, `…_chain4.png`.
