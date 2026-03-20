# chain-inspector

Chain wear inspection pipeline using **FastSAM** + **morphological analysis** (Skeletonization, Distance Transform, Savitzky-Golay smoothing).

## Quick start (offline, adaptive segmenter – no GPU required)

```bash
# 1. Create & activate uv environment
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install core dependencies
uv pip install -e ".[dev]"

# 3. Run offline test on your image
uv run python scripts/run_offline_test.py --image path/to/chain.jpg

# Optional: change reject threshold (default 10%)
uv run python scripts/run_offline_test.py --image chain.jpg --reject-threshold 0.08
```

Outputs saved to `data/samples/`:
- `annotated_<name>.jpg`  – overlay with PASS/REJECT banner
- `mask_<name>.jpg`       – binary segmentation mask
- `thickness_profile_<name>.png` – full thickness curve

---

## Full pipeline (FastSAM, requires GPU or Jetson)

```bash
# Install FastSAM (bundled in ultralytics)
uv pip install -e ".[dev]"          # ultralytics already included

# FastSAM model weights are downloaded automatically on first run
chain-inspect chain.jpg --show
chain-inspect chain.jpg --save-dir results/
chain-inspect chain.jpg --fastsam-variant x    # larger model
```

---

## Pipeline overview

```
Image
  │
  ▼
[Segmentation]
  ├─ FastSAM (default, GPU-accelerated on Jetson Orin NX)
  └─ AdaptiveSegmenter (CLAHE + adaptive threshold, fallback / offline)
  │
  ▼
Binary mask  (chain body = 255)
  │
  ▼
[Distance Transform]   dist[i,j] = distance to nearest background pixel
                       ↔  local radius of chain cross-section
  │
  ▼
[Skeletonization]      centre-line of chain
  │
  ▼
Sample dist along skeleton → raw thickness profile (px)
  │
  ▼
[Savitzky-Golay Filter]  smooth noise while preserving edge geometry
  │
  ▼
reference_thickness = 95th-percentile of profile (un-worn reference)
min_thickness       = minimum in profile
wear_ratio          = (reference − min) / reference
  │
  ▼
wear_ratio > 0.10  →  REJECT
wear_ratio ≤ 0.10  →  PASS
```

---

## Dynamic Thresholding (chain swing compensation)

The `AdaptiveSegmenter` uses **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) with a large tile grid before applying adaptive Gaussian thresholding. This compensates for:
- Uneven illumination caused by chain swinging
- Slight out-of-focus blur at chain edges

---

## Jetson Orin NX deployment notes

- `ultralytics` auto-detects CUDA on JetPack 5.x
- Use `FastSAM-s.pt` for real-time performance (~30+ FPS at 640 px)
- Install Basler pypylon: `uv pip install -e ".[basler]"`
- For TensorRT acceleration: export FastSAM with `model.export(format='engine')`

---

## Project structure

```
chain-inspector/
├── pyproject.toml
├── src/chain_inspector/
│   ├── __init__.py
│   ├── inspector.py      ← core pipeline
│   └── cli.py            ← command-line interface
├── scripts/
│   └── run_offline_test.py
├── tests/
├── data/samples/
└── configs/
```
