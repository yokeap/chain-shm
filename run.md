# Running chain-shm

Pipeline: seg → perspective correction → reconstruction → horizontal wire → area-based wear.
Entry point: `run_offline_test.py`.

## Option A — Docker (Jetson image, recommended)

```bash
docker run --rm --runtime nvidia \
  --user $(id -u):$(id -g) \
  --group-add video \
  -e HOME=/tmp \
  -v ~/git/chain-shm:/workspace \
  -w /workspace \
  chain-inspector:jetson \
  python3 run_offline_test.py --image ./sample/chain.png --model sam_b.pt
```

- `--user $(id -u):$(id -g)`: the image has no `USER` set, so it defaults to root.
  Without this flag, every output file under `debug_seg/` is written as `root`
  and a later native run can't overwrite them ("permission denied").
- `--group-add video`: Jetson GPU device nodes (`/dev/nvmap`, `/dev/nvhost-*`)
  are owned by `root:video` (GID 44), mode 660. Dropping to a non-root user
  without this loses GPU access — SAM silently fails to produce a wire mask
  and every measurement comes back `0.0px`. Root bypasses this check, which is
  why it wasn't obvious until `--user` was added.
- `-e HOME=/tmp`: avoids libraries (ultralytics, matplotlib) failing to write
  config/cache when `$HOME` resolves oddly for a numeric, passwd-less UID.
- `sudo` is not required if your user is in the `docker` group (`groups` includes `docker`).

One-time fix if old outputs are already root-owned (no host `sudo` needed —
a root *container* can chown files it owns just as well):

```bash
docker run --rm -v ~/git/chain-shm:/workspace alpine chown -R $(id -u):$(id -g) /workspace/debug_seg
```

## Option B — Native (no Docker, no GPU/SAM)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python3 run_offline_test.py --image ./sample/chain.png --skip-sam
```

`--skip-sam` swaps the SAM point-prompt wire segmentation for a simple
row-brightness heuristic (`get_wire_mask_simple`) — no GPU or model weights needed.

## CLI flags (`run_offline_test.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--image` | required | input chain image |
| `--model` | `sam_b.pt` | SAM checkpoint (ignored with `--skip-sam`) |
| `--skip-sam` | off | use brightness heuristic instead of SAM |
| `--px-per-mm` | `1.0` | pixel-to-mm scale (unused downstream currently) |
| `--save-dir` | `debug_seg` | where masks/overlays/report.txt go |

## Outputs (in `--save-dir`)

- `mask_full_<stem>.jpg`, `mask_wire_<stem>.jpg`, `mask_vert_<stem>.jpg` — segmentation stages
- `rect_<stem>.jpg` — tilt-corrected image
- `recon_<stem>.jpg` — vertical-link arc reconstruction overlay
- `hwire_<stem>.jpg` — horizontal wire edges + tangent circles
- `full_recon_<stem>.jpg` — combined overlay with wear % labels
- `report.txt` — text summary (d, b, wear % per side) — overwritten every run, not per-image
