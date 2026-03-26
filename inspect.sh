#!/bin/bash
# Usage: ./inspect.sh chain.png
# Usage: ./inspect.sh chain.png --reject-threshold 0.08

sudo docker run --rm --runtime nvidia \
  -v ~/git/chain-shm:/workspace \
  -w /workspace \
  chain-inspector:jetson \
  python3 run_offline_test.py "$@"
