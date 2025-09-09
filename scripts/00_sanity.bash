#!/usr/bin/env bash
set -euo pipefail
uv run urc sanity phi-check --n 256 --d-out 64 --pca-k 16
uv run urc sanity mdn-toy-fit --steps 200 --lr 5e-3
