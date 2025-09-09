# URC — Unified Region Calibration (v0.1.0)

**Goal (v1):** Region-aware *training-time* loss for conditional generators using **level-sets in φ-space**.
This repo currently ships **Phase 0–1**: Torch-only φ-transforms (Standardize → PCA) and a pluggable `PhiImage`.

## Quickstart (φ only for now)
```python
from urc.phi.transforms import StandardScalerTorch, PCATorch
from urc.phi.image_phi import PhiImage
import torch as t

Y = t.randn(1024, 128, device='cpu')
phi = PhiImage(backbone=None, d_out=128, pca_k=64)
phi.fit_stats(Y)
Y_proj = phi.project(Y)   # [1024, 64]
```

## Install (dev)
```bash
uv venv && uv pip install -e .[dev]
uv run pytest -q
```

## Roadmap
- Phase 2: Conditional MDN
- Phase 3: Streaming quantiles
- Phase 4: Level-set losses
- Phase 5: URCModule orchestration
- Phase 6: I/O & state mgmt
- Phase 7: Docs & CI
```
