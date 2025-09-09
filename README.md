# URC — Unified Region Calibration (v0.1.0)

**URC** adds a **training-time, region-aware regularizer** to conditional generative models.  
It projects data to a feature space φ, fits a **conditional density** \(p(y \mid e)\) with a Mixture of Diagonal Gaussians (MDN), maintains **streaming per-class thresholds** \(\tau_\alpha\) (for coverage \(1-\alpha\)), and penalizes generated samples whose **scores** exceed \(\tau_\alpha\).  
This v1 implements the **level-set approach** (deterministic regions via score sublevel sets) and returns a scalar loss to add to your base loss.

- **Torch-only** φ transforms (StandardScaler, PCA) — no NumPy round-trips at training time.
- **Conditional MDN** head (stable `log_prob`, variance floors).
- **Streaming per-class quantiles** (FIFO windows) with warm-up masking.
- **Level-set losses**: acceptance, separation (optional), and a size proxy.
- **URCModule** orchestrates everything and returns a single scalar loss **+ diagnostics**.
- **Save/load** of φ stats, MDN, and quantiles.

> **Planned (not implemented here):** OT-defined regions and CP-based evaluation.

## Install

URC is uv- and pip-friendly.

```bash
# From the repo root
uv sync
# or plain pip:
pip install -e .
