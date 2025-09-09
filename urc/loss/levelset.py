from __future__ import annotations
import torch

def _mask_valid(t: torch.Tensor) -> torch.Tensor:
    """Boolean mask for entries that are finite (not inf/nan)."""
    return torch.isfinite(t)

def acceptance(
    scores_gen: torch.Tensor,
    tau: torch.Tensor,
    mode: str = "softplus",
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Acceptance loss (level-set):
      We want scores_gen <= tau - margin (lower score = more likely under density).
      violation = scores_gen - (tau - margin); positive => outside region.

    Args:
      scores_gen: [B] scores s = -log p(y|e)
      tau:       [B] thresholds; may contain +inf for warmup (masked)
      mode:      "softplus" | "hinge"
      margin:    non-negative relaxation of the boundary inward

    Returns:
      Scalar tensor (mean over valid violating entries if any, else 0).
    """
    assert scores_gen.shape == tau.shape, "scores_gen and tau must align"
    violation = scores_gen - (tau - margin)
    valid = torch.isfinite(tau)
    if valid.any():
        v = violation[valid]
        if mode == "softplus":
            return torch.nn.functional.softplus(v).mean()
        elif mode == "hinge":
            vv = torch.relu(v)
            if (vv > 0).any():
                return vv[vv > 0].mean()
            # ❗ no violating entries → return graph-connected zero
            return (v * 0.0).sum()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    # ❗ no valid taus (e.g., warmup) → graph-connected zero
    return (scores_gen * 0.0).sum()

def separation(
    scores_gen_wrong: torch.Tensor,
    tau_wrong: torch.Tensor,
    margin: float = 0.5,
    mode: str = "softplus",
) -> torch.Tensor:
    """
    Separation loss:
      Make samples improbable under *wrong* conditions:
        target: scores_gen_wrong >= tau_wrong + margin (outside the wrong region)
        violation = (tau_wrong + margin) - scores_gen_wrong; positive => too plausible under wrong cond.

    Args:
      scores_gen_wrong: [B] scores under wrong conditions
      tau_wrong:        [B] thresholds for those wrong conditions (masked if +inf)
      margin:           push-away margin
      mode:             "softplus" | "hinge"

    Returns:
      Scalar tensor (mean over violating entries if any, else 0).
    """
    assert scores_gen_wrong.shape == tau_wrong.shape, "shapes must align"
    violation = (tau_wrong + margin) - scores_gen_wrong
    valid = torch.isfinite(tau_wrong)
    if valid.any():
        v = violation[valid]
        if mode == "softplus":
            sp = torch.nn.functional.softplus(v)
            if (v > 0).any():
                return sp[v > 0].mean()
            return (v * 0.0).sum()     # ❗ graph-connected zero
        elif mode == "hinge":
            vv = torch.relu(v)
            if (vv > 0).any():
                return vv[vv > 0].mean()
            return (v * 0.0).sum()     # ❗ graph-connected zero
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return (scores_gen_wrong * 0.0).sum()  # ❗ graph-connected zero

def size_proxy(Y_in: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Region size proxy via logdet of (shrinkage) covariance of in-region points.

      Σ = (X_c^T X_c) / max(1, n-1) + eps * I,  X_c = Y - mean(Y)

    Returns:
      Scalar logdet(Σ) (finite). If n < 2, returns 0 to avoid instability.
    """
    if Y_in.numel() == 0 or Y_in.shape[0] < 2:
        return torch.zeros((), dtype=Y_in.dtype, device=Y_in.device)

    N, D = Y_in.shape
    Yc = Y_in - Y_in.mean(dim=0, keepdim=True)
    denom = max(1, N - 1)
    Sigma = (Yc.t() @ Yc) / denom
    Sigma = Sigma + eps * torch.eye(D, dtype=Y_in.dtype, device=Y_in.device)

    sign, logabsdet = torch.linalg.slogdet(Sigma)
    if (sign <= 0).any():
        # Rare numerical case: add extra shrinkage once and recompute
        Sigma = Sigma + (10.0 * eps) * torch.eye(D, dtype=Y_in.dtype, device=Y_in.device)
        sign, logabsdet = torch.linalg.slogdet(Sigma)
    return logabsdet
