from __future__ import annotations
import torch

def hit_rate_at_alpha(scores: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Fraction of samples with finite tau and scores <= tau.
    Returns a scalar tensor in [0,1].
    """
    valid = torch.isfinite(tau)
    if not valid.any():
        return torch.zeros((), dtype=scores.dtype, device=scores.device)
    hits = (scores[valid] <= tau[valid]).float().mean()
    return hits

def mean_finite(x: torch.Tensor) -> torch.Tensor:
    m = torch.isfinite(x)
    if not m.any():
        return torch.zeros((), dtype=x.dtype, device=x.device)
    return x[m].mean()

def fraction_valid(tau: torch.Tensor) -> torch.Tensor:
    m = torch.isfinite(tau)
    return m.float().mean()
