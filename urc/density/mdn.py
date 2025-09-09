# urc/density/mdn.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn

LOG2PI = math.log(2.0 * math.pi)

def _stable_logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return (x - m).exp().sum(dim=dim).log() + m.squeeze(dim)

@dataclass
class MDNConfig:
    d_y: int
    d_e: int
    n_comp: int = 4
    hidden: int = 128
    var_floor: float = 1e-4
    use_layernorm: bool = False

class CondMDN(nn.Module):
    """
    Conditional Mixture of Diagonal Gaussians: p(y | e).

    Args:
        d_y: dim of y
        d_e: dim of conditioning e
        n_comp: # mixture components
        hidden: hidden width
        var_floor: minimum variance per dim (numerical guard)
        use_layernorm: optional LN on hidden

    Methods:
        log_prob(y,e): [B]
        nll(y,e): scalar mean NLL
        fit_step(y,e,opt,clip_grad): one optimizer step, returns float loss
    """
    def __init__(self, d_y: int, d_e: int, n_comp: int = 4, hidden: int = 128,
                 var_floor: float = 1e-4, use_layernorm: bool = False):
        super().__init__()
        self.d_y, self.d_e, self.n_comp = d_y, d_e, n_comp
        self.var_floor = var_floor

        layers = [nn.Linear(d_e, hidden), nn.ReLU(inplace=True)]
        if use_layernorm:
            layers.insert(1, nn.LayerNorm(hidden))
        layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

        self.head_mu = nn.Linear(hidden, n_comp * d_y)
        self.head_logvar = nn.Linear(hidden, n_comp * d_y)
        self.head_logit = nn.Linear(hidden, n_comp)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _params(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(e)                                  # [B,H]
        B = e.shape[0]
        mu = self.head_mu(h).view(B, self.n_comp, self.d_y)         # [B,K,D]
        logvar = self.head_logvar(h).view(B, self.n_comp, self.d_y) # [B,K,D]
        # variance floor: var = max(exp(logvar), floor); keep logvar consistent
        var = torch.exp(logvar).clamp_min(self.var_floor)
        logvar = torch.log(var)
        logit = self.head_logit(h)                                   # [B,K]
        return mu, logvar, logit

    def log_prob(self, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        mu, logvar, logit = self._params(e)                 # [B,K,D], [B,K,D], [B,K]
        B, K, D = mu.shape
        y = y.unsqueeze(1).expand(B, K, D)                  # [B,K,D]
        inv_var = torch.exp(-logvar)
        diff2 = (y - mu) ** 2
        # per-component diagonal Gaussian log-prob
        logp_per_dim = -0.5 * (diff2 * inv_var + (LOG2PI + logvar))
        logp = logp_per_dim.sum(dim=-1)                     # [B,K]
        log_mix = torch.log_softmax(logit, dim=-1)          # [B,K]
        return _stable_logsumexp(logp + log_mix, dim=-1)    # [B]

    def nll(self, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(y, e).mean()

    def fit_step(self, y: torch.Tensor, e: torch.Tensor, optimizer: torch.optim.Optimizer,
                 clip_grad: Optional[float] = None) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = self.nll(y, e)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        return float(loss.detach().cpu().item())

def mdn_nll(mdn: CondMDN, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    return mdn.nll(y, e)
