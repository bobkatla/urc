from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Dict

@dataclass
class StandardScalerTorch:
    mean_: torch.Tensor | None = None
    var_: torch.Tensor | None = None
    eps: float = 1e-8

    def to(self, device: torch.device | str):
        if self.mean_ is not None: self.mean_ = self.mean_.to(device)
        if self.var_ is not None: self.var_ = self.var_.to(device)
        return self

    def fit(self, X: torch.Tensor) -> 'StandardScalerTorch':
        # X: [N, D]
        self.mean_ = X.mean(dim=0)
        # population variance (ddof=0) to match StandardScaler behavior
        xm = X - self.mean_
        self.var_ = (xm * xm).mean(dim=0).clamp_min(self.eps)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.mean_ is not None and self.var_ is not None, "Call fit or load stats first."
        return (X - self.mean_) / torch.sqrt(self.var_)

    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        assert self.mean_ is not None and self.var_ is not None, "Call fit or load stats first."
        return Z * torch.sqrt(self.var_) + self.mean_

    def dump_stats(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean_, "var": self.var_, "eps": torch.tensor(self.eps)}

    def load_stats(self, d: Dict[str, torch.Tensor]) -> 'StandardScalerTorch':
        self.mean_ = d["mean"]
        self.var_ = d["var"]
        self.eps = float(d.get("eps", 1e-8))
        return self

@dataclass
class PCATorch:
    mean_: torch.Tensor | None = None
    components_: torch.Tensor | None = None  # [D, K]
    k: int = 2

    def to(self, device: torch.device | str):
        if self.mean_ is not None: self.mean_ = self.mean_.to(device)
        if self.components_ is not None: self.components_ = self.components_.to(device)
        return self

    def fit(self, X: torch.Tensor, k: int | None = None) -> 'PCATorch':
        # X: [N, D]
        if k is not None:
            self.k = k
        self.mean_ = X.mean(dim=0, keepdim=True)  # [1, D]
        Xc = X - self.mean_
        # economy SVD on centered data
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)  # Vh: [min(N,D), D]
        V = Vh.transpose(0,1)  # [D, r]
        self.components_ = V[:, :self.k].contiguous()
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.mean_ is not None and self.components_ is not None, "Call fit first."
        Xc = X - self.mean_
        return Xc @ self.components_  # [N, K]

    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        assert self.mean_ is not None and self.components_ is not None, "Call fit first."
        return Z @ self.components_.transpose(0,1) + self.mean_

    def dump_stats(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean_, "components": self.components_, "k": torch.tensor(self.k)}

    def load_stats(self, d: Dict[str, torch.Tensor]) -> 'PCATorch':
        self.mean_ = d["mean"]
        self.components_ = d["components"]
        self.k = int(d.get("k", self.components_.shape[1]))
        return self
