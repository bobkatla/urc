from __future__ import annotations
from typing import Callable, Optional
import torch
from .transforms import StandardScalerTorch, PCATorch

class PhiImage(torch.nn.Module):
    """Torch-only φ module: backbone -> Standardize -> PCA

    Args:
        backbone: Callable mapping X -> Y features. If None, identity.
        d_out: expected feature dim from backbone. Only used for shape checks.
        pca_k: output dim after PCA.
    """
    def __init__(self, backbone: Optional[Callable]=None, d_out: int=128, pca_k: int=64):
        super().__init__()
        self.backbone = backbone
        self.d_out = d_out
        self.scaler = StandardScalerTorch()
        self.pca = PCATorch(k=pca_k)

    def to(self, device):
        super().to(device)
        self.scaler.to(device)
        self.pca.to(device)
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            Y = X
        else:
            Y = self.backbone(X)
        if Y.dim() != 2:
            Y = Y.flatten(1)
        return Y

    @torch.no_grad()
    def fit_stats(self, Y: torch.Tensor) -> None:
        # Y: [N, D]
        if Y.dim() != 2:
            Y = Y.flatten(1)
        self.scaler.fit(Y)
        Ys = self.scaler.transform(Y)
        self.pca.fit(Ys, k=self.pca.k)

    @torch.no_grad()
    def project(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.forward(X)
        Ys = self.scaler.transform(Y)
        Z = self.pca.transform(Ys)
        return Z

    def dump_stats(self) -> dict:
        return {"scaler": self.scaler.dump_stats(), "pca": self.pca.dump_stats()}

    def load_stats(self, d: dict) -> None:
        self.scaler.load_stats(d["scaler"])
        self.pca.load_stats(d["pca"])
