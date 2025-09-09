from __future__ import annotations
from typing import Dict, List
import torch

class PerClassFIFOQuantiles:
    """Streaming per-class (1-α)-quantiles over a fixed-size FIFO window.

    Maintains a ring buffer of scores per class (on the same device as inputs).
    Warmup: until a class has seen at least `warmup_min` samples, `get_tau_batch`
    returns +inf for that class (so acceptance losses can be masked).

    Args:
        num_classes: number of discrete classes.
        window_size: window length per class.
        warmup_min: minimum samples before the class is considered warm.

    Methods:
        update_many(labels: LongTensor[B], scores: Tensor[B])
        get_tau_batch(labels: LongTensor[B], alpha: float) -> Tensor[B]  # taus for each label
        size(cls: int) -> int
        state_dict() / load_state_dict(state)
    """
    def __init__(self, num_classes: int, window_size: int, warmup_min: int = 128):
        self.C = int(num_classes)
        self.W = int(window_size)
        self.warmup_min = int(warmup_min)

        # Buffers are created lazily on first update to match device/dtype of scores.
        self._buffers: List[torch.Tensor] = []
        self._ptr = torch.zeros(self.C, dtype=torch.long)     # write pointer per class
        self._count = torch.zeros(self.C, dtype=torch.long)   # seen count per class
        self._device = None
        self._dtype = None

        # Internal scratch to avoid reallocs
        self._class_mask = None

    def _lazy_init(self, scores: torch.Tensor):
        if self._device is None:
            self._device = scores.device
            self._dtype = scores.dtype
            self._ptr = self._ptr.to(self._device)
            self._count = self._count.to(self._device)
            self._buffers = [torch.empty(self.W, dtype=self._dtype, device=self._device) for _ in range(self.C)]

    @torch.no_grad()
    def update_many(self, labels: torch.Tensor, scores: torch.Tensor) -> None:
        assert labels.dtype == torch.long, "labels must be int64"
        assert labels.numel() == scores.numel(), "labels and scores must align"
        self._lazy_init(scores)

        # For simplicity, process per class present in this batch
        unique = labels.unique()
        for cls in unique.tolist():
            mask = (labels == cls)
            bs = int(mask.sum().item())
            if bs == 0:
                continue
            s = scores[mask]  # [bs]
            # write into ring buffer
            ptr = int(self._ptr[cls].item())
            cnt = int(self._count[cls].item())

            # If bs > W, keep only the most recent W scores
            if bs >= self.W:
                s = s[-self.W:]
                bs = s.numel()

            end = ptr + bs
            if end <= self.W:
                self._buffers[cls][ptr:end] = s
            else:
                first = self.W - ptr
                self._buffers[cls][ptr:] = s[:first]
                self._buffers[cls][: end - self.W] = s[first:]

            ptr = (ptr + bs) % self.W
            cnt = min(self.W, cnt + bs)

            self._ptr[cls] = ptr
            self._count[cls] = cnt

    @torch.no_grad()
    def get_tau_batch(self, labels: torch.Tensor, alpha: float) -> torch.Tensor:
        """Return per-sample tau for labels; +inf where class not warm.
        """
        assert 0.0 < alpha < 1.0, "alpha must be in (0,1)"
        device = labels.device
        taus = torch.full((labels.numel(),), float('inf'), dtype=self._dtype or torch.float32, device=device)
        if self._device is None:
            return taus  # nothing seen yet

        unique = labels.unique()
        q = 1.0 - alpha
        for cls in unique.tolist():
            idx = (labels == cls).nonzero(as_tuple=False).squeeze(-1)
            n = int(self._count[cls].item())
            if n < self.warmup_min or n == 0:
                taus[idx] = float('inf')
                continue
            buf = self._buffers[cls]
            # Gather last n items in temporal order: [ptr - n, ptr) in ring
            ptr = int(self._ptr[cls].item())
            if n == self.W:
                window = buf
            else:
                # reconstruct last n regardless of wrap
                start = (ptr - n) % self.W
                if start < ptr:
                    window = buf[start:ptr]
                else:
                    window = torch.cat([buf[start:], buf[:ptr]], dim=0)
            tau = torch.quantile(window, q)
            taus[idx] = tau
        return taus

    def size(self, cls: int) -> int:
        return int(self._count[cls].item())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        # Flatten buffers for simple serialization
        flat = torch.stack(self._buffers, dim=0) if self._buffers else torch.empty(0)
        return {
            "buffers": flat,
            "ptr": self._ptr,
            "count": self._count,
            "W": torch.tensor(self.W),
            "warmup_min": torch.tensor(self.warmup_min),
        }

    def load_state_dict(self, state) -> None:
        self.W = int(state["W"].item()) if isinstance(state["W"], torch.Tensor) else int(state["W"])
        self.warmup_min = int(state["warmup_min"].item()) if isinstance(state["warmup_min"], torch.Tensor) else int(state["warmup_min"])
        flat = state["buffers"]
        if flat.numel() == 0:
            self._buffers = []
            self._device = None
            self._dtype = None
        else:
            self._buffers = [flat[i].clone() for i in range(flat.shape[0])]
            self._device = flat.device
            self._dtype = flat.dtype
        self._ptr = state["ptr"].clone().to(self._device) if self._device is not None else state["ptr"].clone()
        self._count = state["count"].clone().to(self._device) if self._device is not None else state["count"].clone()
