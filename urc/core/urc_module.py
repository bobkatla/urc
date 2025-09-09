from __future__ import annotations
from typing import Optional, Callable, Dict
import torch
from torch import nn

from urc.config import URCConfig
from urc.phi.image_phi import PhiImage
from urc.density.mdn import CondMDN
from urc.quantiles import PerClassFIFOQuantiles
from urc.loss.levelset import acceptance as acc_loss_fn, separation as sep_loss_fn, size_proxy
from urc.core.metrics import hit_rate_at_alpha, mean_finite
from urc.device import freeze_parameters

class URCModule(nn.Module):
    """
    Trains a conditional MDN on real (y,e), maintains per-class score thresholds,
    and returns a scalar region-aware loss on generated samples.

    out = urc.step(
        y_real, e_real, labels_real,
        y_gen,  e_gen,  labels_gen,
        y_gen_neg=None, e_gen_neg=None, labels_neg=None,
    )
    """
    def __init__(
        self,
        config: URCConfig,
        backbone: Optional[Callable] = None,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.cfg = config

        # Optional φ
        self.phi: Optional[PhiImage] = None
        if self.cfg.phi is not None:
            self.phi = PhiImage(backbone=backbone, d_out=self.cfg.phi.d_out, pca_k=self.cfg.phi.pca_k)

        # MDN
        m = self.cfg.mdn
        self.mdn = CondMDN(
            d_y=m.d_y, d_e=m.d_e, n_comp=m.n_comp, hidden=m.hidden,
            var_floor=m.var_floor, use_layernorm=m.use_layernorm
        )
        self.mdn_opt = torch.optim.Adam(self.mdn.parameters(), lr=m.lr)

        # Quantiles
        q = self.cfg.quant
        self.quant = PerClassFIFOQuantiles(q.num_classes, q.window_size, q.warmup_min)

        self.loss_cfg = self.cfg.loss
        if device is not None:
            self.to(device)

    @torch.no_grad()
    def project_phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.phi is None:
            return x
        return self.phi.project(x)

    def _labels_from_e(self, e: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if labels is not None:
            assert labels.dtype == torch.long and labels.dim() == 1
            return labels
        if e.dtype.is_floating_point and e.dim() == 2:
            return e.argmax(dim=-1).long()
        raise ValueError("Provide labels or pass one-hot/soft labels in e to infer via argmax.")

    def step(
        self,
        *,
        y_real: torch.Tensor,
        e_real: torch.Tensor,
        labels_real: Optional[torch.Tensor] = None,
        y_gen: torch.Tensor,
        e_gen: torch.Tensor,
        labels_gen: Optional[torch.Tensor] = None,
        y_gen_neg: Optional[torch.Tensor] = None,
        e_gen_neg: Optional[torch.Tensor] = None,
        labels_neg: Optional[torch.Tensor] = None,
        clip_grad_mdn: Optional[float] = 1.0,
    ) -> Dict[str, torch.Tensor]:

        device = y_real.device
        alpha = self.cfg.quant.alpha

        # 1) Update MDN on real
        mdn_loss = self.mdn.nll(y_real, e_real)
        self.mdn_opt.zero_grad(set_to_none=True)
        mdn_loss.backward()
        if clip_grad_mdn is not None:
            torch.nn.utils.clip_grad_norm_(self.mdn.parameters(), max_norm=clip_grad_mdn)
        self.mdn_opt.step()

        # 2) Stream thresholds with real scores
        with torch.no_grad():
            s_real = -self.mdn.log_prob(y_real, e_real)
        lab_real = self._labels_from_e(e_real, labels_real)
        self.quant.update_many(lab_real, s_real)

        # 3) Acceptance on generated (freeze MDN so grads go to y_gen/upstream)
        lab_gen = self._labels_from_e(e_gen, labels_gen)
        with freeze_parameters(self.mdn):
            s_gen = -self.mdn.log_prob(y_gen, e_gen)
        tau = self.quant.get_tau_batch(lab_gen, alpha=alpha)
        loss_acc = acc_loss_fn(s_gen, tau, mode=self.loss_cfg.mode_acc, margin=self.loss_cfg.margin_acc)

        # 4) Optional separation
        loss_sep = torch.zeros((), dtype=y_real.dtype, device=device)
        if y_gen_neg is not None and e_gen_neg is not None:
            lab_neg = self._labels_from_e(e_gen_neg, labels_neg)
            with freeze_parameters(self.mdn):
                s_wrong = -self.mdn.log_prob(y_gen_neg, e_gen_neg)
            tau_wrong = self.quant.get_tau_batch(lab_neg, alpha=alpha)
            loss_sep = sep_loss_fn(s_wrong, tau_wrong, margin=self.loss_cfg.margin_sep, mode=self.loss_cfg.mode_sep)

        # 5) Size proxy over in-region gen points
        valid = torch.isfinite(tau)
        in_region = valid & (s_gen <= tau)
        sp_val = size_proxy(y_gen[in_region].detach())

        # 6) Total
        total = (self.loss_cfg.w_acc * loss_acc
                 + self.loss_cfg.w_sep * loss_sep
                 + self.loss_cfg.w_size * sp_val)

        # 7) Diagnostics
        hit = hit_rate_at_alpha(s_gen.detach(), tau)
        tau_mean = mean_finite(tau)
        n_warm = (torch.isfinite(tau)).sum()

        return {
            "loss": total,
            "loss_acc": loss_acc.detach(),
            "loss_sep": loss_sep.detach(),
            "size_proxy": sp_val.detach(),
            "hit_at_alpha": hit.detach(),
            "tau_mean": tau_mean.detach(),
            "n_warm": n_warm.detach(),
            "mdn_nll_real": mdn_loss.detach(),
        }
