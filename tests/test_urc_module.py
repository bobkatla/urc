# tests/test_urc_module.py
import statistics as st
import torch as t

from urc.config import URCConfig, MDNConfig, QuantileConfig, LossConfig
from urc.core import URCModule
from urc.loss.levelset import acceptance as _acc
from urc.core.metrics import hit_rate_at_alpha as _hit
from urc.device import freeze_parameters


def test_urc_end_to_end_synthetic():
    t.manual_seed(0)
    device = "cpu"

    # Problem setup: 3 classes, 2D features
    C, D, E = 3, 2, 3
    B = 64
    steps = 90  # modest length for stability/speed

    # Class means (well separated)
    means = t.tensor(
        [[+2.0, -2.0],
         [-2.0, +2.0],
         [+2.0, +2.0]],
        dtype=t.float32, device=device
    )
    real_sigma = 0.25

    # === Key stability knobs ===
    # - var_floor raised to 1e-3 to prevent overly sharp MDN (low τ)
    # - MDN LR lowered to 5e-4 for smoother τ dynamics
    # - alpha = 0.05 (q=0.95) → more permissive τ (improves hit@α while still useful)
    cfg = URCConfig(
        phi=None,
        mdn=MDNConfig(
            d_y=D, d_e=E, n_comp=3, hidden=64,
            var_floor=1e-3,   # ↑ variance floor
            lr=5e-4           # ↓ MDN LR
        ),
        quant=QuantileConfig(
            num_classes=C, window_size=256,
            warmup_min=32,   # a bit longer warmup
            alpha=0.05       # more permissive region
        ),
        loss=LossConfig(
            w_acc=1.0, w_sep=0.0, w_size=0.0,
            margin_acc=0.0, mode_acc="softplus"
        ),
    )
    urc = URCModule(cfg, backbone=None, device=device)
    urc.train()

    # "Generator" parameters: we optimize y_gen only via URC loss to test signal flow
    y_gen_param = t.randn(B, D, device=device, requires_grad=True)
    optG = t.optim.Adam([y_gen_param], lr=1e-1)  # a bit higher LR helps move inside τ

    pre_acc_hist, post_acc_hist, hit_hist, warm_hist = [], [], [], []

    for _ in range(steps):
        # Sample a real batch
        labels = t.randint(0, C, (B,), device=device)
        e_real = t.nn.functional.one_hot(labels, num_classes=C).float()
        mu = means[labels]
        y_real = mu + real_sigma * t.randn(B, D, device=device)

        # Mild EMA toward class means to mimic concurrent base loss (helps stability)
        with t.no_grad():
            y_gen_param.data = y_gen_param.data * 0.98 + 0.02 * mu

        y_gen = y_gen_param
        e_gen = e_real.clone()

        # URC step: trains MDN on real, updates quantiles, computes acceptance on current y_gen
        out = urc.step(
            y_real=y_real, e_real=e_real, labels_real=labels,
            y_gen=y_gen,   e_gen=e_gen,   labels_gen=labels,
        )

        # Pre-step acceptance under current MDN/τ
        pre_acc = out["loss_acc"].detach()
        pre_acc_hist.append(float(pre_acc))

        # Update "generator" by URC loss
        optG.zero_grad(set_to_none=True)
        out["loss"].backward()
        optG.step()

        # Recompute acceptance/hit AFTER generator update, with SAME MDN/quant snapshot
        with freeze_parameters(urc.mdn):
            s_gen2 = -urc.mdn.log_prob(y_gen_param, e_gen)
        tau2 = urc.quant.get_tau_batch(labels, alpha=urc.cfg.quant.alpha)

        post_acc = _acc(s_gen2, tau2, mode=urc.cfg.loss.mode_acc, margin=urc.cfg.loss.margin_acc).detach()
        hit2 = _hit(s_gen2.detach(), tau2).detach()

        post_acc_hist.append(float(post_acc))
        hit_hist.append(float(hit2))
        warm_hist.append(int(t.isfinite(tau2).sum()))

    # Must have warmed thresholds for a meaningful test
    assert warm_hist[-1] > B // 2, f"too few warmed taus at end: {warm_hist[-1]} / {B}"

    # Within-step improvement: post-step acceptance < pre-step acceptance (same τ snapshot)
    improvements = [pre - post for pre, post in zip(pre_acc_hist, post_acc_hist)]
    assert st.median(improvements[-20:]) > 0.0, (
        f"Generator isn't improving acceptance within-step: "
        f"median(pre-post last 20)={st.median(improvements[-20:]):.4f}"
    )

    # Coverage shouldn't collapse entirely (loose sanity)
    last10_hit = sum(hit_hist[-10:]) / 10.0
    assert last10_hit > 0.25, f"hit@alpha too low at end: {last10_hit:.3f}"
