# tests/test_io_ckpt.py
import os
import pytest
import torch as t

from urc.config import URCConfig, MDNConfig, QuantileConfig, LossConfig, PhiConfig
from urc.core import URCModule
from urc.io import save_urc, load_urc


def _mk_cfg(phi: bool = False):
    C, D, E = 3, 2, 3
    return URCConfig(
        phi=PhiConfig(d_out=16, pca_k=8) if phi else None,
        mdn=MDNConfig(d_y=D, d_e=E, n_comp=3, hidden=32, var_floor=1e-3, lr=1e-3),
        quant=QuantileConfig(num_classes=C, window_size=64, warmup_min=8, alpha=0.1),
        loss=LossConfig(w_acc=1.0, w_sep=0.0, w_size=0.0, margin_acc=0.0, mode_acc="softplus"),
    )


@pytest.mark.parametrize("with_phi", [False, True])
def test_save_load_parity(tmp_path, with_phi):
    t.manual_seed(0)
    device = "cpu"

    cfg = _mk_cfg(phi=with_phi)
    urc = URCModule(cfg, backbone=None, device=device)
    urc.train()

    # Optionally fit φ stats to make sure they are saved
    if with_phi and urc.phi is not None:
        Y = t.randn(256, cfg.phi.d_out, device=device)
        with t.no_grad():
            urc.phi.fit_stats(Y)

    # Create some activity: train MDN a few steps and update quantiles
    C, D, E = cfg.quant.num_classes, cfg.mdn.d_y, cfg.mdn.d_e
    means = t.tensor([[+2.0, -2.0], [-2.0, +2.0], [+2.0, +2.0]], dtype=t.float32, device=device)
    real_sigma = 0.25

    B = 64
    for _ in range(30):
        labels = t.randint(0, C, (B,), device=device)
        e_real = t.nn.functional.one_hot(labels, num_classes=C).float()
        mu = means[labels]
        y_real = mu + real_sigma * t.randn(B, D, device=device)

        # dummy generated batch (doesn't matter for save test)
        y_gen = mu + 0.5 * t.randn(B, D, device=device)
        e_gen = e_real.clone()

        urc.step(
            y_real=y_real, e_real=e_real, labels_real=labels,
            y_gen=y_gen,   e_gen=e_gen,   labels_gen=labels,
        )

    # Capture a reference forward (scores & taus) on a fixed batch
    labels = (t.arange(B, device=device) % C)
    e = t.nn.functional.one_hot(labels, num_classes=C).float()
    y = means[labels] + 0.3 * t.randn(B, D, device=device)

    urc.eval()
    with t.no_grad():
        s_ref = -urc.mdn.log_prob(y, e)
        tau_ref = urc.quant.get_tau_batch(labels, alpha=cfg.quant.alpha)

    # Save & load
    ckpt_path = tmp_path / "urc_ckpt.pt"
    save_urc(ckpt_path, urc, include_optimizer=True)
    assert os.path.exists(ckpt_path)

    urc2 = load_urc(ckpt_path, map_location=device)
    urc2.eval()

    with t.no_grad():
        s_new = -urc2.mdn.log_prob(y, e)
        tau_new = urc2.quant.get_tau_batch(labels, alpha=cfg.quant.alpha)

    # Parity checks
    assert t.allclose(s_ref, s_new, atol=1e-6), "MDN log_prob parity failed after load"
    assert t.allclose(tau_ref, tau_new, atol=1e-6), "Quantile τ parity failed after load"

    # φ presence respected
    if with_phi:
        assert urc2.phi is not None, "Expected φ to be constructed"
    else:
        assert urc2.phi is None, "Expected no φ in config"
