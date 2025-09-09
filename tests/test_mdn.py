# tests/test_mdn.py
import torch as t
from urc.density.mdn import CondMDN

def test_logprob_finite_and_shapes():
    B, D, E, K = 32, 5, 3, 4
    y = t.randn(B, D)
    e = t.randn(B, E)
    mdn = CondMDN(d_y=D, d_e=E, n_comp=K, hidden=32, var_floor=1e-4)
    lp = mdn.log_prob(y, e)
    assert lp.shape == (B,)
    assert t.isfinite(lp).all()

def test_fit_step_reduces_nll_on_toy_data():
    t.manual_seed(0)
    B, D, E, K = 256, 2, 2, 3

    # Toy conditional structure: component depends on sign of e[:,0]
    e = t.randn(B, E)
    comp = (e[:, 0] > 0).long()  # 0 or 1
    means = t.stack([t.tensor([+2.0, -2.0]), t.tensor([-2.0, +2.0])])  # [2, D]
    y = means[comp] + 0.1 * t.randn(B, D)

    mdn = CondMDN(d_y=D, d_e=E, n_comp=K, hidden=64, var_floor=1e-4)
    opt = t.optim.Adam(mdn.parameters(), lr=5e-3)

    with t.no_grad():
        nll0 = float(mdn.nll(y, e).item())

    nll_last = nll0
    for _ in range(200):
        nll_last = mdn.fit_step(y, e, opt, clip_grad=1.0)

    assert nll_last < nll0 * 0.6, f"NLL drop too small: start={nll0:.3f}, end={nll_last:.3f}"
