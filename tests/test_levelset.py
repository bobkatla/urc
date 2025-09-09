import numpy as np
import torch as t
from urc.loss.levelset import acceptance
from urc.loss.levelset import separation
from urc.loss.levelset import size_proxy


def test_acceptance_masks_inf_and_matches_manual():
    scores = t.tensor([0.1, 0.5, 1.0, 2.0, -0.2, 0.0])
    tau    = t.tensor([0.2, 0.4, float('inf'), 1.5, float('inf'), 0.0])
    # valid indices: 0,1,3,5
    v = scores - tau  # margin=0
    manual = t.nn.functional.softplus(v[[0,1,3,5]]).mean()
    loss = acceptance(scores, tau, mode="softplus", margin=0.0)
    assert t.allclose(loss, manual)

def test_separation_logic_direction():
    # Want scores >= tau + margin; penalize if below
    scores_wrong = t.tensor([0.2, 0.6, 0.7])
    tau_wrong    = t.tensor([0.5, 0.5, 0.5])
    margin = 0.1  # boundary at 0.6
    # Only first is violating by 0.4
    loss = separation(scores_wrong, tau_wrong, margin=margin, mode="hinge")
    assert abs(loss.item() - 0.4) < 1e-6

def test_size_proxy_behaviour_and_stability():
    t.manual_seed(0)
    N, D = 256, 5
    Y = t.randn(N, D)
    s1 = size_proxy(Y, eps=1e-6)
    s2 = size_proxy(2.0 * Y, eps=1e-6)
    # Scaling by 2 multiplies covariance by 4; logdet increases by D*log(4) = 2*D*log(2)
    expected_delta = 2 * D * np.log(2.0)
    assert abs((s2 - s1).item() - expected_delta) < 0.1

    # Stability when N < D (and even for tiny N)
    Y_small = t.randn(2, 7)
    s_small = size_proxy(Y_small, eps=1e-4)
    assert t.isfinite(s_small).all()
