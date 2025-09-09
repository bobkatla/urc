import numpy as np
import torch as t
from urc.quantiles import PerClassFIFOQuantiles

def test_fifo_quantiles_matches_numpy_recent_window():
    t.manual_seed(0)
    C, W = 3, 64
    alpha = 0.1
    q = 1.0 - alpha
    qstream = PerClassFIFOQuantiles(num_classes=C, window_size=W, warmup_min=16)

    # simulate 500 steps with random class and scores
    labels = t.randint(0, C, (500,), dtype=t.long)
    scores = t.randn(500)

    # Python reference buffers
    ref = {c: [] for c in range(C)}
    taus_ref = []

    for i in range(500):
        c = int(labels[i].item())
        s = float(scores[i].item())
        ref[c].append(s)
        if len(ref[c]) > W:
            ref[c].pop(0)
        qstream.update_many(labels[i:i+1], scores[i:i+1])
        # Query tau for this label
        tau_t = qstream.get_tau_batch(labels[i:i+1], alpha=alpha)[0].item()

        if len(ref[c]) < 16:
            assert np.isinf(tau_t)
        else:
            tau_np = float(np.quantile(np.array(ref[c], dtype=np.float32), q))
            assert abs(tau_t - tau_np) < 0.15  # loose tolerance due to small samples

def test_fifo_state_save_load_roundtrip():
    C, W = 2, 8
    q1 = PerClassFIFOQuantiles(C, W, warmup_min=4)
    labels = t.tensor([0,1,0,0,1,1,0,1], dtype=t.long)
    scores = t.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    q1.update_many(labels, scores)

    state = q1.state_dict()
    q2 = PerClassFIFOQuantiles(C, W, warmup_min=4)
    q2.load_state_dict(state)

    alpha = 0.25
    test_labels = t.tensor([0,1,0,1], dtype=t.long)
    tau1 = q1.get_tau_batch(test_labels, alpha=alpha)
    tau2 = q2.get_tau_batch(test_labels, alpha=alpha)
    assert t.allclose(tau1, tau2)
