import json
import click
import torch as t
from urc.phi.image_phi import PhiImage
from urc.density.mdn import CondMDN

@click.group("sanity")
def sanity():
    """Quick sanity checks."""
    pass

@sanity.command("phi-check")
@click.option("--n", default=2048, show_default=True, help="Batch size for synthetic features.")
@click.option("--d-out", default=128, show_default=True, help="Backbone feature dim before PCA.")
@click.option("--pca-k", default=32, show_default=True, help="PCA output dim.")
def phi_check(n: int, d_out: int, pca_k: int):
    """Run a small φ projection sanity check."""
    device = "cuda" if t.cuda.is_available() else "cpu"
    Y = t.randn(n, d_out, device=device)
    phi = PhiImage(backbone=None, d_out=d_out, pca_k=pca_k).to(device)
    phi.fit_stats(Y)
    Z = phi.project(Y)
    click.echo(json.dumps({
        "device": device,
        "shape": list(Z.shape),
        "mean": float(Z.mean().item()),
        "std": float(Z.std().item())
    }))


@sanity.command("mdn-toy-fit")
@click.option("--steps", default=200, show_default=True)
@click.option("--lr", default=5e-3, show_default=True)
def mdn_toy_fit(steps: int, lr: float):
    t.manual_seed(0)
    B, D, E, K = 512, 2, 2, 3
    e = t.randn(B, E)
    comp = (e[:, 0] > 0).long()
    means = t.stack([t.tensor([+2.0, -2.0]), t.tensor([-2.0, +2.0])])
    y = means[comp] + 0.1 * t.randn(B, D)

    mdn = CondMDN(d_y=D, d_e=E, n_comp=K, hidden=64, var_floor=1e-4)
    opt = t.optim.Adam(mdn.parameters(), lr=lr)

    with t.no_grad():
        nll0 = float(mdn.nll(y, e).item())

    nll = nll0
    for _ in range(steps):
        nll = mdn.fit_step(y, e, opt, clip_grad=1.0)

    click.echo(json.dumps({"nll_start": nll0, "nll_end": nll, "improvement": nll0 - nll}))
