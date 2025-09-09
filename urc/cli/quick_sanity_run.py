import json
import click
import torch as t
from urc.phi.image_phi import PhiImage

@click.command("sanity-run")
@click.option("--n", default=2048, show_default=True, help="Batch size for synthetic features.")
@click.option("--d-out", default=128, show_default=True, help="Backbone feature dim before PCA.")
@click.option("--pca-k", default=32, show_default=True, help="PCA output dim.")
def sanity_run(n: int, d_out: int, pca_k: int):
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
