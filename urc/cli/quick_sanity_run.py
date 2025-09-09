import torch as t
from urc.phi.image_phi import PhiImage
import click

@click.command("sanity-run")
def sanity_run():
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    Y = t.randn(2048, 128, device=device)
    phi = PhiImage(backbone=None, d_out=128, pca_k=32).to(device)
    phi.fit_stats(Y)
    Z = phi.project(Y)
    print({'device': device, 'shape': tuple(Z.shape), 'mean': Z.mean().item(), 'std': Z.std().item()})

