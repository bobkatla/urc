from contextlib import contextmanager
import torch

@contextmanager
def autocast_guard(dtype=torch.float32):
    # Placeholder for AMP control if needed later
    try:
        yield
    finally:
        pass

@contextmanager
def freeze_parameters(module: torch.nn.Module):
    """
    Temporarily disables grads for module parameters so that
    grads flow only to inputs (e.g., y_gen) during a forward pass.
    """
    prev = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, r in zip(module.parameters(), prev):
            p.requires_grad_(r)
