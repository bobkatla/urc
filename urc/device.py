from contextlib import contextmanager
import torch

@contextmanager
def autocast_guard(dtype=torch.float32):
    # Simple guard to ensure consistent dtype; extend later if needed
    try:
        yield
    finally:
        pass
