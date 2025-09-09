from typing import Callable, Dict, Optional

_REGISTRY: Dict[str, Callable] = {}

def register(name: str, fn: Callable):
    if name in _REGISTRY:
        raise ValueError(f"Backbone '{name}' is already registered.")
    _REGISTRY[name] = fn

def get(name: str) -> Optional[Callable]:
    return _REGISTRY.get(name, None)
