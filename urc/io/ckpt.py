# urc/io/ckpt.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Callable, Any, Dict
import torch

from urc.config import URCConfig
from urc.core.urc_module import URCModule
from urc.core.state import current_state_version, migrate_state


def _config_to_dict(cfg: URCConfig) -> dict:
    return cfg.model_dump()


def _config_from_dict(d: dict) -> URCConfig:
    # pydantic v2
    return URCConfig(**d)


def save_urc(path: Union[str, Path], urc: URCModule, *, include_optimizer: bool = False) -> None:
    """
    Save a URCModule checkpoint: version, config, φ stats, MDN weights, quantile buffers.

    Args:
        path: file path to save to (e.g., "ckpt_urc.pt").
        urc:  URCModule instance.
        include_optimizer: if True, also saves MDN optimizer state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "version": current_state_version(),
        "config": _config_to_dict(urc.cfg),
        "phi": None,
        "mdn": urc.mdn.state_dict(),
        "quant": urc.quant.state_dict(),
    }
    if urc.phi is not None:
        payload["phi"] = urc.phi.dump_stats()
    if include_optimizer and hasattr(urc, "mdn_opt") and urc.mdn_opt is not None:
        payload["mdn_opt"] = urc.mdn_opt.state_dict()

    torch.save(payload, path)


def load_urc(
    path: Union[str, Path],
    *,
    config: Optional[URCConfig] = None,
    backbone: Optional[Callable] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> URCModule:
    """
    Load a URCModule from a checkpoint produced by `save_urc`.

    Args:
        path: checkpoint path.
        config: if None, uses the config embedded in the checkpoint;
                if provided, overrides the embedded config.
        backbone: optional φ backbone to construct PhiImage when config.phi is not None.
        map_location: torch.load map_location.
        strict: whether to enforce strict MDN state_dict loading.

    Returns:
        A URCModule with restored φ stats, MDN weights, and quantile buffers.
    """
    path = Path(path)
    state = torch.load(path, map_location=map_location)
    state = migrate_state(state)

    cfg = config if config is not None else _config_from_dict(state["config"])
    urc = URCModule(cfg, backbone=backbone, device=map_location)

    # MDN weights
    urc.mdn.load_state_dict(state["mdn"], strict=strict)

    # Quantiles
    urc.quant.load_state_dict(state["quant"])

    # φ stats (if present/configured)
    if cfg.phi is not None and state.get("phi", None) is not None and urc.phi is not None:
        urc.phi.load_stats(state["phi"])

    # Optimizer (optional)
    if "mdn_opt" in state and hasattr(urc, "mdn_opt") and urc.mdn_opt is not None:
        try:
            urc.mdn_opt.load_state_dict(state["mdn_opt"])
        except Exception:
            # Optimizer failures shouldn't block model load
            pass

    return urc
