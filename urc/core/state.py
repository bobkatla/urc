# urc/core/state.py
from __future__ import annotations

STATE_VERSION = "0.1.0"  # bump when checkpoint schema changes


def current_state_version() -> str:
    """Return the current URC checkpoint state version."""
    return STATE_VERSION


def migrate_state(state: dict) -> dict:
    """
    Hook for migrating older checkpoints to the current schema.
    For v0.1.0 there is nothing to do; we just ensure a version is present.
    """
    if "version" not in state:
        state["version"] = "0.1.0"
    # Add future migrations here with if/elif on state["version"].
    return state
