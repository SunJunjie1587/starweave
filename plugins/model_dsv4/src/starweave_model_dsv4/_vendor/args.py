"""Minimal shim for MindSpeed's global args.

The vendored modeling code treats a missing args object as "use the torch
fallback" (its ``_fsdp_args`` helper catches ValueError). Starweave does not
initialize MindSpeed's global training args, so this shim always raises.
"""

from __future__ import annotations


def get_args():
    raise ValueError(
        "MindSpeed args are not initialized; Starweave uses the torch fallback path"
    )
