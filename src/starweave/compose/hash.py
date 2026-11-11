"""Canonical composition hash."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def composition_hash(
    rows: list[dict[str, Any]],
    *,
    framework_version: str,
    plugin_versions: dict[str, str] | None = None,
) -> str:
    payload = {
        "framework": framework_version,
        "plugins": plugin_versions or {},
        "rows": rows,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
