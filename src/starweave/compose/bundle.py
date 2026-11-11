"""Bundle loading. A bundle is a YAML list of patch entries."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..kernel.errors import ComposeError


def load_bundle(name_or_path: str, *, root: Path) -> list[dict]:
    candidate = Path(name_or_path)
    candidates = []
    if candidate.suffix in {".yaml", ".yml"} or candidate.exists():
        candidates.append(candidate)
    candidates.append(root / "bundles" / f"{name_or_path}.yaml")
    for path in candidates:
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if data is None:
                return []
            if not isinstance(data, list):
                raise ComposeError(f"bundle '{path}' must be a YAML list")
            return data
    raise ComposeError(f"bundle not found: {name_or_path}")
