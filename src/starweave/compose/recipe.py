"""Recipe loading. A recipe is bundles + patch + phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..kernel.errors import ComposeError


@dataclass
class Recipe:
    name: str
    path: Path
    root: Path
    bundles: list[str] = field(default_factory=list)
    patch: list[dict] = field(default_factory=list)
    phases: dict[str, dict[str, str]] = field(default_factory=dict)


def load_recipe(path: str | Path) -> Recipe:
    recipe_path = Path(path)
    if not recipe_path.is_file():
        raise ComposeError(f"recipe not found: {recipe_path}")
    data = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ComposeError(f"recipe '{recipe_path}' must be a mapping")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ComposeError(f"recipe '{recipe_path}' is missing 'name'")
    bundles = data.get("bundles") or []
    if not isinstance(bundles, list) or not all(isinstance(b, str) for b in bundles):
        raise ComposeError(f"recipe '{recipe_path}' has invalid 'bundles'")
    patch = data.get("patch") or []
    if not isinstance(patch, list):
        raise ComposeError(f"recipe '{recipe_path}' has invalid 'patch'")
    phases = data.get("phases") or {}
    if not isinstance(phases, dict):
        raise ComposeError(f"recipe '{recipe_path}' has invalid 'phases'")
    return Recipe(
        name=name,
        path=recipe_path,
        root=recipe_path.parent.parent,
        bundles=bundles,
        patch=patch,
        phases=phases,
    )
