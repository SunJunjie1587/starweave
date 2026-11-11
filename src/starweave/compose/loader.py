"""Compose bundles and patches, resolve plugins/configs, and hash the result."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .. import __version__
from ..kernel.errors import ComposeError
from ..kernel.plugin import validate_plugin_class
from .bundle import load_bundle
from .hash import composition_hash
from .patch import Composer
from .recipe import Recipe, load_recipe

_ENTRY_POINT_GROUP = "starweave.plugins"
_MANIFEST_ENV = "STARWEAVE_PLUGIN_MANIFEST"
_MANIFEST_FIRST_ENV = "STARWEAVE_PLUGIN_MANIFEST_FIRST"


@dataclass
class ResolvedRow:
    id: str
    plugin: str
    plugin_cls: type
    config: BaseModel
    disabled: bool
    group: bool
    source: str
    provides: tuple[str, ...]
    requires: tuple[str, ...]


@dataclass
class Composition:
    recipe: Recipe
    run_id: str
    run_dir: Path
    rows: list[ResolvedRow] = field(default_factory=list)
    by_id: dict[str, ResolvedRow] = field(default_factory=dict)
    groups: dict[str, list[ResolvedRow]] = field(default_factory=dict)
    composition_hash: str = ""

    def active(self) -> list[ResolvedRow]:
        return [row for row in self.rows if not row.disabled and not row.group]

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe": str(self.recipe.path),
            "run_id": self.run_id,
            "composition_hash": self.composition_hash,
            "rows": [
                {
                    "id": row.id,
                    "plugin": row.plugin,
                    "config": row.config.model_dump(mode="json"),
                    "disabled": row.disabled,
                    "group": row.group,
                    "source": row.source,
                    "provides": list(row.provides),
                    "requires": list(row.requires),
                }
                for row in self.rows
            ],
            "phases": self.recipe.phases,
        }


def _selected_entry_points():
    entry_points = importlib.metadata.entry_points()
    if hasattr(entry_points, "select"):
        return entry_points.select(group=_ENTRY_POINT_GROUP)
    return entry_points.get(_ENTRY_POINT_GROUP, [])  # pragma: no cover


def _load_entry_point(name: str) -> type | None:
    for entry_point in _selected_entry_points():
        if entry_point.name == name:
            return entry_point.load()
    return None


def _load_manifest() -> dict[str, str]:
    path = os.environ.get(_MANIFEST_ENV)
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ComposeError(f"cannot read plugin manifest '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ComposeError(f"plugin manifest '{path}' must be a mapping")
    return {str(key): str(value) for key, value in data.items()}


def plugin_specs() -> dict[str, str]:
    """Installed entry-point plugins plus an optional development manifest."""
    specs = {
        entry_point.name: str(entry_point.value)
        for entry_point in _selected_entry_points()
    }
    specs.update(_load_manifest())
    return dict(sorted(specs.items()))


def plugin_names() -> list[str]:
    return list(plugin_specs())


def resolve_plugin(name: str) -> type:
    if ":" in name:
        module_name, _, attr = name.partition(":")
        if not module_name or not attr:
            raise ComposeError(f"invalid plugin specifier '{name}'")
        try:
            module = importlib.import_module(module_name)
            plugin_cls = getattr(module, attr)
        except Exception as exc:  # noqa: BLE001
            raise ComposeError(f"cannot import plugin '{name}': {exc}") from exc
        validate_plugin_class(plugin_cls, row=name)
        return plugin_cls

    manifest = _load_manifest()
    spec = manifest.get(name)
    if spec and os.environ.get(_MANIFEST_FIRST_ENV) == "1":
        return resolve_plugin(spec)

    loaded = _load_entry_point(name)
    if loaded is not None:
        validate_plugin_class(loaded, row=name)
        return loaded

    if spec:
        return resolve_plugin(spec)
    raise ComposeError(
        f"unknown plugin '{name}'; install its distribution or check the entry-point name"
    )


_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_string(value: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key.startswith("env:"):
            env_name = key[4:]
            if env_name not in os.environ:
                raise ComposeError(f"environment variable '{env_name}' is not set")
            return os.environ[env_name]
        if key not in variables:
            raise ComposeError(f"unknown composition variable '{key}'")
        return variables[key]

    return _VAR_PATTERN.sub(replace, value)


def resolve_variables(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, variables)
    if isinstance(value, list):
        return [resolve_variables(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: resolve_variables(item, variables) for key, item in value.items()}
    return value


def _set_config_value(config: dict[str, Any], path: list[str], value: Any) -> None:
    current = config
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _apply_set(composer: Composer, assignment: str) -> None:
    if "=" not in assignment:
        raise ComposeError(f"--set expects key=value, got '{assignment}'")
    key, raw = assignment.split("=", 1)
    parts = key.split(".")
    if len(parts) < 2:
        raise ComposeError(f"--set key must be <row>.<field>, got '{key}'")
    row_id, config_path = parts[0], parts[1:]
    row = composer._rows.get(row_id)  # noqa: SLF001 - composer is internal to compose
    if row is None:
        raise ComposeError(f"--set targets unknown row '{row_id}'", row=row_id)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ComposeError(f"invalid --set value for '{key}': {exc}") from exc
    _set_config_value(row.config, config_path, value)
    row.source = "cli:--set"


def load_composition(
    recipe: Recipe | str | Path,
    *,
    run_id: str = "dump",
    run_dir: Path | None = None,
    set_values: list[str] | None = None,
    patch_paths: list[str] | None = None,
    settings: dict[str, Any] | None = None,
) -> Composition:
    if not isinstance(recipe, Recipe):
        recipe = load_recipe(recipe)
    composer = Composer()
    for bundle_name in recipe.bundles:
        composer.apply(load_bundle(bundle_name, root=recipe.root), f"bundle:{bundle_name}")
    composer.apply(recipe.patch, "recipe.patch")
    if settings:
        for row_id, values in settings.items():
            row = composer._rows.get(row_id)  # noqa: SLF001
            if row is None:
                raise ComposeError(f"settings target unknown row '{row_id}'", row=row_id)
            if not isinstance(values, dict):
                raise ComposeError(f"settings for '{row_id}' must be a mapping", row=row_id)
            row.config.update(values)
            row.source = "settings"
    for patch_path in patch_paths or []:
        path = Path(patch_path)
        if not path.is_file():
            raise ComposeError(f"patch overlay not found: {patch_path}")
        composer.apply(yaml.safe_load(path.read_text(encoding="utf-8")) or [], f"patch:{path}")
    for assignment in set_values or []:
        _apply_set(composer, assignment)

    resolved_run_dir = run_dir or Path("runs") / run_id
    variables = {"run.id": run_id, "paths.run_dir": str(resolved_run_dir)}

    rows: list[ResolvedRow] = []
    by_id: dict[str, ResolvedRow] = {}
    groups: dict[str, list[ResolvedRow]] = {}
    for raw in composer.rows:
        plugin_cls = resolve_plugin(raw.plugin)
        config_data = resolve_variables(raw.config, variables)
        try:
            config = plugin_cls.config_model(**config_data)
            # Resolve variables that arrive through config defaults too.
            config = plugin_cls.config_model(
                **resolve_variables(config.model_dump(mode="json"), variables)
            )
        except Exception as exc:  # noqa: BLE001
            raise ComposeError(
                f"invalid config for plugin '{raw.plugin}': {exc}", row=raw.id
            ) from exc
        resolved = ResolvedRow(
            id=raw.id,
            plugin=raw.plugin,
            plugin_cls=plugin_cls,
            config=config,
            disabled=raw.disabled,
            group=raw.group,
            source=raw.source,
            provides=tuple(plugin_cls.provides),
            requires=tuple(plugin_cls.requires),
        )
        rows.append(resolved)
        by_id[raw.id] = resolved
    for group_id, children in composer.groups.items():
        groups[group_id] = [by_id[child.id] for child in children]

    plugin_versions = {
        row.plugin: getattr(row.plugin_cls, "version", "0.1.0") for row in rows
    }
    hash_rows = [
        {
            "id": row.id,
            "plugin": row.plugin,
            "config": row.config.model_dump(mode="json"),
            "disabled": row.disabled,
        }
        for row in rows
    ]
    digest = composition_hash(
        hash_rows,
        framework_version=__version__,
        plugin_versions=plugin_versions,
    )
    return Composition(
        recipe=recipe,
        run_id=run_id,
        run_dir=resolved_run_dir,
        rows=rows,
        by_id=by_id,
        groups=groups,
        composition_hash=digest,
    )
