"""dump-config and dump-schema."""

from __future__ import annotations

import json

from ..compose.loader import load_composition, plugin_names, resolve_plugin
from ..compose.recipe import load_recipe


def format_composition(composition) -> str:
    data = composition.as_dict()
    lines = [
        f"Recipe: {data['recipe']}",
        f"Run: {data['run_id']}",
        f"Composition hash: {data['composition_hash']}",
        "",
        "Rows:",
    ]
    for row in data["rows"]:
        flags = []
        if row["disabled"]:
            flags.append("disabled")
        if row["group"]:
            flags.append("group")
        suffix = f" ({', '.join(flags)})" if flags else ""
        lines.append(f"  [{row['id']}] {row['plugin']}{suffix}")
        lines.append(f"    source: {row['source']}")
        lines.append(f"    provides: {', '.join(row['provides'])}")
        lines.append(f"    requires: {', '.join(row['requires']) or '-'}")
        lines.append(f"    config: {json.dumps(row['config'], sort_keys=True)}")
    lines.append("")
    lines.append("Phases:")
    for name, services in data["phases"].items():
        lines.append(f"  {name}: {json.dumps(services, sort_keys=True)}")
    return "\n".join(lines)


def dump_config(
    recipe_path: str,
    *,
    run_id: str = "dump",
    set_values: list[str] | None = None,
    patch_paths: list[str] | None = None,
    as_json: bool = False,
) -> int:
    recipe = load_recipe(recipe_path)
    composition = load_composition(
        recipe,
        run_id=run_id,
        set_values=set_values,
        patch_paths=patch_paths,
    )
    if as_json:
        print(json.dumps(composition.as_dict(), indent=2, sort_keys=True))
    else:
        print(format_composition(composition))
    return 0


def _plugin_names() -> list[str]:
    return plugin_names()


def dump_schema(plugin_name: str | None = None) -> int:
    if plugin_name:
        plugin_cls = resolve_plugin(plugin_name)
        print(json.dumps(plugin_cls.config_model.model_json_schema(), indent=2, sort_keys=True))
        return 0
    result = {}
    for name in _plugin_names():
        try:
            plugin_cls = resolve_plugin(name)
            result[name] = plugin_cls.config_model.model_json_schema()
        except Exception as exc:  # noqa: BLE001
            result[name] = {"error": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
