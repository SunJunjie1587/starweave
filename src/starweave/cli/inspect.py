"""inspect --plan: compile the static plan without touching a device."""

from __future__ import annotations

from ..compose.loader import load_composition
from ..compose.recipe import load_recipe
from ..plan.compiler import compile_static
from ..plan.plan import format_plan


def inspect_plan(
    recipe_path: str,
    *,
    run_id: str = "dump",
    set_values: list[str] | None = None,
    patch_paths: list[str] | None = None,
    phase: str | None = None,
) -> int:
    recipe = load_recipe(recipe_path)
    composition = load_composition(
        recipe,
        run_id=run_id,
        set_values=set_values,
        patch_paths=patch_paths,
    )
    phases = [phase] if phase else None
    plan = compile_static(composition, phases)
    print(format_plan(plan))
    return 0
