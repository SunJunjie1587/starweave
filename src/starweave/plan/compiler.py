"""Static plan compilation."""

from __future__ import annotations

from ..compose.loader import Composition
from .plan import ExecutionPlan
from .validator import validate_composition


def compile_static(
    composition: Composition,
    phase_names: list[str] | None = None,
) -> ExecutionPlan:
    phases, diagnostics = validate_composition(composition, phase_names)
    return ExecutionPlan(
        run_id=composition.run_id,
        recipe_path=str(composition.recipe.path),
        phases=phases,
        pinned=frozenset(
            seam for phase in phases.values() for seam, b in phase.services.items() if b.pinned
        ),
        rebindable=frozenset(
            seam
            for phase in phases.values()
            for seam, b in phase.services.items()
            if b.rebindable
        ),
        diagnostics=diagnostics,
    )
