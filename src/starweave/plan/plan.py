"""Frozen execution-plan structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..seams.types import StepPlan


@dataclass(frozen=True)
class ServiceBinding:
    id: str
    seam: str
    provider: str
    source: str
    pinned: bool = False
    rebindable: bool = False
    instance: Any = None


@dataclass(frozen=True)
class PhasePlan:
    name: str
    services: dict[str, ServiceBinding]
    step: StepPlan | None
    hooks: tuple[Any, ...]
    composition_hash: str


@dataclass(frozen=True)
class Diagnostic:
    level: str
    code: str
    message: str


@dataclass(frozen=True)
class ExecutionPlan:
    run_id: str
    recipe_path: str
    phases: dict[str, PhasePlan]
    pinned: frozenset[str] = field(default_factory=frozenset)
    rebindable: frozenset[str] = field(default_factory=frozenset)
    diagnostics: tuple[Diagnostic, ...] = ()


def format_plan(plan: ExecutionPlan) -> str:
    lines = [
        f"Run: {plan.run_id}",
        f"Recipe: {plan.recipe_path}",
        "",
        "Services:",
    ]
    seen: set[str] = set()
    for phase in plan.phases.values():
        for seam, binding in phase.services.items():
            if seam in seen:
                continue
            seen.add(seam)
            kind = "pinned" if binding.pinned else "rebindable" if binding.rebindable else "service"
            lines.append(
                f"  [{kind:<10}] {seam:<12} {binding.provider:<24} "
                f"source={binding.source}"
            )
    lines.append("")
    lines.append("Phases:")
    for name, phase in plan.phases.items():
        services = ", ".join(f"{seam}={binding.id}" for seam, binding in phase.services.items())
        hooks = ", ".join(str(hook) for hook in phase.hooks) or "-"
        lines.append(f"  {name}:")
        lines.append(f"    services: {services}")
        lines.append(f"    steps: {'static' if phase.step is None else 'compiled'}")
        lines.append(f"    hooks: {hooks}")
        lines.append(f"    composition_hash: {phase.composition_hash}")
    lines.append("")
    lines.append("Diagnostics:")
    if plan.diagnostics:
        lines.extend(f"  - {item.level} [{item.code}] {item.message}" for item in plan.diagnostics)
    else:
        lines.append("  - none")
    return "\n".join(lines)
