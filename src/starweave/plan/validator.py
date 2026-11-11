"""Static composition validation and phase binding."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from ..compose.loader import Composition, ResolvedRow
from ..kernel.errors import ComposeError
from .plan import Diagnostic, PhasePlan, ServiceBinding

EXCLUSIVE_SEAMS = {
    "device",
    "precision",
    "trainer",
    "inference",
    "task",
    "rollout",
    "reward",
    "algorithm",
    "launcher",
    "collective",
    "fsdp",
    "ep",
}
REGISTRY_SEAMS = {
    "kernel",
    "model",
    "data",
    "optimizer",
    "scheduler",
    "loss",
    "checkpoint",
    "metrics",
    "profiler",
}
POLICY_SEAMS = {"step_policy", "checkpoint_policy", "eval_policy"}

PINNED_SEAMS = {
    "trainer",
    "task",
    "rollout",
    "reward",
    "algorithm",
    "optimizer",
    "scheduler",
    "model",
    "collective",
    "fsdp",
    "ep",
}
REBINDABLE_SEAMS = {"metrics", "profiler", "checkpoint", "inference"}

TRAIN_REQUIRED = {
    "device",
    "trainer",
    "optimizer",
    "scheduler",
    "model",
    "data",
    "collective",
    "fsdp",
    "ep",
}
INFERENCE_REQUIRED = {"device", "model", "inference"}
RL_REQUIRED = {
    "device",
    "precision",
    "model",
    "data",
    "rollout",
    "reward",
    "algorithm",
    "task",
    "optimizer",
    "scheduler",
    "checkpoint",
    "metrics",
}

PHASE_REQUIRED = {
    "train": TRAIN_REQUIRED,
    "eval": {"device", "model", "data", "metrics"},
    "checkpoint": {"checkpoint"},
    "inference": INFERENCE_REQUIRED,
    "rl": RL_REQUIRED,
}


def row_seam(row: ResolvedRow) -> str:
    if not row.provides:
        raise ComposeError(f"plugin '{row.plugin}' provides no seam", row=row.id)
    if len(row.provides) > 1:
        raise ComposeError(
            f"v1 plugins must provide exactly one seam; '{row.plugin}' provides {row.provides}",
            row=row.id,
        )
    return row.provides[0]


def _active_by_seam(
    composition: Composition,
    phase_services: dict[str, str],
) -> dict[str, list[ResolvedRow]]:
    active: dict[str, list[ResolvedRow]] = defaultdict(list)
    for row in composition.rows:
        if row.group:
            continue
        if row.disabled:
            continue
        active[row_seam(row)].append(row)
    for seam, row_id in phase_services.items():
        row = composition.by_id.get(row_id)
        if row is None:
            raise ComposeError(f"phase references unknown row '{row_id}'", row=row_id)
        if seam not in row.provides:
            raise ComposeError(
                f"row '{row_id}' does not provide seam '{seam}'", row=row_id
            )
        if seam in PINNED_SEAMS:
            raise ComposeError(f"pinned seam '{seam}' cannot be rebound", row=row_id)
        if seam not in REBINDABLE_SEAMS:
            raise ComposeError(f"seam '{seam}' is not rebindable", row=row_id)
        active[seam] = [row]
    return active


def _check_cardinality(active: dict[str, list[ResolvedRow]]) -> None:
    for seam, rows in active.items():
        if seam in EXCLUSIVE_SEAMS and len(rows) != 1:
            ids = ", ".join(row.id for row in rows)
            raise ComposeError(
                f"exclusive seam '{seam}' has {len(rows)} providers: {ids}"
            )
        if seam in REGISTRY_SEAMS and not rows:
            raise ComposeError(f"registry seam '{seam}' has no provider")
        if seam not in EXCLUSIVE_SEAMS | REGISTRY_SEAMS | POLICY_SEAMS:
            raise ComposeError(f"unknown seam '{seam}'")


def _check_requires(active: dict[str, list[ResolvedRow]]) -> None:
    available = set(active)
    for rows in active.values():
        for row in rows:
            missing = [seam for seam in row.requires if seam not in available]
            if missing:
                raise ComposeError(
                    f"plugin '{row.plugin}' requires missing seams: {', '.join(missing)}",
                    row=row.id,
                )


def build_phase(
    composition: Composition,
    name: str,
    phase_services: dict[str, str] | None = None,
) -> PhasePlan:
    phase_services = phase_services or {}
    active = _active_by_seam(composition, phase_services)
    _check_cardinality(active)
    _check_requires(active)
    required = PHASE_REQUIRED.get(name, set())
    missing = sorted(seam for seam in required if seam not in active)
    if missing:
        raise ComposeError(f"phase '{name}' is missing required seams: {', '.join(missing)}")

    services: dict[str, ServiceBinding] = {}
    for seam, rows in active.items():
        if seam in POLICY_SEAMS:
            continue
        row = rows[0]
        services[seam] = ServiceBinding(
            id=row.id,
            seam=seam,
            provider=row.plugin,
            source=row.source,
            pinned=seam in PINNED_SEAMS,
            rebindable=seam in REBINDABLE_SEAMS,
        )

    hooks = []
    for seam in ("step_policy", "checkpoint_policy", "eval_policy"):
        for row in sorted(
            active.get(seam, []),
            key=lambda item: getattr(item.plugin_cls, "priority", 100),
        ):
            hooks.append(
                f"{seam}:{row.plugin}({getattr(row.plugin_cls, 'priority', 100)})"
            )
    phase_material = (
        f"{composition.composition_hash}:{name}:"
        f"{json.dumps(phase_services, sort_keys=True, separators=(',', ':'))}"
    )
    phase_hash = hashlib.sha256(phase_material.encode("utf-8")).hexdigest()
    return PhasePlan(
        name=name,
        services=services,
        step=None,
        hooks=tuple(hooks),
        composition_hash=phase_hash,
    )


def validate_composition(
    composition: Composition,
    phase_names: list[str] | None = None,
) -> tuple[dict[str, PhasePlan], tuple[Diagnostic, ...]]:
    phases = phase_names or list(composition.recipe.phases) or ["train"]
    diagnostics: list[Diagnostic] = []
    result: dict[str, PhasePlan] = {}
    for name in phases:
        services = composition.recipe.phases.get(name, {})
        if not isinstance(services, dict):
            raise ComposeError(f"phase '{name}' must map seam to row id")
        result[name] = build_phase(composition, name, services)
    return result, tuple(diagnostics)
