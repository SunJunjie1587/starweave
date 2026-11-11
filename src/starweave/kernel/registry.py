"""Service registry with exclusive/registry/policy cardinality checks."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .errors import ComposeError


class SeamKind(str, Enum):
    EXCLUSIVE = "exclusive"
    REGISTRY = "registry"
    POLICY = "policy"


class ServiceRegistry:
    def __init__(self) -> None:
        self._services: dict[str, Any] = {}
        self._kinds: dict[str, SeamKind] = {}

    def register(self, name: str, service: Any, kind: SeamKind = SeamKind.EXCLUSIVE) -> None:
        if kind is SeamKind.EXCLUSIVE and name in self._services:
            raise ComposeError(f"exclusive seam '{name}' already has a provider")
        self._services[name] = service
        self._kinds[name] = kind

    def get(self, name: str) -> Any:
        return self._services[name]

    def all(self) -> dict[str, Any]:
        return dict(self._services)

    def kind(self, name: str) -> SeamKind:
        return self._kinds[name]
