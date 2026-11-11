"""Layered patch engine.

Semantics follow docs/spec.md §6:
- insert appends rows
- an entry with id and no insert overrides that row; config is replaced wholesale
- group rows are expanded; patches may insert into a group by id
- inject/isolate are reserved and rejected
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..kernel.errors import ComposeError


@dataclass
class Row:
    id: str
    plugin: str
    config: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False
    group: bool = False
    source: str = ""
    children: list["Row"] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str) -> "Row":
        if not isinstance(data, dict):
            raise ComposeError(f"row must be a mapping, got {type(data).__name__}")
        row_id = data.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ComposeError(f"row is missing a non-empty id: {data!r}")
        if data.get("inject") not in (None, [], {}):
            raise ComposeError("'inject' is reserved and not supported in v1", row=row_id)
        if data.get("isolate") not in (None, [], {}):
            raise ComposeError("'isolate' is reserved and not supported in v1", row=row_id)
        plugin = data.get("plugin", "")
        group = bool(data.get("group", False))
        if not group and not plugin:
            raise ComposeError("row is missing 'plugin'", row=row_id)
        raw_config = data.get("config") or {}
        return cls(
            id=row_id,
            plugin=plugin,
            config=dict(raw_config) if isinstance(raw_config, dict) else {},
            disabled=bool(data.get("disabled", False)),
            group=group,
            source=source,
        )


class Composer:
    def __init__(self) -> None:
        self._rows: dict[str, Row] = {}
        self._groups: dict[str, list[Row]] = {}

    @property
    def rows(self) -> list[Row]:
        return list(self._rows.values())

    @property
    def groups(self) -> dict[str, list[Row]]:
        return {key: list(value) for key, value in self._groups.items()}

    def apply(self, entries: list[dict[str, Any]], source: str) -> None:
        if not isinstance(entries, list):
            raise ComposeError(f"patch layer '{source}' must be a YAML list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ComposeError(f"patch entry in '{source}' must be a mapping")
            if "insert" in entry:
                target = entry.get("id")
                if target is not None:
                    self._insert_into_group(target, entry["insert"], source)
                else:
                    self._insert(entry["insert"], source)
                continue
            self._override(entry, source)

    def _insert(self, entries: list[dict[str, Any]], source: str) -> None:
        for entry in entries:
            row = Row.from_dict(entry, source)
            if row.group:
                children = [
                    Row.from_dict(child, source) for child in entry.get("config", [])
                ]
                row.children = children
                self._groups[row.id] = children
                for child in children:
                    self._add(child)
            else:
                self._add(row)

    def _insert_into_group(
        self, group_id: str, entries: list[dict[str, Any]], source: str
    ) -> None:
        if group_id not in self._groups:
            raise ComposeError(f"insert targets unknown group '{group_id}'", row=group_id)
        for entry in entries:
            row = Row.from_dict(entry, source)
            if row.group:
                raise ComposeError("nested groups are not supported in v1", row=row.id)
            self._groups[group_id].append(row)
            self._add(row)

    def _add(self, row: Row) -> None:
        if row.id in self._rows:
            raise ComposeError(f"duplicate row id '{row.id}'", row=row.id)
        self._rows[row.id] = row

    def _override(self, entry: dict[str, Any], source: str) -> None:
        row_id = entry.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ComposeError(f"patch entry without id in '{source}'")
        if row_id not in self._rows:
            raise ComposeError(f"patch targets unknown row '{row_id}'", row=row_id)
        row = self._rows[row_id]
        if "plugin" in entry and entry["plugin"]:
            row.plugin = str(entry["plugin"])
        if "config" in entry:
            value = entry.get("config") or {}
            row.config = dict(value) if isinstance(value, dict) else {}
        if "disabled" in entry:
            row.disabled = bool(entry["disabled"])
        if "group" in entry:
            row.group = bool(entry["group"])
        row.source = source
