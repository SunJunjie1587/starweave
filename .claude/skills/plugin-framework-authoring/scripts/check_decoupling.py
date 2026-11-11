#!/usr/bin/env python3
"""Blocking decoupling validator for plugin-framework plugins.

Usage:
    python check_decoupling.py <plugin-dir> --framework-package <pkg> [--json]

Exit code 0 means the plugin passes every blocking check. Exit code 1 means at
least one error. Warnings do not fail the build, but should be fixed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None  # type: ignore


SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


@dataclass
class Finding:
    level: str
    code: str
    message: str
    path: str = ""


def iter_python_files(root: Path, exclude: list[str] | None = None):
    excluded = tuple(exclude or ())
    for path in root.rglob("*.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if any(token in str(path) for token in excluded):
            continue
        yield path


def parse_python(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return exc


def imported_modules(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module or ""]
    return []


def class_attributes(node: ast.ClassDef) -> set[str]:
    attrs: set[str] = set()
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            attrs.update(
                target.id for target in stmt.targets if isinstance(target, ast.Name)
            )
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            attrs.add(stmt.target.id)
    return attrs


def class_strings(node: ast.ClassDef) -> set[str]:
    values: set[str] = set()
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "provides":
                    for child in ast.walk(stmt.value):
                        if isinstance(child, ast.Constant) and isinstance(
                            child.value, str
                        ):
                            values.add(child.value)
    return values


def nested_get(mapping: dict, dotted: str):
    current = mapping
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _split_section(header: str) -> list[str]:
    parts = re.findall(r'"[^"]*"|[^.]+', header)
    return [part.strip().strip('"').strip("'") for part in parts if part.strip()]


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return re.findall(r'["\']([^"\']*)["\']', raw)
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    return raw


def _parse_toml_fallback(path: Path) -> dict:
    """Small TOML subset parser for Python 3.10 without tomli.

    It supports the pyproject structures this validator needs: section
    headers, string values, and single- or multi-line string arrays.
    """
    data: dict = {}
    section: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = _split_section(line[1:-1])
            continue
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key = key.strip().strip('"').strip("'")
        raw = raw.strip()
        if raw.startswith("["):
            while raw.count("[") > raw.count("]") and index < len(lines):
                raw += " " + lines[index].strip()
                index += 1
        value = _parse_value(raw)
        current = data
        for part in section:
            current = current.setdefault(part, {})
            if not isinstance(current, dict):
                break
        else:
            current[key] = value
    return data


def load_pyproject_data(path: Path) -> dict:
    if tomllib is not None:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    return _parse_toml_fallback(path)


def check_pyproject(root: Path, pkg: str, findings: list[Finding]):
    path = root / "pyproject.toml"
    if not path.exists():
        findings.append(Finding("error", "missing-pyproject", "pyproject.toml not found"))
        return None

    try:
        data = load_pyproject_data(path)
    except Exception as exc:  # noqa: BLE001
        findings.append(
            Finding("error", "invalid-pyproject", f"cannot parse pyproject.toml: {exc}")
        )
        return None

    project = data.get("project", {})
    entry_points = project.get("entry-points", {})
    group = f"{pkg}.plugins"

    if group not in entry_points:
        findings.append(
            Finding(
                "error",
                "missing-entry-point-group",
                f'declare [project.entry-points."{group}"]',
                str(path),
            )
        )
    elif not entry_points[group]:
        findings.append(
            Finding("error", "empty-entry-point-group", f"{group} is empty", str(path))
        )
    else:
        for name, value in entry_points[group].items():
            if ":" not in str(value):
                findings.append(
                    Finding(
                        "warn",
                        "entry-point-format",
                        f"entry point '{name}' should use module:Class or module:function",
                        str(path),
                    )
                )

    for key in entry_points:
        if key.startswith(pkg + ".") and key != group:
            findings.append(
                Finding(
                    "warn",
                    "extra-entry-point-group",
                    f"{key} is not used by discovery; use {group}",
                    str(path),
                )
            )

    api = nested_get(data.get("tool", {}), f"{pkg}.api")
    api_dep = any(
        str(dep).startswith(("framework-api", f"{pkg}-api"))
        for dep in project.get("dependencies", [])
    )
    if api is None and not api_dep:
        findings.append(
            Finding(
                "warn",
                "missing-api-range",
                f'declare [tool."{pkg}".api] or a framework-api dependency',
                str(path),
            )
        )

    return data


def check_imports(
    root: Path,
    pkg: str,
    forbidden: list[str],
    exclude: list[str],
    findings: list[Finding],
):
    for path in iter_python_files(root, exclude):
        tree = parse_python(path)
        if isinstance(tree, SyntaxError):
            findings.append(
                Finding("error", "syntax-error", str(tree), str(path))
            )
            continue
        for node in ast.walk(tree):
            for module in imported_modules(node):
                if not module:
                    continue
                if any(
                    module == item or module.startswith(item + ".")
                    for item in forbidden
                ):
                    findings.append(
                        Finding(
                            "error",
                            "forbidden-sibling-import",
                            f"imports forbidden sibling package {module}",
                            str(path),
                        )
                    )
                if module == pkg or module.startswith(pkg + "."):
                    allowed = module == f"{pkg}.seams" or module.startswith(
                        f"{pkg}.seams."
                    )
                    if not allowed:
                        findings.append(
                            Finding(
                                "error",
                                "forbidden-framework-import",
                                f"imports {module}; plugins may import only {pkg}.seams.*",
                                str(path),
                            )
                        )


def check_plugin_classes(
    root: Path, exclude: list[str], findings: list[Finding]
):
    found = False
    for path in iter_python_files(root, exclude):
        if "tests" in path.parts:
            continue
        tree = parse_python(path)
        if isinstance(tree, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            attrs = class_attributes(node)
            if "provides" not in attrs:
                continue
            found = True
            for required in ("name", "provides", "requires", "config_model"):
                if required not in attrs:
                    findings.append(
                        Finding(
                            "error",
                            "missing-plugin-metadata",
                            f"class {node.name} lacks class attribute '{required}'",
                            str(path),
                        )
                    )

            for stmt in node.body:
                value = None
                targets: list[ast.AST] = []
                if isinstance(stmt, ast.Assign):
                    value = stmt.value
                    targets = list(stmt.targets)
                elif isinstance(stmt, ast.AnnAssign):
                    value = stmt.value
                    targets = [stmt.target]
                if not any(
                    isinstance(target, ast.Name) and target.id == "config_model"
                    for target in targets
                ):
                    continue
                if isinstance(value, ast.Name) and value.id in {
                    "dict",
                    "object",
                    "None",
                    "Any",
                }:
                    findings.append(
                        Finding(
                            "error",
                            "invalid-config-model",
                            f"class {node.name} sets config_model to {value.id}; "
                            "use a pydantic BaseModel subclass",
                            str(path),
                        )
                    )
                elif isinstance(value, ast.Constant):
                    findings.append(
                        Finding(
                            "error",
                            "invalid-config-model",
                            f"class {node.name} sets config_model to a constant; "
                            "use a pydantic BaseModel subclass",
                            str(path),
                        )
                    )
    if not found:
        findings.append(
            Finding(
                "error",
                "no-plugin-class",
                "no class declares a 'provides' class attribute",
            )
        )


def check_hardware(
    root: Path, exclude: list[str], findings: list[Finding]
):
    patterns = [
        ("torch_npu", re.compile(r"\btorch_npu\b")),
        (".npu()", re.compile(r"\.npu\s*\(")),
        ("torch.cuda", re.compile(r"\btorch\.cuda\b")),
        (".cuda()", re.compile(r"\.cuda\s*\(")),
        ("device string", re.compile(r"device\s*=\s*['\"](?:npu|cuda)")),
    ]

    # Device/kernel/acceleration plugins are the only ones allowed to touch
    # hardware APIs directly.
    allow = False
    for path in iter_python_files(root, exclude):
        tree = parse_python(path)
        if isinstance(tree, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                values = class_strings(node)
                if any(
                    token in value
                    for value in values
                    for token in ("device", "kernel", "attention", "accel")
                ):
                    allow = True
        if allow:
            break

    if allow:
        return

    for path in iter_python_files(root, exclude):
        text = path.read_text(encoding="utf-8")
        for label, pattern in patterns:
            if pattern.search(text):
                findings.append(
                    Finding(
                        "error",
                        "hardware-leak",
                        f"uses {label}; use the device/kernel seam or move this into a device plugin",
                        str(path),
                    )
                )


def check_tests(root: Path, findings: list[Finding]):
    candidates = [
        path
        for path in root.rglob("*.py")
        if "tests" in path.parts and path.name.startswith("test_")
    ]
    if not candidates:
        findings.append(
            Finding(
                "error",
                "missing-tests",
                "no tests/test_*.py found; add a contract test",
            )
        )
        return
    for path in candidates:
        if "def test_" not in path.read_text(encoding="utf-8"):
            findings.append(
                Finding(
                    "warn",
                    "empty-test-file",
                    "test file contains no 'def test_'",
                    str(path),
                )
            )


def check_module_globals(
    root: Path, exclude: list[str], findings: list[Finding]
):
    for path in iter_python_files(root, exclude):
        if "tests" in path.parts:
            continue
        tree = parse_python(path)
        if isinstance(tree, SyntaxError):
            continue
        for node in tree.body:
            value = None
            names: list[str] = []
            if isinstance(node, ast.Assign):
                value = node.value
                names = [
                    target.id for target in node.targets if isinstance(target, ast.Name)
                ]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                value = node.value
                names = [node.target.id]

            if any(name.startswith("__") and name.endswith("__") for name in names):
                continue

            if isinstance(value, (ast.List, ast.Dict, ast.Set)):
                findings.append(
                    Finding(
                        "warn",
                        "module-mutable",
                        "module-level mutable state; move it into the plugin instance or ctx",
                        str(path),
                    )
                )


def run(
    plugin_dir: Path,
    pkg: str,
    forbidden: list[str],
    exclude: list[str],
) -> list[Finding]:
    findings: list[Finding] = []
    if not plugin_dir.is_dir():
        return [
            Finding("error", "missing-directory", f"{plugin_dir} is not a directory")
        ]

    check_pyproject(plugin_dir, pkg, findings)
    check_imports(plugin_dir, pkg, forbidden, exclude, findings)
    check_plugin_classes(plugin_dir, exclude, findings)
    check_hardware(plugin_dir, exclude, findings)
    check_tests(plugin_dir, findings)
    check_module_globals(plugin_dir, exclude, findings)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin_dir", type=Path)
    parser.add_argument("--framework-package", default="framework")
    parser.add_argument(
        "--forbid-import",
        action="append",
        default=[],
        help="top-level module to reject, e.g. a sibling plugin package (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="path substring to skip, e.g. _vendor for vendored third-party code (repeatable)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings = run(
        args.plugin_dir,
        args.framework_package,
        args.forbid_import,
        args.exclude,
    )

    if args.json:
        print(json.dumps([asdict(item) for item in findings], indent=2))
    else:
        for item in findings:
            location = f"{item.path}: " if item.path else ""
            print(f"{item.level.upper():5} [{item.code}] {location}{item.message}")
        errors = sum(1 for item in findings if item.level == "error")
        warnings = sum(1 for item in findings if item.level == "warn")
        print(f"\n{errors} error(s), {warnings} warning(s)")

    return 1 if any(item.level == "error" for item in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
