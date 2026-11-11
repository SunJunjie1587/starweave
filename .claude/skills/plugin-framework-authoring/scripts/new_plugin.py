#!/usr/bin/env python3
"""Scaffold a decoupled plugin package from the bundled template.

Example:
    python new_plugin.py ./demo-adamw \
        --distribution demo-adamw \
        --package demo_adamw \
        --framework-package demo
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def replace_text(path: Path, replacements: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--distribution", required=True, help="e.g. demo-adamw")
    parser.add_argument("--package", required=True, help="import package, e.g. demo_adamw")
    parser.add_argument("--framework-package", required=True, help="e.g. demo")
    parser.add_argument("--force", action="store_true", help="write into a non-empty target")
    args = parser.parse_args()

    if not args.package.isidentifier():
        print("--package must be a valid Python identifier", file=sys.stderr)
        return 2

    template = Path(__file__).resolve().parent.parent / "assets" / "plugin-template"
    if not template.is_dir():
        print(f"template not found: {template}", file=sys.stderr)
        return 2

    if args.target.exists() and any(args.target.iterdir()) and not args.force:
        print(
            f"target is not empty: {args.target}; pass --force to overwrite",
            file=sys.stderr,
        )
        return 2

    shutil.copytree(template, args.target, dirs_exist_ok=True)

    src = args.target / "src"
    old_package = src / "example_adamw"
    new_package = src / args.package
    if old_package.exists() and old_package != new_package:
        old_package.rename(new_package)

    replacements = {
        "example_adamw": args.package,
        "example-adamw": args.distribution,
        "framework.plugins": f"{args.framework_package}.plugins",
        "framework.seams": f"{args.framework_package}.seams",
        "framework-api": f"{args.framework_package}-api",
        "[tool.framework]": f"[tool.{args.framework_package}]",
    }

    for path in args.target.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".toml", ".md", ".txt", ".cfg", ".ini"}:
            replace_text(path, replacements)

    print(f"Scaffolded {args.target}")
    print("Next:")
    print("  1. Implement your seam in config.py and plugin.py.")
    print(
        "  2. python scripts/check_decoupling.py "
        f"{args.target} --framework-package {args.framework_package}"
    )
    print("  3. Run the contract tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
