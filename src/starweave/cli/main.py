"""Starweave command-line interface."""

from __future__ import annotations

import argparse
import sys

from ..kernel.errors import BootError, ComposeError, RunError, StarweaveError
from .doctor import doctor
from .dump import dump_config, dump_schema
from .init import init
from .inspect import inspect_plan
from .plugin import plugin_list
from .run import infer, run


def _add_compose_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="ROW.FIELD=VALUE",
        help="override a scalar config leaf (repeatable)",
    )
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        metavar="FILE",
        help="apply a patch overlay after the recipe (repeatable)",
    )
    parser.add_argument("--run-id", default=None, help="run id used for ${run.id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="starweave", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="create a minimal recipe scaffold")
    init_parser.add_argument("directory")

    dump_parser = sub.add_parser("dump-config", help="print the composed tree with sources")
    dump_parser.add_argument("recipe")
    dump_parser.add_argument("--json", action="store_true")
    _add_compose_options(dump_parser)

    schema_parser = sub.add_parser("dump-schema", help="print plugin JSON Schemas")
    schema_parser.add_argument("plugin", nargs="?")

    inspect_parser = sub.add_parser("inspect", help="compile and print the static plan")
    inspect_parser.add_argument("--plan", action="store_true", required=True)
    inspect_parser.add_argument("recipe")
    inspect_parser.add_argument("--phase", default=None)
    _add_compose_options(inspect_parser)

    run_parser = sub.add_parser("run", help="run training phases")
    run_parser.add_argument("recipe")
    run_parser.add_argument("--phase", default=None)
    run_parser.add_argument("--resume", default=None)
    run_parser.add_argument("--allow-config-change", action="store_true")
    _add_compose_options(run_parser)

    infer_parser = sub.add_parser("infer", help="generate from a prompt")
    infer_parser.add_argument("recipe")
    infer_parser.add_argument("--prompt", required=True)
    infer_parser.add_argument("--max-new-tokens", type=int, default=256)
    infer_parser.add_argument("--temperature", type=float, default=0.7)
    infer_parser.add_argument("--top-p", type=float, default=0.9)
    infer_parser.add_argument("--checkpoint", default=None, help="model checkpoint directory")
    _add_compose_options(infer_parser)

    plugin_parser = sub.add_parser("plugin", help="plugin management")
    plugin_sub = plugin_parser.add_subparsers(dest="plugin_command", required=True)
    plugin_sub.add_parser("list", help="list installed plugin entry points")

    sub.add_parser("doctor", help="check the environment and NPU visibility")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return init(args.directory)
        if args.command == "dump-config":
            return dump_config(
                args.recipe,
                run_id=args.run_id or "dump",
                set_values=args.set,
                patch_paths=args.patch,
                as_json=args.json,
            )
        if args.command == "dump-schema":
            return dump_schema(args.plugin)
        if args.command == "inspect":
            return inspect_plan(
                args.recipe,
                run_id=args.run_id or "dump",
                set_values=args.set,
                patch_paths=args.patch,
                phase=args.phase,
            )
        if args.command == "run":
            return run(
                args.recipe,
                phase=args.phase,
                set_values=args.set,
                patch_paths=args.patch,
                resume=args.resume,
                allow_config_change=args.allow_config_change,
                run_id=args.run_id,
            )
        if args.command == "infer":
            return infer(
                args.recipe,
                prompt=args.prompt,
                set_values=args.set,
                patch_paths=args.patch,
                run_id=args.run_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                checkpoint=args.checkpoint,
            )
        if args.command == "plugin":
            if args.plugin_command == "list":
                return plugin_list()
        if args.command == "doctor":
            return doctor()
        parser.error(f"unknown command: {args.command}")
    except ComposeError as exc:
        print(f"compose error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    except BootError as exc:
        print(f"boot error [{exc.code}]: {exc}", file=sys.stderr)
        return 3
    except RunError as exc:
        print(f"run error [{exc.code}]: {exc}", file=sys.stderr)
        return 4
    except StarweaveError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 5
    except Exception as exc:  # noqa: BLE001 - surface plugin/runtime failures with a stable code
        print(f"runtime error: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
