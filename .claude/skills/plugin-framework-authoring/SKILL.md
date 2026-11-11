---
name: plugin-framework-authoring
description: Use when creating, packaging, reviewing, or debugging a plugin for the plugin-framework training or inference stack — including trainers, inference engines, data sources, optimizers, schedulers, losses, checkpoint stores, metrics sinks, profilers, device backends, kernels, and policies. Also use when a plugin must stay fully decoupled, declare provides and requires, register through entry points, and be composable by recipe without importing core internals or other plugins.
license: MIT
metadata:
  category: plugin-framework
  role: plugin-development
  version: "1.0.0"
---

# Authoring a plugin-framework plugin

## Core principle

A plugin is a replaceable implementation of exactly one seam. It may import the framework's **seams package** and its own dependencies. It must never import the kernel, composer, plan compiler, CLI, or another plugin. Removing the plugin from a recipe must not require changes anywhere else.

## When to use

- Adding a trainer, inference engine, data source, optimizer, scheduler, loss, checkpoint store, metrics sink, profiler, device backend, kernel, or policy.
- Reviewing whether an existing plugin is truly decoupled.
- Debugging a plugin that fails composition, imports a sibling plugin, or leaks hardware calls.
- Packaging a plugin as an installable distribution.

Do not use for changing the framework's seam contracts or composition model — use `plugin-framework-build` for that.

## Project binding

This repository is **Starweave**:

- framework distribution: `starweave`
- import package: `src/starweave`
- seams package: `starweave.seams` (`src/starweave/seams/`)
- entry-point group: `starweave.plugins`
- official plugin skill: `.claude/skills/plugin-framework-authoring/`
- reference plugins: `plugins/model_dsv4/` (implements `model`),
  `examples/external_plugin/` (implements `metrics`)
- decoupling check:
  `python .claude/skills/plugin-framework-authoring/scripts/check_decoupling.py <plugin-dir> --framework-package starweave`

At the start of a session, confirm the target repository is Starweave:

1. Read `pyproject.toml` and find the framework distribution and import package.
2. Read `src/starweave/seams/`. The seam Protocols there are the only framework API a plugin may use.
3. Confirm the entry-point group: `starweave.plugins`.
4. If the repository or seams package does not exist yet, stop and use `plugin-framework-build` first.

Never guess the framework package name; a wrong prefix silently breaks discovery.

## The decoupling contract

| Rule | Required |
|---|---|
| Imports | only `<pkg>.seams.*`, the standard library, and your own third-party dependencies |
| Forbidden imports | `<pkg>.kernel`, `<pkg>.compose`, `<pkg>.plan`, `<pkg>.cli`, `<pkg>.plugins.*`, any sibling plugin distribution |
| State | instance state plus `ctx`; no module-level mutable globals |
| Hardware | use the device/kernel seams; no direct `torch_npu`, `.npu()`, `.cuda()`, or device ids unless you are the device backend plugin |
| Config | your own pydantic model only; never read another plugin's config |
| Registration | entry-point group `<pkg>.plugins`; no edits to a central registry |
| Lifecycle | `setup(ctx, config)`; register cleanup with `ctx.effect(...)`; `teardown()` is idempotent |
| Optional dependencies | guarded import with an actionable error naming the missing extra |
| Compatibility | declare the API range as a `framework-api` dependency or `[tool.<pkg>].api` in `pyproject.toml` |
| Tests | a contract test with a fake context; no NPU and no sibling plugins |

## Plugin anatomy

```
<plugin-distribution>/
├── pyproject.toml
├── README.md
├── src/<plugin_pkg>/
│   ├── __init__.py          # export the plugin class and __version__
│   ├── config.py            # pydantic config model
│   └── plugin.py            # plugin class
└── tests/
    ├── test_contract.py
    └── test_config.py
```

## Authoring workflow

1. Identify the seam and its kind: exclusive, registry, or policy.
2. Scaffold from the bundled template:
   `python scripts/new_plugin.py <target> --distribution <dist> --package <pkg> --framework-package <framework>`
   (or copy `assets/plugin-template/` manually and rename the distribution and import package).
3. Implement `config.py` with a pydantic model (`extra="forbid"`).
4. Implement `plugin.py` with the metadata and lifecycle below.
5. Register the entry point in `pyproject.toml`.
6. Write `tests/test_contract.py` against the seam protocol.
7. Run `python scripts/check_decoupling.py <plugin-dir> --framework-package <pkg>`. Treat any failure as blocking.
8. Add a bundle row or recipe patch; verify `dump-config` shows the row and `inspect --plan` binds the plugin.

## Plugin class contract

```python
from __future__ import annotations

from typing import Any
from <pkg>.seams.optimizer import Optimizer          # seam package only
from pydantic import BaseModel


class AdamWConfig(BaseModel):
    model_config = {"extra": "forbid"}
    lr: float = 1e-4
    weight_decay: float = 0.0


class AdamWPlugin:
    name = "adamw"
    provides = ("optimizer",)
    requires = ("device", "model")
    config_model = AdamWConfig
    rebindable = False
    priority = 100

    def setup(self, ctx: Any, config: AdamWConfig) -> None:
        self.ctx = ctx
        self.config = config
        self._optimizer = None
        ctx.effect(self.teardown)

    def build(self, model):
        import torch

        self._optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        return self._optimizer

    def teardown(self) -> None:
        self._optimizer = None
```

Required metadata: `name`, `provides`, `requires`, `config_model`. Declare `rebindable` and `priority` only when the seam uses them. `provides` and `requires` use seam names, not package names.

## Seam kinds

| Kind | Cardinality | Plugin rules |
|---|---|---|
| `exclusive` | exactly one per phase | declare the seam in `provides`; the compiler rejects a second provider |
| `registry` | many | register a unique key; never assume another provider exists |
| `policy` | many, ordered | implement only the hook methods you need; use `priority` for order; must be side-effect safe |

Policies must not call other policies directly. The plan compiler builds the chain.

## Entry points

`pyproject.toml`:

```toml
[project.entry-points."<pkg>.plugins"]
adamw = "<plugin_pkg>.plugin:AdamWPlugin"

[tool.<pkg>]
api = ">=1,<2"
```

Use one group. Do not create one group per seam: discovery, `plugin list`, and `dump-config` all read `<pkg>.plugins`. For local development, a recipe may reference `<plugin_pkg>.plugin:AdamWPlugin` directly, but the package must still ship an entry point.

## Config schema

- pydantic v2, `extra="forbid"`, explicit defaults.
- No config keys that belong to another plugin.
- Validate ranges (`gt`, `ge`, `lt`) and enums.
- The schema is the source of truth for `dump-schema`; do not hand-write a second schema.
- Secrets come from `${env:VAR}` at compose time; never hardcode them.

## Recipe integration

Bundle row:

```yaml
- insert:
    - id: optimizer
      plugin: adamw
      config:
        lr: 2.0e-5
        weight_decay: 0.01
```

Override in a recipe:

```yaml
- id: optimizer
  config:
    lr: 1.0e-5
    weight_decay: 0.0
```

Removal test: delete the row, compose again, and confirm every other plugin still resolves. If something breaks, the plugin was not decoupled.

## Testing

Minimum test set:

1. **Metadata**: `name`, `provides`, `requires`, `config_model` exist and are the right types.
2. **Config**: defaults apply; invalid keys and out-of-range values are rejected.
3. **Contract**: every method of the seam protocol, happy path plus one failure.
4. **Lifecycle**: `setup` → use → `teardown`; resources released; `teardown` twice is safe.
5. **Decoupling**: the plugin imports no forbidden module. The validator covers this.
6. **Optional NPU**: marked and skipped when `torch_npu` is unavailable.

Use a fake context with `get`, `effect`, and emitted-event capture. Do not import the real kernel. Use pytest when available; otherwise write the same tests with `unittest`. The contract matters, not the test runner.

## Validator

`scripts/check_decoupling.py` checks:

- `pyproject.toml` exists and declares an entry point in `<pkg>.plugins`.
- Imports contain only `<pkg>.seams.*` from the framework.
- The plugin class declares `name`, `provides`, `requires`, and `config_model`.
- No `torch_npu`, `.npu()`, `.cuda()`, or hardcoded device outside device/kernel plugins.
- A `tests/` directory with at least one test function exists.
- A `framework-api` constraint is declared.
- Any package passed with `--forbid-import` is not imported (use this to ban known sibling plugins).
- `--exclude _vendor` skips vendored third-party code from the import and hardware checks; the plugin's own `plugin.py` / `config.py` are still checked.

Run it before every review. A failure is blocking, not advisory.

## Rationalization table

| Excuse | Reality |
|---|---|
| "Duck typing avoids version coupling" | It also removes the contract. Import the versioned seam package and declare the API range. |
| "One entry-point group per seam is clearer" | It fragments discovery and forces the core to know every seam. Use `<pkg>.plugins`. |
| "Config can be a plain dict" | Then there is no schema, no validation, and no `dump-schema`. Use pydantic. |
| "Cleanup happens at process exit" | Long jobs and tests need deterministic teardown. Register with `ctx.effect`. |
| "The plugin works without tests" | Contract tests prove removability and catch cross-plugin coupling. |
| "Importing the kernel is convenient" | It makes the plugin non-removable. Use seams and context only. |

## Red flags — stop and fix

- `from <pkg>.kernel import ...`
- `from <pkg>.plugins.optim_adamw import AdamW`
- `torch.npu.set_device(...)` outside the device plugin
- module-level `current_model = ...`
- monkey-patching another plugin or the framework
- plugin config containing another plugin's section
- no entry point; registration requires editing a list in core
- tests require the full framework or an NPU
- the plugin cannot be removed without a code change elsewhere

## References

- `references/plugin-contract.md` — seam protocols, lifecycle, versioning, error model, and per-seam rules.
- `assets/plugin-template/` — a working AdamW plugin package to copy and adapt.
- `scripts/new_plugin.py` — deterministic scaffolder; substitutes distribution, package, and framework names.
- `scripts/check_decoupling.py` — blocking decoupling validator.

**REQUIRED SUB-SKILL:** When the framework itself is missing or a seam contract must change, use `plugin-framework-build`.
