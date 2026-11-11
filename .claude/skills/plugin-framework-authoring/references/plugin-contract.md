# Plugin contract reference

This is the stable contract between the framework and every plugin. If a plugin needs something not described here, it should be added to the seams package first — not reached for through kernel internals.

## 1. Package contract

A plugin distribution:

- is a normal Python package installable with pip/uv;
- depends on the framework's `seams` distribution and declares a `framework-api` range;
- exports one or more plugin classes through the entry-point group `<pkg>.plugins`;
- contains no framework code and no other plugin;
- can be uninstalled without editing the framework or any recipe semantics beyond removing its rows.

Recommended `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "example-adamw"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pydantic>=2", "framework-api>=1,<2", "torch>=2.1"]

[project.entry-points."<pkg>.plugins"]
adamw = "example_adamw.plugin:AdamWPlugin"

[tool.<pkg>]
api = ">=1,<2"

[tool.hatch.build.targets.wheel]
packages = ["src/example_adamw"]
```

## 2. Plugin metadata

Every plugin class declares:

| Attribute | Type | Meaning |
|---|---|---|
| `name` | `str` | unique plugin key used in bundles and logs |
| `provides` | `tuple[str, ...]` | seam names this plugin implements |
| `requires` | `tuple[str, ...]` | seam names that must be available before setup |
| `config_model` | `type[BaseModel]` | pydantic v2 model; `extra="forbid"` |
| `rebindable` | `bool` | optional; only true for metrics, profiler, checkpoint, inference |
| `priority` | `int` | optional; lower runs earlier for policies and ordered chains |
| `optional` | `bool` | optional; failure logs and continues instead of aborting |

`provides` and `requires` use seam names (`optimizer`, `device`, `model`), not class names or package names.

## 3. Lifecycle

```
resolve → validate config → setup(ctx, config) → use → teardown()
```

Rules:

- `setup` does not perform heavy work that belongs to the hot path; build lazily where the seam allows.
- Register every resource cleanup with `ctx.effect(cleanup)` so the framework can unwind in reverse order.
- `teardown` is idempotent and safe to call twice.
- A plugin must not keep a reference to a released service after teardown.
- Optional dependencies are imported inside `setup` or the method that needs them, with an error that names the missing extra.

## 4. Context contract

The context is the only collaboration surface. A plugin may rely on:

- `ctx.get(seam_name)` for services it declared in `requires`;
- `ctx.effect(cleanup)` for teardown;
- typed events emitted by the framework (`step/end`, `phase/enter`, `checkpoint/save`);
- `ctx.paths`, `ctx.run_id`, `ctx.seed` and other documented run metadata.

A plugin must not:

- read a service it did not declare in `requires`;
- reach into another plugin's attributes;
- mutate global state or monkey-patch;
- assume a concrete implementation of another seam.

## 5. Seam kinds

### Exclusive

One provider per phase. Examples: `device`, `precision`, `trainer`, `inference`, `launcher`.

- Declare the seam in `provides`.
- The compiler rejects a second provider unless the recipe explicitly disables the first.
- `rebindable` is false for trainer/optimizer/scheduler/model.

### Registry

Many providers. Examples: `kernel`, `model`, `data`, `optimizer`, `scheduler`, `loss`, `checkpoint`, `metrics`, `profiler`.

- Register a unique key (usually `name`).
- Never assume another provider exists.
- Selection is by name or priority; the recipe decides.
- Providers must be independently constructible and testable.

### Policy

Many stackable providers. Examples: `step_policy`, `checkpoint_policy`, `eval_policy`.

- Implement only the hooks you need.
- Use `priority` for order; lower runs first.
- Policies must be side-effect safe and must not call other policies.
- The compiler builds the chain; the plugin only receives its hook context.

## 6. Config schema

- pydantic v2; `model_config = {"extra": "forbid"}`.
- Defaults must be safe for a CPU-only test.
- No cross-plugin config: if a value belongs to the device, read it through the device seam.
- Validate ranges and enums; fail at compose time, not mid-step.
- The schema feeds `dump-schema`; do not maintain a second schema.

## 7. Error model

| Situation | Behavior |
|---|---|
| Missing optional dependency | raise `PluginDependencyError` naming the extra and install hint |
| Invalid config | pydantic `ValidationError` at compose time, with the row path |
| Required seam unavailable | composition error before setup |
| Runtime failure | raise a plugin-specific error carrying `plugin`, `phase`, `step`, and the original cause |
| Teardown failure | log and continue unwinding the remaining services |

Never swallow an exception silently. Never catch `Exception` and return a default.

## 8. Versioning

- `pyproject.toml` declares the API range in one of two equivalent forms:
  `framework-api>=X,<Y` as a dependency, or `[tool.<pkg>] api = ">=X,<Y"`.
- `requires-python` must be at least the framework's minimum (usually `>=3.10`).
- The composer rejects a plugin whose range excludes the running API.
- A breaking seam change bumps the major API version.
- Additive changes (new optional method with a default) bump the minor version.
- Plugins must not rely on private attributes of seam objects.

## 9. Testing contract

A plugin's `tests/test_contract.py` must cover:

1. metadata exists and has the right types;
2. config defaults and invalid config rejection;
3. each seam method, happy path plus one failure;
4. setup → use → teardown, including double teardown;
5. a fake context that records `effect` callbacks and emitted events;
6. no import of forbidden framework modules (the validator enforces this statically).

A plugin test must pass with only the plugin installed and the seams package available. It must not need the kernel, an NPU, or a sibling plugin.

## 10. Packaging checklist

- [ ] distribution name and import package differ (e.g., `example-adamw` → `example_adamw`)
- [ ] `README.md` states the seam, config keys, and supported framework API
- [ ] entry point group is `<pkg>.plugins`
- [ ] API range declared as `framework-api` or `[tool.<pkg>].api`
- [ ] `requires-python` matches the framework minimum
- [ ] pydantic config with `extra="forbid"`
- [ ] lifecycle cleanup via `ctx.effect`
- [ ] tests pass without NPU
- [ ] validator passes, with `--forbid-import` for every known sibling plugin
- [ ] remove-the-row test passes: compose the recipe without the plugin and confirm the rest still resolves
