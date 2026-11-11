# example-adamw

A minimal, fully decoupled optimizer plugin generated from the official template.

## Structure

- `src/example_adamw/config.py` — pydantic config model
- `src/example_adamw/plugin.py` — plugin metadata and lifecycle
- `tests/test_contract.py` — contract tests with a fake context
- `pyproject.toml` — entry point and framework API range

## Rules

- Import only `<pkg>.seams.*` from the framework.
- Never import the kernel, composer, plan compiler, CLI, or another plugin.
- Register cleanup with `ctx.effect(self.teardown)`.
- Keep the config schema self-contained.
- Declare `provides`, `requires`, and `config_model`.

## Validate

```bash
python scripts/check_decoupling.py . --framework-package <pkg>
```

## If you copied this template manually

Replace `example-adamw` and `example_adamw` with your distribution and import
package. Replace the `framework` prefix in the seam import, the entry-point
group, the tool section, and the `framework-api` dependency with the real
framework package name. The scaffolder (`scripts/new_plugin.py`) does this
automatically.
