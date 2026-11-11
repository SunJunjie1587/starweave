# External plugin example: metrics-memory

This is a complete third-party Starweave plugin. It implements the `metrics`
seam, buffers `MetricEvent`s in memory, and flushes them as JSONL. The plugin
module imports only `starweave.seams` and its own dependencies.

## Install

```bash
pip install -e examples/external_plugin
starweave plugin list | grep metrics-memory
```

## Use in a recipe

The repository recipe `recipes/sft_npu_dsv4_small_metrics_memory.yaml` replaces
the first-party metrics row with this plugin:

```yaml
- id: metrics
  plugin: metrics-memory
  config: {path: "runs/${run.id}/metrics-memory.jsonl"}
```

Run it exactly like the standard recipe:

```bash
starweave run recipes/sft_npu_dsv4_small_metrics_memory.yaml \
  --set trainer.max_steps=3
```

## Validate decoupling

```bash
python ~/.claude/skills/plugin-framework-authoring/scripts/check_decoupling.py \
  examples/external_plugin --framework-package starweave
pytest examples/external_plugin/tests -q
```

Removing this package only removes the `metrics-memory` entry point; the
first-party composition and all other plugins continue to work.
