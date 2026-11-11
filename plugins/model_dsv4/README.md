# starweave-model-dsv4

Official DeepSeek-V4 native model plugin for Starweave.

## Model source

`plugin.py` loads the dsv4 config/modeling in this order:

1. `starweave_model_dsv4/_vendor/` — vendored copies of
   `configuration_deepseek_v4_native.py` and `modeling_deepseek_v4_native.py`
   (source commit `da9858d`, see `_vendor/README.md`).
2. An installed or `PYTHONPATH`-visible MindSpeed-LLM checkout:
   `mindspeed_llm.fsdp2.models.deepseek_v4_native.*`.

The vendored files are present in this workspace. The adapter fallback remains
for environments that prefer to point at a MindSpeed-LLM checkout:

```bash
export PYTHONPATH=/path/to/MindSpeed-LLM-agent-dsv4-complete-20260804:$PYTHONPATH
```

Do not import MindSpeed-LLM core modules from the plugin's own `plugin.py` /
`config.py`.

## Install

```bash
uv pip install -e plugins/model_dsv4 --no-deps
```

## Config

| Field | Meaning |
|---|---|
| `config_path` | path to `configs/dsv4_small.json` |
| `checkpoint_path` | optional file or directory with `model.safetensors` |
| `tokenizer_name_or_path` | local dsv4 tokenizer directory |
| `dtype` | `float32` or `bfloat16` |
| `attn_implementation` | `eager` or `sdpa` |

## Validate

```bash
python3 ~/.claude/skills/plugin-framework-authoring/scripts/check_decoupling.py \
  plugins/model_dsv4 --framework-package starweave --exclude _vendor
```
