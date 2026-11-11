# Vendored dsv4 source

Copied from the user's MindSpeed-LLM checkout with explicit authorization.

- Source: `~/Documents/MindSpeed-LLM-agent-dsv4-complete-20260804/mindspeed_llm/fsdp2/models/deepseek_v4_native/`
- Source commit: `da9858d7d42a972e4e72ad21c420c71476263c81`
- Files:
  - `configuration_deepseek_v4_native.py`
  - `modeling_deepseek_v4_native.py` (imports patched to local `lm_head` / `args`)
  - `lm_head.py` (from `mindspeed_llm/fsdp2/models/common/modules.py`)
  - `args.py` (minimal `get_args()` shim; raises so the torch fallback is used)

Local compatibility patches for transformers 5.2.0:

- `DeepseekV4Config` is decorated with `@dataclass`; `strict` is a no-op because
  `huggingface_hub.dataclasses.strict` rejects transformers 5.2's
  `validate_rope(ignore_keys=...)` signature.
- `validate_rope` accepts the base class's `ignore_keys` keyword and falls back
  to an empty list when the attribute is not initialised yet.
- `sqrtsoftplus` is registered in `ACT2FN` (transformers 5.2 does not ship this
  DeepSeek-V4 router activation). Implementation:
  `sqrt(softplus(x.float())).type_as(x)`. transformers 5.2's `ACT2FN` is a
  `ClassInstantier`, so the registration is an `nn.Module` class, not a plain
  function.

The upstream copyright headers are retained. When updating, re-copy from the
same paths and re-apply the two import replacements:

```text
from mindspeed_llm.fsdp2.models.common.modules import LMHead  ->  from .lm_head import LMHead
from mindspeed_llm.fsdp2.utils.global_vars import get_args     ->  from .args import get_args
```

The vendored files import torch/transformers only when `Dsv4Model.build()` is
called; static `dump-config` / `inspect --plan` do not import them.
