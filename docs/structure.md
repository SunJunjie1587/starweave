# Starweave 目标架构

## 1. 核心原则

一个模型实例 = **一个模型插件 + 一组能力插件 + 一份实例配置**。

模型插件只描述“这是什么模型”，不关心任务、优化器、数据形态。任务由 task 插件决定：

```text
同一个 model-dsv4
  + task-sft        → SFT 训练
  + task-rl         → 强化学习
  + task-inference  → 推理
```

优化器和数据同理：

```text
task-sft + optimizer-adamw
task-sft + optimizer-muon
task-sft + data-text-jsonl
task-rl  + data-multimodal-jsonl
```

换任何一项都不需要改模型插件，也不需要改其他插件。

## 2. 仓库边界

最终拆成两个仓：

```text
starweave/                  # 核心仓：Cordis 式插件内核 + 契约 + 简单 examples
starweave-plugins/          # 插件仓：所有具体实现，按类别分目录
```

本仓只保留：

- Context / Plugin / Service / Event / Scope / 生命周期；
- recipe / bundle / patch / `use` 配置加载；
- runtime：插件解析、实例化、phase 切换、执行；
- `seams/`：**只是契约**，不含任何具体实现；
- `examples/`：toy model + toy task + toy optimizer + toy data，不依赖 NPU。

本仓不包含 device-npu、dsv4、AdamW、Muon、JSONL data、SFT/RL trainer 等实现。

## 3. 插件仓目录

按“能力类别”分目录，每个目录是一个独立发行包：

```text
starweave-plugins/
├── models/
│   ├── model-dsv4/
│   └── model-hf/
├── tasks/
│   ├── task-sft/
│   ├── task-rl/
│   └── task-inference/
├── optimizers/
│   ├── optimizer-adamw/
│   ├── optimizer-muon/
│   └── optimizer-sgd/
├── data/
│   ├── data-text-jsonl/
│   ├── data-text-hf/
│   └── data-multimodal-jsonl/
├── backends/
│   ├── backend-npu/
│   └── backend-gpu/          # v2
├── parallel/
│   ├── parallel-none/
│   ├── parallel-fsdp2/       # v2
│   └── parallel-ep/          # v2
├── checkpoint/
│   └── checkpoint-safetensors/
├── metrics/
│   └── metrics-jsonl/
├── profilers/
│   └── profiler-npu/
└── policies/
    ├── policy-grad-accum/
    ├── policy-grad-clip/
    └── policy-nan-guard/
```

每个目录结构一致：

```text
plugins/optimizers/optimizer-muon/
├── pyproject.toml
├── README.md
├── src/starweave_optimizer_muon/
│   ├── __init__.py
│   ├── config.py
│   └── plugin.py
└── tests/test_contract.py
```

插件之间不互相 import；跨类别只通过 `starweave.seams` 的类型和 Protocol 通信。

## 4. 核心仓目录

```text
starweave/
├── src/starweave/
│   ├── context.py        # Context / Service / Scope
│   ├── plugin.py         # PluginBase / 生命周期
│   ├── events.py
│   ├── loader.py         # recipe / bundle / patch / use 展开
│   ├── runtime.py        # 插件解析、实例化、phase、执行
│   ├── cli.py
│   └── seams/            # 仅契约
│       ├── plugin.py     # PluginBase（也可从 starweave.seams 导入）
│       ├── task.py       # Task Protocol
│       ├── model.py      # ModelBuilder Protocol
│       ├── optimizer.py  # Optimizer Protocol
│       ├── data.py       # DataSource Protocol
│       ├── backend.py    # Device / Precision / Kernel
│       ├── parallel.py   # Collective / FSDP / EP
│       ├── checkpoint.py
│       ├── metrics.py
│       └── types.py      # ModelHandle / DataBundle / StepEvent ...
├── examples/
│   ├── toy-model/
│   ├── toy-task-sft/
│   ├── toy-task-inference/
│   ├── toy-optimizer-sgd/
│   └── recipes/{sft.yaml,inference.yaml}
└── tests/
```

`seams/` 只有 Protocol、pydantic config、共享 dataclass；core 不 import torch、
transformers、torch_npu。如果以后连 `seams/` 也要独立，可以再抽成
`starweave-plugin-api`；v1 不建议继续拆碎。

## 5. 四类插件的契约

### model 插件

只提供模型构建和权重管理，不知道任务是什么：

```python
class Dsv4Model(PluginBase):
    name = "model-dsv4"
    provides = ("model",)
    requires = ("backend",)

    def build(self, ctx, config) -> ModelHandle: ...
```

`ModelHandle` 只暴露 `model`、`tokenizer`、`vocab_size`、`parallel_plan`。SFT/RL/推理都拿同一个句柄。

### task 插件

决定“这次实例在做什么”：

```python
class SFTTask(PluginBase):
    name = "task-sft"
    provides = ("task",)
    requires = ("model", "data", "optimizer", "scheduler", "loss",
                "checkpoint", "metrics", "parallel")

class InferenceTask(PluginBase):
    name = "task-inference"
    provides = ("task",)
    requires = ("model", "data", "metrics")
```

task 只依赖 seam，不依赖具体 model/optimizer/data。SFT task 不 import AdamW，
RL task 不 import dsv4。

### optimizer 插件

只认 `ModelHandle` 和优化器配置：

```python
class Muon(PluginBase):
    name = "optimizer-muon"
    provides = ("optimizer",)
    requires = ("backend", "model")

    def build(self, ctx, model, config): ...
```

换 AdamW → Muon 只换 recipe 里的一个名字。

### data 插件

只负责产出 `DataBundle`，不 import 模型和任务：

```python
class TextJSONL(PluginBase):
    name = "data-text-jsonl"
    provides = ("data",)

    def build(self, ctx, config) -> DataBundle: ...
```

多模态数据换成 `data-multimodal-jsonl`，task 代码不变。

## 6. 实例配置：`use` 直接选插件

实例配置按类别选插件，模型插件保持不动：

```yaml
# instances/dsv4-sft.yaml
name: dsv4-sft
use:
  backend: npu
  model: dsv4
  task: sft
  optimizer: muon
  data: text-jsonl
  checkpoint: safetensors
  metrics: jsonl
config:
  model: {config_path: configs/dsv4_small.json, dtype: bfloat16}
  task: {max_steps: 50, grad_accum: 8}
  optimizer: {lr: 1.0e-4, weight_decay: 0.01}
  data: {path: data/sft_dsv4.jsonl, max_seq_len: 1024}
```

换成 RL：只改 `use.task` 和 `use.data`，model 不变。

```yaml
# instances/dsv4-rl.yaml
name: dsv4-rl
use:
  backend: npu
  model: dsv4          # 同一份模型插件
  task: rl
  optimizer: muon
  data: multimodal-jsonl
  checkpoint: safetensors
```

换成推理：不需要 optimizer/scheduler/loss。

```yaml
# instances/dsv4-inference.yaml
name: dsv4-inference
use:
  backend: npu
  model: dsv4
  task: inference
  data: text-jsonl
config:
  task: {max_new_tokens: 256}
```

`use` 的 key 是类别（seam），value 是插件短名；loader 解析 entry point
`<category>-<name>`，再用插件的 `provides` 校验类别是否匹配。

高级场景仍然可以用 bundle + patch 覆盖默认配置；`use` 是常用路径的简写。

## 7. 运行流程

```text
instance.yaml
  → loader 展开 use：
      {task: sft} → entry point "task-sft" → SFTTask
      {optimizer: muon} → entry point "optimizer-muon" → Muon
      {model: dsv4} → entry point "model-dsv4" → Dsv4Model
  → 校验：每个 exclusive seam 恰好一个 provider；requires 全部满足
  → 按依赖顺序实例化插件
  → task.compile(ctx, services) → 静态 Plan
  → task.execute(ctx)
  → phase 切换时只重绑 metrics/profiler/checkpoint/inference
```

entry point 组仍然是 `starweave.plugins`；插件名建议带类别前缀，避免冲突：

```toml
[project.entry-points."starweave.plugins"]
task-sft = "starweave_task_sft.plugin:SFTTask"

[project.entry-points."starweave.plugins"]
optimizer-muon = "starweave_optimizer_muon.plugin:Muon"
```

## 8. 解耦检查表

| 规则 | 检查方式 |
|---|---|
| 插件只 import `starweave.seams` | `check_decoupling.py` 静态扫描 |
| 模型插件不 import task/optimizer/data | 类别目录依赖检查 |
| task 不 import 具体 optimizer/model/data | 只通过 `ctx.get(...)` 拿 handle |
| data 不 import model/task | 只返回 `DataBundle` |
| 后端插件之外不出现 `torch_npu` / `.npu()` | 硬件泄漏 grep |
| 插件名全局唯一，entry point 格式统一 | `plugin list` + `dump-schema` |
| 删除任一插件，其他插件仍可组合 | 外部插件卸载测试 |

## 9. 迁移步骤

1. **先搬包，不改行为**：把 `src/starweave/plugins/` 按类别搬到
   `starweave-plugins/{models,tasks,optimizers,data,...}/`，每个包自己注册
   entry points；core 删除 29 个 entry points 和 `BUILTIN_PLUGINS` 回退。
2. **引入 task seam**：把现有 `trainer-sft` 和 `inference-hf` 分别包装成
   `task-sft` / `task-inference`，二者都提供 `task`；SFT 和推理共用同一个
   `model` 插件。
3. **加 `use` 配置**：loader 支持 `use: {category: name}` 展开为插件行；
   bundle + patch 保留为高级覆盖层。
4. **新增任务/优化器/数据插件**：RL task、Muon optimizer、多模态 data 都按同一
   契约新增，不改 core。
5. **回归**：现有 recipe、插件名、CLI 命令保持不变；跑 `pytest`、
   `dump-config`、20/50 step 训练、resume、infer、phase、外部插件 validator。

这样本仓最终只剩 Cordis 式插件内核、seam 契约和 toy examples；所有真实能力
（NPU、dsv4、SFT/RL/推理、AdamW/Muon、文本/多模态数据）都在插件仓，按类别
独立安装、独立替换。

## 10. 当前落地状态（2026-11-11）

- 已创建 `plugins/first_party/` 独立发行包，按 `torch_base`、`backend_npu`、
  `parallel_none`、`model_hf`、`inference_hf`、`data_jsonl`、`optim_torch`、
  `trainer_sft`、`policies`、`metrics_jsonl`、`checkpoint_safetensors` 分类。
- core `pyproject.toml` 已删除全部 `starweave.plugins` entry points。
- `BUILTIN_PLUGINS` 已删除；插件发现改为 entry points，测试环境可选使用
  `STARWEAVE_PLUGIN_MANIFEST` / `STARWEAVE_PLUGIN_MANIFEST_FIRST` 开发 manifest。
- `StepPlan` 已移入 `starweave.seams`，trainer 插件不再 import `starweave.plan`。
- 旧 `src/starweave/plugins/` 目录仍保留为备份，未被任何 entry point 引用；
  待确认后删除。
- 本机静态测试 69 passed / 3 skipped；NPU 机 72 passed / 1 skipped；
  3 step NPU smoke 通过。
