# RL 任务如何集成 verl 与 vLLM

## 1. 结论

- **可以把 vLLM 作为一个 rollout 插件“包进来”**：`use.rollout: vllm`，
  由 `rollout` seam 适配 `vllm.LLM` / `AsyncLLMEngine`。
- **可以把 verl 作为一个 RL 后端插件“包进来”**：`use.task: rl-verl`，
  由该插件在内部启动 `RayPPOTrainer`，vLLM 仍由 verl 的 rollout worker 管理。
- **不能通过 Starweave 配置直接改 vLLM/verl 的内部实现**：PagedAttention、
  vLLM scheduler/executor、verl 的 Ray worker group、DataProto 拓扑都不在
  Starweave 的 seam 上。要改这些，只能使用它们自己的扩展点，或者 fork/patch。
- 如果目标是“任意替换 RL 组件”，应实现原生 `task-rl` + `rollout` + `reward`
  seam，把 verl 当参考实现而不是运行时依赖；这是最可组合、但工作量最大的路线。

一句话：**边界可替换，内部不可配置替换；整块可封装，零件要看对方是否暴露。**

## 2. 边界原则

Starweave 的插件只能控制跨 seam 的行为：

```text
task-rl  ←→  rollout  ←→  reward  ←→  model/optimizer/data/parallel
```

一个插件内部的实现细节，对外部是黑盒。vLLM 和 verl 都是“大内部”的系统：

| 系统 | 可被 Starweave 控制的边界 | 不能被 Starweave 控制的内部 |
|---|---|---|
| vLLM | `generate`、`logprobs`、`update_weights`、`sleep/wake_up`、采样参数、TP/显存/长度配置 | PagedAttention、KV block manager、scheduler 内部策略、CUDA/HCCL kernel、executor 拓扑 |
| verl | 启动/停止、配置映射、reward/rollout 扩展、checkpoint 路径 | Ray placement group、WorkerGroup、DataProto 调度、内部 engine 的并行策略 |

如果需要的功能不在上表左列，Starweave 的 `use` 配置无法覆盖它。

## 3. vLLM 作为 rollout 插件

### 3.1 seam

新增 `rollout` seam，而不是复用训练用的 `inference`：

```python
class Rollout(Protocol):
    def generate(self, requests, sampling_params) -> list[RolloutOutput]: ...
    def logprobs(self, requests) -> list[torch.Tensor]: ...
    def update_weights(self, model_state) -> None: ...
    def sleep(self) -> None: ...
    def wake_up(self) -> None: ...
```

`task-rl` 只依赖这些方法，不 import vLLM。

### 3.2 插件

```text
plugins/rollout/rollout-vllm/
├── pyproject.toml       # entry point: rollout-vllm
├── src/starweave_rollout_vllm/
│   ├── config.py        # tensor_parallel_size, gpu_memory_utilization,
│   │                    # max_model_len, dtype, enforce_eager, ...
│   └── plugin.py        # wraps vllm.LLM / AsyncLLMEngine
└── tests/
```

实例配置：

```yaml
use:
  task: rl
  rollout: vllm
  model: dsv4
config:
  rollout:
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.6
    max_model_len: 4096
    enforce_eager: false
```

换 SGLang/TensorRT-LLM 时只换 `use.rollout`，`task-rl` 不变。

### 3.3 真正困难的是权重同步

RL 的 rollout 引擎需要不断拿到 actor 的最新权重。vLLM 常见方式：

- `update_weights` 走 collective（verl 在 CUDA 上用 NCCL；Ascend 对应 HCCL）；
- 序列化 state dict / 共享内存 / IPC；
- vLLM `sleep()` 释放 KV cache，`wake_up()` 重新分配。

这属于 `rollout` seam 的契约，不是配置项。`rollout-vllm` 必须实现
`update_weights`，否则只能做离线推理，不能做 on-policy RL。

### 3.4 Ascend 注意事项

- 原生 vLLM 主要面向 CUDA；Ascend 需要 `vllm-ascend` 插件，且与
  torch_npu/CANN 版本强绑定。
- dsv4 是自定义架构；需要确认 vLLM/vllm-ascend 是否已有对应 model
  implementation。没有的话要写 vLLM model plugin，不能只靠 Starweave 适配。
- vLLM 自己会占用 NPU 显存和进程组；Starweave 的 FSDP/EP 插件不能再假设自己
  独占这些设备。资源所有权必须二选一，见 §6。

## 4. verl 的三种集成方式

### A. 整块封装：`task-rl-verl`

```text
plugins/tasks/task-rl-verl/
  plugin.py   # imports verl, builds RayPPOTrainer, runs it
```

实例：

```yaml
use:
  backend: npu
  model: dsv4
  task: rl-verl
  rollout: vllm
config:
  rl_verl:
    algorithm: grpo
    actor_rollout_ref:
      rollout:
        name: vllm
      actor:
        optim:
          lr: 1.0e-6
```

- 优点：最快获得 verl 的 PPO/GRPO/DAPO、reference/critic/reward、Ray 调度、
  checkpoint、vLLM 权重同步等完整能力。
- 缺点：进入 verl 之后，Starweave 的 optimizer/data/parallel 插件**不再生效**；
  verl 的 Hydra 配置成为内部事实源。你只能通过 verl 自己的扩展点
  （custom worker、reward manager、rollout、engine）覆盖行为。
- 适用：目标是“马上跑 RL”，不要求每个内部组件都可换。

### B. 原生 `task-rl` + 借鉴 verl

```text
task-rl 自己实现 PPO/GRPO 循环
  + rollout-vllm 插件
  + model/optimizer/data/parallel 插件
```

- 优点：所有组件都可替换，完全符合 `use: {task: rl, optimizer: muon, ...}`。
- 缺点：要自己实现 advantage、KL、reference policy、critic、reward、
  sequence packing、权重同步、断点续训、分布式调度；工作量最大。
- 适用：目标是“RL 组件任意组合”，愿意承担实现成本；verl 作为算法参考。

### C. 引擎级适配：`engine-verl`

verl 新版本有 `verl.workers.engine` 分层（FSDPEngine / MegatronEngine /
VeOmniEngine 等）。可以尝试把某个 engine 包装成 Starweave 的
`trainer`/`model` 后端：

```text
task-rl（Starweave 外层循环）
  └── engine-verl（适配 verl 的 FSDP/Megatron engine）
        └── rollout-vllm
```

- 优点：复用 verl 的高性能训练 engine，同时保留外层任务/数据/rollout 组合。
- 缺点：engine 原本跑在 verl 的 worker/Ray 拓扑里，抽出来要处理
  `DataProto`、进程组、权重同步和配置映射；属于高难度适配，不是简单 wrapper。
- 适用：既想要 verl 性能，又不想把整个控制面交给 verl；需要有 verl 内部经验。

### 选择建议

| 目标 | 推荐 |
|---|---|
| 尽快跑通 RL，功能对齐 verl | A：整块封装 `task-rl-verl` |
| 组件任意替换、长期可组合 | B：原生 `task-rl` + `rollout-vllm` |
| 复用 verl 高性能 engine，保留外层编排 | C：`engine-verl` 适配 |
| 只是想把 vLLM 当推理/rollout 引擎 | `rollout-vllm` 插件，独立于 verl |

## 5. RL 需要补充的 seam

当前 v1 的 `model`、`trainer`、`inference` 不足以表达 verl 式 RL。建议补：

| seam | 作用 |
|---|---|
| `task` | SFT / RL / inference 的任务入口 |
| `rollout` | 生成引擎（vLLM/SGLang/...），含 `update_weights` |
| `reward` | 规则 reward 或 reward model |
| `environment` | agentic RL 的 reset/step（可选） |
| `model` | 需要支持按 role 构建：actor / reference / critic / reward |
| `parallel` | 训练侧的 FSDP/EP；与 rollout 的资源边界要显式 |

`model` 的 role 设计示例：

```python
class ModelBuilder(Protocol):
    def build(self, ctx, role: str, config) -> ModelHandle: ...
```

`task-rl` 用 `build("actor")`、`build("reference")`、`build("critic")`
拿多个 handle；SFT/inference 只用 `build("actor")`。当前单例 `model` seam
需要升级为 role-scoped registry，否则 RL 的 reference/critic 无处安放。

`task-rl` 依赖：

```python
requires = ("model", "data", "rollout", "reward", "optimizer",
            "scheduler", "loss", "checkpoint", "metrics", "parallel")
```

GRPO 可以不要 critic，但 reference policy 通常仍需要；PPO 需要 critic。
这些由 `task-rl` 的配置决定，不由 core 决定。

## 6. 资源所有权：不要让 Starweave 和 verl 同时当调度器

RL 里有两套进程/设备拓扑：

- 训练侧：FSDP/EP/Megatron，Starweave `parallel` 或 verl engine；
- 推理侧：vLLM TP/PP，vLLM 自己的 executor。

必须明确谁是 owner：

1. **verl 是 owner**（方式 A）：Starweave 只做配置和启动，不初始化自己的
   process group；vLLM 由 verl 的 rollout worker 管理。
2. **Starweave 是 owner**（方式 B）：`task-rl` 管理 actor 训练和 rollout 的
   资源划分；vLLM 以子进程/独立 engine 运行，通过 `update_weights` 同步。
3. **不要**两边都初始化 HCCL/NCCL 进程组，也不要让两个 launcher 各自
   `set_device`，否则会出现设备争用和集合通信死锁。

## 7. 能力边界矩阵

| 需求 | Starweave 配置能覆盖吗 | 做法 |
|---|---|---|
| 换 RL 算法（PPO/GRPO/DAPO） | 原生 task-rl：是；封装 verl：用 verl 配置 | task 插件或 verl 配置 |
| 换 optimizer（AdamW/Muon） | 原生 task-rl：是；封装 verl：通常否 | optimizer 插件；verl 内部需 adapter |
| 换 rollout 引擎（vLLM/SGLang） | 是（如果都实现 rollout seam） | `use.rollout` |
| 换数据形态（文本/多模态） | 是 | `use.data` |
| 改 vLLM scheduler/PagedAttention | 否 | vLLM 扩展点或 fork |
| 改 verl Ray worker 拓扑 | 否 | verl 扩展点或 fork |
| 复用 verl 的 engine 性能 | 部分（方式 C） | engine adapter |
| 复用 verl 的 reward/rollout 扩展 | 是（在 verl 内部） | verl 自己的插件/worker |

## 8. 推荐落地顺序

1. 先把 `task`、`rollout`、`reward` 三个 seam 定义清楚；SFT/inference 先用
   toy 实现验证组合。
2. 做 `rollout-vllm` 适配，先只支持离线 `generate`，再加 `update_weights`
   和 sleep/wake；在 Ascend 上先确认 `vllm-ascend` 与 dsv4 的可用性。
3. RL 先走方式 A（`task-rl-verl`）拿到可用的训练闭环，作为行为基线。
4. 需要组件替换时，再逐步把 verl 的算法/engine 拆成原生 `task-rl` +
   `engine-verl`（方式 C），最后才考虑完全原生（方式 B）。
5. 无论哪条路线，都不把 vLLM/verl 的内部类泄漏到 Starweave 的 `seams/`；
   适配层只出现在各自插件包内。
