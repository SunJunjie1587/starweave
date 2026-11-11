# 把 verl / vLLM 的能力抽象成 Starweave 插件

## 1. 目标

不是把 verl、vLLM 整体作为黑盒插件，而是：

- `task-rl` 插件在功能和性能上对齐 verl；
- `rollout` 插件在功能和性能上对齐 vLLM；
- 把两者的能力拆成 Starweave 可组合的插件。

关键结论：

> **可以抽象的是“算法、编排、策略、数据形态”；不能抽象的是“每 token 的热路径”。**
> PagedAttention、continuous batching、CUDA/HCCL graph、fused kernel 必须留在
> 一个 rollout engine 插件内部，不能拆成每步调用的 Python 插件。

## 2. 两层抽象

```text
Starweave 层（粗粒度，可组合）
  task-rl
  algorithm / reward / advantage / kl
  rollout（整个引擎）
  parallel（整个训练引擎）
  data / environment / checkpoint / metrics

引擎内部（细粒度，性能敏感，不跨 seam）
  scheduler / block manager / KV cache / attention kernel
  CUDA graph / HCCL / TP/PP executor
  sequence packing / micro-batch pipeline
```

插件边界只出现在粗粒度层：每个 task step、每个 batch、每个 rollout 请求批次、
每个 checkpoint。**不能**为每个 token、每个 layer、每次 scheduler 决策调用一次
Starweave 插件。

## 3. verl 能力拆解

verl 的价值主要不在“模型 kernel”，而在四层：

| 层 | verl 的代表能力 | Starweave 抽象 |
|---|---|---|
| 算法 | PPO / GRPO / DAPO / GSPO、advantage、KL、clip、reward norm | `task-rl` + `algorithm` 插件 |
| 数据/环境 | DataProto、sequence packing、multi-turn/agentic、reward manager | `data` + `environment` + `reward` 插件 |
| 训练引擎 | FSDP/FSDP2、Megatron、VeOmni engine、checkpoint resharding | `parallel` / `engine` 插件 |
| 编排 | Ray WorkerGroup、actor/rollout/ref/critic/reward 角色、权重同步 | `task-rl` + `launcher` / `placement` 插件 |
| rollout | vLLM/SGLang 集成、hybrid engine、sleep/wake | `rollout` 插件 |

拆解后的原则：

1. **算法层自己实现**：PPO/GRPO 的循环、advantage、KL、clip 是纯逻辑，
   适合做成 `algorithm` 插件；这部分不难，也不影响性能。
2. **训练引擎不要重写**：FSDP2、Megatron、VeOmni 本身就是库，直接作为
   `parallel-fsdp2`、`parallel-megatron` 插件使用。verl 也是调用它们。
3. **rollout 不要重写 vLLM 的热路径**：把 vLLM 作为 `rollout-vllm` 插件的
   内部引擎，或者实现一个同等机制的 engine，但都封装在一个插件里。
4. **编排自己实现**：actor/ref/critic/reward 的角色管理、权重同步、
   colocated/disaggregated 资源划分，属于 `task-rl` 的职责。

这样你不需要“把 verl 作为插件”，而是把 verl 的**算法和编排思想**变成 Starweave
插件，把 verl 使用的**底层引擎**（FSDP/Megatron/vLLM）作为插件内部依赖。

## 4. vLLM 能力拆解

vLLM 的能力分为两类：

### 4.1 可以作为 seam 的能力

| 能力 | Starweave 抽象 |
|---|---|
| `generate(prompts, sampling_params)` | `rollout.generate` |
| logprobs / top-k logprobs | `rollout.logprobs` |
| 权重更新 | `rollout.update_weights` |
| 显存让渡 | `rollout.sleep/wake_up` |
| TP/PP、max_model_len、gpu_memory_utilization | `rollout` 插件配置 |
| 量化、LoRA、speculative decoding | `rollout` 插件配置/内部扩展 |

### 4.2 不能拆成 seam 的能力

| 能力 | 原因 |
|---|---|
| PagedAttention / block manager | 每 token/每 block 的热路径，跨 Python 边界会毁掉性能 |
| continuous batching / scheduler | 每次 decode step 都在调度，插件回调太细 |
| CUDA graph / fused kernel | 必须由 engine 内部整体捕获和执行 |
| executor / TP/PP 进程拓扑 | 与 engine 生命周期绑定，不能由外部逐步控制 |

所以 `rollout` seam 的粒度是“一次 generate 调用/一个请求批次”，不是
“一次 scheduler step”。

## 5. 建议的 RL 插件族

```text
plugins/
├── tasks/
│   ├── task-sft/
│   ├── task-rl/
│   └── task-inference/
├── algorithms/
│   ├── algorithm-grpo/
│   ├── algorithm-ppo/
│   └── algorithm-dapo/
├── rewards/
│   ├── reward-rule/
│   └── reward-model/
├── rollouts/
│   ├── rollout-hf/          # 正确性基线，慢
│   ├── rollout-vllm/        # 性能后端，vLLM 作为内部引擎
│   └── rollout-sglang/
├── parallel/
│   ├── parallel-fsdp2/
│   ├── parallel-megatron/
│   └── parallel-veomni/
├── data/
│   ├── data-text-jsonl/
│   ├── data-agentic/
│   └── data-multimodal/
├── environments/
│   └── environment-tool-use/
└── policies/
    ├── policy-advantage/
    ├── policy-kl/
    └── policy-clip/
```

`task-rl` 的依赖：

```python
requires = ("model", "data", "rollout", "reward", "algorithm",
            "optimizer", "scheduler", "loss", "checkpoint",
            "metrics", "parallel")
```

`algorithm` 插件是纯 batch-level 逻辑：

```python
class GRPO(PluginBase):
    name = "algorithm-grpo"
    provides = ("algorithm",)

    def compute_advantages(self, trajectories, rewards): ...
    def policy_loss(self, logprobs, old_logprobs, advantages, ref_logprobs): ...
```

它不碰模型前向、不碰 rollout engine，因此插件边界开销可以忽略。

## 6. 性能对齐策略

功能对齐靠插件拆分；性能对齐靠**不要拆热路径**。

### 6.1 复用同一套底层引擎

verl 的性能来自 FSDP2/Megatron/vLLM，不是来自它自己的 Python 循环。Starweave
要达到同等性能，应该：

- `parallel-fsdp2` 直接使用 PyTorch FSDP2；
- `parallel-megatron` 直接使用 Megatron core；
- `rollout-vllm` 直接在插件内部使用 vLLM engine（或 vllm-ascend）；
- `task-rl` 只做编排，不介入 engine 内部。

这相当于“用 verl 的零件，不用 verl 的控制器”。

### 6.2 粗粒度边界

| 边界 | 频率 | 允许 |
|---|---|---|
| task step / optimizer step | 每 step | 是 |
| rollout.generate 批次 | 每 batch | 是 |
| reward/algorithm 计算 | 每 batch | 是 |
| per-token scheduler 回调 | 每 token | 否 |
| per-layer attention 回调 | 每 layer | 否 |

Starweave 的静态 `StepPlan` 已经把 step 内的调用绑定成 callable；rollout engine
内部不应该再经过 Starweave registry。

### 6.3 权重同步

RL 性能的一个大头是 actor → rollout 的权重同步：

- 优先 HCCL/NCCL broadcast，避免 state_dict 序列化；
- 支持 colocated（同进程/同卡，sleep/wake）和 disaggregated（独立进程组）；
- 同步是 `rollout.update_weights` 的粗粒度操作，不能每层/每参数回调。

### 6.4 基准矩阵

对齐性能必须有可重复的基准：

```text
模型: dsv4-small / 7B / 真实目标模型
硬件: 单卡 / 8 卡 NPU
指标:
  rollout: tokens/s, TTFT, TPOT, batch throughput
  train: step time, MFU, peak memory
  rl: end-to-end step time, weight sync time, rollout/train 占比
基线: verl + vLLM 同配置
```

先做到“同引擎、同并行配置”的 apples-to-apples，再比较 Starweave 的编排开销。
如果编排开销超过 step 时间的 5%，再优化；不要为了插件化牺牲 engine 内部性能。

## 7. 分阶段路线

### 阶段 1：正确性基线

- `task-rl` + `rollout-hf` + 简单 reward，跑通 GRPO 小模型；
- 不追求性能，只验证 seam 组合、权重同步、checkpoint、resume。

### 阶段 2：性能后端

- `rollout-vllm` 插件内部使用 vLLM/vllm-ascend；
- `parallel-fsdp2` 插件直接使用 PyTorch FSDP2；
- 与 verl + vLLM 同配置对比，先对齐单卡/小模型，再上多卡。

### 阶段 3：功能补齐

- PPO（critic）、reference policy、KL 控制；
- sequence packing、dynamic batch、multi-turn/agentic；
- checkpoint resharding、fault tolerance、async rollout；
- 每项功能对应一个 algorithm/reward/environment/data 插件或配置。

### 阶段 4：组合矩阵

- 用 `use` 配置跑 model × task × algorithm × rollout × optimizer × data 的
  组合矩阵；
- 任何组合都不需要改 core；
- 性能回归作为 CI 的一部分。

## 8. 反模式

| 反模式 | 后果 |
|---|---|
| 把 vLLM scheduler 拆成 Starweave policy 插件 | 每 token Python 回调，吞吐骤降 |
| 把 PagedAttention 抽象成 `kv_cache` seam | 跨边界频繁调用，无法用 CUDA graph |
| 把 verl 的 Ray WorkerGroup 拆成多个 Starweave 插件 | 进程/设备所有权冲突，集合通信死锁 |
| 在 task-rl 里自己实现 attention/MoE kernel | 性能不可能对齐 vLLM/Megatron |
| 为了“纯插件”拒绝复用 FSDP2/Megatron/vLLM | 重新实现数年工程，性能更差 |

## 10. 四种 RL 拓扑与插件阻塞

### 10.1 on-policy（同步）

```text
actor → update_weights → rollout.generate → train → actor
```

需要：

- `rollout.update_weights()` 和明确的同步 barrier；
- 每个样本带 `policy_version`，训练只接受当前版本；
- task-rl 控制“生成完再训练”的节奏。

Starweave 现状：不阻塞。`task` 可以表达循环，`rollout` seam 需要补
`update_weights` 和版本字段。

### 10.2 训推共卡（colocated）

训练引擎和 rollout 引擎共享同一组 NPU，时间片切换：

```text
train step → offload/sleep training → wake rollout → generate
          → sleep rollout → resume training
```

阻塞点：

| 阻塞 | 原因 |
|---|---|
| 设备所有权 | 两个 engine 都想 `set_device` / 建进程组 |
| 显存预算 | FSDP 参数/优化器状态与 vLLM KV cache 争用同一张卡 |
| sleep/wake | vLLM 需要释放 KV cache；训练侧需要 offload 参数/优化器 |
| 进程模型 | vLLM 可能在独立进程/executor；FSDP 在另一进程 |
| 生命周期 | 两个 engine 的初始化/销毁顺序必须由同一个 placement 管理 |

需要的 seam：

- `placement`：产出 `PlacementPlan`，声明 train/rollout 共卡、显存预算、
  进程边界；
- `rollout.sleep/wake_up`；
- `parallel` engine 的 offload/resume 接口；
- 同卡权重同步（同进程拷贝、CUDA IPC、HCCL broadcast）。

这是四种模式里工程风险最高的。插件架构能表达，但 **placement/resource seam
是必须新增的**，否则两个插件会各自抢占设备。

### 10.3 训推异卡（disaggregated）

训练和 rollout 使用不同设备组：

```text
train group:   NPU 0-3
rollout group: NPU 4-7
```

阻塞点：

| 阻塞 | 原因 |
|---|---|
| 单进程 runtime | 当前 `Runtime` 在一个进程里建一个模型、一个 device |
| launcher | 需要 worker group / 多进程 / 多机启动 |
| 权重传输 | actor → rollout 跨设备组，需要 HCCL/NCCL/RDMA 或 host staging |
| checkpoint | 训练侧 checkpoint 要能 reshard 到 rollout 的 TP/PP 拓扑 |
| 指标/phase | 多个进程的 metrics、phase barrier、错误传播 |

需要的 seam：

- `launcher` / `worker_group`：启动 train/rollout 两组 worker；
- `placement`：设备组、rank 映射；
- `rollout.update_weights`：跨组传输；
- role-aware `checkpoint`；
- 分布式的 `metrics` / `phase`。

Starweave 现状：**这是当前实现的主要阻塞**。`launcher-local` 是进程内执行，
`collective-none`/`fsdp-none` 是 no-op，runtime 没有 worker group 概念。
不是插件思想的问题，是 runtime 还停在单进程。

### 10.4 推理 PD 分离

P/D 分离是 rollout engine 内部拓扑：

```text
prefill workers → KV cache transfer → decode workers
```

关键：**不要把 P/D 做成 Starweave 的 per-token seam**。它应该完全留在
`rollout-vllm` / 自研 rollout engine 内部：

- `rollout.generate()` 对 task-rl 不变；
- P/D 的 worker 数量、KV 传输后端、调度策略由 rollout 插件配置；
- `placement` 只负责给 prefill/decode 预留设备组，不参与每次 KV 传输。

阻塞点：

- vLLM/vllm-ascend 的 P/D 支持程度需要验证；
- KV transfer 需要 RDMA/HCCL/NIXL 等后端，跨节点时更复杂；
- P/D 调度和 KV cache 生命周期必须由 engine 统一管理。

如果 rollout engine 支持 P/D，Starweave 侧只需要 `placement` 声明资源；
架构上不阻塞。如果自研 engine，P/D 是 engine 内部工作，不是插件边界。

### 10.5 汇总

| 模式 | 架构是否阻塞 | 主要缺口 | 优先级 |
|---|---|---|---|
| on-policy 同步 | 否 | rollout.update_weights、policy_version | P0 |
| 训推共卡 | 需新增 placement | 设备/显存/进程所有权 | P1（高风险） |
| 训推异卡 | 当前 runtime 阻塞 | worker group、launcher、跨组权重传输 | P1 |
| PD 分离 | 否（若 engine 支持） | rollout engine 内部 P/D、KV transfer | P2 |
| transfer queue | 否 | experience queue seam + adapter | P1 |

## 11. verl Transfer Queue 作为插件

可以，但不要把它直接暴露成 seam。定义一个中立的 `experience` seam：

```python
class ExperienceQueue(Protocol):
    def put(self, batch, *, policy_version: int) -> str: ...
    def get(self, *, min_version: int | None = None,
            timeout: float | None = None) -> ExperienceBatch: ...
    def ack(self, batch_id: str) -> None: ...
    def stats(self) -> dict[str, float]: ...
    def close(self) -> None: ...
```

然后提供适配插件：

```text
plugins/experience/
├── experience-local/          # 进程内队列，同步 on-policy
├── experience-shm/            # 同机共享内存，共卡/异进程
├── experience-transfer-verl/  # 适配 verl 的 TransferQueue
└── experience-ray/            # Ray actor 队列，多机
```

`experience-transfer-verl` 内部 import verl 的 TransferQueue，对外只实现
`put/get/ack/stats/close`。这样：

- Starweave 的 `seams/` 不依赖 verl；
- 换成本地队列或 Ray 队列不影响 task-rl；
- 删除 verl 适配插件不影响其他插件。

需要特别定义的语义：

| 语义 | 要求 |
|---|---|
| 样本版本 | 每个 batch 带 `policy_version`；严格 on-policy 只取当前版本 |
| 背压 | 队列容量、put 阻塞/丢弃策略、最大 staleness |
| 传输 | 同机零拷贝、跨机 RDMA/HCCL、CPU staging；dtype/device 明确 |
| 确认 | `ack` 后才释放样本；失败重试/幂等 |
| 生命周期 | checkpoint 前 drain；phase 切换 flush/close |
| 可观测性 | 队列深度、延迟、staleness、吞吐 |

注意区分两种“队列”：

- **experience queue**：rollout 产样本 → trainer 消费，属于 RL 数据面；
- **KV transfer**：prefill → decode 的 KV cache 传输，属于 rollout engine 内部。

不要把两者合并成同一个 seam；它们的粒度、生命周期、性能要求完全不同。

## 12. 当前实现的具体阻塞

| 当前实现 | 阻塞的拓扑 | 需要改成 |
|---|---|---|
| `Runtime` 单进程、单 device | 训推异卡、共卡多进程 | `Cluster`/`WorkerGroup` + placement |
| `launcher-local` 进程内执行 | 异卡/多机 | `launcher-ray` / `launcher-mp` 插件 |
| `model` 单例 handle | PPO 的 actor/ref/critic | role-scoped `ModelBuilder` |
| `TrainingServices` 单模型/单优化器 | 多角色训练 | 按 role 组织 services |
| `checkpoint` 单状态 | 跨拓扑 resharding | role-aware checkpoint seam |
| `collective/fsdp/ep` no-op | 所有多卡模式 | 真实 parallel 插件 |
| `phase` 无分布式 barrier | 异卡/多进程 | 分布式 phase 协调 |
| `Context/Scope` 只在单进程 | 共卡/异卡 | 控制面与 worker 生命周期分离 |

这些是**实现缺口**，不是插件架构的根本矛盾。但其中“单进程 runtime”和
“单例 model”必须先改，否则共卡/异卡只能停留在配置层。

## 13. 需要新增的 seam

```text
seams/
├── placement.py    # PlacementPlan：设备组、共卡/异卡、显存预算、rank 映射
├── rollout.py      # generate / logprobs / update_weights / sleep / wake_up
├── experience.py   # put / get / ack / stats（transfer queue 的抽象）
├── worker.py       # WorkerGroup / launcher 契约
├── model.py        # role-scoped ModelBuilder
└── checkpoint.py   # role-aware、可 reshard
```

其中 `placement` 是共卡/异卡/PD 三种模式的共同基础：先决定“谁在哪些设备上、
是否共卡、进程边界在哪”，再让 rollout/parallel/launcher 插件按计划执行。
没有 placement，各插件会各自假设设备所有权，必然冲突。

## 14. 结论

- **功能对齐**：把 verl 拆成 task、algorithm、reward、data、environment、
  parallel、checkpoint、metrics 插件；把 vLLM 收敛成一个 rollout engine 插件。
- **性能对齐**：复用 FSDP2/Megatron/vLLM 的底层引擎和 kernel；插件边界保持在
  task/batch/engine 级别；热路径不跨 seam。
- **拓扑支持**：on-policy 不阻塞；共卡/异卡需要新增 `placement` + worker group；
  PD 分离留在 rollout engine 内部；transfer queue 用 `experience` seam 适配。
- **当前最大阻塞**：runtime 还是单进程、单模型；先补 role-scoped model 和
  worker/placement 层，四种拓扑才有落地基础。
- **自由组合**：model、task、algorithm、rollout、optimizer、data、parallel
  各自独立替换；组合由 `use` 配置表达。
- **不要**试图把 vLLM 的每个内部机制都变成 Starweave 插件；那是性能的反面。
