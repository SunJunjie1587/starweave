# Starweave v1 规格

- 项目：Starweave
- 命令行：`starweave`
- Python 包：`starweave`
- 入口点组：`starweave.plugins`
- 文档版本：1.1
- 日期：2026-11-10
- 状态：v1 已实现并通过验收（`docs/acceptance.md`）；本规格仍是后续变更的事实源

## 变更记录

- **v1.1（2026-11-10）**：用户裁决 v1 使用 dsv4 小配置单机验证；删除 CPU device backend；真实 284B dsv4 多卡训练列为 v2。
- **v1.0（2026-11-10）**：初稿。

## 阅读指南

| 你想了解 | 看哪一节 |
|---|---|
| 为什么做、做到什么程度 | §1 目标与非目标 |
| 内核怎么分层、插件怎么解耦 | §2 设计原则、§3 分层架构、§4 核心概念 |
| 有哪些 seam、每个 seam 的接口 | §5 Seam 清单与契约 |
| recipe / bundle / patch 怎么写 | §6 组合方言 |
| 启动时怎么校验、怎么编译 | §7 Plan compiler |
| 命令行长什么样 | §8 CLI |
| 目录怎么放 | §9 仓库结构 |
| train/eval/checkpoint 怎么切换 | §10 Phase 机制 |
| 怎么保证可复现 | §11 可复现性 |
| 出错怎么办 | §12 错误模型 |
| 怎么测、怎么验收 | §13 测试策略、§14 验收矩阵 |
| 已知限制和风险 | §15 风险、§16 已知限制 |
| 硬约束是否都覆盖 | §17 自检 |

---

## 1. 目标与非目标

### 1.1 目标

**G1 硬件无关。** 上层 trainer、optimizer、data、inference、checkpoint 不出现任何硬件 API。硬件能力由 `device`、`kernel`、`collective`、`allocator`、`rng` 等 seam 提供。v1 只实现 Ascend NPU backend，**不提供 CPU device backend**；CPU 仅用于运行不依赖设备的单元测试与静态检查。

**G2 全插件化。** trainer、inference、加速特性、数据处理、optimizer、scheduler、loss、checkpoint、metrics、profiler、policy 全部是插件。核心只保留 kernel、seams、compose、plan、CLI。

**G3 任意组合。** 通过 recipe + bundle + patch 组合插件；插件之间不互相 import；组合合法性由 `provides` / `requires` / seam 基数在编译期校验。

**G4 热路径静态。** 启动时把 train step / inference step 编译成不可变的 `ExecutionPlan`；每 step 不查注册表、不做动态分发。

**G5 可观测。** `dump-config` 显示每行来源；`dump-schema` 输出每个插件的 JSON Schema；`inspect --plan` 打印编译后的服务、步骤、pinned/rebindable 和 hooks；`composition_hash` 稳定。

**G6 可复现。** 种子、数据位置、RNG、配置快照、composition_hash 全部进入 checkpoint；支持精确 resume。

**G7 v1 端到端。** 单卡 NPU + dsv4 小配置（缩小层数/专家数/序列长度，保留 CSA/HCA、MoE、mHC、Lightning Indexer 等 dsv4 核心结构），完成 SFT 训练、checkpoint 保存/恢复、推理生成。真实 284B dsv4 多卡训练属于 v2。

### 1.2 非目标

v1 **不实现**，只保留 seam 或文档说明：

- **N1** 真实的多机多卡并行执行（FSDP2、EP、TP、PP、DeepSpeed、HCCL 集合通信编排）。v1 只提供 `collective-none` / `fsdp-none` / `ep-none` no-op 插件；真实并行后端是 v2。
- **N2** RL 全链路（rollout ↔ train）。
- **N3** GPU backend 实现（CUDA 相关代码不进入 v1）。
- **N4** 运行时插件装卸、配置热更新、HMR。
- **N5** Web UI、常驻服务、插件市场。
- **N6** MindSpeed 级融合算子、量化、通信重叠。
- **N7** PEFT / LoRA / QLoRA。
- **N8** 真实 284B DeepSeek-V4 多卡训练，以及 FSDP2 / EP / CP 并行编排。v1 只做 dsv4 小配置单机验证。

如果实现过程中发现非目标被硬约束倒逼必须做，停止并上报，不得擅自扩 scope。

### 1.3 成功标准

| 编号 | 标准 | 验证方式 |
|---|---|---|
| S1 | 换 optimizer/data/model/device 插件只改 YAML，不改核心代码 | M3 外部插件 + recipe 切换测试 |
| S2 | 单卡 NPU 上 dsv4 小配置完成 50 step SFT，loss 下降，checkpoint 可存可载 | M2 验收 |
| S3 | 无 NPU 时静态检查与单元测试通过；NPU 测试 skip；E2E 在 NPU 目标机执行 | M0/M1 验收 |
| S4 | `dump-config` 确定性、带来源；`composition_hash` 稳定 | M0 golden snapshot |
| S5 | `inspect --plan` 能看清 pinned/rebindable/hooks；重绑 pinned 被拒绝 | M0/M3 验收 |
| S6 | 插件层相对裸 PyTorch loop 的开销有测量报告，目标 < 5% | M3 验收 |

---

## 2. 设计原则

继承 deepseek-harness 的五条原则：

| 编号 | 原则 | 在 Starweave 中的落地 |
|---|---|---|
| P1 | 一切能力都是 seam | 插件只依赖 `starweave.seams`；不允许 import kernel/compose/plan/CLI/其他插件 |
| P2 | 激活由依赖可用性决定 | compose 校验 `provides` / `requires`；YAML 行序不是加载顺序 |
| P3 | 组合是分层 patch | bundles → recipe patch → settings → `--set` → `--patch`；config 整体替换 |
| P4 | 横切策略也是插件 | grad-clip、nan-guard、grad-accum、checkpoint/eval policy 叠加在 step seam 上 |
| P5 | 运行时必须可 dump / inspect | `dump-config`、`dump-schema`、`inspect --plan`、`composition_hash` |

补充三条 Starweave 原则：

| 编号 | 原则 | 含义 |
|---|---|---|
| S1 | 导入边界就是解耦边界 | `starweave.seams` 不 import kernel/compose/plan/CLI，也不 import torch |
| S2 | 热路径静态、编排动态 | 热循环编译成静态 plan；长生命周期服务挂在运行期 registry，只在 phase 边界重绑 |
| S3 | 单发行包，插件只依赖 seam | v1 只发布 `starweave` 一个发行包；外部插件依赖它但只允许 import `starweave.seams.*` |
| S4 | 并行策略也是插件 | `collective`、`fsdp`、`ep` 是 seam；v1 提供 no-op 实现，v2 提供 HCCL / FSDP2 / EP 实现，内核不写死并行逻辑 |

### 2.1 关键决策：单发行包与 seam 独立

C7/C8 要求“单发行包 + entry points”，而 dsh 原则要求 seam 独立。v1 的决策是：

- **只发布一个发行包 `starweave`**，包含 kernel、seams、compose、plan、CLI 和第一方插件。
- **`starweave.seams` 是独立导入边界**：它只 import 标准库、`typing` 和 `pydantic`，不 import `starweave.kernel` / `compose` / `plan` / `cli`，也不 import `torch`。
- 外部插件依赖 `starweave`，但只允许 `from starweave.seams... import ...`；`check_decoupling.py` 阻断其他框架导入。
- 未来若需要让插件不依赖整个发行包，可把 `starweave.seams` 拆成独立发行包 `starweave-seams`，插件 import 路径不变。

---

## 3. 分层架构

```
┌────────────────────────────────────────────────────────────┐
│ CLI / recipes / bundles / patches                          │  组合层
├────────────────────────────────────────────────────────────┤
│ plan compiler（validate → ExecutionPlan）                  │  编译层
├────────────────────────────────────────────────────────────┤
│ runtime registry（pinned + rebindable services）           │  运行层
├────────────────────────────────────────────────────────────┤
│ seams（Protocol + dataclass；无 kernel/plan/torch 依赖）   │  接口层
├────────────────────────────────────────────────────────────┤
│ kernel（Context / Plugin / Scope / Events / Errors）       │  内核层
└────────────────────────────────────────────────────────────┘
        ▲
        │ 只依赖 seams
┌───────┴────────────────────────────────────────────────────┐
│ plugins：device / kernel / model / data / optimizer / ...  │  插件层
└────────────────────────────────────────────────────────────┘
```

### 3.1 各层职责

| 层 | 职责 | 允许 import |
|---|---|---|
| kernel | Context、插件生命周期、服务注册、事件、错误 | 仅标准库 + pydantic |
| seams | 全部 Protocol、dataclass、配置基类 | 标准库 + typing + pydantic；**禁止 torch / kernel / compose / plan / cli** |
| compose | bundle / recipe / patch 加载、分层合成、composition_hash | kernel + seams |
| plan | 校验、编译、冻结、执行计划打印 | kernel + seams + compose |
| cli | 命令解析、输出、退出码 | 全部内部模块 |
| plugins | 第一方实现 | **仅 seams** + 自身第三方依赖（torch、transformers 等） |
| 外部插件 | 第三方实现 | **仅 `starweave.seams`** + 自身依赖 |

### 3.2 插件发现

- 第一方插件通过包内 entry points 注册：`[project.entry-points."starweave.plugins"]`。
- 外部插件安装后自动出现在 `starweave plugin list`。
- 开发模式允许 recipe 直接写 `module:Class`，但发布包仍必须提供 entry point。

---

## 4. 核心概念

### 4.1 Plugin

```python
class Plugin(Protocol):
    name: str
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    config_model: type[BaseModel]
    rebindable: bool
    priority: int
    optional: bool

    def setup(self, ctx: Context, config: BaseModel) -> None: ...
    def teardown(self) -> None: ...
```

规则：

- `provides` / `requires` 使用 seam 名，不使用类名或包名。
- `config_model` 必须是 pydantic v2 `BaseModel` 子类，`extra="forbid"`。
- `setup` 只做初始化；重活放到 seam 的 `build` / `compile` 方法。
- 所有资源清理通过 `ctx.effect(cleanup)` 注册；`teardown` 必须幂等。
- `rebindable=True` 只允许 metrics、profiler、checkpoint、inference；trainer/optimizer/scheduler/model 默认 `False`。
- `optional=True` 的插件失败时记录并继续；required 插件失败时中止。

### 4.2 Context

```python
class Context(Protocol):
    run_id: str
    paths: RunPaths          # run_dir / checkpoint_dir / metrics_path
    seed: int
    def get(self, seam: str) -> Any: ...
    def effect(self, cleanup: Callable[[], None]) -> None: ...
    def emit(self, event: str, payload: Any) -> None: ...
    def on(self, event: str, handler: Callable[[Any], None]) -> None: ...
```

- `get` 只允许读取 `requires` 中声明的 seam；读取未声明 seam 在编译期报错。
- `effect` 的清理函数按注册逆序执行。
- `emit` / `on` 是类型化事件总线；事件 payload 见 §11.3。

### 4.3 Seam 三类

| 类型 | 基数 | 选择方式 | 例子 |
|---|---|---|---|
| exclusive | 每个 phase 所需的 seam 恰好一个 provider | 按 seam 绑定；未使用的 exclusive seam 可以缺失 | device、precision、trainer、inference、launcher、collective、fsdp、ep |
| registry | 多个 provider | 按名称或 priority | kernel、model、data、optimizer、scheduler、loss、checkpoint、metrics、profiler |
| policy | 多个 provider，可叠加 | 按 priority 排序 | step_policy、checkpoint_policy、eval_policy |

exclusive seam 也可以声明 `rebindable`（如 inference），但 pinned 服务（trainer/optimizer/scheduler/model）不可重绑。

### 4.4 ExecutionPlan

编译产物，冻结后不可变：

```python
@dataclass(frozen=True)
class ServiceBinding:
    id: str
    seam: str
    provider: str
    instance: Any
    pinned: bool
    rebindable: bool
    source: str                  # 行来源：bundle/recipe/settings/cli/patch

@dataclass(frozen=True)
class StepPlan:
    before_step: tuple[Callable[[StepState], None], ...]
    forward_backward: Callable[[StepState], float]
    after_backward: tuple[Callable[[StepState], None], ...]
    optimizer_step: Callable[[StepState], None]
    after_step: tuple[Callable[[StepState], None], ...]

@dataclass(frozen=True)
class PhasePlan:
    name: str
    services: dict[str, ServiceBinding]
    step: StepPlan | None
    hooks: tuple[Callable[[RunState], None], ...]
    composition_hash: str

@dataclass(frozen=True)
class ExecutionPlan:
    run_id: str
    recipe_path: str
    phases: dict[str, PhasePlan]
    pinned: frozenset[str]
    rebindable: frozenset[str]
    diagnostics: tuple[Diagnostic, ...]
```

`StepPlan` 中的 callable 已经绑定好服务和方法，热路径直接调用，不做 `ctx.get`。

### 4.5 Composition

- **bundle**：可复用的插件行 YAML，可随包分发或本地文件。
- **recipe**：`name`、有序 `bundles`、`patch`、`phases`。
- **patch**：`insert` / 按 id 覆盖 / `disabled` / `group`。
- 合成结果、插件版本、框架版本生成 `composition_hash`。

### 4.6 Scope 与生命周期

- 每个插件一个 scope；`setup` 开始，`teardown` 结束。
- 启动失败时，已 setup 的插件按逆序 teardown。
- 插件不得持有已释放服务的引用；`teardown` 后继续使用视为错误。

### 4.7 Entry points

```toml
[project.entry-points."starweave.plugins"]
adamw = "starweave.plugins.optimizer_adamw:AdamWPlugin"
```

- 名称在全组合内唯一；重复名称由 composer 报错。
- `starweave plugin list` 读取所有已安装 entry points。
- recipe 的 `plugin:` 字段优先按 entry point 名解析，找不到再按 `module:Class` 解析。

---

## 5. Seam 清单与契约

`starweave/seams/` 只包含 Protocol、dataclass、枚举和配置基类，**不 import torch**。torch/transformers 对象以 `Any` 或 opaque handle 传递。

### 5.1 公共类型

```python
@dataclass(frozen=True)
class ParallelPlan:
    shard_modules: tuple[str, ...]        # 例如 ("model.layers.{*}", "lm_head")
    expert_modules: tuple[str, ...]       # 例如 ("model.layers.{*}.mlp.experts",)
    replicate_modules: tuple[str, ...]    # 例如 ("model.hc_head",)

@dataclass(frozen=True)
class ModelHandle:
    model: Any
    tokenizer: Any
    vocab_size: int
    param_count: int
    dtype: str
    parallel_plan: ParallelPlan

@dataclass(frozen=True)
class DataBundle:
    dataset: Any
    collator: Callable[[list[dict[str, Any]]], dict[str, Any]]
    iterator_factory: Callable[[], Iterator[dict[str, Any]]]

@dataclass
class OptimizerHandle:
    optimizer: Any
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...

@dataclass
class SchedulerHandle:
    scheduler: Any
    def step(self) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...

@dataclass
class CheckpointState:
    model_dir: Path
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    step: int
    dataloader_position: int
    rng: dict[str, Any]
    composition_hash: str
    config_snapshot: dict[str, Any]

@dataclass
class StepState:
    step: int
    micro_step: int
    loss: float | None
    lr: float | None
    should_step: bool
    should_abort: bool
    error: BaseException | None

@dataclass(frozen=True)
class StepEvent:
    step: int
    loss: float
    lr: float
    tokens: int
    elapsed_ms: float

@dataclass(frozen=True)
class RunState:
    run_id: str
    phase: str
    step: int
    epoch: int
    best_metric: float | None

@dataclass(frozen=True)
class MetricEvent:
    name: str
    value: float
    step: int
    phase: str
    timestamp: float

class Rebindable(Protocol):
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def release(self) -> None: ...
```

metrics、checkpoint、profiler、inference 插件除实现各自 seam 外，还必须实现 `Rebindable`。`seams` 提供 `BaseRebindable`，默认三个方法都是 no-op；插件按需覆盖。

### 5.2 exclusive seam

#### device

```python
class Device(Protocol):
    name: str
    device_type: str                      # "npu"（v1 唯一实现）
    def set_device(self, index: int) -> None: ...
    def current_device(self) -> int: ...
    def synchronize(self) -> None: ...
    def memory_stats(self) -> dict[str, int]: ...
    def empty_cache(self) -> None: ...
```

`DeviceConfig`：`index: int = 0`。

#### precision

```python
class Precision(Protocol):
    name: str
    dtype: str                            # "float32" | "bfloat16"
    def autocast(self) -> AbstractContextManager: ...
    def cast_model(self, model: Any) -> Any: ...
```

`PrecisionConfig`：`dtype: Literal["float32", "bfloat16"]`。

#### trainer

```python
class Trainer(Protocol):
    name: str
    def compile(self, ctx: Context, services: TrainingServices) -> StepPlan: ...
    def run(self, ctx: Context, plan: StepPlan, max_steps: int) -> Iterator[StepEvent]: ...
```

`TrainingServices`：`model`、`data`、`optimizer`、`scheduler`、`loss`、`device`、`precision`、`step_policies`、`checkpoint`、`metrics`。
`TrainerConfig`：`max_steps: int = 1000`、`micro_batch_size: int = 1`、`log_every: int = 10`。eval / checkpoint 的触发频率由 `eval_policy` / `checkpoint_policy` 插件决定。

#### inference

```python
class InferenceEngine(Protocol):
    name: str
    def load(self, ctx: Context, model: ModelHandle, config: InferenceConfig) -> None: ...
    def generate(self, ctx: Context, requests: list[GenerateRequest]) -> Iterator[GenerateResult]: ...
```

`InferenceConfig`：`max_new_tokens: int = 256`、`temperature: float = 0.7`、`top_p: float = 0.9`、`batch_size: int = 8`、`do_sample: bool = False`。

#### launcher

```python
class Launcher(Protocol):
    name: str
    def run(self, ctx: Context, fn: Callable[[], int]) -> int: ...
```

`LauncherConfig`：无字段。v1 只实现单进程本地启动。

#### collective

```python
class Collective(Protocol):
    name: str
    world_size: int
    rank: int
    def all_reduce(self, tensor: Any, op: str = "sum") -> Any: ...
    def all_gather(self, tensor: Any) -> list[Any]: ...
    def broadcast(self, tensor: Any, src: int = 0) -> Any: ...
    def all_to_all(self, tensor: Any) -> Any: ...
    def barrier(self) -> None: ...
```

`CollectiveConfig`：`backend: Literal["none", "hccl"] = "none"`、`world_size: int = 1`。v1 只实现 `collective-none`（单进程 no-op）。

#### fsdp

```python
class FSDPBackend(Protocol):
    name: str
    def apply(self, ctx: Context, model: Any, plan: ParallelPlan) -> Any: ...
    def no_sync(self) -> AbstractContextManager: ...
    def state_dict(self, model: Any) -> dict[str, Any]: ...
    def load_state_dict(self, model: Any, state: dict[str, Any]) -> None: ...
```

`FSDPConfig`：`enabled: bool = False`、`strategy: Literal["none", "fsdp2"] = "none"`。v1 只实现 `fsdp-none`（不切分）；`fsdp2` 是 v2 插件，接口相同。

#### ep

```python
class EPBackend(Protocol):
    name: str
    def apply(self, ctx: Context, model: Any, plan: ParallelPlan) -> Any: ...
    def dispatch(self, tokens: Any, scores: Any) -> Any: ...
    def combine(self, outputs: Any) -> Any: ...
```

`EPConfig`：`enabled: bool = False`、`dispatcher: Literal["none", "eager", "mc2"] = "none"`、`ep_size: int = 1`。v1 只实现 `ep-none`（全部专家留在本地）；`ep-native` 是 v2 插件，接口相同。

并行插件与模型解耦：模型插件在 `ModelHandle.parallel_plan` 中声明可切分模块、专家模块、复制模块；`fsdp` / `ep` 插件只消费这份 plan，不 import 具体模型代码。

`fsdp.apply` / `ep.apply` 返回包装后的 model；plan compiler 用它替换 `ModelHandle.model`，再构建 optimizer / scheduler。

### 5.3 registry seam

#### kernel

```python
class KernelRegistry(Protocol):
    def register(self, op: str, impl: Callable[..., Any], *, backend: str,
                 dtype: str | None = None, priority: int = 100) -> None: ...
    def get(self, op: str, *, backend: str, dtype: str | None = None) -> Callable[..., Any]: ...
    def list(self) -> list[KernelRef]: ...
```

`KernelConfig`：`attention: Literal["sdpa", "npu_fusion"] = "sdpa"`、`rms_norm: bool = True`。

#### model

```python
class ModelBuilder(Protocol):
    name: str
    def build(self, ctx: Context, config: ModelConfig) -> ModelHandle: ...
```

`ModelConfig`：`name_or_path: str`、`dtype: str = "bfloat16"`、`trust_remote_code: bool = False`、`tokenizer: str | None = None`。

`model-dsv4` 使用自己的 `Dsv4ModelConfig`（见 §5.5.1），不强制复用 `ModelConfig`；`ModelBuilder.build` 接受任意 pydantic `BaseModel`。

#### data

```python
class DataSource(Protocol):
    name: str
    def build(self, ctx: Context, config: DataConfig) -> DataBundle: ...
```

`DataConfig`：`path: str`、`split: str = "train"`、`max_seq_len: int = 2048`、`prompt_field: str = "prompt"`、`response_field: str = "response"`、`batch_size: int = 1`、`shuffle: bool = True`、`seed: int = 42`、`use_chat_template: bool = False`。

SFT JSONL 每行：

```json
{"prompt": "用户输入", "response": "模型回答"}
```

- `prompt` token 的 label 设为 `-100`，不参与 loss；`response` token 参与 loss。
- 序列末尾追加 tokenizer 的 EOS；超过 `max_seq_len` 截断。
- `use_chat_template=true` 时使用 `tokenizer.apply_chat_template` 构造 prompt；否则直接拼接 `prompt + response`。

#### optimizer

```python
class OptimizerFactory(Protocol):
    name: str
    def build(self, ctx: Context, model: ModelHandle, config: OptimizerConfig) -> OptimizerHandle: ...
```

`OptimizerConfig`：`lr: float = 1e-4`、`weight_decay: float = 0.0`、`betas: tuple[float, float] = (0.9, 0.999)`、`eps: float = 1e-8`、`no_decay_keywords: list[str] = ["bias", "norm", "ln_", "embedding"]`。

#### scheduler

```python
class SchedulerFactory(Protocol):
    name: str
    def build(self, ctx: Context, optimizer: OptimizerHandle, total_steps: int,
              config: SchedulerConfig) -> SchedulerHandle: ...
```

`SchedulerConfig`：`warmup_ratio: float = 0.03`、`min_lr_ratio: float = 0.0`。

#### loss

```python
class LossFactory(Protocol):
    name: str
    def build(self, ctx: Context, config: LossConfig) -> Callable[[Any, Any], Any]: ...
```

`LossConfig`：`ignore_index: int = -100`。

#### checkpoint

```python
class CheckpointStore(Protocol):
    name: str
    def save(self, ctx: Context, state: CheckpointState) -> Path: ...
    def load(self, ctx: Context, path: Path) -> CheckpointState: ...
    def latest(self, ctx: Context) -> Path | None: ...
```

`CheckpointConfig`：`root: str = "runs/${run.id}/checkpoints"`、`keep_last: int = 2`。

#### metrics

```python
class MetricsSink(Protocol):
    name: str
    def emit(self, ctx: Context, event: MetricEvent) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

`MetricsConfig`：`path: str = "runs/${run.id}/metrics.jsonl"`。

#### profiler

```python
class Profiler(Protocol):
    name: str
    def start(self, ctx: Context) -> None: ...
    def stop(self) -> None: ...
```

`ProfilerConfig`：`enabled: bool = True`、`output: str | None = None`、`activities: list[str] | None = None`。

### 5.4 policy seam

```python
class StepPolicy(Protocol):
    name: str
    priority: int
    def before_step(self, ctx: Context, state: StepState) -> None: ...
    def after_backward(self, ctx: Context, state: StepState) -> None: ...
    def after_step(self, ctx: Context, state: StepState) -> None: ...
    def on_error(self, ctx: Context, state: StepState) -> None: ...

class CheckpointPolicy(Protocol):
    name: str
    priority: int
    def should_save(self, ctx: Context, state: RunState) -> bool: ...

class EvalPolicy(Protocol):
    name: str
    priority: int
    def should_eval(self, ctx: Context, state: RunState) -> bool: ...
```

seams 提供 `BaseStepPolicy`、`BaseCheckpointPolicy`、`BaseEvalPolicy`，默认 no-op；插件只覆盖需要的方法。

### 5.5 第一方插件与配置字段

| 插件 id | seam | 配置字段 |
|---|---|---|
| `device-npu` | device | `index` |
| `collective-none` | collective | `backend`、`world_size` |
| `fsdp-none` | fsdp | `enabled`、`strategy` |
| `ep-none` | ep | `enabled`、`dispatcher`、`ep_size` |
| `precision-fp32` | precision | `dtype` |
| `precision-bf16` | precision | `dtype` |
| `kernel-torch` | kernel | 无 |
| `kernel-npu` | kernel | `attention`、`rms_norm` |
| `model-hf` | model | `name_or_path`、`dtype`、`trust_remote_code`、`tokenizer` |
| `model-dsv4` | model | `config_path`、`checkpoint_path`、`tokenizer_name_or_path`、`dtype`、`attn_implementation` |
| `data-jsonl` | data | `path`、`max_seq_len`、`prompt_field`、`response_field`、`batch_size`、`shuffle`、`seed`、`use_chat_template` |
| `data-hf` | data | `path`、`split`、`max_seq_len`、`prompt_field`、`response_field`、`batch_size`、`shuffle`、`seed` |
| `optimizer-adamw` | optimizer | `lr`、`weight_decay`、`betas`、`eps`、`no_decay_keywords` |
| `scheduler-cosine` | scheduler | `warmup_ratio`、`min_lr_ratio` |
| `scheduler-linear` | scheduler | `warmup_ratio`、`min_lr_ratio` |
| `loss-ce` | loss | `ignore_index` |
| `checkpoint-safetensors` | checkpoint | `root`、`keep_last` |
| `checkpoint-torch` | checkpoint | `root`、`keep_last` |
| `metrics-jsonl` | metrics | `path` |
| `metrics-noop` | metrics | 无 |
| `profiler-off` | profiler | 无 |
| `profiler-npu` | profiler | `output`、`activities` |
| `trainer-sft` | trainer | `max_steps`、`micro_batch_size`、`log_every` |
| `inference-hf` | inference | `max_new_tokens`、`temperature`、`top_p`、`batch_size`、`do_sample` |
| `launcher-local` | launcher | 无 |
| `policy-grad-clip` | step_policy | `max_norm` |
| `policy-nan-guard` | step_policy | `action`、`threshold` |
| `policy-grad-accum` | step_policy | `steps` |
| `checkpoint-policy-every-n` | checkpoint_policy | `n` |
| `eval-policy-every-n` | eval_policy | `n` |

每个插件必须有：pydantic config model、contract test、README 条目。

表中 `model-dsv4` 属于独立官方插件包 `starweave-model-dsv4`；其余为 `starweave` 主发行包内插件。

`fsdp2`、`ep-native`、`collective-hccl` 是 **v2 插件**；v1 只发布同名 seam 接口与文档，不实现并行执行。v1 的 `collective-none`、`fsdp-none`、`ep-none` 是默认 no-op 实现：单进程、全部专家本地、不切分参数。

#### 5.5.1 model-dsv4 官方插件

`model-dsv4` 是独立官方插件包 `starweave-model-dsv4`，位于 `plugins/model_dsv4/`，不进入 `starweave` 主发行包的运行期依赖。

- **代码来源**：vendoring `mindspeed_llm/fsdp2/models/deepseek_v4_native/` 的 `configuration_deepseek_v4_native.py` 与 `modeling_deepseek_v4_native.py`，保留上游版权头；`LMHead` 与 `get_args` 用最小本地实现替代，避免运行时依赖 `mindspeed_llm`。
- **延迟导入**：`plugin.py` 模块顶层只 import `starweave.seams`、pydantic、标准库；torch / transformers / vendored modeling 在 `build()` 内导入，保证无 torch/NPU 的机器也能 `dump-config` 与 `inspect --plan`。
- **vendored 代码边界**：`_vendor/` 是第三方模型代码，保留上游对 `torch_npu` / `mindspeed` 的可选依赖；插件的 `plugin.py` / `config.py` 不得直接 import `torch_npu`。解耦校验器以 `--exclude _vendor` 豁免 vendored 代码，但插件自身代码仍全量检查，且 `_vendor/` 不得被主包或其他插件 import。
- **可选加速**：vendoring 代码中的 `mindspeed.ops` / `torch_npu` 导入保持 try/except；没有 MindSpeed 时回退到 torch 原生算子。
- **模型类**：`DeepseekV4ForCausalLM`，实现 `GenerationMixin`，可由 `inference-hf` 直接 `generate`。
- **小配置**：`configs/dsv4_small.json`，保留 CSA/HCA、MoE、mHC、Lightning Indexer；缩小层数/专家数/隐藏维度/序列长度。起始值见 §5.5.2。
- **并行声明**：`ModelHandle.parallel_plan` 声明 `shard_modules`（如 `model.layers.{*}`）、`expert_modules`（`model.layers.{*}.mlp.experts`）、`replicate_modules`（`model.hc_head`）；v1 的 `fsdp-none`/`ep-none` 忽略它，v2 的 `fsdp2`/`ep-native` 消费它。
- **权重**：支持从本地 dsv4 checkpoint 目录加载；没有 checkpoint 时可用随机初始化小配置跑通训练链路。
- **checkpoint**：模型 state dict 用 safetensors 保存；optimizer/scheduler 用 `torch.save`；config/tokenizer 一并保存。

#### 5.5.2 dsv4_small 起始配置

`configs/dsv4_small.json` 的起始值（M1 可根据 NPU 显存继续缩小，但必须保留 dsv4 的 CSA/HCA、MoE、mHC、Indexer 结构）：

```json
{
  "model_type": "deepseek_v4_native",
  "vocab_size": 129280,
  "hidden_size": 512,
  "moe_intermediate_size": 256,
  "num_hidden_layers": 4,
  "num_attention_heads": 8,
  "num_key_value_heads": 1,
  "head_dim": 64,
  "q_lora_rank": 128,
  "num_experts_per_tok": 2,
  "n_routed_experts": 8,
  "n_shared_experts": 1,
  "scoring_func": "sqrtsoftplus",
  "norm_topk_prob": true,
  "routed_scaling_factor": 1.5,
  "max_position_embeddings": 2048,
  "rope_theta": 10000.0,
  "layer_types": ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention", "sliding_attention"],
  "mlp_layer_types": ["moe", "moe", "moe", "moe"],
  "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
  "compress_rope_theta": 160000.0,
  "hc_mult": 2,
  "hc_sinkhorn_iters": 4,
  "hc_eps": 1e-6,
  "swiglu_limit": 10.0,
  "sliding_window": 128,
  "o_groups": 2,
  "o_lora_rank": 64,
  "index_n_heads": 4,
  "index_head_dim": 32,
  "index_topk": 64,
  "num_nextn_predict_layers": 0,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-6,
  "use_cache": true,
  "bos_token_id": 0,
  "eos_token_id": 1,
  "tie_word_embeddings": false,
  "attention_bias": false,
  "mlp_bias": false,
  "attention_dropout": 0.0
}
```

`mlp_layer_types` 全部用 `moe`，避免 hash-MoE 需要预训练 `tid2eid` 映射；真实 dsv4 的 `hash_moe` 调度在 v2 接真实 checkpoint 时启用。

`vocab_size` 必须与 `tokenizer_name_or_path` 的 tokenizer 词表一致；默认从仓库根 `tokenizer/` 读取。M1 前把 dsv4 tokenizer 放到该目录，或用 `--set model.tokenizer_name_or_path=/path` 覆盖；M0 静态检查不读取 tokenizer。

---

## 6. 组合方言

### 6.1 bundle

bundle 是一个 YAML 数组，元素是 patch 条目。它随 Python 包分发，或放在仓库的 `bundles/` 下。

```yaml
# bundles/base.yaml
- insert:
    - id: launcher
      plugin: starweave.plugins.launcher_local:LocalLauncher
    - id: collective
      plugin: starweave.plugins.collective_none:NoopCollective
    - id: fsdp
      plugin: starweave.plugins.fsdp_none:NoopFSDP
    - id: ep
      plugin: starweave.plugins.ep_none:NoopEP
    - id: precision
      plugin: starweave.plugins.precision_fp32:FP32Precision
    - id: kernel
      plugin: starweave.plugins.kernel_torch:TorchKernel
    - id: metrics
      plugin: starweave.plugins.metrics_jsonl:JSONLMetrics
      config: {path: "runs/${run.id}/metrics.jsonl"}
    - id: metrics-noop
      plugin: starweave.plugins.metrics_noop:NoopMetrics
    - id: profiler
      plugin: starweave.plugins.profiler_off:NoopProfiler
    - id: checkpoint
      plugin: starweave.plugins.checkpoint_safetensors:SafetensorsCheckpoint
      config: {root: "runs/${run.id}/checkpoints", keep_last: 2}
```

bundle 内容约定：

| bundle | 内容 |
|---|---|
| `base` | 插入 `launcher-local`、`collective-none`、`fsdp-none`、`ep-none`、`precision-fp32`、`kernel-torch`、`metrics-jsonl`、`metrics-noop`、`profiler-off`、`checkpoint-safetensors` |
| `npu` | 插入 `id: device`（插件 `device-npu`）；覆盖 `precision` → `precision-bf16`、`kernel` → `kernel-npu` |
| `hf` | 插入 `model-hf`、`data-jsonl`、`inference-hf` |
| `dsv4` | 覆盖 `id: model` → `model-dsv4`，配置 `configs/dsv4_small.json`；v1 保持 `fsdp-none` / `ep-none` |
| `sft` | 插入 `trainer`（trainer-sft）、`optimizer`（optimizer-adamw）、`scheduler`（scheduler-cosine）、`loss`（loss-ce）、`policy-grad-clip`、`policy-nan-guard`、`policy-grad-accum`、`checkpoint-policy`（checkpoint-policy-every-n）、`eval-policy`（eval-policy-every-n） |
| `inference` | 插入 `model-hf`、`inference-hf`；不插入 trainer/optimizer/scheduler |

所有训练/推理 recipe 至少包含 `base` + `npu`；`npu` 提供唯一的 device backend。`hf`、`dsv4`、`sft`、`inference` 是增量层；`inference` 与 `base` + `npu` 组合使用。`dsv4` 只覆盖 model 行，data/inference 仍由 `hf` 提供。

### 6.2 recipe

```yaml
name: sft-npu-dsv4-small
bundles: [base, npu, hf, dsv4, sft]
patch:
  - id: model
    plugin: dsv4
    config:
      config_path: configs/dsv4_small.json
      tokenizer_name_or_path: tokenizer
      dtype: bfloat16
      attn_implementation: eager
  - id: data
    config: {path: data/sft_dsv4.jsonl, max_seq_len: 1024, batch_size: 1, use_chat_template: false}
  - id: optimizer
    config: {lr: 1.0e-4, weight_decay: 0.01, betas: [0.9, 0.95]}
  - id: trainer
    config: {max_steps: 1000, micro_batch_size: 1, log_every: 10}
  - id: policy-grad-accum
    config: {steps: 8}
  - id: eval-policy
    config: {n: 100}
  - id: checkpoint-policy
    config: {n: 200}
phases:
  train:      {metrics: metrics, profiler: profiler, checkpoint: checkpoint}
  eval:       {metrics: metrics, profiler: profiler}
  checkpoint: {checkpoint: checkpoint, metrics: metrics-noop}
```

### 6.3 patch 条目

字段：

| 字段 | 含义 |
|---|---|
| `id` | 行标识；同一组合内唯一 |
| `plugin` | entry point 名或 `module:Class` |
| `config` | 插件配置；整体替换，不深合并 |
| `disabled` | `true` 时该行不激活；phase 仍可引用 |
| `group` | `true` 时 `config` 是嵌套行列表；composer 展开 |
| `inject` / `isolate` | 预留；v1 在编译期报错 |

规则：

- `insert` 追加行；带 `id` 且无 `insert` 的条目覆盖已有行。
- 覆盖时 `config` **整体替换**；未提供的字段回到插件默认值，不保留旧值。
- `group: true` 的行由 composer 展开，不产生服务；patch 可按 group id 插入。
- 行序不是加载顺序；激活由 `provides` / `requires` 校验决定。
- `disabled` 行仍保留在组合树中，phase 可以选择它；被选中时只为该 phase 实例化。

`group` 示例：

```yaml
- insert:
    - id: policies
      group: true
      config:
        - id: policy-grad-clip
          plugin: starweave.plugins.policy_grad_clip:GradClipPolicy
          config: {max_norm: 1.0}
        - id: policy-nan-guard
          plugin: starweave.plugins.policy_nan_guard:NanGuardPolicy
          config: {action: abort}
```

`group` 行由 composer 展开，不产生服务；patch 可以通过 group id 向内部插入行。

### 6.4 变量

只允许三种变量，禁止任意代码执行：

- `${env:VAR}`：环境变量；未设置时 compose 报错。
- `${run.id}`：本次运行 id。
- `${paths.run_dir}`：本次运行目录。

### 6.5 优先级

```
bundles 顺序 → recipe patch → settings.yaml（可选）→ CLI --set → --patch overlay
```

后者覆盖前者；`config` 整体替换。`--set` 只覆盖标量叶子，不做对象深合并。

`settings.yaml` 是可选层，格式为：

```yaml
trainer:
  max_steps: 1000
optimizer:
  lr: 2.0e-5
```

它只能覆盖已存在行的 `config` 标量字段，不能插入、删除或禁用行；结构变化必须走 recipe patch。

### 6.6 composition_hash

- 对最终组合树做 canonical JSON（键排序、去空白）。
- 加入每个插件的 `name/version`、框架版本、API 版本。
- SHA256 十六进制字符串。
- 写入 run log、checkpoint、`dump-config` 输出。
- 相同输入必须得到相同 hash；patch 变化必须改变 hash。

### 6.7 组合校验

compose 阶段必须拒绝：

- 未知插件名、不可导入的 `module:Class`。
- 重复 `id`。
- `config` 不符合插件 schema。
- 同一 exclusive seam 多个 provider。
- `requires` 不满足。
- 插件 API 版本与框架不兼容。
- `inject` / `isolate` 非空。
- 变量语法非法或环境变量缺失。
- phase 引用了不存在的行 id，或引用了非 rebindable seam。

---

## 7. Plan compiler

### 7.1 输入

- 合成后的组合树（每行带来源层）。
- settings（可选）。
- 目标 phase 列表（默认 `train`、`eval`、`checkpoint`）。
- 运行上下文（run_id、路径、种子）。

### 7.2 校验

按顺序执行：

1. 插件 import 成功；`name` 唯一。
2. `config` 通过 pydantic 校验。
3. 每个 phase 所需的 exclusive seam 恰好一个 provider：`train` 需要 trainer/optimizer/scheduler/model/data/device/collective/fsdp/ep；`inference` 需要 inference/model/device；未使用的 exclusive seam 可以缺失。
4. 每个 `requires` 在组合中存在。
5. API 版本范围兼容。
6. device 能力满足（NPU backend 需要 `torch_npu`；不满足时报错，不做 CPU fallback）。`inspect --plan` 只做静态校验，不 import/probe `torch_npu`；设备可用性在 boot 阶段检查。
7. `device-npu`、`kernel-npu`、`profiler-npu` 不得在模块顶层 `import torch_npu`；延迟到 `setup` 或方法内，保证无 NPU 的机器也能执行 `dump-config` 与 `inspect --plan`。
8. model 与 tokenizer 词表一致。
9. phase 引用合法；pinned seam 不允许重绑。
10. 依赖图无环。
11. 所有 required 插件可实例化；optional 插件失败只记录。

### 7.3 静态计划与运行计划

**静态计划（`inspect --plan`）**：

1. 解析组合树，确认 provider 与 seam 绑定。
2. 执行 §7.2 的 1–5、9–10；6–8、11 属于运行计划（设备探测、NPU 导入规则、词表一致性、插件实例化）。
3. 计算每个 phase 的服务绑定、hook 顺序、pinned/rebindable 集合。
4. 输出 §7.5 的结构；不 import `torch_npu`、不加载模型、不实例化重型服务。

**运行计划（`run` / `infer`）**：

1. 实例化 `device`、`precision`、`collective`。
2. 构建 model（model 插件），得到 `ModelHandle` 与 `parallel_plan`。
3. 应用 `fsdp` 插件（v1 为 `fsdp-none`，no-op）。
4. 应用 `ep` 插件（v1 为 `ep-none`，全部专家本地）。
5. 构建 optimizer / scheduler（基于切分后的 model）。
6. 构建 data iterator。
7. 调用 `trainer.compile(services)` 得到 `StepPlan`。
8. 排序 step policy（priority 升序），绑定到 `before_step` / `after_backward` / `after_step` / `on_error`。
9. 生成每个 phase 的 `PhasePlan`；rebindable 服务在 phase 入口构造、出口按 §10.3 交接。
10. 计算每个 phase 的 `composition_hash`。
11. 冻结 plan 结构：所有 plan dataclass `frozen=True`，不暴露可变容器；service 实例的生命周期仍由 phase 管理。

### 7.4 热路径保证

- `StepPlan` 只包含已绑定的 callable；每 step 不调用 `ctx.get`。
- policy 链在编译期排序；热路径只遍历 tuple。
- metrics 事件在 step 结束后聚合，不在 forward/backward 内做 I/O。
- profiler 只在 phase 或窗口内启停。

### 7.5 inspect --plan 输出

人类可读，至少包含：

```
Run: <run_id>
Recipe: <path>
Composition hash: <sha256>

Services:
  [pinned]      model        model-hf          source=bundles/hf.yaml
  [pinned]      optimizer    optimizer-adamw   source=recipe.patch
  [pinned]      trainer      trainer-sft       source=bundles/sft.yaml
  [rebindable]  metrics      metrics-jsonl     source=bundles/base.yaml
  [rebindable]  checkpoint   checkpoint-safetensors ...

Phases:
  train:
    services: metrics=metrics, profiler=profiler, checkpoint=checkpoint
    steps: before_step(2) -> forward_backward -> after_backward(1) -> optimizer_step -> after_step(1)
    hooks: grad-clip(10), nan-guard(20), grad-accum(30)
  eval: ...
  checkpoint: ...

Diagnostics:
  - none
```

---

## 8. CLI

### 8.1 命令

```bash
starweave init <dir>
starweave run <recipe> [--set k=v ...] [--patch f ...] [--phase train|eval|checkpoint] [--resume ckpt|latest] [--allow-config-change]
starweave infer <recipe> --prompt "..." [--max-new-tokens N] [--temperature T] [--top-p P]
starweave dump-config <recipe> [--json]
starweave dump-schema [plugin]
starweave inspect --plan <recipe>
starweave plugin list
starweave doctor
```

### 8.2 行为

- `init`：创建 `<dir>/recipe.yaml`、`<dir>/data/`、`<dir>/.gitignore`；不覆盖已有文件。
- `run`：compose → compile → 执行 phase 序列；默认执行 `train`，按 policy 触发 `eval` / `checkpoint`。
- `--resume latest`：在 `runs/*/checkpoints/latest` 中选择 mtime 最新的一个；`--resume <path>` 使用显式路径。
- `infer`：compose → compile `inference` 服务 → 加载模型/checkpoint → 生成；输出到 stdout。
- `dump-config`：打印最终组合树；每行带 `source`；`--json` 输出机器可读结构。
- `dump-schema`：不带参数时打印所有插件 schema 的索引；带参数打印单个插件 JSON Schema。
- `inspect --plan`：只编译不运行；打印 §7.5 的结构。
- `plugin list`：列出已安装 entry points 的名称、模块、seam、版本。
- `doctor`：检查 Python、torch、torch_npu、transformers、pydantic、NPU 数量、entry points、写权限；缺少 `torch_npu` 或 0 张 NPU 时明确失败，不提供 CPU fallback。

### 8.3 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | compose / 配置 / 校验错误 |
| 3 | boot 错误（backend 不可用、插件 setup 失败） |
| 4 | 运行期错误（step 失败、NaN abort、OOM） |
| 5 | 用户中断（SIGINT） |

---

## 9. 仓库结构

```
starweave/
├── pyproject.toml
├── README.md
├── docs/
│   ├── spec.md
│   ├── plan.md                      # Stage 2 产物
│   └── acceptance.md                # M3 产物
├── configs/
│   └── dsv4_small.json
├── src/starweave/
│   ├── __init__.py
│   ├── kernel/
│   │   ├── __init__.py
│   │   ├── context.py
│   │   ├── plugin.py
│   │   ├── registry.py
│   │   ├── scope.py
│   │   ├── events.py
│   │   └── errors.py
│   ├── seams/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── device.py
│   │   ├── collective.py
│   │   ├── fsdp.py
│   │   ├── ep.py
│   │   ├── precision.py
│   │   ├── trainer.py
│   │   ├── inference.py
│   │   ├── launcher.py
│   │   ├── kernel.py
│   │   ├── model.py
│   │   ├── data.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── loss.py
│   │   ├── checkpoint.py
│   │   ├── metrics.py
│   │   ├── profiler.py
│   │   └── policy.py
│   ├── compose/
│   │   ├── recipe.py
│   │   ├── patch.py
│   │   ├── bundle.py
│   │   ├── loader.py
│   │   └── hash.py
│   ├── plan/
│   │   ├── compiler.py
│   │   ├── validator.py
│   │   ├── plan.py
│   │   └── executor.py
│   ├── cli/
│   │   ├── main.py
│   │   ├── run.py
│   │   ├── infer.py
│   │   ├── dump.py
│   │   ├── inspect.py
│   │   ├── plugin.py
│   │   └── doctor.py
│   └── plugins/
│       ├── device_npu.py
│       ├── collective_none.py
│       ├── fsdp_none.py
│       ├── ep_none.py
│       ├── precision_fp32.py
│       ├── precision_bf16.py
│       ├── launcher_local.py
│       ├── kernel_torch.py
│       ├── kernel_npu.py
│       ├── model_hf.py
│       ├── data_jsonl.py
│       ├── data_hf.py
│       ├── optimizer_adamw.py
│       ├── scheduler_cosine.py
│       ├── scheduler_linear.py
│       ├── loss_ce.py
│       ├── checkpoint_safetensors.py
│       ├── checkpoint_torch.py
│       ├── metrics_jsonl.py
│       ├── metrics_noop.py
│       ├── profiler_off.py
│       ├── profiler_npu.py
│       ├── trainer_sft.py
│       ├── inference_hf.py
│       ├── policy_grad_clip.py
│       ├── policy_nan_guard.py
│       ├── policy_grad_accum.py
│       ├── checkpoint_policy_every_n.py
│       └── eval_policy_every_n.py
├── plugins/
│   └── model_dsv4/                  # 独立官方插件包 starweave-model-dsv4
│       ├── pyproject.toml
│       ├── README.md
│       ├── src/starweave_model_dsv4/
│       │   ├── __init__.py
│       │   ├── plugin.py
│       │   ├── config.py
│       │   └── _vendor/
│       │       ├── configuration_deepseek_v4_native.py
│       │       ├── modeling_deepseek_v4_native.py
│       │       ├── lm_head.py
│       │       └── args.py
│       └── tests/
├── bundles/
│   ├── base.yaml
│   ├── npu.yaml
│   ├── hf.yaml
│   ├── dsv4.yaml
│   ├── sft.yaml
│   └── inference.yaml
├── recipes/
│   └── sft_npu_dsv4_small.yaml
├── examples/
│   └── external_plugin/
└── tests/
    ├── kernel/
    ├── compose/
    ├── plan/
    ├── seams/
    ├── plugins/
    ├── e2e/
    ├── golden/
    ├── benchmarks/
    └── npu/
```

`plugins/model_dsv4/` 是独立发行包，不在 `src/starweave/` 内；主包不 import 它。`src/starweave/plugins/` 只放第一方、无额外依赖的插件。

每个 Python 目录包含 `__init__.py`；图中为简洁省略重复的 `__init__.py`。

### 9.1 pyproject.toml（关键字段）

```toml
[project]
name = "starweave"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "torch>=2.1",
  "transformers>=4.51",
  "datasets>=2.18",
  "safetensors>=0.4",
  "pydantic>=2.6",
  "PyYAML>=6",
  "typer>=0.12",
]

[project.optional-dependencies]
npu = ["torch_npu>=2.1"]
dsv4 = ["starweave-model-dsv4"]
dev = ["pytest>=8", "ruff>=0.4", "mypy>=1.8"]

[tool.uv.workspace]
members = ["plugins/model_dsv4"]

[tool.uv.sources]
starweave-model-dsv4 = { workspace = true }

[project.scripts]
starweave = "starweave.cli.main:app"

[project.entry-points."starweave.plugins"]
# 第一方插件逐项列出，例如：
adamw = "starweave.plugins.optimizer_adamw:AdamWPlugin"
sft = "starweave.plugins.trainer_sft:SFTTrainer"
```

说明：

- `torch_npu` 是 optional extra：无 NPU 的机器只跑 M0，不装；M1–M3 在 NPU 目标机用 `uv sync --extra npu`。
- `dsv4` extra 通过 uv workspace 安装 `plugins/model_dsv4`；不用 uv 时执行 `pip install -e plugins/model_dsv4`。
- `starweave-model-dsv4` 在自己的 `pyproject.toml` 注册 `dsv4 = "starweave_model_dsv4.plugin:Dsv4Model"`；主发行包不包含它。
- 第一方插件全部注册在 `starweave.plugins` 组。
- 开发环境优先用 `uv`；不可用时 `pip install -e ".[dev]"`。
- `ruff` 与 `mypy` 在 CI 中必须通过。

---

## 10. Phase 机制

### 10.1 Phase

v1 三个 phase：

- `train`：前向、反向、优化器步进、日志。
- `eval`：不更新参数；计算 eval loss / 生成样例。
- `checkpoint`：保存 checkpoint；可异步落盘。

`starweave infer` 额外编译一个 `inference` plan（不是训练 phase），只装载 inference 所需服务，不要求 trainer/optimizer/scheduler。

### 10.2 Pinned 与 rebindable

| 服务 | 分类 | 原因 |
|---|---|---|
| trainer | pinned | 持有训练循环状态 |
| optimizer | pinned | 持有动量、step |
| scheduler | pinned | 持有 step 计数 |
| collective | pinned | 单次运行的通信域 |
| fsdp | pinned | 参数切分策略 |
| ep | pinned | 专家切分策略 |
| model | pinned | 参数是训练状态 |
| data iterator | pinned | 位置是训练状态 |
| metrics | rebindable | 可换 sink，无训练状态 |
| profiler | rebindable | 可启停 |
| checkpoint | rebindable | 可换存储后端 |
| inference | rebindable | phase 专用，可释放显存 |

### 10.3 交接协议

rebindable 服务必须实现 §5.1 的 `Rebindable` 协议：

- `flush()`：把缓冲写出去。
- `close()`：结束资源。
- `release()`：释放显存/文件句柄。

phase 切换顺序：

```
旧服务.flush()
旧服务.close()
旧服务.release()
构造新服务
新服务就绪
```

pinned 服务不参与重绑；plan compiler 在编译期拒绝。

### 10.4 phase 选择

recipe 的 `phases` 按 row id 选择 provider：

```yaml
phases:
  train:      {metrics: metrics, profiler: profiler, checkpoint: checkpoint}
  eval:       {metrics: metrics, profiler: profiler}
  checkpoint: {checkpoint: checkpoint, metrics: metrics-noop}
```

被选中的 row 如果是 `disabled`，只为该 phase 实例化；退出时按交接协议释放。

### 10.5 run log

每个 phase 记录自己的 `composition_hash`；事件见 §11.3。

---

## 11. 可复现性

### 11.1 种子

- 启动时设置 Python `random`、`torch.manual_seed`、`torch.npu.manual_seed_all`（NPU 存在时）。
- DataLoader 使用 `torch.Generator`，种子来自 `ctx.seed`。
- 每个 phase 进入时记录当前 RNG 状态。

### 11.2 Checkpoint 内容

- `model_dir`：HF `save_pretrained` 格式（safetensors + config + tokenizer）。
- `optimizer`：`torch.save` 的 state dict。
- `scheduler`：state dict。
- `step`、`dataloader_position`、`rng`。
- `composition_hash`、`config_snapshot`。
- `latest` 指针：`checkpoints/latest` 文件或目录。

`resume` 行为：

1. 读取 checkpoint。
2. 校验 `composition_hash`；不一致时默认报错，`--allow-config-change` 可显式放宽（v1 只允许非结构字段变化）。
3. 恢复 model、optimizer、scheduler、step、data 位置、RNG。
4. 从下一步继续；loss 轨迹与基线在容差内一致。

### 11.3 run log 事件

JSONL，每行一个事件：

| 事件 | 字段 |
|---|---|
| `run/start` | run_id, recipe, composition_hash, config_snapshot |
| `phase/enter` | phase, composition_hash, step |
| `step/end` | step, loss, lr, tokens, elapsed_ms |
| `eval/end` | step, loss, metrics |
| `ckpt/save` | step, path, bytes, elapsed_ms |
| `run/end` | status, steps, elapsed_ms, best_metric |

metrics / profiler 订阅事件；不得在 forward/backward 内做 I/O。

---

## 12. 错误模型

### 12.1 compose 期

| 错误 | 行为 |
|---|---|
| 未知插件 / 不可导入 | `ComposeError`，带 patch 行路径、候选名称 |
| 重复 id | `ComposeError`，列出冲突行 |
| exclusive seam 多 provider | `ComposeError`，列出两个行 id |
| `requires` 缺失 | `ComposeError`，列出缺失 seam 与提供者 |
| config schema 失败 | `ComposeError`，带 pydantic 错误路径 |
| API 版本不兼容 | `ComposeError`，带插件范围与框架版本 |
| 非法变量 | `ComposeError`，带变量名与行路径 |

### 12.2 boot 期

- backend 不可用（缺少 `torch_npu` 或 0 张 NPU）：required 插件失败 → 逆序 teardown 已建服务，退出码 3；不做 CPU fallback。optional 插件失败 → 记录并继续。
- 插件 `setup` 异常：同上，错误带插件 id。
- teardown 失败：记录，继续卸载其余服务。

### 12.3 运行期

- step 异常：调用 policy 的 `on_error`；`nan-guard` 可以 abort 或 skip；错误带 plugin id / phase / step。
- OOM：释放 rebindable 服务（如 inference / profiler），记录；仍失败则退出码 4。
- 用户中断：捕获 SIGINT，flush metrics，保存 checkpoint（可选），退出码 5。

---

## 13. 测试策略

| 层 | 内容 | 是否依赖 NPU |
|---|---|---|
| kernel | Context、Scope、事件、清理顺序 | 否 |
| compose | patch 语义、优先级、变量、composition_hash | 否 |
| plan | 校验规则、冻结、phase 重绑、pinned 拒绝 | 否 |
| seams | 每个 seam 的 contract test，参数化到实现 | 部分（device/kernel 需要 NPU） |
| e2e NPU tiny | tiny 模型 20 step、resume、infer | 是 |
| NPU | dsv4 小配置 smoke、doctor | 是 |
| golden | `dump-config` 快照 | 否 |
| overhead | 插件 plan vs 裸 PyTorch loop（dispatch 层） | 否 |

测试约束：

- 不在测试中下载模型；NPU e2e 使用本地随机初始化 tiny 模型。
- NPU 测试用 `pytest.mark.npu`，`torch_npu` 缺失时 skip。
- M1–M3 的 NPU 验收在用户提供的 NPU 目标机执行；无 NPU 的机器只跑 M0 静态/单元测试。
- 每个 seam 的 contract test 可被外部插件复用。
- `dump-config` 快照放在 `tests/golden/`，变更需显式更新。

---

## 14. 验收矩阵

环境准备：

```bash
# 无 NPU 机器（只跑 M0）
uv sync --extra dev --extra dsv4

# NPU 目标机（M1–M3）
uv sync --extra npu --extra dsv4
```

不用 uv 时：`pip install -e ".[dev]" -e plugins/model_dsv4`；NPU 目标机额外安装 `torch_npu`。

### M0 内核与组合

```bash
uv run pytest tests/kernel tests/compose tests/plan -q
uv run starweave dump-config recipes/sft_npu_dsv4_small.yaml
uv run starweave inspect --plan recipes/sft_npu_dsv4_small.yaml
```

通过标准：

- 单元测试全过。
- `dump-config` 确定性，显示每行来源。
- `inspect --plan` 显示 services/steps/pinned/rebindable/hooks。
- golden snapshot 通过。
- M0 不依赖 NPU；可在没有 `torch_npu` 的机器执行。可安装 CPU 版 torch/transformers 用于导入和单元测试，但不注册 CPU device backend。

### M1 NPU 端到端（dsv4 小配置）

```bash
uv run starweave doctor
uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=20
uv run starweave run recipes/sft_npu_dsv4_small.yaml --resume latest --set trainer.max_steps=40
uv run starweave infer recipes/sft_npu_dsv4_small.yaml --prompt "hello"
uv run pytest tests/seams tests/e2e -q
```

通过标准：

- doctor 检测到至少 1 张 NPU；无 NPU 时该里程碑不得标记完成。
- loss 下降，run log / checkpoint 生成。
- resume 的 step 与 loss 轨迹与基线一致（容差内）。
- infer 返回非空文本。
- 全部在 NPU 上执行，不做 CPU fallback。

### M2 dsv4 小配置训练与 checkpoint

```bash
uv run starweave doctor
uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=50
uv run pytest tests/npu -q
```

通过标准：

- doctor 检测到至少 1 张 NPU；缺失时明确失败，不进入 M2。
- dsv4 小配置 50 step SFT 完成，loss 下降，checkpoint 可存可载。
- `grep -R -n -E 'torch_npu|\.npu\(' src/starweave` 只命中 device_npu.py / kernel_npu.py / profiler_npu.py。
- dsv4 checkpoint / tokenizer 不可用时使用随机初始化小配置跑通流程，并在报告中标注未验证项。

### M3 phase、文档、验收

```bash
uv run starweave run recipes/sft_npu_dsv4_small.yaml --phase train
uv run starweave run recipes/sft_npu_dsv4_small.yaml --phase eval
uv run starweave dump-schema
uv run starweave plugin list
uv run pytest -q
```

通过标准：

- train → eval → checkpoint 切换；metrics/profiler 重绑生效。
- 重绑 pinned（trainer/optimizer）被编译期拒绝。
- 外部插件包 `pip install -e examples/external_plugin` 后出现在 `plugin list`，并能被 recipe 使用。
- README/docs 完整；插件层开销报告存在。
- 已知限制列出；未验证项不得写成已完成。

---

## 15. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 开发机没有 torch_npu | M1–M3 无法执行 | M0 在本机；M1–M3 在用户提供的 NPU 目标机执行；报告标注未验证 |
| dsv4 tokenizer/checkpoint 不可用 | M1/M2 无法加载 dsv4 | 支持本地 tokenizer 与随机初始化小配置；M1 用随机权重；报告中标注 |
| NPU 显存不足 | dsv4 小配置 bf16 + AdamW 状态可能超限 | 缩小层数/专家数/序列长度；梯度累积、小 micro batch；可重绑 profiler 释放显存；v1 不做 ZeRO |
| dsv4 vendored 代码上游漂移 | 与 MindSpeed-LLM 版本不一致 | vendoring 时记录源 commit；插件自带 contract/smoke test；升级时先跑对齐测试 |
| FSDP2/EP v2 后端未实现 | 真实 284B 无法在 v1 运行 | v1 明确只做 `fsdp-none`/`ep-none`；v2 插件接口已固定，后续替换不污染核心 |
| 插件层开销 | 热路径变慢 | 静态 StepPlan；每 step 零查找；M3 测开销 |
| 组合复杂导致难调试 | 配置错误定位慢 | dump-config 带来源；compose 错误带行路径；golden snapshot |
| API 漂移 | 外部插件失配 | `starweave` API 范围（`[tool.starweave].api` 或依赖范围）；compose 校验；contract tests |
| 单发行包与 seam 解耦冲突 | 插件依赖整个包 | imports 边界 + 校验器；v2 可拆 `starweave-seams` |

---

## 16. v1 已知限制

- 单进程、单设备；v1 不实现真实多机多卡/TP/PP/FSDP2/EP/HCCL，只提供 `fsdp-none`/`ep-none`/`collective-none` no-op 插件。
- v1 只验证 dsv4 小配置；真实 284B DeepSeek-V4 与 FSDP2/EP/CP 属于 v2。
- dsv4 小配置的 `mlp_layer_types` 全部是 `moe`；`hash_moe` 在 v2 接真实 checkpoint 时启用。
- 不提供 CPU device backend；无 NPU 时只能跑静态检查与单元测试，不能跑 E2E 训练/推理。
- 不实现 RL、GPU backend、运行时插件装卸/HMR、Web UI、插件市场。
- 推理使用 HF `generate`，不是高吞吐 serving engine。
- 不支持 PEFT/LoRA、量化、通信重叠、MindSpeed 融合算子；vendored dsv4 代码在缺少 MindSpeed 时回退到 torch 原生算子。
- checkpoint 的 optimizer/scheduler 使用 `torch.save`，跨版本兼容性由 torch 决定。
- `data-hf` 只支持本地/已缓存数据集；不在线下载。
- `starweave.seams` 在 v1 是单发行包内的导入边界，不是独立发行包。

---

## 17. 自检

### 17.1 硬约束覆盖

| 约束 | 设计位置 | 验收位置 |
|---|---|---|
| C1 混合内核 | §3、§4.4、§7 | M0 `inspect --plan`、M3 overhead |
| C2 硬件即插件 | §5.2 device、§5.3 kernel、§9 | M2 grep、M2 NPU 验收 |
| C3 HF 优先 / dsv4 native | §5.3 model/data/checkpoint、§5.5.1 model-dsv4 | M1/M2 |
| S4 并行插件 | §2 S4、§5.2 collective/fsdp/ep、§6 base | M0 inspect、M3 |
| N8 真实 284B dsv4 | §1.2 N8、§16 | v2，不在 v1 验收 |
| C4 组合层 | §6 | M0 compose tests、M3 |
| C5 dsh 原则 | §2、§4、§5、§8 | 全部 |
| C6 最小 phase | §10 | M3 |
| C7 全新仓库/依赖 | §9、pyproject | M0 依赖检查 |
| C8 单发行包/entry points | §2.1、§4.7、§9 | M3 外部插件 |
| C9 语言 | 全文 | 评审 |
| 非目标 | §1.2 | 每个 milestone 检查 |

### 17.2 一致性检查

- 没有 TBD / TODO / 待定 / “后续补”。
- 每个 seam 有 Protocol 和配置字段。
- 每个验收项有可执行命令。
- 单发行包与 seam 导入边界的决策在 §2.1 明确，不再有第二种解释。
- phase 的 rebindable 列表与 §10.2 一致。
- CLI 命令与 §8 一致；验收矩阵使用同一组命令。
- 所有配置字段使用 snake_case；代码标识符英文；文档中文。

### 17.3 评审确认项（已给出决策）

1. **打包**：§2.1 采用单发行包 `starweave` + `starweave.seams` 导入边界。若你要求 v1 就拆成独立 `starweave-seams` 发行包，请在评审时指出。
2. **范围**：§1.2 非目标包含 PEFT/LoRA。若 v1 需要 PEFT/LoRA，请指出并调整 scope；否则按非目标执行。
3. **NPU 验收**：M1–M3 在用户提供的 NPU 目标机执行。当前开发机没有 NPU，只跑 M0 静态/单元测试；M1–M3 未执行前不得写成完成。
4. **并行**：v1 只实现 `collective-none` / `fsdp-none` / `ep-none`；真实 FSDP2 / EP / HCCL 是 v2 插件。若你要求 v1 就实现真实并行后端，请指出——这与单卡 dsv4 小配置 MVP 冲突，需要扩大验收环境。
