# RL 训推精度一致性与统一 Engine

## 1. 先分清两件事

“一个插件”不等于“一条数值路径”。

- 把训练和推理放进同一个 `engine` 插件，只解决**插件解耦**：task-rl 只依赖一个
  seam，插件内部可以共享代码/权重/生命周期。
- 精度一致取决于**计算 logprob 的代码路径**是否相同：相同的权重、dtype、
  attention/MoE kernel、并行切分、logits 处理、tokenizer。

如果同一个 engine 插件内部，训练走 Megatron、rollout 走 vLLM，那么它仍然是
两条数值路径，精度不一致依旧存在。**封装不能替代数值一致性。**

## 2. 训推不一致的来源

| 来源 | 例子 |
|---|---|
| dtype | 训练 bf16、rollout fp16/fp8；logits 是否 upcast 到 fp32 |
| attention kernel | 训练 FlashAttention/自定义 kernel，rollout PagedAttention |
| KV cache | 训练全序列 attention，rollout 分页 KV + 增量 decode |
| MoE | 训练 fused grouped GEMM，rollout 另一套 expert dispatch |
| 并行 | 训练 TP/EP/FSDP 的 all-reduce 顺序与 rollout TP 不同 |
| 权重 | 同步时 cast/量化/round，或 rollout 权重滞后一个版本 |
| logits 处理 | temperature/top-p 的实现、softmax 精度、logsumexp 方式 |
| tokenizer | chat template、特殊 token、截断策略不同 |
| CUDA graph | 图捕获后的 kernel 选择与 eager 路径不同 |

这些差异大多在 1e-3 到 1e-5 量级，但 RL 的 ratio/KL 会放大它们。

## 3. “一个 engine”能解决什么

能解决：

- 模型实现漂移：同一份 layer/MoE/attention 代码；
- 权重版本漂移：同一个权重对象/同一份 state；
- tokenizer/模板漂移；
- 生命周期和显存协调；
- 插件层的设备所有权冲突。

不能自动解决：

- 训练 forward 和推理 forward 仍然是不同代码路径（autograd/全序列 vs
  KV cache/continuous batching）；
- dtype/量化策略不同；
- TP/EP/FSDP 切分不同；
- CUDA/HCCL graph 与 kernel 选择不同；
- 权重同步的 cast/round。

所以：**一个 engine 是必要条件，不是充分条件。** 要根除问题，需要“同一条
logprob 路径 + 同一份数值契约”。

## 4. 三种架构

### A. 统一 engine（最强一致性）

一个插件同时提供训练和推理能力：

```python
class Engine(Protocol):
    def train_step(self, batch) -> TrainResult: ...
    def generate(self, requests, sampling_params) -> RolloutOutput: ...
    def logprobs(self, input_ids, attention_mask) -> Tensor: ...
```

- 训练和 rollout 共用同一份模型代码、权重、dtype、并行布局；
- task-rl 只依赖 `engine` seam，不 import 训练/推理实现，插件不耦合；
- 代价：engine 必须同时支持 autograd 和高效生成。vLLM 不支持 backward，
  PyTorch FSDP 没有高效 KV cache/continuous batching；真正可选的通常是
  Megatron-Core、自研 engine，或“训练 engine + 内部推理模式”。
- 性能通常要在一侧妥协；统一 engine 很难同时达到 vLLM 的 rollout 吞吐和
  FSDP/Megatron 的训练吞吐。

### B. 共享模型实现，两套执行后端

一个 model 插件提供同一份模型定义；训练和 rollout 是两个 engine 插件：

```text
model-dsv4（同一份 layer/MoE 代码）
  ├── train-engine-fsdp2
  └── rollout-engine-vllm
```

- 模型语义一致，但 kernel/并行/KV cache 仍可能不同；
- 需要额外的数值对齐和 logprob 校验；
- 保留独立替换训练/推理后端的能力。

### C. 分离 engine + 训练侧重算 logprob（最实用）

rollout 只负责生成 token；RL loss 使用的 logprob 由训练 engine 重新前向计算：

```text
rollout-vllm.generate() → tokens
train-engine.forward_logprobs(tokens) → logprobs 用于 PPO/GRPO loss
```

- 即使 rollout 用 vLLM、训练用 FSDP/Megatron，loss 中的 `π_θ(a|s)` 也来自
  训练路径；
- 避免直接使用推理引擎 logprob 带来的系统性偏差；
- 代价：多一次训练侧 forward；
- 严格来说，采样仍来自 rollout 策略，不是 100% on-policy；如果要求严格
  on-policy，就要用同一个 engine 采样和算 logprob，或者加重要性修正。

## 5. 对“根源规避”的准确回答

如果目标是“采样分布和训练策略完全一致”：

- 必须用**同一个 engine、同一份权重、同一 dtype、同一并行布局**完成采样和
  logprob 计算；
- 这才是严格 on-policy；
- 代价是放弃“训练/推理各自最优”的异构后端。

如果目标是“RL loss 不被推理引擎的数值误差污染”：

- 不必强求一个 engine；
- 让训练 engine 重算 logprob（方案 C）即可；
- 采样仍可能略微 off-policy，需要接受或用重要性采样修正。

所以：

> **一个 engine 可以从“采样策略”层面根除训推不一致，但前提是采样和 logprob
> 都走同一路径；如果 engine 内部仍是两套后端，就只是把不一致藏进了插件。**
>
> **如果只关心 loss 的数值一致性，训练侧重算 logprob 是更便宜、更现实的根因修复。**

## 6. Starweave 的插件设计

### 6.1 统一 engine 作为单插件

允许一个插件提供一个 `engine` seam，内部包含 train/generate/logprobs：

```python
class UnifiedEngine(PluginBase):
    name = "engine-unified"
    provides = ("engine",)
    requires = ("backend", "model")
```

task-rl 只依赖 `engine`，不 import FSDP/vLLM。插件内部可以复用任何库；内部
模块不是 Starweave 插件，因此不存在跨插件耦合。这符合“功能可以放插件内部，
但插件不可以耦合”。

### 6.2 分离 engine 作为两个插件

```text
train-engine-fsdp2   提供 engine/training
rollout-vllm         提供 rollout
```

task-rl 同时依赖二者，并通过 `forward_logprobs` 统一数值路径。两个插件互不
import，只通过 seam 通信。

### 6.3 共享数值契约

无论 A/B/C，都需要一个显式的 `NumericsConfig`（可以放在 task 配置或
precision 插件）：

```yaml
numerics:
  dtype: bfloat16
  logits_dtype: float32
  attention_backend: flash_attention
  moe_backend: fused
  tp: 2
  ep: 1
  deterministic: false
  rollout_quantization: none
```

训练和 rollout 都必须读取同一份配置；任何一侧偏离都要在启动时报错，而不是
静默运行。

## 7. 一致性验收

固定权重、固定 batch、关闭 dropout：

1. 训练 engine 前向 logits vs rollout engine 前向 logits：
   - max abs diff；
   - top-k token 一致率；
   - logprob 的 KL。
2. 对同一批 token，比较训练侧重算 logprob 与 rollout 返回 logprob：
   - 平均/最大偏差。
3. 跑一个 RL step，检查 ratio 分布是否集中在 1 附近；系统性偏离说明路径不一致。
4. 记录基线：dtype、kernel、TP/EP、权重版本、tokenizer 版本。

建议容差：

- logits max abs diff：bf16 同 kernel 约 1e-3 量级；
- 跨 kernel（FlashAttention vs PagedAttention）可能到 1e-2；
- 超过阈值就禁止上线，而不是“看起来能跑”。

## 8. 与 vLLM / verl 的关系

- vLLM 是推理引擎，没有 backward，不能作为统一 engine 的训练侧；
- verl 的 hybrid engine 是“共卡/共进程编排”，内部仍然是训练引擎 + vLLM
  两套后端，不等于统一数值路径；
- 想复用 vLLM 的吞吐，同时保证 loss 一致性，最现实的是方案 C：
  vLLM 生成 token，训练 engine 重算 logprob；
- 想严格 on-policy，必须接受统一 engine 的性能取舍，或者自己实现一个
  同时支持训练和高效生成的 engine。

## 9. 建议

1. **默认方案**：分离 `train-engine` 和 `rollout`，训练侧重算 logprob；
   这是性能与一致性的最佳折中。
2. **严格 on-policy / 精度优先**：提供 `engine-unified` 插件，采样和 logprob
   走同一路径；先支持小模型/单卡，再扩展并行。
3. **不要**把 vLLM 和训练 engine 塞进同一个插件就宣称“精度一致”；必须用
   §7 的验收证明。
4. 把 `NumericsConfig` 和一致性测试作为 RL 插件发布的门槛。

## 10. 当前实现状态（2026-11-12）

- `rollout-hf` 只负责生成 token。
- `task-rl` 在 rollout 后用训练模型重算 old logprobs（`no_grad`），再执行
  `update_epochs` 轮 PPO 更新；每轮重新前向并计算 ratio clip loss。
- `algorithm-grpo.policy_loss(logprobs, old_logprobs, advantages, mask)` 使用
  `ratio = exp(logprobs - old_logprobs)` 和 `clip_ratio` 的 PPO clipped objective。
- 新增 `rl_clip_fraction` metric，用于观察 ratio 超出 clip 区间的比例。
- NPU 验证：PPO 改动后远程 NPU 主机暂时不可达，尚未重跑；上一版 on-policy
  版本的 83 passed / 1 skipped 是本次改动前的基线。
