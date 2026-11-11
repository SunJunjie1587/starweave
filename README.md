# Starweave

硬件无关、全插件化的训练与推理框架。v1 目标：**单卡 Ascend NPU + DeepSeek-V4
小配置的 SFT 与推理**。设备、模型、数据、优化器、调度器、损失、checkpoint、
metrics、profiler、并行策略（FSDP/EP/collective）全部是插件；组合由 recipe +
bundle + patch 声明，运行时编译成静态 `StepPlan`。

- 规格：`docs/spec.md`
- 实现计划：`docs/plan.md`
- 验收记录：`docs/acceptance.md`
- 插件作者契约：`.claude/skills/plugin-framework-authoring/`

## 架构一览

```text
recipe + bundles + patch
        │
        ▼
   compose（按 id 插入/覆盖/禁用，记录每行来源，计算 composition_hash）
        │
        ▼
   static ExecutionPlan（seam 绑定、pinned/rebindable、phase、hooks）
        │
        ▼
   runtime（实例化插件、构建模型、编译 StepPlan、执行 phase）
        │
        ├── pinned: trainer / optimizer / scheduler / model / collective / fsdp / ep
        └── rebindable: metrics / profiler / checkpoint / inference
```

- 插件只允许 import `starweave.seams`，禁止 import kernel/compose/plan/cli/其他插件。
- 硬件调用只出现在 device/kernel 插件内部；seams 不依赖 torch。
- 热循环不查表：`SFTTrainer.compile()` 把 policy 和 step 函数绑定成 `StepPlan`。
- 每个 phase 有独立的 `composition_hash`，run log 记录 phase 进入/退出与 checkpoint。

## 快速开始（开发机，无 torch）

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]" -e plugins/first_party -e plugins/model_dsv4

python -m starweave.cli.main doctor          # 无 NPU 时退出码 3
python -m starweave.cli.main dump-config recipes/sft_npu_dsv4_small.yaml
python -m starweave.cli.main inspect --plan recipes/sft_npu_dsv4_small.yaml
python -m pytest -q                           # 无 torch 的静态测试
```

## NPU 目标机

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
pip install -e ".[npu,dsv4]" -e plugins/first_party -e plugins/model_dsv4

python -m starweave.cli.main doctor           # 期望看到 8 张 NPU，退出码 0
```

### 训练 / 断点续训

```bash
# 20 step，每 10 step 存一次
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --set trainer.max_steps=20 --set checkpoint-policy.n=10 --run-id npu-m1

# 从 step-000010 续训到 20 step（轨迹与不中断运行逐 step 对齐）
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --set trainer.max_steps=20 --set checkpoint-policy.n=10 \
  --resume step-000010 --run-id npu-m1
```

### 推理

```bash
python -m starweave.cli.main infer recipes/sft_npu_dsv4_small.yaml \
  --checkpoint runs/npu-m1/checkpoints/step-000010 \
  --prompt "hello world" --max-new-tokens 16 --run-id npu-infer
```

### phase

```bash
# eval：加载 checkpoint，跑 4 个 batch 的 no-grad eval
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --phase eval --resume step-000010 --run-id npu-m1 \
  --set trainer.max_steps=20 --set checkpoint-policy.n=10

# checkpoint：切到 checkpoint phase，按 phase 配置的 checkpoint 后端落盘
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --phase checkpoint --resume step-000010 --run-id npu-m1 \
  --set trainer.max_steps=20 --set checkpoint-policy.n=10
```

phase 切换时 metrics 从 `metrics-jsonl` 重绑到 `metrics-noop`（或反向），
旧服务按 `flush → close → release` 交接；pinned seam（model/optimizer/...）
若试图换 provider，编译期和运行期都会拒绝。

### RL（实验性）

```bash
python -m starweave.cli.main run recipes/rl_npu_dsv4_small.yaml \
  --phase rl --run-id rl-smoke
```

当前 RL 链路是 on-policy GRPO 最小实现：

- `task-rl` 负责循环；
- `rollout-hf` 只生成 token；
- `algorithm-grpo` 只算 group advantage 和 policy loss；
- loss 里的 logprob 由**训练模型重新前向计算**（scheme C），不复用 rollout 引擎的
  logprob，避免训推数值路径不一致；
- reward 由 `reward-rule` 提供，真实任务替换成 rule/reward-model 插件即可。

仓库内是合成 tokenizer，真实文本 reward 不可用，所以 smoke recipe 默认带一个
`token_reward` 玩具奖励；真实 assets 下应把它设为 0，使用 reference match 或
reward model。

### 外部插件

```bash
pip install -e examples/external_plugin
python -m starweave.cli.main plugin list | grep metrics-memory
python -m starweave.cli.main run recipes/sft_npu_dsv4_small_metrics_memory.yaml \
  --run-id npu-ext-metrics
```

## 仓库布局

```text
src/starweave/              # core：只有契约和组装器，不注册具体插件
  kernel/                   Context / Scope / EventBus / errors / registry
  seams/                    稳定插件 API：PluginBase、Protocol、config、types
  compose/                  bundle/recipe/patch、变量解析、hash、插件发现
  plan/                     静态 plan 编译、phase 校验、runtime、resume
  cli/                      dump-config / dump-schema / inspect / plugin / doctor / run / infer
plugins/first_party/        # 第一方插件，按能力分模块
  torch_base.py             launcher / fp32 precision / kernel
  backend_npu.py            device / bf16 precision / kernel / profiler
  parallel_none.py          collective / fsdp / ep no-op
  metrics_jsonl.py          metrics
  checkpoint_safetensors.py checkpoint
  model_hf.py               HF model
  inference_hf.py           HF inference
  data_jsonl.py             JSONL/HF data
  optim_torch.py            AdamW / scheduler / CE loss
  trainer_sft.py            SFT trainer
  policies.py               step/checkpoint/eval policies
plugins/model_dsv4/         DeepSeek-V4 官方模型插件（含 vendored 源码）
examples/external_plugin/   最小第三方 metrics 插件
bundles/ recipes/ configs/ data/
tests/{kernel,seams,compose,plan,plugins,cli,golden,e2e,npu,benchmarks}
```

## 测试

```bash
python -m pytest -q                          # 静态：69 passed, 3 skipped（无 torch）
python -m pytest -q tests/e2e tests/npu -m npu
python tests/benchmarks/overhead.py
python .claude/skills/plugin-framework-authoring/scripts/check_decoupling.py \
  examples/external_plugin --framework-package starweave
```

## 已知限制（v1）

- **只有 NPU 后端**，没有 CPU device backend；`doctor` 在无 NPU 时返回 3。
- `fsdp-none` / `ep-none` / `collective-none` 是 no-op；真实 FSDP2、EP、HCCL 是 v2。
- 真实 284B dsv4 多卡训练是 v2；v1 使用 `configs/dsv4_small.json`（4 层 / 8 experts /
  hidden 512 / vocab 129280）。
- 仓库内的 smoke tokenizer 是合成词表，decode 只得到 `<unk>`；真实 tokenizer 请放入
  `tokenizer/` 或用 `--set model.tokenizer_name_or_path=...` 指定。真实 dsv4
  checkpoint 同理通过 `--set model.checkpoint_path=...` 加载。
- transformers 5.2 的 `DynamicCache` 不认识 dsv4 的自定义 compressor cache layer，
  因此 v1 训练禁用 cache，`infer` 也以 `use_cache=false` 运行；KV cache 推理列入 v2。
- vendored dsv4 源码为兼容 transformers 5.2 做了最小 patch（`@dataclass`、
  `validate_rope(ignore_keys=...)`、`cache_position`、`sqrtsoftplus` 注册、
  训练时禁用 cache）；源 commit 记录在
  `plugins/model_dsv4/src/starweave_model_dsv4/_vendor/README.md`。
- 未实现 runtime 安装插件/HMR、Web UI、RL loop、MindSpeed 级融合算子。
