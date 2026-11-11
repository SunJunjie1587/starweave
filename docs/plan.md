# Starweave 实现计划

- 项目：Starweave
- 依据：`docs/spec.md` v1.1（2026-11-10）
- 计划版本：1.0
- 状态：M0–M3 已执行，验收证据见 `docs/acceptance.md`
- 执行方式：逐里程碑，TDD；每个里程碑结束停下等确认，不自动进入下一个

## 0.1 里程碑执行状态

| 里程碑 | 状态 | 证据 |
|---|---|---|
| M0 | 完成 | 静态测试、golden、`dump-config`/`inspect`、validator |
| M1 | 完成 | 20 step、resume、infer（`docs/acceptance.md` §4） |
| M2 | 完成 | 50 step、checkpoint 往返、NPU 测试、无硬件泄漏（§5） |
| M3 | 完成 | eval/checkpoint phase、外部插件、开销测量、验收报告（§6） |

偏差（eval/checkpoint 仍构建完整训练栈、真实 dsv4 tokenizer/checkpoint 缺失、
KV cache 推理延后）记录在 `docs/acceptance.md` §7–§8，不改变本计划的验收标准。

## 1. 目标与验收总览

v1 交付一个硬件无关、全插件化的训练&推理框架，首个真实负载是 **dsv4 小配置单卡 NPU SFT + 推理**。

| 里程碑 | 目标 | 是否需要 NPU | 出口标准 |
|---|---|---|---|
| M0 | 内核、seams、组合、plan、CLI 静态链路；dsv4 插件可被组合 | 否 | `pytest` 单测全过；`dump-config` / `inspect --plan` 确定性输出；golden 快照通过 |
| M1 | dsv4 小配置在单卡 NPU 上 20 step SFT、resume、infer | 是 | loss 下降；run log/checkpoint 生成；resume 轨迹一致；infer 非空 |
| M2 | dsv4 小配置 50 step 稳定训练与 checkpoint 往返 | 是 | 50 step 完成；checkpoint 可存可载；NPU 测试通过；无硬件泄漏 |
| M3 | phase 切换、外部插件、开销测量、文档与总验收 | 是 | train→eval→checkpoint 可切换；pinned 重绑被拒；外部插件可用；验收报告完整 |

## 2. 里程碑依赖图

```text
T0.1 仓库脚手架
  └─ T0.2 kernel
       └─ T0.3 seams
            ├─ T0.4 compose
            │    └─ T0.5 plan
            │         └─ T0.6 CLI
            ├─ T0.7 第一方静态/no-op 插件 + bundles/recipes/configs
            │    └─ T0.8 model-dsv4 插件
            │         └─ T0.9 golden/契约
            │              └─ T0.10 M0 验收
            └──────────────────────────────┘
T0.10 → T1.1 NPU device/kernel + doctor
          └─ T1.2 dsv4 训练链路（5→20 step）
               └─ T1.3 checkpoint/resume
                    └─ T1.4 infer
                         └─ T1.5 seam/e2e 测试
                              └─ T1.6 M1 验收
T1.6 → T2.1 50 step + checkpoint 往返
          └─ T2.2 NPU 测试 + 硬件泄漏检查
               └─ T2.3 M2 验收
T2.3 → T3.1 phase 机制
          └─ T3.2 外部插件示例
               └─ T3.3 开销测量
                    └─ T3.4 文档
                         └─ T3.5 总验收
```

可以并行的任务：

- T0.7 与 T0.4/T0.5/T0.6 在 T0.3 完成后可并行。
- T0.8 在 T0.3 + T0.7 的 recipe 骨架完成后可并行。
- T1.1 与 T1.2 的 dsv4 代码阅读/最小 build 可在 M0 后并行准备，但 T1.2 的 NPU 运行依赖 T1.1。

## 3. M0：内核与组合（无 NPU）

### T0.1 仓库脚手架

**产物**

- `pyproject.toml`：项目元信息、依赖、entry points、`[tool.uv.workspace]`、`[tool.uv.sources]`、pytest/ruff/mypy 配置。
- `README.md`（占位，M3 完善）。
- `src/starweave/{kernel,seams,compose,plan,cli,plugins}/__init__.py`。
- `plugins/model_dsv4/` 工作区成员占位。
- `tests/{kernel,compose,plan,seams,e2e,golden,npu}/`。
- `.gitignore`：`runs/`、`__pycache__/`、`.venv/`、`*.safetensors`（示例数据除外）。

**验收**

```bash
uv sync --extra dev --extra dsv4
uv run python -c "import starweave; print(starweave.__version__)"
uv run starweave --help
```

**回滚点**：脚手架不引入业务逻辑；若 uv workspace 解析失败，退回 `pip install -e ".[dev]" -e plugins/model_dsv4`，计划不变。

### T0.2 kernel

**产物**

- `kernel/context.py`：`Context`、`RunPaths`、`ctx.get`、`ctx.effect`、`ctx.emit/on`。
- `kernel/plugin.py`：`Plugin` Protocol、元数据校验。
- `kernel/registry.py`：seam 注册、exclusive/registry/policy 基数检查。
- `kernel/scope.py`：清理逆序、幂等 teardown。
- `kernel/events.py`：类型化事件总线。
- `kernel/errors.py`：`ComposeError` / `BootError` / `RunError`，带 plugin id / row path。

**测试**：`tests/kernel/test_context.py`、`test_scope.py`、`test_registry.py`、`test_events.py`。

**验收**

```bash
uv run pytest tests/kernel -q
```

**回滚点**：kernel API 在 M0 冻结；后续改动必须先在 `docs/spec.md` 记录并重新评审。

### T0.3 seams

**产物**

- `seams/types.py`：`ModelHandle`、`ParallelPlan`、`DataBundle`、`OptimizerHandle`、`SchedulerHandle`、`CheckpointState`、`StepState`、`StepEvent`、`RunState`、`MetricEvent`、`Rebindable`。
- `seams/device.py`、`precision.py`、`collective.py`、`fsdp.py`、`ep.py`、`trainer.py`、`inference.py`、`launcher.py`、`kernel.py`、`model.py`、`data.py`、`optimizer.py`、`scheduler.py`、`loss.py`、`checkpoint.py`、`metrics.py`、`profiler.py`、`policy.py`。
- 每个 seam 的 pydantic config 基类与 `Base*` 默认实现（policy/rebindable 的 no-op）。

**硬规则**

- `seams/` 不 import `torch`、`kernel`、`compose`、`plan`、`cli`。
- 所有 torch/HF 对象用 `Any` 或 opaque handle。

**测试**：`tests/seams/test_protocols.py`、`test_configs.py`；device/kernel 的 NPU contract 标记 `@pytest.mark.npu`。

**验收**

```bash
uv run pytest tests/seams -q -m "not npu"
grep -R -n -E "^import torch|^from torch" src/starweave/seams && exit 1 || true
```

**回滚点**：seam 签名一旦进入 T0.4 组合测试即冻结；变更需同步改 validator 与 contract tests。

### T0.4 compose

**产物**

- `compose/bundle.py`：bundle YAML 加载、路径锚定。
- `compose/patch.py`：`insert` / 按 id 覆盖 / `disabled` / `group`；config 整体替换；`inject`/`isolate` 拒绝。
- `compose/recipe.py`：recipe 解析、`bundles` 顺序、`patch`、`phases`。
- `compose/loader.py`：优先级合成、变量替换（`${env:VAR}`、`${run.id}`、`${paths.run_dir}`）、每行 `source` 记录。
- `compose/hash.py`：canonical JSON + 插件/框架版本 → `composition_hash`。

**测试**：`tests/compose/test_patch.py`、`test_precedence.py`、`test_variables.py`、`test_hash.py`。

**验收**

```bash
uv run pytest tests/compose -q
uv run starweave dump-config recipes/sft_npu_dsv4_small.yaml
```

**回滚点**：patch 语义以 spec §6 为准；若 group/disabled 组合出现歧义，先补测试再改实现，不引入第二套合并规则。

### T0.5 plan

**产物**

- `plan/validator.py`：§7.2 的 11 条校验；静态/运行分阶段。
- `plan/compiler.py`：静态计划与运行计划；phase 绑定；pinned/rebindable 集合。
- `plan/plan.py`：`ExecutionPlan` / `PhasePlan` / `StepPlan` / `ServiceBinding`，全部 frozen。
- `plan/executor.py`：运行计划执行、phase 交接、reverse teardown。
- `inspect --plan` 的人类可读输出。

**测试**：`tests/plan/test_validator.py`、`test_static_plan.py`、`test_phase_rebind.py`、`test_pinned_reject.py`。

**验收**

```bash
uv run pytest tests/plan -q
uv run starweave inspect --plan recipes/sft_npu_dsv4_small.yaml
```

**回滚点**：静态计划必须不 import/probe `torch_npu`；若 dsv4 插件在静态检查阶段触发 torch 导入，修插件延迟导入，不放松静态检查。

### T0.6 CLI 骨架

**产物**

- `cli/main.py`（typer app）、`run.py`、`infer.py`、`dump.py`、`inspect.py`、`plugin.py`、`doctor.py`。
- 退出码：0/2/3/4/5（spec §8.3）。
- `init` 脚手架；`doctor` 在无 NPU 时报告并退出 3，不崩溃。

**验收**

```bash
uv run starweave init /tmp/sw-init
uv run starweave plugin list
uv run starweave dump-schema
uv run starweave doctor; test $? -eq 3
```

**回滚点**：CLI 只做编排；业务逻辑留在 kernel/compose/plan/plugins。

### T0.7 第一方静态/no-op 插件 + bundles/recipes/configs

**产物**

- `plugins/device_npu.py`（延迟 import torch_npu）、`collective_none.py`、`fsdp_none.py`、`ep_none.py`、`precision_fp32.py`、`precision_bf16.py`、`launcher_local.py`、`kernel_torch.py`、`kernel_npu.py`（延迟 import）、`model_hf.py`、`data_jsonl.py`、`data_hf.py`、`optimizer_adamw.py`、`scheduler_cosine.py`、`scheduler_linear.py`、`loss_ce.py`、`checkpoint_safetensors.py`、`checkpoint_torch.py`、`metrics_jsonl.py`、`metrics_noop.py`、`profiler_off.py`、`profiler_npu.py`（延迟 import）、`trainer_sft.py`、`inference_hf.py`、`policy_*.py`、`checkpoint_policy_every_n.py`、`eval_policy_every_n.py`。
- `bundles/base.yaml`、`npu.yaml`、`hf.yaml`、`dsv4.yaml`、`sft.yaml`、`inference.yaml`。
- `recipes/sft_npu_dsv4_small.yaml`。
- `configs/dsv4_small.json`。

**测试**：每个插件的 config schema 测试；no-op 插件 contract 测试；`dump-config` 覆盖所有行。

**验收**

```bash
uv run pytest tests/plugins -q
uv run starweave dump-config recipes/sft_npu_dsv4_small.yaml
```

**回滚点**：若某个第一方插件拖慢 M0，可先把 `trainer-sft`/`inference-hf` 保留为最小实现，接口不变；不得删除 seam 或把逻辑写进 kernel。

### T0.8 model-dsv4 官方插件

**产物**

- `plugins/model_dsv4/pyproject.toml`：`starweave-model-dsv4`，entry point `dsv4 = "starweave_model_dsv4.plugin:Dsv4Model"`，依赖 `starweave`、torch、transformers、safetensors、pydantic。
- `src/starweave_model_dsv4/plugin.py`：顶层只 import seams/pydantic/stdlib；`build()` 内延迟 import。
- `src/starweave_model_dsv4/config.py`：`Dsv4ModelConfig`。
- `src/starweave_model_dsv4/_vendor/`：`configuration_deepseek_v4_native.py`、`modeling_deepseek_v4_native.py`、`lm_head.py`、`args.py`；保留上游版权头；记录源 commit。
- `tests/`：config、metadata、延迟导入、vendored 代码 smoke。

**验收**

```bash
python3 ~/.claude/skills/plugin-framework-authoring/scripts/check_decoupling.py plugins/model_dsv4 --framework-package starweave --exclude _vendor
uv run pytest plugins/model_dsv4/tests -q
uv run starweave plugin list | grep dsv4
uv run starweave dump-config recipes/sft_npu_dsv4_small.yaml
```

**回滚点**：若 `_vendor` 与上游 API 不兼容，先降级到 2 层/4 experts 的配置验证链路；若仍失败，停止并上报，不得偷偷换成 Qwen/HF 通用模型。

### T0.9 golden 与契约测试

**产物**

- `tests/golden/dump_config_sft_npu_dsv4_small.txt`。
- `composition_hash` 稳定性测试。
- `dump-config` 来源层断言（bundle/patch/override）。
- `dump-schema` 覆盖所有插件。

**验收**

```bash
uv run pytest tests/golden -q
uv run starweave dump-config recipes/sft_npu_dsv4_small.yaml > /tmp/dump1.txt
uv run starweave dump-config recipes/sft_npu_dsv4_small.yaml > /tmp/dump2.txt
diff /tmp/dump1.txt /tmp/dump2.txt
```

**回滚点**：golden 变更必须由用户确认；禁止在 CI 中自动更新快照。

### T0.10 M0 验收

```bash
uv sync --extra dev --extra dsv4
uv run pytest tests/kernel tests/compose tests/plan tests/golden -q
uv run starweave dump-config recipes/sft_npu_dsv4_small.yaml
uv run starweave inspect --plan recipes/sft_npu_dsv4_small.yaml
uv run starweave plugin list
uv run starweave doctor; test $? -eq 3
```

**通过标准**

- 全部单测通过。
- `dump-config` 确定性、带来源。
- `inspect --plan` 显示 services/steps/pinned/rebindable/hooks。
- `plugin list` 能看到 `dsv4`。
- `doctor` 在无 NPU 时明确失败，不崩溃。
- 不依赖 NPU / torch_npu。

**M0 结束：停下等确认，不进入 M1。**

## 4. M1：NPU 端到端（dsv4 小配置，20 step）

### M1 前置条件

- 目标机已安装 `torch_npu` + CANN，且至少 1 张 NPU 可见。
- dsv4 tokenizer 已放到仓库根 `tokenizer/`，或通过 `--set model.tokenizer_name_or_path=/path` 指定。
- 可选：dsv4 小配置 checkpoint 已放到本地；没有 checkpoint 时用随机初始化跑通链路。
- 已执行 `uv sync --extra npu --extra dsv4`。

### T1.1 NPU device / kernel / profiler + doctor

**产物**

- `device_npu.py`：`torch_npu` 延迟导入、`set_device`、`synchronize`、`memory_stats`、`empty_cache`。
- `kernel_npu.py`：注册可选 NPU 算子；无可用算子时回退 `kernel-torch`。
- `profiler_npu.py`：可选启停。
- `doctor`：检测 torch/torch_npu/NPU 数量；缺失时报错退出 3。

**验收**

```bash
uv run starweave doctor
uv run pytest tests/seams -q -m npu
```

**回滚点**：若某 NPU 算子不可用，`kernel-npu` 只注册可用算子；不得让上层感知。

### T1.2 dsv4 训练链路（5→20 step）

**产物**

- `model-dsv4` 在 NPU 上 `build` 小配置（随机初始化或本地 checkpoint）。
- `data-jsonl` 读取 `data/sft_dsv4.jsonl`；prompt mask、EOS。
- `optimizer-adamw`、`scheduler-cosine`、`loss-ce` 接入 `trainer-sft`。
- `trainer-sft.compile` 生成 `StepPlan`；5 step smoke 后 20 step。

**验收**

```bash
uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=5
uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=20
```

**通过标准**：5 step 不崩；20 step loss 下降；run log 生成 `step/end`。

**回滚点**：若显存不足，按顺序缩小：seq 1024 → layers 2 → experts 4；保留 CSA/HCA/MoE/mHC/Indexer 结构。

### T1.3 checkpoint / resume

**产物**

- `checkpoint-safetensors` 保存 model state、optimizer/scheduler、step、data position、RNG、composition_hash、config snapshot。
- `--resume latest` 解析 `runs/*/checkpoints/latest`。
- resume 校验 composition_hash；不一致时报错。

**验收**

```bash
uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=20
uv run starweave run recipes/sft_npu_dsv4_small.yaml --resume latest --set trainer.max_steps=40
```

**通过标准**：resume 从 20 继续到 40；loss 轨迹与不中断基线在容差内一致；checkpoint 可重复加载。

**回滚点**：若 resume 不稳定，先只恢复 model+optimizer+scheduler，再逐步加 data/RNG；不得把“不可精确 resume”写成已完成。

### T1.4 infer

**产物**

- `inference-hf` 调用 dsv4 `generate`；`starweave infer` 输出文本。
- `--prompt`、`--max-new-tokens`、`--temperature`、`--top-p`。

**验收**

```bash
uv run starweave infer recipes/sft_npu_dsv4_small.yaml --prompt "hello"
```

**通过标准**：返回非空文本；不崩；无 NPU 时明确失败。

**回滚点**：若 dsv4 `generate` 与小配置不兼容，先做 greedy 单条生成；不引入 serving engine。

### T1.5 seam contract 与 e2e 测试

**产物**

- `tests/seams/` 对 device/kernel/fsdp/ep/collective 的 NPU/单进程 contract。
- `tests/e2e/test_sft_smoke.py`：20 step + resume + infer。
- `pytest.mark.npu` 标记。

**验收**

```bash
uv run pytest tests/seams tests/e2e -q
```

**回滚点**：contract test 失败时修插件，不修改 seam 契约；契约变更需回到 spec 评审。

### T1.6 M1 验收

```bash
uv run starweave doctor
uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=20
uv run starweave run recipes/sft_npu_dsv4_small.yaml --resume latest --set trainer.max_steps=40
uv run starweave infer recipes/sft_npu_dsv4_small.yaml --prompt "hello"
uv run pytest tests/seams tests/e2e -q
```

**M1 结束：停下等确认，不进入 M2。**

## 5. M2：dsv4 小配置 50 step 与稳定性

### T2.1 50 step 训练 + checkpoint 往返

**产物**

- `sft_npu_dsv4_small.yaml` 50 step 稳定运行。
- checkpoint 保存/加载/继续训练。
- run log 完整。

**验收**

```bash
uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=50
uv run starweave run recipes/sft_npu_dsv4_small.yaml --resume latest --set trainer.max_steps=60
```

**通过标准**：50 step 完成；checkpoint 存/载成功；resume 到 60 正常。

### T2.2 NPU 测试与硬件泄漏检查

**产物**

- `tests/npu/` 的 dsv4 smoke、device/kernel contract。
- 硬件泄漏 grep 检查。

**验收**

```bash
uv run pytest tests/npu -q
grep -R -n -E 'torch_npu|\.npu\(' src/starweave | grep -v -E 'device_npu.py|kernel_npu.py|profiler_npu.py' && exit 1 || true
```

**通过标准**：NPU 测试全过；`src/starweave` 中硬件调用只在三个 NPU 插件文件。

### T2.3 M2 验收

```bash
uv run starweave doctor
uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=50
uv run pytest tests/npu -q
```

**M2 结束：停下等确认，不进入 M3。**

## 6. M3：phase、外部插件、文档、总验收

### T3.1 phase 机制

**产物**

- `PhasePlan` 的 train/eval/checkpoint 绑定。
- rebindable 服务：metrics/profiler/checkpoint/inference；交接 `flush/close/release`。
- pinned 服务：trainer/optimizer/scheduler/model/collective/fsdp/ep；重绑拒绝。
- run log 记录每个 phase 的 composition_hash。

**验收**

```bash
uv run starweave run recipes/sft_npu_dsv4_small.yaml --phase train --set trainer.max_steps=5
uv run starweave run recipes/sft_npu_dsv4_small.yaml --phase eval
uv run starweave inspect --plan recipes/sft_npu_dsv4_small.yaml
uv run pytest tests/plan -q
```

**通过标准**：phase 切换成功；metrics/profiler 重绑生效；pinned 重绑报错。

### T3.2 外部插件示例

**产物**

- `examples/external_plugin/`：一个最小外部插件包（例如 metrics sink 或 scheduler），entry point 为 `starweave.plugins`，只 import `starweave.seams`。
- README 说明安装与 recipe 使用。

**验收**

```bash
uv pip install -e examples/external_plugin
uv run starweave plugin list | grep <example>
uv run pytest examples/external_plugin/tests -q
```

**通过标准**：插件出现在 `plugin list`；可被 recipe 使用；删除该插件不影响其他插件。

### T3.3 开销测量

**产物**

- `tests/benchmarks/overhead.py`：插件 plan vs 裸 PyTorch loop（dispatch 层）。
- 报告写入 `docs/acceptance.md`。

**验收**

```bash
uv run python tests/benchmarks/overhead.py
```

**通过标准**：给出每 step 开销；若 > 5%，定位并优化（绑定 callable、避免每 step 查表）。

### T3.4 文档

**产物**

- `README.md`：安装、快速开始、recipe 示例、NPU 要求、已知限制。
- 每个插件 README 条目。
- `docs/acceptance.md`：验收矩阵结果。
- `docs/spec.md` 与 `docs/plan.md` 状态更新。

**验收**：`uv run mkdocs` 或 README 链接检查；人评审。

### T3.5 总验收

```bash
uv sync --extra npu --extra dsv4
uv run pytest -q
uv run starweave doctor
uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=50
uv run starweave infer recipes/sft_npu_dsv4_small.yaml --prompt "hello"
uv run starweave dump-schema
uv run starweave plugin list
```

**通过标准**

- 全部测试通过。
- dsv4 小配置 50 step + infer 成功。
- 外部插件示例可安装/使用。
- 验收报告包含证据、偏差、已知限制。
- 未执行项明确标注，不写成完成。

**M3 结束：输出最终验收报告。**

## 7. 验收命令汇总

| 里程碑 | 命令 |
|---|---|
| M0 | `uv run pytest tests/kernel tests/compose tests/plan tests/golden -q`；`uv run starweave dump-config recipes/sft_npu_dsv4_small.yaml`；`uv run starweave inspect --plan recipes/sft_npu_dsv4_small.yaml` |
| M1 | `uv run starweave doctor`；`uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=20`；`--resume latest`；`uv run starweave infer ...`；`uv run pytest tests/seams tests/e2e -q` |
| M2 | `uv run starweave run recipes/sft_npu_dsv4_small.yaml --set trainer.max_steps=50`；`uv run pytest tests/npu -q`；硬件泄漏 grep |
| M3 | phase 命令；`uv pip install -e examples/external_plugin`；`uv run python tests/benchmarks/overhead.py`；`uv run pytest -q` |

## 8. 风险与回滚

| 风险 | 触发信号 | 回滚/缓解 |
|---|---|---|
| dsv4 vendored 代码不兼容 | import/build 失败，config `__post_init__` 报错 | 记录源 commit；降到 2 层/4 experts；仍失败则停止上报，不换模型 |
| dsv4 tokenizer/checkpoint 缺失 | M1 启动失败 | 用户提供 tokenizer；无 checkpoint 时随机初始化；报告中标注 |
| NPU 显存不足 | OOM | 依次缩小 seq 2048→1024、layers 4→2、experts 8→4；保留 dsv4 结构 |
| NPU 不可用 | `doctor` 退出 3 | M1–M3 阻塞；不提供 CPU fallback；不标记完成 |
| resume 漂移 | loss 轨迹不一致 | 检查 data position/RNG/optimizer state；修好再进 M2 |
| 插件层开销 > 5% | benchmark 超标 | 检查每 step 是否查表；把 policy 链/绑定 callable 固化 |
| 并行插件被误用 | 有人把 fsdp2/ep-native 接进 v1 | v1 只注册 no-op；`fsdp2`/`ep-native` 不发布 |
| 范围膨胀 | 出现 284B/多卡/RL 需求 | 记录为 v2；当前里程碑不接 |

## 9. 变更控制

- `docs/spec.md` 是事实源；`docs/plan.md` 的变更必须与 spec 一致。
- 任何 seam 签名、组合语义、验收标准变化：先更新 spec，再改 plan，再改代码。
- 不新增 v1 非目标；需要时记录到 v2 backlog。
- 每个里程碑结束提交：代码 + tests + recipes + 报告；不把未验证项写成完成。

## 10. 自检

- [x] 每个任务有产物、依赖、验收命令、回滚点。
- [x] 依赖图无环；M0 不依赖 NPU；M1–M3 依赖用户提供的 NPU。
- [x] 每个 spec 硬约束有对应任务：C1→T0.5/T3.3；C2→T1.1/T2.2；C3→T0.8/T1.2；C4→T0.4；C5→T0.2/T0.3/T0.5；C6→T3.1；C7→T0.1；C8→T3.2；C9→T3.4。
- [x] dsv4 模型是插件（T0.8），fsdp/ep/collective 是插件（T0.7/T1.1）。
- [x] 没有 TBD / TODO / 待定 / “后续补”。
- [x] 每个验收命令可执行；`uv` 不可用时用 `pip` 替代。
