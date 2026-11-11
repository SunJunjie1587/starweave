# Starweave v1 验收记录

日期：2026-11-10
目标机：`<npu-host>`，Ascend910 ×8，`<repo-path>`

> 只记录实际执行过的命令与输出。未执行的项在“未验证”中明确标注，不写成完成。

## 1. 环境

| 项 | 值 |
|---|---|
| 开发机 | macOS，仓库 `/Users/sunjunjie/Documents/MSIS/starweave` |
| 开发机 Python | 预置 `.venv`，无 torch（静态测试） |
| NPU 机 Python | `<repo-path>/.venv/bin/python` 3.11.15 |
| torch / torch_npu | 2.7.1 / 2.7.1（系统包，`.venv` 使用 `--system-site-packages`） |
| transformers | 5.2.0（NPU 机隔离 venv） |
| CANN | 运行前必须 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 与 `source /usr/local/Ascend/nnal/atb/set_env.sh` |
| NPU 数量 | 8（`doctor` 输出 `npu_count: 8`，退出码 0） |

`uv` 在本机不可用且网络受限，按计划回退到 `pip -e --no-build-isolation --no-deps`
与预置 `.venv`。目标机使用同一方式安装 `starweave`、`starweave-model-dsv4`
和示例插件。

## 2. 结果总表

| 里程碑 | 结果 | 关键证据 |
|---|---|---|
| M0 内核/组合/静态 plan | 通过 | 本地静态 `68 passed, 3 skipped`；`dump-config` golden 通过；`dump-config`/`inspect` 确定性；插件 validator 0 error / 0 warning |
| M1 20 step + resume + infer | 通过 | loss 11.9781 → 2.8858；step 10 checkpoint；resume 后 loss 逐 step 对齐（~2e-7）；infer 产出 8 个 token |
| M2 50 step + checkpoint 往返 + NPU 测试 | 通过 | loss 12.4151 → 1.7301；step 25 checkpoint；resume step 26 loss `1.8192211389541626` 与原始逐位相同；NPU 测试通过；无硬件泄漏 |
| M3 phase / 外部插件 / 开销 / 文档 | 通过 | eval phase loss 1.8586；checkpoint phase 落盘 step-000026；pinned 重绑被拒；外部 metrics 插件安装/组合/运行/卸载验证；开销 2.82%–3.87%（模拟步） |

## 3. M0：内核、组合、静态 plan

```text
$ .venv/bin/python -m pytest -q          # 开发机
68 passed, 3 skipped in 0.52s

$ .venv/bin/python -m pytest -q          # NPU 机（含 e2e/npu）
70 passed, 1 skipped in 30.17s
```

跳过项：

- 开发机 3 项：`tests/e2e/test_sft_smoke.py`、`tests/npu/test_environment.py`、
  `tests/npu/test_phase_runtime.py`，原因均为 `No module named 'torch'`。
- NPU 机 1 项：`tests/cli/test_cli.py::test_doctor_without_npu`，因为 `torch_npu`
  已安装，该用例按设计跳过。

静态检查：

```bash
python .claude/skills/plugin-framework-authoring/scripts/check_decoupling.py \
  plugins/model_dsv4 --framework-package starweave --exclude _vendor
# 0 error(s), 0 warning(s)

python .claude/skills/plugin-framework-authoring/scripts/check_decoupling.py \
  examples/external_plugin --framework-package starweave
# 0 error(s), 0 warning(s)
```

硬件泄漏扫描：

```bash
grep -RnE "torch_npu|\.npu\(|device_type=[\"']npu" \
  src/starweave/seams src/starweave/kernel src/starweave/compose src/starweave/plan
# 无输出
```

说明：`src/starweave/cli/doctor.py` 直接 import `torch_npu` 是环境诊断命令，
不在训练/推理热路径；seams/kernel/compose/plan 无硬件依赖。

## 4. M1：20 step、resume、infer

命令：

```bash
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --set trainer.max_steps=20 --set checkpoint-policy.n=10 --run-id npu-m1
```

真实输出（`runs/npu-m1/run.jsonl`）：

| step | loss | lr | elapsed_ms |
|---|---|---|---|
| 0 | 11.97812557220459 | 9.9384e-05 | 341.59 |
| 1 | 9.136560440063477 | 9.7553e-05 | 86.37 |
| 5 | 4.660221576690674 | 7.9389e-05 | 85.18 |
| 10 | 3.2311532497406006 | 4.2178e-05 | 84.99 |
| 15 | 2.9014079570770264 | 9.5492e-06 | 78.16 |
| 19 | 2.8857860565185547 | 0.0 | 79.96 |

- checkpoint：`runs/npu-m1/checkpoints/step-000010`（model.safetensors 289 MB、
  optimizer.pt 577 MB、scheduler.pt、rng.pt、trainer_state.json）。
- 20 step 总耗时约 35.5 s（含模型构建与一次 checkpoint）。

resume：

```bash
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --set trainer.max_steps=20 --set checkpoint-policy.n=10 \
  --resume step-000010 --run-id npu-m1
```

- `phase/enter` 的 `step=11`，执行 step 11–19，共 9 步。
- step 11 loss `3.1165733337402344`，原始不中断运行 `3.1165735721588135`；
  其余 step 同样在 1e-6 以内逐 step 对齐。
- `apply_resume` 返回 `(checkpoint.step + 1, dataloader_position + 1)`；
  skip 循环只把真正消费的 batch 计入跳过数，`StopIteration` 不计。

infer：

```bash
python -m starweave.cli.main infer recipes/sft_npu_dsv4_small.yaml \
  --checkpoint runs/npu-smoke3/checkpoints/step-000002 \
  --prompt "hello world" --max-new-tokens 8 --run-id npu-infer
```

输出：`<unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>`（8 个 token，非空）。
仓库内是合成 tokenizer，真实文本解码需要用户提供 dsv4 tokenizer。

## 5. M2：50 step、checkpoint 往返、NPU 测试

```bash
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --set trainer.max_steps=50 --set checkpoint-policy.n=25 --run-id npu-m2
```

- loss：`12.415124893188477`（step 0）→ `1.730064034461975`（step 49）。
- 稳态约 76–88 ms/step；50 step 总耗时 56.4 s（含一次 checkpoint）。
- checkpoint：`runs/npu-m2/checkpoints/step-000025`。
- resume：

```bash
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --set trainer.max_steps=50 --set checkpoint-policy.n=25 \
  --resume step-000025 --run-id npu-m2
```

- 执行 step 26–49，共 24 步。
- step 26 loss `1.8192211389541626`，与原始 50 step 运行的
  `1.8192211389541626` 逐位相同；step 49 同样逐位相同。
- infer：

```bash
python -m starweave.cli.main infer recipes/sft_npu_dsv4_small.yaml \
  --checkpoint runs/npu-m2/checkpoints/step-000025 \
  --prompt "hello world" --max-new-tokens 16 --run-id npu-m2-infer
```

输出 16 个 `<unk>` token。

NPU 测试：

```bash
.venv/bin/python -m pytest -q tests/e2e tests/npu
# 3 passed（e2e smoke、doctor、phase runtime）
```

## 6. M3：phase、外部插件、开销

### 6.1 phase

eval：

```bash
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --phase eval --resume step-000025 --run-id npu-m2 \
  --set trainer.max_steps=50 --set checkpoint-policy.n=25
```

真实输出：

```text
phase/exit  train       composition_hash=3bec264f...
phase/enter eval        composition_hash=df15f76b...
eval/end    loss=1.8586088120937347 batches=4 step=26
run/end     status=ok phase=eval
```

checkpoint：

```bash
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --phase checkpoint --resume step-000025 --run-id npu-m2 \
  --set trainer.max_steps=50 --set checkpoint-policy.n=25
```

真实输出：

```text
phase/exit  train       composition_hash=3bec264f...
phase/enter checkpoint  composition_hash=edf6cbc9...
ckpt/save   step=26 path=runs/npu-m2/checkpoints/step-000026
run/end     status=ok phase=checkpoint
```

- `metrics` 从 `metrics-jsonl` 重绑到 `metrics-noop`；旧 metrics 按
  `flush → close → release` 交接，checkpoint phase 不写 metrics 行。
- pinned 重绑：`tests/plan/test_plan.py` 断言 `build_phase(..., {"model": "model"})`
  抛 `ComposeError`；`tests/npu/test_phase_runtime.py` 断言运行期
  `switch_phase` 对 pinned 变更抛 `BootError`。

### 6.2 外部插件

```bash
pip install -e examples/external_plugin --no-build-isolation --no-deps
python -m starweave.cli.main plugin list | grep metrics-memory
# metrics-memory  starweave_example_metrics.plugin:MemoryMetrics  starweave-example-metrics

python -m pytest -q examples/external_plugin/tests
# 2 passed

python -m starweave.cli.main run \
  recipes/sft_npu_dsv4_small_metrics_memory.yaml --run-id npu-ext-metrics
# loss: 12.020831108093262 → 9.010860443115234 → 7.385704040527344
# runs/npu-ext-metrics/metrics-memory.jsonl 写入 3 行
```

卸载验证：

```text
pip uninstall -y starweave-example-metrics
plugin list | grep -c metrics-memory        -> 0
dump-config recipes/sft_npu_dsv4_small.yaml -> 退出码 0
pip install -e examples/external_plugin ... -> 重新出现 metrics-memory
```

### 6.3 开销

```bash
.venv/bin/python tests/benchmarks/overhead.py
```

| 机器 | dispatch vs 裸调用 | trainer vs 裸循环 | 绝对开销 | 折算到 77 ms NPU step |
|---|---|---|---|---|
| 开发机 | 1.71% | 2.82% | 64.86 µs/step | 0.0842% |
| NPU 机 | 2.93% | 3.87% | 146.77 µs/step | 0.1906% |

模拟步的 CPU 计算约 2.3–3.8 ms/step；真实 NPU step 约 77 ms。两项均低于计划
的 5% 阈值。

## 7. 偏差与修复记录

1. **transformers 5.2 兼容**：vendored dsv4 源码做了最小 patch：
   - `DeepseekV4Config` 加 `@dataclass`，`strict` 变为 no-op；
   - `validate_rope(ignore_keys=...)` 兼容；
   - `create_sliding_window_causal_mask` 传入 `cache_position`；
   - `sqrtsoftplus` 以 `nn.Module` 类注册（5.2 的 `ACT2FN` 是
     `ClassInstantier`，索引时会实例化，不能注册普通函数）；
   - CausalLM forward 在 `labels is not None` 时强制 `use_cache=False`。
2. **训练 batch 设备**：SFT trainer 在 forward 前把 batch 张量移到
   `device.device_type`；此前 NPU embedding 收到 CPU indices 报错。
3. **resume 语义**：修正为 `step + 1`、`dataloader_position + 1`；skip 循环只统计
   真正消费的 batch，避免 `StopIteration` 被算作跳过。
4. **metrics 落盘**：`JSONLMetrics.emit` 每行 flush，并在 `teardown` 关闭文件；
   此前进程正常退出后 `metrics.jsonl` 仍为 0 字节。
5. **infer 可观测性**：推理结果同时返回 `token_ids`；`InferenceConfig` 增加
   `use_cache` 与 `skip_special_tokens`。合成 tokenizer 下 `skip_special_tokens=false`
   以显示 `<unk>`。
6. **eval/checkpoint phase 的服务集合**：当前 runtime 仍构建完整训练栈
   （trainer/optimizer/data 等），因此 phase 命令需要 train bundle 存在。规范允许
   eval/checkpoint 只装载最小服务集；该优化未在 v1 实现，列为 v2。

## 8. 未验证 / 已知限制

- **真实 dsv4 tokenizer 与真实 dsv4 checkpoint 未在目标机找到**。本轮使用合成
  tokenizer（vocab 129280）与随机初始化；`model.checkpoint_path` 的加载路径已实现
  但未用真实权重验证。
- **真实 284B dsv4 多卡训练、FSDP2、EP、HCCL 未实现**（v2）。v1 的
  `fsdp-none`/`ep-none`/`collective-none` 是 no-op 插件。
- **dsv4 KV cache 推理未验证**：transformers 5.2 的 `DynamicCache` 不构造 V4 自定义
  compressor cache layer，v1 以 `use_cache=false` 推理；列为 v2。
- **未运行 `uv sync`、`ruff`、`mypy`**：本机无 `uv` 且网络受限；静态测试、validator
  与运行时测试已通过，但 lint/type-check 未执行。
- 50 step 的 loss 下降来自 4 行样例数据，只证明链路可训练，不构成模型质量结论。
- 未实现 runtime 安装插件/HMR、Web UI、RL loop、多节点、MindSpeed 级融合算子。

## 9. 复现入口

```bash
# 静态
python -m pytest -q
python -m starweave.cli.main dump-config recipes/sft_npu_dsv4_small.yaml
python -m starweave.cli.main inspect --plan recipes/sft_npu_dsv4_small.yaml

# NPU
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
python -m starweave.cli.main doctor
python -m starweave.cli.main run recipes/sft_npu_dsv4_small.yaml \
  --set trainer.max_steps=50 --set checkpoint-policy.n=25 --run-id accept-m2
python -m starweave.cli.main infer recipes/sft_npu_dsv4_small.yaml \
  --checkpoint runs/accept-m2/checkpoints/step-000025 \
  --prompt "hello" --max-new-tokens 16 --run-id accept-infer
python -m pytest -q tests/e2e tests/npu
python tests/benchmarks/overhead.py
```

## 10. 2026-11-11 包边界重构回归

按“核心只保留契约和组装器、具体实现全部外置”的第一原则，第一方插件从
`src/starweave/plugins/` 迁出到独立发行包 `plugins/first_party/`：

- core `pyproject.toml` 删除全部 `starweave.plugins` entry points；
- `BUILTIN_PLUGINS` 回退删除，改为 entry points + 可选的开发 manifest
  （`STARWEAVE_PLUGIN_MANIFEST`，仅测试用）；
- `StepPlan` 从 `starweave.plan` 移入 `starweave.seams`，trainer 插件不再
  import plan；
- `plugins/first_party/` 按类别拆成 `torch_base`、`backend_npu`、
  `parallel_none`、`model_hf`、`inference_hf`、`data_jsonl`、`optim_torch`、
  `trainer_sft`、`policies`、`metrics_jsonl`、`checkpoint_safetensors`；
- 每个插件仍通过 `starweave.plugins` entry point 注册，插件名、recipe 和 CLI
  不变。

回归结果：

```text
本机:   69 passed, 3 skipped
NPU 机: 72 passed, 1 skipped
entry points: 31（29 first-party + dsv4 + metrics-memory）

3 step NPU smoke（run-id=pack-refactor-smoke）:
step 0 loss 11.984296798706055
step 1 loss 9.48235034942627
step 2 loss 7.852826118469238
checkpoint step-000001/step-000002 正常落盘
run/end status=ok
```

残留：`src/starweave/plugins/` 旧实现目录仍保留为备份，当前 entry points 和
manifest 均指向 `plugins/first_party/`，旧目录不被引用；待确认后删除。
