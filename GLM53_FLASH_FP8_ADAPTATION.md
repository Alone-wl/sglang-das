# GLM-5.3-Flash-FP8 HCU 适配调试日志

## 维护规则

本文档是本任务的交接入口。后续开发者必须遵守：

1. 开始工作前先读“当前状态”“Todo”“未解决问题”。
2. 每次提交必须同步更新本文档，至少更新 Todo、调试记录、验证结果、对应 commit 和未解决问题。
3. Todo 必须反映真实状态：开始处理时标记“进行中”，完成并验证后才能勾选；发现新问题立即追加。
4. 失败实验只保留复现条件、结论和证据；被证伪的猜测移入“已排除项”。
5. commit message 必须独立说明问题、根因、修改、验证和限制，做到 message 即文档。

## 原始需求与约束

- 目标：完成 `glm5.3-flash-fp8` 模型适配，使指定启动和评测流程可用。
- 服务器：`ssh nmz26`；容器：`dev-wl-glm2`。
- 启动脚本：`/home/work/glm/ifb.sh`。
- 代码阅读和修改只在本地 `/Users/wanglong/Code/sglang-das` 进行。修改后本地提交并推送，随后进入容器拉取最新代码。
- 非算子问题先参考官方基线 `/Users/wanglong/Code/sglang`。
- 算子缺失或不兼容先参考 DCU 适配 `/Users/wanglong/Code/sglang-model`，优先复用已有实现，禁止无必要地重写算子。
- 必须从根因修复，禁止拟合输出、屏蔽症状或针对探针 prompt 特判。
- 并行相关问题先回退到纯 TP 复现，再决定是否排查 EP、DeepEP 或 EAGLE。
- SSH 登录失败后立即停止并交给用户处理，禁止重试。ProxyJump 使用 2FA，连续失败可能触发封禁。
- 先运行原始 `ifb.sh`；成功后用 `uv` 创建虚拟环境、安装 EvalScope，并运行 GSM8K 和 MATH-500。流程异常时停止并报告，不私自更换流程。
- 长 prompt 崩溃先关闭 HiCache 验证。关闭后仍失败则继续修复；若消失，记录 HiCache 问题并在关闭状态下继续。

## 当前状态

- 本地分支：`glm5.3-flash`；推送目标：`wl/glm5.3-flash`。
- 当前本地代码提交：`a5f5c1f0dc`；其后只有本文档的状态更新。
- 远端代码：`/home/work/code/sglang-das`；最后确认的远端 commit：`191fc0cc91`，已落后于本地文档提交。
- 模型：`/home/work/GLM-5.3-Flash-Channel-FP8-w8a8`。
- 硬件：8 张 HCU，`gfx938`。
- 主配置：TP=8、EP=8、DeepEP normal、DSA/NSA、FP8-E4M3 KV cache、Mamba、EAGLE。
- 当前临时关闭 HiCache，使用 `/home/work/glm/ifb_nohicache.sh`。这是规避 Mamba backup VMFault 的临时措施，不是最终配置。
- 短 prompt 已能生成连贯文本；短输出乱码的根因已修复。
- Torch fallback 已使纯 TP 冷 prefill 通过 2566/4107 token，但约 8200 token 因复制宽 page table 申请 32.06 GiB 而 OOM；本地已接入现有 AITER top-k，等待 HCU 端到端复测。
- 当前评测阻塞：GSM8K 单请求解码过慢并触发 EvalScope 超时；没有得到有效评分，不能据此判断模型精度。

## Todo（每次提交必须更新）

| 状态 | 优先级 | 事项 | 完成标准 |
| --- | --- | --- | --- |
| 进行中 | M1/P0 | **纯 TP 部署精度正常** | TP=8、EP=1、无 DeepEP、无 EAGLE；长 prompt 稳定；GSM8K/MATH-500 与可信基线对齐；记录配置、分数、截断率和失败样例 |
| 进行中 | M1/P0 | 接入 HCU AITER kpool top-k | 冷/热 prefill 均通过 2561、4096、8192 token；索引语义与参考实现一致；无 OOM/VMFault |
| 待办 | P0 | 完成长 prompt 回归 | HiCache 关闭时按 11、81、641、1281、2561、4096、8192 token 分级验证并记录日志 |
| 待办 | P1 | 定位 GSM8K 极低吞吐 | 分别测 target-only/完整配置的 TTFT、decode tok/s、EAGLE 接受率，明确瓶颈 |
| 待办 | P1 | 完成 GSM8K smoke | 使用合理输出上限和超时先跑 5 条，再跑 20 条；记录截断率、失败样例和分数 |
| 待办 | P1 | 完成 MATH-500 smoke | GSM8K 稳定后执行并记录配置、分数和失败样例 |
| 待办 | P2 | 修复 HiCache Mamba backup VMFault | 开启 HiCache 后长 prompt 不再触发 `transfer_mamba_backup_kernel` VMFault |
| 待办 | P2 | 验证 DeepSeek-V4 norm 修复 | 启动对应模型确认不存在重复归一化 |
| 待办 | P2 | 修复 BF16 KV cache 路径 | `--kv-cache-dtype bfloat16` 不再触发 scaled index K cache 断言 |
| 待办 | P2 | 建立可信精度基线 | 与可信实现、相同 prompt 和采样参数对齐，不只检查文本可读性 |

## 已完成工作与提交

### 继承的适配提交

| Commit | 内容 |
| --- | --- |
| `979baf5a81` | 引入上游 GLM-5.3-Flash 支持 |
| `0258c9987e` | 恢复 HCU DSA index-cache 兼容 |
| `cfcfc6d93d` | 恢复 GLM FP8 no-RoPE 量化辅助逻辑 |
| `f5fd9d01a3` | 移植 GLM DSA K-pool FP8 index 路径 |
| `1d85638859` | 恢复 DSA top-k backend 选择 |
| `f076f02e3b` | HCU 下允许 KDA fused operator 缺失 |
| `8bdf7af339` | 补齐 GLM MTP linear-attention 兼容 |
| `2b74702266` | 统一 Mamba HiCache 布局 |
| `83cc033742` | 增加 SWA branching match 字段 |
| `3514a56c13` | 设置 DSA K-pool 默认 page size |
| `69bfa861a1` | HCU DSA K-pool 支持 page size 1 |
| `1d79fe4681` | HCU 下重映射 CUDA-only FlashMLA backend |

本轮开始时另有 20 个 tracked 文件组成未提交 checkpoint，共 963 行新增、271 行删除；内容覆盖 DSA K-pool/FP8 index、DSA backend、KDA/Mamba/mHC、DeepEP channel-FP8 MoE、scheduler KV/Mamba bookkeeping、logits chunking 和 HiCache allocation。该 checkpoint 在本地与远端均存在，随后由 `2bc63d475f` 保存。

### 本轮提交

| Commit | 内容 | 验证 |
| --- | --- | --- |
| `2bc63d475f` | 保存 20 个文件的 HCU 兼容 checkpoint | `compileall`、`git diff --check`、8 卡服务 `/health` 通过；生成文本错误 |
| `7bedab73d1` | 记录 target-only 首次 prefill 卡死 | 定位到 KDA/DeepEP 等待，随后确认是残留共享内存导致 |
| `3297fc7235` | 记录清理 `/dev/shm` 后 prefill 恢复 | 三组固定 prompt 可重复生成 |
| `a3249b3ccf` | 记录已排除的数值假设 | 多组路径对比，见“已排除项” |
| `e164539dec` | 修复 CUDA graph batch-size helper 的过期参数 | 启动路径验证 |
| `7c68b1bab4` | 适配 schedule batch 重命名后的 CUDA graph 字段 | 启动路径验证 |
| `7d6c10fea3` | 修复 HCU AITER TileLang mHC pre 跳过 layernorm | 单算子及端到端生成验证 |
| `de651cb5e6` | 删除 DeepSeek-V4 已冗余的 HCU `norm_fused` workaround | GLM 已验证；DeepSeek-V4 尚未验证 |
| `0e0835a41c` | 记录 mHC 根因和验证结果 | 文档提交 |
| `e256489848` | 删除 HiCache-off 路径中不存在的 memory 配置字段 | HiCache-off 服务可继续启动 |
| `fe3abe5842` | HCU ragged MQA logits 改用 LightOp | 对 Torch 参考最大绝对误差 `3.8e-06` |
| `191fc0cc91` | 记录四个长 prompt 故障和当前阻塞 | 文档提交 |
| `782f6c8577` | 修正 HiCache-off 和 EvalScope 结论 | 文档提交 |
| `7a03aa8469` | 将调试日志重构为精简中文交接文档 | `git diff --check`；未修改代码 |
| `a5f5c1f0dc` | 缺少融合模块时复用 Torch pooled-history top-k | CPU 语义测试、`compileall`、`git diff --check` 通过；待 HCU 验证 |

## 调试记录

### 1. 初始现象：服务可用，但生成乱码

2026-09-12，8 卡服务启动后首次请求需要约 3 分 47 秒编译和初始化，之后 `/health` 返回 HTTP 200。固定 greedy 请求均完成，但输出为确定性乱码，EAGLE 接受率为 0。

随后构造 target-only 启动脚本，去掉四个 `--speculative-*` 参数。首次 prefill 曾分别卡在 `causal_conv1d_fn` 和 DeepEP `Buffer.__init__` 的 `all_gather_object`。根因不是模型数值或集合通信实现，而是此前异常退出遗留的共享内存和信号量。

清理以下文件后，target-only 服务恢复：

```bash
rm -f /dev/shm/sglang_loads_*.shm \
      /dev/shm/sgl_shm_mm_* \
      /dev/shm/sgl_shm_mq_* \
      /dev/shm/sem.mp-* \
      /dev/shm/torch_*
```

后续每次重启服务前都要清理上述残留，但执行前必须确认没有其他任务正在使用这些对象。

### 2. 纯 TP 复现：排除 EP、DeepEP 和 EAGLE

2026-09-13，使用：

```text
--tp-size 8 --ep-size 1 --moe-a2a-backend none
```

纯 TP 仍产生同类确定性乱码，因此问题位于单 rank 张量路径，与 EP、DeepEP、MoE dispatch 和 EAGLE 无关。

### 3. 乱码根因：mHC 错误报告已融合归一化

真实 GLM 形状下调用 `hc_pre(out_norm_weight=w)`：

```text
norm_fused=True
max|layer_input - RMSNorm reference| = 2.1094
max|layer_input - unnormalized reference| = 0.003906
```

根因链路：

1. `ifb.sh` 设置 `SGLANG_ROCM_USE_AITER_TILELANG_MHC=1`。
2. `mhc_pre()` 进入 AITER `pre_big_fuse_tilelang`。
3. 该算子没有 `norm_weight`/`norm_eps` 参数，实际未执行归一化。
4. `_mhc_pre_dispatch()` 却返回 `norm_fused=True`。
5. `MHCState.attn_split()` 和 `attn_to_mlp()` 因此跳过显式 layernorm，45 层的 `input_layernorm` 和 `post_attention_layernorm` 全部失效。

修复：

- `7d6c10fea3`：存在 `norm_weight` 时不走不带归一化的 AITER 分支，并恢复 mHC post opt-out。
- `de651cb5e6`：删除 DeepSeek-V4 已冗余的调用侧 workaround，防止双重归一化。

验证：

| 检查 | 修复前 | 修复后 |
| --- | --- | --- |
| S=8/64/512，`max|output-reference RMSNorm|` | 2.11 / 2.62 / 3.13 | 0.0156 / 0.0313 / 0.0313 |
| 纯 TP greedy 输出 | 确定性乱码 | 连贯 |
| TP8/EP8 + DeepEP + FP8 KV | 确定性乱码 | 连贯 |
| 完整配置 + EAGLE | 确定性乱码 | 连贯 |
| EAGLE 接受率 | 0 | 0.176 / 0.194；后续观测 0.23-0.28 |
| 20 条文本 sanity | 未通过 | 15/20；不能代替精度评测 |

代表性结果：

```text
Prompt: The capital of France is
修复前: mosself [ eitherNAS4~\n\nuur†/brrardia;Relationicles
修复后: Paris. The official language is French. Currency: Euro (€)...
```

### 4. 长 prompt 故障链

#### 4.1 HiCache Mamba backup VMFault：已规避，未修复

开启 HiCache 时，81-token prefill 曾在所有设备触发：

```text
KERNEL VMFault, Invalid address access, Error code: 3
transfer_mamba_backup_kernel
scheduler_1 exit code -6
```

关闭 HiCache 后，同一分级测试中 VMFault 数为 0，1281 token 可通过，因此当前使用 `ifb_nohicache.sh`。底层问题仍在 `transfer_mamba_backup_kernel` 路径。

早期测试曾记录“HiCache 开启时 1281 token 可通过、2561 token 失败”，与后续干净复核的 81-token VMFault 冲突。当前以可重复的后续复核为准；该阈值可能受缓存和请求状态影响，修复 HiCache 时必须重新测冷、热请求，不能把 81 当作固定边界。

#### 4.2 HiCache-off 读取不存在的配置：已修复

关闭 HiCache 后，`_should_elide_dsa_index_k` 读取不存在的 `memory_config.enable_unified_cache_external_linker`。该字段仅在 GLM 导入代码中出现，官方基线没有。`e256489848` 恢复官方判断条件。

#### 4.3 HCU ragged MQA logits 调用未导入的 `deep_gemm`：已修复

长 prefill 进入 `_get_topk_ragged_kpool_plan` 后，无条件调用只在 CUDA 下导入的 `deep_gemm.fp8_mqa_logits`，HCU 因此报 `NameError`。

`fe3abe5842` 将 HCU 分支改为已有的 `lightop_attention.mqa_logits`。独立算子测试对 Torch 参考的最大绝对误差为 `3.8e-06`。

`clean_logit=True` 的语义是：

```text
sum_h weight * relu(q·k) * k_scale
```

它不负责把行区间外填成 `-inf`；区间屏蔽由后续 top-k transform 根据 `ks`/`ke` 完成。

#### 4.4 缺少 HCU `kpool_topk_transform`：本地已修复，待 HCU 验证

修复前述问题后，约 2561-token prefill 报：

```text
ModuleNotFoundError:
sglang.kernels.ops.moe.kpool_topk_transform
```

调用链：

```text
_get_topk_ragged_kpool_plan
  -> topk_from_pooled_history_logits
  -> kpool_topk_transform
```

上游 GLM 提交包含 Python wrapper 和 CUDA `.cuh`，但本分支未带入。不能直接恢复 `.cuh`：其中使用 CUDA C++、`cuda_fp16.h` 和 SM90 radix top-k，无法在 `gfx938` JIT 编译。

当前 Python fallback 在 `row_starts` 或 `page_table_row_index` 非空时重新抛异常，而 ragged prefill 正好传入 `row_starts=ks_per_q`。

本地已从 `/Users/wanglong/Code/sglang-model/python/sglang/srt/layers/attention/nsa/kpool/kernels.py` 移植 `_torch_topk_pooled_history` 到 `dsa/kpool_fp8_index.py`。该实现处理 `row_starts`、`group_lengths`、pool expansion、page table、tail 和 `out_rows`；融合模块导入失败时进入该 fallback。异常捕获只包围导入，已加载算子的运行错误不会被掩盖。正确性通过后，再按 DCU 参考尝试 `aiter.kpool_topk` 或 LightOp `fast_kpool_topk_transform_fused`；Torch 版本只作保底。

必须测试：

- `row_starts` 为 0 和非零。
- 不同 `group_lengths`，`pool_size=4`，tail 长度 0/1/2/3。
- `page_table`、`page_table_row_index`、`topk_offsets` 和 `out_rows`。
- 无 pooled history、列数小于 top-k、冷 prefill 和 prefix-cache prefill。
- top-k 后将全局列号减去有效窗口起点，恢复局部 pooled-group 编号。

本地验证：

- 从实际源文件 AST 加载 fallback，CPU 语义测试通过。
- 覆盖非零 `row_starts`、page-table 行重映射、`topk_offsets`、`out_rows`、空 history、列数小于 group top-k 和 tail append。
- `compileall` 通过；本机没有 Triton，因此 HCU kernel helper 只能在服务器验证。
- `git diff --check` 通过。

#### 4.5 分级结果

HiCache 关闭且修复 ragged MQA logits 后：

| Prompt token | 结果 |
| ---: | --- |
| 11、41、81、161、321、641、1281 | 通过，约 3.0-3.8 秒 |
| 2561 | `kpool_topk_transform` ModuleNotFoundError；8 个 rank 各一次；无 VMFault |

641-token 记录中出现 `#cached-token: 640`，说明至少部分测试走了 prefix cache。修复后必须补充清空缓存的冷 prefill。

#### 4.6 Torch fallback 的 8K OOM：已定位，AITER 修复待验证

提交 `a5f5c1f0dc` 部署后，纯 TP、无 EAGLE、无 DeepEP、无 HiCache 的冷 prefill 结果：

| Prompt token | 缓存 | 结果 |
| ---: | ---: | --- |
| 16、86、646、1286、2566、4107 | 0 | HTTP 200；无 VMFault |
| 约 8200 | 0 | 所有 rank OOM，进程退出 137 |

OOM 位于 `_torch_topk_pooled_history` 的：

```python
page_table.index_select(0, page_table_row_index)
```

page table 宽度为 1M；约 8K 个 query 行会先复制成约 `8K x 1M x int32`，单 rank 申请 32.06 GiB。该实现语义正确，但不能用于长上下文。

容器中的 `aiter.kpool_topk` 和 LightOp `fast_kpool_topk_transform_fused` 均存在。独立 HCU 测试确认 AITER 支持 `row_starts`、`page_table_row_index`、`seq_lens` 和 tail；有效 group 数超过 128 时，其最终 token 集合与 Torch fallback 完全一致。AITER 会整理输出顺序，因此不能按 Torch `topk` 的分数顺序逐项比较。

本地已按 `sglang-model` 接入 `aiter.kpool_topk`：由 `SGLANG_NSA_KPOOL_AITER_TOPK=1` 控制，仅在 `index_kpool` 为 4/16、`index_topk=2048` 且存在 `seq_lens` 时启用；算子缺失时保留 Torch fallback。待重新提交并执行 8K 冷/热 prefill。

### 5. EvalScope：没有崩溃，实际是超时

GSM8K smoke 配置为 20 条、batch size 1、`max_tokens=2048`。运行 49 分钟仍为 0/20，EvalScope 多次报告请求超时，prediction 目录为空。

服务日志显示：

- `Scheduler hit an exception`：0。
- `Fatal Python error`：0。
- `KERNEL VMFault`：0。
- decode token 持续增加，并非死锁。
- 观测吞吐约 20-50 token/分钟；生成 2048 token 可能需要接近一小时。

结论：这次 GSM8K 没有产生可评分答案，不能算作崩溃，也不能判断正确性。下一次先定位 target-only 与完整配置的吞吐差异，再把 smoke 输出上限设为 256 或 512，并提高请求超时；同时记录 `finish_reason` 和截断率，确认没有系统性截断后再扩大样本。

EvalScope 1.11.1 已安装在 `/home/work/evalscope_uv/.venv`，uv 版本 0.12.13；GSM8K 1319 条、MATH-500 500 条均能加载。

## 已知不可用或受限路径

| 路径 | 结果 |
| --- | --- |
| `--kv-cache-dtype bfloat16` | DSA indexer 报 `Scaled index K cache is not enabled`；不能作为 FP8 KV 问题的绕行方案 |
| DSA `_forward_tilelang` | `libtilelang.so` 在 `GemmNode::InferLayout -> make_hcu_swizzled_layout` 崩溃；`D_V=512`、head 8/16 的 gfx938 路径不可用 |
| `SGLANG_USE_DEEPGEMM_MOE=0` | DeepEP dispatch 不支持当前 `CompressedTensorsW8A8Fp8MoE`，且 HCU 上 JIT DeepGEMM 关闭 |
| 启动参数关闭 mHC post TileLang | 环境变量会在后续 hook 被重新设为 true；如需关闭必须改代码 |

## 待复核的数值路径

以下项尚无独立证据证明有错，但在建立精度基线前不能删除：

- channel-FP8 权重经 `pack_int8_weight_enk_to_w6_low_latency` 后是否保持幅值；应对单个 expert 比较 packed dequant 与原始按通道 BF16 权重。
- checkpoint 的 `kv_cache_scheme` 为 null，但启动参数强制使用 FP8-E4M3 KV；需确认模型期望的 KV 量化约定。
- DSA indexer `weights_proj(x.float())` 与 channel-FP8 权重组合的 HCU 数值行为。
- `Glm5NextForConditionalGeneration` 权重映射、`fused_qkvbfg_a_proj` slice 顺序，以及 MTP `eh_proj/enorm/hnorm` 的 BF16 加载结果。

## 已排除项

- EP、DeepEP、MoE dispatch、EAGLE：纯 TP 仍复现乱码。
- `_forward_aiter_torch_fallback` KV scale：GLM 的 `qk_rope_head_dim=0`，实际走 `triton_sparse_mla_fwd`，不进入怀疑的 einsum 分支。
- AITER fallback head padding：8 个 Q head 扩到 16，输出按相同 factor 取回，索引对应正确。
- TileLang mHC pre 数值实现：关闭该路径时乱码不变；真正问题是错误的 `norm_fused` 状态。
- DeepGEMM channel-FP8 MoE 权重准备：与 DCU 参考使用相同 packer 和 grouped GEMM 调用。
- `kpool_bf16_paged_mqa_logits` FP8 解码、Q/K scale 组合：与 E4M3FN 和算子代数一致。
- `set_mla_kv_buffer_fp8_quant_kernel` 的 `rope_dim=0` 分支：覆盖 GLM 的实际布局。
- `forward_mla_rocm.py` 漂移：仅为配置字段访问方式差异，无数值语义变化。
- GSM8K 服务崩溃：评测对应日志中没有 scheduler exception、fatal error 或 VMFault。

## 未解决问题

1. **P0：AITER kpool top-k 待端到端验证。** Torch fallback 已通过 2566/4107-token 冷 prefill，但 8K 因 page-table 完整复制 OOM；AITER 独立算子测试通过。
2. **P1：解码吞吐异常低。** GSM8K 请求约 20-50 token/分钟，EvalScope 超时且未生成 prediction。
3. **P2：HiCache Mamba backup VMFault。** 当前仅通过关闭 HiCache 规避。
4. **P2：长生成仍有残余质量问题。** 20 条文本 sanity 仅 15/20，缺少失败样例分类和可信基线。
5. **P2：DeepSeek-V4 norm 修复未验证。** `de651cb5e6` 可能影响该模型，需独立回归。
6. **P2：BF16 KV cache 不可用。** DSA indexer 断言 `use_scaled_index_k_cache`。
7. **P2：精度尚未验证。** GSM8K 和 MATH-500 都没有完成有效评分。

## 下一位开发者的执行顺序

1. 更新 Todo，将当前事项标记为进行中。
2. 从 `sglang-model` 移植并验证 `_torch_topk_pooled_history`，不要恢复 CUDA `.cuh`，不要另写新算子。
3. 本地执行语义单测、`compileall`、`git diff --check`；更新本文档并提交，commit message 写清根因、索引语义、验证和限制。
4. 推送 `wl/glm5.3-flash`。进入容器后拉取并用 `git log --oneline -1` 确认 commit。
5. 确认没有其他任务使用相关共享内存后清理 sglang `/dev/shm` 残留，使用 HiCache-off 脚本启动。
6. 先跑独立 HCU 算子 probe，再跑 11 至 8192 token 的冷/热 prefill 分级回归。
7. 正确性通过后定位吞吐，再运行 GSM8K 5/20 条和 MATH-500；每一步同步更新 Todo 和本文档。

## 操作注意事项

- 服务启动约 7-8 分钟。一次启动内批量完成分级探针。
- HCU 算子签名和语义优先用独立 probe 验证，避免用完整服务反复猜测。
- 容器曾需要：

  ```bash
  git fetch wl 'refs/heads/wl/glm5.3-flash'
  git reset --hard FETCH_HEAD
  git log --oneline -1
  ```

  普通 `git fetch wl` 会把目标映射到 `wl/wl/glm5.3-flash`，可能不更新工作树。执行 `reset --hard` 前必须确认工作树没有需要保留的远端修改。
- SSH ControlMaster socket 消失或出现 `Too many authentication failures` 时立即停止，请用户从终端手动执行一次 `ssh nmz26`，禁止自动重试。
- mHC 单测必须在导入 sglang 前设置 `SGLANG_ROCM_USE_AITER_TILELANG_MHC=1`，否则不会覆盖真实分支。
