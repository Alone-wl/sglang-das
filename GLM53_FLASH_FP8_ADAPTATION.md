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
- 当前本地基线：`082ef26dfd`；本提交修复 BF16 KV 下的 kpool index-K 分配。
- 远端代码：`/home/work/code/sglang-das`；最后确认的远端 commit：`082ef26dfd`。
- 模型：`/home/work/GLM-5.3-Flash-Channel-FP8-w8a8`。
- 硬件：8 张 HCU，`gfx938`。
- 主配置：TP=8、EP=8、DeepEP normal、DSA/NSA、FP8-E4M3 KV cache、Mamba、EAGLE。
- 当前临时关闭 HiCache，使用 `/home/work/glm/ifb_nohicache.sh`。这是规避 Mamba backup VMFault 的临时措施，不是最终配置。
- 短 prompt 已能生成连贯文本；短输出乱码的根因已修复。
- AITER kpool top-k 已使纯 TP 的 2568/4109/8209-token 冷 prefill 和 8K prefix-cache 命中请求通过，无 OOM/VMFault。
- TP 解码约 5 tok/s。EvalScope GSM8K 5 条均生成 256 个 `!` 并因 `max_tokens` 截断，得分 0%；当前阻塞是模型数值错误，不再是请求超时。

## Todo（每次提交必须更新）

| 状态 | 优先级 | 事项 | 完成标准 |
| --- | --- | --- | --- |
| 进行中 | M1/P0 | **纯 TP 部署精度正常** | TP=8、EP=1、无 DeepEP、无 EAGLE；长 prompt 稳定；GSM8K/MATH-500 与可信基线对齐；记录配置、分数、截断率和失败样例 |
| 完成 | M1/P0 | 接入 HCU AITER kpool top-k | 2568/4109/8209-token 冷 prefill 和 8K cache hit 通过；无 OOM/VMFault |
| 完成 | M1/P0 | 完成长 prompt 回归 | HiCache 关闭时 8K 冷/热 prefill 通过；低长度已由此前分级覆盖 |
| 进行中 | M1/P0 | 修复 BF16 KV 下的 kpool index-K 分配 | 官方 AMD 配方可启动；短 chat 正确；8K 冷/热通过 |
| 完成 | P1 | 定位 GSM8K 极低吞吐 | 纯 TP 约 5 tok/s；5 条评测可完成，原 20-50 token/分钟来自完整配置/旧路径 |
| 进行中 | M1/P0 | 完成 GSM8K smoke | BF16 KV 配方下先跑 5 条，再跑 20 条；记录截断率、失败样例和分数 |
| 待办 | P1 | 完成 MATH-500 smoke | GSM8K 稳定后执行并记录配置、分数和失败样例 |
| 待办 | P2 | 修复 HiCache Mamba backup VMFault | 开启 HiCache 后长 prompt 不再触发 `transfer_mamba_backup_kernel` VMFault |
| 待办 | P2 | 验证 DeepSeek-V4 norm 修复 | 启动对应模型确认不存在重复归一化 |
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
| `729df85bdc` | 登记 TP 精度 milestone 与 kpool 修复 | 文档提交 |
| `082ef26dfd` | HCU 长 prefill 使用 AITER kpool top-k | 8K 冷/热 prefill 通过；无 OOM/VMFault |
| 本提交 | kpool compression 不再随主 KV dtype 错分配普通 BF16 index-K | 本地 `compileall`、`git diff --check`；HCU 待验证 |

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

`082ef26dfd` 已按 `sglang-model` 接入 `aiter.kpool_topk`：由 `SGLANG_NSA_KPOOL_AITER_TOPK=1` 控制，仅在 `index_kpool` 为 4/16、`index_topk=2048` 且存在 `seq_lens` 时启用；算子缺失时保留 Torch fallback。

端到端结果：

| Prompt token | 缓存 | 结果 |
| ---: | ---: | --- |
| 2568 / 4109 / 8209 | 0 | HTTP 200，分别 4.68 / 5.92 / 11.13 秒 |
| 8205，首次 / 再次 | 0 / 8192 | HTTP 200，11.02 / 1.51 秒 |

### 4.7 官方 AMD 精度配方暴露 BF16 index-K 分配错误

官方仓库 `b26cb8d0d3` 的 GLM-5.3 AMD nightly 使用 TP8、BF16 KV、TileLang DSA、Triton MoE、关闭 CUDA Graph；MI300X/MI355X 的 GSM8K 实测约 97%。当前 TP 脚本使用 FP8 KV 和 AITER DSA，与可信配方不一致。

对齐配方时，服务在 warmup 稳定报错：

```text
get_index_k_with_scale_buffer
AssertionError: Scaled index K cache is not enabled
```

根因是 `0258c9987e` 引入的 HCU index-K 多格式逻辑按主 KV dtype 选择 index-K 格式。BF16 KV 因此分配普通 BF16 index-K；但 GLM-5.3 启用 kpool compression 后，写入/更新算子的 ABI 固定为 packed FP8 K + FP32 scale，必须使用 `IndexKeyCache`。官方基线的 index-K cache 也独立于主 KV dtype。

本次修复仅在 `index_kpool > 1 && index_kpool_compress` 时把错误解析出的 BF16 index-K 改为 scaled FP8；普通 DSA BF16 cache 和 gfx936 INT8 opt-in 不变。HCU 验证结果待补。

### 5. EvalScope：旧超时已解除，当前是确定性重复

GSM8K smoke 配置为 20 条、batch size 1、`max_tokens=2048`。运行 49 分钟仍为 0/20，EvalScope 多次报告请求超时，prediction 目录为空。

服务日志显示：

- `Scheduler hit an exception`：0。
- `Fatal Python error`：0。
- `KERNEL VMFault`：0。
- decode token 持续增加，并非死锁。
- 观测吞吐约 20-50 token/分钟；生成 2048 token 可能需要接近一小时。

纯 TP 下重新运行 5 条、`max_tokens=256` 后，评测在 4 分 14 秒内完成，平均输入 587 token、输出 256 token、约 5.1 tok/s。5 条输出均为连续 `!`，全部因 `max_tokens` 截断，得分 0%。因此旧记录中的超时不是当前主要阻塞，模型数值仍不正确。

隔离结果：关闭 `SGLANG_USE_DEEPGEMM_MOE` 或 CUDA Graph 均不能恢复短 chat 精度；mHC 融合 norm 与 `sglang-model` 的 AITER mHC + 显式 RMSNorm 对纯 Torch 参考的平均误差均约 `0.00112`、最大误差均 `0.03125`，不支持再次修改 mHC。

EvalScope 1.11.1 已安装在 `/home/work/evalscope_uv/.venv`，uv 版本 0.12.13；GSM8K 1319 条、MATH-500 500 条均能加载。

## 已知不可用或受限路径

| 路径 | 结果 |
| --- | --- |
| `--kv-cache-dtype bfloat16` | kpool compression 错分配普通 BF16 index-K；已本地修复，待 HCU 验证 |
| DSA `_forward_tilelang` | `libtilelang.so` 在 `GemmNode::InferLayout -> make_hcu_swizzled_layout` 崩溃；`D_V=512`、head 8/16 的 gfx938 路径不可用 |
| `SGLANG_USE_AITER=1` | 当前 HCU AITER 缺少 `gemm_a8w8_blockscale`，导入阶段失败；继续使用细粒度 AITER 开关 |
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
- DeepGEMM MoE 和 CUDA Graph：纯 TP 分别关闭后，短 chat 的错误输出不变。
- mHC norm 计算精度：融合路径与显式 RMSNorm 路径对 Torch 参考误差相当，不是当前 0% GSM8K 的证据链。

## 未解决问题

1. **P0：BF16 KV 的 kpool index-K 修复待验证。** 修复后需按官方数值配方完成启动、短 chat、8K 冷/热和 GSM8K。
2. **P0：纯 TP 精度错误。** FP8 KV + AITER DSA 配置下 GSM8K 5 条均输出 256 个 `!`，得分 0%。
3. **P2：HiCache Mamba backup VMFault。** 当前仅通过关闭 HiCache 规避。
4. **P2：长生成仍有残余质量问题。** 20 条文本 sanity 仅 15/20，缺少失败样例分类和可信基线。
5. **P2：DeepSeek-V4 norm 修复未验证。** `de651cb5e6` 可能影响该模型，需独立回归。
6. **P2：MATH-500 尚未执行。** 先完成 TP GSM8K 精度门槛。

## 下一位开发者的执行顺序

1. 验证 BF16 KV 下 kpool compression 仍分配 packed scaled index-K，普通 BF16 DSA 分配逻辑不变。
2. 按官方数值配方启动 TP8：BF16 KV、TileLang DSA、Triton MoE、关闭 CUDA Graph；全局 AITER 开关在当前镜像不可用。
3. 跑短 chat 和 8K 冷/热；若 TileLang 在 gfx938 崩溃，分别保留 BF16 KV，只替换 prefill/decode backend 定位。
4. 精度恢复后运行 GSM8K 5/20 条，再运行 MATH-500；记录分数、stop rate、截断率和失败样例。
5. 每次提交同步更新 Todo、调试记录、验证和未解决问题。

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
