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
- 精度门槛：数据集得分在 85-90% 之间即认为精度正常，可推进到后续 milestone。
- 禁止无脑等待。发请求前必须能区分「处理慢」与「真 hang」：优先用流式读取并设置硬性 wall-clock 预算，一旦超出预算就报告状态而不是继续等到超时。单个实验的 timeout 不得设成小时级。

## 性能基准：批处理是当前唯一必要条件（2026-09-13）

在 `tp_bf16_aiter.sh`（TP=8、EP=1、无 DeepEP、BF16 KV、AITER DSA、
**关闭 CUDA graph**、无 EAGLE）上，用流式请求实测并发扩展性。同一状态
重测两轮（`max_tokens` 64 与 128），形状一致：

| 并发 | wall | 生成 token | 聚合 tok/s | 单请求 tok/s |
| --: | --: | --: | --: | --: |
| 1 | 19.2s | 127 | 6.62 | 6.62 |
| 2 | 19.6s | 254 | 12.97 | 6.49 |
| 4 | 19.7s | 502 | 25.53 | 6.38 |
| 8 | 19.8s | 1002 | 50.69 | 6.34 |
| 12 | 20.6s | 1508 | 73.37 | 6.11 |
| 16 | 20.1s | 2016 | 100.38 | 6.27 |

结论：

1. **聚合吞吐随并发线性增长**，到 `--max-running-requests=16` 仍未饱和；
   并发 16 相对串行是 **~15x**。
2. **单请求延迟几乎不随批次变化**（持续 ~6.3 tok/s），说明 bs=1 时约 6s/step
   的耗时主要来自固定的每步开销（launch、dispatch、集合通信延迟），而不是
   算力饱和。
3. 因此 **先说不用做 profiling**：批处理本身就能把验证速度提上来，零代码
   改动、零精度风险。热点算子调优是另一条独立优化路径，不得与「先拿到精度
   数字」混为一谈。

因此评测必须以并发方式运行，不能再用 `--eval-batch-size 1`。此前文档中
「20-50 token/分钟」的旧数字来自完整配置和其他路径，纯 TP 串行实际约
6.3 tok/s。

## Milestone 1 结果：纯 TP 精度

配置（`tp_bf16_aiter.sh`，服务已包含至 `44e500b0f3` 的全部修复）：

```text
--tp-size 8 --ep-size 1 --moe-a2a-backend none --moe-runner-backend triton
--kv-cache-dtype bfloat16 --disable-cuda-graph --disable-piecewise-cuda-graph
（无 EAGLE、无 DeepEP、无 HiCache）
SGLANG_USE_FP8_W8A8_MOE=1
```

### GSM8K

| 规模 | 并发 | 耗时 | EvalScope 分数 | 独立复核 |
| ---: | --: | --: | --: | ---: |
| 5 | 5 | 52s | 100% | 5/5 全部与数据集 gold 一致 |
| 50 | 16 | 2 分 30 秒 | 98% | 49/50 = 98% |
| 1319（全量） | 16 | 约 55 分钟 | **96.66%** | 1274/1319 = 96.6%，与官方一致 |

全量报告：`/home/work/evalscope_uv/out_gsm_full/reports/glm5.3-fp8-tp/gsm8k.json`
（`metrics[0].num=1319`、`score=0.9666`）。性能：平均延迟 38.2s、TTFT 0.64s、
TPOT 0.17s、输出 222 token/题、单请求 5.79 tok/s。

**结论：M1 纯 TP 精度 milestone 达成。** 96.66% 远高于 85-90% 门槛，
且高于官方 AMD nightly 在同配方下 MI300X/MI355X 的 GSM8K 约 97% 水平
（见 §4.7 引用）。1319 条全量、并发 16、`max_tokens=768`，无截断兜底、
无 prompt 特判。

失败形态分析（全量 1319 条）：答错 35 条、未输出 `\boxed{}` 10 条。
错例均为个位数算术偏差（如 idx=12 得 12、gold 13），**无结构性崩溃、
无乱码、无重复 `!`**。与前一阶段「输出 `!`」的现象相比，说明此前的
FP8 MoE `swiglu_limit`、KDA raw beta 和 mHC norm 三项修复共同起效。

`limit=5` 与 `limit=50` 的预测均用独立脚本对照 `openai/gsm8k` 测试集 gold
答案重新提取 `\boxed{}` 并逐一比对，结果与 EvalScope 分数一致。这不是只
看文本是否通顺，也不是依赖 scorer 的单一结论。

50 条中的唯一错误是真实算术失误（idx=12，得 12、gold 13），不是结构性
错误。partial（540 条）误答 7 条，同样为个位数算术偏差，无空答案、无
乱码、无 `!` 重复。

### 关键发现：此前「短 chat 错误」的测量口径有误

`tp_bf16_aiter.sh` 设置了 `--reasoning-parser glm45`。GLM-5.3-Flash 是
**推理模型**，服务会把思考过程放进 `reasoning_content`，而把最终答案放进
`content`。此前用 `max_tokens=64` 发请求时，64 个 token 会被思考过程全部
占用，`content` 返回空字符串——这被误读为「输出错误/空输出」。

放宽 `max_tokens` 后同一批 prompt 的输出完全正确：

```text
Q: What is the capital of France?    -> content='The capital of France is **Paris**.'
Q: 2+2=?                             -> content='**4**'
Q: Water is composed of hydrogen...  -> content='Water is composed of hydrogen and **oxygen**. Its chemical formula is H₂O...'
```

教训：对推理模型做精度判断必须看 `content`，且 `max_tokens` 必须留足
思考预算。评测 `max_tokens` 低于 512 会大面积截断，产生假的「低分」。

## 当前状态

- 本地分支：`glm5.3-flash`；推送目标：`wl/glm5.3-flash`。
- 当前本地基线：`8f0473a36d`；正在修复 HCU FP8 KV cache。
- 远端代码：`/home/work/code/sglang-das`；最后确认的远端 commit：`8f0473a36d`。
- 模型：`/home/work/GLM-5.3-Flash-Channel-FP8-w8a8`。
- 硬件：8 张 HCU，`gfx938`。
- M1 基线配置：TP=8、EP=1、无 DeepEP/EAGLE/HiCache/CUDA graph，BF16 KV + AITER DSA。
- 完整目标配置仍包含 EP、DeepEP、FP8 KV、EAGLE 和 HiCache；这些变量必须在 M1 基线上逐项恢复。
- **M1（纯 TP 精度）已达成**：`tp_bf16_aiter.sh` 配方下 GSM8K 全量 1319 条 = **96.66%**（独立复核 96.6%），高于 85-90% 门槛。
- M2（清理 `979baf5a81` 之后的适配代码）和 M3（开 CUDA graph + EAGLE 后精度与接受率）待办。

## Todo（每次提交必须更新）

| 状态 | 优先级 | 事项 | 完成标准 |
| --- | --- | --- | --- |
| 完成 | M1/P0 | **纯 TP 部署精度正常** | 达成：GSM8K 全量 1319 条 = **96.66%**，独立复核 96.6%，高于 85-90% 门槛；长 prompt 8K 冷/热通过；无截断兜底 |
| 完成 | M1/P0 | 接入 HCU AITER kpool top-k | 2568/4109/8209-token 冷 prefill 和 8K cache hit 通过；无 OOM/VMFault |
| 完成 | M1/P0 | 完成长 prompt 回归 | HiCache 关闭时 8K 冷/热 prefill 通过；低长度已由此前分级覆盖 |
| 完成 | M1/P0 | 修复 BF16 KV 下的 kpool index-K 分配 | BF16 KV 已分配 scaled FP8 index-K，原断言消失并进入 DSA 执行 |
| 完成 | M1/P0 | 修复 KDA safe gate / raw beta 语义 | 调用契约已与官方一致；GSM8K 分数 96.66% 证实端到端生效 |
| 完成 | M1/P0 | 移植并验证 DCU GLM KDA 精度修复 | 关闭 HCU 小网格融合、保留 FP32 中间量；真实 GLM 形状与 naive recurrent 对齐 |
| 完成 | M1/P0 | 补齐 KDA raw beta 调用链 | `KDAAttnBackend -> TritonKDAKernel -> chunk_kda` 全链路传递；20/79/139-token chat 及全量 GSM8K 恢复 |
| 完成 | M1/P0 | 复用 TP FP8 fused MoE 并保留 SwiGLU limit | `SGLANG_USE_FP8_W8A8_MOE=1`；`swiglu_limit=10.0` 进入 clamp+quant 且保持 E4M3FN；GSM8K 96.66% |
| 完成 | P1 | 定位并解除评测吞吐阻塞 | 查出真实瓶颈是每步固定开销（非算力饱和）；用并发 16 把聚合吞吐从 6.6 提到 100 tok/s（15x）；GSM8K 可用批处理完成 |
| 完成 | M1/P0 | 完成 GSM8K 全量 | 1319 条 = 96.66%，独立复核一致；已记录失败样例分布（答错 35、未输出 boxed 10，均为个位数算术偏差） |
| 进行中 | M1/P0 | 修复并验证 HCU FP8 KV cache | 启动参数改为 `--kv-cache-dtype fp8_e4m3`；短 chat 内容正确；确认使用 528-byte 动态分组 scale 布局 |
| 待办 | M1/P1 | 完成 MATH-500 | 用并发配方执行并记录分数、截断率和失败样例 |
| 待办 | M2/P0 | **整理 GLM5.3 适配代码（`979baf5a81` 之后全部 commit）** | review 并清除实验性/workaround 代码，在本文档单独成节说明每一项的去留理由与证据 |
| 待办 | M3/P0 | **开启 CUDA graph + EAGLE（5/1/6）后精度仍正常且接受率非 0** | `--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`；开 CUDA graph；精度达 85%+；接受率 > 0 |
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
| `f17b2903b1` | kpool compression 不再随主 KV dtype 错分配普通 BF16 index-K | 原断言消失；TileLang 进入编译，AITER DSA 可启动 |
| `aa8cc413ba` | 补齐 KDA safe-gate 分支和 raw beta sigmoid | HCU 服务启动；三条短 chat 仍错误，输出形态改变 |
| `009b187323` | 移植 DCU `30a5b3e3704` 的 GLM KDA 精度修复 | KDA 回归测试通过；真实 GLM 形状与 naive recurrent 相对误差低于 0.0038；端到端短 chat 仍错误 |
| `3cfabe3094` | TP 旧 FP8 fused MoE 补齐 GLM SwiGLU limit | HCU warmup 进入该路径；首次实现误用了 INT8 clamp+quant，已在后续提交修正 |
| `fe54e1f611` | FP8 fused MoE align 调用适配当前 LightOp | TP8 服务越过 warmup 并就绪；短 probe 仍乱码，随后定位到激活量化 dtype 错误 |
| `1eaf649dc7` | SwiGLU clamp 后保持 FP8 per-token 量化 | 20-token chat 恢复；79/139/259/506/1287/7K/10K token 仍逐步退化或输出 `!` |
| `44e500b0f3` | KDA Triton wrapper 继续传递 raw beta | HCU 端到端及 GSM8K 全量验证通过 |
| `8f0473a36d` | 登记 M1 GSM8K 全量 96.66% | 文档提交 |
| 本提交 | 恢复 HCU FP8 KV 的动态 scale 存储与 LightOp decode | `compileall`、`git diff --check` 通过；HCU 短请求待验证 |

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

`f17b2903b1` 仅在 `index_kpool > 1 && index_kpool_compress` 时把错误解析出的 BF16 index-K 改为 scaled FP8；普通 DSA BF16 cache 和 gfx936 INT8 opt-in 不变。HCU 验证确认原断言消失，服务进入 DSA 执行。

官方 TileLang DSA 随后在 gfx938 的 TileLang/TVM layout inference 中因 `FloorDiv` 除零崩溃，位置为 `tilelang_sparse_fwd -> sparse_mla_fwd_decode_partial`。该路径与 `sglang-model` 使用同一类实现，当前判断为编译器/架构兼容问题，不重写算子。仅把 DSA backend 换成 AITER 后，BF16 KV 服务可启动；短 chat 不再固定输出 `!`，但仍生成无关内容，说明 FP8 KV 不是唯一根因。

### 4.8 KDA safe gate 和 raw beta 语义丢失

模型配置含 34 个 KDA 层，且 `gate_lower_bound=-5.0`。调用端在 packed GLM extend 路径明确传入：

```text
lower_bound=-5.0
beta_is_raw=True
```

当前 `chunk_kda()` 存在两个回归：

1. 调用 `chunk_kda_fwd_intra()` 时未传 `safe_gate=lower_bound is not None`，导致 safe gate 错走普通 token-parallel intra 公式。
2. 函数签名未声明 `beta_is_raw`，该值被 `**kwargs` 静默吞掉，raw beta 未执行 sigmoid。

官方基线已包含这两处逻辑；DCU 参考提交 `83b848c1a9` 也专门修复 safe gate 向 intra 的传递。自然对数与 `exp2` 的底数转换并非当前问题：现有 `kda_gate_chunk_cumsum(..., scale=RCP_LN2)` 已完成转换。

`aa8cc413ba` 按官方实现补齐两处语义。BF16 KV + AITER DSA 服务启动后，三条 greedy 短 chat 仍生成中文碎片、反斜线或重复短语，均因 64-token 上限截断。输出与修复前不同，说明 KDA 是有效影响因素，但仍有算子精度问题。

### 4.9 DCU 已知 GLM KDA 精度修复

`sglang-model` 的 `30a5b3e3704` 明确用于修复 GLM5-NEXT-70B 精度，当前分支缺少其中两项：

1. 小网格下的 fused diagonal/recompute 在 HCU 上精度不足，DCU 分支固定改用拆分 kernel。
2. 跨 16-token 子块计算 `Aqk/Akk` 前，将门控缩放后的 FP32 中间量显式转成 BF16，会放大 34 层 KDA 的累计误差。

当前代码已具备该提交中的 IEEE dot 配置，因此本次只在 HCU 关闭小网格融合，并删除上述 BF16 强制降精度。门控累加仍沿用当前主线的 `RCP_LN2 + exp2` 约定，不照搬旧分支的 `exp_e`，避免重复乘 `log2(e)`。

验证结果：

- `TestKDAChunkExponentDomain` 三组 HCU 子测试通过。
- 真实 GLM 单 rank 形状 `H=8, K=V=128`，序列长 31/129，启用 `lower_bound=-5` 和 raw beta；对 naive recurrent 的 output 相对误差为 `0.00375/0.00378`，state 相对误差为 `0.00246/0.00232`，最大绝对误差 `3.05e-05`，无非有限值。
- BF16 KV + AITER DSA 的端到端短 chat 仍错误。KDA 单算子证据已充分，停止继续修改 KDA。

### 4.10 TP channel-FP8 MoE 丢失 SwiGLU limit

模型配置声明 `swiglu_limit=10.0`，`glm5_next.py` 已将其写入 `MoeRunnerConfig`。但当前 TP 执行存在两条错误路径：

1. `--moe-runner-backend triton` 并不保证使用 Triton。BF16/HCU 下通用 `fused_moe` 会因 `_use_aiter_moe` 自动转入 AITER；该旧接口只接收 `gemm1_alpha/gemm1_limit`，没有传递 `swiglu_limit`。
2. 可复用的 HCU FP8 fused MoE 由 `SGLANG_USE_FP8_W8A8_MOE=1` 启用，但其 GEMM1 后固定调用无 clamp 的 `fuse_silu_mul_fp8_quant`，同样丢失 limit。

独立检查表明 DSA 不是当前根因：现有 AITER 不含 MLA stage1 ASM，实际回退到 `triton_sparse_mla_fwd`；真实 head dim 下对 Torch 的相对误差约 `0.0022`。相反，GLM 真实 MoE 形状的 AITER channel-FP8 探针与 Torch 参考严重偏离；仓库对应 HCU BF16/channel-FP8 测试也被明确禁用，注明尚需数值验证。

处理方式遵循 DCU 参考实现，不增加新算子：

- TP 启用既有 `SGLANG_USE_FP8_W8A8_MOE=1` 路径。
- 将 `MoeRunnerConfig.swiglu_limit` 经普通 W8A8 和 compressed-tensors 两个入口传入 `fused_moe_fp8_w8a8`。
- limit 存在时先按官方语义原地 clamp gate/up，再复用 LightOp `fuse_silu_mul_fp8_quant`；无 limit 的模型保持原路径。
- HCU 端到端验证必须确认启动环境同时包含 `SGLANG_USE_FP8_W8A8_MOE=1`，否则不会覆盖本次修复。

首次 HCU 启动在 warmup 确认进入 `fused_moe_fp8_w8a8`，随后报错：

```text
AttributeError: module 'lightop.op' has no attribute 'moe_align_block_size_out'
```

当前 LightOp 导出的接口已改为 `moe_align_block_size(..., Is_EP=False, Is_fuse_fill=True)`。`sglang-model` 的旧 FP8 fused MoE 已使用该接口，并在调用前用无效 token id 初始化 padding，避免 LightOp 未写 padding 槽时 GEMM 读取垃圾索引。本次按参考实现同步这两项兼容，不改 align 算法。

align 修复后 TP8 服务成功就绪，但三条固定 probe 仍输出重复乱码。独立 dtype 探针确认首次 limit 实现选错了算子：

```text
fuse_silu_mul_clamp_quant -> torch.int8, max=127
fuse_silu_mul_fp8_quant   -> torch.float8_e4m3fn, max=448
```

前者服务于 W4A8/INT8 激活链路，不能喂给 `moe_gemm_marlin_w8a8_fp8`；将 INT8 字节按 FP8 解释会直接破坏幅值。当前 LightOp 没有 clamp+SiLU+FP8-quant 一体接口，因此按官方通用 MoE 的已有做法，先对 GEMM1 BF16 输出的 gate/up 两半原地限幅，再调用既有 `fuse_silu_mul_fp8_quant`。这只增加 clamp，不新增算子，并保持 GEMM2 输入为 E4M3FN。

修正 dtype 后，20-token chat 已能正确生成 `Paris`，但 79 token 开始出现重复片段，139 token 以上稳定输出 `!`；1.3K/3.3K/7.1K/10K 均复现。marlin GEMM2 配置 A/B 证伪了错误配置猜测：79-token 使用的 `MODE=56` 对 Torch 参考相对误差约 `0.026`，强制换成短请求的 `MODE=57` 反而约为 `1.01`。

### 4.11 KDA Triton wrapper 吞掉 raw beta

长度分级将剩余错误定位到 KDA 状态累计。`KDAAttnBackend.forward_extend()` 根据 gate 布局传入 `beta_is_raw=gate_was_flat`，底层 `chunk_kda()` 也已在 `aa8cc413ba` 支持 raw beta sigmoid，但中间的 `TritonKDAKernel.extend()` 未声明该参数，导致它被 `**kwargs` 静默吞掉。

官方基线提交 `c66a285c94` 明确补齐了 wrapper 的形参和向下传递。本次照搬这两行契约修复，不修改 KDA 算子。此前 KDA 单算子通过只能证明传入正确 beta 后数值正常，不能覆盖这个服务调用链缺口。

### 4.12 FP8 KV 不能使用无 scale 的 512-byte 原始布局

checkpoint 的 `kv_cache_scheme=null` 只表示没有静态校准 scale，不能据此断言 FP8 KV 必须禁用。DCU 参考实现采用运行时动态量化：512 个 latent 按 128 元素分成 4 组，每组保存一个 FP32 scale，因此无 RoPE 的物理行宽是 `512 + 4 * 4 = 528` bytes。

当前代码的 HCU 路径被通用 HIP workaround 错误覆盖：配置器分配 512-byte 原始行，写入时直接 BF16→FP8 cast，DSA backend 又把 `flashmla_auto/flashmla_kv` 强制改成 AITER。这样确实等价于 scale=1.0，可能产生饱和误差。

本提交按 `sglang-model` 恢复既有算子链，不新增量化算子：

1. HCU FP8 DSA 分配 528-byte scaled row，写入复用 `quantize_k_cache_separate`。
2. no-RoPE prefill 使用 HCU FlashMLA sparse BF16 compute；持久化仍为 FP8 scaled row。
3. decode 复用 LightOp `decode_gather_and_up_convert_with_indices`，把选中的 packed FP8 KV 动态反量化为 BF16，再交给 HCU FlashMLA。
4. BF16 KV 继续走已验证的 AITER fallback；generic HIP 的原始布局不变。

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
| `--kv-cache-dtype bfloat16` | index-K 分配已修复；可配合 AITER DSA 启动 |
| DSA `_forward_tilelang` | `libtilelang.so` 在 `GemmNode::InferLayout -> make_hcu_swizzled_layout` 崩溃；`D_V=512`、head 8/16 的 gfx938 路径不可用 |
| `SGLANG_USE_AITER=1` | 当前 HCU AITER 缺少 `gemm_a8w8_blockscale`，导入阶段失败；继续使用细粒度 AITER 开关 |
| 启动参数关闭 mHC post TileLang | 环境变量会在后续 hook 被重新设为 true；如需关闭必须改代码 |

## 待复核的数值路径

以下项尚无独立证据证明有错，但在建立精度基线前不能删除：

- channel-FP8 权重经 `pack_int8_weight_enk_to_w6_low_latency` 后是否保持幅值；应对单个 expert 比较 packed dequant 与原始按通道 BF16 权重。
- FP8 KV 的 528-byte 动态分组 scale 路径已按 DCU 参考恢复，待端到端短请求确认精度。
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

1. **P0：HCU FP8 KV 端到端待验证。** 本地已恢复 528-byte 动态 scale 写入和 LightOp decode，需用 `--kv-cache-dtype fp8_e4m3` 启动并跑短 chat。
2. **P2：HiCache Mamba backup VMFault。** 当前仅通过关闭 HiCache 规避。
3. **P2：DeepSeek-V4 norm 修复未验证。** `de651cb5e6` 可能影响该模型，需独立回归。
4. **P2：MATH-500 尚未执行。** GSM8K M1 已达成，MATH-500 仍需补测。

## 下一位开发者的执行顺序

1. 推送本提交并在容器拉取；把 M1 脚本的 KV 参数单独改为 `fp8_e4m3`，其余变量保持不变。
2. 服务就绪后用足够的生成预算检查 Paris、2+2 和水的组成；分别查看 `reasoning_content` 与 `content`。
3. 短请求正确后补 GSM8K 抽样，再决定是否把 FP8 KV 纳入目标配置。
4. 每次提交同步更新 Todo、调试记录、验证和未解决问题。

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
