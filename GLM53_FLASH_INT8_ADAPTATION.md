# GLM-5.3-Flash INT8 适配交接

## 1. 文档用途

本文记录 `main_glm53_flash` 分支上的 GLM-5.3-Flash INT8 适配现状，供后续开发者直接继续调试。内容以仓库代码和提交记录为准；未实际验证的结论必须标为“待验证”，不得按已完成处理。

本文只覆盖 INT8（W8A8）路径。FP8 权重、FP8 MoE、FP8 KV cache 的适配结论不在本文中复用；如需继续 FP8 工作，应另建或更新对应文档。

## 2. 强制维护规则

每次提交前必须同步更新本文，至少完成以下事项：

1. 更新第 4 节状态和第 9 节 Todo。
2. 在第 8 节登记提交、问题、修改和验证结果。
3. 补充本次使用的启动参数、环境变量、模型路径、机器拓扑和日志位置；敏感信息不得入库。
4. 明确区分静态检查、算子测试、启动验证、请求验证和精度评测，不得用低层测试替代整模型结论。
5. 提交信息必须说明问题表现、根因、修改范围、兼容性影响和验证方式，使 commit message 本身可独立作为变更记录。

若 Todo、代码和提交信息不一致，以代码和可复现结果为准，并立即修正文档。

## 3. 工作约束与参考代码

- 本地开发仓库：`/Users/wanglong/Code/sglang-das`
- 官方基线：`/Users/wanglong/Code/sglang`
- DCU INT8 参考实现：`/Users/wanglong/Code/sglang-model`
- 服务器：`nmz26`
- 容器：`dev-wl-glm2`
- 现场启动脚本：`/home/work/glm/ifb.sh`
- 代码阅读和修改在本地完成；提交并推送后，才在服务器容器中拉取和验证。
- 非算子问题优先对照官方仓库；HCU/DCU 算子调用和数据布局问题优先对照 `sglang-model`，不得在已有实现可复用时另写一套算子。
- 不得提交临时日志、数据 dump、硬编码模型路径、一次性开关或仅为绕过错误而加入的分支。

## 4. 当前基线与结论

| 项目 | 当前值 |
| --- | --- |
| 分支 | `main_glm53_flash` |
| 当前提交 | `db1f092c7bd6a21b9777b53b058a1b2ba82edc43` |
| 合入起点 | `ee0f2375e5fb285e4fb021a9831fa88018244679` |
| 远端跟踪状态 | 当前提交与 `mx_int8/main_glm53_flash` 一致（本文创建前） |
| 工作树注意项 | `.diagnostics/` 为既有未跟踪目录，不属于本任务，不得误提交 |

当前结论：

- GLM-5.3-Flash 模型结构、KDA/DSA、KPool、MTP、LayerSplit、HiCache 以及 HCU INT8 W8A8 的主要代码已移植到当前主线结构。
- Dense INT8 和 MoE INT8 均有实现；HCU MoE 可进入 DeepGEMM W8A8 路径。
- 已有提交记录包含 HCU 单算子、KPool 和 CUDA Graph 回放验证，以及 16 rank 图捕获成功记录。
- 仓库中未找到 GLM-5.3-Flash INT8 的固定启动配方、注册测试或完整精度报告。
- 因此当前状态是“实现基本齐备、局部运行验证通过、整模型精度与完整部署矩阵待补证”，不能标记为生产可用。

## 5. 实现结构

### 5.1 模型与配置

- `python/sglang/srt/models/glm5_next.py`
  - GLM-Next 主模型、稀疏 MoE、MTP、KDA 门控和权重加载。
  - MoE 使用 `moe_quant_config_override or quant_config`，量化配置会下传到专家层。
  - 创建 MoE 时传递 `swiglu_limit`。该参数属于 GLM-5.3-Flash 激活语义，INT8 fused MoE 路径不得忽略。
- `python/sglang/srt/configs/glm5_next.py`
  - 模型配置解析及 GLM-Next 专用配置。

### 5.2 Dense INT8

- `python/sglang/srt/layers/quantization/w8a8_int8.py`
  - `W8A8Int8Config`、`W8A8Int8LinearMethod` 和 `W8A8Int8MoEMethod`。
  - HCU 激活量化使用 LightOp per-token INT8 quant；矩阵乘根据配置进入 HCU INT8 scaled-mm/DeepGEMM 路径。
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`
  - compressed-tensors W8A8 INT8 Dense 方案及权重加载。

### 5.3 MoE INT8

- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py`
  - compressed-tensors INT8 MoE 方案，要求 channel-wise 权重量化。
  - 加载完成后准备 DeepGEMM 所需权重布局。
- `python/sglang/srt/layers/quantization/hcu_deepgemm_w8a8_utils.py`
  - 校验专家权重形状和位宽，完成普通或 ASM 路径的权重打包，并保存原始 shape。
- `python/sglang/srt/layers/moe/ep_moe/layer.py`
  - 识别 HCU W8A8 INT8 DeepGEMM 配置。
  - 覆盖普通 dispatch 和 masked dispatch 两条 EP MoE 路径。
  - contiguous 与 masked 路径都必须把 `swiglu_limit` 传入 fused 激活；后续修改要同时检查两条路径。

相关环境变量：

- `SGLANG_USE_DEEPGEMM_MOE`
- `SGLANG_INT8_DEEPGEMM_ASM`
- `SGLANG_REUSE_W8A8_INT8_EP_MOE_WORKSPACE`

每次验证必须记录这些变量的实际值，避免把后端切换误判为代码回归。

### 5.4 注意力、KPool 与缓存

- `python/sglang/srt/layers/attention/glm5_next/dsa_backend.py`
- `python/sglang/srt/layers/attention/glm5_next/`
- `python/sglang/srt/layers/attention/linear/kda_backend.py`
- `python/sglang/srt/mem_cache/glm5_next.py`
- `python/sglang/kernels/ops/attention/fla/hcu/`

这些模块是 GLM-5.3-Flash 运行基础，不等同于 INT8 权重量化本身。排查精度时必须分开看：

- “INT8 模型”表示权重/激活进入 W8A8 路径，不表示所有缓存都必须为 INT8。
- 主 KV cache 的 dtype 由启动参数决定，可独立使用 BF16 或 FP8。
- DSA index-K/KPool 的压缩格式可能使用 FP8，这是索引注意力的数据格式，不能据此判断主 KV cache 已启用 FP8。

## 6. 关键语义与易错点

### 6.1 `swiglu_limit` 不能丢失

GLM-5.3-Flash 的 fused MoE 激活带 limit 语义。复用通用 INT8 fused MoE 时，必须确认：

- contiguous dispatch 传入 limit；
- masked dispatch 传入 limit；
- TP 与 EP 路径结果一致；
- fallback 路径与 fused 路径的激活定义一致。

只验证算子能够运行不够；limit 丢失通常表现为服务正常但生成质量下降。

### 6.2 KDA gate 语义

`glm5_next.py` 中存在 KDA gate 下界处理；`kda_backend.py` 通过 `beta_is_raw` 区分原始 gate 和已处理 gate。修改张量 shape、flatten 或 backend 选择时，必须确认 gate 是否会被重复变换或漏变换。

### 6.3 FP8 KV 与 INT8 权重无绑定关系

INT8 权重适配不要求同时启用 FP8 KV。首次建立精度基线时建议使用 BF16 KV，把 FP8 KV 作为独立变量验证。是否允许 FP8 KV 应以 checkpoint 配置、实际 scale 路径和精度对比为准，不能只根据 `kv_cache_scheme` 字段或默认 scale 作结论。

### 6.4 图模式不能替代 eager 基线

已有提交修复了图捕获和 replay，但完整验证仍应先得到 eager 模式正确结果，再逐项开启 CUDA Graph、EAGLE、HiCache、LayerSplit 和 P/D。一次只改变一个变量。

## 7. 已有验证证据

| 范围 | 证据 | 可得结论 | 不能推出的结论 |
| --- | --- | --- | --- |
| 静态移植 | `f00d4e16cc` 的语法、diff 和静态检查 | 大规模移植可加载到当前代码结构 | HCU 可启动、精度正常 |
| MHC/KDA 图路径 | `3ed9588430` 记录 AITER MHC 输出、graph replay 对齐，16 rank 完成图捕获 | 对应低层路径和图捕获问题已修复 | 完整 P/D 服务稳定、整模型精度正常 |
| KPool | `5f518ba97c` 记录单 decode、per-query、多计划 decode 和 graph replay 探针通过 | KPool JIT namespace 与调用兼容 | 长上下文、并发请求无误差 |
| 其余运行修复 | `0ab6ac34b9` 至 `db1f092c7b` | 已处理 warmup、通信、HiCache、采样、调度和 HIP stream 等已知问题 | 所有组合均已回归 |
| 整模型准确率 | 无入库报告 | 无 | 不得声称准确率达标 |

## 8. 提交索引

| 提交 | 内容 | 验证/备注 |
| --- | --- | --- |
| `f00d4e16cc` | 从 DCU 参考分支前移植 GLM5-Next 与 HCU 相关能力 | 静态检查；未做 HCU 整体运行 |
| `f2e02183cf` | 导出 LightOp clamp quant root | INT8 激活量化依赖 |
| `026b03c6de` | 对齐 communicator 构造接口 | 主线 API 兼容 |
| `624fd1b0f2` | MoE dispatch 比较 LightOp backend 值 | 避免 backend 判断错误 |
| `8686761473` | 共享 GLM5 NextN loader 常量 | 去除重复常量 |
| `cdd4b8cae4` | 修复 MLA cache sizing 的 runtime getter 名称遮蔽 | 主线 API 兼容 |
| `3ed9588430` | 修复 GLM-Next HCU 启动、MHC、INT8 DeepEP、图元数据和 CP 元数据 | MHC/graph replay 探针通过；16 rank 图捕获完成；完整 P/D 当时仍在继续 |
| `0ab6ac34b9` | 为可选 RMS quant fusion state 提供默认值 | 修复 Prefill NextN warmup |
| `5f518ba97c` | 对齐 KPool JIT namespace | HCU KPool 多类探针和图回放通过 |
| `4a2096ee44` | 恢复 GLM KPool AOT write planner | AOT 写入规划修复 |
| `317aa5d8ef` | 使用主线 scheduler metrics reporter | 主线接口对齐 |
| `6782be0e38` | attention CP rank 间同步 grammar admission | CP 一致性修复 |
| `080d918b66` | 主线裁剪 padded QKV 后对齐 HCU KDA gate | KDA shape/语义修复 |
| `1cba0ae785` | 修复 HCU HiCache mapped pointer 和 layer transfer 所有权 | HiCache 修复 |
| `f2a1cd9b57` | HCU LightOp sampling 前沿用主线 NaN 处理 | 采样行为对齐 |
| `9cd0352166` | HCU reservation 下保留单个未完成 prefill chunk | 调度修复 |
| `2bcad26dd1` | HiCache reload 后修复 draft LayerSplit broadcast 顺序 | 分层/MTP 修复 |
| `db1f092c7b` | AOT TopK 使用正确 HIP current stream API | HIP stream 兼容 |

后续提交按时间顺序追加，不覆盖历史记录。每条至少写清问题、修改和验证。

## 9. Todo

### P0：建立可复现基线

- [ ] 从 `nmz26` 的 `/home/work/glm/ifb.sh` 登记完整启动命令；不得凭记忆补参数。
- [ ] 记录容器镜像/版本、驱动、torch、LightOp、AITER、DeepGEMM、通信库和 SGLang 提交。
- [ ] 记录 INT8 checkpoint 的精确路径、revision、`quantization_config` 和模型配置摘要。
- [ ] 记录机器数量、每机卡数、TP/EP/DP/CP 拓扑及所有相关环境变量。
- [ ] 建立最小 TP eager 基线：关闭 EP、EAGLE、CUDA Graph、HiCache、LayerSplit 和 P/D；KV cache 先用 BF16。
- [ ] 保存固定短请求的输入、参考输出、实际输出和服务日志。
- [ ] 运行固定精度集并登记命令、样本数、随机种子、评分脚本和结果。没有这一步不得关闭 P0。

### P1：验证 INT8 MoE 与 EP

- [ ] 对同一批输入比较 fallback MoE 与 HCU DeepGEMM W8A8 MoE。
- [ ] 分别验证 normal dispatch 和 masked dispatch。
- [ ] 对比启用/禁用 `SGLANG_INT8_DEEPGEMM_ASM` 的结果。
- [ ] 验证 `swiglu_limit` 在 TP、EP、contiguous、masked 四种组合中均生效。
- [ ] 增加覆盖 GLM-5.3-Flash INT8 MoE limit 语义的自动化测试。

### P2：逐项恢复优化

- [ ] CUDA Graph。
- [ ] EAGLE/MTP。
- [ ] HiCache。
- [ ] LayerSplit。
- [ ] P/D 部署。
- [ ] 每开启一项，重跑短请求和精度抽样，并记录相对基线差异。

### P3：工程化

- [ ] 增加 HCU GLM-5.3-Flash INT8 注册启动测试。
- [ ] 增加最小请求回归和可接受的精度门槛。
- [ ] 清理确认无用的兼容分支、调试输出和临时环境变量。
- [ ] 将最终可用启动脚本和环境说明纳入版本管理，避免只保存在服务器。

## 10. 建议调试顺序

1. 固化环境和启动参数，确认实际加载的是当前提交与目标 INT8 checkpoint。
2. 用 TP eager、BF16 KV 运行固定短请求，先确认无启动错误、输出非乱码、重复请求稳定。
3. 检查实际选择的 quant method、Dense GEMM、MoE backend 和 dispatch 类型，防止静默走 fallback。
4. 用同一输入对比 fused/fallback MoE，重点检查 `swiglu_limit`、scale shape、权重布局和 token 排序。
5. 完成精度集基线后，再按 P2 顺序逐项开启优化。
6. 出现问题时先缩小到最小变量；每次只修改一个假设，并把失败结论也写入本文。

## 11. 运行记录模板

后续每轮服务器验证复制以下模板，禁止只写“已验证”或“精度正常”。

```text
日期：
提交：
机器/容器：
模型路径与 revision：
量化配置摘要：
启动脚本及参数：
环境变量：
TP/EP/DP/CP：
KV cache dtype：
CUDA Graph / EAGLE / HiCache / LayerSplit / P-D：
测试请求或数据集：
期望结果：
实际结果：
日志路径：
结论：通过 / 失败 / 信息不足
遗留问题：
```

## 12. 接手检查清单

- [ ] `git status --short`，确认没有误带他人的工作树修改。
- [ ] 阅读本文第 4、6、7、9 节。
- [ ] 阅读待修改路径的官方实现与 `sglang-model` 参考实现。
- [ ] 确认服务器工作树、容器代码和本地提交一致。
- [ ] 确认本轮只改变一个调试变量。
- [ ] 提交前更新本文的状态、Todo、提交索引和验证记录。
- [ ] 提交时只暂存本任务文件，特别注意不要加入 `.diagnostics/`。
