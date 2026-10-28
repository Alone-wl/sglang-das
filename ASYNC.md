# sglang-model → sglang-das 同步与 Handoff

下次同步先读本文件的“续同步步骤”；逐提交结论是审核账本，源码和测试是实现依据。

## 本次快照（2026-10-28，Asia/Shanghai）

| 项目 | 固定值 |
| --- | --- |
| 源仓库 | `/Users/wanglong/Code/sglang-model` |
| 源当前分支 | `origin/sglang_glm_dev`（本地分支名） |
| 源远端 | `https://developer.sourcefind.cn/codes/maxiao1/sglang-model.git` |
| 源上界 / 下次游标 | `d3db5682d1560c4d707feb4d17a92edac1b3c91f` |
| 首个候选的父提交 | `5baf8a39e38c2f57170b9ad5f4fc4cf6d1255700` |
| 目标仓库 | `/Users/wanglong/Code/sglang-das` |
| 目标原分支 | `mx_int8/main_glm53_flash_feature_cleanup_tx` |
| 目标起点 | `1b0a352df3975baa40569aa4ff5d7caea11e1a6c` |
| 同步分支 | `sync/sglang-model-20260915-20261028` |
| 本地源快照 ref | `refs/remotes/sync-source/sglang_glm_dev` |
| 迁移策略 | 在目标当前功能分支之上线性追加；未改为 origin/main，未 merge 两仓库历史 |

范围按 **committer timestamp >= 2026-09-15T00:00:00+08:00**，包含边界当天，且仅限源上界可达提交。全历史按时间过滤核实共 26 条。作者日期早于边界但提交时间落在范围内的 76747d5ada、71a95a45af、2dc4436545、7b361c36b8 也已核对。本次使用本地快照，没有刷新网络远端，因此上界不代表远端最新状态。源 `.gitignore` 和 `.claude/worktrees/` 的未提交改动未纳入。

结论：9 个源提交全部或部分移植，4 个已有覆盖/保留目标差异，13 个因目标缺少对应路径或功能而跳过。新增 8 个实现提交、1 个验证修正提交及本文档提交；f03b2ad357 与 3d08bb73b8 合并移植，避免落下已知安全修复。

## 同步规范

1. 固定源上界、目标起点和分支后枚举候选。区分 author date 与 committer date，按源拓扑顺序分析依赖；跨仓库 SHA 不等不能当作未覆盖的证据。
2. 先找目标的同名/改名/拆分实现、调用者和测试，再判定迁移、部分迁移、已覆盖、跳过。文件路径不同不等于功能不存在；目标已主动删除的路径保持删除。
3. 适用补丁使用 `cherry-pick -x`；重构适配在提交正文记录完整 `Source-commits` 或源 SHA。依赖安全修复可以合并。追加提交保持线性，不穿插两仓库原历史。
4. 冲突按目标结构解决，保留目标独有模式、配置入口和数据格式；部分迁移逐项说明跳过内容。避免整文件采用源版本，尤其 GLM/DSA、encoder、runtime-context、KV allocator 和量化分派。
5. 每个源 SHA 在下表只有一条结论。跳过表示对应源路径未同步，不表示目标不存在同类问题；如果后续引入该功能，应重新审查相关跳过条目。
6. 区分静态/CPU/GPU/服务端验证。没有实测时不把源 commit message 中的性能或精度数字写成目标验证结论。

## 逐提交账本

以下证据路径均相对于目标 `python/sglang/srt/`，以 `kernels/` 开头的除外；测试位于 `test/registered/`。目标 SHA 指本同步分支中的提交，可配合最后的验证修正提交阅读。

| 源完整 SHA | 原主题 | 结论 | 目标提交 | 依据与适配说明 |
| --- | --- | --- | --- | --- |
| `7b10e8c737b86634fe1d07b31856bb26c9b362bc` | fix(nsa): avoid Q quantization JIT specialization by row count | 跳过 | — | 目标 GLM kpool 没有 `_quantize_q_rows`；其 Q/索引量化实现不同。保留 `layers/attention/glm5_next/kpool/kernels.py`，不恢复源 INT8 Q kernel 或已删除测试。 |
| `b878794235c9a29b3c7c604d4ce95b671cb54bda` | perf: 融合 NSA/MoE 输入准备算子并减少 H2D/D2D 搬运 | 跳过 | — | 源 `SGLANG_ENABLE_RUNTIME_FAST_PATH`、decode_input/replay_metadata/input_prepare 路径已被目标 `38cf417d89` 明确删除；不重新引入。 |
| `76747d5adab2bdb077b2c6ebf26a4b97bed90f31` | fix(epd): release prefill RDMA buffer per-part to avoid pool exhaustion | 跳过 | — | 源旧 receiver 的“克隆每个 part 后立即释放 RDMA buffer”路径不存在；目标 `encoder/receiver.py:GlmWaitingRDMARequest` 将 GPU view 生命周期绑定到 mm_inputs。直接套用会提前释放仍被模型使用的存储。 |
| `b822c05fdf33ed6eb8f88d572f776f05b218af44` | fix(disaggregation): stop unsent encoder parts after abort | 已覆盖 | 目标现有实现 | 目标 `encoder/receiver.py` 的 abort 通知、`encoder/transfer_queue.py` 的 abort registry/队列清退及 `test_encoder_abort.py` 已处理未发送任务取消和运行任务排空；保留目标实现。 |
| `71a95a45af9b1240ed5c42094c8014ba85aa8356` | feat(observability): export host mamba pool used/total slot gauges | 迁移 | 2cec40bb28 | 指标写入 `observability/metrics_collector.py`；采样位置改为 `managers/scheduler_components/metrics_reporter.py:_log_hicache_stats`，缺少 Mamba host pool 时清零。 |
| `ce99439c2e8d9e23003efd8dd96a120ae631b382` | fix(disagg): let stub-bootstrap benchmark requests enter the radix cache | 迁移 | abc104c533 | 保留目标 rebootstrap 字段及 pinned-array 导入，在 `Req` 中增加默认关闭的 fake-bootstrap radix opt-in。 |
| `4b98a3bba25d622a2fe1ebfb035f07e1246bc69f` | MTP 复用 NSA DSA init_forward_metadata | 跳过 | — | 源 NSA/Mamba metadata 的共享路径不存在。目标 GLM 使用 DSA backend 与 in/out-graph metadata，Mamba 槽位有目标池翻译；不跨 target/draft 池直接共享源 metadata 对象。 |
| `f03b2ad3574e86a44174fd590abf303b961c3c85` | feat(kda): add env-gated TileLang target_verify kernel (odd-S capable) | 迁移 | 5f9525641d | 与后续安全修复合并移植；目标入口为 `linear/kernels/kda_hcu.py:HcuKDAKernel.target_verify`，kernel 位于 `kernels/ops/attention/fla/hcu/`。 |
| `3d08bb73b8df5025f6bed3ec7731ce1aee755d66` | fix(kda): review fixes + stride/fp16 perf for the TileLang verify kernel | 迁移 | 5f9525641d | 保留 padding/stride/fp16/fallback 修复和预编译；目标开关为 pool 的 `use_hcu_kda_replayssm`，capture API 适配见 `de2b688e33`。 |
| `8fcefa0916bcb9ba1db61db5352ba94e13b8b86a` | perf(encoder): dedup the EPD decode-side multimodal preprocess | 跳过 | — | 目标没有源 content-key encoder cache/probe、`mm_content_key.py` 与对应批处理协议；目标 encoder 已拆为 runtime/preprocessor/server，保留当前协议。 |
| `2387a27403f2aff1834266b35bad0c8374f90af6` | fix(mm): port upstream canonicalize/rendezvous + content-hit guard + scheduler seed | 跳过 | — | 依赖上一条缺失的 content lookup，包括 URL canonicalize、rendezvous/content-hit grid guard；不引入孤立开关或旧 receiver/server。 |
| `2dc443654559859c9f547d27f33f654d90a82692` | feat: support w4a8 deepgemm. | 已覆盖/保留差异 | 目标 241207e321 | 目标已有 SlimQuant W4A8 DeepEP/DeepGEMM 与 HCU compressed-tensors W4A8；保留其 checkpoint 格式与 scheme 分派。源独有 `w4a8_int` compact-config 别名及旧 MoE scheme 不覆盖目标。 |
| `7b361c36b84d1c43aad4455adf2ff61d35ee3a88` | Supports both startup parameters '--enable-kda-replayssm-spec' and '--mamba-ssm-dtype float16'. | 迁移 | 072def91a7 | 参数 hook 已允许低精度并告警；移除 HCU commit kernel 和 MambaPool 中剩余 fp32-only guard，保留 FP32 g/beta 和累加。 |
| `155d946de36506c568f8feb424779674a2990409` | fix w4a8 kernel fusion issue. | 迁移 | 9fb5493552 | 按具体 scheme 公布 prequantized-input 能力，消费已有 INT8/scale；保留目标 LM-head/KV scheme 与签名过滤。通用 factory 修正见 `de2b688e33`。 |
| `a19c0ab7d760e16246150be1d78ddffc2e01496a` | perf(pd-decode): remove host syncs that stall kernel launch on batch-size change | 部分迁移 | 9ab9cc0cc9 | 移植 PREBUILT 小张量 pinned/async H2D 与 KVALLOC 默认关闭，并适配微基准路径。目标 sampling 已异步；源 `new_seq_lens_cpu`/SeqLensCpuView filter/merge 流程已被 relay 管线替代，跳过该部分。 |
| `7e29940662c6078483c6287a354ad7633468c297` | fix(pd-decode): keep a full MTP seed for fake-prefill requests | 跳过 | — | 目标 Req/FutureMap 不使用源 `mtp_topk_indices_tensor`/`mtp_topk_indices_dim` PREBUILT seed 协议，而由 spec algorithm/relay 构造输入；不添加失效字段。 |
| `8c2af29cb973c8d0a06da48f895a95a91e2d880b` | bench: interleave shared-prefix requests and validate warmup | 部分迁移 | 36933d9a4e | 默认 uniform 新数据 round-robin，warmup 取首批真实请求分波发送；保留目标 ordered/Zipf、warmup 剔除及至少一条正式请求的边界。源 auto_benchmark.py/lib 不存在，跳过其 output-length 工具改动。旧数据缓存保留原顺序。 |
| `b41bbf6680c6bf6f423c38d0d40ae1aecf1f8258` | fix(encoder): avoid decoding video pixels for metadata | 跳过 | — | 目标不存在源 metadata-only video decode endpoint/调用链；当前 `encoder/preprocessor.py` 走自己的媒体预处理与 shard 元数据流程。 |
| `b7c10b7f1aea5940eee79d29c85edc4925754694` | fix(encoder): bound CPU video decode buffers | 跳过 | — | 目标没有源 `_decode_indices_parallel`/`sample_video_sglang_sync` CPU bucket decoder，不能把源整段 GLM processor 覆盖回来。 |
| `011f5e65aa53ee8882161b04024de8dde50e547e` | fix(encoder): pin layout and tensor format for video metadata | 跳过 | — | 仅修复源 metadata-only meta/TCHW 输入，依赖 b41bbf6680 的缺失路径；目标没有该 meta 调用。 |
| `74c8e36b5e03e686c5bec431a16b972a7b40d533` | feat: add opt-in CP sharding and prefetch for MLA weights | 跳过 | — | 目标没有源 `zero3_attn_weight.py` / `distributed/attention_cp.py` 的 replicated-attention-weight sharding/prefetch 功能；保留目标 CP/GLM DSA 实现，不导入整套新权重生命周期。 |
| `2e249dcecd7835ee447fb538edc89c2fcf22de66` | perf: optimize CP attention weight prefetch with HIP IPC | 跳过 | — | 依赖上一条缺失的 ZeRO-3 attention-weight prefetch，HIP IPC 权重读取及其更新保护没有目标消费者。 |
| `edec1245bdac657da9c93db0b50485d872d7d6dc` | fix(disagg): close stale ZMQ sockets to a failed prefill on node failure | 部分迁移 | d1b554a679 | 目标 common manager 已统一 Mooncake/NIXL node-failure 并清理 receiver sockets；补充 manager socket/monitor 清理及 send-lock 排空。源 lease-based `ZMQSocketCache` 不存在，保留目标字典缓存。 |
| `bf9db3629eccee8e1c2e41099c49f21ac89ac606` | fix(glm4v): require begin token before mm placeholders in EPD prompt rebuild | 已覆盖 | 目标现有实现 | `multimodal/processors/glm4v.py:build_input_ids_with_timestamps` 已要求完整 begin + modality token 锚点；保留无 begin-ID 时的兼容分支。 |
| `aa1732a1e998e5754faef40c2ffe5b3b150c69ba` | update deepgemm. | 已覆盖 | 目标 4c095c2130 | 目标已有 `m_grouped_i8_gemm_nt_masked` 与 `marlin_i8_masked_weight`，对应工具改名为 `hcu_deepgemm_w8a8_utils.py`。 |
| `d3db5682d1560c4d707feb4d17a92edac1b3c91f` | fix(epd): bound encoder host memory across request lifetimes | 跳过 | — | 针对源旧 encoder HTTP body admission、content-hit-before-reserve、cache storage accounting 与 `/send` 生命周期；目标没有这套 admission/cache 协议。保留目标 scheduler/runtime/transfer_queue，不声称已具备源全部 host-memory 限额保护。 |

## 验证与交付边界

- `python3 test/registered/unit/test_sglang_model_sync_cpu.py`：9/9 通过。实际执行生产 AST 中的函数/方法，使用真实 CPU torch tensors；覆盖 scheme capability、避免重复量化、通用 factory、HiCache 指标、CPU staging、uniform/ordered/Zipf、发送排空后关闭 socket、capture batch sizes 与 slot pitches。
- Python 3.11 对全部 20 个变更 Python 文件做 AST parse/compile；`git diff 1b0a352df3 --check` 通过。
- 尝试完整 package 导入的量化 unittest 时，本机默认 Python 3.9 在已有 `environ.py` 的 `str | None` 注解处失败。Python 3.11 解释器存在，但该解释器没有 torch/完整 serving 依赖。portable CPU tests 避开 package import，不能代替完整模块导入验证。
- 初次交付未运行 GPU/服务测试；2026-10-28 已补充下文的 DCU 验证。迁移后完整服务测试被节点驱动故障阻塞，尚无完整 A/B 结论。
- 所有代码和文档仅本地提交，未推送、未创建 PR、未合并原分支。后续 DCU 验证已在 scnet 创建独立实验部署。

## 运行行为变化

| 项目 | 本次行为 / 保留项 |
| --- | --- |
| `SGLANG_FAKE_BOOTSTRAP_USE_RADIX` | 默认关闭；stub benchmark 可设 1。常规 warmup/health 路径仍默认不插 radix。 |
| `SGLANG_KDA_USE_TILELANG_VERIFY` | 默认关闭；HCU KDA ReplaySSM verify 可设 1，不支持的形状/布局或 host-side kernel 异常回退现有 Triton。设备端 sticky fault 不能由此恢复。 |
| KDA fp16 checkpoint | 参数 hook 的已有告警保留，HCU checkpoint/pool 的 fp32-only guard 已移除；g/beta 仍 FP32。kernel parity 已测，fp16 全服务精度尚未测。 |
| `SGLANG_KVALLOC_KERNEL` | 默认从 true 改为 false；显式设 1 仍可选择 native allocator。微基准已测，完整服务 A/B 待节点恢复。 |
| generated-shared-prefix | 新生成的默认 uniform 请求按组 round-robin；保留 ordered/Zipf；旧缓存保留旧顺序。warmup 仍从正式统计剔除，数量必须小于总请求数。 |
| 共享前缀预热约定 | uniform、G 个组可使用 2*G 条不同 suffix 的预热，请保证还剩正式请求；目标保留按组大小分波发请求。缓存复用还取决于服务端配置，应从服务日志核实。 |

## 续同步步骤

1. 在目标检查工作区和分支，先保留用户改动。读本账本后确认应继续哪个目标分支；本次分支基于 GLM 功能分支，不能直接当作 origin/main 的补丁集。
2. 读取并固定新的源 HEAD。可以从本地源只导入 Git 对象（建议 `git fetch --no-tags /Users/wanglong/Code/sglang-model HEAD:refs/remotes/sync-source/sglang_glm_dev`）；该 ref 是可移动缓存，审核游标始终用本文完整 SHA。
3. 检查 `git merge-base --is-ancestor d3db5682d1560c4d707feb4d17a92edac1b3c91f <NEW_SOURCE_SHA>`。若成立，候选为 `git rev-list --reverse --topo-order d3db5682d1560c4d707feb4d17a92edac1b3c91f..<NEW_SOURCE_SHA>`；增量按 ancestry，不再用日期漏掉回灌旧日期提交。若失败，说明源历史重写，先用 patch-id、主题和实际 diff 对账，重新建立映射。
4. 新建同步分支并逐条检查依赖/目标对应功能。参照上表处理改名：源 `fla/*` → `kernels/ops/attention/fla/hcu/*`，GLM NSA → GLM DSA，encoder 单文件 → `disaggregation/encoder/*`，metrics mixin → `scheduler_components/metrics_reporter.py`，allocator 单文件 → `allocator/paged.py`。这些是本次核对结果，下次仍须以源码为准。
5. 添加新批次表，记录新的源上界、目标起点、每个源 SHA 的结论和验证。仅当所有候选有结论后推进“下次游标”；本次跳过条目随账本保留，不因推进游标而遗忘。
6. 重跑 portable tests、静态 compile/diff 检查和适用完整 package tests。涉及 KDA、量化或分布式变化时，在兼容 HCU 环境运行 GPU/服务集成后再给出部署结论。

## HCU 验证项目与覆盖边界（执行进度见下文）

- 登录 `ssh scnet`，用 `squeue` 选择用户计算节点，默认容器 `dev-wl`；实验脚本/日志统一放容器 `/home/work/agent/`。保持参考启动脚本中的功能。
- 用匹配目标依赖的 Python >=3.10 执行 `test_compressed_tensors_prequantized_input.py`，再执行 `test/registered/attention/test_kda_tilelang_verify.py`，覆盖 S=2..8、padding、fp16、strided pool/ring、真实 HcuKDAKernel dispatch；对照关闭/开启 TileLang，检查回退日志和图捕获。
- 对 fp16 ReplaySSM 运行长序列/实际服务精度：容器内激活 `~/evalscope` 对应环境，使用 `/home/work/eval.sh`，默认 AIME25 与 HumanEval 可并行；给出实际分数与 badcase 截断/复读/事实错误归因，证据不足时标明未知。
- 检查 PD PREBUILT merge、fake-bootstrap radix opt-in 以及 prefill 节点失效时 socket/monitor 清理，尤其在发送进行中触发失效；portable tests 只验证局部锁生命周期。
- 性能评估按项目规则先抓算子 profile：`run_single_bench_d.sh` 发压，任意 rank 的 `#running-req` 达最大并发后等 30 秒，再执行 `/home/work/profile.sh`。本次已执行 before profile 与 allocator 微基准，after 待节点恢复。

## DCU 验证进度与恢复 Handoff（2026-10-28 22:36 CST）

**状态：迁移前全量精度及三轮性能已完成；迁移后启动受 f14r1n15 驱动锁阻塞，完整 A/B 尚未完成。不能把局部 kernel 收益表述为整服务收益。**

### 固定实验与公共修复

- 容器 `dev-wl:/home/work/agent/model-sync-ab-20261028/`；宿主机 `~/agent/model-sync-ab-20261028/`。脚本、快照、依赖、manifest、日志、profiles、results 全部保留。
- before 为 `1b0a352df3975baa40569aa4ff5d7caea11e1a6c`；after 为 `65c61d81cb38d0704c608a92871ca3d24559cce9`。两侧同加 descriptor 异步 staging 修复，已提交 `60f425ec3bd38bc338dea1ce181c178c5b24cb2a`。after runtime 等价于该提交；不能称为完全未修改的原始基线对照。旧同步拷贝会等待在途集合通信；修复保留 fused clear/copy，GPU 数据位一致和异步行为均有回归证据。
- P=f14r1n14（CP8/TP8/EP8、INT8）；D=f14r1n15–16（TP16/DP16/EP16、GLM-5.3-Flash-Int4-ZP-format、SlimQuant W4A8）；client=a12r4n13。各节点8张 BW gfx936 DCU，约64 GiB/卡。
- 保留 DeepEP、MTP EAGLE（5 steps/6 draft tokens）、ReplaySSM、CUDA graph、HiCache（layer_first/kernel/write_through）和 CP。主比较 float32 checkpoint、TileLang verify 关闭。文字请求不覆盖 encoder 多模态路径。
- 两侧同用 `deps-tx/`（Transformers 5.12.1、原厂 sgl_kernel 二进制和同签名 HCU/DCU Python aliases）。HiCache D2H/H2D 的 `_hcu` 名称映射同签名 `_dcu` kernel，D2H 位一致已测；未关闭 HiCache。模型配置/tokenizer/index 哈希与权重文件大小/时间见 manifest。
- 性能客户端固定 before 代码。before 显式 allocator=1 等价于旧默认；after 不设置 `SGLANG_KVALLOC_KERNEL`，采用新默认 false/Triton，属于待比较的默认行为变化。

### 已完成的数据

| 项目 | 迁移前 | 迁移后 |
| --- | --- | --- |
| AIME25 | 27/30，90% | 尚未运行 |
| HumanEval | 163/164，99.39% | 尚未运行 |
| D-only 平均输出吞吐，3轮均值 | 4570.32 token/s | 尚未运行 |
| D-only mean TPOT，3轮均值 | 14.43 ms | 尚未运行 |
| 满载后等待30s的 GPU/CPU profile | 16 rank trace 已采集且可解析 | 尚未采集 |

精度使用 evalscope 和 `/home/work/eval.sh` 实验副本，两数据集并行、batch64、seed42、temperature1、top_p0.95、max_tokens64000、stream=true、enable_thinking=false，HumanEval review_timeout20s。真实 PD/HiCache 16并发门禁通过后运行，不复用预测缓存。AIME25 index12/13/29 与 HumanEval/130 均到64000 tokens且最终答案为空，直接归因为截断/没有最终答案。HumanEval/130 同一代码行重复51次，存在反复推演；AIME25/12反复核对，13/29仍在推导/验证，不能一律归为机械复读，也未证明量化导致错误。原始 reviews/predictions 和各数据集 `analysis.json` 保留证据。

性能为 D-only fake-prefill raw IDs，input81920/shared-prefix79872/output1024/concurrency128，每轮384正式请求，独立预热，共3轮。每轮成功384、生成393216 tokens；平均输出吞吐3798.70/3910.89/6001.37 token/s，mean TPOT13.95/14.33/15.00ms。吞吐波动明显，应同时报告各轮与范围。profile rank0 热点包含 DeepEP、W4A8 GEMM、HCU KDA recurrent、sparse attention；16 rank 汇总见 `results/before/profile-summary.json`。

局部验证：portable CPU9/9、完整 package量化接口3/3、TileLang GPU8/8、descriptor GPU回归2/2通过；KV allocator22/22数据一致。fp16 checkpoint相对FP32累加后转half maxAbs=0.0001220703125，未测fp16全服务精度。TileLang CUDA graph微基准（S6/HV64/K=V128/FP32，N1/4/8）1.63/1.66/1.75倍，但主服务未启用。长extend100000 tokens native/Triton为28.24/0.103ms；短extend BS16/12 tokens为0.037/0.074ms，短调用存在开销权衡。

首轮缺少HiCache alias的评测0个有效样本，已归档 `*-failed-hicache`，不纳入结果。before三轮API压测完整，但运行期间修正提取器导致末尾wrapper exit2；已逐日志验证成功数、tokens及duration/tps一致后保留测量。提取器改为行首精确匹配，避免把Peak当平均吞吐，summary/CSV已重建。

### 当前阻塞与续跑

1. 控制器已终止。f14r1n15旧进程在 `hycu` 驱动 `evict_process_queues_cpsch` mutex 等待中处于D状态，新Python初始化也被阻塞。SIGTERM/SIGKILL不能回收；`hy-smi --hcureset -d 0` 返回 NOT_SUPPORTED；privileged容器挂载debugfs后读取各DCU `gpu_recover`，内核仅记录“HCU reset begin”，等待未解除。未重启宿主机、未删除容器。证据：`logs/after-d0-driver-hang.txt`、`logs/d0-reset-gpu0.log`。
2. 需要先恢复/重启 f14r1n15 宿主机；目前只有停止GPU进程/容器的明确授权，尚未取得宿主机重启授权。检查 `squeue` 分配、dev-wl、GPU小张量同步及无旧D态进程后继续。
3. P14迁移后进程保留，D1已停，D0新旧进程处于驱动等待。先确认P健康；若不健康，使用 `VARIANT=after bash /home/work/agent/model-sync-ab-20261028/scripts/p.sh` 启动并写独立日志。
4. `transport/controller_resume_after_recovery.py` 保存续跑控制器（未执行）：启动after D0/D1→等待P/D健康→满载profile→重启D隔离profile→16并发真实PD门禁→两个数据集精度→停P→三轮D性能。运行前检查已有状态/数据避免重复覆盖，在登录节点nohup运行并保存日志。若profile后驱动再次不能回收，需要设备恢复，不能关闭参考功能绕过。
5. 完成后容器执行 `transport/analyze_profile.py after`、`transport/collect_accuracy.py`；登录节点执行 `transport/collect_performance.py`。核对两侧194个prompt_key一致，比较逐题通过/失败转换及真实输出。profile需核对batch和kernel形状后比较，不仅按总GPU时间推因果。
6. 更新本文并补齐完整A/B报告；说明公共修复/依赖覆盖、默认allocator变化、temperature1单次评测随机性和吞吐波动。当前没有after结果，不能用历史分数或微基准代替。

### 用户授权替代拓扑（2026-10-28 23:20 CST，正在执行）

用户明确允许关闭 E 服务，将 a12r4n13 作为 f14r1n15 的替代节点；已保存 E 的原命令/环境并停止，未删除容器。P14、a12、D1的GPU小张量及同步检查均通过。新拓扑 P=f14r1n14、D0=a12r4n13、D1=f14r1n16，客户端仍在a12运行；相同硬件型号但网络拓扑改变，必须在该拓扑重跑before/after，不能与原基线混算收益。

新目录：`dev-wl:/home/work/agent/model-sync-ab-a12d0-20261028/`，宿主机同名 `~/agent/` 子目录；固定代码与依赖以符号链接复用旧目录，scripts、results、profiles、logs、manifest独立。控制器 `transport/controller_topology2.py`，状态 `manifest/controller-status.json`，日志 `logs/controller.log`。保留主比较配置，仅将D0地址换为10.1.12.73；文字评测不需要运行encoder。原异常节点不再使用，也不重启。

2026-10-29 00:23：新拓扑 before 已完成，16 rank profile解析无错、16并发真实PD/HiCache门禁通过、两个精度job和三轮性能wrapper均exit0。AIME25=25/30（83.33%），HumanEval=163/164（99.39%）。AIME25 index12/14/17到64000 tokens且无最终答案；index18望远镜乘积末项错误（215而非106），index29将极小值条件误解后给出不存在有限和（标准答案240）；HumanEval/116输出完整但将负数优先排序，未通过测试，52642 tokens未达上限，不是截断或超时。这些新拓扑单次样本与旧拓扑不同，不能用旧分数代替本轮基线。

新拓扑 before 三轮吞吐5971.46/4452.98/5435.07 token/s（均值5286.50、样本标准差770.06），mean TPOT14.09/14.55/14.81ms（均值14.48）；每轮成功384、生成393216 tokens。mean TTFT均值1139.19ms，p99 TTFT三轮均值7734.78ms、p99 TPOT30.74ms、p99 ITL67.06ms。结果见新目录 `results/performance-summary.json` 和 `accuracy-summary.json`；after已开始加载，尚无结果。

2026-10-29 00:29：切换after时，替代节点a12r4n13也复现驱动锁：before D0进程结束后，data_parallel/detokenizer/scheduler处于D状态，`evict_process_queues_cpsch` 等待；新Python处于 `gpu_gpukfd_evict_userptr`，未进入after服务初始化。故障不只发生于f14r1n15，且本次发生在before退出阶段，不能归因为after迁移实现。`hy-smi --hcureset -d 0` 同样返回 NOT_SUPPORTED。控制器已终止，after D0/D1已发送停止信号，P14保留；没有宿主机重启或删除容器。

新目录 `logs/after-d0-driver-hang.txt`、`logs/d0-reset-gpu0.log` 保存证据，`manifest/controller-status.json` 为blocked-driver。E停止前的命令和环境在原目录 `manifest/stopped-a12r4n13-*.json`；目前a12驱动阻塞，E也无法恢复。需要恢复/重启a12宿主机或提供额外健康节点。恢复同一拓扑后，核对P健康与GPU同步，再执行新目录 `transport/controller_resume_after.py`（只续after，未执行）；节点再次改变则不能直接混用本次before。注意profile后的D退出/重启也需检查驱动状态，若再次锁住应优先诊断退出路径，不得删减参考功能掩盖故障。当前没有after分数、正式吞吐或profile，不可宣称无精度损失或服务性能收益。
