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
- 未运行 HCU/Triton/TileLang kernel parity、CUDA-graph capture、PD/EPD 集成、AIME25/HumanEval 或目标性能测试。没有本次目标服务的精度/收益结论。
- 所有代码和文档仅本地提交，未推送、未创建 PR、未合并原分支；本次没有部署或修改远端运行环境。

## 运行行为变化

| 项目 | 本次行为 / 保留项 |
| --- | --- |
| `SGLANG_FAKE_BOOTSTRAP_USE_RADIX` | 默认关闭；stub benchmark 可设 1。常规 warmup/health 路径仍默认不插 radix。 |
| `SGLANG_KDA_USE_TILELANG_VERIFY` | 默认关闭；HCU KDA ReplaySSM verify 可设 1，不支持的形状/布局或 host-side kernel 异常回退现有 Triton。设备端 sticky fault 不能由此恢复。 |
| KDA fp16 checkpoint | 参数 hook 的已有告警保留，HCU checkpoint/pool 的 fp32-only guard 已移除；g/beta 仍 FP32，精度待目标 HCU 实测。 |
| `SGLANG_KVALLOC_KERNEL` | 默认从 true 改为 false；显式设 1 仍可选择 native allocator。没有目标性能收益实测。 |
| generated-shared-prefix | 新生成的默认 uniform 请求按组 round-robin；保留 ordered/Zipf；旧缓存保留旧顺序。warmup 仍从正式统计剔除，数量必须小于总请求数。 |
| 共享前缀预热约定 | uniform、G 个组可使用 2*G 条不同 suffix 的预热，请保证还剩正式请求；目标保留按组大小分波发请求。缓存复用还取决于服务端配置，应从服务日志核实。 |

## 续同步步骤

1. 在目标检查工作区和分支，先保留用户改动。读本账本后确认应继续哪个目标分支；本次分支基于 GLM 功能分支，不能直接当作 origin/main 的补丁集。
2. 读取并固定新的源 HEAD。可以从本地源只导入 Git 对象（建议 `git fetch --no-tags /Users/wanglong/Code/sglang-model HEAD:refs/remotes/sync-source/sglang_glm_dev`）；该 ref 是可移动缓存，审核游标始终用本文完整 SHA。
3. 检查 `git merge-base --is-ancestor d3db5682d1560c4d707feb4d17a92edac1b3c91f <NEW_SOURCE_SHA>`。若成立，候选为 `git rev-list --reverse --topo-order d3db5682d1560c4d707feb4d17a92edac1b3c91f..<NEW_SOURCE_SHA>`；增量按 ancestry，不再用日期漏掉回灌旧日期提交。若失败，说明源历史重写，先用 patch-id、主题和实际 diff 对账，重新建立映射。
4. 新建同步分支并逐条检查依赖/目标对应功能。参照上表处理改名：源 `fla/*` → `kernels/ops/attention/fla/hcu/*`，GLM NSA → GLM DSA，encoder 单文件 → `disaggregation/encoder/*`，metrics mixin → `scheduler_components/metrics_reporter.py`，allocator 单文件 → `allocator/paged.py`。这些是本次核对结果，下次仍须以源码为准。
5. 添加新批次表，记录新的源上界、目标起点、每个源 SHA 的结论和验证。仅当所有候选有结论后推进“下次游标”；本次跳过条目随账本保留，不因推进游标而遗忘。
6. 重跑 portable tests、静态 compile/diff 检查和适用完整 package tests。涉及 KDA、量化或分布式变化时，在兼容 HCU 环境运行 GPU/服务集成后再给出部署结论。

## HCU 验证 Handoff（尚未执行）

- 登录 `ssh scnet`，用 `squeue` 选择用户计算节点，默认容器 `dev-wl`；实验脚本/日志统一放容器 `/home/work/agent/`。保持参考启动脚本中的功能。
- 用匹配目标依赖的 Python >=3.10 执行 `test_compressed_tensors_prequantized_input.py`，再执行 `test/registered/attention/test_kda_tilelang_verify.py`，覆盖 S=2..8、padding、fp16、strided pool/ring、真实 HcuKDAKernel dispatch；对照关闭/开启 TileLang，检查回退日志和图捕获。
- 对 fp16 ReplaySSM 运行长序列/实际服务精度：容器内激活 `~/evalscope` 对应环境，使用 `/home/work/eval.sh`，默认 AIME25 与 HumanEval 可并行；给出实际分数与 badcase 截断/复读/事实错误归因，证据不足时标明未知。
- 检查 PD PREBUILT merge、fake-bootstrap radix opt-in 以及 prefill 节点失效时 socket/monitor 清理，尤其在发送进行中触发失效；portable tests 只验证局部锁生命周期。
- 若后续要求性能评估，先按项目规则抓算子 profile：`run_single_bench_d.sh` 发压，任意 rank 的 `#running-req` 达最大并发后等 30 秒，再执行 `/home/work/profile.sh`。微基准 `benchmark/kernels/kvalloc/bench_kvalloc.py` 已适配目标导入路径，但尚未运行。
