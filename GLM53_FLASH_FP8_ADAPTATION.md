# GLM-5.3-Flash-FP8 HCU adaptation log

## Original request and operating constraints

- Target: adapt the `glm5.3-flash-fp8` model until the supplied launch path is usable.
- Remote host: `ssh nmz26`.
- Runtime container: `dev-wl-glm2`.
- Launch script: `/home/work/glm/ifb.sh`.
- All source reading and edits must be performed in the local checkout at
  `/Users/wanglong/Code/sglang-das`.
- After each source fix, commit locally with a detailed commit message, push the
  branch, then enter the remote container and pull the new commit.
- For non-kernel problems, compare the official baseline at
  `/Users/wanglong/Code/sglang` before designing a fix.
- For unsupported or incompatible operators, first reuse the HCU/DCU GLM
  implementation in `/Users/wanglong/Code/sglang-model`; do not write a new
  operator when an established implementation exists.
- Keep this document current with every reproduced problem, diagnosis, fix,
  validation result, corresponding commit, and known unresolved issue so that a
  different engineer can resume without reconstructing the history.

## Environment and branch

- Local branch: `glm5.3-flash`.
- Push remote/branch: `wl/glm5.3-flash`.
- Remote source checkout: `/home/work/code/sglang-das`.
- Remote model: `/home/work/GLM-5.3-Flash-Channel-FP8-w8a8`.
- Hardware visible to the container: 8 HCU devices (`gfx938`).
- Main runtime configuration: TP=8, EP=8, DeepEP normal mode, DSA/NSA,
  FP8-E4M3 KV cache, hierarchical cache, hybrid Mamba extra buffer, and EAGLE
  speculative decoding.

## Inherited work before this debugging session

The branch already contained the upstream GLM-5.3-Flash import and a sequence of
compatibility restores. The most relevant commits immediately preceding this
session are:

| Commit | Purpose |
| --- | --- |
| `979baf5a81` | Import upstream GLM-5.3-Flash support. |
| `0258c9987e` | Restore HCU DSA index-cache compatibility. |
| `cfcfc6d93d` | Restore GLM FP8 no-RoPE quantization helper. |
| `f5fd9d01a3` | Port the GLM DSA K-pool FP8 index path. |
| `1d85638859` | Restore DSA top-k backend resolution. |
| `f076f02e3b` | Make the HCU KDA fused operator import optional. |
| `8bdf7af339` | Complete GLM MTP linear-attention compatibility. |
| `2b74702266` | Normalize the Mamba hierarchical-cache layout. |
| `83cc033742` | Add the SWA branching match-result field. |
| `3514a56c13` | Default the DSA K-pool page size. |
| `69bfa861a1` | Permit page size 1 in the HCU DSA K-pool. |
| `1d79fe4681` | Remap CUDA-only FlashMLA backend names on HCU. |

At session start, 20 tracked files also contained an uncommitted compatibility
checkpoint (963 insertions and 271 deletions). It covers HCU DSA K-pool planning
and FP8 index storage, HCU DSA backend metadata and buffers, KDA/Mamba/MHC
compatibility, DeepEP channel-FP8 MoE weight preparation, scheduler KV/Mamba
bookkeeping, logits chunking, and hierarchical-cache allocation. These changes
were present in both the local and remote checkouts and were preserved intact.

## Debugging chronology

### 2026-09-12: establish the real runtime state

1. Inspected the local branch, worktree, recent commits, remote configuration,
   container launch script, and the actual `PYTHONPATH` from `common.sh`.
2. Found an already running diagnostic server on port 18082. It used the same
   model and source checkout as `ifb.sh`, with `--skip-server-warmup` substituted
   for the configured warmup.
3. An initial `/health` request appeared to hang. `py-spy` showed scheduler
   ranks in HiCache synchronization and MLP-sync collectives. A later request
   and timestamps established that this was not a permanent collective
   deadlock: the first one-token request spent approximately 3 minutes 47
   seconds compiling/initializing, after which `/health` returned HTTP 200.
4. Sent deterministic generation smoke tests through `/generate`:
   - Prompt `Hello`, greedy, 8 output tokens: request completed, but output was
     incoherent; EAGLE accepted no draft tokens.
   - Prompt `1+1=`, greedy, 16 output tokens: request completed, but output was
     incoherent; EAGLE acceptance remained 0%.
5. Conclusion: startup and request plumbing now work, but numerical correctness
   is not yet established. The next isolation step is to compare target-only
   generation and operator-path variants, beginning with the existing DCU GLM
   implementations for MoE, DSA indexing/attention, and Mamba.

Validation of the inherited checkpoint before committing it:

- `python3 -m compileall` passed for every modified Python file.
- `git diff --check` passed.
- The 8-device service loaded the model and returned HTTP 200 from `/health`.
- End-to-end generation completed, but the text was numerically incorrect.

Corresponding checkpoint commit: this document's initial checkpoint commit
(`wip(hcu): checkpoint GLM-5.3 Flash FP8 runtime adaptation`).

### 2026-09-12 (session two): target-only isolation

To decide whether the earlier incoherent output originates in the target
path or the EAGLE draft path, the launcher was copied and stripped of the
four `--speculative-*` flags into `/home/work/glm/target_only.sh`. The
env is identical to `ifb.sh` (`SGLANG_USE_DEEPGEMM_MOE=1`, aiter/hip
FlashMLA remap, DeepEP normal on eight ranks, FP8 KV cache).

Two runs were captured:

1. `target_only_v2.log` first attempt sent a single-token prompt (`Hello`).
   All eight TP ranks stalled inside `causal_conv1d_fn` at
   `python/sglang/srt/layers/attention/linear/kda_backend.py`
   for the full 300-second scheduler watchdog. `causal_conv1d_fn` is a
   Triton kernel with a hard-coded convolution width of four, and a
   1-token prefill is a corner case where the queried window is shorter
   than the kernel. The scheduler debug at kill time reported
   `leaked_mamba_pages={2, 3, 4}`; the mamba pool held the request but
   forward never returned.
2. A rerun with multi-token prompts (`"The quick brown fox …"`, etc.) hung
   identically, this time in the first MoE layer: all eight ranks blocked
   in `torch.distributed.all_gather_object` inside DeepEP's `Buffer.__init__`
   (`python/sglang/srt/layers/moe/token_dispatcher/deepep.py:379`).
   The MainThread native frames were only `libc.so.6`; every rank was
   stuck in the same collective, not diverging. NCCL logs report the
   NUMA-balancing warning and the missing `iommu=pt` boot flag but no
   fatal error before the hang.

Both hangs happen on the first prefill and both start from the same
pretrained-weight state, so they are unlikely to be caused by
per-request state divergence. The plausible root causes are:

- A stale DeepEP shared-memory / semaphore layout left by earlier kills
  that the new run inherits (`/dev/shm/sem.mp-*` was populated). Cleaning
  those between runs is now a required step for the isolation harness.
- A collective-timing dependency where the first DeepEP `Buffer` init
  races the HCU custom-allreduce fence and one rank temporarily leaves
  the collective, blocking `all_gather_object` forever. `common.sh`
  disables the DCU custom allreduce (`USE_DCU_CUSTOM_ALLREDUCE=0`),
  but `parallel_state.py` still defaults to disabling pynccl on HCU
  (`SGLANG_HCU_DISABLE_PYNCCL=true`), routing collectives through NCCL /
  Gloo instead. The earlier successful `target_only.log` run at 11:11
  had run once and then died at 11:24; subsequent starts have never
  reached generation.

Neither corresponds to a numerical bug in the target model. Until a
prefill returns, target-vs-draft correctness cannot be isolated at all,
so the immediate priority is unblocking the first prefill on the
current commit rather than chasing dequant paths.

Candidate divergences flagged from source-only review (all still
unverified — none reproduce standalone without the server):

- `_forward_aiter_torch_fallback` in
  `python/sglang/srt/layers/attention/dsa_backend.py:3140` skips a KV
  scale when casting FP8 KV to bfloat16. On the primary MLA path
  (`layer.head_dim != layer.v_head_dim`) this is used unconditionally
  whenever `aiter.mla_decode_stage1_asm_fwd` is missing.
- `communicator_mhc.attn_to_mlp` zero-pads hidden_states to residual's
  batch size before `hc_post`, papering over a shape mismatch whose
  root cause is not documented (see
  `python/sglang/srt/layers/communicator_mhc.py:94`).
- KDA `forward_target_verify` pads `core_attn_out` with zeros to
  `physical_seq_len`, which sends zero-attention outputs into the sampler
  for the padded positions (see `linear/kda_backend.py:1067`).

### 2026-09-12 (session two, later): first prefill unblocked

The DeepEP `Buffer.__init__` `all_gather_object` hang and the KDA
`causal_conv1d_fn` hang both cleared after clearing the stale sglang
shared-memory state that earlier crashed sglang runs had left behind:

```
rm -f /dev/shm/sglang_loads_*.shm \
      /dev/shm/sgl_shm_mm_* \
      /dev/shm/sgl_shm_mq_* \
      /dev/shm/sem.mp-* \
      /dev/shm/torch_*
```

A fresh `target_only.sh` after that cleanup reached
"fired up and ready to roll" and returned three deterministic /generate
responses back-to-back on port 8080. Because the token IDs match on
repeated runs of the same prompts, the pipeline is deterministic —
subsequent numerical work can rely on that.

The observed outputs (greedy, temperature=0, top_k=1):

```
'The quick brown fox jumps over the lazy' -> '漂浮umo phen率umuwesen禹 SF'
'1 + 1 = 2. 2 + 2 ='                       -> '-minRONinisubstLOTSiple意ortun'
'Once upon a time in a'                    -> ' secondary âanolick inquire goose珍aniwargsSubsetikhbih'
```

So there is definitely a numerical bug on the target path, independent
of EAGLE: this is target-only. The suspects flagged from source review
(aiter torch MLA fallback missing KV scale, MHC `attn_to_mlp` zero pad,
KDA `target_verify` zero pad) can now be tested for real.

Operational rule for the isolation harness going forward: **always
clean `/dev/shm` before starting a new sglang run**; skipping this step
is what caused the earlier "target model hangs on first prefill"
symptom that was misread as a KDA / DeepEP correctness bug.

### 2026-09-12 (session two, still later): numerical hypotheses ruled out

Ran the target-only launcher with `--kv-cache-dtype fp8_e4m3` in a stable
loop (post-shm-cleanup) against three fixed multi-token prompts. The
output is deterministic and identical run-to-run:

```
'The quick brown fox jumps over the lazy' -> '漂浮umo phen率umuwesen禹 SF'
'1 + 1 = 2. 2 + 2 ='                       -> '-minRONinisubstLOTSiple意ortun'
'Once upon a time in a'                    -> ' secondary âanolick inquire goose珍aniwargsSubsetikhbih'
```

Verified against sources; ruled out:

- **`_forward_aiter_torch_fallback` KV scale.** For GLM-5.3-Flash
  `qk_rope_head_dim == 0`, so `attn_mqa.head_dim == kv_lora_rank == 512
  == v_head_dim`. That takes the `triton_sparse_mla_fwd` branch, not
  the einsum loop. The kernel decodes FP8 via `.to(bfloat16)` on the
  raw `float8_e4m3fn`-view KV buffer, which is arithmetically correct
  because `set_mla_kv_buffer_triton_fp8_quant` writes the raw MLA KV
  layout without per-block scales and stores as uint8 aliased to
  `torch.float8_e4m3fn` via `store_dtype`.
- **`_forward_aiter_torch_fallback` head-count MFMA.** `need_pad_heads`
  fires (`num_q_heads = 8 < 16`) and `repeat_interleave` duplicates
  each head so the kernel sees `H=16`; the output stride `[:, ::factor,
  :]` picks the correct duplicate.
- **Tilelang MHC pre.** Rerun with `SGLANG_OPT_USE_TILELANG_MHC_PRE=0`
  produced bit-identical garbled output. (`SGLANG_OPT_USE_TILELANG_MHC_POST=0`
  was not honored — `envs.SGLANG_OPT_USE_TILELANG_MHC_POST.set(True)` runs
  somewhere later in model_hook or server_args; that specific flag is not
  disable-able from the launcher alone. Model-hook line 400 only fires
  for DeepseekV4, not `GlmMoeDsaForCausalLM`.)
- **MHC hc_pre/hc_post kernels.** `sglang-das` diff vs upstream in
  `mhc.py` is a cosmetic refactor; the tilelang and torch dispatches
  are semantically identical.
- **DeepGEMM channel-FP8 MoE weight prep.** The DCU reference at
  `/Users/wanglong/Code/sglang-model` uses the same
  `pack_int8_weight_enk_to_w6_low_latency` packer against FP8 weights
  in `_prepare_dsv4_channel_fp8_deepgemm_weights` and feeds the same
  `m_grouped_fp8_gemm_nt_contiguous`. Packer-name mismatch is not the
  bug.
- **kpool_bf16_paged_mqa_logits kernel.** The FP8 K decode path (bit
  extraction, subnormal handling, NaN sentinel) matches the E4M3FN
  spec; the K side applies its per-slot scale; the Q side folds
  `q_scale` into `weights` via `_get_logits_head_gate`, so the
  algebraic identity `max(q_r·k_r, 0)·w = max(q_fp8·k_fp8, 0)·(w·q_s·k_s)`
  holds.
- **FP8 KV write kernel `set_mla_kv_buffer_fp8_quant_kernel`.** Handles
  the `rope_dim == 0` case (GLM-5.3-Flash's layout) by taking the
  `base + BLOCK <= nope_dim` early branch; the BF16→FP8 downcast is
  done via a typed `tl.store` with no manual scaling, matching how the
  fallback reads it back.
- **bf16 KV variant.** Falls over on the DSA indexer with
  `AssertionError: Scaled index K cache is not enabled` in
  `memory_pool.py:5089` -- the indexer read path assumes the scaled
  index K path but the pool only allocates it in FP8 mode. This is a
  separate correctness bug in the bf16 KV path, not a workaround for
  the FP8 accuracy problem.
- **`_forward_tilelang`.** Crashes in `libtilelang.so`
  `GemmNode::InferLayout` → `make_hcu_swizzled_layout` for the
  gemm_hcu_mmac shape (D_V=512, H=8-or-16). Tilelang DSA prefill/decode
  is not viable on gfx938 at this shape.
- **`SGLANG_USE_DEEPGEMM_MOE=0` variant.** Hard-crashes at the DeepEP
  dispatch: `Dispatch output is not supported: quant_config=...
  scheme=CompressedTensorsW8A8Fp8MoE, use_fp8_w8a8=False,
  has_deepgemm_weights=False`. `deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM`
  is False on HCU, so this scheme has no working dispatch branch
  besides `SGLANG_USE_DEEPGEMM_MOE=1`.
- **`forward_mla_rocm.py` drift.** `md5 diff` vs upstream shows two
  attribute-access diffs only (`get_parallel().dcp_replicate_q_proj`
  vs `get_parallel().config.dcp_replicate_q_proj`); no semantic
  divergence.

Suspects that remain unverified but still possible:

- **Aiter DeepEP normal-mode dispatch payload shape / stride.** The
  scheme's forward path is `deepep dispatch → forward_impl → forward_groupgemm_w8a8_fp8_contiguous`;
  we compared the das and DCU-reference forward_groupgemm bodies and
  they match, but the `pack_int8_weight_enk_to_w6_low_latency` packer
  is called on **compressed-tensors channel-FP8** weights (per token
  `strategy: token`, per channel `strategy: channel`). Whether the
  packer really preserves FP8 magnitude for these strategies on HCU
  gfx938 wasn't validated at the tensor level — a smoke test that
  dumps `w13_weight_deepgemm.dequantize()` vs the original per-channel
  bf16 weight for one expert would settle this.
- **Compressed-tensors kv_cache_scheme is null.** But the launcher
  forces `--kv-cache-dtype fp8_e4m3` regardless. If the model
  checkpoint expects a different KV quantization convention than
  `mla_quantize_for_fp8_no_rope`'s plain `.to(float8_e4m3fn)` cast,
  that would look exactly like this bug. No public description of the
  intended KV quantization was found in the checkpoint's
  `quantization_config`.
- **DSA indexer `weights_proj` path.** `_get_logits_head_gate` uses
  `self.weights_proj(x.float())`. If `weights_proj` weights are
  channel-FP8 quantized, casting `x` to float in Python and doing the
  linear in FP32 while the weights are FP8 (via bf16 upcast) may work,
  but validate that this path exists on HCU.
- **Deep model divergence points not yet inspected:**
  `Glm5NextForConditionalGeneration.__init__` weight-loader mapping,
  the `fused_qkvbfg_a_proj` packed slice ordering, the MTP eh_proj /
  enorm / hnorm paths (excluded from quantization per the config
  `ignore` list — need to confirm they are loaded in bf16).

Operational notes:

- Always clean `/dev/shm` (`sglang_loads_*.shm`, `sgl_shm_mm_*`,
  `sgl_shm_mq_*`, `sem.mp-*`, `torch_*`) before starting a new run.
- `SGLANG_OPT_USE_TILELANG_MHC_POST` cannot be disabled from the
  launcher; toggling it requires a code change in `environ.py` or
  `arg_groups/model_hook.py`.
- SSH goes through a 2FA `zz_jump_wl` ProxyJump. Do not retry after a
  failed attempt (the jump blacklists brute force); when the
  ControlMaster socket at `~/.ssh/control-wanglong3@42.228.13.241:65024`
  is gone, ask the user to `ssh nmz26` once from their terminal to
  rebuild it.

## Known unresolved issues

1. Greedy target output is deterministically incoherent on simple
   prompts. The list of possible root causes has narrowed but no
   candidate has been confirmed. Highest-yield next probes: dump
   `w13_weight_deepgemm.dequantize()` for one expert and compare
   against the raw channel-FP8 weight; then dump the FP8 KV buffer
   after the first write, cast back to bf16, and compare against
   `(cache_k_nope || cache_k_rope).to(bf16)`.
2. `--kv-cache-dtype bfloat16` cannot currently launch: the DSA
   indexer asserts `use_scaled_index_k_cache` unconditionally.
3. The first request after model load can spend several minutes in
   lazy compilation. This must not be mistaken for a scheduler
   deadlock; subsequent requests are fast.
4. Accuracy has not yet been validated against a trusted reference
   response or evaluation set.
5. The supplied `/home/work/glm/ifb.sh` warmup path has not yet
   completed in this session; the inherited server used
   `--skip-server-warmup`.
