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

## Known unresolved issues

1. Greedy target output is incoherent on simple prompts. Because EAGLE verifies
   draft proposals against the target, the 0% draft acceptance is evidence of a
   mismatch but does not by itself identify whether the target path, draft path,
   or both are wrong.
2. The first request after model load can spend several minutes in lazy
   compilation. This must not be mistaken for a scheduler deadlock; subsequent
   health requests are fast.
3. Accuracy has not yet been validated against a trusted reference response or
   evaluation set.
4. The supplied `/home/work/glm/ifb.sh` warmup path has not yet completed in this
   session; the inherited server used `--skip-server-warmup`.
