"""HCU GLM layer-pipelined prefill within the current scheduler lifecycle."""

from __future__ import annotations

import logging
import sys
from functools import partial

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.layers.attention.glm5_next.runtime import (
    get_glm5_next_runtime_args as get_global_server_args,
)
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hcu
from sglang.srt.utils.common import print_info_once, print_warning_once

logger = logging.getLogger(__name__)


def get_pipeline_forward(scheduler, batch, default_forward):
    if (
        not envs.SGLANG_PIPELINED_KV_TRANSFER.get()
        or not is_hcu()
        or scheduler.disaggregation_mode != DisaggregationMode.PREFILL
    ):
        return default_forward
    group_size = _get_pipeline_group_size(scheduler, batch)
    return (
        partial(run_layer_pipeline, scheduler, group_size)
        if group_size
        else default_forward
    )


def _get_pipeline_group_size(self, batch) -> int:
    """Return a GLM layer group size, or zero for the exact mainline path."""
    if not envs.SGLANG_PIPELINED_KV_TRANSFER.get():
        return 0
    args = get_global_server_args()
    if not batch.forward_mode.is_extend() or not batch.reqs:
        return 0
    if any(mm is not None for mm in (batch.multimodal_inputs or [])):
        return 0
    if envs.SGLANG_DISAGG_STAGING_BUFFER.get() or args.enable_hisparse:
        return 0
    if self.ps.pp_size != 1:
        return 0
    if (
        self.enable_unified_memory
        or get_parallel().dcp_enabled
        or any(req.pending_bootstrap for req in batch.reqs)
        or batch.beam_tail is not None
    ):
        return 0
    if batch.input_embeds is not None or batch.replace_embeds is not None:
        return 0
    if any(
        req.return_routed_experts or req.return_indexer_topk or req.return_sampling_mask
        for req in batch.reqs
    ):
        return 0
    if self.enable_overlap and batch.has_grammar:
        return 0
    if (
        args.enable_eplb
        or args.expert_distribution_recorder_mode is not None
        or getattr(args, "enable_expert_distribution_metrics", False)
        or args.elastic_ep_backend is not None
    ):
        return 0

    if not self.spec_algorithm.is_none():
        if (
            not self.spec_algorithm.is_eagle()
            or self.spec_algorithm.is_eagle3()
            or self.spec_algorithm.is_frozen_kv_mtp()
            or args.enable_multi_layer_eagle
        ):
            return 0

    kv_manager = self.disagg_prefill_bootstrap_queue.kv_manager
    if not getattr(kv_manager, "is_hybrid_mla_backend", False):
        return 0

    if any(
        not callable(getattr(self.token_to_kv_pool_allocator, method, None))
        for method in ("pin_pages", "unpin_pages")
    ):
        return 0

    required_sender_methods = (
        "begin_layer_transfer_chunk",
        "send_layers",
        "seal_layer_transfer_chunk",
        "drain_completed_layer_transfer_chunks",
        "complete_layer_transfer_chunk",
        "finalize_layer_transfer",
    )
    if any(
        not callable(getattr(req.disagg_kv_sender, method, None))
        for req in batch.reqs
        for method in required_sender_methods
    ):
        return 0

    model = getattr(getattr(self.tp_worker, "model_runner", None), "model", None)
    runner = self.tp_worker.model_runner
    if runner.canary_manager is not None or runner.msprobe_debugger is not None:
        return 0
    if type(model).__name__ not in (
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
    ) or not callable(getattr(model, "forward_split_prefill", None)):
        return 0

    explicit = envs.SGLANG_PIPELINE_GROUP_SIZE.get()
    if explicit is not None:
        if explicit <= 0:
            print_warning_once(
                "SGLANG_PIPELINE_GROUP_SIZE must be positive; using the "
                "mainline prefill path."
            )
            return 0
        print_info_once(
            f"GLM5 layer-pipelined KV transfer is enabled with group size {explicit}."
        )
        return explicit

    avg_tokens = sum(req.extend_range.length for req in batch.reqs) // len(batch.reqs)
    target_iterations = 10 if avg_tokens < 4096 else 8 if avg_tokens < 8192 else 6
    group_size = max(1, self.model_config.num_hidden_layers // target_iterations)
    print_info_once(
        f"GLM5 layer-pipelined KV transfer is enabled with group size {group_size}."
    )
    return group_size


def _begin_pipelined_kv_page_lease(self, req, page_indices):
    """Pin one logical source-page chunk until all layer sends complete."""
    # run_batch_pipelined executes the producer under forward_stream_ctx
    # when overlap scheduling is enabled.  Allocator free/evict operations
    # remain schedule-stream owned, so establish the pin on that same
    # stream.  Otherwise the next scheduler iteration could read the pin
    # counters before a forward-stream update becomes visible.
    with self.device_module.StreamContext(self.schedule_stream):
        pinned_pages = self.token_to_kv_pool_allocator.pin_pages(page_indices)
    try:
        token = req.disagg_kv_sender.begin_layer_transfer_chunk(page_indices)
    except Exception:
        with self.device_module.StreamContext(self.schedule_stream):
            self.token_to_kv_pool_allocator.unpin_pages(pinned_pages)
        raise

    pending = getattr(self, "_pipelined_kv_page_leases", None)
    if pending is None:
        pending = self._pipelined_kv_page_leases = {}
    if token in pending:
        with self.device_module.StreamContext(self.schedule_stream):
            self.token_to_kv_pool_allocator.unpin_pages(pinned_pages)
        raise RuntimeError(f"Duplicate layer-transfer chunk token: {token}")
    pending[token] = (req.disagg_kv_sender, pinned_pages, req.rid)
    return token


def _drain_pipelined_kv_page_leases(self) -> None:
    """Release completed RDMA source-page leases on the scheduler thread."""
    if not envs.SGLANG_PIPELINED_KV_TRANSFER.get() or not is_hcu():
        return
    kv_manager = self.disagg_prefill_bootstrap_queue.kv_manager
    raise_worker_error = getattr(kv_manager, "raise_if_transfer_worker_failed", None)
    if callable(raise_worker_error):
        # Fatal worker errors take precedence over unpinning. The transfer
        # engine may still own a source pointer after an arbitrary
        # exception, so only process teardown may reclaim those pages.
        raise_worker_error()

    pending = getattr(self, "_pipelined_kv_page_leases", None)
    if not pending:
        return

    # Every sender owned by this scheduler shares one manager-wide
    # completion queue. Drain it once rather than once per request.
    sender = next(iter(pending.values()))[0]
    completions = sender.drain_completed_layer_transfer_chunks()
    with self.device_module.StreamContext(self.schedule_stream):
        for completion in completions:
            token = completion.token
            entry = pending.get(token)
            if entry is None:
                logger.warning(
                    "Ignoring unknown layer-transfer chunk completion: %s",
                    token,
                )
                continue
            _, pinned_pages, _ = entry
            self.token_to_kv_pool_allocator.unpin_pages(pinned_pages)
            del pending[token]


def run_layer_pipeline(scheduler, group_size, batch, on_publish=None, **kwargs):
    """Source layer-group algorithm; run_batch owns isolation, relay and D2H."""
    worker = scheduler.tp_worker
    runner = worker.model_runner
    has_draft = scheduler.spec_algorithm.is_eagle()
    manager = scheduler.disagg_prefill_bootstrap_queue.kv_manager
    num_layers = scheduler.model_config.num_hidden_layers
    owned_ids = set(manager.kv_args.kv_layer_ids)
    draft_ids = tuple(sorted(i for i in owned_ids if i >= num_layers))
    page_size = scheduler.token_to_kv_pool_allocator.page_size
    chunks = []
    final_infos = {}
    pipelined_rids = set()
    for req in batch.reqs:
        last = req is not batch.chunked_req
        start = req.start_send_idx
        end = min(req.extend_range.end, len(req.origin_input_ids))
        if not last:
            end -= end % page_size
        if end < start or (end == start and not last):
            continue
        pages = kv_to_page_indices(
            scheduler.req_to_token_pool.req_to_token[req.kv.req_pool_idx, start:end],
            page_size,
        )
        req.start_send_idx = end
        pipelined_rids.add(req.rid)
        chunks.append((req, pages, last))

    active_leases = []
    try:
        for req, pages, _ in chunks:
            if len(pages):
                _begin_pipelined_kv_page_lease(scheduler, req, pages)
                active_leases.append(req.disagg_kv_sender)
        worker.set_hicache_consumer(batch.hicache_consumer_index)
        fb = ForwardBatch.init_new(
            batch,
            runner,
            capture_hidden_mode=CaptureHiddenMode.FULL if has_draft else None,
            return_hidden_states_before_norm=(
                scheduler.model_worker.need_hidden_states_before_norm
                if has_draft
                else False
            ),
        )
        fb.split_index = 0
        with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
            runner._prepare_eager_forward_batch(fb)
            runner._maybe_execute_deferred_mamba_cow_and_clear(fb)
            fb = runner.eager_runner.load_batch(fb)
            fb.attn_cp_metadata = None
            if hasattr(runner.model, "prepare_forward_batch"):
                runner.model.prepare_forward_batch(fb)
            for start in range(0, num_layers, group_size):
                end = min(start + group_size, num_layers)
                logits = runner.forward_split_prefill(fb, forward_count=end - start)
                event = torch.cuda.Event()
                event.record()
                layer_ids = tuple(i for i in range(start, end) if i in owned_ids)
                if layer_ids:
                    for req, pages, _ in chunks:
                        if len(pages):
                            req.disagg_kv_sender.send_layers(pages, layer_ids, event)
            assert logits is not None, "Final GLM layer group must return logits"
            if fb.global_num_tokens_cpu is not None:
                fb.post_forward_mlp_sync_batch(logits)
            if fb.is_prefill_only:
                next_ids = torch.zeros(
                    len(fb.seq_lens), dtype=torch.long, device=fb.input_ids.device
                )
                if fb.return_logprob and logits.next_token_logits is not None:
                    runner.compute_logprobs_only(logits, fb)
            else:
                next_ids = runner.sample(logits, fb)
        result = GenerationBatchResult(logits_output=logits, next_token_ids=next_ids)
        if has_draft:
            result.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(result.new_seq_lens)
            # The target consumed these deferred operations already.
            batch.mamba_cow_src_indices = None
            batch.mamba_cow_dst_indices = None
            batch.mamba_clear_indices = None
            draft = scheduler.model_worker.draft_worker
            with (
                draft.draft_tp_context(draft.draft_runner.tp_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                result.next_draft_input = draft._draft_extend_for_prefill(
                    batch, logits.hidden_states, next_ids, logits.mm_input_embeds
                )
            event = torch.cuda.Event()
            event.record()
            if draft_ids:
                for req, pages, _ in chunks:
                    if len(pages):
                        req.disagg_kv_sender.send_layers(pages, draft_ids, event)
        for req, pages, last in chunks:
            if last:
                final_infos[req.rid] = (pages, event)
            else:
                req.disagg_kv_sender.complete_layer_transfer_chunk(pages)
                scheduler.disagg_prefill_pending_chunk_rids.add(req.rid)
        result.pipelined_kv_rids = pipelined_rids
        result.pipelined_kv_finalize_infos = final_infos
        return result
    finally:
        active_exception = sys.exc_info()[0] is not None
        first_error = None
        for sender in active_leases:
            try:
                sender.seal_layer_transfer_chunk()
            except Exception as exc:
                logger.exception("Failed to seal a layer-transfer page lease")
                if first_error is None:
                    first_error = exc
        if first_error is not None and not active_exception:
            raise first_error
