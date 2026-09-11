"""Short-request reservation for HCU GLM chunked prefill."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.layers.attention.glm5_next import is_glm5_next_hcu
from sglang.srt.layers.attention.glm5_next.runtime import get_glm5_next_runtime_args
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS, PrefillAdder
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.runtime_context import get_schedule

logger = logging.getLogger(__name__)


@dataclass
class AutoChunkPlan:
    """One-round plan for sharing a long GLM5-Next prefill chunk."""

    chunk_cap: int
    reserved_tokens: int
    short_reqs: List[Req]


class HcuAutoChunkPlanner:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.server_args = get_glm5_next_runtime_args()
        self.enable_prefill_short_req_reserve = True

    def __getattr__(self, name):
        return getattr(self.scheduler, name)

    def _mamba_slots_needed(self, req: Req) -> int:
        """Return the Mamba-pool slots needed to admit ``req``."""
        req_pool = self.req_to_token_pool
        slots = 1 if getattr(req.kv, "mamba_pool_idx", None) is None else 0
        if (
            getattr(req_pool, "enable_mamba_extra_buffer", False)
            and getattr(req.kv, "mamba_ping_pong_track_buffer", None) is None
        ):
            slots += req_pool.mamba_ping_pong_track_buffer_size
        return slots

    def _estimate_auto_chunk_prefix_len(self, req: Req) -> int:
        """Estimate reusable prefix length without mutating cache or request state."""
        fill_ids = list(req.full_untruncated_fill_ids)
        max_prefix_len = max(len(fill_ids) - 1, 0)
        if getattr(req, "return_logprob", False) and req.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, req.logprob_start_len)
        if getattr(req, "positional_embed_overrides", None) is not None:
            max_prefix_len = 0

        existing_len = min(
            len(req.prefix_indices) if req.prefix_indices is not None else 0,
            max_prefix_len,
        )
        if (
            not self.server_args.prefill_short_req_match_prefix
            or max_prefix_len <= existing_len
        ):
            return existing_len

        probe = getattr(self.tree_cache, "probe_prefix_len", None)
        if probe is None or getattr(self.tree_cache, "disable", False):
            return existing_len

        try:
            probed_len = probe(
                RadixKey(
                    token_ids=fill_ids[:max_prefix_len],
                    extra_key=getattr(req, "extra_key", None),
                )
            )
        except Exception as exc:
            if not getattr(self, "_auto_chunk_prefix_probe_warned", False):
                logger.warning(
                    "GLM5-Next auto chunk prefix probe failed; falling back to "
                    "the request's existing prefix: %s",
                    exc,
                )
                self.scheduler._auto_chunk_prefix_probe_warned = True
            return existing_len

        if probed_len is None:
            if not getattr(self, "_auto_chunk_prefix_probe_warned", False):
                logger.warning(
                    "GLM5-Next auto chunk prefix probe is unsupported by %s; "
                    "falling back to the request's existing prefix.",
                    type(self.tree_cache).__name__,
                )
                self.scheduler._auto_chunk_prefix_probe_warned = True
            return existing_len

        return max(existing_len, min(max(int(probed_len), 0), max_prefix_len))

    def _select_auto_chunk_short_reqs(
        self,
        adder: PrefillAdder,
        max_short_reqs: int,
        max_reserve: int,
        full_chunk: int,
    ) -> Tuple[List[Req], int]:
        """Select short requests using conservative token and Mamba budgets.

        GLM5-Next uses a Mamba radix cache whose authoritative prefix match can
        mutate cache state. This dry run therefore uses either prefix information
        already present on the request or the cache's side-effect-free probe;
        the normal admission path remains authoritative.
        """
        if max_short_reqs <= 0 or max_reserve <= 0:
            return [], 0

        req_pool = self.req_to_token_pool
        mamba_pool = getattr(req_pool, "mamba_pool", None)
        if mamba_pool is None:
            return [], 0

        page_size = adder.page_size
        # Reserving input tokens merely moves them from the long request to the
        # short requests.  The additional KV cost is each short request's decode
        # reserve and page overhead, plus the long request's own page overhead.
        kv_headroom = int(adder.rem_total_tokens) - full_chunk - page_size
        if kv_headroom <= 0:
            return [], 0

        mamba_slots_left = int(req_pool.mamba_allocator.available_size())
        selected: List[Req] = []
        reserved = 0
        extra_kv_cost = 0
        max_total_len = self.server_args.prefill_short_req_max_total_len

        for req in self.waiting_queue[: self.server_args.prefill_short_req_scan_depth]:
            if len(selected) >= max_short_reqs:
                break
            if req is self.chunked_req:
                continue
            if max_total_len > 0 and req.seqlen > max_total_len:
                continue
            if req.sampling_params.ignore_eos:
                continue

            prefix_len = self._estimate_auto_chunk_prefix_len(req)
            effective_len = req.seqlen - prefix_len
            if not (0 < effective_len <= self.server_args.prefill_short_req_threshold):
                continue

            chunk_cost = adder.ceil_paged_tokens(effective_len)
            # PrefillAdder requires a new request's input to be strictly less
            # than rem_input_tokens once the long request is in can_run_list.
            # Keep one additional page so the final selected short request does
            # not hit the equality case and get rejected after we compressed
            # the long request for it.
            if reserved + chunk_cost + page_size > max_reserve:
                continue

            # Use the full clipped generation reserve.  This is at least as
            # conservative as add_one_req's remaining-generation admission
            # check and also matches _update_prefill_budget's deduction.
            max_new_tokens = min(
                max(req.sampling_params.max_new_tokens, 0),
                CLIP_MAX_NEW_TOKENS,
            )
            candidate_extra_kv = max_new_tokens + page_size
            if extra_kv_cost + candidate_extra_kv >= kv_headroom:
                continue

            mamba_slots = self._mamba_slots_needed(req)
            if mamba_slots > mamba_slots_left:
                continue

            selected.append(req)
            reserved += chunk_cost
            extra_kv_cost += candidate_extra_kv
            mamba_slots_left -= mamba_slots

        return selected, reserved

    def _plan_auto_chunk(
        self, adder: PrefillAdder, max_short_reqs: int
    ) -> AutoChunkPlan:
        """Plan one GLM5-Next adaptive chunk round without mutating state."""
        page_size = adder.page_size
        rem_chunk_tokens = int(adder.rem_chunk_tokens or 0)
        full_chunk = rem_chunk_tokens // page_size * page_size
        full_plan = AutoChunkPlan(
            chunk_cap=full_chunk, reserved_tokens=0, short_reqs=[]
        )

        # Static deployment compatibility is validated once by ServerArgs.
        # Per-round planning only checks state and resource availability.
        if (
            not self.enable_prefill_short_req_reserve
            or not self.waiting_queue
            or max_short_reqs <= 0
            or full_chunk < 2 * page_size
        ):
            return full_plan

        long_req = self.chunked_req
        if long_req is None:
            return full_plan
        if (
            getattr(long_req, "chunk_starved_rounds", 0)
            >= self.server_args.prefill_long_req_starve_threshold
        ):
            return full_plan

        max_reserve = min(
            int(full_chunk * self.server_args.prefill_short_req_max_reserve_ratio),
            full_chunk - page_size,
        )
        max_reserve = max_reserve // page_size * page_size
        short_reqs, short_input_tokens = self._select_auto_chunk_short_reqs(
            adder, max_short_reqs, max_reserve, full_chunk
        )
        reserved = short_input_tokens + page_size if short_reqs else 0
        chunk_cap = full_chunk - reserved

        # If the long request would finish under this cap, the existing adder
        # already has enough room for short requests; compression is unnecessary.
        if (
            reserved <= 0
            or (len(long_req.full_untruncated_fill_ids) - len(long_req.prefix_indices))
            <= chunk_cap
        ):
            return full_plan

        return AutoChunkPlan(
            chunk_cap=chunk_cap,
            reserved_tokens=reserved,
            short_reqs=short_reqs,
        )

    def _promote_auto_chunk_reqs(self, short_reqs: List[Req]) -> None:
        """Stably promote selected requests within the configured scan window."""
        scan_depth = self.server_args.prefill_short_req_scan_depth
        head = self.waiting_queue[:scan_depth]
        tail = self.waiting_queue[scan_depth:]
        selected_ids = {id(req) for req in short_reqs}
        promoted = [req for req in head if id(req) in selected_ids]
        rest = [req for req in head if id(req) not in selected_ids]
        self.scheduler.waiting_queue = promoted + rest + tail

    def _commit_auto_chunk_plan(
        self,
        long_req: Req,
        plan: AutoChunkPlan,
        scheduled: bool,
    ) -> List[Req]:
        """Commit starvation and queue state after actual long-req admission."""
        if not scheduled:
            return []
        if plan.reserved_tokens <= 0:
            long_req.chunk_starved_rounds = 0
            return []

        long_req.chunk_starved_rounds = getattr(long_req, "chunk_starved_rounds", 0) + 1
        self._promote_auto_chunk_reqs(plan.short_reqs)
        self._log_auto_chunk_intent(long_req, plan)
        return plan.short_reqs

    def _log_auto_chunk_intent(self, long_req: Req, plan: AutoChunkPlan) -> None:
        if not getattr(self, "is_stats_logging_rank", False):
            return
        logger.info(
            "GLM5-Next auto chunk intent: cap=%s reserved=%s short_reqs=%s "
            "starved_rounds=%s queue_len=%s scan_depth=%s rid=%s",
            plan.chunk_cap,
            plan.reserved_tokens,
            len(plan.short_reqs),
            long_req.chunk_starved_rounds,
            len(self.waiting_queue),
            self.server_args.prefill_short_req_scan_depth,
            long_req.rid,
        )

    def _log_auto_chunk_actual(
        self, long_req: Req, short_reqs: List[Req], can_run_list: List[Req]
    ) -> None:
        if not short_reqs or not getattr(self, "is_stats_logging_rank", False):
            return
        admitted_ids = {id(req) for req in can_run_list}
        admitted = [req for req in short_reqs if id(req) in admitted_ids]
        logger.info(
            "GLM5-Next auto chunk actual: admitted=%s/%s tokens=%s rid=%s",
            len(admitted),
            len(short_reqs),
            sum(req.extend_range.length for req in admitted),
            long_req.rid,
        )


def add_chunked_req_with_reservation(scheduler, adder, running_bs):
    req = scheduler.chunked_req
    cfg = get_schedule()
    if not (
        cfg.prefill_short_req_reserve
        and is_glm5_next_hcu(scheduler.model_config.hf_config)
        and scheduler.disaggregation_mode == DisaggregationMode.PREFILL
        and scheduler.chunked_prefill_size
        and not scheduler.enable_unified_memory
    ):
        return adder.add_chunked_req(req)
    max_short = max(scheduler.get_num_allocatable_reqs(running_bs) - 1, 0)
    if cfg.prefill_max_requests is not None:
        max_short = min(max_short, max(cfg.prefill_max_requests - 1, 0))
    planner = HcuAutoChunkPlanner(scheduler)
    plan = planner._plan_auto_chunk(adder, max_short)
    remaining = adder.add_chunked_req(req, max_tokens=plan.chunk_cap)
    planner._commit_auto_chunk_plan(req, plan, req in adder.can_run_list)
    return remaining
