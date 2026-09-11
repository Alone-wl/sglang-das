"""HCU GLM layer-group transfer leases, using main's registered layer IDs."""

from __future__ import annotations

import dataclasses
import queue as thread_queue
import threading
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.utils import KVPoll, filter_kv_indices_for_cp_rank
from sglang.srt.runtime_context import get_parallel


@dataclasses.dataclass
class LayerTransferKVChunk(TransferKVChunk):
    layer_ids: Optional[Tuple[int, ...]] = None
    cuda_event: object = None
    layer_transfer_token: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class LayerTransferChunkCompletion:
    """A layer-pipelined source-page chunk that no worker can read anymore."""

    token: int
    room: int
    page_indices: npt.NDArray[np.int32]


@dataclasses.dataclass
class _LayerTransferChunkLease:
    room: int
    page_indices: npt.NDArray[np.int32]
    outstanding: int = 0
    sealed: bool = False


class LayerTransferManagerMixin:
    def init_layer_transfer_leases(self):
        self._layer_transfer_lease_lock = threading.Lock()
        self._next_layer_transfer_token = 0
        self._layer_transfer_leases = {}
        self._completed_layer_transfer_chunks = thread_queue.SimpleQueue()
        self._transfer_worker_errors = thread_queue.SimpleQueue()

    def begin_layer_transfer_chunk(
        self, room: int, page_indices: npt.NDArray[np.int32]
    ) -> int:
        """Create the lifecycle token for one logical pipelined page chunk.

        This method only records page ids. The scheduler owns allocator pinning
        and must pin the pages before it starts enqueueing layers for the token.
        """
        page_indices = np.asarray(page_indices, dtype=np.int32).copy()
        with self._layer_transfer_lease_lock:
            token = self._next_layer_transfer_token
            self._next_layer_transfer_token += 1
            self._layer_transfer_leases[token] = _LayerTransferChunkLease(
                room=room,
                page_indices=page_indices,
            )
        return token

    def _enqueue_layer_transfer_chunk(
        self,
        queue,
        kv_chunk: LayerTransferKVChunk,
    ) -> None:
        """Enqueue a layer and account for it atomically with worker finish."""
        token = kv_chunk.layer_transfer_token
        if token is None:
            queue.put(kv_chunk)
            return

        with self._layer_transfer_lease_lock:
            lease = self._layer_transfer_leases.get(token)
            if lease is None:
                raise RuntimeError(
                    f"Unknown layer-transfer token {token} for room {kv_chunk.room}"
                )
            if lease.room != kv_chunk.room:
                raise RuntimeError(
                    "Layer-transfer token room mismatch: "
                    f"token={token}, expected={lease.room}, actual={kv_chunk.room}"
                )
            if lease.sealed:
                raise RuntimeError(
                    f"Cannot enqueue sealed layer-transfer token {token}"
                )

            # Account and enqueue under one lease lock. A worker may dequeue
            # the item immediately, but it cannot finish its lease until this
            # lock is released. Roll back if FastQueue rejects the item, so
            # only successfully enqueued entries remain outstanding.
            lease.outstanding += 1
            try:
                queue.put(kv_chunk)
            except BaseException:
                lease.outstanding -= 1
                raise

    def seal_layer_transfer_chunk(self, room: int, token: int) -> None:
        """Declare that the producer will enqueue no more layers for a token."""
        with self._layer_transfer_lease_lock:
            lease = self._layer_transfer_leases.get(token)
            if lease is None:
                raise RuntimeError(
                    f"Unknown layer-transfer token {token} for room {room}"
                )
            if lease.room != room:
                raise RuntimeError(
                    "Layer-transfer token room mismatch while sealing: "
                    f"token={token}, expected={lease.room}, actual={room}"
                )
            if lease.sealed:
                return
            lease.sealed = True
            self._maybe_complete_layer_transfer_chunk_locked(token, lease)

    def _finish_layer_transfer_chunk(self, token: Optional[int]) -> None:
        """Consume one terminal worker entry for a layer-transfer token."""
        if token is None:
            return
        with self._layer_transfer_lease_lock:
            lease = self._layer_transfer_leases.get(token)
            if lease is None:
                raise RuntimeError(f"Unknown completed layer-transfer token {token}")
            if lease.outstanding <= 0:
                raise RuntimeError(
                    f"Layer-transfer token {token} completed more than once"
                )
            lease.outstanding -= 1
            self._maybe_complete_layer_transfer_chunk_locked(token, lease)

    def _maybe_complete_layer_transfer_chunk_locked(
        self, token: int, lease: _LayerTransferChunkLease
    ) -> None:
        if not lease.sealed or lease.outstanding != 0:
            return
        self._layer_transfer_leases.pop(token)
        self._completed_layer_transfer_chunks.put(
            LayerTransferChunkCompletion(
                token=token,
                room=lease.room,
                page_indices=lease.page_indices,
            )
        )

    def drain_completed_layer_transfer_chunks(
        self,
    ) -> List[LayerTransferChunkCompletion]:
        """Return completions for scheduler-thread allocator unpinning."""
        completed = []
        while True:
            try:
                completed.append(self._completed_layer_transfer_chunks.get_nowait())
            except thread_queue.Empty:
                return completed

    def raise_if_transfer_worker_failed(self) -> None:
        """Propagate a fatal transfer-thread error on the scheduler thread."""
        try:
            error = self._transfer_worker_errors.get_nowait()
        except thread_queue.Empty:
            return
        raise error

    def send_kvcache_layers(self, kv_chunk, req, dst_info, dst_indices, executor):
        if dst_info.requires_dcp_relayout:
            raise RuntimeError(
                "Layer-pipelined GLM transfer requires matching page layouts"
            )
        self._validate_envelope_kv_layout(
            dst_info.dst_kv_ptrs, dst_info.dst_kv_item_len, dst_info.dst_attn_tp_size
        )
        owned_ids = self.kv_args.kv_layer_ids
        selected = [
            i for i, layer_id in enumerate(owned_ids) if layer_id in kv_chunk.layer_ids
        ]
        if len(selected) != len(kv_chunk.layer_ids):
            raise RuntimeError(
                "Layer transfer referenced an unregistered local KV layer"
            )
        ret = self._send_kvcache_generic(
            mooncake_session_id=req.mooncake_session_id,
            src_data_ptrs=[self.kv_args.kv_data_ptrs[i] for i in selected],
            dst_data_ptrs=dst_info.dst_kv_ptrs,
            item_lens=[self.kv_args.kv_item_lens[i] for i in selected],
            prefill_data_indices=kv_chunk.prefill_kv_indices,
            dst_data_indices=dst_indices,
            executor=executor,
            src_layer_ids=[owned_ids[i] for i in selected],
            dst_layer_ids=dst_info.dst_kv_layer_ids,
        )
        if ret != 0:
            # An engine failure need not have drained every submitted read.
            # Keep the lease pinned and fail the scheduler instead of recycling.
            raise RuntimeError(
                f"Layer-group KV transfer failed for room {req.room}: {ret}"
            )
        return ret


class LayerTransferSenderMixin:
    _active_layer_transfer_token: Optional[int] = None

    def begin_layer_transfer_chunk(self, page_indices: npt.NDArray[np.int32]) -> int:
        """Begin one logical source-page lease for pipelined layer sends."""
        if self._active_layer_transfer_token is not None:
            raise RuntimeError(
                "Cannot begin a new layer-transfer chunk before sealing token "
                f"{self._active_layer_transfer_token}"
            )
        token = self.kv_mgr.begin_layer_transfer_chunk(
            self.bootstrap_room, page_indices
        )
        self._active_layer_transfer_token = token
        return token

    def seal_layer_transfer_chunk(self) -> None:
        """Idempotently seal the active chunk after its final layer enqueue."""
        token = self._active_layer_transfer_token
        if token is None:
            return
        self.kv_mgr.seal_layer_transfer_chunk(self.bootstrap_room, token)
        self._active_layer_transfer_token = None

    def drain_completed_layer_transfer_chunks(
        self,
    ) -> List[LayerTransferChunkCompletion]:
        """Drain manager-wide completions for scheduler-owned page unpinning."""
        return self.kv_mgr.drain_completed_layer_transfer_chunks()

    def send_layers(
        self,
        kv_indices: npt.NDArray[np.int32],
        layer_ids: Tuple[int, ...],
        cuda_event: object,
    ) -> None:
        """Enqueue one produced GLM hybrid-attention layer group."""
        if not layer_ids:
            raise ValueError("layer_ids must not be empty")

        can_batch_layers = (
            self.kv_mgr.is_hybrid_mla_backend
            and self.kv_mgr.pp_size == 1
            and not self.kv_mgr.enable_custom_mem_pool
            and not self.kv_mgr.enable_staging
        )
        if len(layer_ids) > 1 and not can_batch_layers:
            for layer_id in layer_ids:
                self.send_layers(kv_indices, (layer_id,), cuda_event)
            return

        layer_transfer_token = self._active_layer_transfer_token
        if layer_transfer_token is None:
            raise RuntimeError(
                "send_layers requires begin_layer_transfer_chunk before the "
                "first layer group of each logical source-page chunk"
            )
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))

        if (
            self.kv_mgr.enable_all_cp_ranks_for_transfer
            and not get_parallel().enable_dsa_cache_layer_split
        ):
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
                total_pages=self.num_kv_indices,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            return

        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last_chunk=False,
            layer_ids=layer_ids,
            cuda_event=cuda_event,
            layer_transfer_token=layer_transfer_token,
            trace_ctx=self.trace_ctx.copy_for_thread(),
        )

    def complete_layer_transfer_chunk(self, kv_indices: npt.NDArray[np.int32]) -> None:
        """Advance page bookkeeping after all layers of a non-final chunk."""
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        if self.curr_idx >= self.num_kv_indices:
            raise RuntimeError(
                "Non-final pipelined KV chunk consumed all registered pages: "
                f"room={self.bootstrap_room}, consumed={self.curr_idx}, "
                f"registered={self.num_kv_indices}"
            )

        if self.kv_mgr.is_dummy_cp_rank:
            return
        if (
            self.kv_mgr.enable_all_cp_ranks_for_transfer
            and not get_parallel().enable_dsa_cache_layer_split
        ):
            kv_indices, _ = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
                total_pages=self.num_kv_indices,
            )
        self._record_transfer_indices(kv_indices, None)

    def finalize_layer_transfer(
        self,
        kv_indices: npt.NDArray[np.int32],
        cuda_event: object = None,
        state_indices: Optional[List] = None,
    ) -> None:
        """Send final state/aux metadata without retransmitting KV layers."""
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        if self.curr_idx != self.num_kv_indices:
            raise RuntimeError(
                "Pipelined KV finalize did not consume all registered pages: "
                f"room={self.bootstrap_room}, consumed={self.curr_idx}, "
                f"registered={self.num_kv_indices}"
            )

        if (
            self.kv_mgr.enable_all_cp_ranks_for_transfer
            and not get_parallel().enable_dsa_cache_layer_split
        ):
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
                total_pages=self.num_kv_indices,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
            return

        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            np.empty(0, dtype=np.int32),
            index_slice,
            is_last_chunk=True,
            aux_index=self.aux_index,
            state_indices=state_indices,
            cuda_event=cuda_event,
            trace_ctx=self.trace_ctx.copy_for_thread(),
        )
        self._record_transfer_indices(kv_indices, state_indices)
