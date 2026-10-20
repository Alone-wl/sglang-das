"""CPU abort lifecycle tests, including a real P-to-E HTTP notification."""

import asyncio
import concurrent.futures
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from aiohttp import web

from sglang.srt.disaggregation.encoder import http_server, receiver, server, transfer_queue
from sglang.srt.disaggregation.encoder.receiver import (
    GlmWaitingRDMARequest,
    MMReceiverHTTP,
    WaitingMMRequestStatus,
)
from sglang.srt.disaggregation.encoder.runtime import DPDispatcher, _DP_USER_ABORT
from sglang.srt.disaggregation.encoder.server import MMEncoder, MooncakeDelivery, ReqState
from sglang.srt.disaggregation.encoder.transfer_queue import (
    EncoderAbortRegistry,
    EncoderTransferQueue,
)
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), 3)


def waiting_request(req_id, urls, routes):
    req = GlmWaitingRDMARequest.__new__(GlmWaitingRDMARequest)
    req.rid = req_id
    req.recv_req = SimpleNamespace(rid=req_id, encoder_part_routes=routes)
    req.encoder_urls = urls
    req.status = WaitingMMRequestStatus.PENDING
    req.recv_socket = None
    req.registration_future = None
    req.embeddings_buffer = None
    req.embedding_pool = SimpleNamespace(release=Mock())
    req._pool_slot_id = None
    req._mm_finalizer = None
    req._part_finalizers = []
    req._part_buffers = {0: (torch.ones(4), 7, 16)}
    req._buffer_lock = threading.Lock()
    req._receive_running = True
    req._terminal = False
    req._posted_rdma = False
    req._received_all_parts = False
    return req


def scheduler_receiver(req, rank=0):
    owner = MMReceiverHTTP.__new__(MMReceiverHTTP)
    owner.encoder_transfer_backend = "mooncake"
    owner.tp_rank = rank
    owner.waiting_list = [req]
    return owner


class TestEncoderAbort(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        enc = self.encoder = MMEncoder.__new__(MMEncoder)
        enc.rank = 0
        enc.req_states = {}
        enc.abandoned_req_ids = set()
        enc.background_tasks = set()
        enc.send_timeout = 180
        enc.lifecycle_trace = False
        enc.transfer_queue = EncoderTransferQueue(1)
        enc.ack_queue = EncoderTransferQueue(1, phase="ack")
        enc.transfer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        enc.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        enc.delivery = SimpleNamespace(
            send=MooncakeDelivery(enc).send, release=AsyncMock()
        )
        self.unblock = threading.Event()
        self.tasks = []
        self.patches = [
            patch.object(http_server, "encoder", enc),
            patch.object(http_server, "dp_dispatcher", None),
            patch.object(server.meta_registry, "discard", AsyncMock()),
        ]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        self.unblock.set()
        await asyncio.wait_for(
            asyncio.gather(
                *self.tasks, *self.encoder.background_tasks, return_exceptions=True
            ),
            3,
        )
        self.encoder.transfer_executor.shutdown(wait=True)
        self.encoder.executor.shutdown(wait=True)
        for item in reversed(self.patches):
            item.stop()

    def send(self, req_id, rank=0):
        task = asyncio.create_task(
            self.encoder.send(req_id, "127.0.0.1", 1000 + rank)
        )
        self.tasks.append(task)
        return task

    async def test_abort_during_encode_wakes_send_and_defers_embedding_release(self):
        state = self.encoder._acquire_encode_ref("encoding")
        sending = self.send("encoding")
        await until(lambda: state.active_sends == 1)
        response = await http_server.handle_abort_request({"req_ids": ["encoding"]})
        self.assertEqual(response.status_code, 202)
        with self.assertRaisesRegex(server.InternalError, "released"):
            await sending
        self.assertTrue(state.release_requested)
        self.assertEqual(state.active_encodes, 1)
        self.encoder.delivery.release.assert_not_awaited()
        self.assertIn("encoding", self.encoder.req_states)
        state.embedding_data = SimpleNamespace(embedding=torch.ones(4))
        await self.encoder._release_encode_ref(state)
        self.assertIsNone(state.embedding_data)
        self.assertNotIn("encoding", self.encoder.req_states)
        self.encoder.delivery.release.assert_awaited_once_with(state)

    async def test_early_and_duplicate_abort_are_remembered_for_late_encode(self):
        for _ in range(2):
            response = await http_server.handle_abort_request({"req_ids": ["late"]})
            self.assertEqual(response.status_code, 202)
        await until(lambda: not self.encoder.background_tasks)
        state = self.encoder._acquire_encode_ref("late")
        self.assertTrue(state.release_requested)
        self.assertTrue(state.release_event.is_set())
        with self.assertRaises(server.InternalError):
            await self.send("late")
        await self.encoder._release_encode_ref(state)
        self.assertNotIn("late", self.encoder.req_states)

    async def _abort_sends(self, phase, mixed, all_running=False):
        calls = []
        if all_running:
            attr = "executor" if phase == "ack" else "transfer_executor"
            getattr(self.encoder, attr).shutdown(wait=True)
            setattr(
                self.encoder, attr,
                concurrent.futures.ThreadPoolExecutor(max_workers=8),
            )
            attr = "ack_queue" if phase == "ack" else "transfer_queue"
            setattr(self.encoder, attr, EncoderTransferQueue(8, phase=phase))

        def native(req_id):
            calls.append(req_id)
            if req_id == "blocked":
                if not self.unblock.wait(5):
                    raise RuntimeError("test failed to unblock native work")

        async def send(state, destination):
            await self.encoder._run_transfer_work(
                state.req_id, native, state.req_id, phase=phase
            )

        self.encoder.delivery.send = send
        queue = self.encoder.ack_queue if phase == "ack" else self.encoder.transfer_queue
        self.encoder.req_states["blocked"] = ReqState("blocked")
        running = self.send("blocked")
        await until(lambda: calls == ["blocked"])
        target = "blocked" if mixed or all_running else "queued"
        if target == "queued":
            self.encoder.req_states[target] = ReqState(target)
        state = self.encoder.req_states[target]
        state.embedding_data = SimpleNamespace(embedding=torch.ones(4))
        queued = [
            self.send(target, rank) for rank in range(7 if target == "blocked" else 8)
        ]
        if all_running:
            await until(lambda: queue.snapshot()["running"] == 8)
        else:
            await until(lambda: queue.snapshot()["queued"] == len(queued))
        response = await http_server.handle_abort_request({"req_ids": [target]})
        self.assertEqual(response.status_code, 202)
        if all_running:
            await until(lambda: state.release_requested)
            self.assertTrue(all(not task.done() for task in queued))
        else:
            outcomes = await asyncio.wait_for(
                asyncio.gather(*queued, return_exceptions=True), 2
            )
            self.assertTrue(
                all(isinstance(error, server.InternalError) for error in outcomes)
            )
        self.assertEqual(calls, ["blocked"] * (8 if all_running else 1))
        self.assertFalse(running.done())
        if mixed or all_running:
            self.assertEqual(state.active_sends, 8 if all_running else 1)
            self.assertIsNotNone(state.embedding_data)
        else:
            await until(lambda: target not in self.encoder.req_states)
            self.assertIsNone(state.embedding_data)
        self.unblock.set()
        await running
        if all_running:
            await asyncio.gather(*queued)
        if mixed or all_running:
            await until(lambda: target not in self.encoder.req_states)
            self.assertIsNone(state.embedding_data)

    async def test_abort_eight_queued_rdma_sends(self):
        await self._abort_sends("rdma", mixed=False)

    async def test_abort_eight_queued_ack_sends(self):
        await self._abort_sends("ack", mixed=False)

    async def test_abort_mixed_rdma_keeps_running_source(self):
        await self._abort_sends("rdma", mixed=True)

    async def test_abort_mixed_ack_keeps_running_source(self):
        await self._abort_sends("ack", mixed=True)

    async def test_abort_eight_running_rdma_sends_waits_for_completion(self):
        await self._abort_sends("rdma", mixed=False, all_running=True)

    async def test_abort_eight_running_ack_sends_waits_for_completion(self):
        await self._abort_sends("ack", mixed=False, all_running=True)

    async def test_http_abort_rejects_invalid_ids_without_partial_retirement(self):
        response = await http_server.handle_abort_request({"req_ids": ["valid", ""]})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(hasattr(self.encoder, "_user_aborts"))

    async def test_real_http_notification_retries_and_retires_encoding_part(self):
        req_id = "user_local_part_0"
        state = self.encoder._acquire_encode_ref(req_id)
        calls = []

        async def abort(request):
            payload = await request.json()
            calls.append(payload)
            if len(calls) == 1:
                return web.Response(status=503)
            result = await http_server.handle_abort_request(payload)
            return web.Response(status=result.status_code, body=result.body)

        app = web.Application()
        app.router.add_post("/abort", abort)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        req = waiting_request(
            "user", [f"http://127.0.0.1:{port}"],
            [{"encoder_idx": 0, "req_id": req_id}],
        )
        owner = scheduler_receiver(req)
        try:
            owner.abort_waiting_requests(SimpleNamespace(rid="user", abort_all=False))
            self.assertEqual(req.status, WaitingMMRequestStatus.FAIL)
            self.assertTrue(req._terminal)
            req.embedding_pool.release.assert_not_called()
            await until(lambda: state.release_requested)
            await until(lambda: not owner._abort_futures)
            self.assertEqual(calls, [{"req_ids": [req_id]}] * 2)
            self.assertIn(req_id, self.encoder.req_states)
            await self.encoder._release_encode_ref(state)
            self.assertNotIn(req_id, self.encoder.req_states)
            # Notification acceptance must not free a target whose receive
            # thread still owns it, even after E retired the encoding part.
            req.embedding_pool.release.assert_not_called()
        finally:
            if getattr(owner, "_abort_runner", None) is not None:
                owner._abort_runner.loop.call_soon_threadsafe(owner._abort_runner.loop.stop)
                owner._abort_runner.thread.join(timeout=2)
                owner._abort_runner.loop.close()
            await runner.cleanup()


class TestAbortRouting(unittest.TestCase):
    def test_legacy_routes_use_one_part_per_nonempty_encoder_assignment(self):
        req = waiting_request("legacy", ["e0", "e1", "e2"], [])
        req.num_items_assigned = {"image": [2, 0, 1], "video": [0, 1, 0]}
        self.assertEqual(req.encoder_abort_routes(), [
            {"encoder_idx": 0, "req_id": "legacy_local_part_0"},
            {"encoder_idx": 2, "req_id": "legacy_local_part_1"},
            {"encoder_idx": 1, "req_id": "legacy_local_part_2"},
        ])

    def test_only_tp_leader_notifies_exact_frozen_routes(self):
        notifications = []

        async def notify(parts):
            notifications.append(parts)

        class Runner:
            def submit(self, coroutine):
                asyncio.run(coroutine)
                future = concurrent.futures.Future()
                future.set_result(None)
                return future

        routes = [
            {"encoder_idx": 1, "req_id": "req_local_part_0"},
            {"encoder_idx": 0, "req_id": "req_local_part_1"},
            {"encoder_idx": 1, "req_id": "req_local_part_0"},
        ]
        with patch.object(receiver, "_post_encoder_aborts", notify):
            for rank in range(8):
                req = waiting_request("req", ["http://e0", "http://e1"], routes)
                owner = scheduler_receiver(req, rank)
                owner._abort_runner = Runner()
                owner._abort_futures = set()
                owner.abort_waiting_requests(SimpleNamespace(rid="req", abort_all=False))
                self.assertEqual(req.status, WaitingMMRequestStatus.FAIL)
                self.assertTrue(req._terminal)
                req.embedding_pool.release.assert_not_called()
        self.assertEqual(notifications, [{
            "http://e1": ["req_local_part_0"],
            "http://e0": ["req_local_part_1"],
        }])

    def test_abort_tombstones_expire_and_are_bounded(self):
        now = [0.0]
        with patch.object(transfer_queue, "time", SimpleNamespace(monotonic=lambda: now[0])):
            aborted = EncoderAbortRegistry(ttl=10, max_entries=2)
            self.assertTrue(aborted.remember("a"))
            self.assertFalse(aborted.remember("a"))
            aborted.remember("b")
            aborted.remember("c")
            self.assertNotIn("a", aborted)
            self.assertIn("b", aborted)
            now[0] = 11
            self.assertNotIn("b", aborted)
            self.assertTrue(aborted.remember("b"))


class TestDPAbort(unittest.IsolatedAsyncioTestCase):
    def dispatcher(self):
        return DPDispatcher(1, [object()], [object()], object(), [])

    async def test_owner_receives_abort_on_independent_release_channel(self):
        dispatcher = self.dispatcher()
        dispatcher.req_id_to_rank["req"] = 0
        with patch(
            "sglang.srt.disaggregation.encoder.runtime.async_sock_send", AsyncMock()
        ) as send:
            await dispatcher.abort_requests(["req", "req"])
        send.assert_awaited_once()
        self.assertIs(send.await_args.args[0], dispatcher.release_sockets[0])
        self.assertEqual(unwrap_from_pickle(send.await_args.args[1]), {
            "_dp_type": _DP_USER_ABORT, "req_id": "req",
        })

    async def test_abort_before_dispatch_prevents_late_worker_submission(self):
        dispatcher = self.dispatcher()
        with patch(
            "sglang.srt.disaggregation.encoder.runtime.async_sock_send", AsyncMock()
        ) as send:
            await dispatcher.abort_requests(["late"])
            result = await dispatcher.dispatch({"req_id": "late"})
        send.assert_not_awaited()
        self.assertEqual(result["_error_code"], 400)
        self.assertEqual(dispatcher.pending_counts, [0])

    async def test_failed_worker_notification_can_be_retried(self):
        dispatcher = self.dispatcher()
        dispatcher.req_id_to_rank["req"] = 0
        with patch(
            "sglang.srt.disaggregation.encoder.runtime.async_sock_send", AsyncMock(
                side_effect=[RuntimeError("IPC unavailable"), None]
            )
        ) as send:
            with self.assertRaisesRegex(RuntimeError, "IPC unavailable"):
                await dispatcher.abort_requests(["req"])
            await dispatcher.abort_requests(["req"])
        self.assertEqual(send.await_count, 2)


if __name__ == "__main__":
    unittest.main()
