"""CPU fault injection for Mooncake send retirement; no model/GPU/RDMA needed."""

import asyncio
import concurrent.futures
import gc
import os
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
import zmq

from sglang.srt.disaggregation.encoder import glm_rdma, http_server, server
from sglang.srt.disaggregation.encoder.receiver import EmbeddingData
from sglang.srt.disaggregation.encoder.runtime import send_staged_embedding
from sglang.srt.disaggregation.encoder.server import (
    EncoderMetaRegistry,
    InternalError,
    MMEncoder,
    MooncakeDelivery,
    ReqState,
)
from sglang.srt.disaggregation.encoder.transfer_queue import (
    EncoderTransferQueue,
    TransferRequestReleased,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout=2)


class TestEncoderTransferRetirement(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.unblock = threading.Event()
        self.native_calls = []
        self.ack_calls = []
        self.native_block = None
        self.ack_block = None
        self.ack_error = False
        self.tasks = []
        self.patches = [
            patch.object(
                server,
                "get_disagg",
                return_value=SimpleNamespace(encoder_transfer_backend="mooncake"),
            ),
            patch.object(server, "config_socket"),
            patch.object(server, "encoder_metrics_collector", None),
            patch.object(server.meta_registry, "discard", AsyncMock()),
            patch.object(server.meta_registry, "note_send_done", AsyncMock()),
        ]
        for item in self.patches:
            item.start()

        encoder = self.encoder = MMEncoder.__new__(MMEncoder)
        encoder.rank = 0
        encoder.req_states = {}
        encoder.model_type = "glm_test"
        encoder._element_size = 4
        encoder.transfer_backend = "mooncake"
        encoder.send_timeout = 10
        encoder.lifecycle_trace = False
        encoder.transfer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        encoder.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        encoder.transfer_queue = EncoderTransferQueue(1)
        encoder.ack_queue = EncoderTransferQueue(1, phase="ack")
        encoder.delivery = MooncakeDelivery(encoder)

        def transfer(embedding, session_id, address):
            self.native_calls.append(session_id)
            if session_id == self.native_block:
                if not self.unblock.wait(5):
                    raise RuntimeError("test failed to release native worker")
            return 0

        encoder._glm_source_buffers = SimpleNamespace(transfer=transfer)

        def socket(*_):
            sock = SimpleNamespace(setsockopt=Mock(), close=Mock())
            sock.connect = lambda endpoint: setattr(sock, "endpoint", endpoint)

            def send(*_args, **_kwargs):
                self.ack_calls.append(sock.endpoint)
                if sock.endpoint == self.ack_block:
                    if not self.unblock.wait(5):
                        raise RuntimeError("test failed to release ACK worker")
                    if self.ack_error:
                        raise zmq.Again()
                return SimpleNamespace(wait=Mock())

            sock.send_multipart = send
            return sock

        encoder.sync_context = SimpleNamespace(socket=socket)

    async def asyncTearDown(self):
        self.unblock.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), 3)
        self.encoder.transfer_executor.shutdown(wait=True)
        self.encoder.executor.shutdown(wait=True)
        for item in reversed(self.patches):
            item.stop()

    def stage(self, req_id):
        state = ReqState(req_id, EmbeddingData(
            req_id, 1, 0, None, Modality.IMAGE, embedding=torch.ones((4, 4))
        ))
        state.embedding_ready.set()
        self.encoder.req_states[req_id] = state
        return state

    def send(self, req_id, *, session=None, port=1234):
        task = asyncio.create_task(send_staged_embedding(self.encoder, {
            "req_id": req_id,
            "prefill_host": "127.0.0.1",
            "embedding_port": port,
            "session_id": session or req_id,
            "buffer_address": 1,
            "receive_count": 1,
        }, release_without_count=False))
        self.tasks.append(task)
        return task

    async def block_native(self):
        self.stage("blocked")
        self.native_block = "blocked"
        task = self.send("blocked")
        await until(lambda: "blocked" in self.native_calls)
        return task

    def use_legacy_submission(self):
        async def legacy(req_id, fn, *args, phase="rdma"):
            executor = (
                self.encoder.executor
                if phase == "ack"
                else self.encoder.transfer_executor
            )
            return await server._await_transfer_completion(
                asyncio.get_running_loop().run_in_executor(executor, fn, *args),
                "legacy send",
            )

        self.encoder._run_transfer_work = legacy

    async def test_retire_queued_rdma_while_other_worker_stays_blocked(self):
        blocked = await self.block_native()
        state = self.stage("queued")
        tensor_ref = weakref.ref(state.embedding_data.embedding)
        queued = self.send("queued")
        await until(lambda: self.encoder.transfer_queue.snapshot()["queued"] == 1)

        await asyncio.wait_for(self.encoder.release_request("queued"), 1)
        with self.assertRaisesRegex(InternalError, "retired before rdma"):
            await queued
        self.assertNotIn("queued", self.native_calls)
        self.assertIsNone(state.embedding_data)
        self.assertFalse(blocked.done())
        self.assertEqual(self.encoder.lifecycle_snapshot()["staged_cpu_bytes"], 64)
        self.tasks.remove(queued)
        del queued
        await asyncio.sleep(0)
        gc.collect()
        self.assertIsNone(tensor_ref(), "retired executor waiter still owns tensor")

    async def test_queue_deadline_fails_and_releases_without_starting_native_work(self):
        await self.block_native()
        self.encoder.send_timeout = 0.03
        state = self.stage("expired")
        task = self.send("expired")
        with self.assertRaises(server.MMError) as raised:
            await asyncio.wait_for(task, 1)
        self.assertEqual(raised.exception.code, 504)
        self.assertNotIn("expired", self.native_calls)
        self.assertIsNone(state.embedding_data)
        self.assertEqual(self.encoder.transfer_queue.snapshot()["expired"], 1)

    async def test_client_cancel_discards_queued_send(self):
        await self.block_native()
        state = self.stage("cancelled")
        task = self.send("cancelled")
        await until(lambda: self.encoder.transfer_queue.snapshot()["queued"] == 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertNotIn("cancelled", self.native_calls)
        self.assertIsNone(state.embedding_data)

    async def test_sweeper_retires_queued_send_despite_another_stuck_request(self):
        await self.block_native()
        state = self.stage("stale")
        task = self.send("stale")
        await until(lambda: self.encoder.transfer_queue.snapshot()["queued"] == 1)
        registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=0.02)
        registry.on_release = self.encoder.release_request
        try:
            await registry.publish("stale", 64, 4, 4)
            await until(lambda: "stale" not in self.encoder.req_states)
            await until(
                lambda: not registry._stale_release_tasks and not registry._pending_at
            )
            with self.assertRaises(InternalError):
                await task
            self.assertIsNone(state.embedding_data)
        finally:
            registry._sweeper_task.cancel()
            await asyncio.gather(registry._sweeper_task, return_exceptions=True)

    async def test_ack_backlog_is_retirable_after_rdma_has_completed(self):
        self.stage("blocked")
        self.ack_block = "tcp://127.0.0.1:1234"
        blocked = self.send("blocked")
        await until(lambda: self.ack_block in self.ack_calls)
        state = self.stage("queued-ack")
        queued = self.send("queued-ack", port=1235)
        await until(lambda: self.encoder.ack_queue.snapshot()["queued"] == 1)
        self.assertIn("queued-ack", self.native_calls)

        await asyncio.wait_for(self.encoder.release_request("queued-ack"), 1)
        with self.assertRaisesRegex(InternalError, "retired before ack"):
            await queued
        self.assertNotIn("tcp://127.0.0.1:1235", self.ack_calls)
        self.assertIsNone(state.embedding_data)
        self.assertFalse(blocked.done())

    async def test_eagain_retires_queued_sibling_but_does_not_start_its_ack(self):
        state = self.stage("same")
        self.ack_block = "tcp://127.0.0.1:1234"
        self.ack_error = True
        first = self.send("same")
        await until(lambda: self.ack_block in self.ack_calls)
        second = self.send("same", port=1235)
        await until(lambda: self.encoder.ack_queue.snapshot()["queued"] == 1)
        self.unblock.set()
        results = await asyncio.wait_for(
            asyncio.gather(first, second, return_exceptions=True), 1
        )
        self.assertIsInstance(results[0], zmq.Again)
        self.assertIsInstance(results[1], InternalError)
        self.assertEqual(self.ack_calls, [self.ack_block])
        self.assertIsNone(state.embedding_data)

    async def test_legacy_executor_queue_retains_embedding_after_release_request(self):
        # Reproduce the old unbounded executor submission with the same send
        # and release chain. The worker is controlled, so teardown always drains.
        self.use_legacy_submission()
        await self.block_native()
        state = self.stage("legacy-queued")
        self.send("legacy-queued")
        await until(lambda: state.active_sends == 1)
        release = asyncio.create_task(self.encoder.release_request("legacy-queued"))
        self.tasks.append(release)
        await until(lambda: state.release_requested)
        await asyncio.sleep(0.02)
        self.assertFalse(release.done())
        self.assertIsNotNone(state.embedding_data)
        self.assertEqual(self.encoder.lifecycle_snapshot()["staged_cpu_bytes"], 128)
        self.assertNotIn("legacy-queued", self.native_calls)

    async def test_legacy_ack_queue_retains_embedding_after_successful_rdma(self):
        self.use_legacy_submission()
        self.stage("blocked")
        self.ack_block = "tcp://127.0.0.1:1234"
        self.send("blocked")
        await until(lambda: self.ack_block in self.ack_calls)
        state = self.stage("legacy-ack")
        self.send("legacy-ack", port=1235)
        await until(lambda: "legacy-ack" in self.native_calls)
        release = asyncio.create_task(self.encoder.release_request("legacy-ack"))
        self.tasks.append(release)
        await until(lambda: state.release_requested)
        await asyncio.sleep(0.02)
        self.assertFalse(release.done())
        self.assertIsNotNone(state.embedding_data)
        self.assertEqual(self.encoder.lifecycle_snapshot()["staged_cpu_bytes"], 128)
        self.assertNotIn("tcp://127.0.0.1:1235", self.ack_calls)

    async def test_http_reports_transfer_queue_timeout_as_504(self):
        await self.block_native()
        self.encoder.send_timeout = 0.03
        self.stage("http-expired")
        with (
            patch.object(http_server, "dp_dispatcher", None),
            patch.object(http_server, "encoder", self.encoder),
        ):
            response = await asyncio.wait_for(http_server.handle_send_request({
                "req_id": "http-expired",
                "prefill_host": "127.0.0.1",
                "embedding_port": 1235,
                "session_id": "http-expired",
                "buffer_address": 1,
            }), 1)
        self.assertEqual(response.status_code, 504)
        self.assertNotIn("http-expired", self.encoder.req_states)
        self.assertNotIn("http-expired", self.native_calls)

    async def test_running_native_transfer_is_drained_even_after_repeated_cancel(self):
        task = await self.block_native()
        state = self.encoder.req_states["blocked"]
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        release = asyncio.create_task(self.encoder.release_request("blocked"))
        self.tasks.append(release)
        await asyncio.sleep(0.01)
        self.assertFalse(task.done())
        self.assertFalse(release.done())
        self.assertIsNotNone(state.embedding_data)
        self.unblock.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        await release
        self.assertIsNone(state.embedding_data)

    async def test_retirement_wakes_send_waiting_for_embedding(self):
        state = ReqState("pre-forward")
        self.encoder.req_states[state.req_id] = state
        task = self.send(state.req_id)
        await until(lambda: state.active_sends == 1)
        await asyncio.wait_for(self.encoder.release_request(state.req_id), 1)
        with self.assertRaisesRegex(InternalError, "released"):
            await task
        self.assertEqual(self.native_calls, [])
        self.assertNotIn(state.req_id, self.encoder.req_states)

    async def test_success_counts_destination_and_leaves_no_transfer_waiters(self):
        self.stage("healthy")
        self.assertTrue(await self.send("healthy"))
        server.meta_registry.note_send_done.assert_awaited_once_with(
            "healthy", 1, "127.0.0.1:1234"
        )
        self.assertEqual(self.encoder.transfer_queue.snapshot()["admitted"], 0)
        self.assertEqual(self.encoder.ack_queue.snapshot()["admitted"], 0)
        self.assertEqual(self.encoder.req_states["healthy"].active_sends, 0)

    async def test_glm_send_uses_embedding_address_despite_legacy_pool_setting(self):
        state = self.stage("direct")
        embedding = state.embedding_data.embedding
        engine = Mock()
        engine.transfer_sync.return_value = 0
        with patch.dict(os.environ, {"SGLANG_MC_RDMA_POOL_MAX_MB": "4096"}):
            self.encoder._glm_source_buffers = glm_rdma.GlmSourceBuffers(engine)
            self.assertTrue(await self.send("direct"))
        engine.register.assert_called_once_with(embedding.data_ptr(), embedding.nbytes)
        engine.transfer_sync.assert_called_once_with(
            "direct", embedding.data_ptr(), 1, embedding.nbytes
        )
        engine.deregister.assert_called_once_with(embedding.data_ptr())
        await self.encoder.release_request("direct")


class TestGlmDirectSources(unittest.TestCase):
    def test_concurrent_destinations_keep_source_registered_until_both_finish(self):
        embedding = torch.ones((4, 4))
        engine = Mock()
        sources = glm_rdma.GlmSourceBuffers(engine)
        started = {name: threading.Event() for name in ("first", "second")}
        finish = {name: threading.Event() for name in started}

        def transfer(session, address, destination, size):
            self.assertEqual(address, embedding.data_ptr())
            started[session].set()
            if not finish[session].wait(3):
                raise RuntimeError("test failed to release transfer")
            return 0

        engine.transfer_sync.side_effect = transfer
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                name: executor.submit(sources.transfer, embedding, name, 1)
                for name in started
            }
            try:
                for event in started.values():
                    self.assertTrue(event.wait(2))
                engine.register.assert_called_once_with(
                    embedding.data_ptr(), embedding.nbytes
                )
                finish["first"].set()
                self.assertEqual(futures["first"].result(timeout=2), 0)
                engine.deregister.assert_not_called()
                finish["second"].set()
                self.assertEqual(futures["second"].result(timeout=2), 0)
            finally:
                for event in finish.values():
                    event.set()
        engine.deregister.assert_called_once_with(embedding.data_ptr())

    def test_failed_transfer_keeps_source_registered_through_later_success(self):
        for raises in (False, True):
            with self.subTest(raises=raises):
                embedding = torch.ones((4, 4))
                address = embedding.data_ptr()
                engine = Mock()
                sources = glm_rdma.GlmSourceBuffers(engine)
                engine.transfer_sync.return_value = -1
                if raises:
                    engine.transfer_sync.side_effect = RuntimeError("write failed")
                with patch.object(glm_rdma, "defer_rdma_release") as deferred:
                    if raises:
                        with self.assertRaisesRegex(RuntimeError, "write failed"):
                            sources.transfer(embedding, "failed", 1)
                    else:
                        self.assertEqual(sources.transfer(embedding, "failed", 1), -1)
                engine.deregister.assert_not_called()
                deferred.assert_called_once()
                release, owner = deferred.call_args.args
                self.assertIs(owner, embedding)

                # A successful sibling must not retire the failed write's MR.
                engine.transfer_sync.side_effect = None
                engine.transfer_sync.return_value = 0
                self.assertEqual(sources.transfer(embedding, "healthy", 1), 0)
                engine.register.assert_called_once_with(address, embedding.nbytes)
                engine.deregister.assert_not_called()
                release(owner)
                engine.deregister.assert_called_once_with(address)

    def test_registration_failure_does_not_leave_a_live_refcount(self):
        embedding = torch.ones((4, 4))
        engine = Mock()
        sources = glm_rdma.GlmSourceBuffers(engine)
        engine.register.side_effect = RuntimeError("register failed")
        with self.assertRaisesRegex(RuntimeError, "register failed"):
            sources.transfer(embedding, "failed", 1)
        engine.transfer_sync.assert_not_called()
        engine.deregister.assert_not_called()
        engine.register.side_effect = None
        engine.transfer_sync.return_value = 0
        self.assertEqual(sources.transfer(embedding, "retry", 1), 0)
        self.assertEqual(engine.register.call_count, 2)
        engine.deregister.assert_called_once_with(embedding.data_ptr())


class TestTransferAdmissionRaces(unittest.IsolatedAsyncioTestCase):
    async def test_retirement_wins_when_slot_is_available_without_leaking_permit(self):
        queue = EncoderTransferQueue(1)
        retired = asyncio.Event()
        retired.set()
        with self.assertRaises(TransferRequestReleased):
            async with queue.slot("retired", retired, 1):
                self.fail("retired request was admitted")
        async with queue.slot("next", asyncio.Event(), 0.1):
            self.assertEqual(queue.snapshot()["admitted"], 1)

    async def test_cancelled_waiter_does_not_steal_next_permit(self):
        queue = EncoderTransferQueue(1)
        async with queue.slot("running", asyncio.Event(), 1):
            async def wait():
                async with queue.slot("cancelled", asyncio.Event(), 1):
                    self.fail("cancelled request was admitted")
            task = asyncio.create_task(wait())
            await until(lambda: queue.snapshot()["queued"] == 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        async with queue.slot("next", asyncio.Event(), 0.1):
            self.assertEqual(queue.snapshot()["admitted"], 1)

    async def test_retirement_between_acquisition_and_submission_refunds_permit(self):
        class RetireDuringCleanup(asyncio.Event):
            async def wait(self):
                try:
                    return await super().wait()
                except asyncio.CancelledError:
                    self.set()
                    raise

        queue = EncoderTransferQueue(1)
        with self.assertRaises(TransferRequestReleased):
            async with queue.slot("race", RetireDuringCleanup(), 1):
                self.fail("request retired before submission was admitted")
        async with queue.slot("next", asyncio.Event(), 0.1):
            self.assertEqual(queue.snapshot()["admitted"], 1)


if __name__ == "__main__":
    unittest.main()
