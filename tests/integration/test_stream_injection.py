"""A backend on the application's own clients, against real Redis."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI, Request
from redis.asyncio import Redis, RedisCluster

from redis_fastapi import (
    FastAPIRedis,
    StreamBackend,
    StreamBackendDep,
    StreamConfigurationError,
    StreamWriter,
    add_redis_streams,
)
from redis_fastapi.config import get_settings
from redis_fastapi.stream_backend import probe_streams_support
from tests.conftest import REDIS_URL


def _options(prefix: str, **overrides: object) -> dict[str, object]:
    """Every backend option, so the backend reads no ``REDIS_*`` settings."""
    return {
        "key_prefix": prefix,
        "active_ttl": 600,
        "done_ttl": 120,
        "gone_ttl": 3600,
        "maxlen": 10_000,
        "max_event_bytes": 65_536,
        "heartbeat_interval": 1,
        "alive_ttl": 3,
        "replay_page": 100,
        "max_readers": 256,
        "max_connection_duration": 300,
        "read_retry_window": 30,
        "send_timeout": 10,
        **overrides,
    }


@pytest.fixture(params=["standalone", "cluster", "enterprise"])
async def clients(request) -> AsyncIterator[tuple]:
    topology = request.param
    if topology == "standalone":
        url = REDIS_URL
        factory = Redis
    else:
        url = os.getenv(f"STREAMS_TEST_{topology.upper()}_URL")
        if not url:
            pytest.skip(f"no explicitly supplied {topology} endpoint")
        factory = RedisCluster if topology == "cluster" else Redis
    prefix = f"injected:{uuid.uuid4().hex}"
    commands = factory.from_url(url, socket_timeout=30, max_connections=4)
    readers = factory.from_url(
        url,
        socket_timeout=30,
        max_connections=1,
        client_name=prefix + ":reader",
    )
    available = False
    try:
        try:
            await commands.ping()
            available = True
        except Exception:
            if topology == "standalone":
                pytest.skip("standalone Redis not reachable")
            raise
        if await probe_streams_support(commands) is not True:
            pytest.skip(f"{topology} Redis without resumable-stream support")
        yield commands, readers, prefix
    finally:
        # Remove only this fixture's namespace, including on external endpoints.
        if available:
            async for key in commands.scan_iter(match=f"{prefix}*"):
                await commands.delete(key)
        await readers.aclose()
        await commands.aclose()


@pytest.mark.integration
async def test_injected_http_and_lifecycle(clients, monkeypatch):
    commands, readers, prefix = clients
    monkeypatch.setenv("REDIS_URL", "invalid://ambient")
    monkeypatch.setenv("REDIS_STREAM_MAX_READERS", "invalid")
    monkeypatch.setenv("REDIS_PREFIX", "{invalid}")
    get_settings.cache_clear()
    backend = StreamBackend(commands, blocking_client=readers, **_options(prefix))
    stopped = asyncio.Event()
    started = asyncio.Event()
    blocker = asyncio.Event()

    async def producer(writer: StreamWriter) -> None:
        started.set()
        try:
            await blocker.wait()
        finally:
            stopped.set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with backend:
            yield

    app = FastAPI(lifespan=lifespan)
    add_redis_streams(app, backend=backend)

    @app.get("/events/{stream_id}")
    async def events(stream_id: str, request: Request, streams: StreamBackendDep):
        return await streams.sse(stream_id, request=request, scope="chat:1")

    try:
        async with app.router.lifespan_context(app):
            assert backend.supported is True
            writer = await backend.open(scope="chat:1", stream_id="finished")
            cursor = await writer.append("delta", {"text": "hello"})
            final = await writer.finish("message", {"text": "done"})
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as http:
                response = await http.get("/events/finished")
                assert response.status_code == 200
                assert '"hello"' in response.text and '"done"' in response.text
                response = await http.get(
                    "/events/finished", headers={"Last-Event-ID": cursor}
                )
                assert '"hello"' not in response.text and '"done"' in response.text
                assert (
                    await http.get("/events/finished", headers={"Last-Event-ID": final})
                ).status_code == 204
                assert (await http.get("/events/unknown")).status_code == 404
                assert (
                    await http.get("/events/finished?after=invalid")
                ).status_code == 422
            pending = await backend.open(scope="chat:2", stream_id="pending")
            backend.spawn(pending, producer)
            await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(stopped.wait(), 2)
        assert not backend._tasks
        # Both client objects remain usable after backend shutdown.
        assert await commands.ping()
        assert await readers.ping()
        assert await commands.exists(f"{prefix}:stream:{{chat:1}}:finished")
    finally:
        await backend.aclose()
        get_settings.cache_clear()


@pytest.mark.integration
async def test_two_apps_keep_policy_and_reads_separate(clients):
    commands, readers, prefix = clients
    first = StreamBackend(
        commands,
        blocking_client=readers,
        **_options(prefix + ":a", max_event_bytes=32, max_readers=1),
    )
    second = StreamBackend(
        commands,
        blocking_client=readers,
        **_options(prefix + ":b", max_event_bytes=1000, max_readers=2),
    )
    apps = [FastAPI(), FastAPI()]
    for app, backend in zip(apps, [first, second], strict=True):
        FastAPIRedis(app).streams(backend=backend)
    async with first, second:
        a = await first.open(scope="chat:1", stream_id="same")
        b = await second.open(scope="chat:1", stream_id="same")
        with pytest.raises(ValueError, match="stream_max_event_bytes"):
            await a.append("delta", {"text": "x" * 40})
        await a.finish("message", {"text": "A"})
        await b.finish("message", {"text": "B"})
        for backend, expected in [(first, "A"), (second, "B")]:
            async with await backend.follow("same", scope="chat:1") as events:
                assert [event.data async for event in events] == [{"text": expected}]

        # Observe the blocking XREAD on Redis before publishing through the
        # independent command pool. No scheduling assumption about anext().
        waiting = await first.open(scope="chat:2", stream_id="live")
        async with await first.follow("live", scope="chat:2") as events:
            next_event = asyncio.create_task(anext(events))

            try:
                await asyncio.wait_for(wait_for_xread(commands, prefix), 2)
                await asyncio.wait_for(waiting.finish("message", {"ok": True}), 2)
                assert (await asyncio.wait_for(next_event, 2)).final
            finally:
                next_event.cancel()
                await asyncio.gather(next_event, return_exceptions=True)


@pytest.mark.integration
async def test_failed_app_startup_leaves_external_clients_open(clients):
    commands, readers, prefix = clients
    backend = StreamBackend(commands, blocking_client=readers, **_options(prefix))

    @asynccontextmanager
    async def broken(app: FastAPI):
        async with backend:
            raise RuntimeError("application startup failed")
        yield

    app = FastAPI(lifespan=broken)
    add_redis_streams(app, backend=backend)
    with pytest.raises(RuntimeError, match="application startup"):
        async with app.router.lifespan_context(app):
            pass
    assert not backend._running
    assert await commands.ping() and await readers.ping()
    async with backend:
        with pytest.raises(StreamConfigurationError, match="already running"):
            await backend._initialize()


async def wait_for_xread(commands, prefix):
    def rows(value):
        if isinstance(value, list):
            for item in value:
                yield from rows(item)
        elif isinstance(value, dict):
            if "cmd" in value:
                yield value
            else:
                for item in value.values():
                    yield from rows(item)

    while True:
        kwargs = (
            {"target_nodes": RedisCluster.ALL_NODES}
            if isinstance(commands, RedisCluster)
            else {}
        )
        connections = await commands.client_list(**kwargs)
        if any(
            row.get("name") == prefix + ":reader" and row.get("cmd") == "xread"
            for row in rows(connections)
        ):
            return
        await asyncio.sleep(0.01)


@pytest.mark.integration
async def test_restart_does_not_revive_blocked_readers(clients):
    commands, readers, prefix = clients
    backend = StreamBackend(commands, blocking_client=readers, **_options(prefix))
    await backend._initialize()
    try:
        await backend.open(scope="chat:1", stream_id="live")
        async with await backend.follow("live", scope="chat:1") as events:
            pending = asyncio.create_task(anext(events))
            try:
                await asyncio.wait_for(wait_for_xread(commands, prefix), 2)
                await backend.aclose()
                await backend._initialize()
                with pytest.raises(StopAsyncIteration):
                    await asyncio.wait_for(pending, 2)
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
    finally:
        await backend.aclose()


@pytest.mark.integration
async def test_restart_during_gap_drops_old_buffered_event(clients):
    commands, readers, prefix = clients
    backend = StreamBackend(
        commands, blocking_client=readers, **_options(prefix, replay_page=1)
    )
    async with backend:
        writer = await backend.open(scope="chat:1", stream_id="gap")
        ids = [await writer.append("delta", {"n": n}) for n in range(4)]
        await writer.finish("message", {})
        async with await backend.follow("gap", scope="chat:1") as events:
            assert (await anext(events)).data == {"n": 0}
            await commands.xdel(f"{prefix}:stream:{{chat:1}}:gap", *ids[1:3])
            assert (await anext(events)).type == "stream.gap"
            await backend.aclose()
            await backend._initialize()
            with pytest.raises(StopAsyncIteration):
                await anext(events)


@pytest.mark.integration
async def test_redis_lifespan_runs_an_injected_backend(clients, monkeypatch):
    commands, readers, prefix = clients
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    get_settings.cache_clear()
    backend = StreamBackend(commands, blocking_client=readers, **_options(prefix))
    app = FastAPI()
    FastAPIRedis(app).lifespan().streams(backend=backend)
    try:
        async with app.router.lifespan_context(app):
            assert backend._running and backend.supported is True
            writer = await backend.open(scope="chat:1", stream_id="r1")
            await writer.finish("message", {})
        assert not backend._running
        assert await commands.ping() and await readers.ping()
    finally:
        get_settings.cache_clear()
