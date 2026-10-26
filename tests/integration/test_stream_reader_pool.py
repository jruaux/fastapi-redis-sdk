"""Fluent setup isolates bounded blocking connections from ordinary commands."""

import asyncio

import pytest
from fastapi import FastAPI
from redis.asyncio import RedisCluster

from redis_fastapi import FastAPIRedis
from redis_fastapi.config import get_settings
from redis_fastapi.stream_backend import _keys, _ReadPlan
from redis_fastapi.streams import _SSEResponse

pytestmark = pytest.mark.integration


def policy(monkeypatch, **values):
    for key, value in values.items():
        monkeypatch.setenv("REDIS_" + key.upper(), str(value))
    get_settings.cache_clear()


async def wait_for_blocked(observer, name):
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
            if isinstance(observer, RedisCluster)
            else {}
        )
        clients = await observer.client_list(**kwargs)
        if any(
            row.get("name") == name and row.get("cmd") == "xread"
            for row in rows(clients)
        ):
            return
        await asyncio.sleep(0.01)


async def test_one_blocked_reader_leaves_command_pool_available_and_rejects_excess(
    stream_case, monkeypatch
):
    _, observer, prefix = stream_case
    policy(
        monkeypatch,
        max_connections=1,
        stream_reader_max_connections=1,
        stream_max_readers=5,
        stream_heartbeat_seconds=1,
        stream_alive_ttl_seconds=10,
    )
    app = FastAPI()
    FastAPIRedis(app).lifespan().streams()
    state = None
    reader = None
    async with app.router.lifespan_context(app):
        state = app.state._redis
        backend = app.state._redis_streams
        reader = state.get_stream_reader()
        commands = state.get_async_client()
        assert reader is not commands
        if isinstance(reader, RedisCluster):
            await reader.client_setname(prefix, target_nodes=RedisCluster.ALL_NODES)
            assert all(node.max_connections == 1 for node in reader.get_nodes())
        else:
            await reader.client_setname(prefix)
            assert reader.connection_pool is not commands.connection_pool
            assert reader.connection_pool.max_connections == 1
        writer = await backend.open(scope="chat:1", stream_id="idle")
        # Exercise the real backend blocking loop without the separate baseline
        # Cluster XREAD-in-MULTI replay bug (fixed by the explicit-injection PR).
        plan = _ReadPlan(
            keys=_keys("chat:1", "idle"),
            scope="chat:1",
            stream_id="idle",
            cursor=(0, 0),
            last_seq=0,
        )
        entered = asyncio.Event()
        sent = []

        async def body():
            async for event in backend._events(plan):
                if event is not None:
                    yield "data: done\n\n"

        async def receive():
            await asyncio.Event().wait()

        async def discard(message):
            if message["type"] == "http.response.start":
                entered.set()

        response = _SSEResponse(body(), backend=backend, send_timeout=2)
        task = asyncio.create_task(
            response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, discard
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await asyncio.wait_for(wait_for_blocked(observer, prefix), 2)
            # No connection from the one-slot command pool is held by XREAD.
            await asyncio.wait_for(commands.set(prefix + ":ordinary", "ok"), 2)
            assert await commands.get(prefix + ":ordinary") == b"ok"
            assert backend._readers == 1
            extra = _SSEResponse(body(), backend=backend, send_timeout=2)

            async def capture(message):
                sent.append(message)

            await extra(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, capture
            )
            assert sent[0]["status"] == 503
            assert backend._readers == 1
            await writer.finish("message", {})
            await asyncio.wait_for(task, 2)
            assert backend._readers == 0
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert state.stream_pool is None and state.stream_cluster is None
    assert state._stream_client is None


@pytest.mark.parametrize("cancel", [False, True])
async def test_startup_failure_closes_both_pools(stream_case, monkeypatch, cancel):
    _, observer, _ = stream_case
    policy(monkeypatch, stream_reader_max_connections=1, stream_heartbeat_seconds=0.1)
    app = FastAPI()
    FastAPIRedis(app).lifespan().streams()
    backend = app.state._redis_streams
    captured = {}

    async def broken():
        state = app.state._redis
        captured["state"] = state
        captured["reader"] = state.get_stream_reader()
        await captured["reader"].ping()
        if cancel:
            raise asyncio.CancelledError()
        raise RuntimeError("startup probe failed")

    monkeypatch.setattr(backend, "detect", broken)
    # _start_streams keeps its pre-existing best-effort policy for ordinary
    # probe failures, so inject failure in the application's original lifespan.
    if not cancel:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def failed_app(app):
            raise RuntimeError("application startup failed")
            yield

        # The fluent wrapper must surround the application lifespan.
        app = FastAPI(lifespan=failed_app)
        FastAPIRedis(app).lifespan().streams()
        backend = app.state._redis_streams
        monkeypatch.setattr(backend, "detect", broken)
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        async with app.router.lifespan_context(app):
            pass
    state = captured["state"]
    assert state.stream_pool is None and state.stream_cluster is None
    assert state.async_pool is None and state.async_cluster is None
    assert backend._closing
    assert await observer.ping()
