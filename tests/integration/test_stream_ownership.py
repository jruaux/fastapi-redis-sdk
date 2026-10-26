"""Concurrent creation, takeover and terminal-state fencing against real Redis."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError
from starlette.requests import Request

from redis_fastapi import StreamFenced, StreamNotFound
from redis_fastapi.stream_backend import _keys
from tests.stream_support import gated_exec

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("exclusive", [True, False])
async def test_concurrent_open_has_one_creation_winner(
    stream_case, monkeypatch, exclusive
):
    backend, client, _ = stream_case
    with gated_exec(client, monkeypatch, count=2) as gate:
        tasks = [
            asyncio.create_task(
                backend.open(scope="chat:1", stream_id="same", exclusive=exclusive)
            )
            for _ in range(2)
        ]
        try:
            await asyncio.wait_for(gate["entered"].wait(), 2)
            gate["release"].set()
            writers = await asyncio.wait_for(asyncio.gather(*tasks), 3)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    assert sorted(writer.created for writer in writers) == [False, True]
    winner = next(writer for writer in writers if writer.created)
    loser = next(writer for writer in writers if not writer.created)
    with pytest.raises(StreamFenced):
        await loser.append("delta", {"by": "loser"})
    await winner.append("delta", {"by": "winner"})
    entries = await client.xrange(_keys("chat:1", "same").stream)
    assert [json.loads(fields["d"]) for _, fields in entries] == [{"by": "winner"}]


@pytest.mark.parametrize("finish", [False, True])
async def test_takeover_between_check_and_exec_rejects_stale_write(
    stream_case, monkeypatch, finish
):
    backend, client, _ = stream_case
    old = await backend.open(scope="chat:1", stream_id="race")
    await old.append("delta", {"by": "before"})
    with gated_exec(client, monkeypatch) as gate:
        operation = old.finish if finish else old.append
        pending = asyncio.create_task(
            operation("message" if finish else "delta", {"by": "stale"})
        )
        try:
            await asyncio.wait_for(gate["entered"].wait(), 2)
            current = await backend.attach("race", scope="chat:1")
            gate["release"].set()
            with pytest.raises(StreamFenced):
                await asyncio.wait_for(pending, 2)
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    with pytest.raises(StreamFenced):
        await old.suspend(3600)
    await current.finish("message", {"by": "current"})
    entries = await client.xrange(_keys("chat:1", "race").stream)
    assert [json.loads(fields["d"])["by"] for _, fields in entries] == [
        "before",
        "current",
    ]
    assert [int(fields["q"]) for _, fields in entries] == [1, 2]


async def test_concurrent_appends_on_one_writer_keep_distinct_sequences(stream_case):
    backend, client, _ = stream_case
    writer = await backend.open(scope="chat:1", stream_id="sequence")
    results = await asyncio.gather(
        *[writer.append("delta", {"n": n}) for n in range(20)]
    )
    assert len({result.id for result in results}) == 20
    entries = await client.xrange(_keys("chat:1", "sequence").stream)
    assert [int(fields["q"]) for _, fields in entries] == list(range(1, 21))


async def test_lost_producer_seal_rechecks_a_recovered_heartbeat(
    stream_case, monkeypatch
):
    backend, client, _ = stream_case
    writer = await backend.open(scope="chat:1", stream_id="live")
    await writer.append("delta", {"n": 0})
    keys = _keys("chat:1", "live")
    await client.delete(keys.alive)
    with gated_exec(client, monkeypatch) as gate:
        pending = asyncio.create_task(
            backend._end_lost(SimpleNamespace(keys=keys, stream_id="live"))
        )
        try:
            await asyncio.wait_for(gate["entered"].wait(), 2)
            await writer.append("delta", {"n": 1})
            gate["release"].set()
            await asyncio.wait_for(pending, 2)
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    assert await client.hget(keys.meta, "status") == "running"
    assert all(fields.get("f") != "1" for _, fields in await client.xrange(keys.stream))


async def test_cancel_cannot_recreate_expired_metadata(stream_case, monkeypatch):
    backend, client, _ = stream_case
    await backend.open(scope="chat:1", stream_id="gone")
    keys = _keys("chat:1", "gone")
    with gated_exec(client, monkeypatch) as gate:
        pending = asyncio.create_task(backend.cancel("gone", scope="chat:1"))
        try:
            await asyncio.wait_for(gate["entered"].wait(), 2)
            await client.delete(keys.meta)
            gate["release"].set()
            with pytest.raises(StreamNotFound):
                await asyncio.wait_for(pending, 2)
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    assert not await client.exists(keys.meta)


async def test_final_cursor_is_complete_before_metadata_mirror(
    stream_case, monkeypatch
):
    backend, client, _ = stream_case
    writer = await backend.open(scope="chat:1", stream_id="final")
    entered = asyncio.Event()
    release = asyncio.Event()
    original = backend._after_final

    async def delayed(*args, **kwargs):
        entered.set()
        await asyncio.wait_for(release.wait(), 3)
        await original(*args, **kwargs)

    monkeypatch.setattr(backend, "_after_final", delayed)
    pending = asyncio.create_task(writer.finish("message", {}))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        final_id = (await client.xrevrange(_keys("chat:1", "final").stream, count=1))[
            0
        ][0]
        request = Request(
            {
                "type": "http",
                "headers": [(b"last-event-id", final_id.encode())],
                "query_string": b"",
            }
        )
        response = await backend.sse("final", request=request, scope="chat:1")
        assert response.status_code == 204
    finally:
        release.set()
        await asyncio.wait_for(pending, 2)


async def test_creation_timeout_retry_keeps_the_same_claim(stream_case, monkeypatch):
    backend, client, _ = stream_case
    original = type(client).pipeline
    injected = False

    class LostReply:
        def __init__(self, pipe):
            self.pipe = pipe

        def __getattr__(self, name):
            return getattr(self.pipe, name)

        async def __aenter__(self):
            await self.pipe.__aenter__()
            return self

        async def __aexit__(self, *args):
            return await self.pipe.__aexit__(*args)

        async def execute(self, *args, **kwargs):
            nonlocal injected
            result = await self.pipe.execute(*args, **kwargs)
            if not injected:
                injected = True
                raise RedisTimeoutError("committed claim reply lost")
            return result

    with monkeypatch.context() as patch:
        patch.setattr(
            type(client),
            "pipeline",
            lambda instance, *a, **kw: LostReply(original(instance, *a, **kw)),
        )
        writer = await backend.open(scope="chat:1", stream_id="retry")
    assert writer.created and injected
    assert not (await backend.open(scope="chat:1", stream_id="retry")).created
    await writer.append("delta", {})


async def test_reader_closes_when_started_history_disappears(stream_case):
    backend, client, _ = stream_case
    writer = await backend.open(scope="chat:1", stream_id="missing")
    await writer.append("delta", {})
    keys = _keys("chat:1", "missing")
    await client.delete(keys.stream)
    assert await backend._check_producer(SimpleNamespace(keys=keys)) == "gone"
    assert not await client.exists(keys.stream)
