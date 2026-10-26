"""Configuration and registration contracts for explicit streaming integration."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from redis.asyncio import Redis
from starlette.requests import Request

from redis_fastapi import (
    FastAPIRedis,
    StreamBackend,
    StreamConfigurationError,
    StreamSettings,
    add_redis_streams,
    get_stream_backend,
)
from redis_fastapi.config import RedisSettings, get_settings


def test_settings_are_environment_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_STREAM_MAX_READERS", "invalid")
    monkeypatch.setenv("REDIS_PREFIX", "{invalid}")
    get_settings.cache_clear()
    try:
        assert StreamSettings().stream_max_readers == 256
        backend = StreamBackend(Redis(socket_timeout=30))
        assert backend.settings.prefix == "redis:fastapi"
        with pytest.raises(ValidationError):
            backend.settings.prefix = "changed"
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "values",
    [
        {"stream_max_readers": 0},
        {"stream_heartbeat_interval": 0},
        {"stream_send_timeout": float("nan")},
        {"stream_max_connection_duration": float("inf")},
        {"socket_timeout": 30},
    ],
)
def test_invalid_policy_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        StreamSettings(**values)


def test_explicit_environment_adapter() -> None:
    settings = RedisSettings(
        _env_file=None,
        prefix="adapter",
        socket_timeout=60,
        stream_max_readers=12,
        otel_enabled=True,
    )
    policy = StreamSettings.from_redis_settings(settings)
    assert policy.prefix == "adapter"
    assert policy.stream_max_readers == 12
    assert "socket_timeout" not in policy.model_dump()
    assert "otel_enabled" not in policy.model_dump()


def test_injected_registration_conflicts() -> None:
    app = FastAPI()
    backend = StreamBackend(Redis(socket_timeout=30))
    assert add_redis_streams(app, backend=backend) is backend
    assert add_redis_streams(app, backend=backend) is backend
    for kwargs in [
        {},
        {"backend": StreamBackend(Redis(socket_timeout=30))},
        {"backend": backend, "manage_lifespan": False},
    ]:
        with pytest.raises(StreamConfigurationError, match="conflicting"):
            add_redis_streams(app, **kwargs)
    with pytest.raises(StreamConfigurationError, match="coder"):
        add_redis_streams(app, backend=backend, coder=object)


async def test_dependency_is_app_scoped() -> None:
    apps = [FastAPI(), FastAPI()]
    backends = [
        StreamBackend(Redis(socket_timeout=30)),
        StreamBackend(Redis(socket_timeout=30)),
    ]
    for app, backend in zip(apps, backends, strict=True):
        FastAPIRedis(app).streams(backend=backend, manage_lifespan=False)
        request = Request({"type": "http", "app": app, "headers": []})
        assert await get_stream_backend(request) is backend


@pytest.mark.parametrize("shared_pool", [False, True])
async def test_shared_reader_connections_rejected(shared_pool: bool) -> None:
    commands = Redis(socket_timeout=30)
    reader = (
        Redis(connection_pool=commands.connection_pool) if shared_pool else commands
    )
    backend = StreamBackend(commands, blocking_client=reader)
    with pytest.raises(StreamConfigurationError, match="independent"):
        await backend.initialize()
    assert not backend._running
    await commands.aclose()


async def test_reader_socket_timeout_validated() -> None:
    command = Redis(socket_timeout=1)  # Short command timeout is allowed.
    reader = Redis(socket_timeout=1)
    backend = StreamBackend(command, blocking_client=reader)
    with pytest.raises(StreamConfigurationError, match="socket_timeout"):
        await backend.initialize()
    await reader.aclose()
    await command.aclose()


@pytest.mark.parametrize("failure", [RuntimeError("startup"), asyncio.CancelledError()])
async def test_startup_failure_releases_lifecycle(
    failure: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Redis(socket_timeout=30)
    backend = StreamBackend(client)
    monkeypatch.setattr(backend, "detect", AsyncMock(side_effect=failure))
    with pytest.raises(type(failure)):
        async with backend:
            pytest.fail("startup should fail")
    assert backend._closing
    assert not backend._running
    assert not backend._tasks
    monkeypatch.setattr(backend, "detect", AsyncMock(return_value=True))
    async with backend:
        with pytest.raises(StreamConfigurationError, match="owner"):
            await backend.initialize()
    await client.aclose()


async def test_managed_registration_preserves_lifespan_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    client = Redis(socket_timeout=30)
    backend = StreamBackend(client)
    monkeypatch.setattr(backend, "detect", AsyncMock(return_value=True))

    @asynccontextmanager
    async def original(app: FastAPI):
        assert backend._running
        events.append("start")
        yield {"value": 42}
        assert backend._running
        events.append("stop")

    app = FastAPI(lifespan=original)
    add_redis_streams(app, backend=backend)
    async with app.router.lifespan_context(app) as state:
        assert state == {"value": 42}
    assert events == ["start", "stop"]
    assert not backend._running
    await client.aclose()
