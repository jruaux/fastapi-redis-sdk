"""Reader pool construction preserves identity while enforcing its own cap."""

import pytest

from redis_fastapi.config import get_settings
from redis_fastapi.deps import _PoolState
from redis_fastapi.exceptions import StreamConfigurationError


@pytest.fixture
def settings_env(monkeypatch):
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


async def test_reader_cap_overrides_url_command_cap(settings_env):
    settings_env.setenv(
        "REDIS_URL", "redis://user:pass@localhost:6379/3?max_connections=999"
    )
    settings_env.setenv("REDIS_STREAM_READER_MAX_CONNECTIONS", "2")
    state = _PoolState()
    state.start_stream_reader()
    try:
        assert state.stream_pool.max_connections == 2
        assert state.stream_pool.connection_kwargs["db"] == 3
        assert state.stream_pool.connection_kwargs["username"] == "user"
        assert state.stream_pool.connection_kwargs["password"] == "pass"
        assert get_settings().url.endswith("max_connections=999")
    finally:
        await state.close_stream_reader()


async def test_effective_url_timeout_is_checked(settings_env):
    settings_env.setenv("REDIS_URL", "redis://localhost:6379/0?socket_timeout=1")
    state = _PoolState()
    try:
        with pytest.raises(StreamConfigurationError, match="socket_timeout"):
            state.start_stream_reader()
    finally:
        await state.close_stream_reader()


async def test_tls_reader_keeps_tls_settings(settings_env):
    settings_env.setenv("REDIS_URL", "rediss://user:pass@localhost:6379/0")
    state = _PoolState()
    state.start_stream_reader()
    try:
        from redis.asyncio import SSLConnection

        assert state.stream_pool.connection_class is SSLConnection
        assert state.stream_pool.max_connections == 256
    finally:
        await state.close_stream_reader()


async def test_sentinel_cleanup_attempts_every_close_after_failure():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    first = SimpleNamespace(aclose=AsyncMock(side_effect=RuntimeError("close failed")))
    second = SimpleNamespace(aclose=AsyncMock())
    pool = SimpleNamespace(aclose=AsyncMock())
    state = _PoolState()
    state.stream_pool = pool
    state.stream_sentinel = SimpleNamespace(sentinels=[first, second])
    with pytest.raises(RuntimeError, match="close failed"):
        await state.close_stream_reader()
    first.aclose.assert_awaited_once()
    second.aclose.assert_awaited_once()
    pool.aclose.assert_awaited_once()
    assert state.stream_pool is None and state.stream_sentinel is None
