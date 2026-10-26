"""Explicit, environment-free configuration for resumable streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redis_fastapi.exceptions import StreamConfigurationError

if TYPE_CHECKING:
    from redis_fastapi.config import RedisSettings


class StreamSettings(BaseModel):
    """Stream policy only; the application configures Redis clients separately.

    No environment or .env files are read. Instances are immutable so a writer's
    keys and policy cannot change while a stream is active.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    prefix: str = "redis:fastapi"
    stream_active_ttl: int = Field(default=600, gt=0)
    stream_done_ttl: int = Field(default=120, gt=0)
    stream_gone_ttl: int = Field(default=3600, ge=0)
    stream_maxlen: int = Field(default=10_000, gt=0)
    stream_max_event_bytes: int = Field(default=65_536, gt=0)
    stream_heartbeat_interval: int = Field(default=15, gt=0)
    stream_alive_ttl: int = Field(default=45, gt=0)
    stream_replay_page: int = Field(default=100, gt=0)
    stream_max_readers: int = Field(default=256, gt=0)
    stream_max_connection_duration: int = Field(default=300, gt=0)
    stream_read_retry_window: int = Field(default=30, ge=0)
    stream_send_timeout: int = Field(default=10, gt=0)

    @model_validator(mode="after")
    def _check_policy(self) -> StreamSettings:
        if "{" in self.prefix or "}" in self.prefix:
            raise StreamConfigurationError("The key prefix must not contain braces")
        if self.stream_alive_ttl <= self.stream_heartbeat_interval:
            raise StreamConfigurationError(
                "stream_alive_ttl must be longer than stream_heartbeat_interval"
            )
        return self

    @classmethod
    def from_redis_settings(cls, settings: RedisSettings) -> StreamSettings:
        """Copy resolved streaming fields, excluding connections and telemetry."""
        return cls.model_validate(
            {name: getattr(settings, name) for name in cls.model_fields}
        )
