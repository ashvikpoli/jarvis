"""ModelManager: maps roles ("chat", later "planner", "vision"...) to provider+model."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jarvis.core.types import Message, ProviderHealth, StreamChunk
from jarvis.providers.base import ModelProvider, ProviderError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @classmethod
    def parse(cls, ref: str) -> ModelRef:
        """'ollama:qwen3:14b' -> ModelRef('ollama', 'qwen3:14b'). Splits on the first colon."""
        provider, sep, model = ref.partition(":")
        if not sep or not provider or not model:
            raise ValueError(f"Model ref must look like '<provider>:<model>', got {ref!r}")
        return cls(provider, model)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


class ModelManager:
    def __init__(self, providers: Mapping[str, ModelProvider], roles: Mapping[str, str]):
        self.providers = dict(providers)
        self.roles: dict[str, ModelRef] = {r: ModelRef.parse(ref) for r, ref in roles.items()}

    def set_role(self, role: str, ref: str) -> None:
        self.roles[role] = ModelRef.parse(ref)

    def resolve(self, role: str) -> tuple[ModelProvider, str]:
        if role not in self.roles:
            raise ProviderError(f"No model configured for role {role!r}")
        ref = self.roles[role]
        if ref.provider not in self.providers:
            raise ProviderError(
                f"Role {role!r} uses provider {ref.provider!r}, which isn't enabled. "
                f"Available: {', '.join(self.providers)}"
            )
        return self.providers[ref.provider], ref.model

    async def stream_chat(
        self,
        messages: Sequence[Message],
        role: str = "chat",
        *,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        provider, model = self.resolve(role)
        async for chunk in provider.stream_chat(messages, model, options=options):
            if chunk.done and chunk.usage:
                u = chunk.usage
                log.info(
                    "generation complete",
                    extra={
                        "role": role,
                        "model": f"{provider.name}:{model}",
                        "ttft_ms": round(u.ttft_ms or 0, 1),
                        "total_ms": round(u.total_ms or 0, 1),
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "tokens_per_s": round(u.tokens_per_s or 0, 1),
                    },
                )
            yield chunk

    async def warmup(self, role: str = "chat") -> None:
        provider, model = self.resolve(role)
        await provider.load(model)

    async def unload(self, role: str = "chat") -> None:
        provider, model = self.resolve(role)
        await provider.unload(model)

    async def health(self) -> dict[str, ProviderHealth]:
        return {name: await p.health() for name, p in self.providers.items()}

    async def aclose(self) -> None:
        for p in self.providers.values():
            await p.aclose()
