"""Ollama provider. Talks to the native Windows Ollama server over its HTTP API."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from jarvis.core.config import OllamaSettings
from jarvis.core.types import (
    LoadedModel,
    Message,
    ProviderHealth,
    StreamChunk,
    ToolCall,
    Usage,
)
from jarvis.providers.base import ModelProvider, ProviderError

log = logging.getLogger(__name__)


def _to_ollama(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
        ]
    if m.role == "tool" and m.tool_name:
        out["tool_name"] = m.tool_name
    return out


def _parse_call(raw: dict[str, Any]) -> ToolCall:
    fn = raw.get("function") or {}
    args = fn.get("arguments") or {}
    if isinstance(args, str):  # some models return a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    kw: dict[str, Any] = {"name": fn.get("name", ""), "arguments": args}
    if raw.get("id"):
        kw["id"] = str(raw["id"])
    return ToolCall(**kw)


class OllamaProvider(ModelProvider):
    name = "ollama"

    def __init__(self, settings: OllamaSettings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.timeout_s, connect=5.0),
            transport=transport,
        )

    async def stream_chat(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        opts = dict(options or {})
        # `think` and `keep_alive` are top-level request fields in Ollama; everything
        # else (temperature, num_ctx, ...) goes into `options`.
        think = opts.pop("think", self.settings.think)
        fmt = opts.pop("format", None)  # "json" or a JSON schema → structured output
        body: dict[str, Any] = {
            "model": model,
            "messages": [_to_ollama(m) for m in messages],
            "stream": True,
            "keep_alive": opts.pop("keep_alive", self.settings.keep_alive),
        }
        if opts:
            body["options"] = opts
        if think is not None:
            body["think"] = think
        if tools:
            body["tools"] = tools
        if fmt is not None:
            body["format"] = fmt

        start = time.perf_counter()
        ttft: float | None = None
        answer_ttft: float | None = None
        try:
            # `async with` guarantees the connection closes if our task is cancelled,
            # which makes Ollama stop generating. This is what barge-in relies on.
            async with self._client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode(errors="replace")
                    raise ProviderError(f"Ollama returned {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if err := data.get("error"):
                        raise ProviderError(f"Ollama error: {err}")
                    msg = data.get("message") or {}
                    text, thinking = msg.get("content") or "", msg.get("thinking") or ""
                    now_ms = (time.perf_counter() - start) * 1000
                    if (text or thinking) and ttft is None:
                        ttft = now_ms
                    if text and answer_ttft is None:
                        answer_ttft = now_ms
                    if text or thinking:
                        yield StreamChunk(text=text, thinking=thinking)
                    if raw_calls := msg.get("tool_calls"):
                        yield StreamChunk(tool_calls=[_parse_call(c) for c in raw_calls])
                    if data.get("done"):
                        yield StreamChunk(
                            done=True,
                            usage=Usage(
                                prompt_tokens=data.get("prompt_eval_count"),
                                completion_tokens=data.get("eval_count"),
                                ttft_ms=ttft,
                                answer_ttft_ms=answer_ttft,
                                total_ms=(time.perf_counter() - start) * 1000,
                            ),
                        )
                        return
        except httpx.ConnectError as e:
            raise ProviderError(
                f"Can't reach Ollama at {self.settings.base_url}. Is Ollama running?"
            ) from e

    async def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        data = await self._post_json(
            "/api/embed",
            {"model": model, "input": list(texts), "keep_alive": self.settings.keep_alive},
        )
        vectors = data.get("embeddings") or []
        if len(vectors) != len(texts):
            raise ProviderError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts")
        return vectors

    async def list_models(self) -> list[str]:
        data = await self._get_json("/api/tags")
        return [m["name"] for m in data.get("models", [])]

    async def loaded_models(self) -> list[LoadedModel]:
        data = await self._get_json("/api/ps")
        return [
            LoadedModel(name=m["name"], size_bytes=m.get("size"), vram_bytes=m.get("size_vram"))
            for m in data.get("models", [])
        ]

    async def load(self, model: str, *, options: dict[str, Any] | None = None) -> None:
        # A generate call with no prompt loads the model and returns immediately.
        # Pass the same load-affecting options (num_ctx...) chat will use, or Ollama
        # would reload the model on the first real request.
        opts = {k: v for k, v in (options or {}).items() if k not in ("think", "keep_alive")}
        body: dict[str, Any] = {"model": model, "keep_alive": self.settings.keep_alive}
        if opts:
            body["options"] = opts
        await self._post_json("/api/generate", body)

    async def unload(self, model: str) -> None:
        await self._post_json("/api/generate", {"model": model, "keep_alive": 0})

    async def health(self) -> ProviderHealth:
        try:
            data = await self._get_json("/api/version")
            return ProviderHealth(
                ok=True, version=data.get("version"), detail=self.settings.base_url
            )
        except ProviderError as e:
            return ProviderHealth(ok=False, detail=str(e))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get_json(self, path: str) -> dict[str, Any]:
        return await self._request("GET", path)

    async def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", path, json=body)

    async def _request(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        try:
            resp = await self._client.request(method, path, **kw)
        except httpx.ConnectError as e:
            raise ProviderError(
                f"Can't reach Ollama at {self.settings.base_url}. Is Ollama running?"
            ) from e
        if resp.status_code != 200:
            raise ProviderError(f"Ollama {path} returned {resp.status_code}: {resp.text}")
        return resp.json()
