"""HTTP API. Every future interface (voice daemon, tray widget, phone) is a client of this."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from jarvis import __version__
from jarvis.app import JarvisApp
from jarvis.core.config import Settings, load_settings
from jarvis.core.types import Message, Usage
from jarvis.providers import ProviderError


class ChatRequest(BaseModel):
    messages: list[Message]
    role: str = "chat"
    stream: bool = True


class ChatResponse(BaseModel):
    message: Message
    usage: Usage | None = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def create_api(settings: Settings | None = None, jarvis: JarvisApp | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(api: FastAPI):
        owned = jarvis is None
        api.state.jarvis = jarvis or JarvisApp.create(settings or load_settings())
        yield
        if owned:
            await api.state.jarvis.aclose()

    api = FastAPI(title="Jarvis", version=__version__, lifespan=lifespan)

    def get_jarvis(request: Request) -> JarvisApp:
        return request.app.state.jarvis

    @api.get("/health")
    async def health(request: Request):
        j = get_jarvis(request)
        providers = await j.models.health()
        return {
            "ok": all(p.ok for p in providers.values()),
            "version": __version__,
            "roles": {r: str(ref) for r, ref in j.models.roles.items()},
            "providers": {k: v.model_dump() for k, v in providers.items()},
        }

    @api.get("/v1/models")
    async def models(request: Request):
        j = get_jarvis(request)
        out = {}
        for name, p in j.models.providers.items():
            try:
                out[name] = {
                    "available": await p.list_models(),
                    "loaded": [m.model_dump() for m in await p.loaded_models()],
                }
            except ProviderError as e:
                out[name] = {"error": str(e)}
        return out

    @api.post("/v1/chat")
    async def chat(req: ChatRequest, request: Request):
        j = get_jarvis(request)
        try:
            j.models.resolve(req.role)
        except ProviderError as e:
            raise HTTPException(400, str(e)) from e

        if not req.stream:
            parts: list[str] = []
            usage = None
            try:
                async for chunk in j.chat_stream(req.messages, req.role):
                    parts.append(chunk.text)
                    usage = chunk.usage or usage
            except ProviderError as e:
                raise HTTPException(502, str(e)) from e
            return ChatResponse(message=Message.assistant("".join(parts)), usage=usage)

        async def events() -> AsyncIterator[str]:
            try:
                async for chunk in j.chat_stream(req.messages, req.role):
                    if chunk.thinking:
                        yield _sse({"type": "thinking", "text": chunk.thinking})
                    if chunk.text:
                        yield _sse({"type": "token", "text": chunk.text})
                    if chunk.done:
                        usage = chunk.usage.model_dump() if chunk.usage else None
                        yield _sse({"type": "done", "usage": usage})
            except ProviderError as e:
                yield _sse({"type": "error", "message": str(e)})

        return StreamingResponse(events(), media_type="text/event-stream")

    return api
