"""HTTP API. Every future interface (voice daemon, tray widget, phone) is a client of this."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from jarvis import __version__
from jarvis.app import JarvisApp
from jarvis.conversation import ConversationNotFound
from jarvis.core.config import Settings, load_settings
from jarvis.core.types import Message, StreamChunk, Usage
from jarvis.providers import ProviderError
from jarvis.storage import Conversation, StoredMessage


class ChatRequest(BaseModel):
    messages: list[Message]
    role: str = "chat"
    stream: bool = True


class ChatResponse(BaseModel):
    message: Message
    usage: Usage | None = None


class NewConversation(BaseModel):
    title: str | None = None


class SendMessage(BaseModel):
    content: str
    stream: bool = True


class ConversationDetail(BaseModel):
    conversation: Conversation
    messages: list[StoredMessage]


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _respond(make_stream: Callable[[], AsyncIterator[StreamChunk]], stream: bool):
    """Turn a chunk stream into either an SSE response or a single JSON reply."""
    if not stream:
        parts: list[str] = []
        usage = None
        try:
            async with aclosing(make_stream()) as chunks:
                async for chunk in chunks:
                    parts.append(chunk.text)
                    usage = chunk.usage or usage
        except ProviderError as e:
            raise HTTPException(502, str(e)) from e
        return ChatResponse(message=Message.assistant("".join(parts)), usage=usage)

    async def events() -> AsyncIterator[str]:
        try:
            async with aclosing(make_stream()) as chunks:
                async for chunk in chunks:
                    if chunk.thinking:
                        yield _sse({"type": "thinking", "text": chunk.thinking})
                    if chunk.text:
                        yield _sse({"type": "token", "text": chunk.text})
                    if ev := chunk.event:
                        yield _sse(
                            {
                                "type": "tool_call" if ev.kind == "call" else "tool_result",
                                "status": ev.kind,
                                "id": ev.call.id,
                                "name": ev.call.name,
                                "arguments": ev.call.arguments,
                                "output": ev.output,
                            }
                        )
                    if chunk.done:
                        usage = chunk.usage.model_dump() if chunk.usage else None
                        yield _sse({"type": "done", "usage": usage})
        except ProviderError as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(events(), media_type="text/event-stream")


def create_api(settings: Settings | None = None, jarvis: JarvisApp | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(api: FastAPI):
        owned = jarvis is None
        api.state.jarvis = jarvis or JarvisApp.create(settings or load_settings())
        if owned:
            api.state.jarvis.scheduler.start()  # reminders fire while the server runs
            api.state.jarvis.apps.prefetch()
        yield
        if owned:
            await api.state.jarvis.aclose()

    api = FastAPI(title="Jarvis", version=__version__, lifespan=lifespan)

    def get_jarvis(request: Request) -> JarvisApp:
        return request.app.state.jarvis

    def check_role(j: JarvisApp, role: str) -> None:
        try:
            j.models.resolve(role)
        except ProviderError as e:
            raise HTTPException(400, str(e)) from e

    def get_conversation(j: JarvisApp, conversation_id: int) -> Conversation:
        try:
            return j.conversations.get(conversation_id)
        except ConversationNotFound as e:
            raise HTTPException(404, str(e)) from e

    # ------------------------------------------------------------ system

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

    # ------------------------------------------------------------ stateless chat

    @api.post("/v1/chat")
    async def chat(req: ChatRequest, request: Request):
        j = get_jarvis(request)
        check_role(j, req.role)
        return await _respond(lambda: j.chat_stream(req.messages, req.role), req.stream)

    # ------------------------------------------------------------ conversations

    @api.get("/v1/conversations", response_model=list[Conversation])
    async def list_conversations(request: Request, limit: int = 20):
        return get_jarvis(request).store.list(limit)

    @api.post("/v1/conversations", response_model=Conversation, status_code=201)
    async def create_conversation(body: NewConversation, request: Request):
        return get_jarvis(request).conversations.new(body.title)

    @api.get("/v1/conversations/{conversation_id}", response_model=ConversationDetail)
    async def get_conversation_detail(conversation_id: int, request: Request):
        j = get_jarvis(request)
        conv = get_conversation(j, conversation_id)
        return ConversationDetail(conversation=conv, messages=j.store.messages(conversation_id))

    @api.delete("/v1/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: int, request: Request):
        if not get_jarvis(request).store.delete(conversation_id):
            raise HTTPException(404, f"No conversation with id {conversation_id}")

    @api.get("/v1/tools")
    async def list_tools(request: Request):
        j = get_jarvis(request)
        return [
            {
                "name": t.name,
                "description": t.description,
                "risk": str(t.risk),
                "permission": str(j.policy.overrides.get(t.name) or j.policy.risk_defaults[t.risk]),
            }
            for t in j.registry.tools
        ]

    @api.get("/v1/audit")
    async def audit(request: Request, limit: int = 50):
        return get_jarvis(request).audit.recent(limit)

    @api.post("/v1/conversations/{conversation_id}/messages")
    async def send_message(conversation_id: int, body: SendMessage, request: Request):
        # No approval UI over HTTP yet: tools that would ask are denied.
        j = get_jarvis(request)
        get_conversation(j, conversation_id)
        check_role(j, j.conversations.role)
        return await _respond(
            lambda: j.conversations.send(conversation_id, body.content), body.stream
        )

    return api
