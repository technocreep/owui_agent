"""
OWUI Agent Service
Exposes /v1/chat/completions (OpenAI-compatible) so Open WebUI
can connect to it as an external Connection.

Flow:
  OWUI user → OWUI (RAG/tools injected into messages) →
  POST /v1/chat/completions here →
  PydanticAI agent → back-calls OWUI API for KB/tools →
  streams response back to OWUI
"""

import asyncio
import json
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent import run_agent
from app.models.openai import ChatCompletionRequest, ChatCompletionResponse

app = FastAPI(title="OWUI PydanticAI Agent", version="1.0.0")


@app.get("/v1/models")
async def list_models():
    """OWUI checks this to discover available 'models'."""
    return {
        "object": "list",
        "data": [
            {
                "id": "pydantic-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str = Header(default=""),
):
    # Forward the OWUI API key so agent can call back into OWUI
    owui_token = authorization.replace("Bearer ", "").strip()

    if request.stream:
        return StreamingResponse(
            _stream_response(request, owui_token),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await run_agent(request.messages, owui_token)
    return _build_response(result, request.model)


async def _stream_response(
    request: ChatCompletionRequest, owui_token: str
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    result = await run_agent(request.messages, owui_token)

    # Stream token-by-token (word chunks for simplicity;
    # swap for real token streaming if your LLM backend supports it)
    words = result.split(" ")
    for i, word in enumerate(words):
        chunk_text = word if i == len(words) - 1 else word + " "
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": chunk_text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)  # yield to event loop

    # Final chunk
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def _build_response(content: str, model: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
