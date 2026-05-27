"""
OWUI Agent Service
Exposes /v1/chat/completions (OpenAI-compatible) so Open WebUI
can connect to it as an external Connection.

Flow:
  OWUI user → OWUI (RAG/tools injected into messages) →
  POST /v1/chat/completions here →
  PydanticAI agent → vLLM (reasoning) + OWUI API (KB/tools) + MCP (GIS)
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent import run_agent, VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL, _build_mcp_toolsets
from app.models.openai import ChatCompletionRequest, ChatCompletionResponse

log = logging.getLogger("agent.startup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Startup probe — выполняется один раз при старте контейнера
# ─────────────────────────────────────────────────────────────────────────────

async def _probe_llm() -> bool:
    """
    Реальный запрос к vLLM через агента: max_tokens=1, температура 0.
    Цель — убедиться что модель загружена и отвечает до приёма трафика.
    """
    log.info("━━━  LLM PROBE  ━━━")
    log.info("  endpoint : %s", VLLM_BASE_URL)
    log.info("  model    : %s", VLLM_MODEL)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            r = await client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {VLLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": VLLM_MODEL,
                    "messages": [{"role": "user", "content": "Reply with one word: ready"}],
                    "max_tokens": 4,
                    "temperature": 0,
                },
            )
        if r.status_code == 200:
            reply = r.json()["choices"][0]["message"]["content"].strip()
            log.info("  ✓ LLM reachable — response: %r", reply)
            return True
        else:
            log.error("  ✗ LLM returned HTTP %s: %s", r.status_code, r.text[:200])
            return False
    except Exception as e:
        log.error("  ✗ LLM unreachable: %s", e)
        return False


async def _probe_mcp() -> None:
    """
    Подключается к каждому MCP-серверу и выводит список его инструментов.
    """
    toolsets = _build_mcp_toolsets()
    if not toolsets:
        log.info("━━━  MCP  ━━━  no servers configured")
        return

    for server in toolsets:
        prefix = getattr(server, "_tool_prefix", None) or getattr(server, "tool_prefix", "?")
        url    = getattr(server, "url", "?")
        log.info("━━━  MCP SERVER  ━━━")
        log.info("  prefix : %s", prefix)
        log.info("  url    : %s", url)
        try:
            tools = await server.list_tools()
            if tools:
                log.info("  ✓ %d tool(s):", len(tools))
                for t in tools:
                    desc = (t.description or "").split("\n")[0][:80]
                    log.info("    • %s__%s  —  %s", prefix, t.name, desc)
            else:
                log.warning("  ⚠ server reachable but returned 0 tools")
        except Exception as e:
            log.error("  ✗ MCP unreachable: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────────────────────
    log.info("═══  AGENT STARTUP  ═══")
    llm_ok = await _probe_llm()
    await _probe_mcp()
    if llm_ok:
        log.info("═══  STARTUP COMPLETE — ready to serve  ═══")
    else:
        log.warning("═══  STARTUP COMPLETE — LLM unreachable, requests will fail  ═══")
    yield
    # ── shutdown (ничего особого не нужно) ───────────────────────────────────
    log.info("Agent shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="OWUI PydanticAI Agent", version="1.0.0", lifespan=lifespan)


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
        await asyncio.sleep(0)

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
    """Lightweight liveness check для Docker HEALTHCHECK."""
    return {"status": "ok"}
