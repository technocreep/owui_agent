"""OpenAI-compatible request / response schemas."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Any] | None = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "pydantic-agent"
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # OWUI may pass files/knowledge context already injected into messages,
    # or as extra fields — we accept and ignore unknown fields gracefully
    model_config = {"extra": "allow"}


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]
