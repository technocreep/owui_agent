"""Dependency container injected into every PydanticAI tool call."""

import os
from dataclasses import dataclass, field


@dataclass
class AgentDeps:
    owui_token: str    # bearer token forwarded from OWUI request
    # Self-URL used by sub_agent tool to call this service recursively.
    # Set via AGENT_SELF_URL env (e.g. http://pydantic-agent:8000).
    agent_self_url: str = field(
        default_factory=lambda: os.getenv("AGENT_SELF_URL", "http://pydantic-agent:8000")
    )
