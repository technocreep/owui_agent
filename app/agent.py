"""
PydanticAI agent.

LLM backend: vLLM (напрямую, минуя OWUI).
  - vLLM поднят на хосте/сервере, адрес задаётся через VLLM_BASE_URL.
  - vLLM совместим с OpenAI API, поэтому используем OpenAIModel.

OWUI используется только для:
  - KB / retrieval API (owui_client.py)
  - Делегирования запросов с built-in tools (delegate_to_owui)
"""

from __future__ import annotations

import os

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.models.deps import AgentDeps
from app.models.openai import Message
from app.tools.owui_tools import register_tools

# ---------------------------------------------------------------------------
# vLLM connection — единственный LLM-бэкенд агента
#
# VLLM_BASE_URL  — адрес vLLM сервера, например:
#                  http://192.168.1.10:8000/v1   (хост-машина)
#                  http://vllm:8000/v1            (другой контейнер)
#                  http://host.docker.internal:8000/v1  (Docker Desktop)
#
# VLLM_API_KEY   — если vLLM запущен без --api-key, ставим "EMPTY"
#                  (OpenAI-клиент требует непустую строку)
#
# VLLM_MODEL     — имя модели точно как в vLLM (из --model при запуске),
#                  например "Qwen/Qwen2.5-72B-Instruct"
# ---------------------------------------------------------------------------

VLLM_BASE_URL: str = os.environ["VLLM_BASE_URL"]          # обязательная
VLLM_API_KEY:  str = os.getenv("VLLM_API_KEY", "EMPTY")   # "EMPTY" если без auth
VLLM_MODEL:    str = os.environ["VLLM_MODEL"]              # обязательная

# ---------------------------------------------------------------------------
# GIS MCP Server (arcgis_mcp)
#
# Сервер использует FastMCP 3.x, transport=http → Streamable HTTP на /mcp.
# URL формат: http://<host>:<port>/mcp
#
# Примеры GIS_MCP_URL:
#   http://arcgis-mcp:8000/mcp      ← контейнер в той же Docker-сети
#   http://172.17.0.1:8000/mcp      ← сервис на хосте (Linux Docker)
#   http://host.docker.internal:8000/mcp  ← Docker Desktop
#
# GIS_MCP_TOKEN — JWT-токен если сервер запущен с аутентификацией
#                 (arcgis_mcp поддерживает JWT через GIS_USERNAME/GIS_PASSWORD).
#                 Оставить пустым если auth не нужна.
#
# Важно: arcgis_mcp хранит текущий проект в _state (in-memory).
# При Streamable HTTP состояние живёт в рамках одной MCP-сессии.
# Агент должен вызывать list_projects → get_project_summary в каждом
# новом разговоре, чтобы установить контекст проекта.
# ---------------------------------------------------------------------------
GIS_MCP_URL:   str | None = os.getenv("GIS_MCP_URL")
GIS_MCP_TOKEN: str | None = os.getenv("GIS_MCP_TOKEN")

# ---------------------------------------------------------------------------
# System prompt агента
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    """You are a geology analyst assistant. Respond with technical precision and always cite sources.
Date: {current_datetime}
Language: always match the user's language.

You make all decisions autonomously — never ask for confirmation. You always complete tasks to their logical conclusion.
YOU STRICTLY FOLLOW INSTRUCTIONS AND RESPECT ALL RULES AND CONSTRAINTS.
If a GIS tool returns a markdown map link, use it exactly as-is in your response — never modify the URL.

## Tool Usage Order — DO NOT SKIP STEPS OR USE TOOLS OUT OF ORDER
0. Check conversation history — if the data was already retrieved in this session, use it directly
1. Answer from internal knowledge first
2. Check knowledge base → general info, deposits, drilling, reserves, feasibility studies, stratigraphy
   - Use list_available_knowledge_bases first, then search_knowledge_base or search_multiple_knowledge_bases
   - Max 5 sources per query. Expand search gradually from broad to specific.
   - For definitions/verification of geological terms: ALWAYS search KB "Справочная литература" first
3. ONLY if GIS data is explicitly requested → use gis__ tools:
   - Always start: gis__list_projects → gis__get_project_summary (establishes project context)
   - P0 (fast, manifest): gis__list_layers, gis__describe_layer
   - P1 (reads .gdb): gis__query_features, gis__summarize_layer, gis__search_izuchennost, gis__list_attachments
   - Visualisation: after gis__plot_layer / gis__plot_overlay / gis__plot_histogram
     insert the `markdown` field verbatim — it renders the image in chat.
     After gis__plot_interactive insert the `link` field.
   - Layer styling:
     • Geophysical (units in brackets: мГал, нТл, Э): color_field=<first numeric field from manifest>
     • Geological / stratigraphy: color_field='INDEX'
     • Boreholes: color_field='POINT_Z'
     • Lineaments: color='#00FF00' in plot_overlay
     • Tectonics: different color/linewidth for thrusts vs faults
     • Always show_license=True
   - Data Cube / ML prospectivity: gis__datacube_overview → gis__datacube_block_scores → gis__datacube_block_detail
   - Knowledge Graph (geological context, stratigraphy, work history): gis__geo_context_query
   - Layers with needs_review=true: warn the user the name may be inaccurate
   - Use display_name of layers, never technical layer_id
   - NEVER use GIS tools unless user explicitly asks for GIS data
4. web_search → only if knowledge base returned nothing useful
5. sub_agent → only when the task requires more than 5 tool calls; delegate the full subtask with all context

## Knowledge Verification
If the user asks for a definition or verification of a geological term or concept, always search
the knowledge base "Справочная литература" first — before any other source.
Never answer from internal knowledge alone for such requests.
IF "Справочная литература" is unavailable — tell the user and ask them to fix it.

## Operating Mode
- Work in ReAct mode with intermediate status reports: planned → acted → analyzed → reported
- Always start by planning: break complex tasks into subtasks
- Search knowledge base from general to specific: broad search first, then targeted
- Act autonomously — do not ask for permission
- Quantitative data → Markdown tables
- Every fact must have a source citation
- If data is missing → "No data available for [X]"
- After each tool call: "✓ [result] → [next step]"

## Prohibitions
- NEVER mix data from different objects — use source metadata as the reference point
- NEVER cite without a source
- NEVER give investment advice without a calculation source
- NEVER do what was not asked — do not over-complicate
- NEVER violate tool order or use unnecessary tools
- NEVER use GIS data unless explicitly requested
""",
)


def _build_mcp_toolsets() -> list:
    """
    Собирает список MCP-toolset-ов из env-переменных.

    Каждый MCP-сервер — AbstractToolset: PydanticAI автоматически
    подгружает все инструменты сервера без написания кода под каждый.

    arcgis_mcp использует FastMCP 3.x с transport=http →
    Streamable HTTP (MCP spec 2025-03-26) на пути /mcp.
    Поэтому всегда используем MCPServerHTTP.
    """
    toolsets = []

    if GIS_MCP_URL:
        headers: dict[str, str] = {}
        if GIS_MCP_TOKEN:
            headers["Authorization"] = f"Bearer {GIS_MCP_TOKEN}"

        # arcgis_mcp: FastMCP transport=http → Streamable HTTP на /mcp
        # tool_prefix="gis" → все инструменты получают имя gis__<name>,
        # что исключает коллизии с другими toolset-ами.
        server = MCPServerHTTP(
            url=GIS_MCP_URL,
            headers=headers or None,
            tool_prefix="gis",
        )
        toolsets.append(server)
        print(f"[agent] GIS MCP (arcgis_mcp) registered: {GIS_MCP_URL}")

    # Паттерн для добавления других MCP-серверов:
    # if os.getenv("ANOTHER_MCP_URL"):
    #     toolsets.append(MCPServerHTTP(
    #         url=os.getenv("ANOTHER_MCP_URL"),
    #         tool_prefix="svc",
    #     ))

    return toolsets


def _build_agent() -> Agent[AgentDeps, str]:
    """Собирает агента с vLLM как LLM-бэкендом и MCP-toolset-ами."""
    model = OpenAIModel(
        VLLM_MODEL,
        provider=OpenAIProvider(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
        ),
    )
    mcp_toolsets = _build_mcp_toolsets()

    agent: Agent[AgentDeps, str] = Agent(
        model=model,
        deps_type=AgentDeps,
        output_type=str,
        instructions=SYSTEM_PROMPT,
        toolsets=mcp_toolsets or None,
    )
    register_tools(agent)
    return agent


# Singleton — строится один раз при старте контейнера
_agent = _build_agent()


def _messages_to_prompt(messages: list[Message]) -> str:
    """
    Преобразует список сообщений OWUI в единый prompt-строку.
    OWUI может уже вставить RAG-контекст как system-сообщения — сохраняем всё.
    """
    parts = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            # Мультимодальный контент от OWUI — берём только текстовые части
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if content:
            parts.append(f"{msg.role.upper()}: {content}")
    return "\n\n".join(parts)


async def run_agent(messages: list[Message], owui_token: str) -> str:
    """
    Точка входа из FastAPI-хендлера.
    Инжектирует зависимости и запускает агент против vLLM.
    """
    deps = AgentDeps(
        owui_token=owui_token,
    )
    prompt = _messages_to_prompt(messages)
    result = await _agent.run(prompt, deps=deps)
    # pydantic-ai ≥1.0: финальный ответ в result.response
    return result.response
