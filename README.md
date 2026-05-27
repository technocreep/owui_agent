# OWUI PydanticAI Agent

External agent service that OWUI connects to as an OpenAI-compatible Connection.

## Architecture

```
User
 └─► OWUI (UI, auth, RAG pipeline, built-in tools)
       └─► POST /v1/chat/completions  ──► Agent Service (this repo)
                                               │
                                  ┌────────────┴────────────┐
                                  ▼                         ▼
                           PydanticAI Agent         back-calls OWUI API
                           (tool planning)          /api/v1/knowledge/…
                                                    /api/chat/completions
                                                    (Способ B — OWUI does RAG)
```

Key points:
- OWUI treats the agent as just another OpenAI model
- The agent can call OWUI back for KB queries (direct vector search)
- Or delegate entirely to OWUI's pipeline, which handles web search,
  image generation, and all other built-in tools automatically


## Quick start

```bash
# 1. Clone / copy this directory next to your OWUI compose file

# 2. Create .env
cp .env.example .env
# edit .env — add your OWUI API key

# 3. Make sure OWUI is on a named Docker network
docker network create owui-net   # skip if it already exists
# Add `networks: [owui-net]` to your OWUI service in its compose file

# 4. Build and start the agent
docker compose up -d --build

# 5. Register in OWUI
#    Admin → Settings → Connections → Add connection
#    URL:    http://pydantic-agent:8000
#    Key:    (any string — the agent echoes it back)
#    → Save → the model "pydantic-agent" appears in model list
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OWUI_BASE_URL` | `http://open-webui:3000` | OWUI internal address |
| `AGENT_LLM_BASE_URL` | `http://open-webui:3000/v1` | LLM backend for the agent |
| `AGENT_LLM_API_KEY` | `placeholder` | Key for LLM backend |
| `AGENT_LLM_MODEL` | `llama3.1:8b` | Model name as seen in OWUI |
| `OWUI_DELEGATE_MODEL` | same as above | Model used for Способ B delegation |
| `OWUI_SERVICE_TOKEN` | `` | Agent's own OWUI token for listing KBs |
| `AGENT_SYSTEM_PROMPT` | see agent.py | Override the agent system prompt |

## Available tools

| Tool | What it does |
|---|---|
| `list_available_knowledge_bases` | Lists all KBs the user can access |
| `search_knowledge_base` | Direct vector search against a specific KB |
| `delegate_to_owui` | Delegates to OWUI pipeline (RAG + built-in tools) |

## Adding custom tools

Edit `app/tools/owui_tools.py` and add a new `@agent.tool` decorated
async function inside `register_tools()`. It receives `RunContext[AgentDeps]`
giving access to `ctx.deps.owui_token` for authenticated OWUI calls.

```python
@agent.tool
async def my_tool(ctx: RunContext[AgentDeps], param: str) -> str:
    """Docstring is the tool description shown to the LLM."""
    ...
```
