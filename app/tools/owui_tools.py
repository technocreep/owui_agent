"""
PydanticAI tool definitions — покрытие OWUI KB и веб-поиска.

Загрузка документов — зона OWUI UI + OCR-сервиса, не агента.
Пользователь загружает файл в OWUI → OCR извлекает текст →
OWUI индексирует в вектор-БД. Агент только читает результат.

Группы инструментов:
  1. KB Discovery      — list, find, get files
  2. KB Search         — query/collection, query/doc, multi-KB
  3. KB Management     — только reset (переиндексация)
  4. Content Ingestion — index_web_page, index_text (динамический контент)
  5. Web Search        — через OWUI Perplexity pipeline
"""

from __future__ import annotations
import httpx
from pydantic_ai import RunContext
from app.models.deps import AgentDeps
from app.owui_client import (
    # KB — discovery
    list_knowledge_bases,
    search_knowledge_bases,
    get_knowledge_meta,
    get_knowledge_files,
    # KB — management (только reset, нужен при переиндексации)
    reset_knowledge,
    # Retrieval — поиск
    query_knowledge_base,
    query_single_doc,
    # Retrieval — ingestion (URL и текст, не файлы)
    process_web_url,
    process_text,
    # Web search
    owui_web_search,
)


def register_tools(agent):

    # ── 1. KB Discovery ────────────────────────────────────────────────────

    @agent.tool
    async def list_available_knowledge_bases(ctx: RunContext[AgentDeps]) -> str:
        """
        List all knowledge bases the current user has access to in OWUI.
        Returns names, IDs, and descriptions.
        Call this first to discover what knowledge collections exist.
        """
        kbs = await list_knowledge_bases(ctx.deps.owui_token)
        if not kbs:
            return "No knowledge bases found."
        lines = [
            f"- [{kb['id']}] {kb['name']}"
            + (f": {kb['description']}" if kb.get("description") else "")
            for kb in kbs
        ]
        return "Available knowledge bases:\n" + "\n".join(lines)

    @agent.tool
    async def find_knowledge_base(
        ctx: RunContext[AgentDeps],
        query: str,
    ) -> str:
        """
        Search knowledge base names and descriptions by keyword.
        Useful when you know a topic but not which KB contains it.

        Args:
            query: Keyword or phrase to search for in KB metadata.
        """
        kbs = await search_knowledge_bases(ctx.deps.owui_token, query)
        if not kbs:
            return f"No knowledge bases matching '{query}'."
        lines = [f"- [{kb['id']}] {kb['name']}" for kb in kbs]
        return f"Knowledge bases matching '{query}':\n" + "\n".join(lines)

    @agent.tool
    async def list_files_in_knowledge_base(
        ctx: RunContext[AgentDeps],
        knowledge_id: str,
        filter_query: str | None = None,
    ) -> str:
        """
        List files stored inside a specific knowledge base.
        Useful to understand what documents are indexed before searching.

        Args:
            knowledge_id: KB id from list_available_knowledge_bases.
            filter_query: Optional filename keyword filter.
        """
        files = await get_knowledge_files(
            ctx.deps.owui_token,
            knowledge_id,
            query=filter_query,
        )
        if not files:
            return f"No files found in knowledge base '{knowledge_id}'."
        lines = [
            f"- [{f.get('id', '?')}] {f.get('meta', {}).get('name', f.get('filename', '?'))}"
            for f in files
        ]
        return f"Files in KB '{knowledge_id}':\n" + "\n".join(lines)

    # ── 2. KB Search ───────────────────────────────────────────────────────

    @agent.tool
    async def search_knowledge_base(
        ctx: RunContext[AgentDeps],
        knowledge_id: str,
        query: str,
        num_results: int = 5,
        hybrid: bool = True,
    ) -> str:
        """
        Search a specific knowledge base for relevant document chunks.
        Uses semantic vector search; hybrid=True also uses BM25 keyword
        matching for better recall on exact terms.

        Args:
            knowledge_id: KB id from list_available_knowledge_bases.
            query: Natural language search query.
            num_results: Number of chunks to return (default 5).
            hybrid: Use hybrid BM25+vector search (default True).
        """
        meta = await get_knowledge_meta(ctx.deps.owui_token, knowledge_id)
        collection_name = meta.get("collection_name") or meta.get("id") or knowledge_id
        docs = await query_knowledge_base(
            ctx.deps.owui_token, [collection_name], query, k=num_results, hybrid=hybrid
        )
        if not docs:
            return f"No results in KB '{knowledge_id}' for: {query}"
        parts = [
            f"[{i}] (source: {d['meta'].get('source') or d['meta'].get('name','?')})\n{d['content']}"
            for i, d in enumerate(docs, 1)
        ]
        return "\n\n---\n\n".join(parts)

    @agent.tool
    async def search_multiple_knowledge_bases(
        ctx: RunContext[AgentDeps],
        knowledge_ids: list[str],
        query: str,
        num_results: int = 5,
    ) -> str:
        """
        Search across multiple knowledge bases simultaneously.
        Use when the answer may span several collections.

        Args:
            knowledge_ids: List of KB ids.
            query: Natural language search query.
            num_results: Chunks per KB (default 5).
        """
        collection_names = []
        for kid in knowledge_ids:
            meta = await get_knowledge_meta(ctx.deps.owui_token, kid)
            collection_names.append(meta.get("collection_name") or meta.get("id") or kid)

        docs = await query_knowledge_base(
            ctx.deps.owui_token, collection_names, query, k=num_results
        )
        if not docs:
            return f"No results across {len(knowledge_ids)} KBs for: {query}"
        parts = [
            f"[{i}] (source: {d['meta'].get('source') or d['meta'].get('name','?')})\n{d['content']}"
            for i, d in enumerate(docs, 1)
        ]
        return "\n\n---\n\n".join(parts)

    @agent.tool
    async def search_single_document(
        ctx: RunContext[AgentDeps],
        collection_name: str,
        query: str,
        num_results: int = 5,
    ) -> str:
        """
        Search within a single file-level collection (not a KB).
        Use when you have a specific file's collection_name
        (e.g. from process_web_url or process_text results).

        Args:
            collection_name: Internal collection name (e.g. 'file-<uuid>').
            query: Natural language search query.
            num_results: Number of chunks to return.
        """
        docs = await query_single_doc(
            ctx.deps.owui_token, collection_name, query, k=num_results
        )
        if not docs:
            return f"No results in collection '{collection_name}' for: {query}"
        parts = [
            f"[{i}] {d['content']}" for i, d in enumerate(docs, 1)
        ]
        return "\n\n---\n\n".join(parts)

    # ── 3. KB Management ───────────────────────────────────────────────────
    # Загрузка документов — зона OWUI UI и OCR-сервиса, не агента.
    # Агент не загружает файлы: пользователь делает это через OWUI,
    # OCR-сервис извлекает текст, OWUI индексирует в вектор-БД.
    # Агент только читает результат через search_knowledge_base.
    #
    # Единственная операция управления — reset (нужна при переиндексации).

    @agent.tool
    async def reset_knowledge_base(
        ctx: RunContext[AgentDeps],
        knowledge_id: str,
    ) -> str:
        """
        Clear the vector index of a knowledge base without deleting source files.
        Use only when asked explicitly — e.g. after an embedding model change
        when all documents need to be re-indexed from scratch.

        Args:
            knowledge_id: KB id to reset (from list_available_knowledge_bases).
        """
        await reset_knowledge(ctx.deps.owui_token, knowledge_id)
        return f"Knowledge base '{knowledge_id}' vector index cleared."

    # ── 4. Content Ingestion ───────────────────────────────────────────────

    @agent.tool
    async def index_web_page(
        ctx: RunContext[AgentDeps],
        url: str,
        collection_name: str | None = None,
    ) -> str:
        """
        Fetch a web page, extract its text, and store it in OWUI's
        vector database so it can be searched later.

        Args:
            url: Full URL of the page to index.
            collection_name: Optional vector collection name. Auto-generated
                             from URL hash if not provided.

        Returns the collection_name to use in search_single_document.
        """
        result = await process_web_url(ctx.deps.owui_token, url, collection_name)
        coll = result.get("collection_name", "")
        return (
            f"Page indexed: {url}\n"
            f"Collection: {coll}\n"
            f"Use search_single_document('{coll}', ...) to query it."
        )

    @agent.tool
    async def index_text(
        ctx: RunContext[AgentDeps],
        content: str,
        name: str = "text",
        collection_name: str | None = None,
    ) -> str:
        """
        Store arbitrary text in OWUI's vector database for later retrieval.
        Useful for indexing dynamically generated content (e.g. web search
        results) so it can be searched semantically.

        Args:
            content: Text to index.
            name: Human-readable label for this chunk.
            collection_name: Optional collection name. Auto-generated if omitted.

        Returns the collection_name to use in search_single_document.
        """
        result = await process_text(
            ctx.deps.owui_token, content, name=name, collection_name=collection_name
        )
        coll = result.get("collection_name", "")
        return (
            f"Text indexed as '{name}'.\n"
            f"Collection: {coll}\n"
            f"Use search_single_document('{coll}', ...) to query it."
        )

    # ── 5. Web Search ──────────────────────────────────────────────────────

    @agent.tool
    async def web_search(
        ctx: RunContext[AgentDeps],
        queries: list[str],
    ) -> str:
        """
        Search the web using OWUI's configured provider (Perplexity).
        Returns real-time results with titles, URLs, and content snippets.

        Use when:
        - The user needs current information (news, prices, recent events).
        - Knowledge base search returned nothing useful.
        - The question is about facts that change over time.

        Args:
            queries: 1–3 focused search queries. Multiple queries run in
                     parallel inside OWUI and results are merged.
        """
        result = await owui_web_search(ctx.deps.owui_token, queries)

        if not result.get("status"):
            return "Web search returned no results."

        items = result.get("items", [])
        docs  = result.get("docs", [])

        if not items and not docs:
            return "Web search returned no results."

        content_by_url: dict[str, str] = {
            doc.get("metadata", {}).get("source", ""): doc.get("content", "")
            for doc in docs
            if doc.get("metadata", {}).get("source")
        }

        parts, seen = [], set()
        for i, item in enumerate(items, 1):
            url   = item.get("link", "")
            title = item.get("title", f"Result {i}")
            snip  = item.get("snippet", "")
            body  = content_by_url.get(url, "")
            if url in seen:
                continue
            seen.add(url)
            text = (body or snip)[:800]
            parts.append(f"[{i}] {title}\n{url}\n{text}")

        return ("\n\n---\n\n".join(parts)) if parts else "Web search returned no usable results."

    # ── Sub-Agent ──────────────────────────────────────────────────────────

    @agent.tool
    async def sub_agent(
        ctx: RunContext[AgentDeps],
        task: str,
        context: str = "",
    ) -> str:
        """
        Delegate a complex subtask to a separate agent instance.
        Use ONLY when the current task requires more than 5 tool calls —
        to avoid context overflow and keep the main agent focused.

        The sub-agent has access to the same tools (KB, web search, GIS)
        and runs with the same user token and permissions.

        Args:
            task: Full, self-contained description of the subtask.
                  Include all necessary context — the sub-agent has no
                  memory of the current conversation.
            context: Optional background context to prepend (e.g. project name,
                     relevant facts already retrieved in this session).

        Returns the sub-agent's complete response as a string.
        """
        prompt = f"{context}\n\n{task}".strip() if context else task

        payload = {
            "model": "pydantic-agent",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {ctx.deps.owui_token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            r = await client.post(
                f"{ctx.deps.agent_self_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

        choices = data.get("choices", [])
        if choices:
            return choices[0]["message"]["content"]
        return "Sub-agent returned no response."

