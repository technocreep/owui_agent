"""
OWUI API client.

Покрывает все публичные эндпоинты, нужные агенту:

  Knowledge Base  (/api/v1/knowledge/*)
    - list_knowledge_bases          GET  /
    - search_knowledge_bases        GET  /search
    - get_knowledge_meta            GET  /{id}
    - get_knowledge_files           GET  /{id}/files
    - create_knowledge              POST /create
    - add_file_to_knowledge         POST /{id}/file/add
    - remove_file_from_knowledge    POST /{id}/file/remove
    - reset_knowledge               POST /{id}/reset

  Files  (/api/v1/files/*)
    - upload_file                   POST /
    - get_file_process_status       GET  /{id}/process/status
    - get_file_content              GET  /{id}/data/content

  Retrieval  (/api/v1/retrieval/*)
    - query_knowledge_base          POST /query/collection
    - query_single_doc              POST /query/doc
    - process_web_url               POST /process/web
    - process_text                  POST /process/text

  Web Search  (/api/v1/retrieval/*)
    - owui_web_search               POST /process/web/search

  Chat  (/api/*)
    - owui_chat_completion          POST /api/chat/completions
"""

from __future__ import annotations

import os
from typing import Any

import httpx

OWUI_BASE_URL    = os.getenv("OWUI_BASE_URL", "http://open-webui:3000")
OWUI_SERVICE_TOKEN = os.getenv("OWUI_SERVICE_TOKEN", "")

_TIMEOUT      = httpx.Timeout(30.0)
_LONG_TIMEOUT = httpx.Timeout(120.0)


def _headers(token: str) -> dict[str, str]:
    tok = token or OWUI_SERVICE_TOKEN
    return {
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════

async def list_knowledge_bases(token: str, page: int = 1) -> list[dict[str, Any]]:
    """GET /api/v1/knowledge/ — список всех KB пользователя (с пагинацией)."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.get(
            f"{OWUI_BASE_URL}/api/v1/knowledge/",
            headers=_headers(token),
            params={"page": page},
        )
        r.raise_for_status()
        data = r.json()
        # Ответ: { items: [...], total: N }  или просто list
        return data.get("items", data) if isinstance(data, dict) else data


async def search_knowledge_bases(
    token: str,
    query: str,
    page: int = 1,
) -> list[dict[str, Any]]:
    """GET /api/v1/knowledge/search — полнотекстовый поиск по названиям/описаниям KB."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.get(
            f"{OWUI_BASE_URL}/api/v1/knowledge/search",
            headers=_headers(token),
            params={"query": query, "page": page},
        )
        r.raise_for_status()
        data = r.json()
        return data.get("items", data) if isinstance(data, dict) else data


async def get_knowledge_meta(token: str, knowledge_id: str) -> dict[str, Any]:
    """GET /api/v1/knowledge/{id} — метаданные KB + список файлов."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.get(
            f"{OWUI_BASE_URL}/api/v1/knowledge/{knowledge_id}",
            headers=_headers(token),
        )
        r.raise_for_status()
        return r.json()


async def get_knowledge_files(
    token: str,
    knowledge_id: str,
    query: str | None = None,
    page: int = 1,
) -> list[dict[str, Any]]:
    """GET /api/v1/knowledge/{id}/files — файлы внутри KB (с фильтром и пагинацией)."""
    params: dict[str, Any] = {"page": page}
    if query:
        params["query"] = query
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.get(
            f"{OWUI_BASE_URL}/api/v1/knowledge/{knowledge_id}/files",
            headers=_headers(token),
            params=params,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("items", data) if isinstance(data, dict) else data


async def create_knowledge(
    token: str,
    name: str,
    description: str = "",
) -> dict[str, Any]:
    """POST /api/v1/knowledge/create — создать новую KB."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/v1/knowledge/create",
            headers=_headers(token),
            json={"name": name, "description": description},
        )
        r.raise_for_status()
        return r.json()


async def add_file_to_knowledge(
    token: str,
    knowledge_id: str,
    file_id: str,
) -> dict[str, Any]:
    """POST /api/v1/knowledge/{id}/file/add — добавить уже загруженный файл в KB."""
    async with httpx.AsyncClient(timeout=_LONG_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/v1/knowledge/{knowledge_id}/file/add",
            headers=_headers(token),
            json={"file_id": file_id},
        )
        r.raise_for_status()
        return r.json()


async def remove_file_from_knowledge(
    token: str,
    knowledge_id: str,
    file_id: str,
) -> dict[str, Any]:
    """POST /api/v1/knowledge/{id}/file/remove — удалить файл из KB."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/v1/knowledge/{knowledge_id}/file/remove",
            headers=_headers(token),
            json={"file_id": file_id},
        )
        r.raise_for_status()
        return r.json()


async def reset_knowledge(token: str, knowledge_id: str) -> dict[str, Any]:
    """POST /api/v1/knowledge/{id}/reset — очистить вектор-коллекцию KB (без удаления файлов)."""
    async with httpx.AsyncClient(timeout=_LONG_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/v1/knowledge/{knowledge_id}/reset",
            headers=_headers(token),
        )
        r.raise_for_status()
        return r.json()


# ═══════════════════════════════════════════════════════════════════════════
# Files
# ═══════════════════════════════════════════════════════════════════════════

async def upload_file(
    token: str,
    filename: str,
    content: bytes,
    content_type: str = "application/octet-stream",
    process: bool = True,
) -> dict[str, Any]:
    """
    POST /api/v1/files/ — загрузить файл (multipart/form-data).
    OWUI автоматически запускает embedding в фоне (process_in_background=True).
    Вернёт file_id, который затем можно передать в add_file_to_knowledge.
    """
    files = {"file": (filename, content, content_type)}
    params = {"process": str(process).lower(), "process_in_background": "true"}
    headers = {"Authorization": f"Bearer {token or OWUI_SERVICE_TOKEN}"}
    async with httpx.AsyncClient(timeout=_LONG_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/v1/files/",
            headers=headers,
            files=files,
            params=params,
        )
        r.raise_for_status()
        return r.json()


async def get_file_process_status(token: str, file_id: str) -> str:
    """
    GET /api/v1/files/{id}/process/status
    Возвращает строку статуса: 'completed' | 'failed' | 'pending' | 'processing'.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.get(
            f"{OWUI_BASE_URL}/api/v1/files/{file_id}/process/status",
            headers=_headers(token),
        )
        r.raise_for_status()
        data = r.json()
        return data.get("status", "unknown")


async def get_file_content(token: str, file_id: str) -> str:
    """GET /api/v1/files/{id}/data/content — извлечённый текст файла."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.get(
            f"{OWUI_BASE_URL}/api/v1/files/{file_id}/data/content",
            headers=_headers(token),
        )
        r.raise_for_status()
        data = r.json()
        return data.get("content", "")


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════════════════

async def query_knowledge_base(
    token: str,
    collection_names: list[str],
    query: str,
    k: int = 5,
    hybrid: bool = True,
) -> list[dict[str, Any]]:
    """
    POST /api/v1/retrieval/query/collection
    Векторный (или hybrid BM25+vector) поиск по одной или нескольким коллекциям.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/v1/retrieval/query/collection",
            headers=_headers(token),
            json={
                "collection_names": collection_names,
                "query": query,
                "k": k,
                "hybrid": hybrid,
            },
        )
        r.raise_for_status()
        data = r.json()
        docs: list[dict[str, Any]] = []
        for chunks, metas in zip(
            data.get("documents", [[]]),
            data.get("metadatas", [[]]),
        ):
            for chunk, meta in zip(chunks, metas):
                docs.append({"content": chunk, "meta": meta})
        return docs


async def query_single_doc(
    token: str,
    collection_name: str,
    query: str,
    k: int = 5,
    hybrid: bool = True,
) -> list[dict[str, Any]]:
    """
    POST /api/v1/retrieval/query/doc
    Поиск по одному документу/коллекции (file-level, не KB-level).
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/v1/retrieval/query/doc",
            headers=_headers(token),
            json={
                "collection_name": collection_name,
                "query": query,
                "k": k,
                "hybrid": hybrid,
            },
        )
        r.raise_for_status()
        data = r.json()
        docs: list[dict[str, Any]] = []
        for chunks, metas in zip(
            data.get("documents", [[]]),
            data.get("metadatas", [[]]),
        ):
            for chunk, meta in zip(chunks, metas):
                docs.append({"content": chunk, "meta": meta})
        return docs


async def process_web_url(
    token: str,
    url: str,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """
    POST /api/v1/retrieval/process/web
    Загрузить страницу по URL, извлечь текст и сохранить в вектор-коллекцию.
    Возвращает { collection_name, filename, file: {data: {content}} }.
    """
    payload: dict[str, Any] = {"url": url}
    if collection_name:
        payload["collection_name"] = collection_name
    async with httpx.AsyncClient(timeout=_LONG_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/v1/retrieval/process/web",
            headers=_headers(token),
            json=payload,
            params={"process": "true", "overwrite": "true"},
        )
        r.raise_for_status()
        return r.json()


async def process_text(
    token: str,
    content: str,
    name: str = "text",
    collection_name: str | None = None,
) -> dict[str, Any]:
    """
    POST /api/v1/retrieval/process/text
    Сохранить произвольный текст в вектор-коллекцию.
    Возвращает { collection_name, content }.
    """
    payload: dict[str, Any] = {"content": content, "name": name}
    if collection_name:
        payload["collection_name"] = collection_name
    async with httpx.AsyncClient(timeout=_LONG_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/v1/retrieval/process/text",
            headers=_headers(token),
            json=payload,
        )
        r.raise_for_status()
        return r.json()


# ═══════════════════════════════════════════════════════════════════════════
# Web Search
# ═══════════════════════════════════════════════════════════════════════════

async def owui_web_search(
    token: str,
    queries: list[str],
    bypass_embedding: bool = True,
) -> dict[str, Any]:
    """
    POST /api/v1/retrieval/process/web/search
    Запускает веб-поиск через провайдер, настроенный в OWUI (Perplexity и др.).

    Ответ при BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=true (рекомендуется):
      { status, collection_name: null, filenames, items, docs, loaded_count }
    Ответ при сохранении в вектор-БД:
      { status, collection_names: [...], items, filenames, loaded_count }
    """
    async with httpx.AsyncClient(timeout=_LONG_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/v1/retrieval/process/web/search",
            headers=_headers(token),
            json={"queries": queries},
        )
        r.raise_for_status()
        return r.json()


# ═══════════════════════════════════════════════════════════════════════════
# Chat (Способ B — делегирование OWUI pipeline)
# ═══════════════════════════════════════════════════════════════════════════

async def owui_chat_completion(
    token: str,
    messages: list[dict[str, Any]],
    model: str,
    knowledge_ids: list[str] | None = None,
    stream: bool = False,
) -> Any:
    """
    POST /api/chat/completions
    OWUI выполняет RAG, применяет фильтры и вызывает LLM.
    knowledge_ids → OWUI сам делает retrieval и инжектирует контекст.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if knowledge_ids:
        payload["files"] = [{"type": "collection", "id": kid} for kid in knowledge_ids]
    async with httpx.AsyncClient(timeout=_LONG_TIMEOUT) as c:
        r = await c.post(
            f"{OWUI_BASE_URL}/api/chat/completions",
            headers=_headers(token),
            json=payload,
        )
        r.raise_for_status()
        return r.json()
