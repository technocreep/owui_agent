"""
GIS API client — вызывает arcgis_mcp REST API напрямую.

Сервер поднят как FastAPI (api_server/server.py), все эндпоинты POST.
Базовый URL задаётся через GIS_BASE_URL (без /mcp — это REST, не MCP).
"""

from __future__ import annotations

import os
from typing import Any

import httpx

GIS_BASE_URL:  str | None = os.getenv("GIS_BASE_URL")
GIS_API_TOKEN: str | None = os.getenv("GIS_MCP_TOKEN")   # тот же токен

_TIMEOUT      = httpx.Timeout(30.0)
_LONG_TIMEOUT = httpx.Timeout(120.0)   # для plot_*, query_features


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if GIS_API_TOKEN:
        h["Authorization"] = f"Bearer {GIS_API_TOKEN}"
    return h


async def _post(endpoint: str, payload: dict[str, Any] = {}, long: bool = False) -> Any:
    if not GIS_BASE_URL:
        raise RuntimeError("GIS_BASE_URL not configured")
    url = f"{GIS_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    timeout = _LONG_TIMEOUT if long else _TIMEOUT
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(url, headers=_headers(), json=payload)
        r.raise_for_status()
        return r.json()


# ── Inventory (P0 — читает manifest, быстро) ──────────────────────────────

async def list_projects() -> Any:
    return await _post("/list_projects")

async def get_project_summary(project_id: str) -> Any:
    return await _post("/get_project_summary", {"project_id": project_id})

async def list_layers(
    project_id: str,
    group: str | None = None,
    include_needs_review: bool = False,
    output_format: str = "text",
) -> Any:
    payload: dict[str, Any] = {
        "project_id": project_id,
        "include_needs_review": include_needs_review,
        "output_format": output_format,
    }
    if group:
        payload["group"] = group
    return await _post("/list_layers", payload)

async def describe_layer(project_id: str, layer: str) -> Any:
    return await _post("/describe_layer", {"project_id": project_id, "layer": layer})


# ── Query (P1 — читает .gdb) ──────────────────────────────────────────────

async def query_features(
    project_id: str,
    layer: str,
    filters: str | None = None,
    limit: int = 50,
    fields: str | None = None,
) -> Any:
    payload: dict[str, Any] = {"project_id": project_id, "layer": layer, "limit": limit}
    if filters:
        payload["filters"] = filters
    if fields:
        payload["fields"] = fields
    return await _post("/query_features", payload, long=True)

async def summarize_layer(project_id: str, layer: str) -> Any:
    return await _post("/summarize_layer", {"project_id": project_id, "layer": layer}, long=True)


# ── Izuchennost ────────────────────────────────────────────────────────────

async def search_izuchennost(project_id: str) -> Any:
    return await _post("/search_izuchennost", {"project_id": project_id})

async def get_izuchennost_records(
    project_id: str,
    year_from: int | None = None,
    year_to: int | None = None,
    method: str | None = None,
    organization: str | None = None,
    limit: int = 50,
) -> Any:
    payload: dict[str, Any] = {"project_id": project_id, "limit": limit}
    if year_from:  payload["year_from"]    = year_from
    if year_to:    payload["year_to"]      = year_to
    if method:     payload["method"]       = method
    if organization: payload["organization"] = organization
    return await _post("/get_izuchennost_records", payload)

async def list_attachments(project_id: str, layer: str | None = None) -> Any:
    payload: dict[str, Any] = {"project_id": project_id}
    if layer:
        payload["layer"] = layer
    return await _post("/list_attachments", payload)


# ── Visualisation ─────────────────────────────────────────────────────────

async def plot_layer(
    project_id: str,
    layer_id: str,
    color_field: str | None = None,
    style: str = "auto",
    colormap: str = "viridis",
    show_license: bool = True,
    bbox_wgs84: str | None = None,
    title: str | None = None,
    output_format: str = "png",
) -> Any:
    payload: dict[str, Any] = {
        "project_id": project_id,
        "layer_id": layer_id,
        "style": style,
        "colormap": colormap,
        "show_license": show_license,
        "output_format": output_format,
    }
    if color_field: payload["color_field"] = color_field
    if bbox_wgs84:  payload["bbox_wgs84"]  = bbox_wgs84
    if title:       payload["title"]       = title
    return await _post("/plot_layer", payload, long=True)

async def plot_overlay(
    project_id: str,
    layers: str,
    show_license: bool = True,
    show_legend: bool = True,
    title: str | None = None,
    output_format: str = "png",
) -> Any:
    payload: dict[str, Any] = {
        "project_id": project_id,
        "layers": layers,
        "show_license": show_license,
        "show_legend": show_legend,
        "output_format": output_format,
    }
    if title: payload["title"] = title
    return await _post("/plot_overlay", payload, long=True)

async def plot_relief(
    project_id: str,
    layer_id: str,
    show_rivers: bool = True,
    show_license: bool = True,
    title: str | None = None,
    output_format: str = "png",
) -> Any:
    payload: dict[str, Any] = {
        "project_id": project_id,
        "layer_id": layer_id,
        "show_rivers": show_rivers,
        "show_license": show_license,
        "output_format": output_format,
    }
    if title: payload["title"] = title
    return await _post("/plot_relief", payload, long=True)

async def plot_histogram(
    project_id: str,
    layer_id: str,
    field: str,
    plot_type: str = "histogram",
    group_by: str | None = None,
    bins: int = 50,
    title: str | None = None,
    output_format: str = "png",
) -> Any:
    payload: dict[str, Any] = {
        "project_id": project_id,
        "layer_id": layer_id,
        "field": field,
        "plot_type": plot_type,
        "bins": bins,
        "output_format": output_format,
    }
    if group_by: payload["group_by"] = group_by
    if title:    payload["title"]    = title
    return await _post("/plot_histogram", payload, long=True)


# ── Data Cube / ML ────────────────────────────────────────────────────────

async def datacube_overview(project_id: str, scenario_id: str | None = None) -> Any:
    payload: dict[str, Any] = {"project_id": project_id}
    if scenario_id: payload["scenario_id"] = scenario_id
    return await _post("/datacube_overview", payload)

async def datacube_block_scores(
    project_id: str,
    min_score: float | None = None,
    scenario_id: str | None = None,
) -> Any:
    payload: dict[str, Any] = {"project_id": project_id}
    if min_score is not None: payload["min_score"]   = min_score
    if scenario_id:           payload["scenario_id"] = scenario_id
    return await _post("/datacube_block_scores", payload)

async def datacube_block_detail(
    project_id: str,
    block_id: str,
    scenario_id: str | None = None,
) -> Any:
    payload: dict[str, Any] = {"project_id": project_id, "block_id": block_id}
    if scenario_id: payload["scenario_id"] = scenario_id
    return await _post("/datacube_block_detail", payload)

async def datacube_report_overview(project_id: str) -> Any:
    return await _post("/datacube_report_overview", {"project_id": project_id})

async def datacube_score_overlay(
    project_id: str,
    scenario_id: str | None = None,
    quantile: str = "top10",
    visualization_type: str = "blocks",
    layers: str | None = None,
) -> Any:
    payload: dict[str, Any] = {
        "project_id": project_id,
        "quantile": quantile,
        "visualization_type": visualization_type,
    }
    if scenario_id: payload["scenario_id"] = scenario_id
    if layers:      payload["layers"]      = layers
    return await _post("/datacube_score_overlay", payload, long=True)


# ── Knowledge Graph ───────────────────────────────────────────────────────

async def geo_context_query(query: str, project_id: str | None = None) -> Any:
    payload: dict[str, Any] = {"query": query}
    if project_id: payload["project_id"] = project_id
    return await _post("/geo_context_query", payload)

async def lookup_work_types(codes: list[str]) -> Any:
    return await _post("/lookup_work_types", {"codes": codes})
