"""
GIS tools для PydanticAI агента.
Вызывают arcgis_mcp REST API через gis_client.py.

Все инструменты доступны только если GIS_BASE_URL задан в env.
Если не задан — инструменты не регистрируются.
"""

from __future__ import annotations
import json
from typing import Any

from pydantic_ai import RunContext
from app.models.deps import AgentDeps
from app import gis_client


def _fmt(data: Any) -> str:
    """Форматирует ответ GIS API в строку для агента."""
    if isinstance(data, str):
        return data
    return json.dumps(data, ensure_ascii=False, indent=2)


def _markdown_or_fmt(data: Any) -> str:
    """Возвращает markdown-ссылку на карту если есть, иначе JSON."""
    if isinstance(data, dict):
        if md := data.get("markdown"):
            return md
        if link := data.get("link"):
            return link
    return _fmt(data)


def register_gis_tools(agent) -> bool:
    """
    Регистрирует GIS-инструменты в агенте.
    Возвращает True если GIS_BASE_URL задан, False если нет.
    """
    if not gis_client.GIS_BASE_URL:
        return False

    # ── Inventory (P0) ─────────────────────────────────────────────────────

    @agent.tool
    async def gis_list_projects(ctx: RunContext[AgentDeps]) -> str:
        """
        List all available GIS geological projects.
        ALWAYS call this first before any other GIS tool to get project_id.
        Returns: list of projects with id, name, layer count.
        """
        return _fmt(await gis_client.list_projects())

    @agent.tool
    async def gis_get_project_summary(
        ctx: RunContext[AgentDeps],
        project_id: str,
    ) -> str:
        """
        Get summary of a GIS project: name, map extent, total layers,
        non-empty layers count. Call after gis_list_projects.

        Args:
            project_id: Project ID from gis_list_projects.
        """
        return _fmt(await gis_client.get_project_summary(project_id))

    @agent.tool
    async def gis_list_layers(
        ctx: RunContext[AgentDeps],
        project_id: str,
        group: str | None = None,
        include_needs_review: bool = False,
    ) -> str:
        """
        List layers in a GIS project. Fast — reads from manifest.
        Use display_name when referring to layers in responses.
        Layers with needs_review=true may have inaccurate names — warn the user.

        Args:
            project_id: Project ID from gis_list_projects.
            group: Optional group filter (e.g. 'geophysics', 'geology').
            include_needs_review: Include layers flagged for name review.
        """
        return _fmt(await gis_client.list_layers(
            project_id, group=group, include_needs_review=include_needs_review
        ))

    @agent.tool
    async def gis_describe_layer(
        ctx: RunContext[AgentDeps],
        project_id: str,
        layer: str,
    ) -> str:
        """
        Get detailed metadata for a layer: geometry type, feature count,
        CRS, extent, field names and types. Fast — reads from manifest.

        Args:
            project_id: Project ID.
            layer: Layer display_name, layer_id, or alias.
        """
        return _fmt(await gis_client.describe_layer(project_id, layer))

    # ── Query (P1 — reads .gdb) ────────────────────────────────────────────

    @agent.tool
    async def gis_query_features(
        ctx: RunContext[AgentDeps],
        project_id: str,
        layer: str,
        filters: str | None = None,
        limit: int = 50,
        fields: str | None = None,
    ) -> str:
        """
        Query features from a layer with optional filters. Reads .gdb directly.
        Use only when manifest data is insufficient.

        Args:
            project_id: Project ID.
            layer: Layer display_name, layer_id, or alias.
            filters: JSON string {field: value}, supports >=, <=, partial match.
                     Example: '{"YEAR": {">=": 2010}, "TYPE": "gold"}'
            limit: Max features to return (1-500, default 50).
            fields: Comma-separated field names to return. None = all fields.
        """
        return _fmt(await gis_client.query_features(
            project_id, layer, filters=filters, limit=limit, fields=fields
        ))

    @agent.tool
    async def gis_summarize_layer(
        ctx: RunContext[AgentDeps],
        project_id: str,
        layer: str,
    ) -> str:
        """
        Get statistical summary of a layer's numeric fields.
        Returns min/max/mean/std for each numeric field. Reads .gdb.

        Args:
            project_id: Project ID.
            layer: Layer display_name, layer_id, or alias.
        """
        return _fmt(await gis_client.summarize_layer(project_id, layer))

    # ── Izuchennost ────────────────────────────────────────────────────────

    @agent.tool
    async def gis_search_izuchennost(
        ctx: RunContext[AgentDeps],
        project_id: str,
    ) -> str:
        """
        Search study cards (карты изученности) in the project.
        Returns summary of available geological study records.

        Args:
            project_id: Project ID.
        """
        return _fmt(await gis_client.search_izuchennost(project_id))

    @agent.tool
    async def gis_get_izuchennost_records(
        ctx: RunContext[AgentDeps],
        project_id: str,
        year_from: int | None = None,
        year_to: int | None = None,
        method: str | None = None,
        organization: str | None = None,
        limit: int = 50,
    ) -> str:
        """
        Get filtered geological study records (изученность).

        Args:
            project_id: Project ID.
            year_from: Start year filter (inclusive).
            year_to: End year filter (inclusive).
            method: Work method filter (partial match).
            organization: Organization name filter (partial match).
            limit: Max records (1-200, default 50).
        """
        return _fmt(await gis_client.get_izuchennost_records(
            project_id, year_from=year_from, year_to=year_to,
            method=method, organization=organization, limit=limit
        ))

    @agent.tool
    async def gis_list_attachments(
        ctx: RunContext[AgentDeps],
        project_id: str,
        layer: str | None = None,
    ) -> str:
        """
        List file attachments in the project (PDFs, reports, etc.).
        Returns total count, size, breakdown by table.
        Note: lists only — use gis_geo_context_query to read PDF content.

        Args:
            project_id: Project ID.
            layer: Optional layer filter.
        """
        return _fmt(await gis_client.list_attachments(project_id, layer=layer))

    # ── Visualisation ──────────────────────────────────────────────────────

    @agent.tool
    async def gis_plot_layer(
        ctx: RunContext[AgentDeps],
        project_id: str,
        layer_id: str,
        color_field: str | None = None,
        show_license: bool = True,
        title: str | None = None,
        colormap: str = "viridis",
    ) -> str:
        """
        Render a single GIS layer as a map image.
        Insert the returned markdown link verbatim into the response — it renders the map.

        Styling rules:
        - Geophysical layers (units in brackets: мГал, нТл, Э): color_field=<first numeric field>
        - Geological / stratigraphy: color_field='INDEX'
        - Boreholes: color_field='POINT_Z'
        - Always show_license=True

        Args:
            project_id: Project ID.
            layer_id: Layer display_name or layer_id from gis_list_layers.
            color_field: Field name for color coding (None = single color).
            show_license: Draw license boundary (default True).
            title: Map title (auto-generated if None).
            colormap: Matplotlib colormap name (default 'viridis').
        """
        result = await gis_client.plot_layer(
            project_id, layer_id,
            color_field=color_field,
            show_license=show_license,
            title=title,
            colormap=colormap,
        )
        return _markdown_or_fmt(result)

    @agent.tool
    async def gis_plot_overlay(
        ctx: RunContext[AgentDeps],
        project_id: str,
        layers: str,
        show_license: bool = True,
        title: str | None = None,
    ) -> str:
        """
        Render multiple GIS layers overlaid on one map.
        Insert the returned markdown link verbatim into the response.

        Styling rules:
        - Lineaments: color='#00FF00'
        - Tectonics: use different colors/linewidth for thrusts vs faults
        - Always show_license=True

        Args:
            project_id: Project ID.
            layers: JSON string describing layers to overlay.
                    Example: '[{"layer_id": "faults", "color": "#FF0000"},
                               {"layer_id": "geology", "color_field": "INDEX"}]'
            show_license: Draw license boundary (default True).
            title: Map title (auto-generated if None).
        """
        result = await gis_client.plot_overlay(
            project_id, layers, show_license=show_license, title=title
        )
        return _markdown_or_fmt(result)

    @agent.tool
    async def gis_plot_relief(
        ctx: RunContext[AgentDeps],
        project_id: str,
        layer_id: str,
        show_rivers: bool = True,
        show_license: bool = True,
        title: str | None = None,
    ) -> str:
        """
        Render a relief (topography) map from contour lines layer.
        Insert the returned markdown link verbatim into the response.

        Args:
            project_id: Project ID.
            layer_id: Contour lines layer display_name or layer_id.
            show_rivers: Overlay river layers (default True).
            show_license: Draw license boundary (default True).
            title: Map title (auto-generated if None).
        """
        result = await gis_client.plot_relief(
            project_id, layer_id,
            show_rivers=show_rivers, show_license=show_license, title=title
        )
        return _markdown_or_fmt(result)

    @agent.tool
    async def gis_plot_histogram(
        ctx: RunContext[AgentDeps],
        project_id: str,
        layer_id: str,
        field: str,
        plot_type: str = "histogram",
        group_by: str | None = None,
        bins: int = 50,
        title: str | None = None,
    ) -> str:
        """
        Plot a histogram or bar chart for a numeric field in a layer.
        Insert the returned markdown link verbatim into the response.

        Args:
            project_id: Project ID.
            layer_id: Layer display_name or layer_id.
            field: Field name to analyze (must be numeric).
            plot_type: 'histogram' or 'bar' (default 'histogram').
            group_by: Optional categorical field to group bars by.
            bins: Number of bins for histogram (5-500, default 50).
            title: Chart title (auto-generated if None).
        """
        result = await gis_client.plot_histogram(
            project_id, layer_id, field,
            plot_type=plot_type, group_by=group_by, bins=bins, title=title
        )
        return _markdown_or_fmt(result)

    # ── Data Cube / ML ─────────────────────────────────────────────────────

    @agent.tool
    async def gis_datacube_overview(
        ctx: RunContext[AgentDeps],
        project_id: str,
        scenario_id: str | None = None,
    ) -> str:
        """
        Get ML prospectivity model overview: PR-AUC, CV scores,
        score distribution by decile. ALWAYS call this before
        gis_datacube_block_scores or gis_datacube_block_detail.

        Args:
            project_id: Project ID.
            scenario_id: Scenario ID (None = default scenario).
        """
        return _fmt(await gis_client.datacube_overview(project_id, scenario_id=scenario_id))

    @agent.tool
    async def gis_datacube_block_scores(
        ctx: RunContext[AgentDeps],
        project_id: str,
        min_score: float | None = None,
        scenario_id: str | None = None,
    ) -> str:
        """
        Get prospectivity scores for all blocks. Call after gis_datacube_overview.
        Returns score stats and driver group breakdown.

        Args:
            project_id: Project ID.
            min_score: Filter blocks with score >= min_score (0.0-1.0).
            scenario_id: Scenario ID (None = default).
        """
        return _fmt(await gis_client.datacube_block_scores(
            project_id, min_score=min_score, scenario_id=scenario_id
        ))

    @agent.tool
    async def gis_datacube_block_detail(
        ctx: RunContext[AgentDeps],
        project_id: str,
        block_id: str,
        scenario_id: str | None = None,
    ) -> str:
        """
        Get full detail for a specific block: score, rank, coordinates,
        all feature values, driver contributions.

        Args:
            project_id: Project ID.
            block_id: Block ID from gis_datacube_block_scores (e.g. 'block_2_0').
            scenario_id: Scenario ID (None = default).
        """
        return _fmt(await gis_client.datacube_block_detail(
            project_id, block_id, scenario_id=scenario_id
        ))

    @agent.tool
    async def gis_datacube_score_overlay(
        ctx: RunContext[AgentDeps],
        project_id: str,
        scenario_id: str | None = None,
        quantile: str = "top10",
        visualization_type: str = "blocks",
        layers: str | None = None,
    ) -> str:
        """
        Render prospectivity score map overlaid on geological layers.
        Workflow: gis_datacube_report_overview → gis_datacube_score_overlay.
        Insert the returned markdown link verbatim into the response.

        Args:
            project_id: Project ID.
            scenario_id: Scenario ID (None = default).
            quantile: Which blocks to highlight: 'top10', 'top20', 'top30'.
            visualization_type: 'blocks' or 'heatmap'.
            layers: Optional JSON string of additional layers to overlay.
        """
        result = await gis_client.datacube_score_overlay(
            project_id, scenario_id=scenario_id,
            quantile=quantile, visualization_type=visualization_type,
            layers=layers,
        )
        return _markdown_or_fmt(result)

    # ── Knowledge Graph ────────────────────────────────────────────────────

    @agent.tool
    async def gis_geo_context_query(
        ctx: RunContext[AgentDeps],
        query: str,
        project_id: str | None = None,
    ) -> str:
        """
        Query the geological knowledge graph using natural language.
        Use for: geological context, stratigraphic relations, work history,
        PDF attachment content, terminology lookups.

        Args:
            query: Natural language query in any language.
            project_id: Optional project context for scoped queries.
        """
        return _fmt(await gis_client.geo_context_query(query, project_id=project_id))

    @agent.tool
    async def gis_lookup_work_types(
        ctx: RunContext[AgentDeps],
        codes: list[str],
    ) -> str:
        """
        Look up geological work type descriptions by code.
        Use when izuchennost records contain numeric work type codes
        that need human-readable labels.

        Args:
            codes: List of work type code strings (e.g. ['01', '02', '15']).
        """
        return _fmt(await gis_client.lookup_work_types(codes))

    return True
