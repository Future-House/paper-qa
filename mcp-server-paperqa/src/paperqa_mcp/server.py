"""PaperQA MCP Server - Main server implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import AnyUrl

import paperqa
from paperqa import Docs, Settings, agent_query
from paperqa.agents import build_index, search_query

logger = logging.getLogger(__name__)

# Global state for the server
_docs: Docs | None = None
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        # Check for environment variable to specify settings
        settings_name = os.environ.get("PAPERQA_SETTINGS", "fast")
        _settings = Settings.from_name(settings_name)

        # Override with environment variables if present
        if paper_dir := os.environ.get("PAPERQA_PAPER_DIRECTORY"):
            _settings.agent.index.paper_directory = paper_dir
        if index_dir := os.environ.get("PAPERQA_INDEX_DIRECTORY"):
            _settings.agent.index.index_directory = index_dir

    return _settings


def get_docs() -> Docs:
    """Get or create Docs instance."""
    global _docs
    if _docs is None:
        _docs = Docs()
    return _docs


# Create server instance
app = Server("paperqa-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available PaperQA tools."""
    return [
        Tool(
            name="paperqa_ask",
            description=(
                "Ask a question about scientific papers. The agent will search, "
                "gather evidence, and provide a cited answer based on the documents "
                "in the paper directory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to ask about the papers",
                    },
                    "settings_name": {
                        "type": "string",
                        "description": (
                            "Settings preset to use (fast, high_quality, etc.). "
                            "Defaults to 'fast' or PAPERQA_SETTINGS env var."
                        ),
                        "default": "fast",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="paperqa_add_paper",
            description=(
                "Add a paper (PDF, text file, or URL) to the PaperQA document "
                "collection for future queries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the paper file (PDF, txt, etc.) or URL to download"
                        ),
                    },
                    "citation": {
                        "type": "string",
                        "description": "Optional citation string for the paper",
                    },
                    "docname": {
                        "type": "string",
                        "description": "Optional custom document name",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="paperqa_search",
            description=(
                "Perform a keyword search on the indexed papers. This uses "
                "full-text search to find relevant documents."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "index_name": {
                        "type": "string",
                        "description": "Name of the index to search (default: 'default')",
                        "default": "default",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="paperqa_build_index",
            description=(
                "Build or rebuild the search index from papers in the specified "
                "directory. This enables faster searches."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": (
                            "Directory containing papers to index. If not provided, "
                            "uses the paper_directory from settings."
                        ),
                    },
                    "index_name": {
                        "type": "string",
                        "description": "Name for the index (default: 'default')",
                        "default": "default",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="paperqa_list_docs",
            description="List all documents currently in the PaperQA collection.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="paperqa_get_settings",
            description=(
                "Get the current PaperQA settings configuration, including "
                "paper directory, model settings, and agent configuration."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""

    try:
        if name == "paperqa_ask":
            return await handle_ask(arguments)
        elif name == "paperqa_add_paper":
            return await handle_add_paper(arguments)
        elif name == "paperqa_search":
            return await handle_search(arguments)
        elif name == "paperqa_build_index":
            return await handle_build_index(arguments)
        elif name == "paperqa_list_docs":
            return await handle_list_docs(arguments)
        elif name == "paperqa_get_settings":
            return await handle_get_settings(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.exception(f"Error executing tool {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_ask(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle paperqa_ask tool call."""
    query = arguments["query"]
    settings_name = arguments.get("settings_name", "fast")

    # Load settings
    settings = Settings.from_name(settings_name)

    # Override with environment variables
    if paper_dir := os.environ.get("PAPERQA_PAPER_DIRECTORY"):
        settings.agent.index.paper_directory = paper_dir
    if index_dir := os.environ.get("PAPERQA_INDEX_DIRECTORY"):
        settings.agent.index.index_directory = index_dir

    # Configure logging to be less verbose
    settings.verbosity = 0

    # Run the query
    response = await agent_query(query, settings)

    # Format the response
    result = f"## Question\n{query}\n\n"
    result += f"## Answer\n{response.answer.answer}\n\n"

    if response.answer.sources:
        result += "## Sources\n"
        for source in response.answer.sources:
            result += f"- {source}\n"

    if response.answer.evidence:
        result += f"\n## Evidence ({len(response.answer.evidence)} items)\n"
        for i, evidence in enumerate(response.answer.evidence[:5], 1):  # Show first 5
            result += f"\n**{i}. {evidence.text.name}**\n"
            result += f"> {evidence.text.text[:200]}...\n"

    return [TextContent(type="text", text=result)]


async def handle_add_paper(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle paperqa_add_paper tool call."""
    path = arguments["path"]
    citation = arguments.get("citation")
    docname = arguments.get("docname")

    docs = get_docs()
    settings = get_settings()

    # Check if it's a URL or file path
    if path.startswith("http://") or path.startswith("https://"):
        # Download the file
        import urllib.request
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            urllib.request.urlretrieve(path, tmp_file.name)
            tmp_path = tmp_file.name

        try:
            with open(tmp_path, "rb") as f:
                await docs.aadd(f, citation=citation, docname=docname, settings=settings)
            result = f"Successfully added paper from URL: {path}"
        finally:
            os.unlink(tmp_path)
    else:
        # Local file
        path_obj = Path(path).expanduser().resolve()
        if not path_obj.exists():
            return [TextContent(type="text", text=f"Error: File not found: {path}")]

        with open(path_obj, "rb") as f:
            await docs.aadd(f, citation=citation, docname=docname, settings=settings)
        result = f"Successfully added paper: {path}"

    return [TextContent(type="text", text=result)]


async def handle_search(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle paperqa_search tool call."""
    query = arguments["query"]
    index_name = arguments.get("index_name", "default")

    settings = get_settings()

    # Perform search
    results = await search_query(query, index_name, settings)

    if not results:
        return [TextContent(type="text", text=f"No results found for: {query}")]

    # Format results
    result = f"## Search Results for: {query}\n\n"
    result += f"Found {len(results)} results\n\n"

    for i, (item, path) in enumerate(results[:10], 1):  # Show first 10
        result += f"**{i}.** {path}\n"
        if hasattr(item, "answer"):
            result += f"   {item.answer.answer[:200]}...\n\n"

    return [TextContent(type="text", text=result)]


async def handle_build_index(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle paperqa_build_index tool call."""
    directory = arguments.get("directory")
    index_name = arguments.get("index_name", "default")

    settings = get_settings()

    # Build index
    index = await build_index(
        index_name=index_name,
        directory=directory,
        settings=settings,
    )

    result = f"Successfully built index '{index_name}'"
    if directory:
        result += f" from directory: {directory}"

    return [TextContent(type="text", text=result)]


async def handle_list_docs(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle paperqa_list_docs tool call."""
    docs = get_docs()

    if not docs.docs:
        return [TextContent(type="text", text="No documents in collection.")]

    result = f"## Documents in Collection ({len(docs.docs)})\n\n"

    for dockey, doc in docs.docs.items():
        if isinstance(doc, paperqa.DocDetails):
            result += f"- **{doc.title or 'Untitled'}**\n"
            if doc.authors:
                result += f"  Authors: {', '.join(doc.authors[:3])}\n"
            if doc.year:
                result += f"  Year: {doc.year}\n"
            result += f"  Key: {dockey}\n\n"
        else:
            result += f"- Key: {dockey}\n\n"

    return [TextContent(type="text", text=result)]


async def handle_get_settings(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle paperqa_get_settings tool call."""
    settings = get_settings()

    # Format key settings
    result = "## PaperQA Settings\n\n"
    result += f"**Paper Directory:** {settings.agent.index.paper_directory}\n"
    result += f"**Index Directory:** {settings.agent.index.index_directory}\n"
    result += f"**Agent Type:** {settings.agent.agent_type}\n"
    result += f"**LLM Model:** {settings.llm}\n"
    result += f"**Embedding Model:** {settings.embedding}\n"
    result += f"**Summary LLM:** {settings.summary_llm}\n"
    result += f"**Evidence K:** {settings.answer.evidence_k}\n"
    result += f"**Max Sources:** {settings.answer.max_sources}\n"

    return [TextContent(type="text", text=result)]


def main():
    """Run the MCP server."""
    import sys
    from mcp.server.stdio import stdio_server

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting PaperQA MCP Server")

    # Run the server
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
