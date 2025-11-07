"""Tests for the PaperQA MCP Server."""

import pytest

from paperqa_mcp.server import app, list_tools


@pytest.mark.asyncio
async def test_list_tools():
    """Test that the server can list available tools."""
    tools = await list_tools()

    assert len(tools) > 0

    # Check that expected tools are present
    tool_names = [tool.name for tool in tools]

    expected_tools = [
        "paperqa_ask",
        "paperqa_add_paper",
        "paperqa_search",
        "paperqa_build_index",
        "paperqa_list_docs",
        "paperqa_get_settings",
    ]

    for expected_tool in expected_tools:
        assert expected_tool in tool_names, f"Tool {expected_tool} not found"


@pytest.mark.asyncio
async def test_paperqa_ask_tool_schema():
    """Test that paperqa_ask tool has correct schema."""
    tools = await list_tools()
    ask_tool = next(tool for tool in tools if tool.name == "paperqa_ask")

    assert ask_tool is not None
    assert "query" in ask_tool.inputSchema["properties"]
    assert "query" in ask_tool.inputSchema["required"]


@pytest.mark.asyncio
async def test_paperqa_add_paper_tool_schema():
    """Test that paperqa_add_paper tool has correct schema."""
    tools = await list_tools()
    add_tool = next(tool for tool in tools if tool.name == "paperqa_add_paper")

    assert add_tool is not None
    assert "path" in add_tool.inputSchema["properties"]
    assert "path" in add_tool.inputSchema["required"]


@pytest.mark.asyncio
async def test_paperqa_search_tool_schema():
    """Test that paperqa_search tool has correct schema."""
    tools = await list_tools()
    search_tool = next(tool for tool in tools if tool.name == "paperqa_search")

    assert search_tool is not None
    assert "query" in search_tool.inputSchema["properties"]
    assert "query" in search_tool.inputSchema["required"]
