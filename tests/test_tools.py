import pytest
from tools.registry import get_tools

def test_get_tools():
    tools = get_tools()
    assert tools is not None
    assert len(tools) > 0
    
    tool_names = {t.name for t in tools}
    expected_tools = {"run_bash", "apply_patch", "web_search", "web_fetch"}
    
    for expected in expected_tools:
        assert expected in tool_names, f"Expected tool '{expected}' not found in registered tools: {tool_names}"
