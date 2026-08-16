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

def test_run_bash_in_registry():
    tools = get_tools()
    run_bash_tools = [t for t in tools if t.name == "run_bash"]
    assert len(run_bash_tools) == 1
    run_bash_tool = run_bash_tools[0]
    assert run_bash_tool.description is not None
    assert "bash" in run_bash_tool.description.lower()
