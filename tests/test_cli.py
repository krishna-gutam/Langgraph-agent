import pytest
from unittest.mock import patch, MagicMock
import cli
from langchain_core.messages import AIMessageChunk, ToolMessage

@patch("cli.input", side_effect=["exit"])
@patch("cli.create_graph")
def test_run_cli_exit(mock_create_graph, mock_input):
    mock_graph = MagicMock()
    mock_create_graph.return_value = mock_graph
    
    cli.run_cli()
    
    mock_create_graph.assert_called_once()
    mock_input.assert_called()

@patch("cli.input", side_effect=["Hello", "exit"])
@patch("cli.create_graph")
def test_run_cli_message_and_exit(mock_create_graph, mock_input):
    mock_graph = MagicMock()
    chunk = AIMessageChunk(content="Hi there!")
    mock_graph.stream.return_value = [(chunk, {})]
    
    mock_snapshot = MagicMock()
    mock_snapshot.next = None
    mock_graph.get_state.return_value = mock_snapshot
    
    mock_create_graph.return_value = mock_graph
    
    cli.run_cli()
    
    mock_graph.stream.assert_called_once()

@patch("cli.input", side_effect=["Run tool", "y", "exit"])
@patch("cli.create_graph")
def test_run_cli_tool_approval(mock_create_graph, mock_input):
    mock_graph = MagicMock()
    chunk = AIMessageChunk(content="")
    mock_graph.stream.return_value = [(chunk, {})]
    
    mock_snapshot_tools = MagicMock()
    mock_snapshot_tools.next = ("tools",)
    
    mock_snapshot_done = MagicMock()
    mock_snapshot_done.next = None
    
    tool_msg = ToolMessage(content="Tool output success", tool_call_id="123", name="run_bash")
    mock_snapshot_done.values = {"messages": [tool_msg]}
    
    mock_graph.get_state.side_effect = [mock_snapshot_tools, mock_snapshot_done]
    mock_create_graph.return_value = mock_graph
    
    cli.run_cli()
    
    mock_graph.invoke.assert_called_once_with(None, config={"configurable": {"thread_id": "cli_session"}})

@patch("cli.input", side_effect=["Run tool", "n", "exit"])
@patch("cli.create_graph")
def test_run_cli_tool_denial(mock_create_graph, mock_input):
    mock_graph = MagicMock()
    chunk = AIMessageChunk(content="")
    mock_graph.stream.return_value = [(chunk, {})]
    
    mock_snapshot_tools = MagicMock()
    mock_snapshot_tools.next = ("tools",)
    
    mock_graph.get_state.return_value = mock_snapshot_tools
    mock_create_graph.return_value = mock_graph
    
    cli.run_cli()
    
    mock_graph.invoke.assert_not_called()
