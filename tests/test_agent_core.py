import pytest
from agent_core import create_graph, AgentState
from langgraph.checkpoint.memory import MemorySaver

def test_create_graph_compilation():
    memory = MemorySaver()
    graph = create_graph(checkpointer=memory)
    assert graph is not None
    # Verify graph has expected nodes
    assert "agent" in graph.nodes
    assert "tools" in graph.nodes

def test_graph_initial_state():
    memory = MemorySaver()
    graph = create_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test_thread"}}
    
    state = graph.get_state(config)
    assert state is not None
    assert state.values.get("messages", []) == []
