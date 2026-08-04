import os
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from tools.registry import get_tools

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- HELPER ---
def sanitize_content(content) -> str:
    """Safely extracts a string from LangChain's potential list-based content blocks."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        extracted = []
        for block in content:
            if isinstance(block, str):
                extracted.append(block)
            elif isinstance(block, dict) and "text" in block:
                extracted.append(block["text"])
        return "\n".join(extracted)
    return str(content)

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    model_id = os.getenv("MODEL_ID")
    temperature = float(os.getenv("TEMPERATURE", 0.7))
    return ChatGoogleGenerativeAI(
        model=model_id, temperature=temperature, google_api_key=api_key
    )

# --- CONFIGURATION ---
DEFAULT_SYSTEM_PROMPT = """You are an expert coding assistant. You help users with coding tasks by reading files, executing commands, editing code, writing new files.

Permission rules:
- NEVER make changes automatically.
- Before any action that modifies files, creates files, overwrites files, installs packages, executes scripts, or runs shell/PowerShell commands, first explain:
  1. What you plan to do
  2. Why you want to do it
  3. Which files or commands will be affected

- Ask explicitly for user approval before proceeding.
- Wait for a clear confirmation such as "yes", "approve", or equivalent before executing any modifying action.
- If the user has already explicitly authorized a specific action in the current request, you may proceed only with that specific action.
- If multiple changes are required, summarize all planned changes and request approval once before making them.
- Be concise in your responses.
"""

tools = get_tools()
tools_map = {t.name: t for t in tools}

def agent_node(state: AgentState):
    messages = state["messages"]
    system_prompt = DEFAULT_SYSTEM_PROMPT
    sanitized_messages = [SystemMessage(content=system_prompt)]
    for m in messages:
        if isinstance(m, AIMessage) and not isinstance(m.content, str):
            m.content = sanitize_content(m.content)
        sanitized_messages.append(m)

    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(sanitized_messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        try:
            tool = tools_map[call["name"]]
            tool_result = tool.invoke(call["args"])
            tool_result_str = str(tool_result).strip()
        except Exception as e:
            tool_result_str = f"Error executing tool: {str(e)}"
        results.append(
            ToolMessage(
                content=tool_result_str, tool_call_id=call["id"], name=call["name"]
            )
        )
    return {"messages": results}

def should_continue(state: AgentState):
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def create_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])
