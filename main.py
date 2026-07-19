import os
import re
import sys
import subprocess
import json
import uuid
import sqlite3
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from streamlit_ace import st_ace

from langchain_core.tools import ToolException, tool, StructuredTool

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph and LangChain imports
from typing import Annotated, Sequence, TypedDict, Optional, List
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
    RemoveMessage,
    AIMessageChunk,
    BaseMessage,
)

from tools.registry import get_tools


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Load environment variables
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")

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


def estimate_tokens(messages):
    """
    Estimate token count for a list of messages.
    Prioritizes usage_metadata from AIMessages if available.
    Fallback: character count / 3.675.
    """
    if not messages:
        return 0

    # Try to get tokens from the last AIMessage's usage_metadata
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None):
            return m.usage_metadata.get("total_tokens", 0)

    # Fallback: character count / 3.675
    total_chars = sum(len(str(m.content)) for m in messages)
    return int(total_chars // 3.675)


def get_llm():
    # Check session state first, then environment
    api_key = st.session_state.get("google_api_key_input") or os.getenv(
        "GOOGLE_API_KEY"
    )

    if not api_key:
        return None  # Return None instead of crashing the app

    model_id = st.session_state.get("model_id", MODEL_ID)
    temperature = st.session_state.get("temperature", 0.0)

    return ChatGoogleGenerativeAI(
        model=model_id, temperature=temperature, google_api_key=api_key
    )


# --- CENTRALIZED WORKSPACE MANAGEMENT (moved up: the agent registry lives here too) ---


def get_central_workspace_path():
    # Root directory for all agent data
    root = Path(os.getenv("AGENT_DATA_ROOT")) / "central_workspace_data"
    print(root)
    # Use current directory name as the unique identifier
    dir_name = os.path.basename(os.getcwd())
    path = root / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- DYNAMIC AGENT REGISTRY ---
# Custom agents created at runtime (via the 'architect' agent) are persisted here.
# Because the whole Streamlit script re-executes on every interaction, reloading this
# file at the top of each run is what makes newly created agents "live" without a restart.

def get_agent_registry_path():
    return get_central_workspace_path() / "agents_registry.json"


def load_custom_agents():
    path = get_agent_registry_path()
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_custom_agents(agents):
    path = get_agent_registry_path()
    try:
        with open(path, "w") as f:
            json.dump(agents, f, indent=2)
    except Exception:
        pass


# --- CONFIGURATION & PROMPTS ---

SUPERVISOR_PROMPT = """You are the Supervisor agent. you are part of a agent swarm with shared message state.
Your job is to interact with the user, understand their request, and delegate tasks.
- If the user asks a general question, answer it directly.
- If the request requires complex architecture or step-by-step planning, use the 'transfer_to_planner' tool.
- If the request is a straightforward code edit, use the 'transfer_to_coder' tool.
- If the user wants to learn something, use the 'transfer_to_teacher' tool.
- If the user wants to add, design, or configure a NEW specialized agent for the swarm, use the 'transfer_to_architect' tool.
- When the workers transfer control back to you, review their work and present the final result to the user.
"""

PLANNER_PROMPT = """You are an Planner agent. you are part of a agent swarm with shared message state.
Your job is toInterview me relentlessly about every aspect of the plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.
Ask the questions one at a time. And create a detailed step-by-step plan.
Do NOT write final code. Once your plan is complete, you MUST use the 'transfer_to_supervisor' tool to hand the plan back, or 'transfer_to_coder' if immediate execution is needed.
"""

CODER_PROMPT = """You are an Coder agent. you are part of a agent swarm with shared message state.
Your job is to execute plans and write code using your tools.
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
- Once you have successfully completed the coding task and verified it, you MUST use the 'transfer_to_supervisor' tool to report back.
"""

TEACHER_PROMPT = """
You are a Teacher Agent in a multi-agent swarm with shared state.

Objective:
- Resolve the user's question by teaching, not merely answering.

Rules:
- Be direct, precise, and information-dense.
- Prioritize understanding over verbosity.
- Break complex topics into clear steps.
- Ask questions only when required to make progress.
- Do not use filler, motivational language, or unnecessary acknowledgements.
- Do not repeat information already present in the shared state.
- Stay focused on the user's current learning objective.

Completion:
- When the user has received a complete explanation or indicates satisfaction, call `transfer_to_supervisor`.
"""

ARCHITECT_PROMPT = """You are the Architect agent. you are part of an agent swarm with shared message state.
Your job is to design and instantiate NEW specialized agents when the existing swarm
(Supervisor, Planner, Coder, Teacher) cannot adequately cover a class of task the user needs.

Process:
- Clarify with the user what the new agent should own, which tools it needs, and which existing
  agents should be able to hand work off to it (reachable_from). Ask one question at a time.
- Use 'list_agents' first to see what already exists so you don't create an overlapping agent.
- Write a complete system_prompt for the new agent, in the same style as the other agents' prompts:
  state its role and rules, and instruct it to call 'transfer_to_supervisor' when its task is done.
- Only grant tool_names that are actually needed for the agent's stated job (least privilege).
- Call 'create_agent' to persist the definition. The agent becomes active starting next turn.
- Use 'delete_agent' if the user asks to remove a custom agent that is no longer needed.
- Once done (or if you decide not to create an agent), call 'transfer_to_supervisor' to hand back control.
"""

# --- BUILT-IN AGENT DEFINITIONS ---
# name -> {prompt, tools: [base tool names], transfers_to: [agent names it can hand off to]}
BUILT_IN_AGENTS = {
    "supervisor": {
        "prompt": SUPERVISOR_PROMPT,
        "tools": ["web_search", "read_file", "run_powershell", "web_fetch", "fetch_code"],
        "transfers_to": ["planner", "coder", "teacher", "architect","get_youtube_transcript"],
    },
    "planner": {
        "prompt": PLANNER_PROMPT,
        "tools": ["read_file", "web_search", "web_fetch", "fetch_code", "write_file", "apply_patch"],
        "transfers_to": ["supervisor", "coder"],
    },
    "coder": {
        "prompt": CODER_PROMPT,
        "tools": ["write_file", "fetch_code", "read_file", "run_powershell", "apply_patch"],
        "transfers_to": ["supervisor", "planner"],
    },
    "teacher": {
        "prompt": TEACHER_PROMPT,
        "tools": ["web_search", "read_file", "run_powershell", "web_fetch", "write_file"],
        "transfers_to": ["supervisor"],
    },
    "architect": {
        "prompt": ARCHITECT_PROMPT,
        "tools": ["create_agent", "list_agents", "delete_agent"],
        "transfers_to": ["supervisor"],
    },
}

RESERVED_NAMES = {"tools", "start", "end"}

# --- 1. BASE TOOLS ---
all_base_tools = get_tools()
base_tools_map = {t.name: t for t in all_base_tools}
RAW_TOOL_NAMES = set(base_tools_map.keys())  # tools that exist independent of the agent swarm


# --- 2. META TOOLS: the agent-creation toolkit used by the 'architect' agent ---

@tool
def create_agent(
    name: str,
    system_prompt: str,
    tool_names: List[str],
    reachable_from: Optional[List[str]] = None,
) -> str:
    """Create a brand new specialized agent and add it to the multi-agent swarm.

    Args:
        name: unique lowercase snake_case id for the agent, e.g. 'researcher'. Cannot reuse
              an existing agent name.
        system_prompt: the full system prompt describing this agent's role, rules, and
              completion criteria. It should instruct the agent to call 'transfer_to_supervisor'
              when its task is done.
        tool_names: subset of the available base tools this agent is allowed to use. Grant the
              minimum needed for its job.
        reachable_from: names of existing agents that should be able to hand off to this new
              agent (defaults to ['supervisor'] if omitted).
    """
    reachable_from = reachable_from or ["supervisor"]
    custom = load_custom_agents()
    existing_names = set(BUILT_IN_AGENTS.keys()) | {a["name"] for a in custom}

    clean_name = (name or "").strip().lower()
    if not re.match(r"^[a-z][a-z0-9_]*$", clean_name):
        return (
            f"Error: '{name}' is not a valid agent name. Use lowercase letters, digits, "
            f"and underscores, starting with a letter."
        )
    if clean_name in existing_names or clean_name in RESERVED_NAMES:
        return f"Error: agent name '{clean_name}' is already in use."
    if not system_prompt or not system_prompt.strip():
        return "Error: system_prompt cannot be empty."

    bad_tools = [t for t in (tool_names or []) if t not in RAW_TOOL_NAMES]
    if bad_tools:
        return f"Error: unknown tool(s) {bad_tools}. Available tools: {sorted(RAW_TOOL_NAMES)}"

    bad_sources = [s for s in reachable_from if s not in existing_names]
    if bad_sources:
        return (
            f"Error: unknown source agent(s) {bad_sources} in reachable_from. "
            f"Existing agents: {sorted(existing_names)}"
        )

    custom.append(
        {
            "name": clean_name,
            "prompt": system_prompt,
            "tools": list(tool_names or []),
            "transfers_to": ["supervisor"],
            "reachable_from": reachable_from,
        }
    )
    save_custom_agents(custom)
    return (
        f"Agent '{clean_name}' created with tools {tool_names}, reachable from {reachable_from}. "
        f"It will be live in the swarm starting next turn."
    )


@tool
def list_agents() -> str:
    """List all agents currently in the swarm, built-in and custom, with their tools."""
    custom = load_custom_agents()
    lines = []
    for n, d in BUILT_IN_AGENTS.items():
        lines.append(f"[built-in] {n}: tools={d['tools']}")
    for a in custom:
        lines.append(
            f"[custom] {a['name']}: tools={a.get('tools', [])}, "
            f"reachable_from={a.get('reachable_from', ['supervisor'])}"
        )
    return "\n".join(lines) if lines else "No agents found."


@tool
def delete_agent(name: str) -> str:
    """Delete a previously created custom agent by name. Built-in agents cannot be deleted."""
    clean_name = (name or "").strip().lower()
    if clean_name in BUILT_IN_AGENTS:
        return f"Error: '{clean_name}' is a built-in agent and cannot be deleted."
    custom = load_custom_agents()
    new_custom = [a for a in custom if a["name"] != clean_name]
    if len(new_custom) == len(custom):
        return f"Error: no custom agent named '{clean_name}' found."
    save_custom_agents(new_custom)
    return f"Agent '{clean_name}' deleted. It will be removed from the swarm starting next turn."


META_TOOLS = [create_agent, list_agents, delete_agent]
for t in META_TOOLS:
    base_tools_map[t.name] = t


# --- 3. ASSEMBLE THE FULL AGENT DEFINITION SET (built-in + custom, dynamically loaded) ---

AGENT_DEFS = {name: dict(d, tools=list(d["tools"]), transfers_to=list(d["transfers_to"])) for name, d in BUILT_IN_AGENTS.items()}

for a in load_custom_agents():
    AGENT_DEFS[a["name"]] = {
        "prompt": a["prompt"],
        "tools": list(a.get("tools", [])),
        "transfers_to": list(a.get("transfers_to", ["supervisor"])),
    }

# Wire reachability: give each listed source agent a way to hand off to each custom agent
for a in load_custom_agents():
    for source in a.get("reachable_from", ["supervisor"]):
        if source in AGENT_DEFS and a["name"] not in AGENT_DEFS[source]["transfers_to"]:
            AGENT_DEFS[source]["transfers_to"].append(a["name"])


# --- 4. AUTO-GENERATE HANDOFF TOOLS FOR EVERY AGENT ---

def make_transfer_tool(target_name: str):
    def _transfer(who_are_you: str, who_you_want_to_transfer_to: str) -> str:
        return f"Transferred from {who_are_you} to {target_name}."

    return StructuredTool.from_function(
        func=_transfer,
        name=f"transfer_to_{target_name}",
        description=f"Transfer the conversation to the '{target_name}' agent.",
    )


for agent_name in AGENT_DEFS:
    t = make_transfer_tool(agent_name)
    base_tools_map[t.name] = t


# --- 5. RESOLVE EACH AGENT'S FULL TOOL LIST (base tools + its transfer tools) ---

AGENT_TOOLS = {}
for name, d in AGENT_DEFS.items():
    tool_objs = [base_tools_map[tn] for tn in d["tools"] if tn in base_tools_map]
    transfer_objs = [
        base_tools_map[f"transfer_to_{target}"]
        for target in d["transfers_to"]
        if f"transfer_to_{target}" in base_tools_map
    ]
    AGENT_TOOLS[name] = tool_objs + transfer_objs

ALLOWED_TOOLS_BY_AGENT = {name: [t.name for t in tools] for name, tools in AGENT_TOOLS.items()}


# --- 6. GENERIC AGENT NODE FACTORY ---

def make_agent_node(name: str, prompt: str, tools: list):
    def node(state: AgentState):
        llm = get_llm().bind_tools(tools)
        response = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
        response.name = name
        return {"messages": [response]}

    return node


def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    results = []

    # Identify which agent requested the tool call
    caller_name = getattr(last_message, "name", "supervisor")

    for call in last_message.tool_calls:
        try:
            tool_name = call["name"]

            # --- STRICT AUTHORIZATION CHECK ---
            allowed_tools = ALLOWED_TOOLS_BY_AGENT.get(caller_name, [])

            if tool_name not in allowed_tools:
                raise PermissionError(
                    f"Agent '{caller_name}' is not authorized to use '{tool_name}'."
                    f"Please use only your designated tools: {allowed_tools} or transfer to the correct agent."
                )

            # Execution
            tool_obj = base_tools_map[tool_name]
            tool_result = tool_obj.invoke(call["args"])
            tool_result_str = str(tool_result).strip()

        except Exception as e:
            # If the authorization fails, the agent gets the error message instead of the tool output
            tool_result_str = f"Error executing tool: {str(e)}"

        results.append(
            ToolMessage(content=tool_result_str, tool_call_id=call["id"], name=tool_name)
        )

    return {"messages": results}


def route_start(state: AgentState) -> str:
    """Determine which agent should handle the new user message."""
    messages = state.get("messages", [])

    # Scan backward through the conversation to find the last AI that spoke
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            # Route to whichever agent generated this message, falling back to supervisor
            # if that agent no longer exists (e.g. name missing, or a custom agent was deleted).
            return msg.name if msg.name in AGENT_DEFS else "supervisor"

    # Default to the supervisor if this is a brand new conversation
    return "supervisor"


def agent_router(state: AgentState):
    """Determines what happens after an agent speaks."""
    messages = state["messages"]

    if not messages:
        return END

    last_message = messages[-1]

    # 1. If the agent called a tool, route to the tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 2. Otherwise the agent is talking to the user directly; yield control back.
    return END


def tool_router(state: AgentState):
    """Determines what happens after a tool is executed."""
    messages = state["messages"]

    if not messages:
        return "supervisor"
    last_message = messages[-1]  # This is the ToolMessage

    # 1. Did we just execute a handoff tool? If so, route to the new agent (with a safe fallback
    #    to supervisor in case a custom agent was deleted since this transfer was issued).
    if last_message.name and last_message.name.startswith("transfer_to_"):
        target = last_message.name[len("transfer_to_"):]
        return target if target in AGENT_DEFS else "supervisor"

    # 2. If it was a normal tool (like read_file), route back to whoever called it.
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.name if msg.name in AGENT_DEFS else "supervisor"

    return "supervisor"  # Fallback


# --- 7. BUILD THE GRAPH FROM THE (POSSIBLY CUSTOM-EXTENDED) AGENT_DEFS ---

workflow = StateGraph(AgentState)

workflow.add_node("tools", tool_node)
for name, d in AGENT_DEFS.items():
    workflow.add_node(name, make_agent_node(name, d["prompt"], AGENT_TOOLS[name]))

workflow.add_conditional_edges(START, route_start)
for name in AGENT_DEFS:
    workflow.add_conditional_edges(name, agent_router)
workflow.add_conditional_edges("tools", tool_router)


# We pass `cwd` as an argument so Streamlit caches a SEPARATE database connection per workspace folder
@st.cache_resource
def get_memory(cwd: str):
    workspace_data_path = get_central_workspace_path()
    thread_file = workspace_data_path / ".current_thread.txt"
    if not thread_file.exists():
        with open(thread_file, "w") as f:
            f.write("1")
    db_path = workspace_data_path / "agent_workspace_memory.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


# Re-fetch memory and compile graph based on the CURRENT working directory
memory = get_memory(os.getcwd())
app = workflow.compile(checkpointer=memory, interrupt_before=["tools"])

# --- PROJECT MANAGEMENT ---

HISTORY_FILE = Path(os.getenv("AGENT_DATA_ROOT")) / "history_file" / ".gemini_agent_projects.json"
print("HISTORY_FILE")
print(HISTORY_FILE)


def load_recent_projects():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_recent_project(path):
    projects = load_recent_projects()
    if path in projects:
        projects.remove(path)
    projects.insert(0, path)
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(projects, f)
    except Exception:
        pass


# --- THREAD PERSISTENCE VIA TEXT FILE ---


def get_all_thread_ids():
    """Queries the SQLite database for all unique thread IDs."""
    db_path = get_central_workspace_path() / "agent_workspace_memory.db"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        return threads
    except Exception:
        return []


def get_saved_thread_id():
    state_file = get_central_workspace_path() / ".current_thread.txt"
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return f.read().strip()
        except Exception:
            pass
    return None


def save_thread_id(tid):
    state_file = get_central_workspace_path() / ".current_thread.txt"
    try:
        with open(state_file, "w") as f:
            f.write(tid)
    except Exception:
        pass


def switch_thread(tid):
    save_thread_id(tid)
    st.session_state.thread_id = tid
    # LangGraph automatically initializes empty states for new thread_ids when the graph is invoked.
    st.rerun()


def switch_workspace_environment():
    """Helper to update session state with the thread ID of the newly activated workspace."""
    saved_id = get_saved_thread_id()
    if saved_id:
        st.session_state.thread_id = saved_id
    else:
        new_id = str(uuid.uuid4())
        save_thread_id(new_id)
        st.session_state.thread_id = new_id
    # Rerun immediately to refresh the UI and DB connections
    st.rerun()


# --- STREAMLIT UI ---

st.set_page_config(page_title="----------", layout="wide")

# Ensure a thread ID is loaded or created on every refresh
if "thread_id" not in st.session_state:
    saved_id = get_saved_thread_id()
    if saved_id:
        st.session_state.thread_id = saved_id
    else:
        new_id = str(uuid.uuid4())
        save_thread_id(new_id)
        st.session_state.thread_id = new_id

thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- SIDEBAR: Workspace & Controls ---
with st.sidebar:
    with st.container(border=True):
        # Conversation Selection
        all_threads = get_all_thread_ids()

        if st.session_state.thread_id not in all_threads:
            all_threads.insert(0, st.session_state.thread_id)

        current_idx = all_threads.index(st.session_state.thread_id)

        selected_thread = st.selectbox(
            "Switch Conversation",
            all_threads,
            index=current_idx,
            format_func=lambda x: x[:8],
        )

        if selected_thread != st.session_state.thread_id:
            switch_thread(selected_thread)

        if st.button("➕ New Conversation", key="new_conv_btn"):
            st.session_state.show_new_thread_input = True

        if st.session_state.get("show_new_thread_input", False):
            custom_id = st.text_input("Enter Thread ID:", key="custom_thread_id_input")
            col1, col2 = st.columns(2)
            if col1.button("Create"):
                new_id = custom_id if custom_id else str(uuid.uuid4())
                st.session_state.show_new_thread_input = False
                switch_thread(new_id)
            if col2.button("Cancel"):
                st.session_state.show_new_thread_input = False
                st.rerun()

    with st.container(border=True):
        recent_projects = load_recent_projects()
        project_opts = ["Current Directory"] + recent_projects
        selected_proj = st.selectbox("Switch Workspace", project_opts)

        if st.button("➕ Create New Project", key="new_proj_btn"):
            st.session_state.show_new_project_input = True

        if st.session_state.get("show_new_project_input", False):
            new_path = st.text_input(
                "Enter absolute path for new project:", key="new_proj_path_input"
            )
            col1, col2 = st.columns(2)
            if col1.button("Create Project"):
                if new_path and os.path.isdir(os.path.dirname(new_path)):
                    os.makedirs(new_path, exist_ok=True)
                    st.session_state.show_new_project_input = False
                    os.chdir(new_path)
                    save_recent_project(new_path)
                    switch_workspace_environment()
                    st.rerun()
                else:
                    st.error("Invalid path provided.")
            if col2.button("Cancel Project"):
                st.session_state.show_new_project_input = False
                st.rerun()

        if selected_proj != "Current Directory":
            if os.path.exists(selected_proj) and os.getcwd() != selected_proj:
                os.chdir(selected_proj)
                save_recent_project(selected_proj)
                switch_workspace_environment()

        st.caption(f"**Active:** `{os.getcwd()}`")

    with st.container(border=True):
        try:
            state_snapshot = app.get_state(thread_config)
            messages_for_token = state_snapshot.values.get("messages", [])
            token_count = estimate_tokens(messages_for_token)
            st.metric(label="Conversation Tokens", value=token_count)
        except Exception:
            st.metric(label="Conversation Tokens", value="N/A")

        auto_approve = st.checkbox("Auto-Approve Tools", value=False)

        if st.button("⏮️ Undo First Turn", use_container_width=True):
            current_state = app.get_state(thread_config)
            state_messages = current_state.values.get("messages", [])
            human_indices = [
                i for i, m in enumerate(state_messages) if isinstance(m, HumanMessage)
            ]
            if human_indices:
                first_human_idx = human_indices[0]
                if len(human_indices) > 1:
                    next_human_idx = human_indices[1]
                    target_messages = state_messages[first_human_idx:next_human_idx]
                else:
                    target_messages = state_messages[first_human_idx:]
                messages_to_remove = [
                    RemoveMessage(id=msg.id) for msg in target_messages if msg.id
                ]
                if messages_to_remove:
                    app.update_state(thread_config, {"messages": messages_to_remove})
                    st.rerun()

        if st.button("↩️ Undo Last Turn", use_container_width=True):
            current_state = app.get_state(thread_config)
            state_messages = current_state.values.get("messages", [])
            human_indices = [
                i for i, m in enumerate(state_messages) if isinstance(m, HumanMessage)
            ]
            if human_indices:
                target_idx = human_indices[-1]
                messages_to_remove = [
                    RemoveMessage(id=msg.id)
                    for msg in state_messages[target_idx:]
                    if msg.id
                ]
                if messages_to_remove:
                    app.update_state(thread_config, {"messages": messages_to_remove})
                    st.rerun()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            current_state = app.get_state(thread_config)
            state_messages = current_state.values.get("messages", [])
            human_indices = [
                i for i, m in enumerate(state_messages) if isinstance(m, HumanMessage)
            ]
            if human_indices:
                target_idx = human_indices[0]
                messages_to_remove = [
                    RemoveMessage(id=msg.id)
                    for msg in state_messages[target_idx:]
                    if msg.id
                ]
                if messages_to_remove:
                    app.update_state(thread_config, {"messages": messages_to_remove})
            st.rerun()

    with st.container(border=True):
        st.subheader("🧩 Custom Agents")
        custom_agents_ui = load_custom_agents()
        if not custom_agents_ui:
            st.caption("None yet — ask the Architect agent to create one.")
        for a in custom_agents_ui:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                with st.expander(a["name"]):
                    st.caption(f"Tools: {a.get('tools', [])}")
                    st.caption(f"Reachable from: {a.get('reachable_from', ['supervisor'])}")
                    st.text(a.get("prompt", ""))
            with col2:
                if st.button("🗑️", key=f"del_agent_{a['name']}"):
                    remaining = [x for x in custom_agents_ui if x["name"] != a["name"]]
                    save_custom_agents(remaining)
                    st.rerun()

    with st.container(border=True):
        notes_file = get_central_workspace_path() / "notes.md"
        if notes_file.exists():
            with open(notes_file, "r") as f:
                current_notes = f.read()
        else:
            current_notes = ""

        sidebar_notes = st.text_area(
            "Quick Notes:", value=current_notes, height=200, key="sidebar_notes"
        )
        if st.button("Save Quick Notes"):
            with open(notes_file, "w") as f:
                f.write(sidebar_notes)
            st.success("Notes saved!")

# --- MAIN CHAT INTERFACE ---

current_state = app.get_state(thread_config)
graph_messages = current_state.values.get("messages", [])
snapshot = app.get_state(thread_config)

(
    tab_chat,
    tab_edit,
    tab_logs,
    tab_history,
    tab_settings,
) = st.tabs(
    [
        "💬 Chat Interface",
        "📝 Editor",
        "📜 Message Logs",
        "🕒 Manage History",
        "⚙️ Settings",
    ]
)

with tab_history:
    st.subheader("Conversation Threads")
    thread_ids = get_all_thread_ids()

    for tid in thread_ids:
        col1, col2, col3, col4 = st.columns([0.85, 0.05, 0.05, 0.05])

        config = {"configurable": {"thread_id": tid}}
        state = app.get_state(config)
        messages = state.values.get("messages", [])

        last_human = "No human message"
        last_ai = "No AI message"

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and last_human == "No human message":
                last_human = sanitize_content(msg.content)
            if isinstance(msg, AIMessage) and last_ai == "No AI message":
                last_ai = sanitize_content(msg.content)

        with col1:
            with st.expander(f"Thread: {tid}"):
                st.write(f"**Last Human:** {last_human}")
                st.write(f"**Last AI:** {last_ai}")

        with col2:
            if st.button("D", key=f"del_thread_{tid}"):
                db_path = get_central_workspace_path() / "agent_workspace_memory.db"
                conn = sqlite3.connect(db_path, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (tid,))
                cursor.execute("DELETE FROM writes WHERE thread_id = ?", (tid,))
                conn.commit()
                conn.close()
                st.rerun()
        with col3:
            new_id = st.text_input(
                "New ID", key=f"rename_input_{tid}", label_visibility="collapsed"
            )
        with col4:
            if st.button("R", key=f"rename_btn_{tid}"):
                if new_id and new_id != tid:
                    db_path = get_central_workspace_path() / "agent_workspace_memory.db"
                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE checkpoints SET thread_id = ? WHERE thread_id = ?",
                        (new_id, tid),
                    )
                    cursor.execute(
                        "UPDATE writes SET thread_id = ? WHERE thread_id = ?",
                        (new_id, tid),
                    )
                    conn.commit()
                    conn.close()
                    st.rerun()


with tab_settings:
    st.subheader("🧠 Context Management")

    st.text_input(
        "Google API Key",
        type="password",
        key="google_api_key_input",
        help="Enter your Google API Key here if not set in .env",
    )

    st.text_input("Model ID", value=MODEL_ID, key="model_id")

    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        key="temperature",
    )

    try:
        state_snapshot = app.get_state(thread_config)
    except Exception:
        pass


with tab_logs:
    st.subheader("Full Message History")
    for i, msg in enumerate(graph_messages):
        msg_type = type(msg).__name__
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            label = f"{i}: {msg_type} | {msg.name}"
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_names = [call["name"] for call in msg.tool_calls]
                label += f" ({', '.join(tool_names)})"

            with st.expander(label):
                st.write(msg)
        with col2:
            if st.button("🗑️", key=f"del_{i}"):
                if msg.id:
                    app.update_state(
                        thread_config, {"messages": [RemoveMessage(id=msg.id)]}
                    )
                    st.rerun()


def ui_read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def ui_write_file(path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


with tab_edit:
    st.subheader("File Editor")

    all_files = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d
            for d in dirs
            if d
            not in [
                ".venv",
                ".git",
                "__pycache__",
                "node_modules",
                "workspace_data",
                "papers",
            ]
        ]
        for f in files:
            all_files.append(os.path.join(root, f))

    edit_path = st.selectbox("Select a file to edit:", all_files)

    if st.button("Load File"):
        if os.path.exists(edit_path):
            st.session_state.edit_content = ui_read_file(edit_path)
            st.session_state.editor_key = str(uuid.uuid4())
        else:
            st.error("File not found.")

    if "edit_content" in st.session_state:
        if "editor_key" not in st.session_state:
            st.session_state.editor_key = "ace_editor_initial"

        new_content = st_ace(
            value=st.session_state.edit_content,
            language="python",
            theme="monokai",
            key=st.session_state.editor_key,
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("💾 Save Changes", use_container_width=True):
                result = ui_write_file(edit_path, new_content)
                if "Error" in result:
                    st.error(result)
                else:
                    st.success(result)
                    st.session_state.edit_content = new_content

        with col2:
            if st.button("🔄 Reset Unsaved Changes", use_container_width=True):
                if os.path.exists(edit_path):
                    st.session_state.edit_content = ui_read_file(edit_path)
                    st.session_state.editor_key = str(uuid.uuid4())
                    st.rerun()

with tab_chat:
    for msg in graph_messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(sanitize_content(msg.content))
        elif isinstance(msg, AIMessage):
            content = sanitize_content(msg.content)
            if content.strip():
                with st.chat_message("assistant"):
                    st.markdown(content)
        elif isinstance(msg, ToolMessage):
            with st.chat_message("tool", avatar="🔧"):
                tool_name = msg.name if msg.name else "Tool"
                with st.expander(f"Result from {tool_name}", expanded=False):
                    st.code(msg.content, language="markdown")

    if snapshot.next == ("tools",):
        last_msg = snapshot.values["messages"][-1]

        with st.chat_message("assistant"):
            st.warning(
                "⚠️ **The Agent has requested to execute the following tool(s):**"
            )
            for call in last_msg.tool_calls:
                tool_name = call.get("name", "Unknown Tool")
                args = call.get("args", {})

                with st.expander(f"Tool Call: {tool_name}", expanded=True):
                    for key, value in args.items():
                        if key == "justification":
                            continue
                        label = key
                        lang = "python"
                        st.markdown(f"**{label}:**")
                        st.code(str(value), language=lang)

                    if "justification" in args:
                        st.markdown("**Justification:**")
                        st.code(args["justification"], language="python")

            if auto_approve:
                st.info("Auto-approving because the checkbox is ticked...")
                app.invoke(None, config=thread_config)
                st.rerun()
            else:
                col1, col2, col3 = st.columns([0.4, 0.3, 0.3])

                if col1.button("✅ Approve Action"):
                    with st.status("Executing tools...", expanded=True) as status:
                        app.invoke(None, config=thread_config)
                        status.update(
                            label="Action complete!", state="complete", expanded=False
                        )
                    st.rerun()

                if col2.button("❌ Deny Action"):
                    state_messages = snapshot.values.get("messages", [])
                    human_indices = [
                        i
                        for i, m in enumerate(state_messages)
                        if isinstance(m, HumanMessage)
                    ]
                    if human_indices:
                        target_idx = human_indices[-1]
                        removals = [
                            RemoveMessage(id=msg.id)
                            for msg in state_messages[target_idx:]
                            if msg.id
                        ]
                        app.update_state(thread_config, {"messages": removals})
                    st.error("Action denied and forgotten from agent memory.")
                    st.rerun()

                with col3:
                    with st.popover("💬 Provide Feedback"):
                        feedback_text = st.text_area("Tell the agent what to change:")

                        if st.button("Submit Feedback"):
                            if feedback_text.strip():
                                tool_messages = [
                                    ToolMessage(
                                        tool_call_id=call["id"],
                                        name=call["name"],
                                        content=f"User interrupted: {feedback_text}",
                                    )
                                    for call in last_msg.tool_calls
                                ]

                                app.update_state(
                                    thread_config,
                                    {"messages": tool_messages},
                                    as_node="tools",
                                )

                                st.session_state.resume_agent = True
                                st.rerun()
                            else:
                                st.warning("Please enter some feedback.")

    if st.session_state.get("resume_agent", False):
        st.session_state.resume_agent = False

        with st.chat_message("assistant"):
            with st.spinner("Processing your feedback..."):
                app.invoke(None, config=thread_config)

        st.rerun()

current_llm = get_llm()

if current_llm is None:
    with tab_chat:
        st.warning(
            "⚠️ **API Key Required:** Please navigate to the **⚙️ Settings** tab and enter your Google API Key to start coding."
        )

    st.chat_input("Enter API key in settings to chat...", disabled=True)

else:
    if prompt := st.chat_input("What would you like to do?"):
        safe_prompt = sanitize_content(prompt)

        with tab_chat:
            with st.chat_message("user"):
                st.markdown(safe_prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk, metadata in app.stream(
                    {"messages": [HumanMessage(content=safe_prompt)]},
                    config=thread_config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        safe_chunk_text = sanitize_content(chunk.content)
                        full_response += safe_chunk_text
                        message_placeholder.markdown(full_response + "▌")

                if full_response:
                    message_placeholder.markdown(full_response)

                new_snapshot = app.get_state(thread_config)
                if new_snapshot.next == ("tools",):
                    st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")