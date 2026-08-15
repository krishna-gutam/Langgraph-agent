import os
import json
import uuid
import sqlite3
from pathlib import Path
import streamlit as st
from streamlit_ace import st_ace

from agent_core import (
    create_graph,
    get_llm,
    sanitize_content,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    RemoveMessage,
    AIMessageChunk,
)



# We pass `cwd` as an argument so Streamlit caches a SEPARATE database connection per workspace folder
# --- CENTRALIZED WORKSPACE MANAGEMENT ---


def get_central_workspace_path():
    # Root directory for all agent data
    root = Path(os.getenv("AGENT_DATA_ROOT")) / "central_workspace_data"
    # Use current directory name as the unique identifier
    dir_name = os.path.basename(os.getcwd())
    path = root / dir_name # e.g., D:\Apps\agent\app\core\central_workspace_data\project_name
    path.mkdir(parents=True, exist_ok=True)
    return path


@st.cache_resource
def get_memory():
    workspace_data_path = get_central_workspace_path()
    thread_file = workspace_data_path / ".current_thread.txt"
    if not thread_file.exists():
        with open(thread_file, "w") as f:
            f.write("1")
    db_path = workspace_data_path / "agent_workspace_memory.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


# Re-fetch memory and compile graph based on the CURRENT working directory
memory = get_memory()
app = create_graph(checkpointer=memory)

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

HISTORY_FILE = Path(os.getenv("AGENT_DATA_ROOT")) / "history_file" / ".gemini_agent_projects.json"


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
        # st.subheader("💬 Conversations")
        all_threads = get_all_thread_ids()

        # FIX: Ensure the current session thread (even if brand new and not in DB yet) is in the list
        if st.session_state.thread_id not in all_threads:
            all_threads.insert(0, st.session_state.thread_id)

        current_idx = all_threads.index(st.session_state.thread_id)

        selected_thread = st.selectbox(
            "Switch Conversation",
            all_threads,
            index=current_idx,
            format_func=lambda x: x[:8],  # + "..." # Show truncated ID
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

        # st.subheader("📂 Project Workspace")
        recent_projects = load_recent_projects()

        # Project Selection
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

        # Workspace switching logic
        if selected_proj != "Current Directory":
            if os.path.exists(selected_proj) and os.getcwd() != selected_proj:
                os.chdir(selected_proj)
                save_recent_project(selected_proj)
                switch_workspace_environment()

        st.caption(f"**Active:** `{os.getcwd()}`")

    with st.container(border=True):
        # st.subheader("⚙️ Agent Controls")

        # Token count in sidebar
        try:
            state_snapshot = app.get_state(thread_config)
            messages_for_token = state_snapshot.values.get("messages", [])
            token_count = estimate_tokens(
                messages_for_token
            )
            st.metric(label="Conversation Tokens", value=token_count)
        except Exception:
            st.metric(label="Conversation Tokens", value="N/A")

        auto_approve = st.checkbox("Auto-Approve Tools", value=False)

        if st.button("⏮️ Undo First Turn", use_container_width=True):
            current_state = app.get_state(thread_config)
            state_messages = current_state.values.get("messages", [])

            # Identify all user inputs
            human_indices = [
                i for i, m in enumerate(state_messages) if isinstance(m, HumanMessage)
            ]

            if human_indices:
                first_human_idx = human_indices[0]

                # Determine where the first turn ends (right before the second human message)
                if len(human_indices) > 1:
                    next_human_idx = human_indices[1]
                    target_messages = state_messages[first_human_idx:next_human_idx]
                else:
                    # If there's only one turn, remove everything from that point onward
                    target_messages = state_messages[first_human_idx:]

                # Create RemoveMessage modifiers for that specific slice
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
            current_state = app.get_state(thread_config)  #
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
        # st.subheader("🗒️ Collect Your Thoughts")
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


# Add tabs for Chat View, Raw Log View, Editor, and Prompt Generator
(
    tab_chat,
    tab_edit,
    tab_logs,
    tab_history,
) = st.tabs(
    [
        "💬 Chat Interface",
        "📝 Editor",
        "📜 Message Logs",
        "🕒 Manage History",
    ]
)

with tab_history:
    st.subheader("Conversation Threads")
    thread_ids = get_all_thread_ids()

    for tid in thread_ids:
        col1, col2, col3, col4 = st.columns([0.85, 0.05, 0.05, 0.05])

        # Get last messages
        config = {"configurable": {"thread_id": tid}}
        state = app.get_state(config)
        messages = state.values.get("messages", [])

        last_human = "No human message"
        last_ai = "No AI message"

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and last_human == "No human message":
                last_human = sanitize_content(msg.content)  # [:50] + "..."
            if isinstance(msg, AIMessage) and last_ai == "No AI message":
                last_ai = sanitize_content(msg.content)  # [:50] + "..."

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

with tab_logs:
    st.subheader("Full Message History")
    for i, msg in enumerate(graph_messages):
        msg_type = type(msg).__name__
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            # Determine label: show tool name if it's a tool call
            label = f"{i}: {msg_type}"
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


# --- Helper functions for the UI ---
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

    # Get list of files for selection
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
            # Generate a unique key to force the Ace editor to remount
            st.session_state.editor_key = str(uuid.uuid4())
        else:
            st.error("File not found.")

    if "edit_content" in st.session_state:
        # Fallback in case the key isn't set yet
        if "editor_key" not in st.session_state:
            st.session_state.editor_key = "ace_editor_initial"

        new_content = st_ace(
            value=st.session_state.edit_content,
            language="python",
            theme="monokai",
            key=st.session_state.editor_key,  # Use the dynamic key here
        )

        # Use columns to put Save and Reset side-by-side cleanly
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("💾 Save Changes", use_container_width=True):
                result = ui_write_file(edit_path, new_content)
                if "Error" in result:
                    st.error(result)
                else:
                    st.success(result)
                    # Update the session state so the new baseline is the saved text
                    st.session_state.edit_content = new_content

        with col2:
            if st.button("🔄 Reset Unsaved Changes", use_container_width=True):
                if os.path.exists(edit_path):
                    # 1. Re-read the fresh file from disk using standard function
                    st.session_state.edit_content = ui_read_file(edit_path)
                    # 2. Change the key to force Streamlit to wipe the widget memory
                    st.session_state.editor_key = str(uuid.uuid4())
                    # 3. Rerun to instantly show the changes
                    st.rerun()

with tab_chat:
    # 1. RENDER CHAT HISTORY DIRECTLY FROM LANGGRAPH MEMORY
    for msg in graph_messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(sanitize_content(msg.content))
        elif isinstance(msg, AIMessage):
            content = sanitize_content(msg.content)
            # Only render if there is actual text (ignores silent tool-calls)
            if content.strip():
                with st.chat_message("assistant"):
                    st.markdown(content)
        # ---> NEW: CHECK FOR TOOL RESPONSES <---
        elif isinstance(msg, ToolMessage):
            # Using a custom avatar for tools makes them visually distinct
            with st.chat_message("tool", avatar="🔧"):
                tool_name = msg.name if msg.name else "Tool"
                with st.expander(f"Result from {tool_name}", expanded=False):
                    # st.text or st.code works best here to preserve formatting of raw outputs
                    st.code(msg.content, language="markdown")

    # 2. CHECK FOR PENDING TOOL APPROVALS
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
                    # 1. Iterate over all arguments dynamically
                    for key, value in args.items():
                        # Skip justification here so we can pin it to the bottom
                        if key == "justification":
                            continue
                        
                        # Format the dictionary key into a neat label (e.g., "old_code" -> "Old Code")
                        label = key
                        
                        # Optional: Catch specific keys to change syntax highlighting
                        lang = "python"
                        
                        st.markdown(f"**{label}:**")
                        # Convert value to string to prevent errors if the LLM passes a list/int
                        st.code(str(value), language=lang)
                        
                    # 2. Render justification last
                    if "justification" in args:
                        st.markdown("**Justification:**")
                        st.code(args["justification"], language="python")

            if auto_approve:
                st.info("Auto-approving because the checkbox is ticked...")
                app.invoke(None, config=thread_config)
                st.rerun()
            else:
                # Create 3 columns instead of 2
                col1, col2, col3 = st.columns([0.4, 0.3, 0.3])

                if col1.button("✅ Approve Action"):
                    with st.status("Executing tools...", expanded=True) as status:
                        # Resume graph execution
                        app.invoke(None, config=thread_config)
                        status.update(
                            label="Action complete!", state="complete", expanded=False
                        )
                    st.rerun()

                if col2.button("❌ Deny Action"):
                    # Handle Deny: Roll back state memory
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

                # ---> NEW: FEEDBACK BUTTON <---
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

                                # 1. Instantly update the state
                                app.update_state(
                                    thread_config,
                                    {"messages": tool_messages},
                                    as_node="tools",
                                )

                                # 2. Set a flag to trigger the LLM outside the popover
                                st.session_state.resume_agent = True

                                # 3. Rerun immediately
                                st.rerun()
                            else:
                                st.warning("Please enter some feedback.")

    # 3. HANDLE PENDING GRAPH EXECUTIONS
    if st.session_state.get("resume_agent", False):
        # Clear the flag immediately so it doesn't loop
        st.session_state.resume_agent = False

        # Show a spinner directly in the chat feed
        with st.chat_message("assistant"):
            with st.spinner("Processing your feedback..."):
                # Run the agent
                app.invoke(None, config=thread_config)

        # Rerun one last time to render the new Agent message
        st.rerun()
# 3. HANDLE NEW USER INPUT
# Check if we have a valid LLM setup before allowing input
current_llm = get_llm()

if current_llm is None:
    # Render a friendly warning in the main chat area
    with tab_chat:
        st.warning(
            "⚠️ **API Key Required:** Please navigate to the **⚙️ Settings** tab and enter your Google API Key to start coding."
        )

    # Render a disabled chat input so they know where to type later
    st.chat_input("Enter API key in settings to chat...", disabled=True)

else:
    # Normal execution if the key exists
    if prompt := st.chat_input("What would you like to do?"):
        safe_prompt = sanitize_content(prompt)

        with tab_chat:
            with st.chat_message("user"):
                st.markdown(safe_prompt)
            # ... (keep the rest of your existing stream/execution block here)

        with st.chat_message("assistant"):
            # Create an empty placeholder to dump the text into word-by-word
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Use stream_mode="messages" to get token-by-token streaming
                for chunk, metadata in app.stream(
                    {"messages": [HumanMessage(content=safe_prompt)]},
                    config=thread_config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        # FIX: Sanitize the chunk before concatenating!
                        safe_chunk_text = sanitize_content(chunk.content)

                        # Now it is guaranteed to be a string
                        full_response += safe_chunk_text

                        # Update the UI instantly with a "typing cursor" effect
                        message_placeholder.markdown(full_response + "▌")

                # Once the stream is fully complete, remove the cursor
                if full_response:
                    message_placeholder.markdown(full_response)

                # Check if the graph paused because it wants to use a tool
                new_snapshot = app.get_state(thread_config)
                if new_snapshot.next == ("tools",):
                    st.rerun()  # Immediately rerun to show the "Approve Tool" UI

            except Exception as e:
                st.error(f"Error: {str(e)}")
