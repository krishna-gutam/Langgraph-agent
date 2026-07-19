# Project Context: Multi-Agent Orchestration Framework

This project is a comprehensive multi-agent orchestration framework built with LangGraph, LangChain, and Streamlit. It provides a robust environment for managing agent workflows, conversation persistence, and project workspaces.

## Key Features
- **Agent Swarm:** Includes built-in agents (Supervisor, Planner, Coder, Teacher, Architect) with dynamic creation and management capabilities.
- **Dynamic Agent Management:** The Architect agent can create, list, and delete agents at runtime.
- **Persistence:** Uses SQLite to maintain conversation state and thread history across sessions.
- **Streamlit UI:** A feature-rich interface for:
    - Managing conversation threads and history.
    - Configuring LLM settings (API keys, model IDs, temperature).
    - Inspecting logs and message history.
    - Editing and managing files within the project workspace.
- **Tool Orchestration:** Implements a secure agent-tool interaction model with human-in-the-loop controls.

## Core Components
- `main.py`: The central entry point containing the graph construction, agent definitions, UI layout, and orchestration logic.
- **Agent Nodes:** Logic for LLM invocation and tool execution.
- **Memory Management:** SQLite-based persistence for thread state.
- **Workspace Management:** Utilities for file system interaction and project directory tracking.
