# LangGraph Coding Agent

A powerful, modular, and autonomous AI coding assistant built with **LangGraph**, **Streamlit**, and **Google Gemini**. This agent is designed to handle complex coding tasks by planning, executing, and verifying code changes, with a unique "Overseer" autopilot mode.

## 🚀 Key Features

*   **Interactive Streamlit UI**: A full-featured web interface for chatting, file editing, managing conversation history, and monitoring agent activity.
*   **CLI Interface**: A lightweight, terminal-based interface for quick tasks.
*   **Overseer (Autopilot) Mode**: A meta-agent that acts as a proxy for the user. It plans, steers, and verifies the coding agent's work toward a specific goal, pausing only when human intervention is required.
*   **Sophisticated Tooling**:
    *   **`apply_patch`**: A robust, fuzzy-matching file editor that handles indentation and whitespace variations gracefully.
    *   **`run_bash`**: A secure, environment-aware bash executor that automatically manages a project-specific `.venv`.
    *   **`web_search` & `web_fetch`**: Integrated research capabilities using Tavily and BeautifulSoup.
*   **Persistent Memory**: Uses SQLite to maintain conversation state across sessions.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Create a `.env` file in the root directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    MODEL_ID=gemini-2.0-flash # or your preferred model
    TAVILY_API_KEY=your_tavily_api_key
    AGENT_DATA_ROOT=./data # Directory for workspace and history
    ```

## 📖 Usage

### Streamlit UI
Launch the web interface to start coding:
```bash
streamlit run main.py
```
*   **Chat**: Interact with the agent directly.
*   **Editor**: Load, edit, and save files directly from the browser.
*   **Overseer**: Set a goal, define a step limit, and let the agent work autonomously.

### CLI
For quick, terminal-based interactions:
```bash
python cli.py
```

## 🛠️ Available Tools

| Tool | Description |
| :--- | :--- |
| `apply_patch` | Edits files using an 8-tier fuzzy matching strategy to ensure accurate code replacement. |
| `run_bash` | Executes shell commands in a sandboxed, auto-managed virtual environment. |
| `web_search` | Performs advanced web searches via Tavily. |
| `web_fetch` | Scrapes and cleans text content from URLs. |

## 🎯 The Overseer Architecture

The **Overseer** is a meta-agent designed to steer the coding agent without being part of the core LangGraph state.

*   **How it works**: It observes the conversation transcript and generates the next `HumanMessage`. This keeps the agent's history clean and makes the feature modular.
*   **Decision Loop**: It evaluates the current state against the user's goal and returns a JSON decision:
    *   `continue`: Provides the next instruction.
    *   `goal_reached`: Signals completion.
    *   `needs_human`: Escalates to the user for critical decisions (e.g., destructive actions, ambiguous goals).
*   **Safety**: It is programmed to avoid repeating instructions and to verify results before declaring victory.


