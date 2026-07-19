# Coding Agent — Build Plan

## Vision

A Python-based autonomous coding agent powered by Claude, built incrementally.
Each phase is independently useful and production-deployable.

---

## Architecture Overview

```
Trigger (user / webhook / cron)
         ↓
    Guardrails IN          ← validate input before anything runs
         ↓
    Memory READ            ← inject relevant past context
         ↓
    Planner Agent          ← decompose task into subtasks
         ↓
  ┌──────┴──────┐
Coder         Tester       ← specialist sub-agents
Agent         Agent
  └──────┬──────┘
         ↓
    Tool Calls             ← built-in tools + MCP servers
         ↓
    Memory WRITE           ← store what worked
         ↓
    Guardrails OUT         ← validate output before acting
         ↓
    Observability          ← log full trace
         ↓
    Act in the world
```

---

## Core Design Principles

- **LLM + Tools** — Claude is the brain, tools are the hands
- **One tool per skill** — each tool does one thing well
- **MCP for external skills** — plug in any MCP server for new capabilities
- **Multiple triggers** — user chat, webhooks, cron jobs all use the same agent loop
- **Incremental complexity** — ship after every phase

---

## Phase 1 — Foundation: LLM + Tools + Agent Loop

**Goal:** Working agent that can write, read, and run code.

### Tools
| Tool | Skill |
|---|---|
| `write_file(path, content)` | Create or edit a code file |
| `read_file(path)` | Read a file from disk |
| `run_code(path)` | Execute file, return stdout + stderr |

### Agent Loop
```
User prompt
    ↓
Claude decides: respond OR call a tool
    ↓
If tool → execute → feed result back to Claude
    ↓
Claude loops until task is done
    ↓
Final response to user
```

### Deliverables
- [ ] `agent.py` — core loop
- [ ] `tools.py` — 3 built-in tools
- [ ] Basic CLI to run the agent

---

## Phase 2 — Event-Driven: Webhooks + Queue

**Goal:** Agent reacts to external events, not just user input.

### Triggers
| Source | Event | Agent Action |
|---|---|---|
| GitHub | PR opened | Review code, post comment |
| GitHub | Tests fail | Read error, auto-fix, push patch |
| Slack | Bug report | Triage, scaffold fix, open PR |
| Cron | Scheduled | Audit codebase, suggest refactors |

### Architecture
```
External Service
    ↓ (POST /webhook)
FastAPI endpoint       ← return 200 immediately
    ↓
Push to queue          ← Redis / Celery
    ↓
Background worker      ← runs the agent loop
    ↓
Act in the world
```

### Deliverables
- [ ] `server.py` — FastAPI webhook endpoints
- [ ] `worker.py` — background task runner
- [ ] Queue setup (Redis + Celery)
- [ ] GitHub webhook handler

---

## Phase 3 — Observability: Tracing + Logging

**Goal:** See exactly what the agent did, how long it took, what it cost.

### What to Log
| Event | Data Captured |
|---|---|
| LLM call | tokens in/out, duration, tool called |
| Tool call | tool name, args, result, duration |
| Agent run | run ID, trigger source, total cost, outcome |

### Implementation
```python
# Every run gets a unique ID
run_id = uuid.uuid4()

# Wrap every LLM call
traced_claude_call(messages, tools, run_id)

# Wrap every tool call
traced_tool_call(tool_name, args, result, run_id)
```

### Options
- **DIY** — log to JSON files or a database
- **Langfuse** — open-source, full trace UI (recommended)
- **LangSmith** — managed, pairs well with LangChain

### Deliverables
- [ ] `observability.py` — trace wrapper
- [ ] Structured JSON logs per run
- [ ] Langfuse integration (optional)

---

## Phase 4 — Guardrails: Safety Layer

**Goal:** Prevent dangerous or unintended actions.

### Input Guardrails (before any tool runs)
```python
def guardrail_in(user_input):
    blocked = ["rm -rf", "drop table", "os.system"]
    if any(b in user_input for b in blocked):
        raise ValueError("Blocked input")
```

### Output Guardrails (before each tool call executes)
```python
def guardrail_out(tool_name, tool_args):
    # Human approval for risky tools
    if tool_name in ["run_code", "git_push"]:
        confirm = input(f"Allow {tool_name}? y/n: ")
        if confirm != "y":
            raise PermissionError("Rejected")
    
    # Structural validation
    if tool_name == "write_file":
        assert tool_args["path"].endswith(".py")
```

### Rules to Implement
- [ ] Block dangerous shell patterns
- [ ] Whitelist allowed file paths
- [ ] Max tool calls per run (budget limit)
- [ ] Human-in-the-loop for destructive actions
- [ ] Output schema validation per tool

### Deliverables
- [ ] `guardrails.py` — input + output checks
- [ ] Config file for rules (easy to update without code changes)

---

## Phase 5 — Memory: RAG + Episodic

**Goal:** Agent learns from past runs and brings relevant context to new tasks.

### Memory Types
| Type | What | Storage |
|---|---|---|
| Short-term | Conversation history | In-memory (already have) |
| Long-term | Successful solutions, patterns | ChromaDB / FAISS |
| Episodic | "Last time I fixed this repo, I did X" | Vector DB with metadata |

### Flow
```python
# Before planning — retrieve relevant memory
context = memory.search("how did I fix auth bugs before?")

# Inject into prompt
prompt = f"Past experience:\n{context}\n\nCurrent task: {task}"

# After success — store what worked
memory.store("fixed null pointer in auth.py by checking for None before access")
```

### Deliverables
- [ ] `memory.py` — read/write interface
- [ ] ChromaDB setup + embedding config
- [ ] Memory injection into planner prompt
- [ ] Auto-store after successful task completion

---

## Phase 6 — Planning: Task Decomposition

**Goal:** Agent breaks complex tasks into ordered subtasks and executes them step by step.

### Planner Output (JSON)
```json
[
  {"id": 1, "task": "read existing code",      "depends_on": []},
  {"id": 2, "task": "write the function",      "depends_on": [1]},
  {"id": 3, "task": "write tests",             "depends_on": [2]},
  {"id": 4, "task": "run tests, fix if fail",  "depends_on": [3]}
]
```

### Execution
- Respect `depends_on` ordering
- Each subtask runs through the full agent loop
- Failed subtask → retry or escalate to user

### Deliverables
- [ ] `planner.py` — task decomposition prompt + JSON parser
- [ ] `executor.py` — run subtasks in dependency order
- [ ] Retry logic with configurable max attempts

---

## Phase 7 — Multi-Agent: Orchestration

**Goal:** Split responsibilities across focused specialist agents for scalability.

### Agents
| Agent | System Prompt Focus | Tools |
|---|---|---|
| Orchestrator | Plan, delegate, never code | `delegate_to_coder`, `delegate_to_tester` |
| Coder | Write clean Python | `read_file`, `write_file`, `run_code` |
| Tester | Write and run pytest tests | `read_file`, `write_file`, `run_code` |
| Reviewer | Code review, suggest improvements | `read_file`, `post_comment` |

### Communication
```
Orchestrator
    ├── → Coder Agent   (subtask + context)
    │         ↓ (code written)
    └── → Tester Agent  (code + test request)
               ↓ (pass/fail + details)
    ← Result bubbles back to Orchestrator
    ← Orchestrator reports to user
```

### Deliverables
- [ ] `agents/orchestrator.py`
- [ ] `agents/coder.py`
- [ ] `agents/tester.py`
- [ ] `agents/reviewer.py`
- [ ] Inter-agent message passing protocol

---

## MCP Server Integration

Add new skills at any phase by connecting MCP servers.

### Recommended MCP Servers for a Coding Agent
| MCP Server | Skills Unlocked |
|---|---|
| `mcp-server-git` | commit, push, branch, diff |
| `mcp-server-github` | PRs, issues, comments |
| `mcp-server-brave-search` | search docs, Stack Overflow |
| `mcp-server-filesystem` | broader file system access |
| `mcp-server-memory` | persistent knowledge graph |
| `mcp-server-postgres` | query/update databases |

### Usage in Code
```python
mcp_servers = [
    {"type": "stdio", "command": "uvx mcp-server-git"},
    {"type": "url",   "url": "https://mcp.brave.com/sse"},
]

response = claude(messages=messages, tools=local_tools, mcp_servers=mcp_servers)
```

---

## Tech Stack

| Layer | Choice |
|---|---|
| LLM | Claude API (claude-sonnet-4) |
| Language | Python 3.11+ |
| Web server | FastAPI |
| Queue | Redis + Celery |
| Vector DB | ChromaDB |
| Observability | Langfuse |
| MCP servers | uvx / hosted URLs |
| Package manager | uv |

---

## Folder Structure (end state)

```
coding-agent/
├── agent.py               # core agent loop
├── tools.py               # built-in tools
├── planner.py             # task decomposition
├── executor.py            # subtask runner
├── memory.py              # read/write memory
├── guardrails.py          # safety checks
├── observability.py       # tracing + logging
├── server.py              # FastAPI + webhook endpoints
├── worker.py              # background task runner
├── agents/
│   ├── orchestrator.py
│   ├── coder.py
│   ├── tester.py
│   └── reviewer.py
├── config.py              # rules, limits, model settings
└── requirements.txt
```

---

## Milestone Summary

| Phase | What You Get | Est. Complexity |
|---|---|---|
| 1 — Foundation | Working coding agent (CLI) | Low |
| 2 — Webhooks | Event-driven, async execution | Medium |
| 3 — Observability | Full trace visibility | Low |
| 4 — Guardrails | Safe to run in production | Low |
| 5 — Memory | Learns from past runs | Medium |
| 6 — Planning | Handles complex multi-step tasks | Medium |
| 7 — Multi-agent | Scalable, specialist agents | High |