# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-03-09

### Added
- **Comprehensive Test Suite**:
  - `tests/test_agent_core.py`: Verifies LangGraph compilation (`create_graph`) and initial agent state initialization with checkpointer.
  - `tests/test_tools.py`: Verifies the tool registry successfully loads all expected tools (`run_bash`, `apply_patch`, `web_search`, `web_fetch`).
  - `tests/test_cli.py`: Covers command-line interface execution (`run_cli`), handling user messaging, immediate exit, tool approval (`y`), and tool denial (`n`).
- **Git Tracking**: Staged all new test files and changelog into git version control.
