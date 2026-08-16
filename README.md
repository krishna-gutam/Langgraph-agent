# Langgraph-agent

This project is a modular LangGraph-based agent designed for various tasks.

## Structure
- `agent_core.py`: Core logic and graph definitions.
- `codingagent2.py`: Main entry point for the agent.
- `tools/`: Contains various tools used by the agent, including `bash_exec`.
- `utils/`: Helper functions.

## Best Practices
- **Automatic Function Calling (AFC):** When implementing automatic function calling, please use `Chat.send_message` and `Chat.send_message_stream` instead of `Models.generate_content` and `Models.generate_content_stream`. Direct use of the latter is discouraged.

