import os
import sys
from agent_core import create_graph, sanitize_content
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

def run_cli():
    print("--- Coding Agent CLI ---")
    print("Type 'exit' to quit.")
    
    # Use MemorySaver for a transient CLI session
    memory = MemorySaver()
    app = create_graph(checkpointer=memory)
    
    thread_config = {"configurable": {"thread_id": "cli_session"}}
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Stream the response
        print("Agent: ", end="", flush=True)
        try:
            full_response = ""
            for chunk, metadata in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=thread_config,
                stream_mode="messages",
            ):
                if hasattr(chunk, "content") and chunk.content:
                    # Sanitize the chunk content before printing
                    safe_chunk = sanitize_content(chunk.content)
                    print(safe_chunk, end="", flush=True)
                    full_response += safe_chunk
            print()
            
            # Check for tool calls
            snapshot = app.get_state(thread_config)
            if snapshot.next == ("tools",):
                print("\n[!] Agent wants to execute a tool. Approve? (y/n)")
                choice = input("> ").lower()
                if choice == 'y':
                    print("Executing...")
                    app.invoke(None, config=thread_config)
                    # Show the result
                    new_snapshot = app.get_state(thread_config)
                    print("Result:", sanitize_content(new_snapshot.values["messages"][-1].content))
                else:
                    print("Action denied.")
                    
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    run_cli()
