"""
overseer.py
-----------
A small meta-agent that acts on the user's behalf.

The user gives it a GOAL. Every time the main coding agent finishes a turn,
the Overseer reads the conversation and returns ONE plain string: the next
message to send to the coding agent.

That string is fed into the graph as a normal HumanMessage.

It has no tools, no JSON schema, and no opinion about when to stop.
It runs until the user presses Stop.
"""

import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# How much context the Overseer gets. Both are capped because this loop can
# run indefinitely - without limits every prompt would grow until it blew
# through the context window.
MAX_TRANSCRIPT_MESSAGES = 40
MAX_CHARS_PER_MESSAGE = 1200
MAX_PAST_INSTRUCTIONS = 12


OVERSEER_SYSTEM_PROMPT = """You are the Overseer. You are standing in for a human user \
who is not at the keyboard. You do not write code and you do not call tools. Your only \
job is to write the next message that a separate coding agent will receive.

You are given THE GOAL, the instructions you have already sent, and the conversation so far.

Rules:
  - Reply with the message itself. No JSON, no preamble, no quotes, no "Instruction:" label.
  - One concrete next step at a time. Not a plan, not a ten-item checklist.
  - Be specific about files, functions and commands. "Fix the bug" is useless.
    "Open tools/registry.py and make get_tools() log the module name when an import fails"
    is useful.
  - Verify rather than trust. Prefer asking it to run the code or re-read the changed file
    over accepting its summary.
  """


def get_overseer_llm():
    """
    Runs on OVERSEER_MODEL_ID if set, otherwise the main MODEL_ID.
    Low temperature - steering should be boring.
    """
    api_key = os.getenv("GOOGLE_API_KEY2")
    model_id = os.getenv("OVERSEER_MODEL_ID") or os.getenv("MODEL_ID")
    if not api_key or not model_id:
        return None
    return ChatGoogleGenerativeAI(
        model=model_id,
        temperature=float(os.getenv("OVERSEER_TEMPERATURE", 0.2)),
        google_api_key=api_key,
    )


def _as_text(content) -> str:
    """
    Gemini sometimes returns content as a list of blocks instead of a string.
    Flatten it before any string operation.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _truncate(text, limit: int = MAX_CHARS_PER_MESSAGE) -> str:
    text = _as_text(text)
    return text #if len(text) <= limit else text[:limit] + "\n... [truncated]"


def build_transcript(messages) -> str:
    """Flatten LangGraph state messages into compact plain text."""
    if not messages:
        return "(nothing yet - the coding agent has not been given anything)"

    lines = []
    for msg in messages[-MAX_TRANSCRIPT_MESSAGES:]:
        if isinstance(msg, HumanMessage):
            lines.append(f"USER: {_truncate(msg.content)}")

        elif isinstance(msg, AIMessage):
            body = _truncate(msg.content)
            if body.strip():
                lines.append(f"CODING_AGENT: {body}")
            for call in getattr(msg, "tool_calls", []) or []:
                args = {k: _truncate(v, 300) for k, v in (call.get("args") or {}).items()}
                lines.append(f"CODING_AGENT wants to run `{call.get('name')}` with {args}")

        elif isinstance(msg, ToolMessage):
            lines.append(f"TOOL_RESULT [{msg.name}]: {_truncate(msg.content, 800)}")

    return "\n\n".join(lines)


def next_instruction(goal: str, messages, past_instructions=None) -> str:
    """
    Return the next message to send to the coding agent, as a plain string.

    Raises RuntimeError if the Overseer cannot produce one (missing config,
    API failure, empty response). That is a genuine failure, not the Overseer
    deciding to stop - the caller surfaces it and ends the run.
    """
    llm = get_overseer_llm()
    if llm is None:
        raise RuntimeError("No GOOGLE_API_KEY / MODEL_ID configured for the Overseer.")

    past = past_instructions or []
    recent = past[-MAX_PAST_INSTRUCTIONS:]
    if recent:
        offset = len(past) - len(recent)
        past_block = "\n".join(
            f"  {offset + i + 1}. {_truncate(p, 300)}" for i, p in enumerate(recent)
        )
        if offset:
            past_block = f"  ... {offset} earlier omitted ...\n" + past_block
    else:
        past_block = "  (none - this is your opening instruction)"

    user_block = f"""THE GOAL:
{goal}

INSTRUCTIONS YOU HAVE ALREADY SENT:
{past_block}

THE CONVERSATION SO FAR:
{build_transcript(messages)}

Write the next message to send to the coding agent."""

    try:
        response = llm.invoke(
            [("system", OVERSEER_SYSTEM_PROMPT), ("human", user_block)]
        )
    except Exception as e:
        raise RuntimeError(f"Overseer LLM call failed: {e}") from e

    instruction = _as_text(getattr(response, "content", "")).strip()
    if not instruction:
        raise RuntimeError("Overseer returned an empty message.")
    return instruction