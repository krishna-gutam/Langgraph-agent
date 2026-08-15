"""
overseer.py
-----------
The "Overseer": a meta-agent that acts on the user's behalf.

The user hands it a GOAL. It then watches the main coding agent's
conversation and repeatedly decides what to *say* to it next, until the
goal is reached, it gets stuck, or it hits the step budget.

Design notes:
  * The Overseer is deliberately NOT a node in the main LangGraph graph.
    It has no tools and never mutates graph state directly. Its only
    output is the text of the next HumanMessage.
  * This keeps the main agent's checkpoint history clean (it just looks
    like a very decisive user) and makes the whole feature removable
    without touching agent_core.py.
  * One decision per Streamlit rerun. The UI driver in main.py owns the
    loop; this module is stateless.
"""

import os
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_MAX_STEPS = 8

# How much of the conversation the Overseer gets to see.
MAX_TRANSCRIPT_MESSAGES = 40
MAX_CHARS_PER_MESSAGE = 1200


OVERSEER_SYSTEM_PROMPT = """You are the Overseer. You are a proxy for a human user \
who is not at the keyboard right now. You do not write code and you do not call tools. \
Your only job is to steer a separate coding agent by writing the next message it receives.

You will be given:
  - THE GOAL: what the human wants achieved.
  - THE TRANSCRIPT: the conversation so far between you (shown as USER) and the coding agent.
  - Your own previous instructions, so you do not repeat yourself.

How to steer well:
  - Give ONE concrete next instruction at a time. Not a plan, not a checklist of ten items.
  - Be specific about files, functions, and commands. "Fix the bug" is useless.
    "Open tools/registry.py and make get_tools() skip modules that fail to import,
     logging the module name" is useful.
  - The coding agent is required to ask permission before modifying files or running
    commands. You are authorised to grant that permission on the human's behalf when
    the proposed action clearly serves THE GOAL. In that case just say so plainly,
    e.g. "Approved, go ahead and apply that edit."
  - If the agent goes off-track, redirect it firmly. If it hallucinates a file or an API,
    tell it to read the actual file first.
  - Verify before declaring victory. Prefer asking the agent to run the code or re-read
    the changed file over trusting its summary.
  - Never invent facts about the codebase. If you do not know, tell the agent to go look.

Escalate to the human (status "needs_human") when, and only when:
  - the goal is ambiguous in a way that changes what should be built,
  - an action is destructive or irreversible and not clearly implied by the goal
    (deleting data, force-pushing, rotating credentials, spending money),
  - the agent has failed at the same thing three times and you have no new angle,
  - a secret, credential, or payment is required.

Reply with a single JSON object and nothing else. No prose, no markdown fences.

{
  "assessment": "one sentence on where the coding agent currently stands",
  "status": "continue" | "goal_reached" | "needs_human",
  "instruction": "the exact text to send to the coding agent (or, if needs_human, the question to put to the human)",
  "reason": "one sentence on why this is the right next move"
}

If status is "goal_reached", leave "instruction" as an empty string."""


def get_overseer_llm():
    """
    The Overseer can run on a different (usually cheaper/faster) model than the
    main agent. Set OVERSEER_MODEL_ID in .env to override; otherwise it falls
    back to MODEL_ID. Temperature is held low - steering should be boring.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    model_id = os.getenv("OVERSEER_MODEL_ID") or os.getenv("MODEL_ID")
    if not model_id:
        return None
    temperature = float(os.getenv("OVERSEER_TEMPERATURE", 0.2))
    return ChatGoogleGenerativeAI(
        model=model_id, temperature=temperature, google_api_key=api_key
    )


def _truncate(text: str, limit: int = MAX_CHARS_PER_MESSAGE) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


def build_transcript(messages, max_messages: int = MAX_TRANSCRIPT_MESSAGES) -> str:
    """
    Flattens LangGraph state messages into a compact plain-text transcript.
    Tool results are truncated hard - the Overseer needs the shape of what
    happened, not every byte of a file dump.
    """
    if not messages:
        return "(no conversation yet - the coding agent has not been given anything)"

    lines = []
    for msg in messages[-max_messages:]:
        if isinstance(msg, HumanMessage):
            lines.append(f"USER: {_truncate(msg.content)}")

        elif isinstance(msg, AIMessage):
            content = _as_text(msg.content)
            if content.strip():
                lines.append(f"CODING_AGENT: {_truncate(content)}")
            for call in getattr(msg, "tool_calls", []) or []:
                args = {k: _truncate(v, 300) for k, v in (call.get("args") or {}).items()}
                lines.append(
                    f"CODING_AGENT wants to run tool `{call.get('name')}` with {args}"
                )

        elif isinstance(msg, ToolMessage):
            lines.append(f"TOOL_RESULT [{msg.name}]: {_truncate(msg.content, 800)}")

    return "\n\n".join(lines)


def _as_text(content) -> str:
    """
    Gemini may return content as a list of blocks rather than a plain string.
    Flatten it before any string operation. Mirrors agent_core.sanitize_content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # text blocks, and function-call blocks that carry no text
                if "text" in block:
                    parts.append(str(block["text"]))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _extract_json(raw) -> "dict | None":
    """Pull a JSON object out of an LLM response that may be wrapped in fences."""
    raw = _as_text(raw)
    if not raw.strip():
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost braces.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def decide_next_step(
    goal: str,
    messages,
    step: int,
    max_steps: int,
    past_instructions=None,
) -> dict:
    """
    Ask the Overseer what to tell the coding agent next.

    Returns a dict with keys: assessment, status, instruction, reason.
    Never raises - failures come back as status "needs_human" so the UI
    stops the loop and hands control to the actual human.
    """
    llm = get_overseer_llm()
    if llm is None:
        return {
            "assessment": "Overseer could not start.",
            "status": "needs_human",
            "instruction": "No GOOGLE_API_KEY / MODEL_ID configured for the Overseer.",
            "reason": "Missing configuration.",
        }

    past = past_instructions or []
    past_block = (
        "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(past))
        if past
        else "  (none yet - this is your opening instruction)"
    )

    user_block = f"""THE GOAL:
{goal}

YOUR PREVIOUS INSTRUCTIONS:
{past_block}

STEP BUDGET: this is step {step + 1} of {max_steps}. \
If you are near the end of the budget, drive towards the smallest useful \
finished state rather than starting something new.

THE TRANSCRIPT:
{build_transcript(messages)}

Decide the next move. JSON only."""

    prompt = [
        ("system", OVERSEER_SYSTEM_PROMPT),
        ("human", user_block),
    ]

    for attempt in range(2):
        try:
            response = llm.invoke(prompt)
        except Exception as e:  # network, quota, safety block...
            return {
                "assessment": "Overseer call failed.",
                "status": "needs_human",
                "instruction": f"Overseer LLM error: {e}",
                "reason": "Cannot continue without a decision.",
            }

        parsed = _extract_json(getattr(response, "content", ""))
        if isinstance(parsed, dict) and parsed.get("status"):
            # Coerce every field to a string - the caller does .strip() on
            # "instruction" and a stray list here would crash the UI.
            decision = {
                "assessment": _as_text(parsed.get("assessment", "")),
                "status": str(parsed.get("status", "")).strip(),
                "instruction": _as_text(parsed.get("instruction", "")),
                "reason": _as_text(parsed.get("reason", "")),
            }
            if decision["status"] not in ("continue", "goal_reached", "needs_human"):
                decision["status"] = "continue"
            return decision

        # One retry with a blunter nudge.
        prompt = prompt + [
            ("ai", _as_text(getattr(response, "content", "")) or "(empty)"),
            ("human", "That was not valid JSON. Reply with the JSON object only."),
        ]

    return {
        "assessment": "Overseer returned unparseable output twice.",
        "status": "needs_human",
        "instruction": "The Overseer could not produce a valid decision. Take over manually.",
        "reason": "Malformed model output.",
    }