from langchain_core.tools import tool
from utils.helpers import normalize_content, sanitize_content, get_llm
import time

@tool
def read_file(path: str, query: str, justification: str) -> str:
    """Reads a file and extracts relevant parts for the query.

    Args:
        path: Path to the file.
        query: The specific information you are looking for.
        justification: Explain why you need to read this file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Normalize in-memory for the LLM
        content = normalize_content(raw_content)

        # Split into lines and add line numbers
        lines = content.splitlines()
        total_lines = len(lines)

        # If file is small enough, process in one go
        if total_lines <= 500:
            numbered_content = "".join(
                [f"{i + 1}: {line}\n" for i, line in enumerate(lines)]
            )
            llm = get_llm()
            prompt = _build_prompt(path, query, numbered_content, 1, total_lines)
            response = llm.invoke(prompt)
            return sanitize_content(response.content)

        # Divide into base chunks of 500 lines, then pad the start of each
        # chunk (except the first) with the last 50 lines of the previous
        # chunk, so consecutive chunks overlap by 50 lines (~550 lines each).
        chunk_size = 500
        overlap = 50

        all_summaries = []
        llm = get_llm()

        for base_start in range(0, total_lines, chunk_size):
            base_end = min(base_start + chunk_size, total_lines)

            # Extend the start backwards by `overlap` lines (except for the first chunk)
            start_idx = max(0, base_start - overlap)
            end_idx = base_end

            # Build numbered chunk
            chunk_lines = lines[start_idx:end_idx]
            numbered_chunk = "".join(
                [f"{start_idx + i + 1}: {line}\n" for i, line in enumerate(chunk_lines)]
            )

            # Build prompt with context
            prompt = _build_prompt(
                path,
                query,
                numbered_chunk,
                start_idx + 1,
                end_idx
            )

            # Call LLM
            response = llm.invoke(prompt)
            summary = sanitize_content(response.content)
            all_summaries.append(f"--- Lines {start_idx + 1}:{end_idx} ---\n{summary}")

            # Rate limiting: 4-second gap between calls (skip after last chunk)
            if end_idx < total_lines:
                time.sleep(4)

        # Combine all summaries
        combined = "\n\n".join(all_summaries)

        # Optional: Final synthesis pass if there are multiple chunks
        if len(all_summaries) > 1:
            synthesis_prompt = f"""You are a code analysis assistant.
The following are summaries from different sections of the same file.
Synthesize them into a single coherent summary of the file structure relevant to the query.
Note: section summaries may overlap by ~50 lines at their boundaries; merge/dedupe
any repeated descriptions rather than listing them twice.

File Path: {path}
Query: {query}

Section Summaries:
{combined}

Provide a unified summary in the format:
line(start:end) is about [description]
"""
            time.sleep(4)  # Rate limit before final call
            final_response = llm.invoke(synthesis_prompt)
            return sanitize_content(final_response.content)

        return combined

    except Exception as e:
        return f"Error reading file: {str(e)}"


def _build_prompt(path: str, query: str, numbered_content: str, start_line: int, end_line: int) -> str:
    """Builds the LLM prompt for a chunk."""
    return f"""You are a code analysis assistant.
Analyze the following file content (lines {start_line} to {end_line}) and provide a summary of the structure relevant to the user's query.
Instead of returning the full code, return a list of sections in the format:
line(start:end) is about [description]

File Path: {path}
Query: {query}

Content:
{numbered_content}
"""