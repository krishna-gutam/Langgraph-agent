from langchain_core.tools import tool
from utils.helpers import normalize_content

@tool
def fetch_code(path: str, start_line: int, end_line: int, justification: str) -> str:
    """Fetches code from a file between start_line and end_line (inclusive).

    Args:
        path: Path to the file.
        start_line: The line number to start reading from.
        end_line: The line number to stop reading at.
        justification: Explain why you need to fetch this specific block of code.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Normalize in-memory
        content = normalize_content(raw_content)
        lines = content.splitlines()

        # Adjust for 0-based indexing
        start = max(0, start_line - 1)
        end = min(len(lines), end_line)

        selected_lines = lines[start:end]

        return "\n".join(selected_lines)
    except Exception as e:
        return f"Error fetching code: {str(e)}"
