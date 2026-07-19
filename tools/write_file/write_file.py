from langchain_core.tools import tool

@tool
def write_file(path: str, content: str, justification: str) -> str:
    """Writes content to a file (overwrites existing or creates new).

    Args:
        path: Path to the file.
        content: The complete content to write into the file.
        justification: Explain why this file is being created or overwritten.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"
