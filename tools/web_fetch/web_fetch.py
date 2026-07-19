import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

@tool
def web_fetch(url: str, justification: str) -> str:
    """Fetches the text content from a given URL.

    Args:
        url: The URL to fetch.
        justification: Explain why you need to fetch this URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator="\n")
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        return text[:5000]  # Limit to 5000 characters
    except Exception as e:
        return f"Error fetching URL: {e}"
