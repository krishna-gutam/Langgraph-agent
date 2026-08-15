import os
import json
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# Assuming these constants are defined somewhere accessible or need to be passed
# For now, we'll import them or define them here if needed.
# MODEL_ID = "gemini-1.5-flash"
# GLOBAL_REGISTRY_PATH = Path("skills")

def sanitize_content(content) -> str:
    """Safely extracts a string from LangChain's potential list-based content blocks."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        extracted = []
        for block in content:
            if isinstance(block, str):
                extracted.append(block)
            elif isinstance(block, dict) and "text" in block:
                extracted.append(block["text"])
        return "\n".join(extracted)
    return str(content)

def normalize_content(content: str) -> str:
    """Normalizes file content: converts tabs to spaces, ensures consistent line endings."""
    # Convert tabs to 4 spaces
    content = content.replace("\t", "    ")
    # Normalize line endings to \n
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    return content

def get_llm():
    # Check session state first, then environment
    api_key = os.getenv(
        "GOOGLE_API_KEY"
    )

    if not api_key:
        return None  # Return None instead of crashing the app

    # Assuming MODEL_ID is available in the global scope or imported
    # For now, we'll use a default or assume it's available
    model_id = os.getenv("MODEL_ID")
    temperature = os.getenv("TEMPERATURE")

    return ChatGoogleGenerativeAI(
        model=model_id, temperature=temperature, google_api_key=api_key
    )

def get_llm2():
    # Check session state first, then environment
    api_key = st.session_state.get("google_api_key_input") or os.getenv(
        "GOOGLE_API_KEY"
    )

    if not api_key:
        return None  # Return None instead of crashing the app

    # Assuming MODEL_ID is available in the global scope or imported
    # For now, we'll use a default or assume it's available
    model_id = "gemma-4-31b-it"
    temperature = st.session_state.get("temperature", 0.0)

    return ChatGoogleGenerativeAI(
        model=model_id, temperature=temperature, google_api_key=api_key
    )
