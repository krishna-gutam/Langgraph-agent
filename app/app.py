import os
import streamlit as st
from rag_engine import build_rag_graph

st.set_page_config(page_title="Streamlit + LangGraph RAG App", layout="centered")

st.title("📚 LangGraph RAG Application with Streamlit")
st.markdown("Index your documents and query them using a LangGraph-powered RAG pipeline.")

# Sidebar for API Key configuration
st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input("Google Gemini API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))

if api_key_input:
    os.environ["GOOGLE_API_KEY"] = api_key_input

# Main Interface
st.subheader("1. Add Knowledge Base Documents")
doc_input = st.text_area(
    "Enter text documents (separate multiple documents or paragraphs if desired, or paste a single large text):",
    placeholder="Type or paste your reference text here..."
)

st.subheader("2. Ask a Question")
question_input = st.text_input("Enter your question based on the documents above:")

if st.button("Run RAG Pipeline", type="primary"):
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Please provide a valid Google Gemini API Key in the sidebar or via environment variables.")
    elif not doc_input.strip():
        st.warning("Please provide some document text to index.")
    elif not question_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing through LangGraph workflow..."):
            try:
                app_graph = build_rag_graph()
                
                initial_state = {
                    "documents": [doc_input],
                    "question": question_input,
                    "vectorstore": None,
                    "context": [],
                    "answer": ""
                }
                
                result = app_graph.invoke(initial_state)
                
                st.success("Pipeline executed successfully!")
                
                st.markdown("### 🤖 Answer")
                st.write(result.get("answer", "No answer generated."))
                
                st.markdown("### 🔍 Retrieved Context Chunks")
                context_docs = result.get("context", [])
                if context_docs:
                    for i, doc in enumerate(context_docs):
                        with st.expander(f"Context Chunk {i+1}"):
                            st.write(doc.page_content)
                else:
                    st.info("No context chunks retrieved.")
                    
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")

