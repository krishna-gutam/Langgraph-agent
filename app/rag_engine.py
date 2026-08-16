import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

# Initialize embeddings and LLM with Gemini models
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Define state structure for LangGraph workflow
class RAGState(TypedDict):
    documents: List[str]
    question: str
    vectorstore: FAISS
    context: List[Document]
    answer: str

def create_vectorstore_node(state: RAGState) -> RAGState:
    """Chunks documents and creates/adds them to a FAISS vector store."""
    docs = state.get("documents", [])
    if not docs:
        state["vectorstore"] = None
        return state

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.create_documents(docs)
    
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    state["vectorstore"] = vectorstore
    return state

def retrieve_node(state: RAGState) -> RAGState:
    """Retrieves relevant document chunks based on the question."""
    vectorstore = state.get("vectorstore")
    question = state.get("question", "")
    
    if not vectorstore or not question:
        state["context"] = []
        return state
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)
    state["context"] = retrieved_docs
    return state

def generate_node(state: RAGState) -> RAGState:
    """Generates an answer using the LLM based on the retrieved context and question."""
    question = state.get("question", "")
    context_docs = state.get("context", [])
    
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question accurately using only the provided context below.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    
    if not context_text:
        state["answer"] = "No relevant context found to answer the question."
        return state
        
    messages = prompt.format_messages(context=context_text, question=question)
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state

def build_rag_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(RAGState)
    
    workflow.add_node("create_vectorstore", create_vectorstore_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("create_vectorstore")
    workflow.add_edge("create_vectorstore", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
