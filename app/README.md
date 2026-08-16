# 📚 Streamlit + LangGraph RAG Application

A Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, **LangGraph**, and **FAISS**.

---

## 🚀 Project Structure

- `app.py`: Streamlit user interface and interactive workflow runner.
- `rag_engine.py`: Core RAG logic including document chunking, FAISS vector store creation, retrieval, LLM generation, and the LangGraph state graph workflow.
- `requirements.txt`: Python package dependencies.
- `.env.example`: Template for environment variables (Google Gemini API key).
- `README.md`: Project documentation and usage guide.

---

## 🛠️ Setup Instructions

### 1. Clone or Navigate to the Project Directory
Make sure you are in the project root directory.

### 2. Install Dependencies
Install all required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_google_api_key_here
```
*(Alternatively, you can input your API key directly in the sidebar of the Streamlit app interface).*

---

## ▶️ Running the Application

Launch the Streamlit app by running:
```bash
python -m streamlit run app.py
```

Open the provided local URL (usually `http://localhost:8501`) in your browser to interact with the application.

---

## 💡 How It Works

1. **Document Indexing Node**: Text provided in the UI is split into manageable chunks using `RecursiveCharacterTextSplitter` and embedded into a **FAISS** vector store using Google Gemini Embeddings.
2. **Retrieval Node**: Given your question, the pipeline queries the FAISS vector store to retrieve the top $k$ relevant context chunks.
3. **Generation Node**: The retrieved context chunks and your question are passed to **Gemini 1.5 Flash** via a prompt template to synthesize an accurate, context-grounded response.
4. **LangGraph Workflow**: Manages state transition across the steps (`create_vectorstore` ➔ `retrieve` ➔ `generate`).
