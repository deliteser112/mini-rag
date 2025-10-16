# app.py

import os
import logging
from typing import List, Tuple, Dict, Any

import streamlit as st
from ingest import load_documents, chunk_documents
from store import build_faiss_index, save_index, load_index, search_index
from llm import call_llm
from langchain.schema import Document

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------
# Constants
# -------------------------------
INDEX_PATH = "data/faiss_index"
TOP_K = 3  # number of retrieved chunks per query
RAW_DOCS_FOLDER = "data/raw"

# -------------------------------
# Streamlit caching
# -------------------------------

@st.cache_data(show_spinner=False)
def load_and_chunk_documents(folder_path: str) -> List[Document]:
    """
    Load documents from folder and chunk them.
    Cached so repeated runs are faster.
    """
    raw_docs = load_documents(folder_path)
    if not raw_docs:
        return []
    chunks = chunk_documents(raw_docs)
    return chunks

@st.cache_resource(show_spinner=False)
def get_or_build_index(chunks: List[Document]) -> "FAISS":
    """
    Load existing FAISS index or build from chunks.
    Cached to avoid rebuilding.
    """
    if os.path.exists(INDEX_PATH):
        index = load_index(INDEX_PATH)
    else:
        if not chunks:
            st.warning("No documents to build FAISS index.")
            return None
        index = build_faiss_index(chunks)
        save_index(index, INDEX_PATH)
    return index

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Mini RAG LLM App", layout="wide")
st.title("📚 Mini RAG + Groq LLM Demo")

# --- File uploader ---
uploaded_files = st.file_uploader(
    "Upload documents (.pdf, .txt, .docx, .md)", 
    type=["pdf", "txt", "docx", "md"], 
    accept_multiple_files=True
)

# Save uploaded files
if uploaded_files:
    os.makedirs(RAW_DOCS_FOLDER, exist_ok=True)
    for f in uploaded_files:
        fpath = os.path.join(RAW_DOCS_FOLDER, f.name)
        with open(fpath, "wb") as out_file:
            out_file.write(f.read())
    st.success(f"Saved {len(uploaded_files)} files to {RAW_DOCS_FOLDER}")

# --- Load & chunk documents (cached) ---
chunks = load_and_chunk_documents(RAW_DOCS_FOLDER)
if not chunks:
    st.warning("No documents loaded. Upload files or check folder.")
    st.stop()
else:
    st.info(f"Loaded {len(chunks)} chunks from documents.")

# --- Load or build FAISS index (cached) ---
index = get_or_build_index(chunks)
if not index:
    st.error("FAISS index is missing. Cannot proceed.")
    st.stop()

# --- Query input ---
query = st.text_input("Enter your question:")

if st.button("Get Answer") and query:
    with st.spinner("Retrieving documents..."):
        results: List[Tuple[Document, float]] = search_index(query, index, top_k=TOP_K)
        retrieved_for_llm: List[Tuple[str, Dict[str, Any]]] = [
            (doc.page_content, doc.metadata) for doc, _ in results
        ]

    with st.spinner("Calling LLM..."):
        answer = call_llm(query, retrieved_for_llm)["answer"]

    # --- Display results ---
    st.subheader("🔹 Retrieved Context")
    for doc, score in results:
        st.markdown(f"**Source:** {doc.metadata.get('source')} | **Score:** {score:.3f}")
        st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
        st.write("---")

    st.subheader("📝 LLM Answer")
    st.write(answer)
