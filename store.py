# mini_rag/store.py

import os
import pickle
import logging
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Constants ---
INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------
# 1️⃣ Build FAISS index
# ------------------------------------------------------------
def build_faiss_index(docs: List[Document]) -> FAISS:
    """
    Builds a FAISS index from LangChain Document chunks.
    """
    if not docs:
        raise ValueError("No documents provided to build index.")
    
    logging.info(f"Building FAISS index from {len(docs)} chunks...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embedder)
    logging.info("✅ FAISS index built successfully.")
    return vectorstore


# ------------------------------------------------------------
# 2️⃣ Save / Load index
# ------------------------------------------------------------
def save_index(index: FAISS, path: str = INDEX_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    index.save_local(path)
    logging.info(f"💾 Index saved to {path}")


def load_index(path: str = INDEX_PATH) -> FAISS:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No FAISS index found at {path}")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index = FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)
    logging.info(f"📂 Loaded FAISS index from {path}")
    return index


# ------------------------------------------------------------
# 3️⃣ Query index
# ------------------------------------------------------------
def search_index(query: str, index: FAISS, top_k: int = 3) -> List[Tuple[Document, float]]:
    """
    Searches the FAISS index for relevant chunks.
    Returns a list of (Document, similarity_score).
    """
    results = index.similarity_search_with_score(query, k=top_k)
    logging.info(f"🔍 Retrieved {len(results)} matches for query: {query[:50]}...")
    return results


# ------------------------------------------------------------
# 🔹 Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    from ingest import load_and_prepare_corpus

    # Step 1: Load + chunk docs
    docs = load_and_prepare_corpus("data")

    # Step 2: Build FAISS index
    index = build_faiss_index(docs)

    # Step 3: Save
    save_index(index)

    # Step 4: Search
    results = search_index("What is the main topic?", index)
    for doc, score in results:
        print(f"Score: {score:.3f} | Source: {doc.metadata.get('source')}")
        print(doc.page_content[:200], "\n---\n")
