# store.py
import os
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Any

from langchain.schema import Document


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# Defaults / constants

DEFAULT_INDEX_DIR = Path("data") / "faiss_index"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


_EMBEDDER: Optional[HuggingFaceEmbeddings] = None



# Helper: embedder factory (cached)

def get_embedder() -> HuggingFaceEmbeddings:
    """
    Lazy-create and return a HuggingFaceEmbeddings instance.
    Keeps a single instance in `_EMBEDDER` to avoid reloading the model repeatedly.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        logging.info(f"Initializing embedder model: {EMBEDDING_MODEL}")
        _EMBEDDER = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMBEDDER



# Build / Save / Load

def build_faiss_index(docs: List[Document]) -> FAISS:
    """
    Build a FAISS vectorstore from a list of LangChain Document objects.
    Returns the vectorstore instance.
    """
    if not docs:
        raise ValueError("No documents provided to build index.")
    logging.info(f"Building FAISS index from {len(docs)} chunks...")
    embedder = get_embedder()
    vectorstore = FAISS.from_documents(docs, embedder)
    logging.info("✅ FAISS index built successfully.")
    return vectorstore


def save_index(index: FAISS, path: str | Path = DEFAULT_INDEX_DIR) -> None:
    """
    Save a FAISS vectorstore to `path` (directory). Creates the directory if needed.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving FAISS index to {str(path)} ...")
    
    index.save_local(str(path))
    logging.info("💾 Index saved.")


def load_index(path: str | Path = DEFAULT_INDEX_DIR) -> FAISS:
    """
    Load a FAISS vectorstore from `path` (directory).
    Returns the vectorstore instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No FAISS index found at {path}")
    logging.info(f"Loading FAISS index from {str(path)} ...")
    embedder = get_embedder()
    index = FAISS.load_local(str(path), embedder, allow_dangerous_deserialization=True)
    logging.info("📂 FAISS index loaded.")
    return index



# Incremental add

def add_to_index(index: FAISS, docs: List[Document], save_path: Optional[str | Path] = None) -> FAISS:
    """
    Incrementally add documents to an existing FAISS vectorstore.
    - `index` is an existing FAISS vectorstore instance (created with an embedder)
    - `docs` is a list of LangChain Document objects to be added
    - If `save_path` is provided, the index will be saved after addition.
    Returns the updated index.
    """
    if not docs:
        logging.info("add_to_index called with empty docs list — nothing to add.")
        return index

    logging.info(f"Adding {len(docs)} documents to FAISS index...")
    try:
        
        index.add_documents(docs)
        logging.info("✅ Added documents to index.")
    except Exception as e:
        logging.error(f"Failed to add documents to FAISS index: {e}")
        raise

    if save_path:
        try:
            save_index(index, save_path)
        except Exception as e:
            logging.warning(f"Failed to save index after add: {e}")

    return index



# Querying

def search_index(query: str, index: FAISS, top_k: int = 3) -> List[Tuple[Document, float]]:
    """
    Search the FAISS index for the most relevant chunks.
    Returns a list of (Document, similarity_score) tuples.
    """
    if index is None:
        logging.warning("search_index called with `index=None`.")
        return []

    try:
        results = index.similarity_search_with_score(query, k=top_k)
        logging.info(f"🔍 Retrieved {len(results)} matches for query: {query[:80]}...")
        return results
    except Exception as e:
        logging.error(f"Error while searching index: {e}")
        raise

