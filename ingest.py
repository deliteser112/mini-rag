
import os
import logging
from typing import List, Dict, Callable
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Supported loaders (extensible dict) ---
LOADERS: Dict[str, Callable[[str], object]] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".html": UnstructuredHTMLLoader,
    ".docx": UnstructuredWordDocumentLoader,   
    ".md": UnstructuredMarkdownLoader,         
}


def load_documents(folder_path: str = "data/raw") -> List[Document]:

    docs: List[Document] = []

    if not os.path.exists(folder_path):
        logging.warning(f"Folder {folder_path} does not exist. Skipping.")
        return docs

    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        ext = os.path.splitext(fname)[1].lower()

        if not os.path.isfile(fpath):
            continue  # skip dirs

        loader_cls = LOADERS.get(ext)
        
        if not loader_cls:
            logging.warning(f"Unsupported file format skipped: {fname}")
            continue

        try:
            loader = loader_cls(fpath)
            file_docs = loader.load()
            # Add filename as metadata for traceability
            for d in file_docs:
                d.metadata["source"] = fname
            docs.extend(file_docs)
            logging.info(f"Loaded {len(file_docs)} docs from {fname}")
        except Exception as e:
            logging.error(f"Error loading {fname}: {e}")
            continue

    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> List[Document]:
    if not docs:
        logging.warning("No documents provided for chunking.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(docs)
    logging.info(f"Split {len(docs)} docs into {len(chunks)} chunks.")
    return chunks


def load_and_prepare_corpus(
    folder_path: str = "data/raw",
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Full pipeline: load + chunk documents.
    Returns a list of LangChain Document objects ready for embeddings.
    """
    raw_docs = load_documents(folder_path)
    if not raw_docs:
        logging.warning("No documents loaded from corpus.")
        return []

    chunks = chunk_documents(raw_docs, chunk_size, chunk_overlap)
    return chunks

