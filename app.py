# app.py
import os
import logging
import hashlib
import pickle
import html
from pathlib import Path
from typing import List, Tuple, Dict, Any
import streamlit as st
from langchain.schema import Document
import ingest
import store  
from llm import call_llm


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# Paths & Constants (auto-create)

RAW_DOCS_FOLDER = Path(os.getenv("RAW_DOCS_FOLDER", "data"))
INDEX_DIR = RAW_DOCS_FOLDER / "faiss_index"
CACHE_DIR = RAW_DOCS_FOLDER / ".cache_files"

RAW_DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TOP_K = 7
DEFAULT_TEMPERATURE = 0.0


# Helper utilities

def file_md5(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_chunks_checksum(chunks: List[Document]) -> str:
    h = hashlib.md5()
    
    for doc in sorted(chunks, key=lambda d: (str(d.metadata.get("source", "")), d.page_content[:50])):
        h.update(doc.page_content.encode("utf-8"))
    return h.hexdigest()


def escape_and_highlight(text: str, query: str) -> str:
    """Escape HTML and highlight simple occurrences of query with <mark>."""
    if not query:
        return html.escape(text)
    escaped = html.escape(text)
    q_escaped = html.escape(query)
    
    try:
        import re
        pattern = re.compile(re.escape(q_escaped), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)
    except Exception:
        highlighted = escaped.replace(q_escaped, f"<mark>{q_escaped}</mark>")
    return highlighted



@st.cache_data(show_spinner=False, persist=True)
def load_and_chunk_documents_incremental(folder_path: Path) -> List[Document]:
    """
    Incrementally loads & chunks documents in folder_path.
    Caches per-file chunks in CACHE_DIR/.cache_<filename>.pkl keyed by MD5 hash.
    """
    supported_exts = set(ingest.LOADERS.keys())  # e.g. {".pdf", ".txt", ".docx", ...}
    all_chunks: List[Document] = []

    
    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in supported_exts:
            continue

        file_hash = file_md5(file_path)
        cache_file = CACHE_DIR / f".cache_{file_path.name}.pkl"

        
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, dict) and cached.get("hash") == file_hash and "chunks" in cached:
                    
                    cached_chunks = cached["chunks"]
                   
                    for d in cached_chunks:
                        if "source" not in d.metadata:
                            d.metadata["source"] = file_path.name
                    all_chunks.extend(cached_chunks)
                    continue
            except Exception as e:
                logging.warning(f"Failed to load cache for {file_path.name}: {e}")

        
        loader_cls = ingest.LOADERS.get(ext)
        if loader_cls is None:
            logging.warning(f"No loader for extension {ext}, skipping {file_path.name}")
            continue

        try:
            loader = loader_cls(str(file_path))
            file_docs = loader.load()
            
            for d in file_docs:
                d.metadata["source"] = file_path.name
            
            chunks = ingest.chunk_documents(file_docs)
            
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump({"hash": file_hash, "chunks": chunks}, f)
            except Exception as e:
                logging.warning(f"Couldn't write cache for {file_path.name}: {e}")
            all_chunks.extend(chunks)
        except Exception as e:
            logging.error(f"Error loading/chunking {file_path.name}: {e}")
            continue

    return all_chunks



# Load or (incrementally) build FAISS index

@st.cache_resource(show_spinner=False)
def get_or_build_index(chunks: List[Document]):
    """
    Loads existing FAISS index from INDEX_DIR if checksum matches.
    If checksum changed:
      - if store.add_to_index exists: add only new chunks
      - else: rebuild index fully
    Caches the loaded index resource to avoid reloading on reruns.
    """
    # ensure INDEX_DIR exists
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    checksum_file = INDEX_DIR / "chunks.checksum"
    hashes_file = INDEX_DIR / "chunks_hashes.pkl"  # saved list of chunk hashes for incremental diff

    current_checksum = compute_chunks_checksum(chunks) if chunks else ""

    index = None
    previous_checksum = None
    if checksum_file.exists():
        previous_checksum = checksum_file.read_text()

    # If checksum matches and index exists, load it
    try:
        if previous_checksum == current_checksum:
            # Attempt to load
            try:
                logging.info("Checksum unchanged — loading existing FAISS index.")
                index = store.load_index(str(INDEX_DIR))
                return index
            except Exception as e:
                logging.warning(f"Failed to load existing index despite matching checksum: {e}")
                # fallthrough to rebuild
    except Exception:
        pass

    # If here, checksum changed or no index → attempt incremental update if possible
    if INDEX_DIR.exists():
        
        prev_hashes = []
        if hashes_file.exists():
            try:
                with open(hashes_file, "rb") as f:
                    prev_hashes = pickle.load(f)
            except Exception as e:
                logging.warning(f"Could not load prev hashes: {e}")
                prev_hashes = []

        
        cur_hashes = []
        for d in chunks:
            
            cur_hashes.append(hashlib.md5(d.page_content.encode("utf-8")).hexdigest())

       
        new_chunks = []
        for i, h in enumerate(cur_hashes):
            if h not in prev_hashes:
                new_chunks.append(chunks[i])

        
        add_fn = getattr(store, "add_to_index", None)
        if add_fn and new_chunks:
            try:
                logging.info(f"Adding {len(new_chunks)} new chunks incrementally to existing index...")
                index = store.load_index(str(INDEX_DIR))
                add_fn(index, new_chunks)
                
                store.save_index(index, str(INDEX_DIR))
                with open(checksum_file, "w") as f:
                    f.write(current_checksum)
                with open(hashes_file, "wb") as f:
                    pickle.dump(cur_hashes, f)
                return index
            except Exception as e:
                logging.warning(f"Incremental add failed: {e} — falling back to rebuild.")

   
    if not chunks:
        logging.warning("No chunks available to build index.")
        return None

    logging.info("Building FAISS index (full rebuild)...")
    index = store.build_faiss_index(chunks)
    
    try:
        store.save_index(index, str(INDEX_DIR))
        with open(checksum_file, "w") as f:
            f.write(current_checksum)
        cur_hashes = [hashlib.md5(d.page_content.encode("utf-8")).hexdigest() for d in chunks]
        with open(hashes_file, "wb") as f:
            pickle.dump(cur_hashes, f)
    except Exception as e:
        logging.warning(f"Error saving index/checksum: {e}")

    return index



# Streamlit UI
st.set_page_config(page_title="Mini RAG + Groq LLM", layout="wide")
st.title("📚 Mini RAG + Groq LLM Demo")

# Sidebar controls
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=DEFAULT_TOP_K)
temperature = st.sidebar.slider("LLM temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05)
rebuild_index_button = st.sidebar.button("Force rebuild index")

# Upload area
st.markdown("### Upload documents")
uploaded_files = st.file_uploader(
    "Upload documents (.pdf, .txt, .docx, .md)",
    type=[ext.lstrip(".") for ext in ingest.LOADERS.keys()],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Saving {len(uploaded_files)} file(s) to `{RAW_DOCS_FOLDER}`...")
    for f in uploaded_files:
        dest = RAW_DOCS_FOLDER / f.name
        with open(dest, "wb") as out:
            out.write(f.read())
    st.success("Saved uploaded files. They will be processed and indexed automatically.")

    try:
        load_and_chunk_documents_incremental.clear()
    except Exception:
        pass

# Load & chunk documents (incremental + cached)
with st.spinner("Loading & chunking documents..."):
    chunks = load_and_chunk_documents_incremental(RAW_DOCS_FOLDER)

if not chunks:
    st.warning("No documents found in the data folder. Upload documents to start.")
    st.stop()
else:
    st.info(f"Loaded {len(chunks)} chunks from documents.")

# Force rebuild handling
if rebuild_index_button:
    try:
       
        get_or_build_index.clear()
        checksum_file = INDEX_DIR / "chunks.checksum"
        hashes_file = INDEX_DIR / "chunks_hashes.pkl"
        if checksum_file.exists():
            checksum_file.unlink()
        if hashes_file.exists():
            hashes_file.unlink()
        st.success("Index cache cleared. Rebuilding index now...")
    except Exception as e:
        st.warning(f"Could not clear index cache: {e}")

# Load or build FAISS index (cached resource)
with st.spinner("Loading / building FAISS index (may take a while on first run)..."):
    index = get_or_build_index(chunks)

if index is None:
    st.error("FAISS index not available. Please upload documents or check logs.")
    st.stop()

# Query UI
st.markdown("### Ask a question")
query = st.text_input("Enter your question", key="query_input")

if st.button("Get Answer") and query:
    
    try:
        with st.spinner("Retrieving top documents..."):
            results: List[Tuple[Document, float]] = store.search_index(query, index, top_k=top_k)
    except Exception as e:
        st.error(f"Search failed: {e}")
        results = []

    retrieved_for_llm: List[Tuple[str, Dict[str, Any]]] = []
    for doc, score in results:

        if "source" not in doc.metadata:
            doc.metadata["source"] = doc.metadata.get("source", "unknown")
        retrieved_for_llm.append((doc.page_content, doc.metadata))

    # Call LLM
    answer = ""
    try:
        with st.spinner("Calling LLM..."):
            resp = call_llm(query, retrieved_for_llm, temperature=temperature)
            answer = resp.get("answer") if isinstance(resp, dict) else str(resp)
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        answer = ""

    # Display retrieved context
    st.subheader("🔹 Retrieved Context")
    if not results:
        st.info("No relevant documents found.")
    else:
        for i, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            header = f"{i}. {source} — score {score:.3f}"
            with st.expander(header, expanded=(i == 1)):
                # show metadata
                md = doc.metadata.copy()
                st.markdown("**Metadata:**")
                st.json(md)
                # highlight occurrences of query in the snippet
                snippet = doc.page_content[:5000]  # guard long output
                hl = escape_and_highlight(snippet, query)
                st.markdown(hl, unsafe_allow_html=True)
                # show full content button
                if st.button(f"Show full text #{i}", key=f"full_{i}"):
                    st.text_area(f"Full document text — {source}", value=doc.page_content, height=300)

    # Display LLM answer
    st.subheader("📝 LLM Answer")
    if answer:
        # Plain text display first
        st.write(answer)
        # Then a copyable text area for convenience
        st.text_area("Answer (copyable)", value=answer, height=200)
    else:
        st.info("No answer returned from the LLM.")


# Footer
st.markdown("---")
st.caption("Mini RAG — incremental ingestion, cached index, and Groq LLM (via call_llm).")
