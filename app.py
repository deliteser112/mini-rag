import streamlit as st
from ingest import build_documents_for_index
from store import VectorStore
from llm import build_prompt, call_openai
import tempfile
import os

st.set_page_config(page_title="mini_RAG", layout="centered")
st.title("mini_RAG — Mini Retrieval-Augmented Generation")

vs = VectorStore()

st.sidebar.header("Ingest documents")
uploaded = st.sidebar.file_uploader("Upload PDF / TXT", accept_multiple_files=True)
if st.sidebar.button("Ingest uploaded"):
    if not uploaded:
        st.sidebar.warning("Please upload at least one file.")
    else:
        tmp_paths = []
        for up in uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tmp:
                tmp.write(up.getvalue())
                tmp_paths.append(tmp.name)
        docs = build_documents_for_index(tmp_paths)
        vs.add_documents(docs)
        st.sidebar.success(f"Ingested {len(docs)} chunks.")

st.sidebar.markdown("---")
query = st.text_area("Ask a question about the uploaded docs", height=120)
top_k = st.sidebar.slider("Top K results", 1, 10, 5)

if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        results = vs.search(query, top_k=top_k)
        for r in results:
            st.write(f"**Source:** {r['metadata'].get('id')} — Score: {r['score']:.3f}")
        prompt = build_prompt(results, query)
        with st.spinner("Calling LLM..."):
            try:
                ans = call_openai(prompt)
                st.subheader("Answer")
                st.write(ans)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
