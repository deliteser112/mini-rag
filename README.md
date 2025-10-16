# mini_rag

A minimal Retrieval-Augmented Generation (RAG) demo using:

- **Streamlit** for the web interface  
- **FAISS** for vector search  
- **LangChain** for document chunking and embeddings  
- **Groq LLM** as the backend language model  

## Installation

```bash
git clone https://github.com/deliteser112/mini_rag.git
cd mini_rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="sk-..."
streamlit run app.py
```
