# mini_rag

A minimal Retrieval-Augmented Generation (RAG) demo using:

- **Streamlit** for the web interface  
- **FAISS** for vector search  
- **LangChain** for document chunking and embeddings  
- **Groq LLM** as the backend language model

  <img width="1411" height="797" alt="image" src="https://github.com/user-attachments/assets/d510195f-16a5-48bf-bf48-8eefdc5ee01b" />

  <img width="1391" height="807" alt="image" src="https://github.com/user-attachments/assets/07e245f6-d60d-494a-9605-c487241cda79" />

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
