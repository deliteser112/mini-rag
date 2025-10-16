# llm.py

import os
import logging
from typing import List, Dict, Any, Tuple
from groq import Groq

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------
# Default model
# -------------------------------
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# -------------------------------
# Groq client singleton
# -------------------------------
_groq_client: Groq | None = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _groq_client = Groq(api_key=api_key)
        logging.info("✅ Groq client initialized.")
    return _groq_client


def build_messages(
    query: str,
    retrieved: List[Tuple[str, Dict[str, Any]]],
    system_prompt: str = "You are a helpful assistant. Use the context passages to answer."
) -> List[Dict[str, str]]:
    """
    Converts retrieved chunks + user query into Groq chat messages.
    
    retrieved: List of tuples (chunk_text, metadata)
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    if retrieved:
        context_lines = []
        for idx, (text, meta) in enumerate(retrieved):
            src = meta.get("source", "unknown")
            chunk_id = meta.get("chunk", "?")
            context_lines.append(f"[{idx}] (from {src} chunk {chunk_id}): {text}")
        context_block = "\n\n".join(context_lines)
        messages.append({"role": "system", "content": f"Reference passages:\n{context_block}"})
    
    messages.append({"role": "user", "content": query})
    return messages


# -------------------------------
# Call Groq LLM
# -------------------------------
def call_llm(
    query: str,
    retrieved: List[Tuple[str, Dict[str, Any]]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Sends the query + retrieved context to Groq chat API.
    Returns dict with 'answer', 'raw', 'usage', etc.
    """
    client = get_groq_client()
    messages = build_messages(query, retrieved)

    # Prepare documents payload (optional, enables citations)
    documents_payload = [{"text": text, "metadata": meta} for text, meta in retrieved]

    logging.info("Sending request to Groq API...")
    resp = client.chat.completions.create(
        messages=messages,
        model=model,
        documents=documents_payload,
        enable_citations=True,
        temperature=temperature
    )

    # Extract first choice
    choice = resp.choices[0]
    answer = choice.message.content

    result = {
        "answer": answer,
        "raw": resp,
        "usage": getattr(resp, "usage", None),
        "choice": choice
    }

    logging.info("✅ Response received from Groq API.")
    return result


# -------------------------------
# Quick test
# -------------------------------
if __name__ == "__main__":
    # Example test (replace with FAISS retrieval)
    dummy_retrieved = [
        ("Transformers are neural networks based on self-attention.", {"source": "GPT-2.pdf", "chunk": 0}),
        ("BERT uses bidirectional encoding for NLP tasks.", {"source": "BERT.pdf", "chunk": 1})
    ]
    query = "Explain the transformer architecture."
    response = call_llm(query, dummy_retrieved)
    print("Answer:", response["answer"])
