# llm.py

import os
from typing import List, Dict, Any, Tuple
from groq import Groq
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Default model — you might make this configurable via config.py or environment
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Initialize client once
_groq_client = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def build_messages(
    query: str,
    retrieved: List[Tuple[str, Dict[str, Any]]],
    system_prompt: str = "You are a helpful assistant. Use provided context passages to answer."
) -> List[Dict[str, str]]:
    """
    Build a list of message dicts for Groq chat completion.
    retrieved: list of tuples (chunk_text, metadata)
    """
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    
    # Add passages as “context” message
    # You can combine them into one message or multiple messages
    context_lines = []
    for idx, (text, meta) in enumerate(retrieved):
        src = meta.get("source", "unknown")
        chunk_id = meta.get("chunk", None)
        context_lines.append(f"[{idx}] (from {src} chunk {chunk_id}): {text}")
    context_block = "\n\n".join(context_lines)
    messages.append({
        "role": "system",
        "content": f"Here are some reference passages:\n{context_block}"
    })
    
    # Final user message
    messages.append({"role": "user", "content": query})
    return messages


def call_llm(
    query: str,
    retrieved: List[Tuple[str, Dict[str, Any]]],
    top_k: int = 3,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Call Groq chat completion API with the query + retrieved context.
    Returns dict with keys: "answer", "choices", "usage", etc.
    """
    client = get_groq_client()
    # Build messages
    messages = build_messages(query, retrieved)
    
    # Also build the `documents` payload Groq expects for context
    # Each document can be dict with text + metadata
    documents_payload = []
    for (text, meta) in retrieved:
        doc = {
            "text": text,
            "metadata": meta
        }
        documents_payload.append(doc)
    
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
    content = choice.message.content
    # You might also parse citations etc.
    result = {
        "answer": content,
        "raw": resp,
        "usage": resp.usage if hasattr(resp, "usage") else None,
        "choice": choice
    }
    return result
