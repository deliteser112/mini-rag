import os
from typing import List, Dict
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def build_prompt(retrieved: List[Dict], query: str) -> str:
    contexts = []
    for i, r in enumerate(retrieved):
        text = r.get("text") or r.get("metadata", {}).get("text") or "<no-text>"
        src = r.get("metadata", {}).get("source", f"chunk_{i}")
        contexts.append(f"Source [{src}] (score={r.get('score',0):.3f}):\n{text}\n")
    context_block = "\n---\n".join(contexts)
    prompt = f"""You are a helpful assistant. Use only the provided sources to answer the user's question.

SOURCES:
{context_block}

QUESTION:
{query}

ANSWER:"""
    return prompt

def call_openai(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 512):
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not found.")
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
