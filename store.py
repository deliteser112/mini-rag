import numpy as np
import faiss
import os
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

class VectorStore:
    def __init__(self, dim: int = None, index_path: str = "faiss_index/index.faiss", meta_path: str = "faiss_index/metadata.json"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.emb_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.index = None
        self.metadata = []
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self._load()
        else:
            self.index = None
            self.dim = None

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.dim = self.index.d

    def embed_texts(self, texts: List[str]):
        return self.emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def add_documents(self, docs: List[Dict]):
        texts = [d["text"] for d in docs]
        embeddings = self.embed_texts(texts).astype("float32")
        n, d = embeddings.shape
        if self.index is None:
            self.dim = d
            index = faiss.IndexFlatIP(self.dim)
            self.index = index
        faiss.normalize_L2(embeddings)
        for dmeta in docs:
            self.metadata.append({"id": dmeta["id"], **(dmeta.get("metadata") or {})})
        self.index.add(embeddings)
        self._save()

    def search(self, query: str, top_k: int = 5):
        q_emb = self.embed_texts([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append({"score": float(score), "metadata": self.metadata[idx]})
        return results
