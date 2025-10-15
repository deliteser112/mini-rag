from typing import List
from PyPDF2 import PdfReader
from pathlib import Path
from utils import clean_text, chunk_text

def extract_text(file_path: str) -> str:
    p = Path(file_path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text = []
        reader = PdfReader(str(p))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
        return clean_text("\n".join(text))
    else:
        return clean_text(p.read_text(encoding="utf-8", errors="ignore"))

def build_documents_for_index(file_paths: List[str], chunk_size: int = 1000, overlap: int = 200):
    docs = []
    for path in file_paths:
        text = extract_text(path)
        base = Path(path).name
        for i, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            docs.append({
                "id": f"{base}__chunk_{i}",
                "text": chunk,
                "metadata": {"source": base, "chunk": i}
            })
    return docs
