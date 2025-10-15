import re
from pathlib import Path

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def read_file_bytes(path: str) -> bytes:
    path = Path(path)
    return path.read_bytes()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        yield text[start:end]
        start += chunk_size - overlap
