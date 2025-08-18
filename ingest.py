import os
import json
import pickle
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import faiss
from tqdm import tqdm
from pypdf import PdfReader
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurações
DATA_DIR = "pdfs"   # pasta onde ficam seus PDFs
INDEX_DIR = "index" # pasta onde vai salvar o índice
os.makedirs(INDEX_DIR, exist_ok=True)

@dataclass
class ChunkMeta:
    source: str
    page: int
    text_preview: str

def read_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    """Extrai texto de cada página do PDF"""
    pages = []
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            pages.append((i + 1, txt))
    return pages

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Divide texto em pedaços menores"""
    text = " ".join(text.split())  # remove espaços extras
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks

def embed_openai(texts: List[str]) -> np.ndarray:
    """Gera embeddings com a OpenAI"""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    vecs = []
    for i in range(0, len(texts), 100):  # processa em lotes
        batch = texts[i:i+100]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        for d in resp.data:
            vecs.append(d.embedding)
        time.sleep(0.2)
    arr = np.array(vecs, dtype="float32")
    faiss.normalize_L2(arr)
    return arr

def main():
    pdf_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_paths:
        print(f"⚠️ Nenhum PDF encontrado em '{DATA_DIR}'. Coloque os PDFs lá e rode de novo.")
        return

    chunks = []
    metas: List[ChunkMeta] = []

    print(f"📂 Lendo PDFs da pasta {DATA_DIR}...")
    for pdf in tqdm(pdf_paths, desc="Processando PDFs"):
        pages = read_pdf_text(pdf)
        for page_num, page_text in pages:
            page_chunks = chunk_text(page_text)
            for ch in page_chunks:
                preview = ch[:200].replace("\n", " ")
                metas.append(ChunkMeta(source=os.path.basename(pdf), page=page_num, text_preview=preview))
                chunks.append(ch)

    print(f"🧠 Gerando embeddings ({len(chunks)} trechos)...")
    emb = embed_openai(chunks)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # Salvar índice e metadados
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "metas.pkl"), "wb") as f:
        pickle.dump(metas, f)
    with open(os.path.join(INDEX_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"num_chunks": len(chunks)}, f, indent=2)

    print("✅ Index criado com sucesso em ./index/")

if __name__ == "__main__":
    main()