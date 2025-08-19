import os
import glob
import json
import pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import faiss
from openai import OpenAI

INDEX_DIR = "index"
TXT_DIR = "txts"
EMB_MODEL = "text-embedding-3-small"

FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "faiss_meta.pkl")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

@dataclass
class Meta:
    source: str
    chunk_id: int
    text_preview: str

def read_txts(txt_dir: str) -> List[Tuple[str, str]]:
    files = sorted(glob.glob(os.path.join(txt_dir, "**/*.txt"), recursive=True))
    items = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            items.append((fp, f.read()))
    return items

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())  # normaliza espaços
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_corpus(files: List[Tuple[str, str]]) -> Tuple[List[str], List[Meta]]:
    texts = []
    metas = []
    for src, content in files:
        chunks = chunk_text(content)
        for i, ch in enumerate(chunks):
            # preview curto pra manter pickle leve
            preview = ch[:1000]
            texts.append(ch)
            metas.append(Meta(source=os.path.relpath(src), chunk_id=i, text_preview=preview))
    return texts, metas

def embed_all(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    vecs = []
    BATCH = 100
    pbar = tqdm(range(0, len(texts), BATCH), desc="🔤 Gerando embeddings")
    for i in pbar:
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=model, input=batch)
        for item in resp.data:
            vecs.append(item.embedding)
    arr = np.array(vecs, dtype="float32")
    faiss.normalize_L2(arr)
    return arr

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não definido (Actions secret).")

    os.makedirs(INDEX_DIR, exist_ok=True)

    files = read_txts(TXT_DIR)
    if not files:
        raise RuntimeError("Nenhum .txt encontrado em txts/.")
    print(f"📂 {len(files)} arquivo(s) .txt encontrados.")

    texts, metas = build_corpus(files)
    print(f"🧠 Total de chunks: {len(texts)}")

    client = OpenAI(api_key=api_key)
    emb = embed_all(client, texts, EMB_MODEL)

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)

    manifest = {"emb_model": EMB_MODEL, "chunks": len(texts)}
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("✅ Index criado em index/")

if __name__ == "__main__":
    main()
