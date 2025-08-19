# build_index.py
import os, time, pathlib, json
from typing import List, Dict
import numpy as np
import faiss, pickle
from openai import OpenAI

TXT_DIR = pathlib.Path("txts")
OUT_DIR = pathlib.Path("index")
OUT_DIR.mkdir(exist_ok=True)

INDEX_PATH = OUT_DIR / "faiss.index"
META_PATH  = OUT_DIR / "faiss_meta.pkl"
MANIFEST   = OUT_DIR / "manifest.json"

EMBED_MODEL = "text-embedding-3-small"

# chunks maiores para aparecer nome/fato no contexto
MAX_CHARS_PER_CHUNK = 1800
CHUNK_OVERLAP = 200

def read_txts() -> List[Dict]:
    files = sorted(TXT_DIR.rglob("*.txt"))
    if not files:
        raise RuntimeError("Nenhum .txt encontrado em txts/.")
    items = []
    for p in files:
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            content = p.read_text(encoding="latin-1", errors="ignore")
        content = " ".join(content.split())
        i, n = 0, len(content)
        while i < n:
            j = min(i + MAX_CHARS_PER_CHUNK, n)
            chunk = content[i:j]
            items.append({"file": p.name, "text": chunk})
            if j == n: break
            i = max(0, j - CHUNK_OVERLAP)
    return items

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    vecs = []
    B = 100
    for k in range(0, len(texts), B):
        batch = texts[k:k+B]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                vecs.extend([d.embedding for d in resp.data])
                break
            except Exception:
                time.sleep(1.5*(attempt+1))
                if attempt == 4:
                    raise
        time.sleep(0.1)
    arr = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(arr)  # cos-sim via produto interno
    return arr

def main():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não definido (Actions secret).")
    client = OpenAI(api_key=api_key)

    items = read_txts()
    texts = [it["text"] for it in items]
    # GUARDE um preview LONGO (até 1600 chars) — é isso que vai para o LLM
    metas = [{"file": it["file"], "text_preview": it["text"][:1600]} for it in items]

    emb = embed_texts(client, texts)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)

    MANIFEST.write_text(json.dumps({
        "num_chunks": len(texts),
        "embed_model": EMBED_MODEL,
        "dim": int(emb.shape[1]),
        "max_chars_per_chunk": MAX_CHARS_PER_CHUNK,
        "chunk_overlap": CHUNK_OVERLAP,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Index salvo em {INDEX_PATH} e {META_PATH}. Chunks: {len(texts)} | Dim: {emb.shape[1]}")

if __name__ == "__main__":
    main()
