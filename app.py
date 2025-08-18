import os
import io
import json
import time
import zipfile
import pickle
import pathlib
import numpy as np
import faiss
import streamlit as st
from typing import List, Tuple
from openai import OpenAI

# =========================
# CONFIGURAÇÃO / CONSTANTES
# =========================
st.set_page_config(page_title="LAVO - Reforma Tributária", page_icon="📄", layout="wide")

# Diretórios dentro do container da nuvem
UPLOAD_DIR = pathlib.Path("uploads")
INDEX_DIR  = pathlib.Path("index")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# Modelos OpenAI (somente NUVEM)
EMBED_MODEL = "text-embedding-3-small"   # barato e bom
CHAT_MODEL  = "gpt-4o-mini"               # respostas do chat

# Preço aproximado de embeddings (USD por 1M tokens)
PRICE_PER_MTOK = 0.02

# Limites de chunk
MAX_CHARS_PER_CHUNK = 1800
CHUNK_OVERLAP       = 150

# =========================
# SECRETS / AUTENTICAÇÃO
# =========================
def get_secret(name: str, default: str = "") -> str:
    # prioriza st.secrets (nuvem). Se rodar local, pode cair em os.getenv.
    return st.secrets.get(name, os.getenv(name, default))

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
USERS = {
    "rafael souza": get_secret("APP_PASS_RAFAEL"),
    "alex montu":   get_secret("APP_PASS_ALEX"),
}

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# FUNÇÕES DE SUPORTE
# =========================
def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def estimate_cost_chars(total_chars: int) -> Tuple[int, float]:
    # ~4 chars ≈ 1 token
    est_tokens = max(1, total_chars // 4)
    usd = (est_tokens / 1_000_000) * PRICE_PER_MTOK
    return est_tokens, usd

def embed_batch(texts: List[str], model: str, client: OpenAI) -> np.ndarray:
    """Gera embeddings em lotes com retry/backoff."""
    vecs: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # retentativa simples
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                for d in resp.data:
                    vecs.append(d.embedding)
                break
            except Exception as e:
                time.sleep(1.5 * (attempt + 1))
                if attempt == 4:
                    raise e
        time.sleep(0.15)
    arr = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(arr)  # para usar IP como cosseno
    return arr

def build_index_from_txt_files(txt_paths: List[pathlib.Path]) -> Tuple[faiss.Index, list]:
    """Lê TXT do disco, chunk, calcula embeddings (OpenAI) e retorna (índice, metas)."""
    metas = []
    chunks = []
    for p in txt_paths:
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            content = p.read_text(encoding="latin-1", errors="ignore")
        for ch in chunk_text(content):
            metas.append({"source": p.name, "page": 1, "text_preview": ch[:240]})
            chunks.append(ch)

    if not chunks:
        raise RuntimeError("Nenhum texto útil encontrado nos .txt enviados.")

    total_chars = sum(len(c) for c in chunks)
    tokens, usd = estimate_cost_chars(total_chars)
    st.info(f"Estimativa de custo de embeddings: ~{tokens:,} tokens (~US${usd:,.2f}).")

    # Confirmação do usuário
    if not st.session_state.get("confirmed_embeddings", False):
        if st.button("✅ Confirmo gerar embeddings (OpenAI, custo estimado acima)"):
            st.session_state.confirmed_embeddings = True
        st.stop()

    with st.spinner("Gerando embeddings na nuvem (OpenAI)…"):
        emb = embed_batch(chunks, EMBED_MODEL, client)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, metas

def save_index(index: faiss.Index, metas: list):
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "metas.pkl", "wb") as f:
        pickle.dump(metas, f)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_chunks": index.ntotal,
        "embed_model": EMBED_MODEL,
        "index_type": "faiss.IndexFlatIP (normalized)",
        "chunk_chars": MAX_CHARS_PER_CHUNK,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    with open(INDEX_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def zip_index() -> bytes:
    """Compacta ./index em um ZIP em memória e retorna bytes."""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ["index.faiss", "metas.pkl", "manifest.json"]:
            fpath = INDEX_DIR / name
            if fpath.exists():
                zf.write(fpath, arcname=f"index/{name}")
    mem.seek(0)
    return mem.read()

def load_resources_or_error():
    idx = INDEX_DIR / "index.faiss"
    mta = INDEX_DIR / "metas.pkl"
    mft = INDEX_DIR / "manifest.json"
    if not (idx.exists() and mta.exists() and mft.exists()):
        st.error("⚠️ Nenhum índice encontrado. Vá na aba **Admin → Construir índice** para gerar um.")
        st.stop()
    index = faiss.read_index(str(idx))
    with open(mta, "rb") as f:
        metas = pickle.load(f)
    with open(mft, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return index, metas, manifest

def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def retrieve(index: faiss.Index, metas: list, query: str, k: int = 5):
    q = embed_query(query).reshape(1, -1)
    D, I = index.search(q, k)
    hits = []
    for pos, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if idx == -1: 
            continue
        m = metas[idx]
        hits.append((pos, score, m))
    return hits

def answer_with_context(question: str, hits: List[Tuple[int, float, dict]], nome: str) -> str:
    context_parts = []
    for rank, score, m in hits:
        context_parts.append(f"[{rank}] {m['text_preview']}")
    contexto = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": """Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group.
Sempre pergunte o nome da pessoa e use nas respostas.
Seja objetiva, clara e traga exemplos contábeis e fiscais práticos.
Comporte-se como professora de cursinho, mas nunca diga isso.
Cite só leis, PECs, pareceres e nomes de professores ou relatores.
Nunca mencione arquivos, PDFs, apresentações ou materiais de aula.
Se não souber, diga: “Ainda estou estudando, mas logo aprendo e voltamos a falar.”"""},
        {"role": "user", "content": f"Nome da pessoa: {nome}\n\n<contexto>\n{contexto}\n</contexto>\n\nPergunta: {question}"}
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=600,
    )
    return resp.choices[0].message.content

# =========================
# LOGIN
# =========================
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔒 Login")
    col1, col2 = st.columns(2)
    with col1:
        user = st.text_input("Usuário (ex.: Rafael Souza)").strip()
    with col2:
        password = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        u = user.lower()
        if u in USERS and USERS[u] == password:
            st.session_state.auth = True
            st.session_state.user_name = user
            st.success(f"Bem-vindo, {user}!")
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos")
    st.stop()

# =========================
# LAYOUT: TABS
# =========================
tab_admin, tab_chat = st.tabs(["🛠️ Admin (Construir índice)", "💬 Chat"])

# -------------------------
# ADMIN
# -------------------------
with tab_admin:
    st.subheader("Upload de .TXT e construção do índice (OpenAI embeddings)")

    # Upload múltiplo
    uploaded = st.file_uploader(
        "Envie seus arquivos .txt (pode mandar vários)",
        type=["txt"],
        accept_multiple_files=True
    )

    if uploaded:
        # Salvar arquivos enviados
        saved_paths = []
        for up in uploaded:
            safe_name = up.name.replace("/", "_")
            path = UPLOAD_DIR / safe_name
            path.write_bytes(up.read())
            saved_paths.append(path)
        st.success(f"{len(saved_paths)} arquivo(s) salvo(s) em /uploads.")

        # Botão para construir índice
        if st.button("🚀 Construir índice agora (OpenAI)"):
            st.session_state.confirmed_embeddings = False  # força confirmação de custo
            index, metas = build_index_from_txt_files(saved_paths)
            save_index(index, metas)
            st.success("Índice criado e salvo em ./index/ ✅")

    # Se já existir índice, oferecer download do zip
    if (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "metas.pkl").exists():
        st.divider()
        st.caption("Baixar o índice atual (para versionar ou backup)")
        data = zip_index()
        st.download_button(
            label="📦 Baixar index.zip",
            data=data,
            file_name="index.zip",
            mime="application/zip"
        )

# -------------------------
# CHAT
# -------------------------
with tab_chat:
    st.subheader("Converse com a LAVO")
    index, metas, manifest = load_resources_or_error()

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    q = st.chat_input("Digite sua pergunta para a LAVO…")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        st.session_state.history.append(("user", q))

        with st.chat_message("assistant"):
            with st.spinner("Consultando…"):
                hits = retrieve(index, metas, q, k=5)
                resp = answer_with_context(q, hits, st.session_state.get("user_name") or "colega")
                st.markdown(resp)
                st.session_state.history.append(("assistant", resp))
