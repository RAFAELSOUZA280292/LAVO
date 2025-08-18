import os, time, pathlib
from typing import List, Tuple
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

# =============== CONFIG GERAL ===============
st.set_page_config(page_title="LAVO - Reforma Tributária", page_icon="📄", layout="centered")

TXT_DIR = pathlib.Path("txts")               # <-- seus .txt no próprio repo
EMBED_MODEL = "text-embedding-3-small"      # OpenAI embeddings
CHAT_MODEL  = "gpt-4o-mini"                  # OpenAI chat

PRICE_PER_MTOK = 0.02                        # US$ por 1M tokens de embedding (aprox)
MAX_CHARS_PER_CHUNK = 1800
CHUNK_OVERLAP       = 150

# Limite duro de custo (em tokens) para embeddings no boot (~4 chars ≈ 1 token)
MAX_EMBED_TOKENS = int(
    st.secrets.get("MAX_EMBED_TOKENS", os.getenv("MAX_EMBED_TOKENS", "2000000"))
)  # padrão ~2M tokens (~US$0.04 aprox)

# Secrets / auth (tudo via Streamlit Secrets na nuvem)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
USERS = {
    "rafael souza": st.secrets.get("APP_PASS_RAFAEL", os.getenv("APP_PASS_RAFAEL", "")),
    "alex montu":   st.secrets.get("APP_PASS_ALEX",   os.getenv("APP_PASS_ALEX", "")),
}
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group.
Sempre pergunte o nome da pessoa e use nas respostas.
Seja objetiva, clara e traga exemplos contábeis e fiscais práticos.
Comporte-se como professora de cursinho, mas nunca diga isso.
Cite só leis, PECs, pareceres e nomes de professores ou relatores.
Nunca mencione arquivos, PDFs, apresentações ou materiais de aula.
Se não souber, diga: “Ainda estou estudando, mas logo aprendo e voltamos a falar.”
"""

# =============== FUNÇÕES ===============
def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join((text or "").split())
    if not text: return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n: break
        start = max(0, end - overlap)
    return chunks

def estimate_tokens_from_chars(total_chars: int) -> int:
    # ~4 chars ≈ 1 token
    return max(1, total_chars // 4)

def read_all_txts(txt_dir: pathlib.Path):
    files = sorted(txt_dir.rglob("*.txt"))
    if not files:
        raise RuntimeError("Nenhum .txt encontrado na pasta 'txts/'. Suba seus arquivos no GitHub.")
    metas, chunks = [], []
    for p in files:
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            content = p.read_text(encoding="latin-1", errors="ignore")
        for ch in chunk_text(content):
            metas.append({"source": p.name, "page": 1, "text_preview": ch[:240]})
            chunks.append(ch)
    if not chunks:
        raise RuntimeError("Os .txt foram lidos, mas não há conteúdo útil.")
    return metas, chunks

def embed_batch_openai(texts: List[str]) -> np.ndarray:
    vecs = []
    batch = 100
    for i in range(0, len(texts), batch):
        part = texts[i:i+batch]
        # retry simples
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=part)
                vecs.extend([d.embedding for d in resp.data])
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
                if attempt == 4:
                    raise
        time.sleep(0.1)
    arr = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(arr)  # cosseno via inner-product
    return arr

@st.cache_resource(show_spinner=True)
def build_index_cached() -> Tuple[faiss.Index, list, dict]:
    """Constrói o índice 1x por servidor e cacheia (memória da nuvem)."""
    metas, chunks = read_all_txts(TXT_DIR)

    total_chars = sum(len(c) for c in chunks)
    est_tokens = estimate_tokens_from_chars(total_chars)
    if est_tokens > MAX_EMBED_TOKENS:
        usd = (est_tokens / 1_000_000) * PRICE_PER_MTOK
        raise RuntimeError(
            f"Volume muito alto ({est_tokens:,} tokens ≈ US${usd:,.2f}). "
            f"Reduza os .txt ou aumente MAX_EMBED_TOKENS em Secrets."
        )

    emb = embed_batch_openai(chunks)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    manifest = {
        "num_chunks": len(chunks),
        "embed_model": EMBED_MODEL,
        "estimated_tokens": est_tokens,
        "estimated_cost_usd": round((est_tokens/1_000_000)*PRICE_PER_MTOK, 4),
    }
    return index, metas, manifest

def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(v.reshape(1, -1))
    return v

def retrieve(index: faiss.Index, metas: list, query: str, k: int = 5):
    q = embed_query(query).reshape(1, -1)
    D, I = index.search(q, k)
    hits = []
    for pos, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if idx == -1: continue
        m = metas[idx]
        hits.append((pos, score, m))
    return hits

def answer_with_context(question: str, hits: List[Tuple[int, float, dict]], nome: str) -> str:
    contexto = "\n\n".join(f"[{rank}] {m['text_preview']}" for rank, score, m in hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Nome da pessoa: {nome}\n\n<contexto>\n{contexto}\n</contexto>\n\nPergunta: {question}"}
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0.0, max_tokens=600
    )
    return resp.choices[0].message.content

# =============== LOGIN ===============
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔒 Login")
    user = st.text_input("Usuário (ex.: Rafael Souza)").strip()
    pwd  = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        ukey = (user or "").lower()
        if ukey in USERS and USERS[ukey] == (pwd or ""):
            st.session_state.auth = True
            st.session_state.user_name = user
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos")
    st.stop()

# =============== CHAT ÚNICO ===============
st.title("📚 LAVO — Especialista em Reforma Tributária")

with st.spinner("Preparando base de conhecimento (primeiro carregamento gera embeddings)…"):
    try:
        index, metas, manifest = build_index_cached()
    except Exception as e:
        st.error(f"Não foi possível preparar a base: {e}")
        st.stop()

st.caption(f"Base carregada • chunks: {manifest['num_chunks']} • "
           f"custo único de embeddings (estimado): ~US${manifest['estimated_cost_usd']:.2f}")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

q = st.chat_input("Digite sua pergunta para a LAVO…")
if q:
    with st.chat_message("user"): st.markdown(q)
    st.session_state.history.append(("user", q))

    with st.chat_message("assistant"):
        with st.spinner("Consultando…"):
            hits = retrieve(index, metas, q, k=5)
            nome = st.session_state.get("user_name") or "colega"
            ans = answer_with_context(q, hits, nome)
            st.markdown(ans)
            st.session_state.history.append(("assistant", ans))
