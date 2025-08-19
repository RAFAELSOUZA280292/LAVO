# app.py
# LAVO - Especialista em Reforma Tributária (RAG + FAISS + OpenAI)
# Revisão: Hybrid Search (FAISS + BM25), metadados JSONL estáveis e citações
# Observação: requer index/faiss.index e index/metas.jsonl (ou faiss_meta.pkl legado)

import os
import json
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import faiss
import streamlit as st
from rank_bm25 import BM25Okapi

# ===== OpenAI (SDK v1)
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("Instale: pip install openai>=1.40.0")

# ----------------------------------------------------------------------
# 🔧 HOTFIX para deserializar faiss_meta.pkl legado
#     (o pickle procura por uma classe Meta no módulo em execução)
# ----------------------------------------------------------------------
try:
    from dataclasses import dataclass
    @dataclass
    class Meta:
        source: str
        chunk_id: int
        text_preview: str = ""
        text: str = ""
        title: str = ""
except Exception:
    pass

# -----------------------------
# Segredos / Config
# -----------------------------
def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

def _to_float(v: str, d: float) -> float:
    try:
        return float(v)
    except:
        return d

def _to_int(v: str, d: int) -> int:
    try:
        return int(v)
    except:
        return d

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY não encontrado nos Secrets.")
    st.stop()

CHAT_MODEL = _get_secret("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = _to_float(_get_secret("TEMPERATURE", "0.2"), 0.2)
MAX_TOKENS = _to_int(_get_secret("MAX_TOKENS", "1200"), 1200)

USERS: Dict[str, str] = {}
u1 = _get_secret("APP_USER_RAF", "").strip()
p1 = _get_secret("APP_PASS_RAF", "")
u2 = _get_secret("APP_USER_ALEX", "").strip()
p2 = _get_secret("APP_PASS_ALEX", "")
if u1 and p1:
    USERS[u1.lower()] = p1
if u2 and p2:
    USERS[u2.lower()] = p2

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Arquivos do índice
# -----------------------------
INDEX_DIR = "index"
FAISS_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_PKL = os.path.join(INDEX_DIR, "faiss_meta.pkl")       # legado
META_JSONL = os.path.join(INDEX_DIR, "metas.jsonl")        # novo estável
MANIFEST_JSON = os.path.join(INDEX_DIR, "manifest.json")   # info do índice

# -----------------------------
# Prompt
# -----------------------------
SYSTEM_PROMPT = (
    "Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group. "
    "Seja objetiva e clara. Quando fizer sentido, traga UM exemplo numérico simples "
    "com notação brasileira (R$ 1.234,56 e 12%). "
    "Cite leis/EC/LC apenas quando forem estritamente necessárias e de forma enxuta. "
    "Não invente referência legal. "
    "Nunca mencione arquivos internos (.txt, .pdf). "
    "Se a pergunta estiver vaga, peça UMA precisão curta. "
    "Evite listas padronizadas; responda no formato adequado ao que foi perguntado."
)

# -----------------------------
# Login
# -----------------------------
def _norm_name(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()

def login_box():
    st.title("🧑‍🏫 LAVO - Especialista em Reforma Tributária")
    with st.container():
        st.subheader("Login")
        nome = st.text_input("Nome (igual ao cadastro)", key="login_nome")
        senha = st.text_input("Senha", type="password", key="login_senha")
        if st.button("Entrar", type="primary"):
            ok = USERS.get(_norm_name(nome)) == (senha or "")
            if ok:
                st.session_state.auth = True
                st.session_state.user_name = " ".join((nome or "").split()).strip()
                st.success(f"Bem-vindo, {st.session_state.user_name}!")
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")
    st.stop()

# -----------------------------
# Utilidades
# -----------------------------
def _coerce_dict_list(obj_list) -> List[Dict[str, Any]]:
    out = []
    for x in obj_list or []:
        if isinstance(x, dict):
            out.append(x)
        else:
            d = {}
            for attr in ("text", "text_preview", "title", "source", "path", "chunk_id"):
                if hasattr(x, attr):
                    d[attr] = getattr(x, attr)
            if not d and hasattr(x, "__dict__"):
                try:
                    d = {k: v for k, v in x.__dict__.items() if not k.startswith("__")}
                except:
                    d = {}
            out.append(d)
    return out

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except:
                continue
    return items

def _chunks_count(metas): 
    return len(metas or [])

# -----------------------------
# Loaders (cacheados)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_index() -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    # 1) FAISS
    if not os.path.exists(FAISS_FILE):
        return None, [], {}
    try:
        idx = faiss.read_index(FAISS_FILE)
    except Exception as e:
        st.error(f"Falha ao carregar FAISS: {e}")
        return None, [], {}

    # 2) Metas JSONL (novo formato preferencial)
    metas: List[Dict[str, Any]] = []
    if os.path.exists(META_JSONL):
        try:
            metas = _read_jsonl(META_JSONL)
        except Exception as e:
            st.error(f"Falha ao ler metas.jsonl: {e}")
            metas = []

    # 3) Fallback para PKL legado
    if not metas and os.path.exists(META_PKL):
        try:
            with open(META_PKL, "rb") as f:
                raw = pickle.load(f)
            if isinstance(raw, list):
                metas = _coerce_dict_list(raw)
            else:
                metas = _coerce_dict_list([raw])
        except Exception as e:
            st.error(f"Falha ao carregar faiss_meta.pkl: {e}")
            metas = []

    # 4) Manifest
    manifest = {}
    if os.path.exists(MANIFEST_JSON):
        try:
            with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    return idx, metas, manifest

# -----------------------------
# Embeddings / Busca
# -----------------------------
def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def _faiss_search(index, query_vec: np.ndarray, k: int = 5):
    D, I = index.search(query_vec.reshape(1, -1), k)
    return D[0], I[0]

@st.cache_resource(show_spinner=False)
def _bm25_index(metas: List[Dict[str, Any]]):
    """Cria BM25 sobre os textos/previas para complementar a busca vetorial."""
    docs = []
    for m in metas:
        txt = str(m.get("text") or m.get("text_preview") or "")
        tokens = txt.lower().split()
        docs.append(tokens)
    return BM25Okapi(docs)

def _hybrid_rank(
    question: str,
    index,
    metas: List[Dict[str, Any]],
    k_faiss: int = 8,
    k_bm25: int = 12,
    top_k: int = 6,
) -> List[int]:
    """
    Combina FAISS (cosine) + BM25: normaliza scores e faz rerank linear simples.
    Retorna índices finais de 'metas'.
    """
    if not metas:
        return []

    # FAISS
    faiss_idxs = []
    faiss_scores = {}
    if index is not None:
        qv = embed_query(question)
        D, I = _faiss_search(index, qv, k=k_faiss)
        for score, idx in zip(D, I):
            if 0 <= idx < len(metas):
                faiss_idxs.append(idx)
                faiss_scores[idx] = float(score)

    # BM25
    bm25 = _bm25_index(metas)
    bm25_scores = bm25.get_scores(question.lower().split())
    bm25_top = np.argsort(bm25_scores)[::-1][:k_bm25].tolist()

    # União
    pool = set(faiss_idxs) | set(bm25_top)
    if not pool:
        return []

    # Normalização min-max
    def _norm(vals: Dict[int, float]):
        if not vals:
            return {}
        arr = np.array(list(vals.values()), dtype=float)
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmax - vmin < 1e-9:
            return {k: 0.0 for k in vals}
        return {k: (v - vmin) / (vmax - vmin) for k, v in vals.items()}

    f_full = {i: faiss_scores.get(i, 0.0) for i in pool}
    b_full = {i: float(bm25_scores[i]) for i in pool}

    f_n = _norm(f_full)
    b_n = _norm(b_full)

    combo = {i: 0.55 * f_n.get(i, 0.0) + 0.45 * b_n.get(i, 0.0) for i in pool}
    reranked = sorted(combo.items(), key=lambda x: x[1], reverse=True)
    final_idxs = [i for i, _ in reranked[:top_k]]
    return final_idxs

def _build_context(metas: List[Dict[str, Any]], idxs: List[int], max_chars=2800) -> Tuple[str, List[Dict[str, Any]]]:
    parts, total = [], 0
    used = []
    for rank, i in enumerate(idxs, start=1):
        if i < 0 or i >= len(metas):
            continue
        m = metas[i] or {}
        snippet = str(m.get("text") or m.get("text_preview") or "").replace("\u200b", " ")
        title = str(m.get("title", ""))[:200]
        source = str(m.get("source", ""))
        piece = f"[{rank}] {title}\n{snippet}\n"
        parts.append(piece)
        total += len(piece)
        used.append({
            "rank": rank,
            "title": title,
            "source": source,
            "chunk_id": m.get("chunk_id"),
            "preview": snippet[:400]
        })
        if total >= max_chars:
            break
    return "\n".join(parts).strip(), used

# -----------------------------
# Resposta (RAG)
# -----------------------------
def answer_with_rag(question: str, user_name: str, index, metas, top_k=6) -> Tuple[str, List[Dict[str, Any]]]:
    contexto = ""
    citations: List[Dict[str, Any]] = []
    if index is not None and metas:
        idxs = _hybrid_rank(question, index, metas, k_faiss=8, k_bm25=12, top_k=top_k)
        contexto, citations = _build_context(metas, idxs, max_chars=2800)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": (
             f"Usuário: {user_name}\n"
             + (f"<contexto>\n{contexto}\n</contexto>\n" if contexto else "")
             + f"Pergunta: {question}\n"
             "- Seja direto e útil; se couber, traga UM pequeno exemplo com números em notação brasileira.\n"
             "- Evite fórmulas que quebrem a renderização."
         )},
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip(), citations

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="LAVO - Reforma Tributária", page_icon="📄", layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    login_box()

user_name = st.session_state.get("user_name", "Usuário")

st.title("🧑‍🏫 LAVO - Especialista em Reforma Tributária")

index, metas, manifest = load_index()
if not index or not metas:
    st.warning(
        "⚠️ Base não encontrada. Gere `index/faiss.index` e `index/metas.jsonl` "
        "(ou `faiss_meta.pkl`) via GitHub Actions a partir dos `.txt`. "
        "Depois de commitados em `index/`, recarregue esta página."
    )
else:
    emb_model = manifest.get("emb_model", "desconhecido")
    st.caption(f"Base carregada • trechos: {_chunks_count(metas)} • modelo de embedding: {emb_model}")

# Histórico
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

user_q = st.chat_input("Pergunte algo sobre Reforma Tributária…")
if user_q:
    st.session_state.history.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            try:
                text, cites = answer_with_rag(user_q, user_name, index, metas, top_k=6)
            except Exception as e:
                text, cites = f"Desculpe, ocorreu um erro ao gerar a resposta. Detalhe: {e}", []

            st.markdown(text)
            if cites:
                with st.expander("🔎 Fontes utilizadas nesta resposta"):
                    for c in cites:
                        st.markdown(
                            f"- **[{c['rank']}]** *{c['title']}* — `{c['source']}` (chunk {c.get('chunk_id')})\n\n"
                            f"  > {c['preview']}…"
                        )

            st.session_state.history.append(("assistant", text))

st.caption("Dica: se quiser respostas com números, peça: “traga 1 exemplo com números”.")
