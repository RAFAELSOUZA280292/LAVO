# app.py
# LAVO - Especialista em Reforma Tributária (RAG + FAISS + OpenAI)

import os
import json
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import faiss
import streamlit as st

# ===== OpenAI (SDK v1)
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("Instale: pip install openai>=1.40.0")

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
    try: return float(v)
    except: return d

def _to_int(v: str, d: int) -> int:
    try: return int(v)
    except: return d

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
if u1 and p1: USERS[u1.lower()] = p1
if u2 and p2: USERS[u2.lower()] = p2

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Arquivos do índice
# -----------------------------
INDEX_DIR = "index"
FAISS_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_PKL  = os.path.join(INDEX_DIR, "faiss_meta.pkl")   # legado
META_JSON = os.path.join(INDEX_DIR, "manifest.json")    # fallback moderno

# -----------------------------
# Prompt
# -----------------------------
SYSTEM_PROMPT = (
    "Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group. "
    "Seja objetiva, clara e, quando fizer sentido, traga um pequeno exemplo numérico simples "
    "com notação brasileira (R$ 1.234,56 e 12%). "
    "Use linguagem acessível, cite leis/PECs/LCs apenas quando realmente necessárias; não invente. "
    "Nunca mencione arquivos internos (.txt, .pdf). "
    "Se a pergunta estiver vaga, peça uma pequena precisão. "
    "Evite listas padronizadas repetitivas; ajuste o formato ao que foi perguntado."
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
        nome  = st.text_input("Nome (igual ao cadastro)", key="login_nome")
        senha = st.text_input("Senha", type="password", key="login_senha")
        if st.button("Entrar", type="primary"):
            if USERS.get(_norm_name(nome)) == (senha or ""):
                st.session_state.auth = True
                st.session_state.user_name = " ".join((nome or "").split()).strip()
                st.success(f"Bem-vindo, {st.session_state.user_name}!")
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")
    st.stop()

# -----------------------------
# Index loader robusto
# -----------------------------
def _coerce_dict_list(obj_list) -> List[Dict[str, Any]]:
    """Converte itens não-dict para dict sem quebrar."""
    out = []
    for x in obj_list:
        if isinstance(x, dict):
            out.append(x)
        else:
            d = {}
            # tenta extrair atributos comuns
            for attr in ("text", "text_preview", "title", "source", "path"):
                if hasattr(x, attr):
                    d[attr] = getattr(x, attr)
            # fallback: __dict__
            if not d and hasattr(x, "__dict__"):
                try:
                    d = {k: v for k, v in x.__dict__.items() if k[:2] != "__"}
                except:
                    d = {}
            out.append(d)
    return out

@st.cache_resource(show_spinner=False)
def load_index() -> Tuple[Any, List[Dict[str, Any]]]:
    # 1) FAISS
    if not os.path.exists(FAISS_FILE):
        return None, []
    try:
        idx = faiss.read_index(FAISS_FILE)
    except Exception as e:
        st.error(f"Falha ao carregar FAISS: {e}")
        return None, []

    # 2) Metadados (pkl legado) -> tenta e converte
    metas: List[Dict[str, Any]] = []
    if os.path.exists(META_PKL):
        try:
            with open(META_PKL, "rb") as f:
                raw = pickle.load(f)
            # se vier lista de objetos, converte para dict
            if isinstance(raw, list):
                metas = _coerce_dict_list(raw)
            else:
                # pode ser dict com chave 'items', etc.
                if hasattr(raw, "items"):
                    metas = _coerce_dict_list(list(raw.values()))
                else:
                    metas = _coerce_dict_list([raw])
        except Exception as e:
            st.error(f"Falha ao carregar índice: {e}")
            metas = []

    # 3) Fallback moderno: manifest.json
    if not metas and os.path.exists(META_JSON):
        try:
            with open(META_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                metas = _coerce_dict_list(data)
            elif isinstance(data, dict) and "items" in data:
                metas = _coerce_dict_list(data["items"])
        except Exception as e:
            st.error(f"Falha ao ler manifest.json: {e}")

    return idx, metas

def _chunks_count(metas): return len(metas or [])

# -----------------------------
# Embeddings / Busca
# -----------------------------
def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def _search(index, vec: np.ndarray, k: int = 5):
    D, I = index.search(vec.reshape(1, -1), k)
    return D[0], I[0]

def _build_context(metas: List[Dict[str, Any]], idxs, max_chars=2800) -> str:
    parts, total = [], 0
    for rank, i in enumerate(idxs, start=1):
        if i < 0 or i >= len(metas): continue
        m = metas[i] or {}
        snippet = str(m.get("text") or m.get("text_preview") or "").replace("\u200b", " ")
        title   = str(m.get("title", ""))[:120]
        piece = f"[{rank}] {title}\n{snippet}\n"
        parts.append(piece); total += len(piece)
        if total >= max_chars: break
    return "\n".join(parts).strip()

# -----------------------------
# Resposta (RAG)
# -----------------------------
def answer_with_rag(question: str, user_name: str, index, metas, k=5) -> str:
    contexto = ""
    if index is not None and metas:
        qv = embed_query(question)
        _, I = _search(index, qv, k=k)
        contexto = _build_context(metas, I)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": (
             f"Usuário: {user_name}\n"
             + (f"<contexto>\n{contexto}\n</contexto>\n" if contexto else "")
             + f"Pergunta: {question}\n"
             "- Seja direto e útil; se couber, traga UM pequeno exemplo com números em notação brasileira.\n"
             "- Evite caracteres que quebrem renderização (nada de fórmulas especiais)."
         )},
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()

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

index, metas = load_index()
if not index or not metas:
    st.warning(
        "⚠️ Nenhum índice encontrado. Gere `index/faiss.index` e `index/faiss_meta.pkl` "
        "ou `index/manifest.json` via GitHub Actions a partir dos `.txt`. "
        "Depois de commitados em `index/`, recarregue esta página."
    )
else:
    st.caption(f"Base carregada • trechos: {_chunks_count(metas)}")

# Histórico
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

user_q = st.chat_input("Pergunte algo sobre Reforma Tributária…")
if user_q:
    st.session_state.history.append(("user", user_q))
    with st.chat_message("user"): st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            try:
                text = answer_with_rag(user_q, user_name, index, metas, k=5)
            except Exception as e:
                text = f"Desculpe, ocorreu um erro ao gerar a resposta. Detalhe: {e}"
            st.markdown(text)
            st.session_state.history.append(("assistant", text))

st.caption("Dica: se quiser respostas com mais exemplos, peça explicitamente “traga 1 exemplo com números”.")
