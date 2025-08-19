# app.py
# LAVO - Especialista em Reforma Tributária (RAG + FAISS + OpenAI)
# Requisitos: streamlit, openai, faiss-cpu, numpy

import os
import json
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import faiss
import streamlit as st

# ==== OpenAI client (SDK v1.x)
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError(
        "Pacote 'openai' não encontrado. Instale: pip install openai>=1.40.0"
    )

# -----------------------------
# Utilidades de Config/Secrets
# -----------------------------
def _get_secret(key: str, default: str = "") -> str:
    # Prioriza st.secrets, cai para variáveis de ambiente
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


def _to_float(val: str, default: float) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _to_int(val: str, default: int) -> int:
    try:
        return int(val)
    except Exception:
        return default


# -----------------------------
# Carrega Config
# -----------------------------
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY não encontrado (verifique Secrets).")
    st.stop()

CHAT_MODEL = _get_secret("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = _to_float(_get_secret("TEMPERATURE", "0.2"), 0.2)
MAX_TOKENS = _to_int(_get_secret("MAX_TOKENS", "1200"), 1200)

# Credenciais (nome é case-insensitive; senha é case-sensitive)
USERS: Dict[str, str] = {}
user_raf = _get_secret("APP_USER_RAF", "").strip()
pass_raf = _get_secret("APP_PASS_RAF", "")
user_alex = _get_secret("APP_USER_ALEX", "").strip()
pass_alex = _get_secret("APP_PASS_ALEX", "")

if user_raf and pass_raf:
    USERS[user_raf.lower()] = pass_raf
if user_alex and pass_alex:
    USERS[user_alex.lower()] = pass_alex

# Cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Parâmetros de índice
# -----------------------------
INDEX_DIR = "index"
FAISS_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "faiss_meta.pkl")

# -----------------------------
# Prompt de Sistema
# -----------------------------
SYSTEM_PROMPT = (
    "Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group. "
    "Seja objetiva, clara e, quando fizer sentido, traga exemplos práticos numéricos simples, "
    "com notação brasileira (R$ 1.234,56) e percentuais (12%). "
    "Use linguagem acessível (professora de cursinho), mas não diga isso explicitamente. "
    "Cite leis/PECs/LCs/pareceres apenas quando forem realmente relevantes; não invente referências. "
    "Nunca mencione arquivos internos (.txt, .pdf). "
    "Se a pergunta estiver vaga, peça uma pequena precisão. "
    "Evite listas padrão repetitivas; ajuste o formato ao que foi perguntado."
)

# -----------------------------
# Funções de Login
# -----------------------------
def norm_name(s: str) -> str:
    # Normaliza espaços e case
    return " ".join((s or "").split()).strip().lower()


def login_box() -> str:
    st.title("🧑‍🏫 LAVO - Especialista em Reforma Tributária")
    with st.container():
        st.subheader("Login")
        nome = st.text_input("Nome (igual ao cadastro)", value="", key="login_nome")
        senha = st.text_input("Senha", value="", type="password", key="login_senha")
        if st.button("Entrar", type="primary"):
            nkey = norm_name(nome)
            ok_pass = USERS.get(nkey)
            if ok_pass and senha.strip() == ok_pass.strip():
                st.session_state["auth"] = True
                st.session_state["user_name"] = " ".join(nome.split()).strip()
                st.success(f"Bem-vindo, {st.session_state['user_name']}!")
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")
    st.stop()


# -----------------------------
# Recursos: carregar índice
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_index() -> Tuple[Any, List[Dict[str, Any]]]:
    """Carrega índice FAISS e metadados. Retorna (index, metas)."""
    if not (os.path.exists(FAISS_FILE) and os.path.exists(META_FILE)):
        return None, []
    try:
        idx = faiss.read_index(FAISS_FILE)
        with open(META_FILE, "rb") as f:
            metas = pickle.load(f)
        return idx, metas
    except Exception as e:
        st.error(f"Falha ao carregar índice: {e}")
        return None, []


def chunks_count(metas: List[Dict[str, Any]]) -> int:
    return len(metas or [])


# -----------------------------
# Embeddings & Busca
# -----------------------------
def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec


def search(index, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(query_vec.reshape(1, -1), k)
    return D[0], I[0]


def build_context(metas: List[Dict[str, Any]], idxs: np.ndarray, max_chars: int = 2800) -> str:
    parts = []
    total = 0
    for rank, i in enumerate(idxs, start=1):
        if i < 0 or i >= len(metas):
            continue
        m = metas[i]
        # garantimos textos “seguros” (sem fórmulas estranhas)
        snippet = str(m.get("text", m.get("text_preview", ""))).replace("\u200b", " ")
        title = str(m.get("title", ""))[:120]
        piece = f"[{rank}] {title}\n{snippet}\n"
        total += len(piece)
        parts.append(piece)
        if total >= max_chars:
            break
    return "\n".join(parts).strip()


# -----------------------------
# Chat / Geração de resposta
# -----------------------------
def answer_with_rag(
    question: str,
    user_name: str,
    index,
    metas: List[Dict[str, Any]],
    k: int = 5,
) -> str:
    # Recupera contexto (se houver índice)
    contexto = ""
    if index is not None and metas:
        qv = embed_query(question)
        _, I = search(index, qv, k=k)
        contexto = build_context(metas, I)

    # Mensagens
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Usuário: {user_name}\n"
                + (f"<contexto>\n{contexto}\n</contexto>\n" if contexto else "")
                + f"\nPergunta: {question}\n"
                "Regras de resposta:\n"
                "- Seja direta e útil; se couber, traga UM pequeno exemplo numérico em notação brasileira.\n"
                "- Não use caracteres de fórmula matemática que causem rendering estranho (use texto simples)."
            ),
        },
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
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    login_box()

user_name = st.session_state.get("user_name", "Usuário")

st.title("🧑‍🏫 LAVO - Especialista em Reforma Tributária")

# Carrega índice
index, metas = load_index()
if not index or not metas:
    st.warning(
        "⚠️ Nenhum índice encontrado. Gere `index/faiss.index` e `index/faiss_meta.pkl` via GitHub Actions "
        "a partir dos seus `.txt`. Depois de commitados em `index/`, recarregue esta página."
    )
else:
    st.caption(f"Base carregada • trechos: {chunks_count(metas)}")

# Histórico
if "history" not in st.session_state:
    st.session_state["history"] = []

# Render histórico
for role, content in st.session_state["history"]:
    with st.chat_message(role):
        st.markdown(content)

# Entrada do usuário
user_q = st.chat_input("Pergunte algo sobre Reforma Tributária…")
if user_q:
    st.session_state["history"].append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            try:
                text = answer_with_rag(user_q, user_name, index, metas, k=5)
            except Exception as e:
                text = (
                    "Desculpe, ocorreu um erro ao gerar a resposta. "
                    f"Detalhe técnico: {e}"
                )
            st.markdown(text)
            st.session_state["history"].append(("assistant", text))

# Rodapé discreto
st.caption(
    "Dica: se quiser respostas com mais exemplos, peça explicitamente “traga 1 exemplo com números”."
)
