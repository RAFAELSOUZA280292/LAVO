import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import List
import numpy as np
import faiss
import streamlit as st

# ------- OpenAI -------
from openai import OpenAI

# ========= CONFIG =========
INDEX_DIR = "index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "faiss_meta.pkl")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")
EMB_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = """
Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group.
Regras:
- Sempre cumprimente usando o NOME do usuário se disponível.
- Responda SOMENTE com base no <CONTEXTO> fornecido. Não use conhecimento externo.
- Cite leis/PECs/pareceres/pessoas apenas se aparecerem no CONTEXTO.
- Traga exemplos contábeis e fiscais práticos quando fizer sentido.
- Não mencione “PDF”, “arquivo”, “material” nem “chunk”.
- Se o CONTEXTO não trouxer a informação pedida, diga: “Ainda estou estudando, mas logo aprendo e voltamos a falar.”

Formato:
1) Saudação curta usando o nome
2) Resposta objetiva em 3–8 linhas
3) (se útil) bullets com passos ou números
4) (se houver no contexto) Referências normativas citando número/ano/artigo
"""

# ========= UTILS =========
def escape_currency(text: str) -> str:
    """Evita que Markdown interprete $ como LaTeX."""
    return text.replace("R$", "R\\$")

def fmt_moeda(valor: float) -> str:
    s = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R\\$ {s}"

# ========= LOGIN =========
def get_users_from_secrets():
    users = {}
    # defina seus usuários no Streamlit Secrets
    # [secrets] -> APP_USER_RAF="Rafael Souza", APP_PASS_RAF="..."
    for k in st.secrets:
        if k.startswith("APP_USER_"):
            suf = k.split("APP_USER_")[1]
            user_name = st.secrets[k]
            pass_key = f"APP_PASS_{suf}"
            if pass_key in st.secrets:
                users[user_name] = st.secrets[pass_key]
    return users

def login_screen() -> str:
    st.title("🔐 Login")
    st.caption("Acesso restrito · LAVO")
    user = st.text_input("Usuário (nome completo)", key="login_user")
    pwd = st.text_input("Senha", type="password", key="login_pwd")
    ok = st.button("Entrar")
    if ok:
        users = get_users_from_secrets()
        if user in users and pwd == users[user]:
            st.session_state.auth = True
            st.session_state.nome_usuario = user
            st.success("Autenticado!")
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos.")
    st.stop()

# ========= CARREGAR ÍNDICE =========
@dataclass
class Meta:
    source: str
    chunk_id: int
    text_preview: str

def load_index():
    if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
        st.warning(
            "⚠️ Nenhum índice encontrado. Gere `index/faiss.index` e `index/faiss_meta.pkl` via GitHub Actions a partir de `txts/`."
        )
        st.stop()

    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        metas: List[Meta] = pickle.load(f)

    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    return index, metas, manifest

# ========= EMBEDDINGS =========
def openai_client() -> OpenAI:
    # pega da Cloud (Secrets) ou env
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        st.error("OPENAI_API_KEY não configurada em Secrets/variáveis de ambiente.")
        st.stop()
    return OpenAI(api_key=api_key)

def embed_query(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMB_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

# ========= BUSCA HÍBRIDA =========
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except Exception:
    BM25_AVAILABLE = False

def prepare_bm25(metas: List[Meta]):
    if not BM25_AVAILABLE:
        return None, None
    corpus = [m.text_preview for m in metas]
    tokenized = [c.lower().split() for c in corpus]
    return BM25Okapi(tokenized), corpus

def rrf_fuse(lists: List[List[int]], k_rrf: int = 60, limit: int = 10) -> List[int]:
    scores = {}
    for lst in lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked][:limit]

def retrieve_hybrid(client: OpenAI, index, metas: List[Meta], bm25, corpus, query: str,
                    k_faiss: int = 12, k_bm25: int = 20, k_final: int = 8) -> List[int]:
    # FAISS
    q = embed_query(client, query).reshape(1, -1)
    _, I = index.search(q, k_faiss)
    faiss_ids = I[0].tolist()

    # BM25
    if bm25 is not None:
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        bm25_ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:k_bm25]
    else:
        bm25_ranked = []

    fused = rrf_fuse([faiss_ids, bm25_ranked], limit=max(k_faiss, k_bm25))
    return fused[:k_final]

def build_context(metas: List[Meta], ids: List[int], max_chars: int = 4500) -> str:
    parts = []
    used = set()
    total = 0
    for idx in ids:
        if idx in used:
            continue
        s = metas[idx].text_preview
        if total + len(s) > max_chars:
            break
        parts.append(s)
        total += len(s)
        used.add(idx)
    return "\n\n---\n\n".join(parts)

def answer_with_context(client: OpenAI, question: str, user_name: str, index, metas, bm25, corpus) -> str:
    doc_ids = retrieve_hybrid(client, index, metas, bm25, corpus, question, k_final=8)
    contexto = build_context(metas, doc_ids, max_chars=4500)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"NOME: {user_name}\n\n<CONTEXTO>\n{contexto}\n</CONTEXTO>\n\nPERGUNTA: {question}"}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=800,
    )
    text = resp.choices[0].message.content.strip()
    text = escape_currency(text)
    return text

# ========= UI =========
st.set_page_config(page_title="LAVO - Reforma Tributária", page_icon="📚", layout="centered")

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    login_screen()

# Carregar índice
index, metas, manifest = load_index()
bm25, corpus = prepare_bm25(metas)
client = openai_client()

# Header
st.markdown("## 🧠 LAVO - Especialista em Reforma Tributária")
st.caption(f"Base carregada • trechos: **{len(metas)}**")

# Histórico
if "history" not in st.session_state:
    st.session_state.history = []

# Render histórico
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# Entrada
user_q = st.chat_input("Faça sua pergunta para a LAVO...")
if user_q:
    with st.chat_message("user"):
        st.markdown(escape_currency(user_q))
    st.session_state.history.append(("user", escape_currency(user_q)))

    with st.chat_message("assistant"):
        with st.spinner("Consultando a base..."):
            nome = st.session_state.get("nome_usuario", "amigo")
            ans = answer_with_context(client, user_q, nome, index, metas, bm25, corpus)
            st.markdown(ans)
            st.session_state.history.append(("assistant", ans))
