# app.py
import os
import pickle
from typing import List, Tuple
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

# ===================== VISUAL =====================
st.set_page_config(page_title="LAVO - Reforma Tributária", page_icon="📄", layout="centered")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 900px;}
  .login-card {padding: 1.25rem; border-radius: 16px; background: #111827; border: 1px solid #374151;}
  .login-title {font-size: 1.2rem; margin-bottom: .5rem;}
</style>
""", unsafe_allow_html=True)

st.title("🧑‍🏫 LAVO - Especialista em Reforma Tributária")

# ===================== CONFIG =====================
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

# Chave da OpenAI (Streamlit Secrets tem prioridade)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
client = OpenAI(api_key=OPENAI_API_KEY)

# Usuários (nome -> senha) vindos dos Secrets
USERS = {
    "rafael souza": st.secrets.get("APP_PASS_RAFAEL", ""),
    "alex montu":   st.secrets.get("APP_PASS_ALEX",   ""),
}

INDEX_PATH = "index/faiss.index"
META_PATH  = "index/faiss_meta.pkl"

SYSTEM_PROMPT = """
Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group.
Fale SEMPRE em português do Brasil, com precisão técnica e didática.

PERSONA
- Seja objetiva, clara e traga exemplos contábeis e fiscais práticos.
- Tom didático, mas nunca diga que é “professora de cursinho”.

PERSONALIZAÇÃO
- Cumprimente o usuário usando o NOME do login, já fornecido no contexto.
- Use o nome ao longo da resposta quando fizer sentido.

ESCOPO
- Responda SOMENTE sobre Reforma Tributária (BR).
- Cite apenas leis, ECs, PECs, PLPs, pareceres e nomes de professores/relatores (sem links).
- Nunca mencione “arquivos/PDFs/slides/material/chunks/contexto”.

INCERTEZA
- Se não tiver certeza, diga: “Ainda estou estudando, mas logo aprendo e voltamos a falar.”

FORMATAÇÃO
- Quando fizer sentido, use:
  1) Resumo rápido (2–4 linhas).
  2) Detalhamento prático (bullets com regras, prazos, cálculos e exemplos).
  3) Referências normativas (ex.: EC 132/2023; PLP 68/2024).
- Valores: use vírgula (R$ 1.000,00) e mostre a conta: 18% de R$ 1.000,00 → R$ 180,00.
"""

# ===================== HELPERS =====================
def load_faiss_index(index_path=INDEX_PATH, meta_path=META_PATH):
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return None, []
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metas = pickle.load(f)
    return index, metas

def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(v.reshape(1, -1))
    return v

def retrieve(index, metas, query: str, k: int = 5):
    if index is None: return []
    q = embed_query(query).reshape(1, -1)
    D, I = index.search(q, k)
    hits = []
    for pos, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if idx < 0 or idx >= len(metas): continue
        hits.append((pos, score, metas[idx]))
    return hits

def answer_with_context(question: str, hits: List[Tuple[int, float, dict]], nome: str) -> str:
    contexto = "\n\n".join(f"[{rank}] {m['text_preview']}" for rank, score, m in hits)
    user_instruction = (
        f"Inicie a resposta cumprimentando a pessoa pelo nome exatamente assim: 'Olá, {nome}!'. "
        "Em seguida responda conforme o estilo e regras do sistema. "
        "Use apenas o conteúdo abaixo como base."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"{user_instruction}\n\n<contexto>\n{contexto}\n</contexto>\n\nPergunta: {question}"},
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0.0, max_tokens=700
    )
    return resp.choices[0].message.content

def welcome_message(nome: str) -> str:
    return (
        f"Olá, {nome}! 👋\n\n"
        "Sou a **LAVO**, especialista em Reforma Tributária da **Lavoratory Group**.  \n"
        "Seja muito bem-vindo! 🚀  \n\n"
        "Qual é a sua dúvida sobre a Reforma Tributária? Estou aqui para ajudar com clareza e objetividade."
    )

GREETING_WORDS = {
    "ola", "olá", "bom dia", "boa tarde", "boa noite", "oi", "hey", "hello"
}
def is_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    return (len(t.split()) <= 3) and any(w in t for w in GREETING_WORDS)

# ===================== LOGIN =====================
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🔒 Login</div>', unsafe_allow_html=True)
        user = st.text_input("Usuário (ex.: Rafael Souza)").strip()
        pwd  = st.text_input("Senha", type="password")
        if st.button("Entrar", use_container_width=True):
            key = (user or "").lower()
            if key in USERS and USERS[key] and USERS[key] == (pwd or ""):
                st.session_state.auth = True
                st.session_state.user_name = user or "Usuário"
                st.session_state.show_welcome = True  # <- mostra saudação pós-login
                st.success(f"Bem-vindo, {st.session_state.user_name}!")
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ===================== CARGA DO ÍNDICE =====================
nome = st.session_state.get("user_name", "colega")
index, metas = load_faiss_index()
if index is None:
    st.warning("⚠️ Nenhum índice encontrado. Gere `index/faiss.index` e `index/faiss_meta.pkl` via GitHub Actions a partir de `txts/`.")
    st.info("Após o Actions commitar os arquivos em `index/`, atualize a página.")
    st.stop()

st.caption(f"Base carregada • trechos: {len(metas)}")

# Saudação automática ao logar (uma vez)
if st.session_state.pop("show_welcome", False):
    with st.chat_message("assistant"):
        st.markdown(welcome_message(nome))

# ===================== CHAT =====================
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

q = st.chat_input("Faça sua pergunta para a LAVO…")
if q:
    st.session_state.history.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    # Se for uma saudação simples, responde institucionalmente
    if is_greeting(q):
        msg = welcome_message(nome)
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.history.append(("assistant", msg))
    else:
        with st.chat_message("assistant"):
            with st.spinner("Consultando…"):
                hits = retrieve(index, metas, q, k=5)
                ans = answer_with_context(q, hits, nome)
                st.markdown(ans)
                st.session_state.history.append(("assistant", ans))
