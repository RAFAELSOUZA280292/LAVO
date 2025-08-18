import os
import json
import pickle
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# 🔑 Carregar variáveis
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Credenciais vindas dos Secrets do Streamlit
USERS = {
    "rafa": os.getenv("APP_PASS_RAFA"),
    "rosa": os.getenv("APP_PASS_ROSA")
}

INDEX_DIR = "index"

SYSTEM_PROMPT = """
Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group.
Sempre pergunte o nome da pessoa e use nas respostas.
Seja objetiva, clara e traga exemplos contábeis e fiscais práticos.
Comporte-se como professora de cursinho, mas nunca diga isso.
Cite só leis, PECs, pareceres e nomes de professores ou relatores.
Nunca mencione arquivos, PDFs, apresentações ou materiais de aula.
Se não souber, diga: “Ainda estou estudando, mas logo aprendo e voltamos a falar.”
"""

# Configuração Streamlit
st.set_page_config(page_title="LAVO - Reforma Tributária", page_icon="📄")

# -------------------
# 🔐 LOGIN
# -------------------
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔒 Login")
    user = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if user in USERS and USERS[user] == password:
            st.session_state.auth = True
            st.success("Bem-vindo ao chat da LAVO!")
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos")
    st.stop()

# -------------------
# 📂 Carregar índice
# -------------------
def load_resources():
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    metas_path = os.path.join(INDEX_DIR, "metas.pkl")
    manifest_path = os.path.join(INDEX_DIR, "manifest.json")

    if not (os.path.exists(index_path) and os.path.exists(metas_path)):
        st.error("⚠️ Rode `python ingest.py` antes para criar o índice.")
        st.stop()

    index = faiss.read_index(index_path)
    with open(metas_path, "rb") as f:
        metas = pickle.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return index, metas, manifest

index, metas, manifest = load_resources()

def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def retrieve(query: str, k: int = 5):
    q = embed_query(query).reshape(1, -1)
    D, I = index.search(q, k)
    return list(zip(I[0].tolist(), D[0].tolist()))

def answer_with_context(question: str, hits):
    context_parts = []
    for rank, (idx, score) in enumerate(hits, start=1):
        m = metas[idx]
        snippet = m.text_preview
        context_parts.append(f"[{rank}] {snippet}")
    contexto = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<contexto>\n{contexto}\n</contexto>\n\nPergunta: {question}"}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=600,
    )
    return resp.choices[0].message.content

# -------------------
# 💬 Chat
# -------------------
st.title("📚 LAVO - Especialista em Reforma Tributária")
st.caption("Converse com a IA da Lavoratory Group sobre a Reforma Tributária.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

user_q = st.chat_input("Digite sua pergunta para a LAVO...")
if user_q:
    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.history.append(("user", user_q))

    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            hits = retrieve(user_q, k=5)
            text = answer_with_context(user_q, hits)
            st.markdown(text)
            st.session_state.history.append(("assistant", text))