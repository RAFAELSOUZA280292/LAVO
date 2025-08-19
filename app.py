# app.py
import os
import json
import re
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import streamlit as st

# FAISS é opcional para a UI inicial; só exigimos quando o índice existir
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from openai import OpenAI

# ======================================================================================
# CONFIG GERAL
# ======================================================================================

st.set_page_config(page_title="LAVO - Especialista em Reforma Tributária", page_icon="🧑‍🏫")

# Carrega segredos (Streamlit Cloud)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY não encontrado em Secrets.")
    st.stop()

# Usuários (nome completo → senha)
# Exemplo em Secrets:
# APP_USERS = {"Rafael Souza":"Ra@15062017","Alex Montu":"Lavoratory@753"}
raw_users = st.secrets.get("APP_USERS", None)
if isinstance(raw_users, str):
    try:
        USERS: Dict[str, str] = json.loads(raw_users)
    except Exception:
        USERS = {}
elif isinstance(raw_users, dict):
    USERS = raw_users
else:
    USERS = {}

if not USERS:
    # fallback de emergência (evite em produção)
    USERS = {"Admin": "admin"}

INDEX_DIR = "index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(INDEX_DIR, "faiss_meta.pkl")

# Cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================================================
# PROMPT DA LAVO
# ======================================================================================

SYSTEM_PROMPT = """
Você é a **LAVO**, especialista em Reforma Tributária da Lavoratory Group.

Estilo:
- Sempre trate a pessoa pelo **nome** (se conhecido).
- Seja objetiva, clara e **didática**, como **professora de cursinho**, mas **nunca diga isso**.
- Quando útil, traga **exemplos contábeis e fiscais práticos** e **numéricos redondinhos**.
- Cite **apenas** leis, ECs, PECs, LCs, pareceres e nomes de professores/relatores. **Não cite PDFs, arquivos, anexos, slides**.
- Se a pergunta for vaga, faça **uma** pergunta de esclarecimento (no fim) e **já entregue** um esboço de resposta.
- **Não use modelos prontos/repetitivos** (“O que muda no dia a dia”, “Financeiro/Fiscal/TI/Compras/Comercial”…). Responda **sob medida**.
- Se **não souber**, diga: “Ainda estou estudando, mas logo aprendo e voltamos a falar.” (e diga **onde** você procuraria).
- Evite formatação quebrada de moeda (ex.: “R 100 , 00”). Use sempre **R$ 100,00**.
- Quando fizer contas, mostre de forma simples (poucas linhas) e apenas quando ajudar.

Contexto a seguir são **recortes normativos/explicativos**; use-os para fundamentar a resposta **sem mencionar que vieram de documentos**.
"""

# ======================================================================================
# UTILITÁRIOS
# ======================================================================================

def load_index() -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Carrega FAISS + metadados, se existirem.
    """
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH)):
        return None, []
    if faiss is None:
        st.error("FAISS não está disponível no ambiente.")
        st.stop()
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        metas = pickle.load(f)
    return index, metas


def embed_query(text: str) -> np.ndarray:
    """
    Gera embedding da consulta usando OpenAI.
    """
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    # Normalização melhora a busca de cosseno no FAISS
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def search(index, metas: List[Dict[str, Any]], query: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    Busca top-k trechos no FAISS.
    """
    if index is None or not metas:
        return []
    q = embed_query(query).reshape(1, -1).astype("float32")
    D, I = index.search(q, k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        if 0 <= idx < len(metas):
            m = metas[idx]
            # tentamos chaves comuns: 'text', 'text_preview', 'chunk'
            text = m.get("text") or m.get("text_preview") or m.get("chunk") or ""
            src = m.get("src") or m.get("source") or ""
            hits.append({"text": str(text), "score": float(score), "src": src})
    return hits


def build_messages(nome: str, pergunta: str, trechos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Monta as mensagens do chat com o contexto.
    """
    # Junta o contexto em blocos curtos
    contexto_parts = []
    for i, h in enumerate(trechos, start=1):
        snippet = h["text"].strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        contexto_parts.append(f"[{i}] {snippet}")

    contexto = "\n\n".join(contexto_parts) if contexto_parts else ""

    # Monta o USER com tags para separar contexto de pergunta
    user_content = []
    if nome:
        user_content.append(f"NOME: {nome}\n")
    if contexto:
        user_content.append("<contexto>\n" + contexto + "\n</contexto>\n")
    user_content.append("Pergunta do usuário:\n" + pergunta.strip())

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_content)},
    ]
    return messages


def postprocess(text: str) -> str:
    """
    Limpa problemas frequentes vindos do modelo (espaços entre dígitos, R $, etc).
    Mantém formatação do Streamlit (markdown).
    """
    s = text

    # Remove espaços entre dígitos (ex.: 1 000 -> 1000)
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)

    # Normaliza "R $" para "R$"
    s = s.replace("R $", "R$")
    s = re.sub(r"R\s+\$", "R$", s)

    # "R 100,00" -> "R$ 100,00"
    s = re.sub(r"\bR\s+(\d)", r"R$ \1", s)

    # Evita "R$100,00" sem espaço (opcional; estética)
    s = re.sub(r"R\$(\d)", r"R$ \1", s)

    # Remove espaços antes de vírgula e ponto
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"\s+\.", ".", s)

    # Remove espaços logo após "R$"
    s = re.sub(r"R\$\s+(\d)", r"R$ \1", s)

    return s.strip()


def is_greeting(text: str) -> bool:
    t = text.strip().lower()
    return t in {"oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "hey", "hi", "hello"}


def greet(nome: str) -> str:
    base = f"Olá, {nome}! " if nome else "Olá! "
    return base + (
        "Sou a LAVO, especialista em Reforma Tributária. "
        "Diga o tema e o contexto (empresa/ramo/valor aproximado) e eu te explico com exemplos práticos. "
        "Ex.: “Calcule IBS/CBS numa venda de R$ 12.500,00 (varejo)”, ou “Como funciona o Split Payment no varejo?”."
    )


# ======================================================================================
# UI - LOGIN
# ======================================================================================

def login_ui() -> str:
    st.title("🧑‍🏫 LAVO - Especialista em Reforma Tributária")

    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
        st.session_state.user_name = ""

    if not st.session_state.auth_ok:
        with st.form("login"):
            user = st.text_input("Nome (igual ao cadastro)", placeholder="Rafael Souza")
            pwd = st.text_input("Senha", type="password")
            ok = st.form_submit_button("Entrar")

        if ok:
            # Nome deve bater exatamente (case sensitive) para simplificar; ajuste se quiser
            if user in USERS and USERS[user] == pwd:
                st.session_state.auth_ok = True
                st.session_state.user_name = user
                st.success(f"Bem-vindo, {user}!")
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")
        st.stop()

    return st.session_state.user_name


# ======================================================================================
# APP
# ======================================================================================

def main():
    nome = login_ui()

    # Carrega índice (se existir)
    index, metas = load_index()
    total_trechos = len(metas) if metas else 0

    st.caption(f"Base carregada • trechos: {total_trechos}")

    # Histórico do chat
    if "chat" not in st.session_state:
        st.session_state.chat: List[Tuple[str, str]] = []
        # Mensagem de boas-vindas
        st.session_state.chat.append(("assistant", greet(nome)))

    # Render histórico
    for role, content in st.session_state.chat:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    pergunta = st.chat_input("Escreva sua pergunta para a LAVO…")
    if not pergunta:
        return

    # Mostra a pergunta
    st.session_state.chat.append(("user", pergunta))
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Se for apenas saudação, responda com boas-vindas inteligente
    if is_greeting(pergunta):
        resposta = greet(nome)
        resposta = postprocess(resposta)
        st.session_state.chat.append(("assistant", resposta))
        with st.chat_message("assistant"):
            st.markdown(resposta)
        return

    # Busca contexto (se houver índice)
    trechos = search(index, metas, pergunta, k=6) if index is not None else []

    # Monta mensagens
    messages = build_messages(nome=nome, pergunta=pergunta, trechos=trechos)

    # Chamada ao modelo
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=900,
        )
        raw = resp.choices[0].message.content or ""
        resposta = postprocess(raw)
    except Exception as e:
        resposta = f"Falhou ao consultar a IA: {e}"

    # Exibe
    st.session_state.chat.append(("assistant", resposta))
    with st.chat_message("assistant"):
        st.markdown(resposta)


if __name__ == "__main__":
    main()
