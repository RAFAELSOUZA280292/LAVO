# app.py
import os
import re
import json
import pickle
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------
# Configuração básica
# ------------------------------------------------------------
st.set_page_config(page_title="LAVO - Especialista em Reforma Tributária", page_icon="🧑‍🏫", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    st.error("Faltou definir OPENAI_API_KEY nos Secrets do Streamlit.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Usuários (nome de login -> senha)
USERS = {
    "Rafael Souza": st.secrets.get("APP_PASS_RAFASOUZA"),
    "Alex Montu": st.secrets.get("APP_PASS_ALEXMONTU"),
}

INDEX_DIR = "index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "faiss_meta.pkl")

SYSTEM_PROMPT = (
    "Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group. "
    "Regras de estilo IMPORTANTES:\n"
    "1) Use sempre Markdown simples (títulos, listas, negritos). **Nunca** use LaTeX, "
    "símbolos matemáticos tipo \\( \\) ou expressões formatadas.\n"
    "2) Seja objetiva, clara e traga exemplos contábeis e fiscais práticos com números formatados como R$ 1.234,56 e 12%.\n"
    "3) Cite somente leis, PECs, pareceres e nomes de professores/relatores. **Nunca** mencione arquivos, PDFs, anexos ou materiais de aula.\n"
    "4) Se não souber, diga: “Ainda estou estudando, mas logo aprendo e voltamos a falar.”\n"
    "5) Responda sempre em português do Brasil. Não use jargões desnecessários.\n"
    "6) Ao falar de ‘Split Payment’, ‘IBS’, ‘CBS’ etc., explique com listas e passos práticos.\n"
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_index():
    if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
        return None, None
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        metas = pickle.load(f)
    return index, metas

def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def retrieve(index, metas, query: str, k: int = 6):
    q = embed_query(query).reshape(1, -1)
    D, I = index.search(q, k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        if 0 <= idx < len(metas):
            m = metas[idx]
            hits.append({
                "score": float(score),
                "text": m.get("text_preview", m.get("text", ""))[:1200],
                "source": m.get("source", ""),
                "chunk": m.get("chunk_id", idx),
            })
    return hits

def build_context(hits):
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] {h['text']}")
    return "\n\n".join(parts)

_num_fix_regexes = [
    # Junta quebras dentro de números
    (re.compile(r"R\$\s*\n\s*"), "R$ "),
    (re.compile(r"(\d)\s*\n\s*(\d)"), r"\1\2"),
    # Remove espaços errados em vírgula de moeda/percentual
    (re.compile(r"(\d)\s*,\s*(\d{2})"), r"\1,\2"),
    (re.compile(r"(\d)\s*%\b"), r"\1%"),
    # Normaliza setas/bullets que quebram
    (re.compile(r"\s*→\s*"), " → "),
]

def sanitize_numbers(text: str) -> str:
    out = text
    for rgx, repl in _num_fix_regexes:
        out = rgx.sub(repl, out)
    # Evita duplicações tipo "R 100,00" -> "R$ 100,00"
    out = re.sub(r"\bR\s+(\d)", r"R$ \1", out)
    return out

def answer(question: str, user_name: str, index, metas):
    # Recupera contexto
    hits = retrieve(index, metas, question, k=6) if index is not None else []
    contexto = build_context(hits) if hits else ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Usuário: {user_name}\n\n"
                f"<contexto>\n{contexto}\n</contexto>\n\n"
                "Regras de formatação da resposta:\n"
                "- Cumprimente usando o nome do usuário.\n"
                "- Use subtítulos (###), listas com bullets e destaques em **negrito**.\n"
                "- Formate valores como R$ 1.000,00 e percentuais como 12%.\n"
                "- Não use LaTeX, nem símbolos matemáticos estranhos.\n\n"
                f"Pergunta: {question}"
            ),
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=900,
    )
    text = resp.choices[0].message.content
    return sanitize_numbers(text)

# ------------------------------------------------------------
# UI: Login
# ------------------------------------------------------------
if "auth" not in st.session_state:
    st.session_state.auth = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.auth:
    st.title("🔐 Login")
    user = st.selectbox("Usuário", list(USERS.keys()))
    pwd = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if USERS.get(user) and USERS[user] == pwd:
            st.session_state.auth = True
            st.session_state.user_name = user
            st.success(f"Bem-vindo, {user}!")
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos.")
    st.stop()

# ------------------------------------------------------------
# UI: App principal
# ------------------------------------------------------------
st.markdown(
    f"# 🧑‍🏫 LAVO - Especialista em Reforma Tributária\n"
    f"**Base carregada** • índice FAISS + textos • Usuário: **{st.session_state.user_name}**"
)

index, metas = load_index()
if index is None:
    st.warning(
        "⚠️ Nenhum índice encontrado. Gere `index/faiss.index` e `index/faiss_meta.pkl` "
        "(via GitHub Actions) a partir dos seus `.txt`. Depois atualize a página."
    )
    st.stop()

# Caixa de histórico do chat
if "history" not in st.session_state:
    st.session_state.history = []

# Mensagem de abertura (somente primeira vez)
if not st.session_state.history:
    welcome = sanitize_numbers(f"""
👋 Olá, **{st.session_state.user_name}**!

Sou a **LAVO**, especialista em Reforma Tributária da **Lavoratory Group**.
Posso te ajudar com **IBS, CBS, Split Payment, créditos, regime de apuração, regras de não-cumulatividade** e muito mais.

### Exemplos do que posso fazer
- Explicar **Split Payment** com um exemplo simples.
- Simular o cálculo de **IBS** e **CBS** para um valor (ex.: R$ 500,00) usando **listas e passos práticos**.
- Citar leis/PECs e apontar **riscos e cuidados** sem juridiquês.

Conte sua dúvida e vamos direto ao ponto. 🙂
""")
    st.session_state.history.append(("assistant", welcome))

# Render do histórico
for role, content in st.session_state.history:
    with st.chat_message("assistant" if role == "assistant" else "user"):
        st.markdown(content)

# Entrada do usuário
prompt = st.chat_input("Digite sua pergunta...")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando a base e preparando uma resposta…"):
            try:
                out = answer(prompt, st.session_state.user_name, index, metas)

                # Se detetar pergunta típica de cálculo, reforça formato didático
                if re.search(r"\b(calcul|cálcul|como calcular|IBS|CBS)\b", prompt, re.I):
                    # apenas garante markdown ‘limpo’
                    out = out.replace("## ", "### ")  # desce um nível para ficar uniforme

                st.markdown(out)
                st.session_state.history.append(("assistant", out))
            except Exception as e:
                st.error("Ocorreu um erro ao gerar a resposta.")
                st.exception(e)
