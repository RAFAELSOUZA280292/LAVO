import os
from typing import List, Tuple
import streamlit as st
from openai import OpenAI
import faiss, pickle
import numpy as np

# ==========================
# CONFIGURAÇÕES
# ==========================
st.set_page_config(page_title="LAVO - Reforma Tributária", page_icon="📄", layout="centered")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
client = OpenAI(api_key=OPENAI_API_KEY)

# Usuários e senhas vindos dos Secrets
USERS = {
    "rafael souza": st.secrets.get("APP_PASS_RAFAEL", ""),
    "alex montu":   st.secrets.get("APP_PASS_ALEX",   ""),
}

SYSTEM_PROMPT = """
Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group.
Fale SEMPRE em português do Brasil, com precisão técnica e didática.

PERSONA E ESTILO
- Seja objetiva, clara e traga exemplos contábeis e fiscais práticos.
- Comporte-se como professora de cursinho (tom didático), mas nunca diga isso.
- Mantenha respostas concisas, evitando jargões desnecessários.
- Sempre que possível, organize em tópicos curtos.

PERSONALIZAÇÃO
- Sempre cumprimente o usuário usando o NOME obtido no login (ex.: “Olá, Rafael Souza!”).
- Use o nome ao longo da conversa.

ESCOPO E FONTES
- Responda SOMENTE sobre Reforma Tributária (BR): IVA/IBS/CBS, transição, regimes, não cumulatividade, créditos, alíquotas, repartição de receitas, Comitê Gestor etc.
- Baseie-se APENAS no conteúdo fornecido pelo sistema/assistente (contexto recuperado), sem inventar informações externas.
- Cite somente leis, PECs, ECs, PLPs, pareceres e nomes de professores/relatores (sem links).
- Nunca mencione “arquivos”, “PDFs”, “slides”, “materiais de aula”, “chunks” ou “contexto recuperado”.

INCERTEZA E LIMITES
- Se a informação não estiver no contexto ou você não tiver certeza, diga:
  “Ainda estou estudando, mas logo aprendo e voltamos a falar.”
- Se o pedido fugir do escopo, informe que o tema foge da sua atuação e redirecione para Reforma Tributária.

FORMATAÇÃO DA RESPOSTA
- Estruture, quando fizer sentido:
  1) **Resumo rápido** (2–4 linhas).
  2) **Detalhamento prático** (bullets com regras, prazos, cálculos e exemplos).
  3) **Referências normativas** (ex.: EC 132/2023; PLP 68/2024; Parecer XYZ).

NÚMEROS E VALORES
- Sempre escreva valores em reais com vírgula (ex.: R$ 1.000,00).
- Sempre destaque o percentual (ex.: 18%).
- Em exemplos numéricos, mostre a conta completa.
  Ex.: “Produto de R$ 1.000,00 com alíquota de 18% → ICMS de R$ 180,00”.
"""

# ==========================
# FAISS: carregar índice
# ==========================
def load_faiss_index(index_path="faiss.index", meta_path="faiss_meta.pkl"):
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
    if index is None:
        return []
    q = embed_query(query).reshape(1, -1)
    D, I = index.search(q, k)
    hits = []
    for pos, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if idx < 0 or idx >= len(metas):
            continue
        hits.append((pos, score, metas[idx]))
    return hits

def answer_with_context(question: str, hits: List[Tuple[int, float, dict]], nome: str) -> str:
    contexto = "\n\n".join(f"[{rank}] {m['text_preview']}" for rank, score, m in hits)
    user_instruction = (
        f"Inicie a resposta cumprimentando a pessoa pelo nome exatamente assim: 'Olá, {nome}!'. "
        "Em seguida responda conforme o estilo e regras do sistema.\n"
        "Use apenas o conteúdo abaixo como base."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"{user_instruction}\n\n<contexto>\n{contexto}\n</contexto>\n\nPergunta: {question}"},
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0.0, max_tokens=600
    )
    return resp.choices[0].message.content

# ==========================
# LOGIN (usuário + senha)
# ==========================
if "auth" not in st.session_state:
    st.session_state.auth = False

st.title("🧑‍🏫 LAVO - Especialista em Reforma Tributária")

if not st.session_state.auth:
    st.subheader("🔒 Login")
    col1, col2 = st.columns(2)
    with col1:
        user_input = st.text_input("Usuário (ex.: Rafael Souza)").strip()
    with col2:
        pwd_input  = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        key = (user_input or "").lower()
        if key in USERS and USERS[key] and USERS[key] == (pwd_input or ""):
            st.session_state.auth = True
            st.session_state.user_name = user_input.strip() or "Usuário"
            st.success(f"Bem-vindo, {st.session_state.user_name}!")
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos.")
    st.stop()

# ==========================
# CHAT
# ==========================
nome = st.session_state.get("user_name", "colega")

index, metas = load_faiss_index()
if index is None:
    st.warning("⚠️ Nenhum índice encontrado. Gere `faiss.index` e `faiss_meta.pkl` (via GitHub Actions) a partir dos seus .txt.")
    st.info("Depois do índice no repositório, basta atualizar a página que o chat fica disponível.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

q = st.chat_input("Digite sua pergunta para a LAVO…")
if q:
    st.session_state.history.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Consultando…"):
            hits = retrieve(index, metas, q, k=5)
            ans = answer_with_context(q, hits, nome)
            st.markdown(ans)
            st.session_state.history.append(("assistant", ans))
