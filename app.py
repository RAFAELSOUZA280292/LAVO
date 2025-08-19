# app.py
import os, json, pickle, re
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
EMBED_MODEL  = "text-embedding-3-small"
CHAT_MODEL   = os.getenv("CHAT_MODEL", "gpt-4o")       # modelo principal
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4o")     # re-ranking

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
client = OpenAI(api_key=OPENAI_API_KEY)

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
- **NUNCA quebre números, moedas (ex.: R$ 1.000,00), percentuais (ex.: 18%) ou siglas (ex.: IBS/CBS) com quebras de linha ou espaços no meio. Mantenha-os em linha única e contínua.**
"""

# ===================== SANITIZADOR DE FORMATAÇÃO =====================
def tidy_text(s: str) -> str:
    if not s:
        return s

    # Normalizar "R $", "R  $", "R\n$" -> "R$"
    s = re.sub(r'R\s*\$\s*', 'R$', s)

    # Remover quebras/espaços entre dígitos, pontos, vírgulas e % (ex.: "1  . 000 , 00" -> "1.000,00")
    s = re.sub(r'(?<=\d)[\s\u00A0]+(?=[\d\.,%])', '', s)
    s = re.sub(r'(?<=[\d\.,])[\s\u00A0]+(?=\d)', '', s)

    # Garantir espaço após "R$" quando seguido de número (ex.: "R$1000" -> "R$ 1000")
    s = re.sub(r'R\$(?=\d)', r'R$ ', s)

    # Colar siglas quebradas por espaços/linhas: I B S -> IBS, C B S -> CBS, etc.
    def join_acronyms(m):
        return m.group(0).replace(' ', '').replace('\n', '')
    s = re.sub(r'\b(?:[A-Z]\s+){1,}[A-Z]\b', join_acronyms, s)

    # Remover espaços extras ao redor de "/" em siglas compostas (IBS / CBS -> IBS/CBS)
    s = re.sub(r'\s*/\s*', '/', s)

    return s

# ===================== HELPERS DE BUSCA =====================
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

def retrieve_candidates(index, metas, query: str, top_k: int = 30):
    if index is None: return []
    q = embed_query(query).reshape(1, -1)
    D, I = index.search(q, top_k)

    q_tokens = set((query or "").lower().split())
    def kw_score(text: str):
        t = (text or "").lower()
        return sum(1 for w in q_tokens if w in t)

    hits = []
    for pos, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if idx < 0 or idx >= len(metas): continue
        m = metas[idx]
        hits.append((pos, float(score), m.get("text_preview", "")))
    hits.sort(key=lambda h: kw_score(h[2]), reverse=True)
    return hits[:10]

def llm_rerank(question: str, candidates: List[Tuple[int, float, str]]):
    if not candidates:
        return []
    bundle = [{"id": rid, "text": txt[:1600]} for rid, _, txt in candidates]
    prompt = (
        "Você é um reranker de trechos. Dada a PERGUNTA e uma lista numerada de CANDIDATOS, "
        "atribua uma nota de relevância de 0 a 5 (5 = responde diretamente). "
        "Retorne um JSON com a chave 'ranking' contendo uma lista ordenada por "
        "nota decrescente, com objetos {\"id\": <id>, \"score\": <0..5>, "
        "\"answer_hint\": \"frase CURTA com a informação central, se houver\"}.\n"
        "- Use somente o texto dos candidatos.\n"
        "- Se a pergunta começar com 'quem', privilegie nomes próprios claros."
    )
    content = {"pergunta": question, "candidatos": bundle}
    resp = client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[
            {"role": "system", "content": "Você é preciso, conciso e sempre retorna JSON válido."},
            {"role": "user", "content": prompt + "\n\n" + json.dumps(content, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        ranking = data.get("ranking", [])
        ranking.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        return ranking[:5]
    except Exception:
        return [{"id": rid, "score": 0} for rid, _, _ in candidates[:5]]

def build_context_from_ranking(candidates, ranking):
    id_to_text = {rid: txt for rid, _, txt in candidates}
    picked = []
    for item in ranking[:3]:
        rid = item.get("id")
        if rid in id_to_text:
            picked.append(id_to_text[rid])
    return "\n\n".join(picked)

def answer_with_context(question: str, index, metas, nome: str) -> str:
    candidates = retrieve_candidates(index, metas, question, top_k=30)
    ranking = llm_rerank(question, candidates)
    contexto = build_context_from_ranking(candidates, ranking)

    quem_mode = question.strip().lower().startswith("quem ")
    instr_quem = (
        "Se a pergunta for 'Quem ...?', responda de forma direta com o NOME encontrado no contexto, "
        "em uma linha, e finalize. Se não houver nome no contexto, diga: "
        "'Ainda estou estudando, mas logo aprendo e voltamos a falar.'"
    )
    user_instruction = (
        f"Inicie a resposta cumprimentando a pessoa pelo nome exatamente assim: 'Olá, {nome}!'. "
        "Em seguida responda conforme o estilo e regras do sistema. "
        f"{instr_quem if quem_mode else ''} "
        "Use apenas o conteúdo abaixo como base e não invente normas que não estejam nele."
    )
    if not contexto.strip():
        return f"Olá, {nome}! Ainda estou estudando, mas logo aprendo e voltamos a falar."

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"{user_instruction}\n\n<contexto>\n{contexto}\n</contexto>\n\nPergunta: {question}"},
        ],
        temperature=0.0,
        max_tokens=700,
    )
    return resp.choices[0].message.content

def welcome_message(nome: str) -> str:
    return (
        f"Olá, {nome}! 👋\n\n"
        "Sou a **LAVO**, especialista em Reforma Tributária da **Lavoratory Group**.  \n"
        "Seja muito bem-vindo! 🚀  \n\n"
        "Qual é a sua dúvida sobre a Reforma Tributária? Estou aqui para ajudar com clareza e objetividade."
    )

GREETING_WORDS = {"ola","olá","bom dia","boa tarde","boa noite","oi","hey","hello"}
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
                st.session_state.show_welcome = True
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

    if is_greeting(q):
        msg = welcome_message(nome)
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.history.append(("assistant", msg))
    else:
        with st.chat_message("assistant"):
            with st.spinner("Consultando…"):
                raw = answer_with_context(q, index, metas, nome)
                fixed = tidy_text(raw)  # <<<<<< CORREÇÃO DE FORMATAÇÃO AQUI
                st.markdown(fixed)
                st.session_state.history.append(("assistant", fixed))
