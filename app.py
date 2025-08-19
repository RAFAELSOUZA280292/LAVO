# app.py
import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import List
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

# ===================== Config Básica =====================
st.set_page_config(page_title="LAVO - Reforma Tributária", page_icon="📚", layout="wide")

INDEX_DIR = "index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "faiss_meta.pkl")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")

EMB_MODEL = "text-embedding-3-small"
CHAT_MODEL = st.secrets.get("CHAT_MODEL", os.getenv("CHAT_MODEL", "gpt-4o"))
DEFAULT_TEMP = float(st.secrets.get("TEMPERATURE", os.getenv("TEMPERATURE", "0.3")))
DEFAULT_MAXTOK = int(st.secrets.get("MAX_TOKENS", os.getenv("MAX_TOKENS", "1400")))

SYSTEM_PROMPT = """
Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group.
- Responda como uma consultora sênior: clara, direta, precisa e prática.
- Adapte o tom e a estrutura à pergunta do usuário (sem respostas engessadas).
- Use apenas o <CONTEXTO>; não traga conhecimento externo, a menos que o modo híbrido esteja LIGADO (instrução do usuário indicará isso).
- Cite leis/PECs/pareceres/pessoas apenas se aparecerem no CONTEXTO.
- Traga exemplos contábeis/fiscais quando forem úteis.
- Nunca mencione “PDF”, “arquivo”, “material” ou “chunk”.
- Se não houver base suficiente no CONTEXTO e o modo híbrido estiver DESLIGADO, responda apenas:
  “Ainda estou estudando, mas logo aprendo e voltamos a falar.”
- Formate moeda como “R$ 1.000,00” e percentuais como “12%”, sem quebras de linha.
Nota: Para saudações simples (ex.: “oi”, “olá”, “bom dia”), responda com boas-vindas amigáveis sem necessidade de contexto.
"""

# ===================== Sidebar: knobs de conversa =====================
with st.sidebar:
    st.header("⚙️ Ajustes da conversa")
    tone = st.selectbox(
        "Tom da resposta",
        ["Natural", "Executivo (direto ao ponto)", "Professoral (exemplos)"],
        index=0
    )
    depth = st.select_slider(
        "Profundidade",
        options=["Curta", "Média", "Detalhada"],
        value="Média"
    )
    hybrid_mode = st.toggle("Modo híbrido (complementar além do CONTEXTO quando faltar)", value=False)

def style_booster(tone: str, depth: str, hybrid: bool) -> str:
    bits = []
    if tone == "Executivo (direto ao ponto)":
        bits.append("Seja concisa, focada em decisão e impacto prático. Evite prolixidade.")
    elif tone == "Professoral (exemplos)":
        bits.append("Inclua exemplos contábeis/fiscais claros com números quando ajudar a compreensão.")
    else:
        bits.append("Fale naturalmente como consultora sênior, ajustando detalhes à pergunta.")
    if depth == "Curta":
        bits.append("Responda em 5–8 linhas, sem detalhes desnecessários.")
    elif depth == "Detalhada":
        bits.append("Aprofunde com nuances práticas, mas sem enrolação.")
    if hybrid:
        bits.append(
            "Se o CONTEXTO não cobrir totalmente, complemente com conhecimento geral consolidado "
            "em trechos iniciados por 'Complemento geral:'."
        )
    else:
        bits.append("Use exclusivamente o CONTEXTO; não extrapole além dele.")
    return " ".join(bits)

# ===================== Helpers de Formatação =====================
_num_fix_regexes = [
    (re.compile(r"R\$\s*\n\s*"), "R$ "),
    (re.compile(r"(\d)\s*\n\s*(\d)"), r"\1\2"),
    (re.compile(r"(\d)\s*,\s*(\d{2})"), r"\1,\2"),
    (re.compile(r"(\d)\s*%\b"), r"\1%"),
    (re.compile(r"\s*→\s*"), " → "),
]

def sanitize_numbers(text: str) -> str:
    out = text
    for rgx, repl in _num_fix_regexes:
        out = rgx.sub(repl, out)
    out = re.sub(r"\bR\s+(\d)", r"R$ \1", out)
    return out

def escape_currency(text: str) -> str:
    return text.replace("R$", "R\\$")

# ===================== Login (via Secrets) =====================
def get_users_from_secrets() -> dict:
    users = {}
    for k in st.secrets:
        if k.startswith("APP_USER_"):
            suf = k.split("APP_USER_")[1]
            user_name = st.secrets[k]
            pass_key = f"APP_PASS_{suf}"
            if pass_key in st.secrets:
                users[user_name] = st.secrets[pass_key]
    return users

def login_screen():
    st.title("🔐 Login · LAVO")
    st.caption("Acesso restrito")
    user = st.text_input("Usuário (nome completo)")
    pwd = st.text_input("Senha", type="password")
    if st.button("Entrar", type="primary", use_container_width=True):
        users = get_users_from_secrets()
        if user in users and pwd == users[user]:
            st.session_state.auth = True
            st.session_state.nome_usuario = user
            st.success("Autenticado!")
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos.")
    st.stop()

if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    login_screen()

# ===================== OpenAI Client =====================
def openai_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        st.error("OPENAI_API_KEY não configurada em Secrets/variáveis de ambiente.")
        st.stop()
    return OpenAI(api_key=api_key)

client = openai_client()

# ===================== Index / Metas =====================
@dataclass
class Meta:
    source: str
    chunk_id: int
    text_preview: str

def load_index():
    if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
        st.warning("⚠️ Nenhum índice encontrado. Gere `index/faiss.index` e `index/faiss_meta.pkl` via GitHub Actions a partir de `txts/`.")
        st.stop()
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        metas: List[Meta] = pickle.load(f)
    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    return index, metas, manifest

index, metas, manifest = load_index()

# ===================== BM25 (híbrido com FAISS) =====================
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

bm25, corpus = prepare_bm25(metas)

# ===================== Embeddings / Busca =====================
def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMB_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def rrf_fuse(lists: List[List[int]], k_rrf: int = 60, limit: int = 10) -> List[int]:
    scores = {}
    for lst in lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked][:limit]

def retrieve_hybrid(query: str, k_faiss: int = 12, k_bm25: int = 20, k_final: int = 8) -> List[int]:
    # FAISS
    q = embed_query(query).reshape(1, -1)
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

def build_context(ids: List[int], max_chars: int = 4500) -> str:
    parts, used, total = [], set(), 0
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

# ===================== User message (com estilo da sidebar) =====================
def make_user_message(question: str, nome: str, contexto: str, style_hint: str, hybrid: bool) -> str:
    return (
        f"NOME: {nome}\n\n"
        f"<CONTEXTO>\n{contexto}\n</CONTEXTO>\n\n"
        f"{style_hint}\n\n"
        + (
            "O modo híbrido está LIGADO: se o CONTEXTO não cobrir totalmente, complemente com conhecimento geral "
            "somente em trechos iniciados por 'Complemento geral:'.\n\n"
            if hybrid else
            "O modo híbrido está DESLIGADO: use exclusivamente o CONTEXTO; se não houver base suficiente, use a frase padrão de incerteza.\n\n"
        )
        f"PERGUNTA: {question}"
    )

# ===================== Geração da Resposta (RAG) =====================
def answer_with_context(question: str, nome: str, tone: str, depth: str, hybrid: bool) -> str:
    try:
        doc_ids = retrieve_hybrid(question, k_final=8)
        contexto = build_context(doc_ids, max_chars=4500)
    except Exception:
        q = embed_query(question).reshape(1, -1)
        D, I = index.search(q, 6)
        texto_parts = []
        for idx in I[0].tolist():
            if 0 <= idx < len(metas):
                texto_parts.append(metas[idx].text_preview)
        contexto = "\n\n---\n\n".join(texto_parts)

    if not contexto.strip() and not hybrid:
        return f"Olá, {nome}! Ainda estou estudando, mas logo aprendo e voltamos a falar."

    style_hint = style_booster(tone, depth, hybrid)
    user_msg = make_user_message(question, nome, contexto, style_hint, hybrid)

    # Ajuste fino conforme knobs
    temperature = DEFAULT_TEMP + (0.1 if tone == "Professoral (exemplos)" else 0.0)
    max_tokens = DEFAULT_MAXTOK + (400 if depth == "Detalhada" else (-300 if depth == "Curta" else 0))

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content.strip()
    text = sanitize_numbers(text)
    text = escape_currency(text)
    return text

# ===================== Small talk / Ajuda (bypass RAG) =====================
SMALLTALK_RE = re.compile(r"\b(oi|olá|ola|e ?a[ií]|hello|hey|hi|bom dia|boa tarde|boa noite)\b", re.I)
HELP_RE = re.compile(r"\b(ajuda|help|como usar|o que você faz|capacidade|exemplos?)\b", re.I)

def is_greeting(q: str) -> bool:
    return bool(SMALLTALK_RE.search(q.strip()))

def is_help(q: str) -> bool:
    return bool(HELP_RE.search(q.strip()))

def is_empty_or_short(q: str) -> bool:
    return len(q.strip()) < 2

def reply_smalltalk(nome: str) -> str:
    return escape_currency(
        f"Olá, **{nome}**! Sou a **LAVO**. Posso ajudar com IBS, CBS, Split Payment, regimes, transição e apuração. "
        "Manda sua dúvida ou peça um exemplo prático."
    )

def reply_help(nome: str) -> str:
    return escape_currency(
        f"Claro, **{nome}**! Exemplos do que posso responder:\n"
        "- “Quais são as leis base da Reforma?”\n"
        "- “Explique Split Payment com um exemplo de R$ 1.000,00.”\n"
        "- “Como fica o crédito no novo sistema?”\n"
        "- “Quais os riscos operacionais para varejo?”\n"
        "Faça sua pergunta 🙂"
    )

# ===================== UI Principal =====================
st.markdown("## 🧠 LAVO - Especialista em Reforma Tributária")
st.caption(f"Base carregada • chunks: **{len(metas)}** • modelo: {CHAT_MODEL}")

if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    welcome = escape_currency(
        f"Olá, **{st.session_state.get('nome_usuario','amigo')}**! "
        "Sou a **LAVO**. Pergunte o que quiser sobre Reforma Tributária (IBS, CBS, Split Payment, regimes, transição etc.). "
        "Quando fizer sentido, trago exemplos práticos com números."
    )
    st.session_state.history.append(("assistant", welcome))

# Render histórico
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# Entrada do usuário
user_q = st.chat_input("Digite sua pergunta para a LAVO...")
if user_q:
    nome = st.session_state.get("nome_usuario", "amigo")
    with st.chat_message("user"):
        st.markdown(escape_currency(user_q))
    st.session_state.history.append(("user", escape_currency(user_q)))

    with st.chat_message("assistant"):
        with st.spinner("Consultando a base…"):
            try:
                if is_empty_or_short(user_q) or is_greeting(user_q):
                    ans = reply_smalltalk(nome)
                elif is_help(user_q):
                    ans = reply_help(nome)
                else:
                    ans = answer_with_context(user_q, nome, tone, depth, hybrid_mode)

                st.markdown(ans)
                st.session_state.history.append(("assistant", ans))
            except Exception as e:
                st.error("Ocorreu um erro ao gerar a resposta.")
                st.exception(e)
