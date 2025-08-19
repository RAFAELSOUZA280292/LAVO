# app.py
import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import List, Tuple
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
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """
Você é a **LAVO**, especialista em Reforma Tributária da Lavoratory Group.

OBJETIVO
- Entregar respostas didáticas, completas e aplicáveis ao dia a dia das empresas, SEM enrolação.
- Sempre cumprir: saudação com o NOME do usuário; conteúdo objetivo + exemplos práticos + plano de ação + referências.

ESCOPO
- Responda SOMENTE com base no <CONTEXTO> fornecido. Não use conhecimento externo.
- Cite apenas leis, ECs/PECs, PLPs/LCs, pareceres e nomes de professores/relatores que APARECEM no CONTEXTO.
- Nunca mencione “PDF”, “arquivo”, “material”, “chunk” ou “base”.

ESTILO (professora de cursinho, sem falar isso)
- Linguagem simples, direta e segura.
- Dê EXEMPLOS CONTÁBEIS/FISCAIS PRÁTICOS (com valores, contas e impactos de fluxo de caixa).
- Use sempre Markdown simples (títulos ###, listas -, **negritos**). **Nunca** use LaTeX.

FORMATAÇÃO OBRIGATÓRIA DA RESPOSTA
1) Saudação: “Olá, {NOME}!”
2) **Resumo rápido:** 2–4 linhas no máximo.
3) **O que muda no dia a dia:** bullets com impactos operacionais (Financeiro, Fiscal, TI, Compras, Comercial).
4) **Exemplos práticos:** pelo menos 2; use números (ex.: R$ 100.000,00; 12%; 30 dias) e explique a conta.
5) **Próximos passos (plano de ação):** bullets curtos, priorizados (1–5).
6) **Referências normativas:** só as que aparecerem no CONTEXTO (ex.: EC 132/2023; LC 214/2025; Art. 31).

INCERTEZA
- Se o CONTEXTO não trouxer a informação pedida, responda apenas:
“Ainda estou estudando, mas logo aprendo e voltamos a falar.”

REGRAS DE NÚMEROS
- Formate moeda como “R$ 1.000,00” e percentuais como “12%”.
- Nunca quebre números em linhas diferentes.
"""

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
    # Evita que Markdown/LaTeX quebre "R$"
    return text.replace("R$", "R\\$")

def fmt_moeda(valor: float) -> str:
    s = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R\\$ {s}"

# ===================== Login (via Secrets) =====================
def get_users_from_secrets() -> dict:
    """
    Em Secrets (Streamlit Cloud), cadastre pares:
      APP_USER_RAF="Rafael Souza"
      APP_PASS_RAF="Ra@15062017"
      APP_USER_ALEX="Alex Montu"
      APP_PASS_ALEX="Lavoratory@753"
    """
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

# ===================== Prompt do Usuário Guiado =====================
def build_guided_user_instruction(question: str, nome: str, contexto: str) -> str:
    return (
        f"NOME: {nome}\n\n"
        f"<CONTEXTO>\n{contexto}\n</CONTEXTO>\n\n"
        "Produza a resposta seguindo EXATAMENTE o formato obrigatório descrito no system prompt. "
        "Use apenas informações presentes no CONTEXTO. Não invente. "
        "Se não houver base suficiente no CONTEXTO, responda com a frase padrão de incerteza.\n\n"
        f"PERGUNTA: {question}"
    )

# ===================== Geração da Resposta =====================
def answer_with_context(question: str, nome: str) -> str:
    try:
        doc_ids = retrieve_hybrid(question, k_final=8)
        contexto = build_context(doc_ids, max_chars=4500)
    except Exception:
        # fallback simples, caso BM25 não esteja disponível
        q = embed_query(question).reshape(1, -1)
        D, I = index.search(q, 6)
        texto_parts = []
        for idx in I[0].tolist():
            if 0 <= idx < len(metas):
                texto_parts.append(metas[idx].text_preview)
        contexto = "\n\n---\n\n".join(texto_parts)

    if not contexto.strip():
        return f"Olá, {nome}! Ainda estou estudando, mas logo aprendo e voltamos a falar."

    user_instruction = build_guided_user_instruction(question, nome, contexto)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_instruction},
        ],
        temperature=0.1,
        max_tokens=1100,
    )
    text = resp.choices[0].message.content.strip()
    text = sanitize_numbers(text)
    text = escape_currency(text)
    return text

# ===================== UI Principal =====================
st.markdown("## 🧠 LAVO - Especialista em Reforma Tributária")
st.caption(f"Base carregada • chunks: **{len(metas)}** • modelo: {CHAT_MODEL}")

if "history" not in st.session_state:
    st.session_state.history = []

# Mensagem de abertura (uma vez)
if not st.session_state.history:
    welcome = escape_currency(f"""
👋 Olá, **{st.session_state.get('nome_usuario','amigo')}**!

Sou a **LAVO**, especialista em Reforma Tributária da **Lavoratory Group**.
Posso te ajudar com **IBS, CBS, Split Payment, regimes especiais/favorecidos, cashback, transição e apuração**.

### Exemplos do que posso fazer
- Explicar **Split Payment** com um exemplo de fluxo de caixa.
- Simular cálculo de **IBS/CBS** em um caso simples (ex.: R$ 500,00; 12%/6%).
- Sugerir **próximos passos** para adequar ERP, contratos e governança.

Manda a sua dúvida e eu respondo no formato didático.
""")
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
        with st.spinner("Consultando a base e montando a resposta…"):
            try:
                ans = answer_with_context(user_q, nome)
                st.markdown(ans)
                st.session_state.history.append(("assistant", ans))
            except Exception as e:
                st.error("Ocorreu um erro ao gerar a resposta.")
                st.exception(e)
