# app.py
# LAVO - Especialista em Reforma Tributária (RAG + FAISS + BM25 + Context-First Engine)
# Novidade: âncora de seção — se a pergunta bater com um título/termo do contexto,
#           a resposta sai diretamente do trecho correspondente (sem passar pelo modelo).

import os
import re
import json
import pickle
from typing import List, Tuple, Dict, Any, Optional
from datetime import date, datetime

import numpy as np
import faiss
import streamlit as st
from rank_bm25 import BM25Okapi

# ===== OpenAI (SDK v1)
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("Instale: pip install openai>=1.40.0")

# ----------------------------------------------------------------------
# 🔧 HOTFIX para deserializar faiss_meta.pkl legado
# ----------------------------------------------------------------------
try:
    from dataclasses import dataclass
    @dataclass
    class Meta:
        source: str
        chunk_id: int
        text_preview: str = ""
        text: str = ""
        title: str = ""
except Exception:
    pass

# -----------------------------
# Segredos / Config
# -----------------------------
def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

def _to_float(v: str, d: float) -> float:
    try: return float(v)
    except: return d

def _to_int(v: str, d: int) -> int:
    try: return int(v)
    except: return d

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY não encontrado nos Secrets.")
    st.stop()

CHAT_MODEL = _get_secret("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = _to_float(_get_secret("TEMPERATURE", "0.2"), 0.2)
MAX_TOKENS = _to_int(_get_secret("MAX_TOKENS", "1200"), 1200)

USERS: Dict[str, str] = {}
u1 = _get_secret("APP_USER_RAF", "").strip()
p1 = _get_secret("APP_PASS_RAF", "")
u2 = _get_secret("APP_USER_ALEX", "").strip()
p2 = _get_secret("APP_PASS_ALEX", "")
if u1 and p1: USERS[u1.lower()] = p1
if u2 and p2: USERS[u2.lower()] = p2

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Arquivos do índice
# -----------------------------
INDEX_DIR = "index"
FAISS_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_PKL  = os.path.join(INDEX_DIR, "faiss_meta.pkl")       # legado
META_JSONL = os.path.join(INDEX_DIR, "metas.jsonl")         # novo estável
MANIFEST_JSON = os.path.join(INDEX_DIR, "manifest.json")    # info do índice

# -----------------------------
# Prompt
# -----------------------------
SYSTEM_PROMPT = (
    "Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group.\n"
    "POLÍTICA DE RESPOSTA:\n"
    "1) SEMPRE priorize as informações do <contexto> quando ele existir; se o contexto já trouxer a resposta, responda com base nele.\n"
    "2) Somente complemente com conhecimento próprio se o contexto não cobrir o necessário.\n"
    "3) Só inclua exemplo numérico se o usuário pedir explicitamente. Quando houver exemplo, use notação brasileira (R$ 1.234,56 e 12%).\n"
    "4) Cite leis/EC/LC apenas quando estritamente necessárias e de forma enxuta; não invente.\n"
    "5) Nunca mencione arquivos internos (.txt, .pdf).\n"
    "6) Não quebre números em linhas; mantenha 'R$' colado ao valor (ex.: R$ 1.000,00) e evite espaços entre dígitos.\n"
)

# -----------------------------
# Login
# -----------------------------
def _norm_name(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()

def login_box():
    st.title("🧑‍🏫 LAVO - Especialista em Reforma Tributária")
    with st.container():
        st.subheader("Login")
        nome  = st.text_input("Nome (igual ao cadastro)", key="login_nome")
        senha = st.text_input("Senha", type="password", key="login_senha")
        if st.button("Entrar", type="primary"):
            ok = USERS.get(_norm_name(nome)) == (senha or "")
            if ok:
                st.session_state.auth = True
                st.session_state.user_name = " ".join((nome or "").split()).strip()
                st.success(f"Bem-vindo, {st.session_state.user_name}!")
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")
    st.stop()

# -----------------------------
# Utilidades
# -----------------------------
def _coerce_dict_list(obj_list) -> List[Dict[str, Any]]:
    out = []
    for x in obj_list or []:
        if isinstance(x, dict):
            out.append(x)
        else:
            d = {}
            for attr in ("text", "text_preview", "title", "source", "path", "chunk_id"):
                if hasattr(x, attr):
                    d[attr] = getattr(x, attr)
            if not d and hasattr(x, "__dict__"):
                try:
                    d = {k: v for k, v in x.__dict__.items() if not k.startswith("__")}
                except:
                    d = {}
            out.append(d)
    return out

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                items.append(json.loads(line))
            except:
                continue
    return items

def _chunks_count(metas): return len(metas or [])

# ===== Sanitização =====
_EXAMPLE_TAG_RE = re.compile(r'(?is)\bexemplo\s+num[eé]rico.*?(?:\n\n|$)')

def _wants_example(question: str) -> bool:
    q = (question or "").lower()
    keys = ("exemplo", "exemplifique", "simule", "cálculo", "calculo", "número", "numeros", "números")
    return any(k in q for k in keys)

def _clean_output(s: str) -> str:
    if not s: 
        return s
    s = s.replace("\r\n", "\n").replace("\u200b", " ")
    s = re.sub(r'(?<!\n)\n(?!\n)', ' ', s)            # quebra simples -> espaço
    s = re.sub(r'(?<=\d)\s+(?=\d)', '', s)            # 1 000 -> 1000
    s = re.sub(r'R\s*\$', 'R$', s)                    # R $ -> R$
    s = re.sub(r'R\$(\s+)(?=\d)', 'R$', s)            # R$  1 -> R$1
    s = re.sub(r'\s+%', '%', s)                       # 12 % -> 12%
    s = re.sub(r'[ \t]{2,}', ' ', s)                  # múltiplos espaços
    s = re.sub(r'\n{3,}', '\n\n', s)                  # quebras em excesso
    return s.strip()

def _strip_example_if_unwanted(text: str, question: str) -> str:
    if _wants_example(question):
        return text
    return _EXAMPLE_TAG_RE.sub('', text).strip()

# ===== Datas & contagens (determinístico) =====
BR_DATE_RE = re.compile(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})')
CTX_TODAY_RE = re.compile(r'(?i)\bhoje\s+é\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})')
CTX_DAYS_RE  = re.compile(r'(?i)\b(restam|faltam)\s+(\d{1,4})\s+dias\b')
CTX_TARGET_RE = re.compile(r'(?i)\b(em|no|para)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})')

def _parse_br_date_str(s: str) -> Optional[date]:
    m = BR_DATE_RE.search(s or "")
    if not m: return None
    d, mth, y = map(int, m.groups())
    try:
        return date(y, mth, d)
    except ValueError:
        return None

def _date_from_context_text(txt: str) -> Optional[date]:
    m = CTX_TODAY_RE.search(txt or "")
    if m:
        return _parse_br_date_str(m.group(1))
    return _parse_br_date_str(txt or "")

def _target_from_context_text(txt: str) -> Optional[date]:
    m = CTX_TARGET_RE.search(txt or "")
    if m:
        return _parse_br_date_str(m.group(2))
    return None

def _days_from_context_text(txt: str) -> Optional[int]:
    m = CTX_DAYS_RE.search(txt or "")
    if m:
        try:
            return int(m.group(2))
        except:
            return None
    return None

def _best_candidate_snippet(context_text: str) -> str:
    paras = [p.strip() for p in context_text.split("\n") if p.strip()]
    for p in paras:
        if CTX_DAYS_RE.search(p) or CTX_TODAY_RE.search(p):
            return p
    return context_text[:2000]

def _extract_ref_and_target_from_question(q: str) -> Tuple[date, date]:
    ref = _parse_br_date_str(q) or date.today()
    all_dates = BR_DATE_RE.findall(q or "")
    target = None
    for d, mth, y in all_dates:
        dt = _parse_br_date_str(f"{d}/{mth}/{y}")
        if dt and dt.year >= 2026:
            target = dt
            break
    if not target:
        target = date(2026, 1, 1)
    return ref, target

def ctx_reference_is_useful(ctx_ref: Optional[date], ctx_days: Optional[int]) -> bool:
    return (ctx_ref is not None) and (ctx_days is not None)

def _recompute_days_paragraph(paragraph: str, question: str) -> Optional[str]:
    if not paragraph:
        return None
    ctx_ref = _date_from_context_text(paragraph)
    ctx_days = _days_from_context_text(paragraph)
    ctx_target = _target_from_context_text(paragraph)
    if not ctx_reference_is_useful(ctx_ref, ctx_days):
        return None
    q_ref, q_target = _extract_ref_and_target_from_question(question)
    target = q_target or ctx_target or date(2026, 1, 1)
    new_days = (target - q_ref).days
    if new_days > 0:
        return f"Faltam **{new_days} dias** para o início em {target.strftime('%d/%m/%Y')}."
    elif new_days == 0:
        return f"**Hoje** ({q_ref.strftime('%d/%m/%Y')}) é o início."
    else:
        return f"O início ({target.strftime('%d/%m/%Y')}) já passou há **{abs(new_days)} dias**."

# ===== Detecção de intenção "dias"
DAYS_INTENT_RE = re.compile(
    r'(?i)(faltam\s+quantos\s+dias|quantos\s+dias\s+faltam|quantos\s+dias|dias\s+faltam|quanto\s+tempo).*?(reforma|ibs|cbs|in[ií]cio)',
    re.DOTALL
)
def _is_days_intent(q: str) -> bool:
    return bool(DAYS_INTENT_RE.search(q or ""))

# ===== Âncora de seção (genérico + sinônimos)
SYNONYMS = {
    "imposto do pecado": ["imposto do pecado", "imposto seletivo", "is"],
    "imposto seletivo": ["imposto seletivo", "imposto do pecado", "is"],
    "iva dual": ["iva dual", "iva", "ibs e cbs"],
    "ibs": ["ibs", "imposto sobre bens e serviços"],
    "cbs": ["cbs", "contribuição sobre bens e serviços"],
    "split payment": ["split payment", "pagamento dividido", "split"],
}

HEADER_RE = re.compile(r'^\s*\d+\s*[-–]\s', re.IGNORECASE)  # ex.: "7 - O QUE É ..."

def _anchor_answer_from_context(question: str, context_text: str) -> Optional[str]:
    q = (question or "").lower().strip()
    if not q or not context_text:
        return None

    # gera candidatos (palavras-chave + sinônimos)
    cand = set()
    for key, vals in SYNONYMS.items():
        if key in q:
            cand.update(vals)
    # também acrescenta n-grams simples da própria pergunta (até 4 palavras)
    tokens = [t for t in re.split(r'[^a-zà-ú0-9]+', q) if t]
    for n in range(4, 0, -1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n]).strip()
            if len(phrase) >= 4:
                cand.add(phrase)

    if not cand:
        return None

    # varre o contexto por linha; ao bater, coleta o bloco daquela "seção"
    lines = context_text.splitlines()
    lower = [ln.lower() for ln in lines]

    def matches_any(idx: int) -> bool:
        s = lower[idx]
        return any(c in s for c in cand)

    # encontra a primeira linha que casa
    hit = None
    for i in range(len(lines)):
        if matches_any(i):
            hit = i
            break
    if hit is None:
        return None

    # coleta bloco: inclui a linha encontrada e segue até próxima seção/linha vazia longa
    start = hit
    # se a linha anterior for um cabeçalho tipo "7 - ...", puxa desde lá
    j = hit
    while j-1 >= 0 and not lines[j-1].strip() == "" and not HEADER_RE.match(lines[j]):
        j -= 1
    # se havia um header imediatamente antes, usa-o como início
    if j-1 >= 0 and HEADER_RE.match(lines[j-1]):
        start = j-1
    else:
        start = j

    block = [lines[start].strip()]
    i = hit + 1
    empty_run = 0
    while i < len(lines):
        ln = lines[i]
        if HEADER_RE.match(ln):  # nova seção numerada
            break
        if not ln.strip():
            empty_run += 1
            if empty_run >= 2:  # dois parágrafos vazios seguidos -> parar
                break
            block.append("")  # mantém separação de parágrafos
            i += 1
            continue
        empty_run = 0
        block.append(ln.strip())
        # limite de segurança
        if len("\n".join(block)) > 1500:
            break
        i += 1

    answer = "\n".join([b for b in block if b is not None]).strip()
    # Se o bloco for curto demais, não vale a pena
    if len(answer) < 40:
        return None
    return answer

# -----------------------------
# Loaders (cacheados)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_index() -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    if not os.path.exists(FAISS_FILE):
        return None, [], {}
    try:
        idx = faiss.read_index(FAISS_FILE)
    except Exception as e:
        st.error(f"Falha ao carregar FAISS: {e}")
        return None, [], {}

    metas: List[Dict[str, Any]] = []
    if os.path.exists(META_JSONL):
        try:
            metas = _read_jsonl(META_JSONL)
        except Exception as e:
            st.error(f"Falha ao ler metas.jsonl: {e}")
            metas = []

    if not metas and os.path.exists(META_PKL):
        try:
            with open(META_PKL, "rb") as f:
                raw = pickle.load(f)
            if isinstance(raw, list):
                metas = _coerce_dict_list(raw)
            else:
                metas = _coerce_dict_list([raw])
        except Exception as e:
            st.error(f"Falha ao carregar faiss_meta.pkl: {e}")
            metas = []

    manifest = {}
    if os.path.exists(MANIFEST_JSON):
        try:
            with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    return idx, metas, manifest

# -----------------------------
# Embeddings / Busca
# -----------------------------
def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def _faiss_search(index, query_vec: np.ndarray, k: int = 5):
    D, I = index.search(query_vec.reshape(1, -1), k)
    return D[0], I[0]

@st.cache_resource(show_spinner=False)
def _bm25_index(metas: List[Dict[str, Any]]):
    docs = []
    for m in metas:
        txt = str(m.get("text") or m.get("text_preview") or "")
        tokens = txt.lower().split()
        docs.append(tokens)
    return BM25Okapi(docs)

def _hybrid_rank(
    question: str,
    index,
    metas: List[Dict[str, Any]],
    k_faiss: int = 8,
    k_bm25: int = 12,
    top_k: int = 6,
) -> List[int]:
    if not metas:
        return []

    faiss_idxs, faiss_scores = [], {}
    if index is not None:
        qv = embed_query(question)
        D, I = _faiss_search(index, qv, k=k_faiss)
        for score, idx in zip(D, I):
            if 0 <= idx < len(metas):
                faiss_idxs.append(idx)
                faiss_scores[idx] = float(score)

    bm25 = _bm25_index(metas)
    bm25_scores = bm25.get_scores(question.lower().split())
    bm25_top = np.argsort(bm25_scores)[::-1][:k_bm25].tolist()

    pool = set(faiss_idxs) | set(bm25_top)
    if not pool:
        return []

    def _norm(vals: Dict[int, float]):
        if not vals: return {}
        arr = np.array(list(vals.values()), dtype=float)
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmax - vmin < 1e-9:
            return {k: 0.0 for k in vals}
        return {k: (v - vmin) / (vmax - vmin) for k, v in vals.items()}

    f_full = {i: faiss_scores.get(i, 0.0) for i in pool}
    b_full = {i: float(bm25_scores[i]) for i in pool}

    f_n = _norm(f_full)
    b_n = _norm(b_full)

    combo = {i: 0.55 * f_n.get(i, 0.0) + 0.45 * b_n.get(i, 0.0) for i in pool}
    reranked = sorted(combo.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in reranked[:top_k]]

def _build_context(metas: List[Dict[str, Any]], idxs: List[int], max_chars=2800) -> str:
    parts, total = [], 0
    for rank, i in enumerate(idxs, start=1):
        if i < 0 or i >= len(metas): 
            continue
        m = metas[i] or {}
        snippet = str(m.get("text") or m.get("text_preview") or "").replace("\u200b", " ")
        title   = str(m.get("title", ""))[:200]
        piece = f"[{rank}] {title}\n{snippet}\n"
        parts.append(piece); total += len(piece)
        if total >= max_chars:
            break
    return "\n".join(parts).strip()

# -----------------------------
# Context-First Answer Engine (com âncora de seção)
# -----------------------------
def _context_first_transform(question: str, context_text: str) -> Optional[str]:
    # 1) ÂNCORA: se a pergunta contém termo que aparece no contexto, retorna o bloco daquela seção
    anchored = _anchor_answer_from_context(question, context_text)
    if anchored:
        return anchored

    # 2) Regra de "faltam quantos dias" (recalcula determinístico)
    if _is_days_intent(question):
        para = _best_candidate_snippet(context_text)
        fixed = _recompute_days_paragraph(para, question)
        if fixed:
            return fixed

    # (Novas transformações genéricas podem ser plugadas aqui)
    return None

# -----------------------------
# Resposta (RAG) com política context-first
# -----------------------------
def answer_with_rag(question: str, user_name: str, index, metas, top_k=6) -> str:
    context_text = ""
    if index is not None and metas:
        idxs = _hybrid_rank(question, index, metas, k_faiss=8, k_bm25=12, top_k=top_k)
        context_text = _build_context(metas, idxs, max_chars=2800)

    # 1) Context-first transform: responde direto do contexto quando possível
    if context_text:
        direct = _context_first_transform(question, context_text)
        if direct:
            return _clean_output(direct)

    # 2) Caso não dê, consulta o modelo (forçado a usar contexto primeiro)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": (
             f"Usuário: {user_name}\n"
             + (f"<contexto>\n{context_text}\n</contexto>\n" if context_text else "")
             + f"Pergunta: {question}\n"
             "- Use o contexto acima como base; responda de forma direta e fiel ao texto."
         )},
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    text = resp.choices[0].message.content.strip()
    text = _strip_example_if_unwanted(text, question)
    return _clean_output(text)

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="LAVO - Reforma Tributária", page_icon="📄", layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    login_box()

user_name = st.session_state.get("user_name", "Usuário")

st.title("🧑‍🏫 LAVO - Especialista em Reforma Tributária")

index, metas, manifest = load_index()
if not index or not metas:
    st.warning(
        "⚠️ Base não encontrada. Gere `index/faiss.index` e `index/metas.jsonl` "
        "(ou `faiss_meta.pkl`) via GitHub Actions a partir dos `.txt`. "
        "Depois de commitados em `index/`, recarregue esta página."
    )
else:
    emb_model = manifest.get("emb_model", "desconhecido")
    st.caption(f"Base carregada • trechos: {_chunks_count(metas)} • modelo de embedding: {emb_model}")

# Histórico
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

user_q = st.chat_input("Pergunte algo sobre Reforma Tributária…")
if user_q:
    st.session_state.history.append(("user", user_q))
    with st.chat_message("user"): 
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            try:
                text = answer_with_rag(user_q, user_name, index, metas, top_k=6)
            except Exception as e:
                text = f"Desculpe, ocorreu um erro ao gerar a resposta. Detalhe: {e}"

            st.markdown(text)
            st.session_state.history.append(("assistant", text))

st.caption("Dica: pergunte por um termo específico (ex.: “imposto do pecado”, “split payment”) para ver a resposta ancorada na seção correta da base.")
