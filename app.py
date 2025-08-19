# app.py
# LAVO - RAG básico estilo NotebookLM (FAISS + BM25 + síntese)
# Ajustes: sem "Fontes", tom conversacional e TEMPERATURE=0.35

import os
import re
import json
import pickle
from typing import List, Tuple, Dict, Any, Optional
from datetime import date

import numpy as np
import faiss
import streamlit as st
from rank_bm25 import BM25Okapi

# ===== OpenAI (SDK v1)
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("Instale: pip install openai>=1.40.0")

# --------------------------------------------------------
# HOTFIX para pkl legado (classe Meta ausente no unpickle)
# --------------------------------------------------------
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

CHAT_MODEL  = _get_secret("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = _to_float(_get_secret("TEMPERATURE", "0.35"), 0.35)  # <— mais natural
MAX_TOKENS  = _to_int(_get_secret("MAX_TOKENS", "1100"), 1100)

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
INDEX_DIR  = "index"
FAISS_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_PKL   = os.path.join(INDEX_DIR, "faiss_meta.pkl")   # legado
META_JSONL = os.path.join(INDEX_DIR, "metas.jsonl")      # novo estável
MANIFEST   = os.path.join(INDEX_DIR, "manifest.json")

# -----------------------------
# Prompt (conversacional)
# -----------------------------
SYSTEM_PROMPT = (
    "Você é a LAVO, especialista em Reforma Tributária da Lavoratory Group.\n"
    "Estilo: direto, claro e humano, como uma conversa; responda em 1–3 parágrafos (sem listas se não forem necessárias).\n"
    "Use o <contexto> quando existir para fatos específicos (leis, datas, nomes). "
    "Se o contexto não trouxer esses fatos, explique de forma geral SEM inventar números de lei/artigos/datas.\n"
    "Nunca mencione arquivos ou índice. Não copie blocos longos; sintetize com suas palavras. "
    "Só traga exemplo numérico se o usuário pedir. Use notação brasileira (R$ 1.234,56; 12%). "
    "Não quebre valores monetários em linhas; mantenha 'R$' colado ao número."
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
# Utilidades & Sanitização
# -----------------------------
def _coerce_dict_list(obj_list) -> List[Dict[str, Any]]:
    out = []
    for x in obj_list or []:
        if isinstance(x, dict):
            out.append(x)
        else:
            d = {}
            for attr in ("text", "text_preview", "title", "source", "path", "chunk_id"):
                if hasattr(x, attr): d[attr] = getattr(x, attr)
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
            try: items.append(json.loads(line))
            except: continue
    return items

def _chunks_count(metas): return len(metas or [])

_EXAMPLE_TAG_RE = re.compile(r'(?is)\bexemplo\s+num[eé]rico.*?(?:\n\n|$)')

def _wants_example(question: str) -> bool:
    q = (question or "").lower()
    keys = ("exemplo", "exemplifique", "simule", "cálculo", "calculo", "número", "numeros", "números")
    return any(k in q for k in keys)

def _clean_output(s: str) -> str:
    if not s: return s
    s = s.replace("\r\n", "\n").replace("\u200b", " ")
    s = re.sub(r'(?<!\n)\n(?!\n)', ' ', s)
    s = re.sub(r'(?<=\d)\s+(?=\d)', '', s)
    s = re.sub(r'R\s*\$', 'R$', s)
    s = re.sub(r'R\$(\s+)(?=\d)', 'R$', s)
    s = re.sub(r'\s+%', '%', s)
    s = re.sub(r'[ \t]{2,}', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def _strip_example_if_unwanted(text: str, question: str) -> str:
    return text if _wants_example(question) else _EXAMPLE_TAG_RE.sub('', text).strip()

# -----------------------------
# Datas & contagens ("quantos dias")
# -----------------------------
BR_DATE_RE     = re.compile(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})')
CTX_TODAY_RE   = re.compile(r'(?i)\bhoje\s+é\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})')
CTX_DAYS_RE    = re.compile(r'(?i)\b(restam|faltam)\s+(\d{1,4})\s+dias\b')
CTX_TARGET_RE  = re.compile(r'(?i)\b(em|no|para)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})')

def _parse_br_date_str(s: str) -> Optional[date]:
    m = BR_DATE_RE.search(s or "")
    if not m: return None
    d, mth, y = map(int, m.groups())
    try: return date(y, mth, d)
    except ValueError: return None

def _date_from_context_text(txt: str) -> Optional[date]:
    m = CTX_TODAY_RE.search(txt or "");  return _parse_br_date_str(m.group(1)) if m else _parse_br_date_str(txt or "")

def _target_from_context_text(txt: str) -> Optional[date]:
    m = CTX_TARGET_RE.search(txt or ""); return _parse_br_date_str(m.group(2)) if m else None

def _days_from_context_text(txt: str) -> Optional[int]:
    m = CTX_DAYS_RE.search(txt or "");   return int(m.group(2)) if m else None

def _best_candidate_snippet(context_text: str) -> str:
    paras = [p.strip() for p in context_text.split("\n") if p.strip()]
    for p in paras:
        if CTX_DAYS_RE.search(p) or CTX_TODAY_RE.search(p): return p
    return context_text[:2000]

def _extract_ref_and_target_from_question(q: str) -> Tuple[date, date]:
    ref = _parse_br_date_str(q) or date.today()
    target = None
    for d, mth, y in BR_DATE_RE.findall(q or ""):
        dt = _parse_br_date_str(f"{d}/{mth}/{y}")
        if dt and dt.year >= 2026: target = dt; break
    return ref, (target or date(2026, 1, 1))

def _recompute_days_paragraph(paragraph: str, question: str) -> Optional[str]:
    if not paragraph: return None
    ctx_ref    = _date_from_context_text(paragraph)
    ctx_days   = _days_from_context_text(paragraph)
    ctx_target = _target_from_context_text(paragraph)
    if not (ctx_ref and ctx_days is not None): return None
    q_ref, q_target = _extract_ref_and_target_from_question(question)
    target = q_target or ctx_target or date(2026, 1, 1)
    new_days = (target - q_ref).days
    if new_days > 0:   return f"Faltam **{new_days} dias** para o início em {target.strftime('%d/%m/%Y')}."
    if new_days == 0:  return f"**Hoje** ({q_ref.strftime('%d/%m/%Y')}) é o início."
    return f"O início ({target.strftime('%d/%m/%Y')}) já passou há **{abs(new_days)} dias**."

# -----------------------------
# Carregamento do índice (cacheado)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_index() -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    if not os.path.exists(FAISS_FILE): return None, [], {}
    try: idx = faiss.read_index(FAISS_FILE)
    except Exception as e:
        st.error(f"Falha ao carregar FAISS: {e}"); return None, [], {}
    metas: List[Dict[str, Any]] = []
    if os.path.exists(META_JSONL):
        try: metas = _read_jsonl(META_JSONL)
        except Exception as e: st.error(f"Falha ao ler metas.jsonl: {e}"); metas = []
    if not metas and os.path.exists(META_PKL):
        try:
            with open(META_PKL, "rb") as f: raw = pickle.load(f)
            metas = _coerce_dict_list(raw if isinstance(raw, list) else [raw])
        except Exception as e:
            st.error(f"Falha ao carregar faiss_meta.pkl: {e}"); metas = []
    manifest = {}
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST, "r", encoding="utf-8") as f: manifest = json.load(f)
        except Exception: manifest = {}
    return idx, metas, manifest

# -----------------------------
# Busca (FAISS + BM25) com reforço lexical, nomes e force-find
# -----------------------------
def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def _faiss_search(index, query_vec: np.ndarray, k: int = 12):
    D, I = index.search(query_vec.reshape(1, -1), k);  return D[0], I[0]

@st.cache_resource(show_spinner=False)
def _bm25_index(metas: List[Dict[str, Any]]):
    docs = []
    for m in metas:
        txt = str(m.get("text") or m.get("text_preview") or "")
        docs.append(txt.lower().split())
    return BM25Okapi(docs)

SYNONYMS = {
    "split payment": ["split payment", "pagamento dividido", "split"],
    "imposto do pecado": ["imposto do pecado", "imposto seletivo", "is"],
    "imposto seletivo": ["imposto seletivo", "imposto do pecado", "is"],
    "iva dual": ["iva dual", "iva", "ibs e cbs"],
    "ibs": ["ibs", "imposto sobre bens e serviços"],
    "cbs": ["cbs", "contribuição sobre bens e serviços"],
}

NAME_SEQ_RE = re.compile(r'((?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-zà-ú]+(?:\s+|$)){2,})')

def _name_queries(question: str) -> List[str]:
    if not question:
        return []
    seqs = [m.group(1).strip() for m in NAME_SEQ_RE.finditer(question)]
    seqs += [s.lower() for s in seqs]
    seen, out = set(), []
    for s in seqs:
        k = s.strip()
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out

def _keywords_from_question(q: str) -> List[str]:
    ql = (q or "").lower()
    keys = set()
    for k, syns in SYNONYMS.items():
        if k in ql: keys.update(syns)
    for name in _name_queries(q):
        keys.add(name.lower())
    toks = [t for t in re.split(r'[^a-zà-ú0-9]+', ql) if t]
    for n in range(4, 0, -1):
        for i in range(len(toks)-n+1):
            phrase = " ".join(toks[i:i+n]).strip()
            if len(phrase) >= 4:
                keys.add(phrase)
    return list(keys)[:32]

def _hybrid_rank(question: str, index, metas: List[Dict[str, Any]],
                 k_faiss: int = 12, k_bm25: int = 24, top_k: int = 10) -> List[int]:
    if not metas: return []
    faiss_idxs, faiss_scores = [], {}
    if index is not None:
        qv = embed_query(question)
        D, I = _faiss_search(index, qv, k=k_faiss)
        for score, idx in zip(D, I):
            if 0 <= idx < len(metas): faiss_idxs.append(idx); faiss_scores[idx] = float(score)
    bm25 = _bm25_index(metas)
    bm25_scores = bm25.get_scores(question.lower().split())
    bm25_top = np.argsort(bm25_scores)[::-1][:k_bm25].tolist()

    keys = _keywords_from_question(question)
    bonus = {}
    if keys:
        for i, m in enumerate(metas):
            txt = (m.get("text") or m.get("text_preview") or "").lower()
            sc = sum(1 for kw in keys if kw in txt)
            if sc: bonus[i] = float(sc)

    pool = set(faiss_idxs) | set(bm25_top) | set(list(bonus.keys())[:k_bm25])
    if not pool: return []

    def _norm(vals: Dict[int, float]):
        if not vals: return {}
        arr = np.array(list(vals.values()), dtype=float)
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmax - vmin < 1e-9: return {k: 0.0 for k in vals}
        return {k: (v - vmin) / (vmax - vmin) for k, v in vals.items()}

    f_full = {i: faiss_scores.get(i, 0.0) for i in pool}
    b_full = {i: float(bm25_scores[i]) for i in pool}
    x_full = {i: bonus.get(i, 0.0)       for i in pool}

    f_n = _norm(f_full); b_n = _norm(b_full); x_n = _norm(x_full)
    combo = {i: 0.50*f_n.get(i,0.0) + 0.35*b_n.get(i,0.0) + 0.15*x_n.get(i,0.0) for i in pool}
    reranked = sorted(combo.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in reranked[:top_k]]

# -----------------------------
# Contexto (sem exibir fontes)
# -----------------------------
PARA_SPLIT_RE = re.compile(r'\n\s*\n')

def _extract_paragraph_hits(text: str, keywords: List[str], question: str = "") -> List[str]:
    if not text:
        return []
    paras = [p.strip() for p in PARA_SPLIT_RE.split(text) if p.strip()]
    if not paras:
        return []
    names = [n.strip() for n in _name_queries(question) if n.strip()]
    if names:
        L = [n.lower() for n in names]
        name_hits = [p for p in paras if any(n in p.lower() for n in L)]
        if name_hits:
            return name_hits[:2]
    if keywords:
        scored = []
        for p in paras:
            pl = p.lower()
            score = sum(1 for kw in keywords if kw in pl)
            if score: scored.append((score, p))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return [p for _, p in scored[:3]]
    return paras[:2]

def _nice_title(m: Dict[str, Any]) -> str:
    t = (m.get("title") or "").strip()
    if t: return t[:120]
    src = (m.get("source") or m.get("path") or "").replace("\\", "/")
    return os.path.basename(src)[:120] if src else "Documento"

def _build_context(metas: List[Dict[str, Any]], idxs: List[int], question: str,
                   max_chars=4400) -> str:
    keys = _keywords_from_question(question)
    parts, total = [], 0
    for i in idxs:
        if i < 0 or i >= len(metas): continue
        m = metas[i] or {}
        raw = str(m.get("text") or m.get("text_preview") or "").replace("\u200b", " ")
        hits = _extract_paragraph_hits(raw, keys, question)
        if not hits: continue
        title = _nice_title(m)
        block = f"{title}\n" + "\n\n".join(hits)
        parts.append(block)
        total += len(block)
        if total >= max_chars: break
    if not parts and idxs:
        i = idxs[0]; m = metas[i] or {}
        title = _nice_title(m)
        raw = str(m.get("text") or m.get("text_preview") or "")
        parts = [f"{title}\n{raw[:1800]}"]
    return ("\n\n---\n\n").join(parts).strip()

def _force_find_name_context(metas: List[Dict[str, Any]], question: str, max_chars=2000) -> str:
    names = [n.lower() for n in _name_queries(question)]
    if not names: 
        return ""
    parts, total = [], 0
    for m in metas:
        raw = str(m.get("text") or m.get("text_preview") or "")
        title = _nice_title(m)
        low = (raw + "\n" + title).lower()
        if any(n in low for n in names):
            hits = _extract_paragraph_hits(raw, [], question)
            block = f"{title}\n" + "\n\n".join(hits)
            if block.strip():
                parts.append(block); total += len(block)
                if total >= max_chars:
                    break
    return ("\n\n---\n\n").join(parts).strip()

# -----------------------------
# Resposta + "quantos dias"
# -----------------------------
DAYS_INTENT_RE = re.compile(
    r'(?i)(faltam\s+quantos\s+dias|quantos\s+dias\s+faltam|quantos\s+dias|dias\s+faltam|quanto\s+tempo).*?(reforma|ibs|cbs|in[ií]cio)',
    re.DOTALL
)

def answer_with_rag(question: str, user_name: str, index, metas, top_k=10) -> str:
    # 0) Atalho determinístico para "quantos dias"
    if DAYS_INTENT_RE.search(question or ""):
        context_text = ""
        if index is not None and metas:
            idxs = _hybrid_rank(question, index, metas, top_k=top_k)
            context_text = _build_context(metas, idxs, question, max_chars=3000)
        para = _best_candidate_snippet(context_text) if context_text else ""
        fixed = _recompute_days_paragraph(para, question)
        if fixed: return _clean_output(fixed)

    # 1) Contexto via índice
    context_text = ""
    if index is not None and metas:
        idxs = _hybrid_rank(question, index, metas, top_k=top_k)
        context_text = _build_context(metas, idxs, question, max_chars=4400)

    # 2) Se vazio e houver nome, force-find
    if not context_text and metas:
        forced = _force_find_name_context(metas, question, max_chars=2000)
        if forced:
            context_text = forced

    # 3) Sempre responder (com ou sem contexto)
    user_content = (
        f"Usuário: {user_name}\n"
        + (f"<contexto>\n{context_text}\n</contexto>\n" if context_text else "")
        + f"Pergunta: {question}\n"
        "- Regras: use o contexto para fatos específicos (leis/datas/nomes). "
        "Sem contexto suficiente, responda de forma geral e NÃO invente números de lei/artigos/datas."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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
    st.caption(f"Base carregada • trechos: {_chunks_count(metas)} • embedding: {emb_model}")

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
                text = answer_with_rag(user_q, user_name, index, metas, top_k=10)
            except Exception as e:
                text = f"Desculpe, ocorreu um erro ao gerar a resposta. Detalhe: {e}"
            st.markdown(text)
            st.session_state.history.append(("assistant", text))

st.caption("Dica: peça exemplos explicitamente (“traga 1 exemplo com números”) quando quiser simulações.")
