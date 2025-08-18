import os
import streamlit as st
from openai import OpenAI
import faiss
import pickle
from typing import List, Tuple

# ==========================
# Configurações iniciais
# ==========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ==========================
# PROMPT DO SISTEMA
# ==========================
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
# Funções auxiliares
# ==========================
def load_faiss_index(path="faiss.index", meta_path="faiss_meta.pkl"):
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return None, []
    index = faiss.read_index(path)
    with open(meta_path, "rb") as f:
        metas = pickle.load(f)
    return index, metas

def embed_text(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def search_index(query: str, index, metas, k=3):
    if index is None:
        return []
    q_emb = embed_text(query)
    D, I = index.search([q_emb], k)
    hits = []
    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(metas):
            continue
        hits.append((rank+1, D[0][rank], metas[idx]))
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
         "content": f"{user_instruction}\n\n<contexto>\n{contexto}\n</contexto>\n\nPergunta: {question}"}
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=600,
    )
    return resp.choices[0].message.content

# ==========================
# INTERFACE STREAMLIT
# ==========================
st.set_page_config(page_title="LAVO - Reforma Tributária", layout="centered")

st.title("🧑‍🏫 LAVO - Especialista em Reforma Tributária")

# Login simples
if "user_name" not in st.session_state:
    nome = st.text_input("Digite seu nome para iniciar:", "")
    if nome:
        st.session_state["user_name"] = nome

if "user_name" in st.session_state:
    nome = st.session_state["user_name"]
    st.success(f"Bem-vindo, {nome}!")

    index, metas = load_faiss_index()
    if index is None:
        st.warning("⚠️ Nenhum índice encontrado. Suba os arquivos .txt e rode o indexador no GitHub Actions.")
    else:
        question = st.text_area("Digite sua pergunta sobre Reforma Tributária:", "")
        if st.button("Perguntar") and question.strip():
            hits = search_index(question, index, metas, k=3)
            answer = answer_with_context(question, hits, nome)
            st.markdown(answer)
