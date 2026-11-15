
import os
import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from groq import Groq

st.set_page_config(page_title="Asistente Académico RAG", page_icon="🎓", layout="wide")
st.title("🎓 Asistente Académico Institucional")
st.caption("Chatbot RAG para orientación estudiantil, bienestar y permanencia académica.")

def secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)

GEMINI_API_KEY = secret("GEMINI_API_KEY")
GROQ_API_KEY = secret("GROQ_API_KEY")
QDRANT_URL = secret("QDRANT_URL")
QDRANT_API_KEY = secret("QDRANT_API_KEY")

if not all([GEMINI_API_KEY, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    st.error("Configura GEMINI_API_KEY, GROQ_API_KEY, QDRANT_URL y QDRANT_API_KEY en Secrets.")
    st.stop()

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "openai/gpt-oss-120b"
VECTOR_SIZE = 768
COLLECTION_NAME = "aca1_educacion_gemini2"
TOP_K = 5

@st.cache_resource
def clients(gemini_key, groq_key, qdrant_url, qdrant_key):
    return (
        genai.Client(api_key=gemini_key),
        Groq(api_key=groq_key),
        QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=90)
    )

gemini_client, groq_client, qdrant_client = clients(
    GEMINI_API_KEY, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY
)

SYSTEM_INSTRUCTION = """
Eres un asistente virtual académico del pilar Educación.
Responde en español utilizando exclusivamente el contexto documental.
No inventes normas, fechas, requisitos ni procedimientos.
Si la información no está en el contexto, dilo claramente.
Incluye referencias [1], [2], etc. junto a las afirmaciones respaldadas.
"""

def retrieve(question):
    emb = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=f"task: question answering | query: {question}",
        config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE)
    ).embeddings[0].values
    result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME, query=emb, limit=TOP_K, with_payload=True
    )
    items = []
    for rank, point in enumerate(result.points, start=1):
        p = point.payload or {}
        items.append({
            "rank": rank, "score": float(point.score),
            "source": p.get("source", "Fuente no identificada"),
            "chunk": p.get("chunk_index", "N/D"),
            "text": p.get("text", "")
        })
    return items

def answer(question):
    items = retrieve(question)
    if not items or not any(x["text"].strip() for x in items):
        return "No encontré evidencia documental suficiente para responder.", []
    context = "\n\n".join(
        f"[{x['rank']}] Fuente: {x['source']} | Fragmento: {x['chunk']}\n{x['text']}"
        for x in items
    )
    prompt = f"""{SYSTEM_INSTRUCTION}

CONTEXTO DOCUMENTAL:
{context}

PREGUNTA:
{question}

Responde con claridad, cita [n] y advierte si falta evidencia.
"""
    completion = groq_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_completion_tokens=900
    )
    response_text = completion.choices[0].message.content or "No se pudo generar respuesta."
    return response_text, items

with st.sidebar:
    st.header("Panel documental")
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        st.success("Qdrant conectado")
        st.metric("Vectores indexados", info.points_count or 0)
        st.caption(f"Colección: {COLLECTION_NAME}")
        st.caption(f"Dimensión: {VECTOR_SIZE}")
        st.caption(f"Embeddings: {EMBEDDING_MODEL}")
        st.caption(f"Generación: Groq · {GENERATION_MODEL}")
        points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME, limit=1000,
            with_payload=["source"], with_vectors=False
        )
        counts = {}
        for point in points:
            src = (point.payload or {}).get("source", "Fuente no identificada")
            counts[src] = counts.get(src, 0) + 1
        st.subheader("Documentos indexados")
        if counts:
            st.dataframe(pd.DataFrame(
                [{"Documento": k, "Fragmentos": v} for k, v in sorted(counts.items())]
            ), hide_index=True, use_container_width=True)
    except Exception as exc:
        st.error(f"No fue posible consultar Qdrant: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Pregunta sobre reglamentos, bienestar o permanencia...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Consultando documentos y generando respuesta..."):
            try:
                response_text, items = answer(question)
                st.markdown(response_text)
                st.session_state.last_sources = items
                if items:
                    with st.expander("Fuentes y fragmentos recuperados", expanded=True):
                        for item in items:
                            st.markdown(
                                f"**[{item['rank']}] {item['source']}** · "
                                f"Fragmento {item['chunk']} · Similitud {item['score']:.4f}"
                            )
                            st.write(item["text"])
                            st.divider()
            except Exception as exc:
                response_text = f"No fue posible completar la consulta: {type(exc).__name__}: {exc}"
                st.error(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})

if st.session_state.last_sources:
    with st.expander("Última recuperación documental"):
        st.dataframe(pd.DataFrame([{
            "Fuente": x["source"], "Fragmento": x["chunk"],
            "Similitud": round(x["score"], 4)
        } for x in st.session_state.last_sources]), hide_index=True, use_container_width=True)
