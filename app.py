import os
import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from groq import Groq

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Asistente Académico | Educación",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "openai/gpt-oss-120b"
VECTOR_SIZE = 768
COLLECTION_NAME = "aca1_educacion_gemini2"
TOP_K = 5

# Identidad visual: azul marino, cian y violeta.
st.markdown(
    """
    <style>
    :root {
        --navy: #14213d;
        --navy-2: #1d3557;
        --cyan: #16a6c9;
        --violet: #7957d5;
        --surface: rgba(127, 151, 190, .09);
        --surface-2: rgba(127, 151, 190, .15);
        --muted: #9aa9bd;
    }
    .stApp {
        background: radial-gradient(ellipse at 15% 0%, rgba(22,166,201,.12), transparent 35%),
                    radial-gradient(ellipse at 90% 10%, rgba(121,87,213,.12), transparent 32%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111d35 0%, #172944 100%);
        border-right: 1px solid rgba(148,163,184,.18);
    }
    [data-testid="stSidebar"] * { color: #edf4ff; }
    .hero {
        padding: 1.35rem 1.5rem;
        border: 1px solid rgba(91, 171, 208, .28);
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(20,33,61,.96), rgba(29,53,87,.92) 62%, rgba(71,55,126,.88));
        box-shadow: 0 12px 35px rgba(0,0,0,.16);
        margin-bottom: 1.2rem;
    }
    .hero-kicker { color: #8ee6f5; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { color: #fff !important; font-size: clamp(1.7rem, 3vw, 2.45rem); margin: .35rem 0 .25rem 0; }
    .hero p { color: #d4e1f2; margin: 0; font-size: .98rem; }
    .section-title { font-size: 1.02rem; font-weight: 700; margin: .5rem 0 .65rem; }
    .stat-card {
        background: var(--surface);
        border: 1px solid rgba(148,163,184,.20);
        border-radius: 14px;
        padding: .8rem .9rem;
        min-height: 88px;
    }
    .stat-label { color: var(--muted); font-size: .76rem; }
    .stat-value { font-size: 1.45rem; font-weight: 750; margin-top: .15rem; }
    .source-card {
        border: 1px solid rgba(22,166,201,.24);
        background: var(--surface);
        border-radius: 12px;
        padding: .75rem .9rem;
        margin: .35rem 0;
    }
    .source-name { font-weight: 700; font-size: .88rem; }
    .source-meta { color: var(--muted); font-size: .76rem; margin-top: .2rem; }
    .hint { color: var(--muted); font-size: .82rem; }
    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 16px;
        padding: .35rem .7rem;
        background: rgba(127,151,190,.045);
    }
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(22,166,201,.38);
        font-weight: 650;
    }
    .stChatInput textarea { border-radius: 14px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str):
    """Lee secretos de Streamlit Cloud o variables de entorno locales."""
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")

# -----------------------------------------------------------------------------
# Encabezado
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Pilar Educación · Asistente inteligente</div>
      <h1>🎓 Asistente Académico Institucional</h1>
      <p>Una forma sencilla de consultar información sobre reglamentos, bienestar y permanencia estudiantil.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

required_secrets = {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
    "QDRANT_URL": QDRANT_URL,
    "QDRANT_API_KEY": QDRANT_API_KEY,
}
missing = [key for key, value in required_secrets.items() if not value]
if missing:
    st.error("Faltan credenciales en Streamlit → Settings → Secrets: " + ", ".join(missing))
    st.stop()


@st.cache_resource
def create_clients(gemini_key, groq_key, qdrant_url, qdrant_key):
    return (
        genai.Client(api_key=gemini_key),
        Groq(api_key=groq_key),
        QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=90),
    )


try:
    gemini_client, groq_client, qdrant_client = create_clients(
        GEMINI_API_KEY, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY
    )
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    qdrant_ready = True
    qdrant_error = None
except Exception as exc:
    qdrant_ready = False
    qdrant_error = f"{type(exc).__name__}: {exc}"


SYSTEM_INSTRUCTION = """
Eres un orientador académico virtual del pilar Educación. Conversa en español de
Colombia con un tono cercano, claro, respetuoso y natural. Ayuda al estudiante a
entender la información institucional; no respondas como si estuvieras copiando
un documento ni uses lenguaje excesivamente técnico.

REGLAS DE RESPUESTA:
- Basa las respuestas únicamente en el CONTEXTO DOCUMENTAL entregado.
- Explica primero la idea principal con palabras sencillas y organiza los detalles
  en viñetas cuando ayude a la lectura.
- No inventes requisitos, fechas, beneficios, artículos ni procedimientos.
- Si la pregunta pide una lista completa y los fragmentos no contienen todos los
  elementos, aclara que la información recuperada es parcial.
- Si no hay evidencia suficiente, dilo con amabilidad e indica qué información
  no aparece en los documentos consultados.
- Incluye referencias [1], [2], etc. junto a las afirmaciones que respaldan.
- Evita repetir la pregunta y evita frases genéricas como “según el contexto”.
- Termina, cuando sea útil, ofreciendo orientar sobre el siguiente paso.
"""


def retrieve(question: str):
    embedding_response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=f"task: question answering | query: {question}",
        config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE),
    )
    query_vector = embedding_response.embeddings[0].values
    result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    )
    fragments = []
    for rank, point in enumerate(result.points, start=1):
        payload = point.payload or {}
        fragments.append(
            {
                "rank": rank,
                "score": float(point.score),
                "source": payload.get("source", "Fuente no identificada"),
                "chunk": payload.get("chunk_index", "N/D"),
                "text": payload.get("text", ""),
            }
        )
    return fragments


def generate_answer(question: str):
    fragments = retrieve(question)
    usable = [item for item in fragments if item["text"].strip()]
    if not usable:
        return (
            "No encontré información suficiente en los documentos consultados para responder con seguridad. "
            "Puedes intentar formular la pregunta con un poco más de detalle.",
            fragments,
        )

    context = "\n\n".join(
        f"[{item['rank']}] Documento: {item['source']} | Fragmento: {item['chunk']}\n{item['text']}"
        for item in usable
    )
    user_prompt = f"""CONTEXTO DOCUMENTAL:
{context}

CONSULTA DEL ESTUDIANTE:
{question}

Prepara una respuesta clara, natural y útil. Incluye las referencias [n] que
correspondan. No afirmes que la lista es completa si los fragmentos no lo prueban.
"""
    completion = groq_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.25,
        max_completion_tokens=1400,
    )
    response_text = (completion.choices[0].message.content or "").strip()
    if not response_text:
        response_text = "No pude generar una respuesta en este momento. Inténtalo nuevamente."
    return response_text, fragments


# -----------------------------------------------------------------------------
# Barra lateral: estado de servicios y corpus
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 Panel del sistema")
    if qdrant_ready:
        st.success("Base documental conectada")
        st.metric("Fragmentos indexados", f"{collection_info.points_count or 0:,}")
        st.caption(f"Colección: `{COLLECTION_NAME}`")
        st.caption(f"Dimensión vectorial: {VECTOR_SIZE}")
        st.divider()
        st.markdown("**Modelos activos**")
        st.caption(f"🔹 Embeddings · {EMBEDDING_MODEL}")
        st.caption(f"🟣 Generación · Groq / {GENERATION_MODEL}")
        st.divider()
        st.markdown("**Documentos disponibles**")
        try:
            points, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=["source"],
                with_vectors=False,
            )
            counts = {}
            for point in points:
                source = (point.payload or {}).get("source", "Fuente no identificada")
                counts[source] = counts.get(source, 0) + 1
            for source, count in sorted(counts.items()):
                st.markdown(f"📄 **{source}**  \n<span class='hint'>{count} fragmentos</span>", unsafe_allow_html=True)
        except Exception as exc:
            st.caption(f"No se pudo listar el corpus: {type(exc).__name__}")
    else:
        st.error("No fue posible conectar con Qdrant")
        st.caption(qdrant_error)
        st.info("Verifica QDRANT_URL y QDRANT_API_KEY en Settings → Secrets.")

    st.divider()
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()
    st.caption("Las respuestas se construyen a partir de los documentos institucionales indexados.")

if not qdrant_ready:
    st.stop()

# -----------------------------------------------------------------------------
# Chat principal
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

left, right = st.columns([0.72, 0.28])
with left:
    st.markdown("### 💬 Conversa con el asistente")
    st.markdown(
        "<span class='hint'>Puedes preguntar con tus propias palabras. El asistente buscará información en los documentos institucionales.</span>",
        unsafe_allow_html=True,
    )
with right:
    st.markdown("<div style='text-align:right; padding-top:.5rem'><span class='hint'>Respuestas con fuentes</span></div>", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("**¿Por dónde quieres empezar?**")
    suggestions = [
        "¿Qué servicios ofrece Bienestar Institucional?",
        "¿Cuáles son los derechos y deberes de los estudiantes?",
        "¿Qué condiciones debo cumplir para acceder a bienestar?",
    ]
    cols = st.columns(len(suggestions))
    for index, suggestion in enumerate(suggestions):
        with cols[index]:
            if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 Consultar fuentes ({len(message['sources'])})"):
                for item in message["sources"]:
                    st.markdown(
                        f"**[{item['rank']}] {item['source']}** · Fragmento {item['chunk']} · "
                        f"Coincidencia {item['score']:.3f}"
                    )
                    excerpt = item["text"].strip()
                    st.caption(excerpt[:1400] + ("…" if len(excerpt) > 1400 else ""))
                    st.divider()

pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Escribe aquí tu consulta sobre Educación...")
question_to_process = question or pending

if question_to_process:
    st.session_state.messages.append({"role": "user", "content": question_to_process})
    with st.chat_message("user"):
        st.markdown(question_to_process)
    with st.chat_message("assistant"):
        with st.spinner("Revisando los documentos para darte una orientación..."):
            try:
                response_text, sources = generate_answer(question_to_process)
                st.markdown(response_text)
                if sources:
                    with st.expander(f"📚 Fuentes consultadas ({len(sources)})", expanded=False):
                        for item in sources:
                            st.markdown(
                                f"**[{item['rank']}] {item['source']}** · Fragmento {item['chunk']} · "
                                f"Coincidencia {item['score']:.3f}"
                            )
                            excerpt = item["text"].strip()
                            st.caption(excerpt[:1400] + ("…" if len(excerpt) > 1400 else ""))
                            st.divider()
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text, "sources": sources}
                )
                st.session_state.last_sources = sources
            except Exception as exc:
                friendly_error = (
                    "No pude completar la consulta. Inténtalo nuevamente en unos momentos. "
                    f"(Detalle técnico: {type(exc).__name__}: {exc})"
                )
                st.error(friendly_error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": friendly_error, "sources": []}
                )

st.markdown(
    "<div class='hint' style='text-align:center; padding:1.2rem 0 .4rem'>"
    "Asistente Académico Institucional · Pilar Educación · Sistema de recuperación aumentada (RAG)"
    "</div>",
    unsafe_allow_html=True,
)
