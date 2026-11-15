import os
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from groq import Groq

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Asistente Académico Institucional",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "openai/gpt-oss-120b"
VECTOR_SIZE = 768
COLLECTION_NAME = "aca1_educacion_gemini2"
TOP_K = 5
ASSET_DIR = Path(__file__).parent

# -----------------------------------------------------------------------------
# Estilos visuales: tema oscuro, azul noche, violeta y cian
# -----------------------------------------------------------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root {
  --bg: #070d20; --bg2: #0b1530; --panel: #111d3c;
  --line: rgba(130,153,255,.22); --text: #f4f7ff;
  --muted: #aab7d4; --cyan: #22d3ee; --violet: #8b5cf6;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background: radial-gradient(ellipse at 72% 0%, rgba(65,55,160,.23), transparent 36%),
              radial-gradient(ellipse at 95% 80%, rgba(0,190,220,.12), transparent 30%),
              linear-gradient(135deg, #070d20 0%, #0a1230 55%, #0b1028 100%);
  color: var(--text);
}
[data-testid="stHeader"] { background: rgba(5,10,25,.55); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #101b3b 0%, #0c1530 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
[data-testid="stSidebar"] * { color: #edf3ff; }
.block-container { max-width: 1500px; padding-top: 1.7rem; padding-bottom: 2rem; }
.hero {
  position: relative; overflow: hidden; display:flex; align-items:center; gap:1.25rem;
  padding: 1.25rem 1.5rem; border:1px solid rgba(116,132,255,.34);
  border-radius:24px; margin-bottom:1.15rem;
  background: linear-gradient(112deg, rgba(18,31,70,.96), rgba(26,30,78,.92) 58%, rgba(31,24,76,.88));
  box-shadow: 0 18px 55px rgba(0,0,0,.24), inset 0 1px rgba(255,255,255,.05);
}
.hero:after { content:''; position:absolute; width:330px; height:180px; right:-100px; bottom:-130px;
  background:#7138ef; filter:blur(70px); opacity:.32; }
.hero-copy { position:relative; z-index:1; flex:1; }
.kicker { color:#67e8f9; font-size:.72rem; letter-spacing:.16em; font-weight:800; text-transform:uppercase; }
.hero h1 { color:#fff !important; font-size:clamp(1.65rem,3vw,2.45rem); line-height:1.12; margin:.35rem 0 .55rem; font-weight:800; }
.hero h1 span { background:linear-gradient(90deg,#a78bfa,#38bdf8); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { color:#c6d2ed; margin:0; font-size:.96rem; max-width:780px; }
.online-pill { position:relative; z-index:2; white-space:nowrap; border:1px solid rgba(45,230,168,.45);
  border-radius:999px; padding:.45rem .8rem; color:#6ee7b7; background:rgba(6,95,70,.22); font-size:.78rem; font-weight:700; }
.mascot { width:125px; height:125px; object-fit:contain; position:relative; z-index:1; filter:drop-shadow(0 0 18px rgba(79,70,229,.38)); }
.section-label { color:#dbeafe; font-size:1rem; font-weight:750; margin:.55rem 0 .7rem; }
.topic-card {
  border:1px solid rgba(103,125,255,.38); border-radius:17px; padding:.9rem .95rem;
  background:linear-gradient(145deg,rgba(24,42,83,.92),rgba(27,30,75,.82));
  min-height:94px; box-shadow:0 8px 24px rgba(0,0,0,.13); transition:all .18s ease;
}
.topic-icon { display:inline-flex; align-items:center; justify-content:center; width:34px;height:34px;border-radius:11px;
  background:linear-gradient(135deg,rgba(34,211,238,.22),rgba(139,92,246,.35)); font-size:1.1rem; margin-bottom:.45rem; }
.topic-title { color:#f6f8ff; font-size:.88rem; font-weight:700; line-height:1.25; }
.topic-sub { color:#9eadd0; font-size:.72rem; margin-top:.2rem; }
.stat-card { background:linear-gradient(145deg,rgba(30,47,91,.76),rgba(20,28,64,.8)); border:1px solid var(--line); border-radius:15px; padding:.8rem .9rem; }
.stat-label { color:#9eadd0; font-size:.72rem; }
.stat-value { color:#fff; font-size:1.3rem; font-weight:800; margin-top:.15rem; }
.status-ok { border:1px solid rgba(52,211,153,.25); background:rgba(6,95,70,.25); color:#6ee7b7; border-radius:12px; padding:.75rem .85rem; font-weight:700; }
.status-bad { border:1px solid rgba(248,113,113,.3); background:rgba(127,29,29,.24); color:#fca5a5; border-radius:12px; padding:.75rem .85rem; }
.sidebar-brand { padding:.3rem .2rem 1rem; }
.sidebar-brand h2 { font-size:1.1rem; margin:.35rem 0 .25rem; color:#fff; }
.sidebar-brand p { color:#aab7d4; font-size:.8rem; line-height:1.4; }
.sidebar-nav { border:1px solid rgba(139,92,246,.32); border-radius:12px; background:linear-gradient(90deg,rgba(109,40,217,.3),rgba(59,130,246,.12)); padding:.65rem .8rem; font-weight:700; margin:.6rem 0 1rem; }
[data-testid="stChatMessage"] { border:1px solid rgba(125,145,220,.2); border-radius:18px; background:rgba(22,34,70,.68); padding:.75rem .9rem; box-shadow:0 8px 22px rgba(0,0,0,.12); }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color:#edf3ff; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background:linear-gradient(100deg,rgba(37,67,130,.78),rgba(34,57,111,.7)); }
.stButton > button { border-radius:12px; border:1px solid rgba(111,128,255,.42); background:linear-gradient(110deg,rgba(73,47,153,.55),rgba(26,87,140,.45)); color:#f7f8ff; font-weight:700; min-height:3rem; }
.stButton > button:hover { border-color:#22d3ee; color:#fff; background:linear-gradient(110deg,rgba(109,40,217,.65),rgba(8,145,178,.45)); }
.stChatInput textarea { border:1px solid rgba(139,120,255,.55) !important; border-radius:16px !important; background:#111b39 !important; color:#f8fafc !important; }
[data-testid="stExpander"] { border:1px solid rgba(139,120,255,.35); border-radius:14px; background:rgba(19,29,62,.62); }
[data-testid="stMetric"] { background:rgba(30,47,91,.58); padding:.7rem .8rem; border-radius:13px; border:1px solid var(--line); }
hr { border-color:rgba(130,153,255,.2); }
.small-muted { color:#9eadd0; font-size:.78rem; }
footer { visibility:hidden; }
@media(max-width:850px) { .hero { gap:.5rem; padding:1rem; } .mascot { width:82px;height:82px; } .online-pill { display:none; } }
</style>
""", unsafe_allow_html=True)


def get_secret(name: str):
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

required = {"GEMINI_API_KEY": GEMINI_API_KEY, "GROQ_API_KEY": GROQ_API_KEY,
            "QDRANT_URL": QDRANT_URL, "QDRANT_API_KEY": QDRANT_API_KEY}
missing = [k for k, v in required.items() if not v]

# -----------------------------------------------------------------------------
# Clientes y conexión
# -----------------------------------------------------------------------------
@st.cache_resource
def create_clients(gemini_key, groq_key, qdrant_url, qdrant_key):
    return (genai.Client(api_key=gemini_key), Groq(api_key=groq_key),
            QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=90))

qdrant_ready, qdrant_error = False, None
collection_info = None
gemini_client = groq_client = qdrant_client = None
if not missing:
    try:
        gemini_client, groq_client, qdrant_client = create_clients(
            GEMINI_API_KEY, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY)
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        qdrant_ready = True
    except Exception as exc:
        qdrant_error = f"{type(exc).__name__}: {exc}"

# -----------------------------------------------------------------------------
# Recuperación y generación RAG (se conserva la lógica validada)
# -----------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
Eres un orientador académico virtual del pilar Educación. Conversa en español de
Colombia con un tono cercano, claro, respetuoso y natural. Ayuda al estudiante a
entender la información institucional; no respondas como si estuvieras copiando
un documento ni uses lenguaje excesivamente técnico.

REGLAS:
- Basa las respuestas únicamente en el CONTEXTO DOCUMENTAL entregado.
- Explica primero la idea principal con palabras sencillas y organiza detalles
  en viñetas cuando ayude a la lectura.
- No inventes requisitos, fechas, beneficios, artículos ni procedimientos.
- Si la pregunta pide una lista completa y los fragmentos no contienen todos los
  elementos, aclara que la información recuperada es parcial.
- Si no hay evidencia suficiente, dilo con amabilidad.
- Incluye referencias [1], [2], etc. junto a las afirmaciones que respaldan.
- Evita repetir la pregunta y las frases genéricas.
"""


def retrieve(question: str):
    embedding_response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=f"task: question answering | query: {question}",
        config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE),
    )
    query_vector = embedding_response.embeddings[0].values
    result = qdrant_client.query_points(collection_name=COLLECTION_NAME,
        query=query_vector, limit=TOP_K, with_payload=True)
    fragments = []
    for rank, point in enumerate(result.points, start=1):
        payload = point.payload or {}
        fragments.append({"rank": rank, "score": float(point.score),
            "source": payload.get("source", "Fuente no identificada"),
            "chunk": payload.get("chunk_index", "N/D"),
            "text": payload.get("text", "")})
    return fragments


def generate_answer(question: str):
    fragments = retrieve(question)
    usable = [x for x in fragments if str(x["text"]).strip()]
    if not usable:
        return ("No encontré información suficiente en los documentos consultados. "
                "¿Puedes reformular la pregunta o darme un poco más de contexto?", fragments)
    context = "\n\n".join(
        f"[{x['rank']}] Documento: {x['source']} | Fragmento: {x['chunk']}\n{x['text']}"
        for x in usable)
    prompt = f"CONTEXTO DOCUMENTAL:\n{context}\n\nCONSULTA DEL ESTUDIANTE:\n{question}\n\nResponde de forma natural, clara y útil. Cita las fuentes [n]. No inventes información ni afirmes que una lista es completa si los fragmentos no lo demuestran."
    completion = groq_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "system", "content": SYSTEM_INSTRUCTION},
                  {"role": "user", "content": prompt}],
        temperature=0.25, max_completion_tokens=1400)
    answer = (completion.choices[0].message.content or "").strip()
    return answer or "No pude generar una respuesta en este momento. Inténtalo nuevamente.", fragments


def render_sources(sources, expanded=False):
    with st.expander(f"📚 Fuentes de información ({len(sources)})", expanded=expanded):
        for item in sources:
            st.markdown(f"**[{item['rank']}] {item['source']}**")
            st.caption(f"Fragmento {item['chunk']} · Similitud {item['score']:.3f}")
            excerpt = str(item["text"]).strip()
            st.markdown(excerpt[:1100] + ("…" if len(excerpt) > 1100 else ""))
            if item != sources[-1]:
                st.divider()

# -----------------------------------------------------------------------------
# Barra lateral
# -----------------------------------------------------------------------------
with st.sidebar:
    mascot_path = ASSET_DIR / "mascota_asistente.png"
    if mascot_path.exists():
        st.image(str(mascot_path), use_container_width=True)
    st.markdown("""
      <div class="sidebar-brand">
        <h2>🎓 Asistente Académico</h2>
        <p>Orientación, bienestar y permanencia estudiantil</p>
      </div>
      <div class="sidebar-nav">▣ &nbsp; Conversación</div>
    """, unsafe_allow_html=True)

    st.markdown("### ◈ Estado del sistema")
    if missing:
        st.markdown('<div class="status-bad">Faltan credenciales de conexión</div>', unsafe_allow_html=True)
        st.caption("Configura GEMINI_API_KEY, GROQ_API_KEY, QDRANT_URL y QDRANT_API_KEY en Secrets.")
    elif qdrant_ready:
        st.markdown('<div class="status-ok">● &nbsp; Base documental conectada</div>', unsafe_allow_html=True)
        st.markdown("<div class='stat-card'><div class='stat-label'>VECTORES INDEXADOS</div><div class='stat-value'>" + f"{collection_info.points_count or 0:,}" + "</div></div>", unsafe_allow_html=True)
        st.caption(f"Colección · `{COLLECTION_NAME}`")
        st.caption(f"Dimensión vectorial · **{VECTOR_SIZE}**")
    else:
        st.markdown('<div class="status-bad">No fue posible conectar con Qdrant</div>', unsafe_allow_html=True)
        st.caption(qdrant_error or "Revisa las credenciales en Secrets.")

    st.divider()
    st.markdown("### ✦ Modelos activos")
    st.markdown("<div class='small-muted'>🔹 Embeddings<br><b>Gemini Embedding 2</b><br><br>🟣 Generación<br><b>Groq · GPT-OSS-120B</b></div>", unsafe_allow_html=True)
    st.divider()
    if qdrant_ready:
        st.markdown("### ▤ Base documental")
        try:
            points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME,
                limit=1000, with_payload=["source"], with_vectors=False)
            counts = {}
            for point in points:
                name = (point.payload or {}).get("source", "Fuente no identificada")
                counts[name] = counts.get(name, 0) + 1
            for name, count in sorted(counts.items()):
                st.markdown(f"📄 **{name}**")
                st.caption(f"{count} fragmentos")
        except Exception as exc:
            st.caption(f"No se pudo listar el corpus: {type(exc).__name__}")
    if st.button("↺  Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_question", None)
        st.rerun()
    st.markdown("<p class='small-muted'>Respuestas fundamentadas en documentos institucionales.</p>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Encabezado principal
# -----------------------------------------------------------------------------
mascot_html = ""
mascot_file = ASSET_DIR / "mascota_asistente.png"
if mascot_file.exists():
    mascot_html = f'<img class="mascot" src="data:image/png;base64,{__import__("base64").b64encode(mascot_file.read_bytes()).decode()}" />'

st.markdown(f"""
<div class="hero">
  {mascot_html}
  <div class="hero-copy">
    <div class="kicker">PILAR EDUCACIÓN · ORIENTACIÓN VIRTUAL</div>
    <h1>¡Hola! Soy tu <span>Asistente Académico</span></h1>
    <p>Puedo ayudarte con información sobre bienestar institucional, reglamento estudiantil,
    permanencia académica y procesos de la universidad.</p>
  </div>
  <div class="online-pill">● En línea</div>
</div>
""", unsafe_allow_html=True)

if missing:
    st.error("Configura las credenciales de la aplicación en Settings → Secrets para comenzar.")
    st.stop()
if not qdrant_ready:
    st.warning("La interfaz está disponible, pero no se pudo conectar con la base documental. Revisa QDRANT_URL y QDRANT_API_KEY.")
    st.stop()

# -----------------------------------------------------------------------------
# Preguntas sugeridas
# -----------------------------------------------------------------------------
st.markdown('<div class="section-label">✦ ¿Qué te gustaría consultar?</div>', unsafe_allow_html=True)
topics = [
    ("📘", "Reglamento estudiantil", "Derechos, deberes y normas" , "¿Cuáles son los derechos y deberes de los estudiantes?"),
    ("💜", "Bienestar institucional", "Servicios y actividades", "¿Qué servicios ofrece Bienestar Institucional?"),
    ("🎓", "Permanencia académica", "Requisitos y acompañamiento", "¿Qué condiciones debo cumplir para acceder a bienestar?"),
    ("📋", "Procesos académicos", "Orientación institucional", "¿Qué información hay sobre los procesos académicos?"),
]
cols = st.columns(4)
for i, (icon, title, sub, prompt) in enumerate(topics):
    with cols[i]:
        st.markdown(f'<div class="topic-card"><div class="topic-icon">{icon}</div><div class="topic-title">{title}</div><div class="topic-sub">{sub}</div></div>', unsafe_allow_html=True)
        if st.button("Consultar", key=f"topic_{i}", use_container_width=True):
            st.session_state.pending_question = prompt
            st.rerun()

st.markdown('<div class="section-label">💬 Conversación</div>', unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Escribe tu pregunta en el campo inferior o elige uno de los temas para comenzar.")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])

pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Escribe aquí tu pregunta...")
question_to_process = question or pending
if question_to_process:
    st.session_state.messages.append({"role": "user", "content": question_to_process})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question_to_process)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Estoy revisando los documentos para orientarte..."):
            try:
                answer, sources = generate_answer(question_to_process)
                st.markdown(answer)
                if sources:
                    render_sources(sources)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as exc:
                friendly = f"No pude completar la consulta. Inténtalo nuevamente. (Detalle: {type(exc).__name__}: {exc})"
                st.error(friendly)
                st.session_state.messages.append({"role": "assistant", "content": friendly, "sources": []})

st.markdown("<div class='small-muted' style='text-align:center;padding:1.4rem 0 .4rem'>Asistente Académico Institucional · Pilar Educación · RAG</div>", unsafe_allow_html=True)
