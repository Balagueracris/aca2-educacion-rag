import os
import base64
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from groq import Groq

st.set_page_config(page_title="Asistente Académico Institucional", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "openai/gpt-oss-120b"
VECTOR_SIZE = 768
COLLECTION_NAME = "aca1_educacion_gemini2"
TOP_K = 5
ASSET_DIR = Path(__file__).parent
MASCOT_PATH = ASSET_DIR / "mascota_asistente.png"

# Paleta y componentes inspirados directamente en la propuesta visual aprobada.
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root{--bg:#071126;--panel:#101d3d;--panel2:#17264b;--line:rgba(130,151,255,.28);--white:#f6f8ff;--muted:#b3bfdc;--cyan:#25d9f2;--violet:#8b43f5;--purple:#6b2bd9;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 80% 10%,rgba(85,45,190,.24),transparent 35%),radial-gradient(ellipse at 12% 85%,rgba(0,174,220,.15),transparent 32%),linear-gradient(135deg,#060d20 0%,#0a1734 58%,#07152b 100%);color:var(--white)}
[data-testid="stHeader"]{background:rgba(5,11,27,.35)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#111d40 0%,#0b1530 100%);border-right:1px solid rgba(139,158,255,.25)}
[data-testid="stSidebar"]>div:first-child{padding:1.15rem .95rem}
[data-testid="stSidebar"] *{color:#eef3ff}
.block-container{max-width:1480px;padding-top:1.15rem;padding-bottom:1.6rem}
/* Sidebar */
.brand{display:flex;align-items:center;gap:12px;padding:.4rem .2rem .25rem}
.brand-mark{font-size:2.25rem;filter:drop-shadow(0 0 12px rgba(115,77,255,.5))}
.brand-title{font-size:1.1rem;font-weight:800;color:#fff;line-height:1.2}
.brand-sub{font-size:.78rem;color:#aebde0;line-height:1.4;margin:.55rem 0 1.1rem}
.nav-item{display:flex;align-items:center;gap:12px;padding:.72rem .85rem;border-radius:13px;margin:.42rem 0;color:#eaf0ff;font-weight:700;border:1px solid transparent}
.nav-active{background:linear-gradient(100deg,#7029d6,#3545a6);border-color:rgba(176,140,255,.45);box-shadow:0 8px 25px rgba(91,42,200,.25)}
.nav-icon{font-size:1.15rem;width:25px;text-align:center}
.side-panel{border:1px solid rgba(160,139,255,.5);border-radius:17px;background:linear-gradient(145deg,rgba(28,42,86,.95),rgba(19,30,65,.95));padding:.85rem .9rem;box-shadow:0 12px 28px rgba(0,0,0,.18)}
.side-status{display:inline-flex;align-items:center;gap:7px;border-radius:99px;padding:.45rem .75rem;background:rgba(0,116,82,.45);border:1px solid rgba(54,226,174,.35);color:#8af4c9;font-size:.78rem;font-weight:700}
.side-stat{display:flex;justify-content:space-between;align-items:center;padding:.55rem 0;border-bottom:1px solid rgba(157,174,230,.13);font-size:.82rem;color:#dce6ff}
.side-stat:last-child{border:0}.side-stat b{font-size:1rem;color:#fff}
.side-section{font-weight:800;color:#fff;font-size:.95rem;margin:.9rem 0 .55rem}
.side-caption{color:#aebbd9;font-size:.78rem;line-height:1.6}
/* Hero */
.hero{position:relative;display:flex;align-items:center;gap:1.1rem;padding:.65rem .4rem 1.1rem;margin-bottom:.85rem;min-height:190px}
.hero-art{width:205px;min-width:205px;height:190px;object-fit:contain;filter:drop-shadow(0 0 25px rgba(91,76,255,.45));}
.hero-copy{flex:1;min-width:0}
.hero-title{font-size:clamp(1.65rem,3vw,2.5rem);font-weight:800;line-height:1.12;margin:.25rem 0 .55rem;color:#fff}
.hero-title span{background:linear-gradient(90deg,#a78bfa 0%,#7c9dff 52%,#27d9f2 100%);-webkit-background-clip:text;background-clip:text;color:transparent}
.kicker{font-size:.72rem;letter-spacing:.16em;font-weight:800;color:#5de8f5;text-transform:uppercase}
.hero-desc{font-size:.95rem;color:#d1dcf5;line-height:1.55;max-width:800px}
.online{position:absolute;right:.4rem;top:.25rem;border:1px solid rgba(51,230,172,.45);background:rgba(4,91,72,.28);color:#6ff0bf;padding:.5rem .9rem;border-radius:99px;font-size:.78rem;font-weight:800}
/* topic cards */
.section-title{font-size:1rem;font-weight:800;color:#f5f7ff;margin:.5rem 0 .7rem}
.topic-card{height:110px;border:1px solid rgba(95,157,255,.38);border-radius:17px;padding:.8rem .85rem;background:linear-gradient(145deg,rgba(18,55,91,.9),rgba(24,34,83,.92));box-shadow:0 9px 24px rgba(0,0,0,.18);}
.topic-card.purple{background:linear-gradient(145deg,rgba(63,31,119,.92),rgba(31,34,88,.94));border-color:rgba(172,100,255,.48)}
.topic-card.teal{background:linear-gradient(145deg,rgba(5,83,105,.9),rgba(18,43,83,.94));border-color:rgba(37,217,242,.45)}
.topic-top{display:flex;align-items:center;gap:.75rem}
.topic-icon{width:46px;height:46px;display:flex;align-items:center;justify-content:center;border-radius:13px;background:linear-gradient(145deg,rgba(28,206,239,.27),rgba(117,64,232,.45));font-size:1.45rem;box-shadow:inset 0 1px rgba(255,255,255,.12)}
.topic-name{font-weight:800;font-size:.9rem;color:#fff;line-height:1.25}
.topic-sub{color:#b9c9e9;font-size:.72rem;margin:.42rem 0 0 3.55rem}
.stButton>button{border:1px solid rgba(139,112,255,.52);border-radius:12px;background:linear-gradient(100deg,#6427c9,#174b85);color:#fff;font-weight:800;min-height:2.75rem;transition:all .18s ease}
.stButton>button:hover{border-color:#36e4f4;background:linear-gradient(100deg,#7c35e8,#087c9d);color:#fff;box-shadow:0 0 20px rgba(48,191,238,.2)}
/* chat */
[data-testid="stChatMessage"]{border:1px solid rgba(123,146,220,.25);border-radius:18px;background:linear-gradient(130deg,rgba(25,42,82,.94),rgba(19,32,67,.94));padding:.9rem 1rem;box-shadow:0 10px 28px rgba(0,0,0,.16)}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]{color:#f3f6ff}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){background:linear-gradient(110deg,rgba(43,58,112,.95),rgba(31,49,91,.95));}
.stChatInput textarea{border:1px solid rgba(146,112,255,.65)!important;border-radius:16px!important;background:linear-gradient(100deg,#121d3c,#18284c)!important;color:#fff!important}
.stChatInput textarea:focus{border-color:#27d9f2!important;box-shadow:0 0 0 1px #27d9f2!important}
[data-testid="stExpander"]{border:1px solid rgba(153,119,255,.45);border-radius:15px;background:linear-gradient(110deg,rgba(35,39,91,.88),rgba(18,45,77,.88))}
[data-testid="stAlert"]{border-radius:14px}
hr{border-color:rgba(146,165,230,.22)}
.small-muted{font-size:.76rem;color:#9eafd2}
footer{visibility:hidden}
@media(max-width:900px){.hero{gap:.2rem}.hero-art{width:125px;min-width:125px;height:140px}.online{position:static;margin-left:auto}.hero-title{font-size:1.55rem}.topic-card{height:112px}}
@media(max-width:600px){.hero{flex-wrap:wrap}.hero-art{width:100px;min-width:100px;height:110px}.online{display:none}}
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
required = {"GEMINI_API_KEY": GEMINI_API_KEY, "GROQ_API_KEY": GROQ_API_KEY, "QDRANT_URL": QDRANT_URL, "QDRANT_API_KEY": QDRANT_API_KEY}
missing = [k for k, v in required.items() if not v]

@st.cache_resource
def create_clients(gemini_key, groq_key, qdrant_url, qdrant_key):
    return genai.Client(api_key=gemini_key), Groq(api_key=groq_key), QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=90)

qdrant_ready, qdrant_error, collection_info = False, None, None
gemini_client = groq_client = qdrant_client = None
if not missing:
    try:
        gemini_client, groq_client, qdrant_client = create_clients(GEMINI_API_KEY, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY)
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        qdrant_ready = True
    except Exception as exc:
        qdrant_error = f"{type(exc).__name__}: {exc}"

SYSTEM_INSTRUCTION = """
Eres un orientador académico virtual del pilar Educación. Conversa en español de Colombia con tono cercano, claro, respetuoso y natural. Ayuda al estudiante a entender la información institucional sin sonar como una copia literal del documento.

REGLAS:
- Basa tus respuestas únicamente en el CONTEXTO DOCUMENTAL entregado.
- Comienza con una respuesta directa y sencilla; organiza los detalles en viñetas cuando sea útil.
- No inventes requisitos, fechas, beneficios, artículos ni procedimientos.
- Si la pregunta pide una lista completa y los fragmentos no contienen todos los elementos, aclara que la información recuperada es parcial.
- Si no hay evidencia suficiente, dilo con amabilidad y explica qué dato no aparece.
- Incluye referencias [1], [2], etc. junto a las afirmaciones que respaldan.
- Evita repetir la pregunta y las frases genéricas.
"""


def retrieve(question: str):
    embedding_response = gemini_client.models.embed_content(model=EMBEDDING_MODEL, contents=f"task: question answering | query: {question}", config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE))
    query_vector = embedding_response.embeddings[0].values
    result = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=TOP_K, with_payload=True)
    fragments = []
    for rank, point in enumerate(result.points, start=1):
        payload = point.payload or {}
        fragments.append({"rank": rank, "score": float(point.score), "source": payload.get("source", "Fuente no identificada"), "chunk": payload.get("chunk_index", "N/D"), "text": payload.get("text", "")})
    return fragments


def generate_answer(question: str):
    fragments = retrieve(question)
    usable = [x for x in fragments if str(x["text"]).strip()]
    if not usable:
        return "No encontré información suficiente en los documentos consultados. ¿Puedes reformular la pregunta o darme un poco más de contexto?", fragments
    context = "\n\n".join(f"[{x['rank']}] Documento: {x['source']} | Fragmento: {x['chunk']}\n{x['text']}" for x in usable)
    prompt = f"CONTEXTO DOCUMENTAL:\n{context}\n\nCONSULTA DEL ESTUDIANTE:\n{question}\n\nResponde de forma natural, clara y útil. Cita las fuentes [n]. No inventes información ni afirmes que una lista es completa si los fragmentos no lo demuestran."
    completion = groq_client.chat.completions.create(model=GENERATION_MODEL, messages=[{"role":"system","content":SYSTEM_INSTRUCTION},{"role":"user","content":prompt}], temperature=0.25, max_completion_tokens=1400)
    answer = (completion.choices[0].message.content or "").strip()
    return answer or "No pude generar una respuesta en este momento. Inténtalo nuevamente.", fragments


def render_sources(sources, expanded=False):
    with st.expander(f"📚  Fuentes de información ({len(sources)})", expanded=expanded):
        for item in sources:
            st.markdown(f"**[{item['rank']}] {item['source']}**")
            st.caption(f"Fragmento {item['chunk']}  ·  Similitud {item['score']:.3f}")
            excerpt = str(item["text"]).strip()
            st.markdown(excerpt[:1100] + ("…" if len(excerpt) > 1100 else ""))
            if item != sources[-1]: st.divider()

# Sidebar fiel al panel lateral del mockup.
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">🎓</div><div class="brand-title">Asistente<br>Académico<br>Institucional</div></div><div class="brand-sub">Orientación, bienestar y<br>permanencia estudiantil</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item nav-active"><span class="nav-icon">▢</span> Conversación</div><div class="nav-item"><span class="nav-icon">▤</span> Base documental</div><div class="nav-item"><span class="nav-icon">▥</span> Estado del sistema</div><div class="nav-item"><span class="nav-icon">ⓘ</span> Acerca del proyecto</div>', unsafe_allow_html=True)
    st.divider()
    if missing:
        st.markdown('<div class="side-status">● Credenciales pendientes</div>', unsafe_allow_html=True)
        st.caption("Configura las cuatro credenciales en Streamlit Secrets.")
    elif qdrant_ready:
        st.markdown('<div class="side-panel"><div class="side-status">● Base documental conectada</div><div class="side-stat"><span>▤ &nbsp; Vectores indexados</span><b>' + f"{collection_info.points_count or 0:,}" + '</b></div><div class="side-stat"><span>▱ &nbsp; Colección</span><small>' + COLLECTION_NAME + '</small></div><div class="side-stat"><span>⬡ &nbsp; Dimensión</span><b>768</b></div><div class="side-stat"><span>▦ &nbsp; Modelos activos</span></div><div class="side-caption">🔹 Gemini Embedding 2<br>🟣 Groq · GPT-OSS-120B</div></div>', unsafe_allow_html=True)
    else:
        st.error("No fue posible conectar con Qdrant")
        st.caption(qdrant_error or "Revisa QDRANT_URL y QDRANT_API_KEY.")
    st.markdown('<div class="side-section">▤ &nbsp; Base documental</div>', unsafe_allow_html=True)
    if qdrant_ready:
        try:
            points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=1000, with_payload=["source"], with_vectors=False)
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

# Hero: mascota protagonista + bienvenida + estado.
mascot_html = ""
if MASCOT_PATH.exists():
    mascot_b64 = base64.b64encode(MASCOT_PATH.read_bytes()).decode("utf-8")
    mascot_html = f'<img class="hero-art" src="data:image/png;base64,{mascot_b64}" alt="Mascota del asistente académico">'
st.markdown(f"""<div class=\"hero\">{mascot_html}<div class=\"hero-copy\"><div class=\"kicker\">PILAR EDUCACIÓN · ORIENTACIÓN VIRTUAL</div><div class=\"hero-title\">¡Hola! Soy tu <span>Asistente Académico</span></div><div class=\"hero-desc\">Puedo ayudarte con información sobre bienestar institucional, reglamento estudiantil, permanencia académica y procesos de la universidad.</div></div><div class=\"online\">● En línea</div></div>""", unsafe_allow_html=True)

if missing:
    st.error("Configura GEMINI_API_KEY, GROQ_API_KEY, QDRANT_URL y QDRANT_API_KEY en Settings → Secrets.")
    st.stop()
if not qdrant_ready:
    st.warning("La interfaz está disponible, pero no se pudo conectar con Qdrant. Revisa las credenciales.")
    st.stop()

# Tarjetas de acceso rápido con iconografía y color individual.
st.markdown('<div class="section-title">✦ &nbsp; ¿Qué te gustaría consultar?</div>', unsafe_allow_html=True)
topics = [
    ("📖", "Reglamento estudiantil", "Derechos, deberes y normas", "¿Cuáles son los derechos y deberes de los estudiantes?", ""),
    ("♡", "Bienestar institucional", "Servicios y actividades", "¿Qué servicios ofrece Bienestar Institucional?", "purple"),
    ("🎓", "Permanencia académica", "Requisitos y acompañamiento", "¿Qué condiciones debo cumplir para acceder a bienestar?", "teal"),
    ("▤", "Procesos académicos", "Orientación institucional", "¿Qué información hay sobre los procesos académicos?", "purple"),
]
cols = st.columns(4, gap="medium")
for i, (icon, title, sub, prompt, style) in enumerate(topics):
    with cols[i]:
        st.markdown(f'<div class="topic-card {style}"><div class="topic-top"><div class="topic-icon">{icon}</div><div class="topic-name">{title}</div></div><div class="topic-sub">{sub}</div></div>', unsafe_allow_html=True)
        if st.button("Consultar  ›", key=f"topic_{i}", use_container_width=True):
            st.session_state.pending_question = prompt
            st.rerun()

st.markdown('<div class="section-title" style="margin-top:1.2rem">💬 &nbsp; Conversación</div>', unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
if not st.session_state.messages:
    st.markdown('<div style="border:1px solid rgba(125,149,224,.22);border-radius:16px;padding:1rem 1.2rem;background:linear-gradient(110deg,rgba(24,42,79,.85),rgba(17,34,66,.85));color:#cbd8f5">¡Hola! Soy tu asistente académico. Escribe tu consulta o elige uno de los temas para comenzar.</div>', unsafe_allow_html=True)
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])
pending = st.session_state.pop("pending_question", None)
question = st.chat_input("✎  Escribe aquí tu pregunta...")
question_to_process = question or pending
if question_to_process:
    st.session_state.messages.append({"role":"user","content":question_to_process})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question_to_process)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("El asistente está consultando la base documental..."):
            try:
                answer, sources = generate_answer(question_to_process)
                st.markdown(answer)
                if sources: render_sources(sources)
                st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})
            except Exception as exc:
                friendly = f"No pude completar la consulta. Inténtalo nuevamente. (Detalle: {type(exc).__name__}: {exc})"
                st.error(friendly)
                st.session_state.messages.append({"role":"assistant","content":friendly,"sources":[]})
st.markdown('<div class="small-muted" style="text-align:center;padding:1.2rem 0 .3rem">Asistente Académico Institucional · Pilar Educación · Recuperación Aumentada (RAG)</div>', unsafe_allow_html=True)
