# ACA 2 – Asistente Académico RAG | Educación

Chatbot RAG que reutiliza la colección `aca1_educacion_gemini2` de Qdrant Cloud.

## Arquitectura
1. Gemini Embedding 2 vectoriza la pregunta (768 dimensiones).
2. Qdrant Cloud recupera fragmentos relevantes.
3. Groq Cloud con el modelo openai/gpt-oss-120b genera una respuesta basada en el contexto recuperado.
4. Streamlit muestra chat, historial, fuentes y panel documental.

## Requisitos
- Python 3.10 o superior.
- API Key de Gemini (embeddings).
- API Key de Groq (generación).
- Endpoint y API Key de Qdrant Cloud.
- Colección `aca1_educacion_gemini2` creada previamente.

## Ejecución local
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Secrets
Crea `.streamlit/secrets.toml` localmente (NO lo subas a GitHub):
```toml
GEMINI_API_KEY = "TU_CLAVE_GEMINI"
GROQ_API_KEY = "TU_CLAVE_GROQ"
QDRANT_URL = "TU_ENDPOINT_QDRANT"
QDRANT_API_KEY = "TU_CLAVE_QDRANT"
```
En Streamlit Community Cloud, configura las mismas variables en App settings > Secrets.

## Seguridad
No publiques API Keys, `.env` ni `secrets.toml`.
