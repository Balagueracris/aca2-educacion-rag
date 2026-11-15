# ACA2 – Asistente Académico Institucional RAG

Aplicación Streamlit para orientación académica del pilar Educación.

## Arquitectura
- Gemini Embedding 2: vectoriza las consultas en 768 dimensiones.
- Qdrant Cloud: recupera fragmentos de `aca1_educacion_gemini2`.
- Groq (`openai/gpt-oss-120b`): genera respuestas fundamentadas y citas.

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

Configura estas variables en Streamlit Secrets o en el entorno local:
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

No publiques credenciales en GitHub.
