import os
import io
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
from langdetect import detect, DetectorFactory
from groq import Groq
import easyocr

# -------- Config / clave --------
st.set_page_config(page_title="OCR + LLM (Groq)", page_icon="🧠")

from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error('Falta **GROQ_API_KEY**. En Secrets agrega:\n```toml\nGROQ_API_KEY = "gsk_xxx"\n```')
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
DetectorFactory.seed = 0

@st.cache_resource(show_spinner=True)
def load_reader():
    return easyocr.Reader(["es","en","pt","fr","de","it"], gpu=False)

def detect_lang(text: str, fallback="es"):
    try:
        code = detect(text) if text and text.strip() else fallback
        return code if code in {"es","en","pt","fr","de","it"} else fallback
    except Exception:
        return fallback

def system_prompt(target_lang: str, tone: str):
    tones = {
        "Preciso y neutro": "preciso, claro, profesional y natural",
        "Conversacional": "cálido, cercano y conversacional",
        "Técnico/conciso": "técnico, conciso y directo al punto",
    }
    return (
        f"Eres un asistente útil. Responde en {target_lang} con un tono {tones.get(tone,'preciso y natural')}.\n"
        "Sé exacto, evita relleno y si te piden JSON, entrégalo **válido**. "
        "Si el texto OCR tiene ruido, marca supuestos con cautela."
    )

st.sidebar.header("⚙️ Ajustes")
model = st.sidebar.selectbox("Modelo", ["llama-3.3-70b-versatile","llama-3.1-8b-instant"], 0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
max_tokens = st.sidebar.slider("Máx. tokens", 128, 2000, 800, 64)
force_lang = st.sidebar.selectbox("Idioma de salida", ["Detectar automáticamente","es","en","pt","fr","de","it"], 0)
tone = st.sidebar.selectbox("Tono", ["Preciso y neutro","Conversacional","Técnico/conciso"], 0)
task = st.sidebar.text_area("Instrucción al LLM", "Resume el texto y, si procede, devuelve un JSON con campos clave.", height=90)

st.title("🧠 OCR + LLM (Groq) · Multilenguaje")
uploaded = st.file_uploader("Imagen (JPG/PNG)", type=["jpg","jpeg","png"])
run = st.button("Procesar", type="primary", use_container_width=True)

if run:
    if not uploaded:
        st.warning("Sube una imagen primero.")
        st.stop()

    st.subheader("📷 Imagen")
    st.image(uploaded, use_column_width=True)

    with st.spinner("Leyendo (OCR)…"):
        reader = load_reader()
        image = Image.open(uploaded).convert("RGB")
        np_img = np.array(image)             # ✅ conversión a ndarray
        ocr_results = reader.readtext(np_img)
        text = "\n".join([item[1] for item in ocr_results]).strip()

    st.subheader("🧾 Texto OCR")
    if not text:
        st.info("No se detectó texto.")
        st.stop()
    st.text_area("Resultado", text, height=280)

    lang = detect_lang(text, "es") if force_lang == "Detectar automáticamente" else force_lang
    st.toast(f"Idioma: {lang}", icon="🌐")

    messages = [
        {"role": "system", "content": system_prompt(lang, tone)},
        {"role": "user", "content": f"Tarea: {task}\n\nTexto OCR:\n{text[:6000]}"},
    ]

    with st.spinner("Consultando LLM…"):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(temperature),
                max_completion_tokens=int(max_tokens),
            )
            llm_text = completion.choices[0].message.content if completion.choices else ""
        except Exception as e:
            st.error("Error al llamar a Groq.")
            with st.expander("Detalle técnico"):
                st.exception(e)
            st.stop()

    st.subheader("🧠 Respuesta del LLM")
    st.markdown(llm_text or "_Sin contenido_")
else:
    st.info("Sube una imagen y pulsa **Procesar**.")
