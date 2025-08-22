import os
import io
import time
from pathlib import Path

import streamlit as st
from PIL import Image
import easyocr
from langdetect import detect, DetectorFactory
from groq import Groq

# ---------- CONFIG ----------
st.set_page_config(page_title="OCR + LLM (Groq) · Multilenguaje", page_icon="🧠", layout="wide")
DetectorFactory.seed = 0  # langdetect determinista

# ---------- KEY ----------
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error(
        "No se encontró **GROQ_API_KEY**. En Streamlit Cloud ve a **Settings → Secrets** y agrega:\n\n"
        '```toml\nGROQ_API_KEY = "gsk_xxx..."\n```'
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------- HELPERS ----------
@st.cache_resource(show_spinner=True)
def load_ocr_reader(langs=("es","en","pt","fr","de","it")):
    # Cambia langs si quieres soportar otros
    return easyocr.Reader(list(langs), gpu=False)

def detect_lang(text: str, fallback="es"):
    try:
        code = detect(text) if text and text.strip() else fallback
        # normaliza códigos comunes
        mapping = {"pt":"pt", "es":"es", "en":"en", "fr":"fr", "de":"de", "it":"it"}
        return mapping.get(code, fallback)
    except Exception:
        return fallback

def human_system_prompt(target_lang: str, style: str):
    # Pequeña guía de estilo y calidad
    styles = {
        "Preciso y neutro": "preciso, claro, profesional y natural",
        "Conversacional": "cálido, cercano y conversacional",
        "Técnico/conciso": "técnico, conciso y directo al punto"
    }
    return (
        f"Eres un asistente útil. Responde en **{target_lang}** con un tono {styles.get(style, 'preciso y natural')}.\n"
        "Sé exacto, evita relleno, lista pasos si ayudan, y si te piden JSON entrégalo **válido**.\n"
        "Si el texto OCR tiene ruido, indica supuestos con cautela."
    )

def call_groq(model:str, messages:list, temperature:float, max_tokens:int):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
        max_completion_tokens=int(max_tokens),
    )

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Ajustes")
model = st.sidebar.selectbox(
    "Modelo Groq",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.2-11b-vision-preview"],
    index=0,
    help="Elige 70B para mejor calidad; 8B es más rápido."
)
temperature = st.sidebar.slider("Creatividad (temperature)", 0.0, 1.0, 0.2, 0.1)
max_tokens = st.sidebar.slider("Máx. tokens de salida", 128, 2000, 800, 64)
force_lang = st.sidebar.selectbox(
    "Idioma de respuesta",
    ["Detectar automáticamente", "es", "en", "pt", "fr", "de", "it"],
    index=0
)
style = st.sidebar.selectbox("Tono", ["Preciso y neutro", "Conversacional", "Técnico/conciso"], index=0)
task = st.sidebar.text_area(
    "Instrucción al LLM",
    "Resume el texto y extrae datos clave en JSON (si procede).",
    height=90
)
st.sidebar.caption("Tip: cambia la instrucción para extraer campos específicos (fecha, total, etc.).")

# ---------- MAIN UI ----------
st.title("🧠 OCR + LLM (Groq) · Multilenguaje")
st.write("Sube una imagen con texto. Haré OCR y luego pediré al LLM un análisis **preciso y humanizado** en el idioma adecuado.")

uploaded = st.file_uploader("Imagen (JPG/PNG)", type=["jpg","jpeg","png"])
run = st.button("Procesar", type="primary", use_container_width=True)

col1, col2 = st.columns(2)

if run:
    if not uploaded:
        st.warning("Sube una imagen primero.")
        st.stop()

    # Muestra la imagen
    with col1:
        st.subheader("📷 Imagen")
        st.image(uploaded, use_column_width=True)

    # OCR
    with st.spinner("Leyendo texto (OCR)…"):
        reader = load_ocr_reader()
        image = Image.open(uploaded).convert("RGB")
        ocr_results = reader.readtext(image)
        extracted_text = "\n".join([item[1] for item in ocr_results]).strip()

    with col2:
        st.subheader("🧾 Texto OCR")
        if extracted_text:
            st.text_area("Resultado OCR", extracted_text, height=300)
        else:
            st.info("No se detectó texto. Revisa la nitidez/idioma de la imagen.")
            st.stop()

    # Idioma
    lang = detect_lang(extracted_text, fallback="es") if force_lang == "Detectar automáticamente" else force_lang
    st.toast(f"Idioma de salida: {lang}", icon="🌐")

    # Prompt
    system_msg = human_system_prompt(lang, style)
    user_msg = f"Tarea: {task}\n\nTexto OCR:\n{extracted_text[:6000]}"  # evita contextos gigantes

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # LLM
    with st.spinner("Consultando LLM en Groq…"):
        try:
            completion = call_groq(model, messages, temperature, max_tokens)
            llm_text = completion.choices[0].message.content if completion.choices else ""
        except Exception as e:
            st.error("⚠️ Error al llamar a Groq. Revisa:\n- API key válida y con permisos\n- Modelo disponible\no intenta con otro modelo/menor tamaño.")
            with st.expander("Detalle técnico"):
                st.exception(e)
            st.stop()

    # Resultado
    st.subheader("🧠 Respuesta del LLM")
    if llm_text.strip():
        st.markdown(llm_text)
    else:
        st.info("No llegó contenido. Prueba con otro modelo o baja `temperature`.")

    # Metadatos mínimos
    with st.expander("Detalles de la ejecución"):
        st.write({
            "modelo": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "idioma_salida": lang,
            "tokens_llm_aprox": "depende del input (OCR truncado a 6000 chars)"
        })
else:
    st.info("Sube una imagen y pulsa **Procesar**.")
