# main.py
import io
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

import easyocr  # OCR
from groq import Groq  # LLM (Groq SDK)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Falta GROQ_API_KEY en .env")

app = FastAPI(title="OCR + LLM (Groq) API", version="0.1.0")

# Inicializamos OCR una vez (más rápido que por request)
# Idiomas: agrega códigos según necesites, ej. ["es", "en"]
reader = easyocr.Reader(["es", "en"], gpu=False)

# Cliente Groq
client = Groq(api_key=GROQ_API_KEY)

class LLMRequest(BaseModel):
    task: Optional[str] = "Resume el texto y extrae los datos clave en JSON."
    model: Optional[str] = "llama-3.3-70b-versatile"
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 800

@app.post("/ocr-llm")
async def ocr_llm_endpoint(
    image: UploadFile = File(..., description="Imagen con texto"),
    task: Optional[str] = Form(None),
    model: Optional[str] = Form("llama-3.3-70b-versatile"),
    temperature: Optional[float] = Form(0.2),
    max_tokens: Optional[int] = Form(800),
):
    """
    1) Lee imagen -> OCR (EasyOCR)
    2) Envía el texto al LLM (Groq) con la 'task' indicada
    3) Devuelve JSON con el texto extraído y la respuesta del LLM
    """
    try:
        # Cargar imagen
        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        # OCR
        # readtext retorna lista de [bbox, text, conf]; nos quedamos con los textos
        ocr_results = reader.readtext(img)
        extracted_text = "\n".join([item[1] for item in ocr_results]).strip()

        if not extracted_text:
            return JSONResponse(
                status_code=200,
                content={
                    "extracted_text": "",
                    "llm_output": "",
                    "note": "No se detectó texto en la imagen."
                },
            )

        # Prompt al LLM
        user_task = task or "Resume el texto y extrae los datos clave en JSON."
        messages = [
            {"role": "system", "content": "Eres un asistente que devuelve respuestas útiles y, cuando se pida, JSON válido."},
            {"role": "user", "content": f"Tarea: {user_task}\n\nTexto OCR:\n{extracted_text}"}
        ]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(temperature),
            max_completion_tokens=int(max_tokens),
        )
        llm_text = completion.choices[0].message.content if completion.choices else ""

        return {
            "extracted_text": extracted_text,
            "llm_output": llm_text,
            "model": model,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"ok": True, "service": "OCR + LLM (Groq)"}
