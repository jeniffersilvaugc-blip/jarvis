import os

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import logging
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env
load_dotenv()

logging.basicConfig(level=logging.INFO)

# Optionally create an OpenAI client at startup if available
client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

'''
JSON
{
 "mensagem": "<qualquer texto>",
 "id_anterior": "<opcional>"
}
'''
class ChatMessage(BaseModel):
    mensagem: str
    id_anterior: Optional[str] = None


def _extract_response_text(resp):
    """Extrai o texto final do retorno da API de respostas do OpenAI."""
    try:
        output_text = getattr(resp, "output_text", None)
        if output_text:
            return output_text.strip()
    except Exception:
        pass

    try:
        if getattr(resp, "output", None):
            for item in resp.output:
                if getattr(item, "content", None):
                    for content in item.content:
                        text = getattr(content, "text", None)
                        if text:
                            return str(text).strip()
    except Exception:
        pass

    try:
        if resp["output"]:
            for item in resp["output"]:
                for content in item.get("content", []):
                    text = content.get("text")
                    if text:
                        return str(text).strip()
    except Exception:
        pass

    return None


@app.post("/chat")
async def chat(message: ChatMessage):
    mensagem = (message.mensagem or "").strip()
    if not mensagem:
        raise HTTPException(status_code=400, detail="mensagem não pode estar vazia")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")

    global client
    if client is None:
        client = OpenAI(api_key=api_key)

    request_kwargs = {
        "model": "gpt-4o-mini",
        "input": [
            {
                "role": "system",
                "content": (
                    "Você é o Jarvis, um assistente virtual do SENAI, e sua conversa deve parecer natural, humana e espontânea. "
                    "Fale em português do Brasil com linguagem clara, acolhedora e profissional. "
                    "Responda como uma pessoa que conversa de verdade: direta, amigável, útil e sem soar robótico. "
                    "Evite frases prontas, repetições, excesso de formalidade e respostas genéricas. "
                    "Se a pessoa estiver buscando ajuda, seja prestativo e objetivo, mas com um toque humano. "
                    "Explique em linguagem simples, e se for preciso, dê passos claros. "
                    "Atenda dúvidas sobre cursos, matrículas, suporte, orientação e assuntos relacionados ao SENAI. "
                    "Se não souber algo, diga com honestidade e sugira o próximo passo mais útil."
                ),
            },
            {"role": "user", "content": mensagem},
        ],
        "max_output_tokens": 500,
        "temperature": 0.8,
    }

    if message.id_anterior:
        request_kwargs["previous_response_id"] = message.id_anterior

    try:
        resp = client.responses.create(**request_kwargs)
        reply = _extract_response_text(resp)

        if not reply:
            reply = "Não consegui montar uma resposta adequada agora. Tente novamente em alguns instantes."

    except Exception as e:
        tb = traceback.format_exc()
        logging.exception("Error calling OpenAI Responses API")
        last_line = tb.splitlines()[-1] if tb else ""
        detail = f"{type(e).__name__}: {str(e)} -- {last_line}"
        raise HTTPException(status_code=500, detail=detail)

    response_id = getattr(resp, "id", None)
    return {"mensagem": mensagem, "resposta": reply, "id": response_id}