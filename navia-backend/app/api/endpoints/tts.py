"""
============================================================================
NAVIA Backend - Endpoint de Text-to-Speech
============================================================================
Convierte texto en español a audio WAV usando Piper TTS.
Usado por la app móvil para leer descripciones con voz natural.
============================================================================
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["Text-to-Speech"])


class TtsRequest(BaseModel):
    """Solicitud de síntesis de voz."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Texto en español a sintetizar",
    )


@router.post(
    "",
    summary="Sintetizar voz desde texto",
    description="Convierte texto en español a audio WAV usando Piper TTS (modelo VITS).",
    response_class=Response,
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Audio WAV sintetizado",
        },
        503: {"description": "TTS no disponible"},
    },
)
async def synthesize_speech(request: TtsRequest) -> Response:
    """Convierte texto a audio WAV."""
    if not settings.TTS_ENABLED:
        raise HTTPException(status_code=503, detail="TTS deshabilitado en configuración")

    try:
        from app.services.tts_service import get_tts_service

        tts_service = get_tts_service()
        audio_data = tts_service.synthesize(request.text)

        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-cache",
            },
        )

    except RuntimeError as e:
        logger.error(f"TTS no disponible: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en TTS: {e}")
        raise HTTPException(status_code=500, detail=f"Error en síntesis: {str(e)}")
