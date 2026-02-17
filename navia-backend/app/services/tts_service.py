"""
============================================================================
NAVIA Backend - Servicio de Text-to-Speech (Piper TTS)
============================================================================
Genera audio WAV a partir de texto en español usando Piper TTS.
Piper usa modelos VITS (ONNX) que corren eficientemente en CPU.

El modelo se descarga automáticamente la primera vez (~100MB).
Inferencia típica: ~0.5-1.5 segundos por oración en CPU.
============================================================================
"""

import io
import wave
import logging
from typing import Optional
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class PiperTtsService:
    """
    Servicio de síntesis de voz con Piper TTS.

    Carga el modelo VITS en español de forma lazy (al primer uso).
    Genera audio WAV mono 16-bit.
    """

    def __init__(self):
        self._voice = None
        self._synthesize_fn = None
        self._loaded = False

    def _ensure_loaded(self):
        """Carga el modelo Piper en el primer uso."""
        if self._loaded:
            return

        try:
            from piper import PiperVoice

            model_name = settings.TTS_MODEL_NAME

            # Buscar modelo en directorio de modelos
            models_dir = Path(settings.MODELS_DIR) if hasattr(settings, 'MODELS_DIR') else Path("models")
            model_path = models_dir / f"{model_name}.onnx"
            config_path = models_dir / f"{model_name}.onnx.json"

            if model_path.exists() and config_path.exists():
                logger.info(f"Cargando modelo Piper TTS desde: {model_path}")
                self._voice = PiperVoice.load(str(model_path), str(config_path))
            else:
                # Intentar descarga automática
                logger.info(f"Modelo no encontrado en {model_path}, intentando descarga...")
                self._voice = PiperVoice.load(model_name)

            self._loaded = True
            logger.info(f"Piper TTS cargado: {model_name}")

        except ImportError:
            logger.warning("piper-tts no instalado. TTS backend no disponible.")
            raise RuntimeError("piper-tts no está instalado")
        except Exception as e:
            logger.error(f"Error cargando Piper TTS: {e}")
            raise RuntimeError(f"No se pudo cargar Piper TTS: {e}")

    def synthesize(self, text: str) -> bytes:
        """
        Sintetiza texto a audio WAV.

        Args:
            text: Texto en español a sintetizar

        Returns:
            Audio WAV como bytes (mono, 16-bit, 22050 Hz)
        """
        self._ensure_loaded()

        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")

        # Truncar textos muy largos
        max_len = settings.TTS_MAX_TEXT_LENGTH
        if len(text) > max_len:
            text = text[:max_len]

        sample_rate = settings.TTS_SAMPLE_RATE

        try:
            audio_buffer = io.BytesIO()

            with wave.open(audio_buffer, 'wb') as wav_file:
                # synthesize_wav configura formato WAV automáticamente
                self._voice.synthesize_wav(text, wav_file)

            audio_buffer.seek(0)
            return audio_buffer.read()

        except Exception as e:
            logger.error(f"Error sintetizando audio: {e}")
            raise RuntimeError(f"Error en síntesis TTS: {e}")

    def get_info(self) -> dict:
        """Información del servicio TTS."""
        return {
            "status": "loaded" if self._loaded else "not_loaded",
            "model": settings.TTS_MODEL_NAME,
            "sample_rate": settings.TTS_SAMPLE_RATE,
            "enabled": settings.TTS_ENABLED,
        }


# ============================================================================
# SINGLETON
# ============================================================================

_tts_service: Optional[PiperTtsService] = None


def get_tts_service() -> PiperTtsService:
    """Obtiene la instancia singleton del servicio TTS."""
    global _tts_service
    if _tts_service is None:
        _tts_service = PiperTtsService()
    return _tts_service
