"""
========================================================================
NAVIA Backend - Servicio de Text-to-Speech (Piper TTS + gTTS Fallback)
========================================================================
Genera audio WAV a partir de texto en español usando Piper TTS.
Si Piper no está disponible, usa gTTS como fallback.

Piper usa modelos VITS (ONNX) que corren eficientemente en CPU.
gTTS usa los servicios de Google para síntesis de voz.
========================================================================
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
    Servicio de síntesis de voz con Piper TTS y gTTS fallback.
    """

    def __init__(self):
        self._voice = None
        self._loaded = False
        self._use_gtts = False

    def _ensure_loaded(self):
        """Carga el backend TTS configurado (gTTS o Piper)."""
        if self._loaded:
            return

        backend = getattr(settings, 'TTS_BACKEND', 'gtts')

        if backend == 'gtts':
            # gTTS: voz femenina de Google, requiere internet en el servidor
            logger.info("TTS backend: gTTS (voz femenina Google)")
            self._use_gtts = True
            self._loaded = True
            return

        # backend == 'piper'
        try:
            from piper import PiperVoice

            model_name = settings.TTS_MODEL_NAME
            models_dir = Path(settings.MODELS_DIR) if hasattr(settings, 'MODELS_DIR') else Path("models")
            model_path = models_dir / f"{model_name}.onnx"
            config_path = models_dir / f"{model_name}.onnx.json"

            if model_path.exists() and config_path.exists():
                logger.info(f"Cargando modelo Piper TTS desde: {model_path}")
                try:
                    self._voice = PiperVoice.load(str(model_path), config_path=str(config_path))
                except TypeError:
                    self._voice = PiperVoice.load(str(model_path), str(config_path))
                self._loaded = True
                logger.info(f"Piper TTS cargado: {model_name}")
            else:
                raise FileNotFoundError(
                    f"Modelo Piper no encontrado: {model_path}. "
                    f"Cambia TTS_BACKEND=gtts o descarga el modelo."
                )

        except ImportError:
            logger.warning("piper-tts no instalado. Usando gTTS como fallback.")
            self._use_gtts = True
            self._loaded = True
        except Exception as e:
            logger.warning(f"Error cargando Piper TTS: {e}. Usando gTTS como fallback.")
            self._use_gtts = True
            self._loaded = True

    def synthesize(self, text: str) -> bytes:
        """
        Sintetiza texto a audio WAV.
        """
        self._ensure_loaded()

        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")

        max_len = settings.TTS_MAX_TEXT_LENGTH
        if len(text) > max_len:
            text = text[:max_len]

        if self._use_gtts:
            return self._synthesize_gtts(text)
        else:
            return self._synthesize_piper(text)

    def _synthesize_piper(self, text: str) -> bytes:
        """Síntesis con Piper TTS."""
        from piper.config import SynthesisConfig

        try:
            audio_buffer = io.BytesIO()

            # length_scale > 1 = más lento → más natural y menos robótico
            # noise_scale controla variación de tono (más variado = menos monótono)
            syn_config = SynthesisConfig(
                length_scale=1.15,   # 15% más lento que el default
                noise_scale=0.667,   # variación de tono natural (default Piper)
                noise_w_scale=0.8,   # variación de duración de fonemas
            )

            with wave.open(audio_buffer, 'wb') as wav_file:
                self._voice.synthesize_wav(text, wav_file, syn_config=syn_config)

            audio_buffer.seek(0)
            return audio_buffer.read()

        except Exception as e:
            logger.error(f"Error sintetizando con Piper: {e}")
            raise RuntimeError(f"Error en síntesis TTS: {e}")

    def _synthesize_gtts(self, text: str) -> bytes:
        """Síntesis con gTTS (Google TTS) como fallback."""
        try:
            from gtts import gTTS

            mp3_buffer = io.BytesIO()
            tts = gTTS(text=text, lang='es', slow=False)
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)

            return mp3_buffer.read()

        except ImportError:
            raise RuntimeError("gTTS no está instalado. Ejecuta: pip install gTTS")
        except Exception as e:
            logger.error(f"Error sintetizando con gTTS: {e}")
            raise RuntimeError(f"Error en síntesis TTS: {e}")

    def get_info(self) -> dict:
        """Información del servicio TTS."""
        return {
            "status": "loaded" if self._loaded else "not_loaded",
            "backend": "gTTS" if self._use_gtts else "piper",
            "model": settings.TTS_MODEL_NAME,
            "sample_rate": settings.TTS_SAMPLE_RATE,
            "enabled": settings.TTS_ENABLED,
        }


_tts_service: Optional[PiperTtsService] = None


def get_tts_service() -> PiperTtsService:
    """Obtiene la instancia singleton del servicio TTS."""
    global _tts_service
    if _tts_service is None:
        _tts_service = PiperTtsService()
    return _tts_service


def reset_tts_service() -> None:
    """Fuerza recarga del servicio TTS (útil tras instalar modelo)."""
    global _tts_service
    _tts_service = None
