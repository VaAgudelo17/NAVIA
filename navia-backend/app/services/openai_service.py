"""
Servicio OpenAI para NAVIA - Validación y narrativa empática en modo exploración.

Usa GPT-4o Vision para dos tareas sobre las detecciones de YOLO:
1. Validar cuáles objetos están realmente en la imagen (descarta falsos positivos)
2. Generar una descripción natural y empática para el usuario con discapacidad visual

Si el servicio no está disponible (sin API key, error de red, rate limit),
exploration_service cae automáticamente al generador de narrativa local.
"""

import base64
import io
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpenAIService:

    def __init__(self):
        self._client = None
        self._model = "gpt-4o"
        self._enabled = False
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._enabled

        self._initialized = True

        try:
            from app.core.config import settings

            if not settings.OPENAI_API_KEY or not settings.OPENAI_ENABLED:
                logger.info("[OpenAI] Desactivado o sin API key configurada")
                return False

            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.OPENAI_TIMEOUT,
            )
            self._model = settings.OPENAI_MODEL
            self._enabled = True
            logger.info(f"[OpenAI] Inicializado correctamente con modelo {self._model}")
            return True

        except ImportError:
            logger.warning(
                "[OpenAI] Paquete 'openai' no instalado. "
                "Ejecuta: pip install openai>=1.0.0"
            )
            return False
        except Exception as e:
            logger.error(f"[OpenAI] Error al inicializar: {e}")
            return False

    def validate_and_describe(
        self,
        image: np.ndarray,
        enriched_objects: List[Dict],
    ) -> Optional[str]:
        """
        Valida las detecciones de YOLO y genera una descripción empática.

        Args:
            image: Imagen en formato OpenCV (BGR)
            enriched_objects: Objetos enriquecidos por el pipeline de exploración
                              (cada uno tiene name_es, position, distance_label, confidence)

        Returns:
            Descripción en español lista para TTS, o None si el servicio no está disponible.
        """
        if not self._ensure_initialized():
            return None

        try:
            image_b64 = self._encode_image(image)
            if not image_b64:
                return None

            objects_text = self._build_objects_text(enriched_objects)
            prompt = self._build_prompt(objects_text)

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    # "high" para que GPT-4o pueda identificar objetos
                                    # pequeños: gafas, esmalte, tablet, etc.
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=150,
                temperature=0.4,
            )

            result = response.choices[0].message.content
            if result:
                result = result.strip()
                logger.info(f"[OpenAI] Descripción generada: {result[:100]}")
                return result

            return None

        except Exception as e:
            logger.warning(f"[OpenAI] Falló la generación de descripción: {e}")
            return None

    def _encode_image(self, image: np.ndarray) -> Optional[str]:
        """Codifica la imagen BGR a JPEG base64, redimensionando si supera 768px."""
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = rgb.shape[:2]
            max_dim = 768
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                rgb = cv2.resize(
                    rgb,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            from PIL import Image

            pil_img = Image.fromarray(rgb)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)

            return base64.b64encode(buffer.read()).decode("utf-8")

        except Exception as e:
            logger.debug(f"[OpenAI] Error codificando imagen: {e}")
            return None

    def _build_objects_text(self, enriched_objects: List[Dict]) -> str:
        """
        Envía solo posición y distancia a GPT-4o, sin los nombres de YOLO.
        Así GPT-4o identifica libremente qué hay en cada zona sin ser anclado
        por las (posiblemente erróneas) etiquetas del detector.
        """
        lines = []
        for i, obj in enumerate(enriched_objects, 1):
            position = obj.get("position", {}).get("combined", "frente a ti")
            distance = obj.get("distance_label", "")
            line = f"- Zona {i}: {position}"
            if distance:
                line += f", {distance}"
            lines.append(line)
        return "\n".join(lines)

    def _build_prompt(self, objects_text: str) -> str:
        return (
            "Eres NAVIA, asistente visual para personas con discapacidad visual.\n\n"
            "El sensor detectó actividad en estas zonas de la imagen:\n"
            f"{objects_text}\n\n"
            "Tu tarea: genera una descripción breve (máximo 2 oraciones) de los objetos "
            "presentes, de forma cálida y empática, como si guiaras a alguien con los ojos cerrados.\n\n"
            "Reglas estrictas:\n"
            "- Describe SOLO objetos concretos: muebles, ropa, accesorios, electrodomésticos, "
            "personas, animales, alimentos, etc.\n"
            "- IGNORA completamente: paredes, piso, techo, cielo, suelo, fondo, superficies "
            "genéricas y cualquier elemento de ambiente o textura.\n"
            "- NO menciones texto, letreros, palabras ni números visibles en la imagen.\n"
            "- Indica posición solo cuando sea útil: \"Frente a ti...\", "
            "\"A tu izquierda...\", \"A tu derecha...\".\n"
            "- Lenguaje natural y directo. Sin tecnicismos.\n\n"
            "Responde ÚNICAMENTE con la descripción en español, sin explicaciones adicionales."
        )


_openai_service: Optional[OpenAIService] = None


def get_openai_service() -> OpenAIService:
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service
