"""
============================================================================
NAVIA Backend - Servicio de Gemini Vision (OCR Avanzado)
============================================================================
Usa Gemini 2.0 Flash para leer documentos desde fotos de celular.

En una sola llamada obtiene:
- Texto completo (OCR)
- Tipo de documento
- Campos estructurados (fechas, montos, emails, telefonos, IDs)
- Narrativa natural en español para TTS

Fallback: Si Gemini falla (sin internet, rate limit, API key invalida),
el SmartReadingService cae automaticamente a Tesseract local.

SDK: google-genai (Google Generative AI SDK v1)
============================================================================
"""

import json
import base64
import logging
import time
from typing import Optional, Dict, Any

import numpy as np
import cv2

from app.core.config import settings

logger = logging.getLogger(__name__)

# El prompt está diseñado para que Gemini devuelva JSON estructurado
# compatible con el formato que SmartReadingService espera.
_GEMINI_PROMPT = """Eres un asistente de lectura para personas ciegas. Analiza esta imagen de un documento o texto.

INSTRUCCIONES:
1. Extrae TODO el texto visible en la imagen (OCR completo)
2. Clasifica el tipo de documento
3. Extrae campos estructurados (fechas, montos, emails, teléfonos, identificaciones)
4. Genera una narrativa natural EN ESPAÑOL optimizada para lectura por voz (TTS)

TIPOS DE DOCUMENTO VÁLIDOS:
- "factura" (facturas, invoices)
- "recibo" (recibos de pago, comprobantes)
- "carta" (cartas formales, comunicados)
- "formulario" (formularios para llenar)
- "documento_informativo" (artículos, resoluciones, avisos oficiales)
- "etiqueta" (etiquetas de productos, información nutricional, medicamentos)
- "menu" (menús de restaurante, carta de comidas)
- "tarjeta" (tarjetas de presentación, business cards)
- "chat" (capturas de WhatsApp, Telegram, Instagram u otras apps de mensajería)
- "notificacion" (notificaciones push, alertas del sistema, avisos de apps)
- "login" (pantallas de inicio de sesión, registro, verificación)
- "red_social" (posts de Instagram, Twitter/X, TikTok, Facebook, YouTube, LinkedIn)
- "noticia" (noticias, artículos de prensa, blogs, portadas de revista)
- "correo" (emails, Gmail, Outlook, bandeja de entrada)
- "presentacion" (diapositivas, PowerPoint, Google Slides, Keynote)
- "configuracion" (pantallas de ajustes del teléfono, settings de apps)
- "imagen_visual" (imagen con muy poco o ningún texto, fotos)
- "desconocido" (si no encaja en ningún tipo)

REGLAS PARA LA NARRATIVA:
- Comienza indicando el tipo de documento: "Este documento es una factura."
- Para facturas/recibos: destaca montos totales y fecha
- Para cartas: menciona remitente, destinatario y asunto
- Para etiquetas: ingredientes clave, calorías, vencimiento
- Para menús: platos y precios
- Para chats: indica que es una conversación, lee los mensajes principales
- Para notificaciones: lee el contenido de la alerta
- Para login: indica qué servicio es, qué campos hay (usuario, contraseña)
- Para redes sociales: menciona la plataforma, usuario, contenido del post, interacciones
- Para noticias: lee el titular y resumen del contenido
- Para correos: menciona remitente, asunto y resumen del mensaje
- Para presentaciones: lee título de la diapositiva y contenido
- Para configuración: lista las opciones visibles en la pantalla
- Usa lenguaje natural, como si le describieras a alguien que no puede ver
- NO uses markdown, asteriscos, ni formatos especiales
- Máximo 200 palabras en la narrativa
- Si no hay texto legible, di "No se detectó texto claro en la imagen."

Responde ÚNICAMENTE con JSON válido (sin markdown, sin ```json):
{
  "raw_text": "texto completo extraído tal cual aparece",
  "document_type": "tipo del documento",
  "narrative": "narrativa natural en español para TTS",
  "extracted_fields": {
    "dates": ["lista de fechas encontradas"],
    "amounts": ["lista de montos monetarios"],
    "emails": ["lista de emails"],
    "phones": ["lista de teléfonos"],
    "ids": ["lista de identificaciones RIF/NIT/cédula"],
    "headers": ["encabezados o títulos principales"],
    "totals": ["líneas de totales ej: Total: $50.00"]
  },
  "word_count": 0,
  "confidence": 95
}"""


class GeminiService:
    """
    Servicio de OCR avanzado usando Gemini Vision.
    
    Envía la imagen directamente a Gemini (sin preprocesamiento)
    y obtiene OCR + clasificación + narrativa en una sola llamada.
    """

    # Segundos que se espera antes de volver a intentar Gemini tras un 429
    _RATE_LIMIT_COOLDOWN = 60
    # Reintentos para errores transitorios (timeout, error de red) — NO para 429
    _TRANSIENT_RETRIES = 1
    _TRANSIENT_RETRY_DELAY = 3  # segundos entre reintentos

    def __init__(self):
        self._client = None
        self._available = False
        self._init_error: Optional[str] = None
        self._rate_limited_until: float = 0  # circuit breaker timestamp
        self._initialize()

    def _initialize(self):
        """Inicializa el cliente de Gemini."""
        if not settings.GEMINI_ENABLED:
            self._init_error = "Gemini deshabilitado en configuración"
            logger.info("[Gemini] Deshabilitado en configuración")
            return

        if not settings.GEMINI_API_KEY:
            self._init_error = "GEMINI_API_KEY no configurada"
            logger.warning("[Gemini] API key no configurada. Usando solo Tesseract.")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self._available = True
            logger.info(f"[Gemini] Inicializado con modelo {settings.GEMINI_MODEL}")
        except ImportError:
            self._init_error = "google-genai no instalado (pip install google-genai)"
            logger.warning(f"[Gemini] {self._init_error}")
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"[Gemini] Error inicializando: {e}")

    @property
    def is_available(self) -> bool:
        """Indica si Gemini está listo para usar (respeta el cooldown de rate limit)."""
        if not self._available or self._client is None:
            return False
        if time.time() < self._rate_limited_until:
            remaining = int(self._rate_limited_until - time.time())
            logger.debug(f"[Gemini] En cooldown por rate limit, {remaining}s restantes")
            return False
        return True

    def analyze_document(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analiza un documento/imagen usando Gemini Vision.

        Args:
            image: Imagen en formato OpenCV (BGR numpy array)

        Returns:
            Dict con los campos del análisis, o None si falla.
            Formato compatible con SmartReadingService.
        """
        if not self.is_available:
            logger.debug(f"[Gemini] No disponible: {self._init_error}")
            return None

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return None

        # Codificar imagen una sola vez (se reutiliza en reintentos)
        success, jpeg_bytes = cv2.imencode(
            '.jpg', image,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not success:
            logger.error("[Gemini] Error codificando imagen a JPEG")
            return None
        image_bytes = jpeg_bytes.tobytes()
        logger.info(f"[Gemini] Enviando imagen ({len(image_bytes) // 1024} KB)")

        gen_config_kwargs: dict = {
            "temperature": 0.1,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",
        }
        try:
            gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except AttributeError:
            pass

        last_error: Optional[Exception] = None
        attempts = 1 + self._TRANSIENT_RETRIES

        for attempt in range(attempts):
            try:
                response = self._client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        _GEMINI_PROMPT,
                    ],
                    config=types.GenerateContentConfig(**gen_config_kwargs),
                )

                response_text = response.text
                if not response_text:
                    logger.warning("[Gemini] Respuesta vacía")
                    return None

                logger.info(f"[Gemini] Respuesta recibida ({len(response_text)} chars)")
                result = self._parse_response(response_text)
                if result is None:
                    return None

                logger.info(
                    f"[Gemini] Documento: {result.get('document_type', '?')}, "
                    f"{result.get('word_count', 0)} palabras, "
                    f"confianza: {result.get('confidence', 0)}%"
                )
                return result

            except Exception as e:
                error_str = str(e)
                last_error = e

                if "429" in error_str or "quota" in error_str.lower() or "resource_exhausted" in error_str.lower():
                    # Rate limit: activar circuit breaker, no reintentar
                    self._rate_limited_until = time.time() + self._RATE_LIMIT_COOLDOWN
                    logger.warning(
                        f"[Gemini] Rate limit alcanzado. "
                        f"Cooldown de {self._RATE_LIMIT_COOLDOWN}s activado."
                    )
                    return None

                if "403" in error_str or "api_key" in error_str.lower():
                    logger.error("[Gemini] API key inválida o sin permisos")
                    return None

                is_transient = (
                    "timeout" in error_str.lower()
                    or "deadline" in error_str.lower()
                    or "503" in error_str
                    or "500" in error_str
                    or "connection" in error_str.lower()
                )
                if is_transient and attempt < attempts - 1:
                    logger.warning(
                        f"[Gemini] Error transitorio (intento {attempt + 1}/{attempts}): {e}. "
                        f"Reintentando en {self._TRANSIENT_RETRY_DELAY}s…"
                    )
                    time.sleep(self._TRANSIENT_RETRY_DELAY)
                    continue

                logger.warning(f"[Gemini] Error: {e}")
                return None

        logger.warning(f"[Gemini] Falló tras {attempts} intentos: {last_error}")
        return None

    def classify_scene(self, image: np.ndarray) -> Optional[str]:
        """
        Clasifica el entorno visible en una frase corta para el inicio de navegación.
        Llamada síncrona — ejecutar desde run_in_executor para no bloquear el event loop.

        Returns:
            Frase como "Estás en la calle." o None si falla.
        """
        if not self.is_available:
            return None

        try:
            from google import genai
            from google.genai import types

            success, jpeg_bytes = cv2.imencode(
                '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if not success:
                return None

            prompt = (
                "Eres un asistente para personas ciegas. "
                "Mira la imagen y di en una sola frase corta en qué tipo de lugar está la persona, "
                "empezando siempre con 'Estás en'. "
                "Ejemplos: 'Estás en la calle.', 'Estás en un parque.', "
                "'Estás en una habitación.', 'Estás en un supermercado.', "
                "'Estás en una oficina.', 'Estás en un transporte público.', "
                "'Estás en un pasillo.', 'Estás en un restaurante.'. "
                "Responde ÚNICAMENTE con esa frase, sin explicaciones."
            )

            gen_config_kwargs: Dict[str, Any] = {
                "temperature": 0.1,
                "max_output_tokens": 60,
            }
            try:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except AttributeError:
                pass

            response = self._client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=jpeg_bytes.tobytes(), mime_type="image/jpeg"
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(**gen_config_kwargs),
            )

            text = response.text.strip() if response.text else None
            if text:
                if not text.endswith('.'):
                    text += '.'
                logger.info(f"[SceneIntro] {text}")
            return text

        except Exception as e:
            logger.warning(f"[SceneIntro] Gemini error: {e}")
            return None

    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parsea la respuesta JSON de Gemini.
        Maneja casos donde Gemini envuelve el JSON en markdown.
        """
        text = response_text.strip()

        # Remover markdown code block si Gemini lo envuelve
        if text.startswith("```"):
            # Buscar el primer newline después de ```
            first_newline = text.index('\n') if '\n' in text else 3
            text = text[first_newline + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"[Gemini] JSON inválido: {e}")
            logger.warning(f"[Gemini] Respuesta raw ({len(response_text)} chars): {response_text[:500]}")
            return None

        # Validar campos requeridos
        if not isinstance(data, dict):
            logger.warning("[Gemini] Respuesta no es un diccionario")
            return None

        # Asegurar que todos los campos existen con valores por defecto
        result = {
            "raw_text": data.get("raw_text", ""),
            "document_type": data.get("document_type", "desconocido"),
            "narrative": data.get("narrative", ""),
            "word_count": data.get("word_count", 0),
            "confidence": data.get("confidence", 0),
            "extracted_fields": {
                "dates": [],
                "amounts": [],
                "emails": [],
                "phones": [],
                "ids": [],
                "headers": [],
                "totals": [],
            },
        }

        # Parsear extracted_fields si existe
        ef = data.get("extracted_fields", {})
        if isinstance(ef, dict):
            for field in ["dates", "amounts", "emails", "phones", "ids", "headers", "totals"]:
                val = ef.get(field, [])
                if isinstance(val, list):
                    result["extracted_fields"][field] = [str(v) for v in val if v]

        # Validar tipo de documento
        valid_types = {
            "factura", "recibo", "carta", "formulario",
            "documento_informativo", "etiqueta", "menu",
            "tarjeta", "chat", "notificacion", "login",
            "red_social", "noticia", "correo", "presentacion",
            "configuracion", "imagen_visual", "desconocido",
        }
        if result["document_type"] not in valid_types:
            logger.info(f"[Gemini] Tipo '{result['document_type']}' no reconocido, usando 'desconocido'")
            result["document_type"] = "desconocido"

        # Calcular word_count si no lo devolvió
        if result["word_count"] == 0 and result["raw_text"]:
            result["word_count"] = len(result["raw_text"].split())

        # Validar narrative
        if not result["narrative"]:
            if result["raw_text"]:
                result["narrative"] = f"Se detectó texto. {' '.join(result['raw_text'].split()[:50])}"
            else:
                result["narrative"] = "No se detectó texto claro en la imagen."

        return result


# ============================================================================
# SINGLETON
# ============================================================================

_gemini_service: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Factory function para obtener el servicio Gemini (singleton)."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
