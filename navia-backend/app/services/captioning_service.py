"""
============================================================================
NAVIA Backend - Servicio de Captioning (Florence-2)
============================================================================
Genera descripciones naturales de escenas usando Florence-2, un modelo
multimodal de Microsoft que soporta múltiples tareas de visión.

Florence-2-base (~0.5B params, ~1GB RAM):
- Soporta tareas: <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>
- Genera captions en inglés que se combinan con datos YOLO en español
- Tiempo de inferencia: ~2-4 segundos en CPU

IMPORTANTE:
- Solo se usa en endpoints HTTP (/analyze/scene, /analyze/exploracion)
- NUNCA se usa en WebSocket tiempo real (demasiado lento)
- Se carga lazy: solo cuando se necesita por primera vez
- Fallback silencioso si el modelo no está disponible
============================================================================
"""

import numpy as np
import cv2
import logging
from typing import Optional, Dict, List

from app.core.config import settings
from app.models.schemas import DetectedObject

logger = logging.getLogger(__name__)

# ============================================================================
# Diccionario de traducción para conceptos comunes de Florence-2
# No es traducción completa, sino mapeo de patrones frecuentes
# para enriquecer la descripción en español
# ============================================================================

_SCENE_TRANSLATIONS: Dict[str, str] = {
    # Escenas / ambientes
    "living room": "sala de estar",
    "bedroom": "dormitorio",
    "kitchen": "cocina",
    "bathroom": "baño",
    "office": "oficina",
    "dining room": "comedor",
    "hallway": "pasillo",
    "garage": "garaje",
    "garden": "jardín",
    "park": "parque",
    "street": "calle",
    "sidewalk": "acera",
    "parking lot": "estacionamiento",
    "store": "tienda",
    "restaurant": "restaurante",
    "classroom": "aula",
    "hospital": "hospital",
    "lobby": "vestíbulo",
    "staircase": "escaleras",
    "elevator": "ascensor",
    # Descriptores
    "well-lit": "bien iluminado",
    "dimly lit": "con poca luz",
    "bright": "brillante",
    "dark": "oscuro",
    "cluttered": "desordenado",
    "clean": "limpio",
    "spacious": "espacioso",
    "narrow": "estrecho",
    "crowded": "concurrido",
    "empty": "vacío",
    "outdoor": "exterior",
    "indoor": "interior",
    # Elementos comunes
    "wooden floor": "piso de madera",
    "carpet": "alfombra",
    "tile floor": "piso de baldosa",
    "ceiling": "techo",
    "wall": "pared",
    "window": "ventana",
    "door": "puerta",
    "table": "mesa",
    "chair": "silla",
    "couch": "sofá",
    "bed": "cama",
    "desk": "escritorio",
    "shelf": "estante",
    "lamp": "lámpara",
    "television": "televisor",
    "computer": "computadora",
    "phone": "teléfono",
    "book": "libro",
    "plant": "planta",
    "picture": "cuadro",
    "mirror": "espejo",
    "curtain": "cortina",
    "rug": "alfombra",
}


class CaptioningService:
    """
    Servicio de captioning de escenas usando Florence-2.

    Genera descripciones naturales que complementan la detección de objetos
    YOLO-World, proporcionando contexto de escena que YOLO no captura
    (ej: "una cocina iluminada" vs solo "estufa, refrigerador, mesa").

    El caption en inglés se usa como contexto para construir una
    descripción enriquecida en español combinada con datos YOLO.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self._initialized = False
        self._available = True

    def _ensure_initialized(self) -> None:
        """
        Carga lazy del modelo Florence-2.

        Solo se ejecuta una vez. Si falla, marca el servicio como
        no disponible para evitar reintentos costosos.
        """
        if self._initialized:
            return

        self._initialized = True

        if not settings.CAPTIONING_ENABLED:
            self._available = False
            logger.info("Captioning desactivado (CAPTIONING_ENABLED=False)")
            return

        try:
            from transformers import AutoProcessor, AutoModelForCausalLM

            model_name = settings.CAPTIONING_MODEL
            logger.info(f"Cargando modelo de captioning: {model_name}")

            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model.eval()

            self._available = True
            logger.info("Modelo Florence-2 cargado exitosamente")

        except Exception as e:
            logger.warning(
                f"No se pudo cargar Florence-2: {e}. "
                "Las descripciones de escena usarán solo YOLO + reglas."
            )
            self._available = False

    @property
    def is_available(self) -> bool:
        """Indica si el modelo de captioning está disponible."""
        self._ensure_initialized()
        return self._available and self.model is not None

    def generate_caption(
        self, image: np.ndarray, detailed: bool = False
    ) -> Optional[str]:
        """
        Genera un caption en inglés para la imagen.

        Args:
            image: Imagen BGR (OpenCV format)
            detailed: Si True, usa <DETAILED_CAPTION> para más detalle

        Returns:
            Caption en inglés, o None si falla
        """
        self._ensure_initialized()
        if not self._available or self.model is None:
            return None

        try:
            import torch
            from PIL import Image as PILImage

            # Convertir BGR → RGB → PIL
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb)

            # Elegir task prompt
            task = "<DETAILED_CAPTION>" if detailed else "<CAPTION>"

            # Preparar inputs
            inputs = self.processor(
                text=task,
                images=pil_image,
                return_tensors="pt",
            )

            # Generar caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=128,
                    num_beams=3,
                    do_sample=False,
                )

            # Decodificar
            caption = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            # Post-procesar: extraer texto entre tags de tarea
            caption = self.processor.post_process_generation(
                caption, task=task, image_size=pil_image.size
            )

            # El resultado es un dict con la tarea como key
            if isinstance(caption, dict):
                caption = caption.get(task, "")

            return caption.strip() if caption else None

        except Exception as e:
            logger.error(f"Error generando caption: {e}")
            return None

    def generate_scene_description_spanish(
        self,
        image: np.ndarray,
        detected_objects: Optional[List[DetectedObject]] = None,
        detailed: bool = True,
    ) -> str:
        """
        Genera una descripción de escena en español combinando Florence-2
        con los datos de objetos detectados por YOLO-World.

        Estrategia híbrida:
        1. Florence-2 genera caption en inglés (contexto general de escena)
        2. Se traduce parcialmente con diccionario estático
        3. Se enriquece con nombres de objetos YOLO (ya en español)
        4. Se genera una oración natural en español

        Args:
            image: Imagen BGR (OpenCV format)
            detected_objects: Objetos detectados por YOLO (opcional)
            detailed: Si True, usa caption detallado

        Returns:
            Descripción en español (string vacío si falla)
        """
        # Obtener caption de Florence-2
        caption_en = self.generate_caption(image, detailed=detailed)
        if not caption_en:
            return ""

        # Traducir conceptos conocidos del caption
        caption_es = self._translate_caption(caption_en)

        # Si hay objetos detectados, enriquecer la descripción
        if detected_objects:
            scene_context = self._build_scene_context(
                caption_es, detected_objects
            )
            return scene_context

        return caption_es

    def _translate_caption(self, caption_en: str) -> str:
        """
        Traduce un caption del inglés al español usando diccionario estático.

        No es traducción perfecta, sino reemplazo de conceptos clave
        para dar contexto en español.

        Args:
            caption_en: Caption en inglés de Florence-2

        Returns:
            Caption parcialmente traducido al español
        """
        result = caption_en.lower()

        # Reemplazar frases conocidas (más largas primero para evitar colisiones)
        sorted_translations = sorted(
            _SCENE_TRANSLATIONS.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )
        for en, es in sorted_translations:
            result = result.replace(en, es)

        return result

    def _build_scene_context(
        self,
        caption_es: str,
        objects: List[DetectedObject],
    ) -> str:
        """
        Combina el caption traducido con datos de objetos YOLO para
        generar una descripción contextual coherente en español.

        Args:
            caption_es: Caption parcialmente traducido
            objects: Objetos detectados por YOLO con nombres en español

        Returns:
            Descripción contextual en español
        """
        # Extraer los objetos principales (top 5 por confianza)
        top_objects = sorted(objects, key=lambda o: o.confidence, reverse=True)[:5]
        object_names = list(dict.fromkeys(o.name_es for o in top_objects))

        # Detectar tipo de escena del caption
        scene_type = self._detect_scene_type(caption_es)

        # Construir descripción
        parts = []

        if scene_type:
            parts.append(f"La escena muestra {scene_type}")
        else:
            parts.append("En la escena se observa")

        if object_names:
            if len(object_names) == 1:
                parts[0] += f" con {object_names[0]}"
            elif len(object_names) == 2:
                parts[0] += f" con {object_names[0]} y {object_names[1]}"
            else:
                last = object_names[-1]
                rest = ", ".join(object_names[:-1])
                parts[0] += f" con {rest} y {last}"

        # Agregar información de distancia si hay objetos cercanos
        close_objects = [
            o for o in top_objects if o.distance_zone == "muy_cerca"
        ]
        if close_objects:
            close_names = list(dict.fromkeys(o.name_es for o in close_objects))
            if len(close_names) == 1:
                parts.append(f"{close_names[0].capitalize()} muy cerca")
            else:
                parts.append("Varios objetos muy cerca")

        return ". ".join(parts) + "."

    @staticmethod
    def _detect_scene_type(caption: str) -> Optional[str]:
        """
        Detecta el tipo de escena a partir del caption.

        Returns:
            Descripción del tipo de escena en español, o None
        """
        scene_keywords = {
            "sala de estar": "una sala de estar",
            "dormitorio": "un dormitorio",
            "cocina": "una cocina",
            "baño": "un baño",
            "oficina": "una oficina",
            "comedor": "un comedor",
            "pasillo": "un pasillo",
            "garaje": "un garaje",
            "jardín": "un jardín",
            "parque": "un parque",
            "calle": "una calle",
            "acera": "una acera",
            "estacionamiento": "un estacionamiento",
            "tienda": "una tienda",
            "restaurante": "un restaurante",
            "aula": "un aula",
            "escaleras": "unas escaleras",
        }

        caption_lower = caption.lower()
        for keyword, description in scene_keywords.items():
            if keyword in caption_lower:
                return description

        return None


# Singleton
_captioning_service: Optional[CaptioningService] = None


def get_captioning_service() -> CaptioningService:
    """Factory function para obtener el servicio de captioning."""
    global _captioning_service
    if _captioning_service is None:
        _captioning_service = CaptioningService()
    return _captioning_service
