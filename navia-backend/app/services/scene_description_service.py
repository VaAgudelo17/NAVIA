"""
============================================================================
NAVIA Backend - Servicio de Descripción de Escenas
============================================================================
Este módulo combina OCR y detección de objetos para generar descripciones
completas de imágenes, optimizadas para ser convertidas a audio (TTS).

Arquitectura:
- Utiliza OCRService para extracción de texto
- Utiliza ObjectDetectionService para identificar objetos
- Combina ambos resultados en una descripción coherente

Diseño para accesibilidad:
- Descripciones en lenguaje natural (no técnico)
- Priorización de información importante
- Oraciones claras y concisas
- Evita redundancia

Mejoras implementadas:
- Captioning con Florence-2 (image-to-text) para contexto de escena
- Prioridad semántica para ordenar objetos por relevancia

Mejoras futuras posibles:
- Detección de expresiones faciales
- Descripción de colores dominantes
============================================================================
"""

import numpy as np
from typing import Dict, Optional
import logging

from app.core.config import settings
from app.services.ocr_service import get_ocr_service
from app.services.object_detection_service import get_object_detection_service, GENDER_MAP
from app.models.schemas import SceneDescriptionResponse, ExplorationResponse, DetectedObject
from app.utils.image_utils import get_image_info

# Configurar logging
logger = logging.getLogger(__name__)


class SceneDescriptionService:
    """
    Servicio principal para generar descripciones de escenas.

    Orquesta los servicios de OCR y detección de objetos para
    crear una descripción unificada de la imagen.

    Uso:
        service = SceneDescriptionService()
        descripcion = service.describe_scene(imagen_cv2)
    """

    def __init__(self):
        """Inicializa el servicio y sus dependencias."""
        self.ocr_service = None
        self.detection_service = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """
        Inicializa los servicios de forma lazy.

        Lazy initialization permite que la aplicación arranque rápido
        y solo cargue los modelos cuando realmente se necesitan.
        """
        if not self._initialized:
            logger.info("Inicializando servicios de descripción...")
            self.ocr_service = get_ocr_service()
            self.detection_service = get_object_detection_service()
            self._initialized = True
            logger.info("Servicios inicializados correctamente")

    def describe_scene(self, image: np.ndarray) -> SceneDescriptionResponse:
        """
        Genera una descripción completa de la escena.

        Proceso:
        1. Ejecutar OCR para extraer texto
        2. Ejecutar detección de objetos
        3. Combinar resultados en descripción coherente
        4. Formatear para TTS

        Args:
            image: Imagen en formato OpenCV (numpy array BGR)

        Returns:
            SceneDescriptionResponse con descripción completa
        """
        self._ensure_initialized()

        try:
            # Obtener información de la imagen
            image_info = get_image_info(image)

            # Paso 1: Extraer texto (OCR)
            logger.info("Ejecutando OCR...")
            ocr_result = self.ocr_service.extract_text(image)

            # Paso 2: Detectar objetos
            logger.info("Detectando objetos...")
            detection_result = self.detection_service.detect_objects(image)

            # Paso 3: Generar caption con Florence-2 (si habilitado)
            caption = None
            if settings.CAPTIONING_ENABLED:
                try:
                    from app.services.captioning_service import get_captioning_service
                    captioning = get_captioning_service()
                    caption = captioning.generate_scene_description_spanish(
                        image,
                        detected_objects=detection_result.get("objects", []),
                        detailed=True,
                    )
                    if caption:
                        logger.info(f"Caption generado: {caption[:80]}...")
                except Exception as e:
                    logger.debug(f"Captioning no disponible: {e}")

            # Paso 4: Generar descripción combinada
            description = self._generate_combined_description(
                ocr_result,
                detection_result,
                image_info,
                caption=caption,
            )

            # Asignar prioridades semánticas a los objetos
            objects_list = detection_result.get("objects", [])
            if settings.SEMANTIC_PRIORITY_ENABLED:
                try:
                    from app.services.semantic_priority_service import get_semantic_priority_service
                    priority_svc = get_semantic_priority_service()
                    priority_svc.assign_priorities(objects_list)
                except Exception:
                    pass

            # Construir respuesta
            return SceneDescriptionResponse(
                success=True,
                message="Escena analizada correctamente",
                description=description,
                detected_text=ocr_result.get("text", ""),
                has_text=ocr_result.get("has_text", False),
                caption=caption,
                objects=[DetectedObject(**obj.model_dump()) if hasattr(obj, 'model_dump')
                         else obj for obj in objects_list],
                object_count=detection_result.get("object_count", 0),
                processing_details={
                    "ocr_confidence": ocr_result.get("confidence"),
                    "ocr_word_count": ocr_result.get("word_count"),
                    "image_dimensions": f"{image_info['width']}x{image_info['height']}",
                    "captioning_enabled": settings.CAPTIONING_ENABLED,
                    "has_caption": caption is not None,
                }
            )

        except Exception as e:
            logger.error(f"Error describiendo escena: {e}")
            return SceneDescriptionResponse(
                success=False,
                message=f"Error durante el análisis: {str(e)}",
                description="No fue posible analizar la imagen.",
                detected_text="",
                has_text=False,
                objects=[],
                object_count=0
            )

    def _generate_combined_description(
        self,
        ocr_result: Dict,
        detection_result: Dict,
        image_info: Dict,
        caption: Optional[str] = None,
    ) -> str:
        """
        Genera una descripción en lenguaje natural.

        La descripción está optimizada para TTS (Text-to-Speech):
        - Oraciones cortas y claras
        - Sin abreviaciones
        - Información organizada jerárquicamente
        - Evita símbolos y caracteres especiales

        Si hay caption de Florence-2, se usa como contexto inicial de escena.
        Luego se agregan detalles de objetos y texto.

        Args:
            ocr_result: Resultado del servicio OCR
            detection_result: Resultado del servicio de detección
            image_info: Información de dimensiones de la imagen
            caption: Caption de Florence-2 en español (opcional)

        Returns:
            Descripción en español para TTS
        """
        parts = []

        # --- PARTE 0: Caption de Florence-2 (contexto de escena) ---
        if caption:
            parts.append(caption)

        # --- PARTE 1: Objetos detectados ---
        objects = detection_result.get("objects", [])
        if objects:
            # Usar el resumen generado por el servicio de detección
            object_summary = detection_result.get("summary", "")
            if object_summary:
                parts.append(object_summary)

            # Agregar detalle de ubicación para objetos principales
            main_objects = objects[:3]  # Top 3 por confianza
            location_details = self._describe_locations(main_objects, image_info)
            if location_details:
                parts.append(location_details)
        else:
            parts.append("No se identificaron objetos específicos en la imagen.")

        # --- PARTE 2: Texto detectado ---
        if ocr_result.get("has_text"):
            text = ocr_result.get("text", "").strip()
            word_count = ocr_result.get("word_count", 0)

            if word_count > 0:
                if word_count <= 10:
                    # Texto corto: leerlo completo
                    parts.append(f"La imagen contiene texto que dice: {text}")
                elif word_count <= 50:
                    # Texto medio: resumen + texto completo
                    parts.append(
                        f"Se encontró un texto con {word_count} palabras. "
                        f"El contenido es: {text}"
                    )
                else:
                    # Texto largo: solo las primeras palabras
                    preview = ' '.join(text.split()[:20])
                    parts.append(
                        f"La imagen contiene un texto extenso de "
                        f"aproximadamente {word_count} palabras. "
                        f"Comienza con: {preview}..."
                    )
        else:
            parts.append("No se detectó texto legible en la imagen.")

        # Unir todas las partes
        description = " ".join(parts)

        return description

    def _describe_locations(
        self,
        objects: list,
        image_info: Dict
    ) -> str:
        """
        Describe la ubicación de los objetos principales.

        Divide la imagen en una cuadrícula 3x3:
        - Superior (izquierda, centro, derecha)
        - Centro (izquierda, centro, derecha)
        - Inferior (izquierda, centro, derecha)

        Args:
            objects: Lista de objetos detectados
            image_info: Información de la imagen

        Returns:
            Descripción de ubicaciones
        """
        if not objects:
            return ""

        width = image_info["width"]
        height = image_info["height"]

        # Divisiones de la imagen
        third_w = width / 3
        third_h = height / 3

        descriptions = []

        for obj in objects:
            if not hasattr(obj, 'bounding_box'):
                continue

            bbox = obj.bounding_box

            # Calcular centro del objeto
            center_x = (bbox.x_min + bbox.x_max) / 2
            center_y = (bbox.y_min + bbox.y_max) / 2

            # Determinar posición horizontal
            if center_x < third_w:
                pos_h = "a la izquierda"
            elif center_x > 2 * third_w:
                pos_h = "a la derecha"
            else:
                pos_h = "en el centro"

            # Determinar posición vertical
            if center_y < third_h:
                pos_v = "en la parte superior"
            elif center_y > 2 * third_h:
                pos_v = "en la parte inferior"
            else:
                pos_v = ""  # Centro vertical no se menciona

            # Construir descripción de ubicación
            if pos_v:
                location = f"{pos_v} {pos_h}"
            else:
                location = pos_h

            name = obj.name_es if hasattr(obj, 'name_es') else obj.name
            gender = GENDER_MAP.get(name, "m")
            article = "La" if gender == "f" else "El"
            descriptions.append(f"{article} {name} está {location}")

        if descriptions:
            return ". ".join(descriptions) + "."

        return ""

    def describe_exploration(self, image: np.ndarray) -> ExplorationResponse:
        """
        Genera una descripción estructurada del entorno para el modo Exploración.

        A diferencia de describe_scene (narrativo), este método produce
        descripciones en segunda persona, espaciales y estructuradas:
        "Hay una mesa frente a ti y una silla a la izquierda."

        Args:
            image: Imagen en formato OpenCV (numpy array BGR)

        Returns:
            ExplorationResponse con descripción estructurada
        """
        self._ensure_initialized()

        try:
            image_info = get_image_info(image)

            # Ejecutar OCR y detección
            ocr_result = self.ocr_service.extract_text(image)
            detection_result = self.detection_service.detect_objects(image)

            objects = detection_result.get("objects", [])

            # Ordenar por prioridad semántica (high primero)
            if settings.SEMANTIC_PRIORITY_ENABLED and objects:
                try:
                    from app.services.semantic_priority_service import get_semantic_priority_service
                    priority_svc = get_semantic_priority_service()
                    priority_svc.assign_priorities(objects)
                    objects = priority_svc.sort_by_priority(objects)
                except Exception:
                    pass

            parts = []

            # Describir objetos con posición espacial
            if objects:
                descriptions = []
                width = image_info["width"]

                for obj in objects[:6]:  # Top 6 objetos por confianza
                    if not hasattr(obj, 'bounding_box'):
                        continue

                    bbox = obj.bounding_box
                    center_x = (bbox.x_min + bbox.x_max) / 2
                    ratio = center_x / width if width > 0 else 0.5

                    if ratio < 0.33:
                        pos = "a la izquierda"
                    elif ratio > 0.66:
                        pos = "a la derecha"
                    else:
                        pos = "frente a ti"

                    name = obj.name_es
                    gender = GENDER_MAP.get(name, "m")
                    article = "una" if gender == "f" else "un"

                    # Agregar zona de distancia si es relevante
                    zone_text = ""
                    if obj.distance_zone == "muy_cerca":
                        zone_text = " muy cerca"
                    elif obj.distance_zone == "cerca":
                        zone_text = " cerca"

                    descriptions.append(f"{article} {name}{zone_text} {pos}")

                if descriptions:
                    if len(descriptions) == 1:
                        parts.append(f"Hay {descriptions[0]}.")
                    else:
                        items = ", ".join(descriptions[:-1]) + f" y {descriptions[-1]}"
                        parts.append(f"Hay {items}.")
            else:
                parts.append("No se detectan objetos en el entorno.")

            # Agregar texto detectado
            if ocr_result.get("has_text"):
                text = ocr_result.get("text", "").strip()
                word_count = ocr_result.get("word_count", 0)
                if word_count <= 15:
                    parts.append(f"Se lee el texto: {text}")
                else:
                    preview = ' '.join(text.split()[:15])
                    parts.append(f"Se lee un texto que dice: {preview}...")

            description = " ".join(parts)

            return ExplorationResponse(
                success=True,
                message="Entorno explorado correctamente",
                description=description,
                detected_text=ocr_result.get("text", ""),
                has_text=ocr_result.get("has_text", False),
                objects=[DetectedObject(**obj.model_dump()) if hasattr(obj, 'model_dump')
                         else obj for obj in objects],
                object_count=len(objects),
            )

        except Exception as e:
            logger.error(f"Error en exploración: {e}")
            return ExplorationResponse(
                success=False,
                message=f"Error durante la exploración: {str(e)}",
                description="No fue posible explorar el entorno.",
                detected_text="",
                has_text=False,
                objects=[],
                object_count=0,
            )

    def analyze_text_only(self, image: np.ndarray) -> Dict:
        """
        Analiza solo el texto de una imagen (sin detección de objetos).

        Útil cuando el usuario solo necesita leer texto.

        Args:
            image: Imagen en formato OpenCV

        Returns:
            Resultado del OCR
        """
        self._ensure_initialized()
        return self.ocr_service.extract_text(image)

    def analyze_objects_only(self, image: np.ndarray) -> Dict:
        """
        Analiza solo los objetos de una imagen (sin OCR).

        Útil cuando el usuario solo necesita identificar objetos.

        Args:
            image: Imagen en formato OpenCV

        Returns:
            Resultado de la detección de objetos
        """
        self._ensure_initialized()
        return self.detection_service.detect_objects(image)


# Instancia global del servicio
scene_service: Optional[SceneDescriptionService] = None


def get_scene_description_service() -> SceneDescriptionService:
    """
    Factory function para obtener el servicio de descripción de escenas.

    Returns:
        Instancia del servicio
    """
    global scene_service
    if scene_service is None:
        scene_service = SceneDescriptionService()
    return scene_service
