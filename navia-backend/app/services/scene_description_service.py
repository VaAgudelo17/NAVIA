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

Mejoras futuras posibles:
- Integrar modelos de captioning (image-to-text)
- Detección de expresiones faciales
- Reconocimiento de escenas (interior/exterior, cocina, oficina)
- Descripción de colores dominantes
============================================================================
"""

import numpy as np
from typing import Dict, Optional
import logging

from app.services.ocr_service import get_ocr_service
from app.services.object_detection_service import get_object_detection_service
from app.models.schemas import SceneDescriptionResponse, DetectedObject
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

            # Paso 3: Generar descripción combinada
            description = self._generate_combined_description(
                ocr_result,
                detection_result,
                image_info
            )

            # Construir respuesta
            return SceneDescriptionResponse(
                success=True,
                message="Escena analizada correctamente",
                description=description,
                detected_text=ocr_result.get("text", ""),
                has_text=ocr_result.get("has_text", False),
                objects=[DetectedObject(**obj.model_dump()) if hasattr(obj, 'model_dump')
                         else obj for obj in detection_result.get("objects", [])],
                object_count=detection_result.get("object_count", 0),
                processing_details={
                    "ocr_confidence": ocr_result.get("confidence"),
                    "ocr_word_count": ocr_result.get("word_count"),
                    "image_dimensions": f"{image_info['width']}x{image_info['height']}"
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
        image_info: Dict
    ) -> str:
        """
        Genera una descripción en lenguaje natural.

        La descripción está optimizada para TTS (Text-to-Speech):
        - Oraciones cortas y claras
        - Sin abreviaciones
        - Información organizada jerárquicamente
        - Evita símbolos y caracteres especiales

        Args:
            ocr_result: Resultado del servicio OCR
            detection_result: Resultado del servicio de detección
            image_info: Información de dimensiones de la imagen

        Returns:
            Descripción en español para TTS
        """
        parts = []

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
            descriptions.append(f"La {name} está {location}")

        if descriptions:
            return ". ".join(descriptions) + "."

        return ""

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
