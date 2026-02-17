"""
============================================================================
NAVIA Backend - Servicio de Prioridad Semántica
============================================================================
Asigna prioridad a objetos detectados según su relevancia para asistencia
visual y navegación segura.

Tres niveles de prioridad:
  - high: obstáculos directos, personas, vehículos, escaleras, puertas
  - medium: muebles grandes que obstruyen el paso
  - low: decoración, objetos pequeños, comida

El sistema usa un mapa estático (O(1) por objeto, cero latencia adicional)
para las ~298 clases del vocabulario YOLO-World.

Opcionalmente, puede usar CLIP para análisis más detallado en endpoints
HTTP (nunca en tiempo real por costo computacional).
============================================================================
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

from app.core.config import settings
from app.models.schemas import DetectedObject

logger = logging.getLogger(__name__)


# ============================================================================
# Mapa de prioridad estática por nombre en español (name_es)
# Basado en relevancia para navegación de personas con discapacidad visual
# ============================================================================

PRIORITY_HIGH = {
    # --- Personas (impredecibles, requieren atención) ---
    "persona", "niño", "bebé", "persona en silla de ruedas",

    # --- Animales (movimiento impredecible) ---
    "perro", "gato",

    # --- Vehículos (peligro crítico) ---
    "carro", "autobús", "camión", "motocicleta", "bicicleta",
    "scooter eléctrico", "taxi", "ambulancia", "patrulla",

    # --- Infraestructura de calle (obstáculos de navegación) ---
    "semáforo", "señal de pare", "paso peatonal", "poste",
    "hidrante", "cono de tráfico", "barrera vial",
    "borde de acera", "tapa de alcantarilla", "reductor de velocidad",
    "farola", "señal de estacionamiento", "señal de construcción",

    # --- Escaleras y cambios de nivel ---
    "escaleras", "escalera mecánica", "ascensor", "pasamanos",

    # --- Puertas y accesos ---
    "puerta", "puerta corrediza", "portón",

    # --- Peligros domésticos ---
    "cuchillo", "tijeras", "estufa", "horno",

    # --- Objetos en movimiento / transporte ---
    "silla de ruedas", "cochecito de bebé", "carrito de compras",
    "patineta",

    # --- Señales de seguridad ---
    "señal de advertencia", "señal de piso mojado",
    "señal de salida", "alarma de incendio", "alarma contra incendios",
    "extintor",
}

PRIORITY_MEDIUM = {
    # --- Muebles grandes que obstruyen el paso ---
    "silla", "mesa", "escritorio", "sofá", "sillón",
    "cama", "banco", "mesa de centro", "mesa de comedor",
    "mostrador", "mecedora", "taburete",
    "mueble de televisor", "mesa de noche",

    # --- Electrodomésticos grandes ---
    "refrigerador", "lavadora", "secadora", "lavavajillas",
    "microondas", "calentador de agua",

    # --- Estructuras y barreras ---
    "cerca", "árbol", "arbusto",
    "contenedor de basura", "contenedor de reciclaje",

    # --- Objetos de baño (resbaladizos) ---
    "inodoro", "lavamanos", "bañera", "ducha",
    "báscula",

    # --- Objetos grandes en el suelo ---
    "alfombra", "tapete", "maleta",
    "aspiradora",
}

# Todo lo que no está en HIGH ni MEDIUM es LOW (decoración, objetos pequeños)


class SemanticPriorityService:
    """
    Servicio de priorización semántica para objetos detectados.

    Asigna prioridad basada en relevancia para navegación asistiva:
    - high: requiere atención inmediata (obstáculos, peligros, personas)
    - medium: puede obstruir el paso (muebles grandes)
    - low: información contextual (decoración, objetos pequeños)
    """

    def __init__(self):
        self._clip_model = None
        self._clip_processor = None
        self._clip_initialized = False
        self._clip_available = False

    def get_priority(self, name_es: str) -> str:
        """
        Obtiene la prioridad de un objeto por su nombre en español.

        Args:
            name_es: Nombre del objeto en español

        Returns:
            "high", "medium", o "low"
        """
        if name_es in PRIORITY_HIGH:
            return "high"
        if name_es in PRIORITY_MEDIUM:
            return "medium"
        return "low"

    def assign_priorities(
        self, objects: List[DetectedObject]
    ) -> List[DetectedObject]:
        """
        Asigna campo priority a cada objeto detectado.

        Modifica los objetos in-place y también los retorna.

        Args:
            objects: Lista de objetos detectados por YOLO

        Returns:
            Misma lista con campo priority asignado
        """
        for obj in objects:
            obj.priority = self.get_priority(obj.name_es)
        return objects

    def filter_by_priority(
        self,
        objects: List[DetectedObject],
        min_priority: str = "medium"
    ) -> List[DetectedObject]:
        """
        Filtra objetos manteniendo solo los de prioridad >= min_priority.

        Args:
            objects: Lista de objetos detectados
            min_priority: Prioridad mínima ("high", "medium", "low")

        Returns:
            Lista filtrada de objetos
        """
        priority_levels = {"high": 3, "medium": 2, "low": 1}
        min_level = priority_levels.get(min_priority, 1)

        return [
            obj for obj in objects
            if priority_levels.get(
                obj.priority or self.get_priority(obj.name_es), 1
            ) >= min_level
        ]

    def sort_by_priority(
        self, objects: List[DetectedObject]
    ) -> List[DetectedObject]:
        """
        Ordena objetos por prioridad (high primero) y luego por confianza.

        Args:
            objects: Lista de objetos detectados

        Returns:
            Lista ordenada: high → medium → low, dentro de cada nivel por confianza
        """
        priority_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            objects,
            key=lambda obj: (
                priority_order.get(
                    obj.priority or self.get_priority(obj.name_es), 2
                ),
                -obj.confidence,
            ),
        )

    # ========================================================================
    # CLIP (opcional) - Solo para endpoints HTTP
    # ========================================================================

    def _ensure_clip_initialized(self) -> None:
        """Carga lazy del modelo CLIP. Solo si CLIP_ENABLED=True."""
        if self._clip_initialized:
            return

        self._clip_initialized = True

        if not settings.CLIP_ENABLED:
            self._clip_available = False
            return

        try:
            from transformers import CLIPProcessor, CLIPModel

            logger.info(f"Cargando modelo CLIP: {settings.CLIP_MODEL}")
            self._clip_processor = CLIPProcessor.from_pretrained(
                settings.CLIP_MODEL
            )
            self._clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL)
            self._clip_model.eval()
            self._clip_available = True
            logger.info("Modelo CLIP cargado exitosamente")
        except Exception as e:
            logger.warning(f"No se pudo cargar CLIP: {e}")
            self._clip_available = False

    def compute_clip_relevance(
        self,
        image: np.ndarray,
        objects: List[DetectedObject]
    ) -> Dict[str, float]:
        """
        Calcula score de relevancia CLIP por objeto.

        Compara cada región de objeto contra prompts de asistencia visual
        para determinar qué tan relevante es para navegación.

        Solo funciona si CLIP_ENABLED=True. Retorna dict vacío si no.

        Args:
            image: Imagen BGR (OpenCV)
            objects: Objetos detectados con bounding boxes

        Returns:
            Dict {name_es: relevance_score} donde score ∈ [0, 1]
        """
        self._ensure_clip_initialized()
        if not self._clip_available:
            return {}

        import torch
        from PIL import Image as PILImage
        import cv2

        # Prompts que definen "relevante para navegación asistiva"
        relevance_prompts = [
            "obstacle blocking the walking path",
            "person standing or walking nearby",
            "moving vehicle on the road",
            "stairs or steps ahead",
            "danger warning sign",
        ]
        # Prompt negativo para calibrar
        irrelevance_prompts = [
            "decorative object on a shelf",
            "picture hanging on a wall",
        ]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scores: Dict[str, float] = {}

        for obj in objects:
            bb = obj.bounding_box
            crop = rgb[bb.y_min:bb.y_max, bb.x_min:bb.x_max]
            if crop.size == 0:
                continue

            pil_crop = PILImage.fromarray(crop)

            all_prompts = relevance_prompts + irrelevance_prompts
            inputs = self._clip_processor(
                text=all_prompts,
                images=pil_crop,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image.softmax(dim=-1)[0]

            # Score = suma de probabilidades de prompts relevantes
            relevance_score = float(
                logits[:len(relevance_prompts)].sum()
            )
            scores[obj.name_es] = round(relevance_score, 3)

        return scores


# Singleton
_semantic_priority_service: Optional[SemanticPriorityService] = None


def get_semantic_priority_service() -> SemanticPriorityService:
    """Factory function para obtener el servicio de prioridad semántica."""
    global _semantic_priority_service
    if _semantic_priority_service is None:
        _semantic_priority_service = SemanticPriorityService()
    return _semantic_priority_service
