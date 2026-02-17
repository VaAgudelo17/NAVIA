"""
============================================================================
NAVIA Backend - Servicio de Navegación
============================================================================
Transforma detecciones de objetos en instrucciones cortas de navegación
espacial para personas con discapacidad visual.

Ejemplo de salida: "Camino libre al frente. Silla cerca a la izquierda."

Principios:
- Instrucciones cortas y accionables (máximo 4 frases)
- Solo reporta obstáculos relevantes (muy_cerca y cerca)
- Posición espacial: izquierda, frente, derecha
- Prioriza objetos más cercanos
============================================================================
"""

import logging
from typing import List

from app.core.config import settings
from app.models.schemas import DetectedObject, BoundingBox

logger = logging.getLogger(__name__)


class NavigationService:
    """
    Genera instrucciones de navegación a partir de objetos detectados.

    Divide la imagen en 3 columnas verticales:
    - Izquierda (0-33% del ancho)
    - Frente/Centro (33-66%)
    - Derecha (66-100%)

    Solo reporta objetos en zonas muy_cerca y cerca.
    """

    def generate_navigation_summary(
        self,
        objects: List[DetectedObject],
        img_width: int,
        img_height: int,
    ) -> dict:
        """
        Genera instrucciones de navegación.

        Args:
            objects: Objetos detectados con zonas de distancia
            img_width: Ancho de la imagen en píxeles
            img_height: Alto de la imagen en píxeles

        Returns:
            dict con: instruction (str), obstacles (list), path_clear (bool)
        """
        if not objects:
            return {
                "instruction": "No se detectan obstáculos.",
                "obstacles": [],
                "path_clear": True,
            }

        # Filtrar por prioridad semántica: ignorar objetos decorativos
        relevant_objects = objects
        if settings.SEMANTIC_PRIORITY_ENABLED:
            try:
                from app.services.semantic_priority_service import get_semantic_priority_service
                priority_service = get_semantic_priority_service()
                priority_service.assign_priorities(objects)
                relevant_objects = priority_service.filter_by_priority(
                    objects, min_priority="medium"
                )
            except Exception:
                relevant_objects = objects

        # Filtrar solo objetos cercanos (obstáculos)
        obstacles = [
            obj for obj in relevant_objects
            if obj.distance_zone in ("muy_cerca", "cerca")
        ]

        if not obstacles:
            return {
                "instruction": "Camino libre.",
                "obstacles": [],
                "path_clear": True,
            }

        # Ordenar: muy_cerca primero, luego cerca. Dentro de cada zona, mayor confianza primero
        zone_priority = {"muy_cerca": 0, "cerca": 1}
        obstacles.sort(
            key=lambda o: (zone_priority.get(o.distance_zone, 2), -o.confidence)
        )

        # Clasificar obstáculos por posición
        left_obs = []
        center_obs = []
        right_obs = []

        for obj in obstacles:
            pos = self._get_spatial_position(obj.bounding_box, img_width)
            entry = (obj.name_es, obj.distance_zone, pos)
            if pos == "a la izquierda":
                left_obs.append(entry)
            elif pos == "a la derecha":
                right_obs.append(entry)
            else:
                center_obs.append(entry)

        # Generar frases
        phrases = []
        path_clear = len(center_obs) == 0

        if path_clear:
            phrases.append("Camino libre al frente")

        # Agregar obstáculos (máximo 4 frases total)
        all_obs = []
        # Primero centro (más peligroso), luego izquierda, luego derecha
        all_obs.extend(center_obs)
        all_obs.extend(left_obs)
        all_obs.extend(right_obs)

        for name_es, zone, position in all_obs:
            if len(phrases) >= 4:
                break
            zone_label = "muy cerca" if zone == "muy_cerca" else "cerca"
            phrases.append(f"{name_es.capitalize()} {zone_label} {position}")

        instruction = ". ".join(phrases) + "."

        return {
            "instruction": instruction,
            "obstacles": obstacles,
            "path_clear": path_clear,
        }

    def _get_spatial_position(self, bbox: BoundingBox, img_width: int) -> str:
        """
        Determina la posición horizontal de un objeto.

        Args:
            bbox: Bounding box del objeto
            img_width: Ancho de la imagen

        Returns:
            "a la izquierda", "al frente", o "a la derecha"
        """
        center_x = (bbox.x_min + bbox.x_max) / 2
        ratio = center_x / img_width if img_width > 0 else 0.5

        if ratio < 0.33:
            return "a la izquierda"
        elif ratio > 0.66:
            return "a la derecha"
        else:
            return "al frente"


# Instancia global
_navigation_service = None


def get_navigation_service() -> NavigationService:
    global _navigation_service
    if _navigation_service is None:
        _navigation_service = NavigationService()
    return _navigation_service
