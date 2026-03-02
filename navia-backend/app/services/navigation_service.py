"""
============================================================================
NAVIA Backend - Servicio de Navegación (DEPRECADO)
============================================================================
Este servicio fue reemplazado por NavigationGuidanceService que unifica
navegación + riesgo en un solo pipeline optimizado para movilidad peatonal.

Este archivo se mantiene por compatibilidad. Todas las funciones
redirigen al nuevo servicio unificado.

Usar en su lugar:
    from app.services.navigation_guidance_service import get_navigation_guidance_service
============================================================================
"""

import logging
from typing import List

from app.models.schemas import DetectedObject

logger = logging.getLogger(__name__)


class NavigationService:
    """
    DEPRECADO: Usar NavigationGuidanceService en su lugar.

    Este wrapper redirige al nuevo servicio unificado para mantener
    compatibilidad con código que aún no ha sido actualizado.
    """

    def __init__(self):
        from app.services.navigation_guidance_service import get_navigation_guidance_service
        self._guidance = get_navigation_guidance_service()
        logger.warning(
            "NavigationService está deprecado. "
            "Usar NavigationGuidanceService en su lugar."
        )

    def generate_navigation_summary(
        self,
        objects: List[DetectedObject],
        img_width: int,
        img_height: int,
    ) -> dict:
        """Redirige al pipeline unificado de NavigationGuidanceService."""
        result = self._guidance.generate_guidance(
            objects, img_width, img_height,
            track_movement=False,
        )
        return {
            "instruction": result["instruction"],
            "obstacles": result["obstacles"],
            "path_clear": result["path_clear"],
        }


# Instancia global (compatibilidad)
_navigation_service = None


def get_navigation_service() -> NavigationService:
    """DEPRECADO: Usar get_navigation_guidance_service() en su lugar."""
    global _navigation_service
    if _navigation_service is None:
        _navigation_service = NavigationService()
    return _navigation_service
