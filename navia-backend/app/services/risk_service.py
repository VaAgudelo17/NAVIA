"""
============================================================================
NAVIA Backend - Servicio de Evaluación de Riesgo (DEPRECADO)
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

from app.models.schemas import DetectedObject, RiskAlert

logger = logging.getLogger(__name__)


class RiskService:
    """
    DEPRECADO: Usar NavigationGuidanceService en su lugar.

    Este wrapper redirige al nuevo servicio unificado para mantener
    compatibilidad con código que aún no ha sido actualizado.
    """

    def __init__(self):
        from app.services.navigation_guidance_service import get_navigation_guidance_service
        self._guidance = get_navigation_guidance_service()
        logger.warning(
            "RiskService está deprecado. "
            "Usar NavigationGuidanceService en su lugar."
        )

    def evaluate_risk(
        self,
        objects: List[DetectedObject],
        img_width: int,
    ) -> dict:
        """Redirige al pipeline unificado de NavigationGuidanceService."""
        # Usar altura estándar ya que el antiguo API no la pasaba
        result = self._guidance.generate_guidance(
            objects, img_width, 480,
            track_movement=False,
        )
        return {
            "has_danger": result["has_danger"],
            "priority": result["priority"],
            "alert_text": result["instruction"],
            "dangers": [],  # El nuevo sistema usa obstacle_details
        }


# Instancia global (compatibilidad)
_risk_service = None


def get_risk_service() -> RiskService:
    """DEPRECADO: Usar get_navigation_guidance_service() en su lugar."""
    global _risk_service
    if _risk_service is None:
        _risk_service = RiskService()
    return _risk_service
