"""
============================================================================
NAVIA Backend - Servicio de Evaluación de Riesgo
============================================================================
Clasifica objetos detectados por nivel de peligro y genera alertas cortas.

Niveles de peligro:
- critical: Peligro físico inmediato (vehículos, escaleras, huecos)
- high: Potencialmente peligroso (cuchillos, estufas, perros)
- medium: Requiere atención (personas, postes, vallas)

Regla: Solo genera alertas de voz para:
- critical: siempre (cualquier zona)
- high: solo si está muy_cerca
- medium: nunca se anuncia (solo se incluye en datos)

Ejemplo de salida: "Peligro. Carro muy cerca a la izquierda."
============================================================================
"""

import logging
from typing import List, Dict, Optional

from app.models.schemas import DetectedObject, BoundingBox, RiskAlert

logger = logging.getLogger(__name__)


# ============================================================================
# CLASIFICACIÓN DE PELIGRO
# ============================================================================
# Las claves son los nombres en español (name_es) de WORLD_CLASSES_ES
# para hacer el mapeo directo desde los objetos detectados.

DANGER_OBJECTS: Dict[str, str] = {
    # --- CRITICAL: Peligro físico inmediato ---
    # Vehículos
    "carro": "critical",
    "autobús": "critical",
    "camión": "critical",
    "motocicleta": "critical",
    "bicicleta": "critical",
    "scooter eléctrico": "critical",
    "taxi": "critical",
    "ambulancia": "critical",
    "patrulla": "critical",
    # Caídas y obstáculos de suelo
    "escaleras": "critical",
    "escalera mecánica": "critical",
    "tapa de alcantarilla": "critical",
    "reductor de velocidad": "critical",
    "borde de acera": "critical",
    # Barreras
    "barrera vial": "critical",
    "cono de tráfico": "critical",

    # --- HIGH: Potencialmente peligroso ---
    "cuchillo": "high",
    "tijeras": "high",
    "estufa": "high",
    "horno": "high",
    "perro": "high",
    "señal de construcción": "high",
    "contenedor de basura": "high",

    # --- MEDIUM: Requiere atención ---
    "persona": "medium",
    "niño": "medium",
    "poste": "medium",
    "farola": "medium",
    "hidrante": "medium",
    "cerca": "medium",
    "portón": "medium",
    "semáforo": "medium",
    "carrito de compras": "medium",
    "cochecito de bebé": "medium",
    "silla de ruedas": "medium",
    "árbol": "medium",
    "arbusto": "medium",
}


class RiskService:
    """
    Evalúa el nivel de riesgo de los objetos detectados y genera alertas.

    Solo genera alertas de voz para objetos con peligro real:
    - critical: siempre alerta
    - high: solo si muy_cerca
    - medium: solo datos, no alerta de voz
    """

    def evaluate_risk(
        self,
        objects: List[DetectedObject],
        img_width: int,
    ) -> dict:
        """
        Evalúa riesgos y genera texto de alerta.

        Args:
            objects: Objetos detectados con zonas de distancia
            img_width: Ancho de la imagen para calcular posición

        Returns:
            dict con: has_danger, priority, alert_text, dangers[]
        """
        if not objects:
            return {
                "has_danger": False,
                "priority": "none",
                "alert_text": "",
                "dangers": [],
            }

        dangers: List[RiskAlert] = []
        max_priority = "none"
        priority_order = {"critical": 3, "high": 2, "medium": 1, "none": 0}

        for obj in objects:
            danger_level = DANGER_OBJECTS.get(obj.name_es)
            if danger_level is None:
                continue

            position = self._get_spatial_position(obj.bounding_box, img_width)

            alert = RiskAlert(
                object_name=obj.name_es,
                danger_level=danger_level,
                distance_zone=obj.distance_zone or "lejos",
                position=position,
            )
            dangers.append(alert)

            # Actualizar prioridad máxima
            if priority_order.get(danger_level, 0) > priority_order.get(max_priority, 0):
                max_priority = danger_level

        # Filtrar alertas que merecen anuncio de voz
        voice_alerts = self._filter_voice_alerts(dangers)

        # Generar texto de alerta
        alert_text = self._generate_alert_text(voice_alerts)

        has_danger = len(voice_alerts) > 0

        return {
            "has_danger": has_danger,
            "priority": max_priority if has_danger else "none",
            "alert_text": alert_text,
            "dangers": dangers,
        }

    def _filter_voice_alerts(self, dangers: List[RiskAlert]) -> List[RiskAlert]:
        """
        Filtra alertas que merecen anuncio de voz.

        Reglas:
        - critical: siempre (cualquier zona excepto lejos)
        - high: solo si muy_cerca
        - medium: nunca
        """
        voice_alerts = []
        for d in dangers:
            if d.danger_level == "critical" and d.distance_zone != "lejos":
                voice_alerts.append(d)
            elif d.danger_level == "high" and d.distance_zone == "muy_cerca":
                voice_alerts.append(d)
        return voice_alerts

    def _generate_alert_text(self, alerts: List[RiskAlert]) -> str:
        """
        Genera texto corto de alerta para TTS.

        Máximo 2 frases. Prioriza critical sobre high.
        """
        if not alerts:
            return ""

        # Ordenar: critical primero, luego high. muy_cerca primero.
        zone_priority = {"muy_cerca": 0, "cerca": 1, "lejos": 2}
        level_priority = {"critical": 0, "high": 1}
        alerts.sort(
            key=lambda a: (
                level_priority.get(a.danger_level, 9),
                zone_priority.get(a.distance_zone, 9),
            )
        )

        phrases = []
        for alert in alerts[:2]:  # Máximo 2 frases
            name = alert.object_name
            pos = alert.position

            if alert.danger_level == "critical" and alert.distance_zone == "muy_cerca":
                phrases.append(f"Peligro. {name.capitalize()} muy cerca {pos}")
            elif alert.danger_level == "critical":
                phrases.append(f"Atención. {name.capitalize()} {pos}")
            elif alert.danger_level == "high" and alert.distance_zone == "muy_cerca":
                phrases.append(f"Precaución. {name.capitalize()} muy cerca {pos}")

        return ". ".join(phrases) + "." if phrases else ""

    def _get_spatial_position(self, bbox: BoundingBox, img_width: int) -> str:
        """Determina posición horizontal del objeto."""
        center_x = (bbox.x_min + bbox.x_max) / 2
        ratio = center_x / img_width if img_width > 0 else 0.5

        if ratio < 0.33:
            return "a la izquierda"
        elif ratio > 0.66:
            return "a la derecha"
        else:
            return "al frente"


# Instancia global
_risk_service = None


def get_risk_service() -> RiskService:
    global _risk_service
    if _risk_service is None:
        _risk_service = RiskService()
    return _risk_service
