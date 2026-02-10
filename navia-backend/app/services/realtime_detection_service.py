"""
============================================================================
NAVIA Backend - Servicio de Detección en Tiempo Real
============================================================================
Gestiona el estado por sesión WebSocket para detección continua.

Funcionalidades:
- Tracking de objetos entre frames consecutivos
- Detección de cambios (objetos que aparecen/desaparecen)
- Estabilización para evitar falsos positivos por parpadeo
- Exponential smoothing de profundidad por objeto
- Zone persistence: solo confirma cambio de zona tras N frames
============================================================================
"""

import logging
from typing import List, Dict, Set, Optional, Tuple

from app.core.config import settings
from app.models.schemas import DetectedObject
from app.services.depth_estimation_service import DepthEstimationService, ZONE_LABELS

logger = logging.getLogger(__name__)


class DepthSmoother:
    """
    Suaviza valores de profundidad usando exponential smoothing
    y requiere persistencia de zona para evitar fluctuaciones.

    Exponential smoothing:
        smoothed[t] = α * raw[t] + (1 - α) * smoothed[t-1]

    Zone persistence:
        Un cambio de zona solo se confirma tras N frames consecutivos
        con la misma nueva zona. Esto evita que el TTS "se vuelva loco"
        alternando entre zonas en los bordes de los umbrales.
    """

    def __init__(
        self,
        alpha: float = settings.DEPTH_SMOOTHING_ALPHA,
        persistence_frames: int = settings.DEPTH_ZONE_PERSISTENCE
    ):
        self.alpha = alpha
        self.persistence_frames = persistence_frames
        # Valor suavizado por objeto
        self.smoothed: Dict[str, float] = {}
        # Streak de zona actual: name -> (zone, consecutive_count)
        self.zone_streak: Dict[str, Tuple[str, int]] = {}
        # Zona confirmada por objeto
        self.confirmed_zone: Dict[str, str] = {}

    def update(self, name: str, raw_depth: float) -> str:
        """
        Actualiza profundidad de un objeto y retorna su zona confirmada.

        Args:
            name: Nombre del objeto (name_es)
            raw_depth: Valor crudo de profundidad [0, 1]

        Returns:
            Zona confirmada: "muy_cerca", "cerca" o "lejos"
        """
        # Exponential smoothing
        prev = self.smoothed.get(name)
        if prev is not None:
            smoothed = self.alpha * raw_depth + (1 - self.alpha) * prev
        else:
            smoothed = raw_depth
        self.smoothed[name] = smoothed

        # Clasificar zona desde valor suavizado
        new_zone = DepthEstimationService.depth_to_zone(smoothed)

        # Tracking de persistencia
        prev_zone, count = self.zone_streak.get(name, (None, 0))
        if new_zone == prev_zone:
            count += 1
        else:
            count = 1
        self.zone_streak[name] = (new_zone, count)

        # Primera detección: asignar zona inmediatamente
        if name not in self.confirmed_zone:
            self.confirmed_zone[name] = new_zone
        # Cambio de zona: requiere persistencia
        elif count >= self.persistence_frames:
            self.confirmed_zone[name] = new_zone

        return self.confirmed_zone[name]

    def cleanup(self, active_names: Set[str]) -> None:
        """Elimina tracking de objetos que ya no están presentes."""
        stale = [n for n in self.smoothed if n not in active_names]
        for n in stale:
            del self.smoothed[n]
            self.zone_streak.pop(n, None)
            self.confirmed_zone.pop(n, None)


class RealtimeSessionState:
    """
    Estado de una sesión de detección en tiempo real.

    Mantiene un historial de objetos detectados para determinar
    qué cambió entre frames y decidir cuándo activar TTS.

    Estabilización:
    - Un objeto se considera 'apareció' tras 2 frames consecutivos
    - Un objeto se considera 'desapareció' tras 3 frames ausente
    - Un cambio de zona requiere N frames consecutivos (zone persistence)
    """

    def __init__(self):
        self.confirmed_objects: Set[str] = set()
        self.candidate_appeared: Dict[str, int] = {}
        self.candidate_disappeared: Dict[str, int] = {}
        self.frame_count: int = 0
        # Suavizador de profundidad
        self.depth_smoother = DepthSmoother()
        # Zonas previas para detectar cambios
        self.previous_zones: Dict[str, str] = {}

    def compute_changes(
        self,
        current_objects: List[DetectedObject],
        raw_depths: Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Compara detecciones actuales con las confirmadas anteriores.
        Aplica smoothing de profundidad y detecta cambios de zona.

        Args:
            current_objects: Objetos detectados en el frame actual
            raw_depths: {name_es: raw_depth_value} para smoothing

        Returns:
            Dict con appeared, disappeared, zone_changes,
            smoothed_zones, has_significant_change
        """
        self.frame_count += 1
        current_set = set(obj.name_es for obj in current_objects)

        # --- TRACKING DE OBJETOS ---

        # Objetos nuevos (en current pero no confirmados)
        new_candidates = current_set - self.confirmed_objects
        # Objetos ausentes (confirmados pero no en current)
        missing_candidates = self.confirmed_objects - current_set

        # Estabilización de apariciones (requiere 2 frames)
        appeared = []
        for name in new_candidates:
            self.candidate_appeared[name] = self.candidate_appeared.get(name, 0) + 1
            if self.candidate_appeared[name] >= 2:
                appeared.append(name)
                self.confirmed_objects.add(name)
                self.candidate_appeared.pop(name, None)

        # Limpiar candidatos que ya no están presentes
        stale = [n for n in self.candidate_appeared if n not in new_candidates]
        for name in stale:
            self.candidate_appeared.pop(name, None)

        # Estabilización de desapariciones (requiere 3 frames ausente)
        disappeared = []
        for name in missing_candidates:
            self.candidate_disappeared[name] = self.candidate_disappeared.get(name, 0) + 1
            if self.candidate_disappeared[name] >= 3:
                disappeared.append(name)
                self.confirmed_objects.discard(name)
                self.candidate_disappeared.pop(name, None)

        # Limpiar candidatos que reaparecieron
        reappeared = [n for n in self.candidate_disappeared if n not in missing_candidates]
        for name in reappeared:
            self.candidate_disappeared.pop(name, None)

        # --- SMOOTHING DE PROFUNDIDAD ---

        smoothed_zones: Dict[str, str] = {}
        zone_changes: List[Dict[str, str]] = []

        if raw_depths:
            for name, depth in raw_depths.items():
                zone = self.depth_smoother.update(name, depth)
                smoothed_zones[name] = zone

            # Limpiar objetos que desaparecieron del smoother
            self.depth_smoother.cleanup(current_set)

            # Detectar cambios de zona (para TTS)
            for name, zone in smoothed_zones.items():
                prev = self.previous_zones.get(name)
                if prev and prev != zone:
                    zone_changes.append({
                        "name": name,
                        "from_zone": prev,
                        "to_zone": zone,
                    })

            self.previous_zones = smoothed_zones.copy()

        has_significant_change = (
            len(appeared) > 0
            or len(disappeared) > 0
            or any(zc["to_zone"] == "muy_cerca" for zc in zone_changes)
        )

        return {
            "appeared": appeared,
            "disappeared": disappeared,
            "zone_changes": zone_changes,
            "smoothed_zones": smoothed_zones,
            "has_significant_change": has_significant_change,
            "current_objects": list(current_set),
        }

    def reset(self):
        """Reinicia el estado de la sesión."""
        self.confirmed_objects.clear()
        self.candidate_appeared.clear()
        self.candidate_disappeared.clear()
        self.frame_count = 0
        self.depth_smoother = DepthSmoother()
        self.previous_zones.clear()
