"""
============================================================================
NAVIA Backend - Servicio de Detección en Tiempo Real
============================================================================
Gestiona el estado por sesión WebSocket para detección continua.

Funcionalidades:
- Tracking espacial con ByteTrack (IoU + Kalman filter)
- Detección de cambios (objetos que aparecen/desaparecen)
- Estabilización para evitar falsos positivos por parpadeo
- Exponential smoothing de profundidad por track_id individual
- Zone persistence: solo confirma cambio de zona tras N frames
- Prioridad semántica: ordena objetos por relevancia para asistencia
- Fallback a tracking por nombre si ByteTrack no está disponible
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

    Con ByteTrack, las keys son track_id (int) en vez de name_es (str),
    permitiendo suavizar dos objetos del mismo tipo independientemente.
    """

    def __init__(
        self,
        alpha: float = settings.DEPTH_SMOOTHING_ALPHA,
        persistence_frames: int = settings.DEPTH_ZONE_PERSISTENCE
    ):
        self.alpha = alpha
        self.persistence_frames = persistence_frames
        # Valor suavizado por key (track_id o name_es)
        self.smoothed: Dict = {}
        # Streak de zona actual: key -> (zone, consecutive_count)
        self.zone_streak: Dict = {}
        # Zona confirmada por key
        self.confirmed_zone: Dict = {}

    def update(self, key, raw_depth: float) -> str:
        """
        Actualiza profundidad de un objeto y retorna su zona confirmada.

        Args:
            key: Identificador del objeto (track_id int o name_es str)
            raw_depth: Valor crudo de profundidad [0, 1]

        Returns:
            Zona confirmada: "muy_cerca", "cerca" o "lejos"
        """
        # Exponential smoothing
        prev = self.smoothed.get(key)
        if prev is not None:
            smoothed = self.alpha * raw_depth + (1 - self.alpha) * prev
        else:
            smoothed = raw_depth
        self.smoothed[key] = smoothed

        # Clasificar zona desde valor suavizado
        new_zone = DepthEstimationService.depth_to_zone(smoothed)

        # Tracking de persistencia
        prev_zone, count = self.zone_streak.get(key, (None, 0))
        if new_zone == prev_zone:
            count += 1
        else:
            count = 1
        self.zone_streak[key] = (new_zone, count)

        # Primera detección: asignar zona inmediatamente
        if key not in self.confirmed_zone:
            self.confirmed_zone[key] = new_zone
        # Cambio de zona: requiere persistencia
        elif count >= self.persistence_frames:
            self.confirmed_zone[key] = new_zone

        return self.confirmed_zone[key]

    def cleanup(self, active_keys: set) -> None:
        """Elimina tracking de objetos que ya no están presentes."""
        stale = [k for k in self.smoothed if k not in active_keys]
        for k in stale:
            del self.smoothed[k]
            self.zone_streak.pop(k, None)
            self.confirmed_zone.pop(k, None)


class RealtimeSessionState:
    """
    Estado de una sesión de detección en tiempo real.

    Mantiene un historial de objetos detectados para determinar
    qué cambió entre frames y decidir cuándo activar TTS.

    Con ByteTrack (TRACKING_ENABLED=True):
    - Tracking espacial por track_id (IoU + Kalman filter)
    - Cada objeto individual tiene su propio suavizado de profundidad
    - Cambios se reportan por name_es para TTS

    Sin ByteTrack (fallback):
    - Tracking por nombre (name_es) como antes
    - Compatible con el sistema original

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

        # ByteTrack (si habilitado)
        self._tracker = None
        self._tracking_enabled = settings.TRACKING_ENABLED
        if self._tracking_enabled:
            try:
                from app.services.tracking_service import TrackingService
                self._tracker = TrackingService()
                logger.info("ByteTrack habilitado para sesión realtime")
            except Exception as e:
                logger.warning(f"ByteTrack no disponible: {e}. Usando tracking por nombre.")
                self._tracking_enabled = False

        # Prioridad semántica (si habilitado)
        self._priority_service = None
        if settings.SEMANTIC_PRIORITY_ENABLED:
            try:
                from app.services.semantic_priority_service import get_semantic_priority_service
                self._priority_service = get_semantic_priority_service()
            except Exception as e:
                logger.warning(f"Prioridad semántica no disponible: {e}")

        # Mapeo track_id -> name_es para ByteTrack
        self._track_names: Dict[int, str] = {}
        # Track IDs confirmados (equivalente a confirmed_objects pero por track_id)
        self._confirmed_tracks: Set[int] = set()
        self._candidate_track_appeared: Dict[int, int] = {}
        self._candidate_track_disappeared: Dict[int, int] = {}
        # Contador de tracks activos
        self.tracked_count: int = 0

    def compute_changes(
        self,
        current_objects: List[DetectedObject],
        raw_depths: Optional[Dict[str, float]] = None,
        img_shape: Optional[Tuple[int, int]] = None,
    ) -> dict:
        """
        Compara detecciones actuales con las confirmadas anteriores.
        Aplica smoothing de profundidad y detecta cambios de zona.

        Si ByteTrack está habilitado, usa track_id para estabilización
        interna pero reporta cambios por name_es para TTS.

        Args:
            current_objects: Objetos detectados en el frame actual
            raw_depths: {name_es: raw_depth_value} para smoothing
            img_shape: (height, width) del frame, requerido para ByteTrack

        Returns:
            Dict con appeared, disappeared, zone_changes,
            smoothed_zones, has_significant_change, tracked_count
        """
        self.frame_count += 1

        # Asignar prioridades a los objetos
        if self._priority_service:
            self._priority_service.assign_priorities(current_objects)

        # Decidir estrategia de tracking
        if self._tracking_enabled and self._tracker and img_shape:
            return self._compute_changes_tracked(
                current_objects, raw_depths, img_shape
            )
        else:
            return self._compute_changes_by_name(
                current_objects, raw_depths
            )

    def _compute_changes_tracked(
        self,
        current_objects: List[DetectedObject],
        raw_depths: Optional[Dict[str, float]],
        img_shape: Tuple[int, int],
    ) -> dict:
        """Tracking con ByteTrack: usa track_id para estabilización."""

        # Paso 1: Actualizar ByteTrack
        tracked = self._tracker.update(current_objects, img_shape)
        self.tracked_count = len(tracked)

        # Paso 2: Construir conjuntos por track_id
        current_track_ids = set()
        track_to_name: Dict[int, str] = {}
        track_to_depth: Dict[int, float] = {}

        for t in tracked:
            tid = t["track_id"]
            name = t["name_es"]
            current_track_ids.add(tid)
            track_to_name[tid] = name
            self._track_names[tid] = name

            # Depth por track_id (usar raw_depths del objeto)
            if raw_depths and name in raw_depths:
                track_to_depth[tid] = raw_depths[name]

        # Paso 3: Estabilización por track_id
        new_tracks = current_track_ids - self._confirmed_tracks
        missing_tracks = self._confirmed_tracks - current_track_ids

        appeared_names = []
        for tid in new_tracks:
            self._candidate_track_appeared[tid] = (
                self._candidate_track_appeared.get(tid, 0) + 1
            )
            if self._candidate_track_appeared[tid] >= 2:
                appeared_names.append(track_to_name.get(tid, "desconocido"))
                self._confirmed_tracks.add(tid)
                self._candidate_track_appeared.pop(tid, None)

        # Limpiar candidatos que ya no están
        stale = [t for t in self._candidate_track_appeared if t not in new_tracks]
        for tid in stale:
            self._candidate_track_appeared.pop(tid, None)

        disappeared_names = []
        for tid in missing_tracks:
            self._candidate_track_disappeared[tid] = (
                self._candidate_track_disappeared.get(tid, 0) + 1
            )
            if self._candidate_track_disappeared[tid] >= 3:
                name = self._track_names.get(tid, "desconocido")
                disappeared_names.append(name)
                self._confirmed_tracks.discard(tid)
                self._candidate_track_disappeared.pop(tid, None)
                self._track_names.pop(tid, None)

        reappeared = [
            t for t in self._candidate_track_disappeared
            if t not in missing_tracks
        ]
        for tid in reappeared:
            self._candidate_track_disappeared.pop(tid, None)

        # Paso 4: Smoothing de profundidad por track_id
        smoothed_zones: Dict[str, str] = {}
        zone_changes: List[Dict[str, str]] = []

        if raw_depths:
            for tid, depth in track_to_depth.items():
                zone = self.depth_smoother.update(tid, depth)
                name = track_to_name.get(tid, "desconocido")
                # Guardar por name_es para compatibilidad con el resto del sistema
                smoothed_zones[name] = zone

            self.depth_smoother.cleanup(current_track_ids)

            # Detectar cambios de zona
            for name, zone in smoothed_zones.items():
                prev = self.previous_zones.get(name)
                if prev and prev != zone:
                    zone_changes.append({
                        "name": name,
                        "from_zone": prev,
                        "to_zone": zone,
                    })

            self.previous_zones = smoothed_zones.copy()

        # Deduplicar nombres (varios tracks del mismo tipo)
        appeared_unique = list(set(appeared_names))
        disappeared_unique = list(set(disappeared_names))

        has_significant_change = (
            len(appeared_unique) > 0
            or len(disappeared_unique) > 0
            or any(zc["to_zone"] == "muy_cerca" for zc in zone_changes)
        )

        # Construir lista de nombres actuales para compatibilidad
        current_names = list(set(track_to_name.values()))

        return {
            "appeared": appeared_unique,
            "disappeared": disappeared_unique,
            "zone_changes": zone_changes,
            "smoothed_zones": smoothed_zones,
            "has_significant_change": has_significant_change,
            "current_objects": current_names,
            "tracked_count": self.tracked_count,
        }

    def _compute_changes_by_name(
        self,
        current_objects: List[DetectedObject],
        raw_depths: Optional[Dict[str, float]],
    ) -> dict:
        """Fallback: tracking por nombre (sistema original)."""
        current_set = set(obj.name_es for obj in current_objects)

        # Objetos nuevos
        new_candidates = current_set - self.confirmed_objects
        missing_candidates = self.confirmed_objects - current_set

        # Estabilización de apariciones (requiere 2 frames)
        appeared = []
        for name in new_candidates:
            self.candidate_appeared[name] = (
                self.candidate_appeared.get(name, 0) + 1
            )
            if self.candidate_appeared[name] >= 2:
                appeared.append(name)
                self.confirmed_objects.add(name)
                self.candidate_appeared.pop(name, None)

        stale = [n for n in self.candidate_appeared if n not in new_candidates]
        for name in stale:
            self.candidate_appeared.pop(name, None)

        # Estabilización de desapariciones (requiere 3 frames ausente)
        disappeared = []
        for name in missing_candidates:
            self.candidate_disappeared[name] = (
                self.candidate_disappeared.get(name, 0) + 1
            )
            if self.candidate_disappeared[name] >= 3:
                disappeared.append(name)
                self.confirmed_objects.discard(name)
                self.candidate_disappeared.pop(name, None)

        reappeared = [
            n for n in self.candidate_disappeared
            if n not in missing_candidates
        ]
        for name in reappeared:
            self.candidate_disappeared.pop(name, None)

        # Smoothing de profundidad
        smoothed_zones: Dict[str, str] = {}
        zone_changes: List[Dict[str, str]] = []

        if raw_depths:
            for name, depth in raw_depths.items():
                zone = self.depth_smoother.update(name, depth)
                smoothed_zones[name] = zone

            self.depth_smoother.cleanup(current_set)

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
            "tracked_count": None,
        }

    def reset(self):
        """Reinicia el estado de la sesión."""
        self.confirmed_objects.clear()
        self.candidate_appeared.clear()
        self.candidate_disappeared.clear()
        self.frame_count = 0
        self.depth_smoother = DepthSmoother()
        self.previous_zones.clear()
        # Reset ByteTrack
        if self._tracker:
            self._tracker.reset()
        self._track_names.clear()
        self._confirmed_tracks.clear()
        self._candidate_track_appeared.clear()
        self._candidate_track_disappeared.clear()
        self.tracked_count = 0
