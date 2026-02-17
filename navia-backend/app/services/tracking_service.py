"""
============================================================================
NAVIA Backend - Servicio de Tracking de Objetos (ByteTrack)
============================================================================
Tracking temporal de objetos entre frames consecutivos usando ByteTrack
integrado en la librería ultralytics.

ByteTrack usa asociación IoU en dos etapas:
1. Alta confianza: asocia detecciones de alta confianza con tracks existentes
2. Baja confianza: re-asocia detecciones de baja confianza con tracks no
   emparejados, recuperando detecciones que otros trackers descartarían

Ventajas sobre tracking por nombre (sistema anterior):
- Distingue objetos individuales del mismo tipo (dos sillas = dos tracks)
- Predice posición con Kalman filter cuando un objeto se pierde brevemente
- Reduce flickering porque un track persiste aunque falle una detección
- No requiere modelo de ReID (lightweight, ideal para CPU)
============================================================================
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

from app.core.config import settings
from app.models.schemas import DetectedObject

logger = logging.getLogger(__name__)


class TrackingService:
    """
    Servicio de tracking que envuelve ByteTrack de ultralytics.

    Asigna track_id persistentes a cada detección entre frames,
    permitiendo estabilización temporal sin depender del nombre de clase.

    Los track_ids se usan internamente para:
    - Suavizado de profundidad por objeto individual
    - Estabilización de apariciones/desapariciones
    - Priorización consistente de objetos en TTS
    """

    def __init__(self):
        self.tracker = None
        self.frame_id = 0
        self._class_name_map: Dict[int, str] = {}
        self._init_tracker()

    def _init_tracker(self) -> None:
        """Inicializa el tracker ByteTrack con configuración de settings."""
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace

            args = IterableSimpleNamespace(
                track_high_thresh=settings.TRACK_HIGH_THRESH,
                track_low_thresh=settings.TRACK_LOW_THRESH,
                new_track_thresh=settings.TRACK_LOW_THRESH,
                track_buffer=settings.TRACK_BUFFER_FRAMES,
                match_thresh=settings.TRACK_MATCH_THRESH,
            )
            self.tracker = BYTETracker(args, frame_rate=10)
            logger.info("ByteTrack inicializado correctamente")
        except Exception as e:
            logger.warning(
                f"No se pudo inicializar ByteTrack: {e}. "
                "Se usará tracking por nombre como fallback."
            )
            self.tracker = None

    @property
    def is_available(self) -> bool:
        """Indica si ByteTrack está disponible."""
        return self.tracker is not None

    def update(
        self,
        detections: List[DetectedObject],
        img_shape: Tuple[int, int],
    ) -> List[Dict]:
        """
        Actualiza el tracker con las detecciones del frame actual.

        Convierte DetectedObject a formato numpy que ByteTrack espera,
        ejecuta la asociación, y retorna las detecciones con track_id.

        Args:
            detections: Objetos detectados por YOLO en este frame
            img_shape: (height, width) del frame actual

        Returns:
            Lista de dicts con:
                - track_id: int (ID persistente del track)
                - object: DetectedObject original
                - name_es: str (nombre en español para referencia rápida)
        """
        if not self.is_available or not detections:
            # Fallback: retornar detecciones sin track_id real
            return [
                {
                    "track_id": i,
                    "object": det,
                    "name_es": det.name_es,
                }
                for i, det in enumerate(detections)
            ]

        self.frame_id += 1

        # Construir array [N, 6]: x1, y1, x2, y2, confidence, class_id
        # ByteTrack necesita class_id numérico; usamos hash del nombre
        det_array = np.zeros((len(detections), 6), dtype=np.float32)
        det_to_obj: Dict[int, DetectedObject] = {}

        for i, det in enumerate(detections):
            bb = det.bounding_box
            # Generar class_id numérico consistente a partir del nombre
            class_id = hash(det.name_es) % 10000
            self._class_name_map[class_id] = det.name_es

            det_array[i] = [
                bb.x_min, bb.y_min, bb.x_max, bb.y_max,
                det.confidence, class_id,
            ]
            det_to_obj[i] = det

        try:
            # ByteTrack espera formato Results-like de ultralytics
            # Pasamos el array directamente
            tracks = self.tracker.update(det_array, img_shape)
        except Exception as e:
            logger.warning(f"Error en ByteTrack update: {e}")
            return [
                {
                    "track_id": i,
                    "object": det,
                    "name_es": det.name_es,
                }
                for i, det in enumerate(detections)
            ]

        # Mapear tracks de vuelta a DetectedObject
        tracked_results = []

        for track in tracks:
            track_id = int(track.track_id)
            # track.tlbr contiene [x1, y1, x2, y2]
            track_bbox = track.tlbr

            # Encontrar la detección original más cercana por IoU
            best_idx = self._match_track_to_detection(
                track_bbox, det_array[:, :4]
            )

            if best_idx is not None and best_idx in det_to_obj:
                original_obj = det_to_obj[best_idx]
                name_es = original_obj.name_es
            else:
                # Track sin detección asociable, usar class_name del mapa
                cls_id = int(track.cls) if hasattr(track, 'cls') else 0
                name_es = self._class_name_map.get(cls_id, "desconocido")
                # Crear objeto con las coordenadas del track
                original_obj = None

            if original_obj is not None:
                tracked_results.append({
                    "track_id": track_id,
                    "object": original_obj,
                    "name_es": name_es,
                })

        return tracked_results

    @staticmethod
    def _match_track_to_detection(
        track_bbox: np.ndarray,
        det_bboxes: np.ndarray,
    ) -> Optional[int]:
        """
        Encuentra la detección más cercana a un track por IoU.

        Args:
            track_bbox: [x1, y1, x2, y2] del track
            det_bboxes: [N, 4] array de bounding boxes de detecciones

        Returns:
            Índice de la detección más cercana, o None
        """
        if len(det_bboxes) == 0:
            return None

        # Calcular IoU entre track y todas las detecciones
        tx1, ty1, tx2, ty2 = track_bbox[:4]

        dx1 = det_bboxes[:, 0]
        dy1 = det_bboxes[:, 1]
        dx2 = det_bboxes[:, 2]
        dy2 = det_bboxes[:, 3]

        # Intersección
        ix1 = np.maximum(tx1, dx1)
        iy1 = np.maximum(ty1, dy1)
        ix2 = np.minimum(tx2, dx2)
        iy2 = np.minimum(ty2, dy2)

        inter_w = np.maximum(0, ix2 - ix1)
        inter_h = np.maximum(0, iy2 - iy1)
        intersection = inter_w * inter_h

        # Unión
        track_area = max(0, (tx2 - tx1) * (ty2 - ty1))
        det_areas = np.maximum(0, (dx2 - dx1) * (dy2 - dy1))
        union = track_area + det_areas - intersection

        ious = np.where(union > 0, intersection / union, 0)

        best_idx = int(np.argmax(ious))
        if ious[best_idx] > 0.3:
            return best_idx
        return None

    def reset(self) -> None:
        """Reinicia el tracker para una nueva sesión."""
        self.frame_id = 0
        self._class_name_map.clear()
        self._init_tracker()
