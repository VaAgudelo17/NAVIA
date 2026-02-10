"""
============================================================================
NAVIA Backend - Servicio de Estimación de Profundidad
============================================================================
Estimación de profundidad monocular usando Depth Anything V2.

Convierte una imagen 2D en un mapa de profundidad relativa,
permitiendo clasificar objetos en zonas de distancia:
  - muy_cerca (peligro): objetos que requieren atención inmediata
  - cerca: objetos a distancia moderada
  - lejos (seguro): objetos distantes

El modelo Depth Anything V2 Small (~98MB) se descarga automáticamente
la primera vez. Funciona en CPU con rendimiento aceptable.
============================================================================
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple, Dict

from app.core.config import settings

logger = logging.getLogger(__name__)

# Etiquetas legibles para cada zona (TTS y display)
ZONE_LABELS: Dict[str, str] = {
    "muy_cerca": "muy cerca",
    "cerca": "cerca",
    "lejos": "lejos",
}


class DepthEstimationService:
    """
    Servicio de estimación de profundidad monocular.

    Usa Depth Anything V2 Small para generar mapas de profundidad
    a partir de imágenes RGB. Los valores de profundidad se normalizan
    a [0, 1] donde valores más altos = objetos más cercanos.
    """

    def __init__(self):
        self.pipe = None
        self._initialized = False
        self._available = True

    def _ensure_initialized(self) -> None:
        """Carga el modelo de forma lazy. Fallback silencioso si falla."""
        if self._initialized:
            return
        try:
            from transformers import pipeline
            logger.info(f"Cargando modelo de profundidad: {settings.DEPTH_MODEL}")
            self.pipe = pipeline(
                "depth-estimation",
                model=settings.DEPTH_MODEL,
                device="cpu",
            )
            self._initialized = True
            logger.info("Modelo de profundidad cargado exitosamente")
        except Exception as e:
            logger.warning(
                f"No se pudo cargar Depth Anything: {e}. "
                "Se usará heurística de bounding box como fallback."
            )
            self._available = False
            self._initialized = True

    @property
    def is_available(self) -> bool:
        """Indica si el modelo de profundidad está disponible."""
        self._ensure_initialized()
        return self._available and self.pipe is not None

    def estimate_depth_map(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Genera un mapa de profundidad normalizado [0, 1].

        Valores más altos = objetos más cercanos a la cámara.

        Args:
            image: Imagen BGR (OpenCV format)

        Returns:
            numpy array float32 [H, W] normalizado [0, 1], o None si falla
        """
        self._ensure_initialized()
        if not self._available or self.pipe is None:
            return None

        try:
            from PIL import Image as PILImage

            # Convertir BGR → RGB → PIL
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb)

            img_h, img_w = image.shape[:2]

            # Ejecutar inferencia
            result = self.pipe(pil_image)

            # Usar predicted_depth (tensor con mayor precisión)
            pred_depth = result["predicted_depth"]
            depth_np = pred_depth.squeeze().cpu().numpy()

            # Redimensionar al tamaño original de la imagen
            depth_resized = cv2.resize(
                depth_np, (img_w, img_h),
                interpolation=cv2.INTER_LINEAR
            )

            # Normalizar a [0, 1] donde mayor = más cerca
            d_min = depth_resized.min()
            d_max = depth_resized.max()
            if d_max > d_min:
                normalized = (depth_resized - d_min) / (d_max - d_min)
            else:
                normalized = np.zeros_like(depth_resized, dtype=np.float32)

            return normalized.astype(np.float32)

        except Exception as e:
            logger.error(f"Error en estimación de profundidad: {e}")
            return None

    @staticmethod
    def get_object_depth(
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        img_shape: Tuple[int, int]
    ) -> float:
        """
        Obtiene la profundidad mediana de un objeto en su bounding box.

        Usa el 60% central del bbox para mayor robustez (evita bordes
        donde la profundidad puede ser ruidosa).

        Args:
            depth_map: Mapa de profundidad [H, W] normalizado
            bbox: (x_min, y_min, x_max, y_max) en coordenadas de imagen
            img_shape: (height, width) de la imagen original

        Returns:
            Valor de profundidad [0, 1] (mayor = más cerca)
        """
        d_h, d_w = depth_map.shape[:2]
        img_h, img_w = img_shape

        # Escalar coordenadas del bbox al tamaño del depth map
        scale_x = d_w / img_w
        scale_y = d_h / img_h

        x1 = max(0, int(bbox[0] * scale_x))
        y1 = max(0, int(bbox[1] * scale_y))
        x2 = min(d_w, int(bbox[2] * scale_x))
        y2 = min(d_h, int(bbox[3] * scale_y))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Usar centro 60% del bbox para robustez
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rw = max(1, int((x2 - x1) * 0.3))
        rh = max(1, int((y2 - y1) * 0.3))

        rx1 = max(0, cx - rw)
        ry1 = max(0, cy - rh)
        rx2 = min(d_w, cx + rw)
        ry2 = min(d_h, cy + rh)

        region = depth_map[ry1:ry2, rx1:rx2]
        if region.size == 0:
            return 0.0

        return float(np.median(region))

    @staticmethod
    def depth_to_zone(depth_value: float) -> str:
        """
        Convierte un valor de profundidad a zona de distancia.

        Zonas (configurables via settings):
          - muy_cerca: > DEPTH_ZONE_MUY_CERCA (default 0.7)
          - cerca: > DEPTH_ZONE_CERCA (default 0.35)
          - lejos: el resto
        """
        if depth_value > settings.DEPTH_ZONE_MUY_CERCA:
            return "muy_cerca"
        elif depth_value > settings.DEPTH_ZONE_CERCA:
            return "cerca"
        return "lejos"

    @staticmethod
    def zone_to_label(zone: str) -> str:
        """Convierte zona programática a etiqueta legible."""
        return ZONE_LABELS.get(zone, zone)

    @staticmethod
    def bbox_heuristic_zone(bbox_area: float, img_area: float) -> str:
        """
        Fallback: estima zona basándose en tamaño del bounding box.

        Se usa cuando Depth Anything no está disponible.
        """
        ratio = bbox_area / img_area if img_area > 0 else 0
        if ratio > 0.25:
            return "muy_cerca"
        elif ratio > 0.06:
            return "cerca"
        return "lejos"


# Singleton
_depth_service: Optional[DepthEstimationService] = None


def get_depth_estimation_service() -> DepthEstimationService:
    """Factory function para obtener el servicio de profundidad."""
    global _depth_service
    if _depth_service is None:
        _depth_service = DepthEstimationService()
    return _depth_service
