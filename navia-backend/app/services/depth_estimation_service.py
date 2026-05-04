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

import hashlib
import numpy as np
import cv2
import logging
from typing import Optional, Tuple, Dict

from app.core.config import settings

logger = logging.getLogger(__name__)

# Tamaño interno de Depth Anything V2 Small (optimización de resize)
_DEPTH_INTERNAL_SIZE = 518

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
        # Caché del último depth map para evitar recalcular el mismo frame
        self._last_image_hash: Optional[str] = None
        self._last_depth_map: Optional[np.ndarray] = None

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

    def estimate_depth_map(
        self, image: np.ndarray, use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """
        Genera un mapa de profundidad normalizado [0, 1].

        Valores más altos = objetos más cercanos a la cámara.

        Optimizaciones:
        - Resize a 518px antes de inferencia (tamaño interno del modelo)
        - Caché del último resultado para evitar recomputar el mismo frame

        Args:
            image: Imagen BGR (OpenCV format)
            use_cache: Si True, retorna caché si la imagen es la misma

        Returns:
            numpy array float32 [H, W] normalizado [0, 1], o None si falla
        """
        self._ensure_initialized()
        if not self._available or self.pipe is None:
            return None

        try:
            from PIL import Image as PILImage

            img_h, img_w = image.shape[:2]

            # Verificar caché: hash rápido basado en muestra de píxeles
            if use_cache:
                img_hash = self._compute_image_hash(image)
                if img_hash == self._last_image_hash and self._last_depth_map is not None:
                    # Retornar caché redimensionado si es necesario
                    cached = self._last_depth_map
                    if cached.shape[:2] != (img_h, img_w):
                        cached = cv2.resize(
                            cached, (img_w, img_h),
                            interpolation=cv2.INTER_LINEAR
                        )
                    return cached

            # Convertir BGR → RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize para optimizar inferencia (el modelo usa 518px internamente)
            max_dim = max(img_h, img_w)
            if max_dim > _DEPTH_INTERNAL_SIZE:
                scale = _DEPTH_INTERNAL_SIZE / max_dim
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                rgb_resized = cv2.resize(rgb, (new_w, new_h))
            else:
                rgb_resized = rgb

            pil_image = PILImage.fromarray(rgb_resized)

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

            result_map = normalized.astype(np.float32)

            # Actualizar caché
            if use_cache:
                self._last_image_hash = img_hash
                self._last_depth_map = result_map

            return result_map

        except Exception as e:
            logger.error(f"Error en estimación de profundidad: {e}")
            return None

    @staticmethod
    def _compute_image_hash(image: np.ndarray) -> str:
        """
        Calcula un hash rápido de la imagen para comparación de caché.

        Usa una muestra de píxeles para velocidad (no hash completo).
        """
        # Tomar muestra de 100 píxeles distribuidos uniformemente
        h, w = image.shape[:2]
        step_h = max(1, h // 10)
        step_w = max(1, w // 10)
        sample = image[::step_h, ::step_w].tobytes()
        return hashlib.md5(sample).hexdigest()

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
        if ratio > 0.15:    # antes 0.25: advierte "muy cerca" desde el 15% del frame
            return "muy_cerca"
        elif ratio > 0.04:  # antes 0.06: detecta "cerca" desde el 4% del frame
            return "cerca"
        return "lejos"

    @staticmethod
    def filter_depth_coherent(
        objects: list,
        depth_map: np.ndarray,
    ) -> list:
        """
        Descarta detecciones de YOLO cuyo depth dentro de su bbox NO es
        coherente con la zona declarada. Reduce falsos positivos del tipo
        "perro muy cerca" cuando en realidad el depth dice que el bbox
        está lejos (probablemente YOLO confundió un patrón).

        Reglas:
        - Si declara "muy_cerca" → depth medio del bbox debe ser > 0.40
        - Si declara "cerca"     → depth medio debe ser > 0.20
        - "lejos" no se filtra (no aporta seguridad descartar lejanos)

        Es un FILTRO conservador: solo descarta cuando hay desacuerdo
        FUERTE. Si la diferencia es marginal, mantiene la detección.
        """
        if depth_map is None or not objects:
            return objects

        h, w = depth_map.shape[:2]
        kept = []
        for obj in objects:
            if obj.bounding_box is None or obj.distance_zone is None:
                kept.append(obj)
                continue

            bbox = obj.bounding_box
            x1 = max(0, int(bbox.x_min))
            y1 = max(0, int(bbox.y_min))
            x2 = min(w, int(bbox.x_max))
            y2 = min(h, int(bbox.y_max))
            if x2 <= x1 or y2 <= y1:
                kept.append(obj)
                continue

            bbox_depth = float(np.mean(depth_map[y1:y2, x1:x2]))

            zone = obj.distance_zone
            keep = True
            if zone == "muy_cerca" and bbox_depth < 0.40:
                keep = False
            elif zone == "cerca" and bbox_depth < 0.20:
                keep = False

            if keep:
                kept.append(obj)
        return kept

    @staticmethod
    def detect_phantom_obstacle(
        depth_map: np.ndarray,
        yolo_objects: list,
        threshold: float = 0.55,
        min_area_ratio: float = 0.10,
    ) -> Optional[Dict]:
        """
        Encuentra UNA región muy cercana en el depth map que YOLO NO
        clasificó (paredes, árboles, obstáculos desconocidos).

        Devuelve un dict con bbox, depth promedio y posición horizontal,
        o None si no encuentra nada significativo.

        Algoritmo:
        1. Crea máscara de zonas con depth > threshold
        2. Borra de la máscara las zonas cubiertas por bboxes de YOLO
        3. Encuentra la mayor componente conexa restante
        4. Si su área >= min_area_ratio del frame, es un obstáculo fantasma
        """
        if depth_map is None:
            return None

        h, w = depth_map.shape[:2]
        img_area = h * w
        if img_area == 0:
            return None

        # 1. Máscara binaria de zonas muy cercanas
        close_mask = (depth_map > threshold).astype(np.uint8)

        # 2. Borrar zonas cubiertas por YOLO bboxes (con margen del 5% para
        #    evitar detectar el "halo" del depth alrededor del objeto)
        for obj in yolo_objects:
            if obj.bounding_box is None:
                continue
            bbox = obj.bounding_box
            margin_x = int((bbox.x_max - bbox.x_min) * 0.05)
            margin_y = int((bbox.y_max - bbox.y_min) * 0.05)
            x1 = max(0, int(bbox.x_min) - margin_x)
            y1 = max(0, int(bbox.y_min) - margin_y)
            x2 = min(w, int(bbox.x_max) + margin_x)
            y2 = min(h, int(bbox.y_max) + margin_y)
            close_mask[y1:y2, x1:x2] = 0

        # 3. Encontrar componentes conexas
        # Usamos cv2.connectedComponentsWithStats para no agregar dependencia
        try:
            import cv2
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                close_mask, connectivity=8
            )
        except Exception:
            return None

        if num_labels <= 1:
            return None  # Solo background

        # Buscar componente más grande (excluye label 0 = background)
        largest_label = 0
        largest_area = 0
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > largest_area:
                largest_area = area
                largest_label = i

        if largest_label == 0:
            return None

        # 4. Verificar que sea suficientemente grande
        if largest_area / img_area < min_area_ratio:
            return None

        # Extraer bbox del componente
        x = int(stats[largest_label, cv2.CC_STAT_LEFT])
        y = int(stats[largest_label, cv2.CC_STAT_TOP])
        bw = int(stats[largest_label, cv2.CC_STAT_WIDTH])
        bh = int(stats[largest_label, cv2.CC_STAT_HEIGHT])

        # Profundidad promedio dentro del componente
        component_mask = (labels == largest_label)
        component_depth = float(np.mean(depth_map[component_mask]))

        # Posición horizontal del centro del componente
        cx = x + bw / 2
        ratio_x = cx / w
        if ratio_x < 0.35:
            position = "a tu izquierda"
        elif ratio_x > 0.65:
            position = "a tu derecha"
        else:
            position = "frente a ti"

        return {
            "bbox": (x, y, x + bw, y + bh),
            "depth": component_depth,
            "area_ratio": largest_area / img_area,
            "position": position,
        }

    @staticmethod
    def compute_directional_depth(
        depth_map: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calcula profundidad promedio por columna (izquierda, centro, derecha).

        Útil para instrucciones de navegación: "el camino está más libre
        a la derecha" o "obstáculos al frente".

        Args:
            depth_map: Mapa de profundidad [H, W] normalizado [0, 1]

        Returns:
            Dict con profundidad promedio por zona:
            - "izquierda": float [0,1]
            - "centro": float [0,1]
            - "derecha": float [0,1]
            Valores más altos = objetos más cercanos
        """
        h, w = depth_map.shape[:2]
        third = w // 3

        return {
            "izquierda": float(np.mean(depth_map[:, :third])),
            "centro": float(np.mean(depth_map[:, third:2 * third])),
            "derecha": float(np.mean(depth_map[:, 2 * third:])),
        }


# Singleton
_depth_service: Optional[DepthEstimationService] = None


def get_depth_estimation_service() -> DepthEstimationService:
    """Factory function para obtener el servicio de profundidad."""
    global _depth_service
    if _depth_service is None:
        _depth_service = DepthEstimationService()
    return _depth_service
