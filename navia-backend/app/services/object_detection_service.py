"""
============================================================================
NAVIA Backend - Servicio de Detección de Objetos
============================================================================
Este módulo implementa detección de objetos usando YOLOv8.

¿Qué es YOLO?
- "You Only Look Once" - arquitectura de detección de objetos en tiempo real
- A diferencia de otros métodos que analizan la imagen múltiples veces,
  YOLO procesa toda la imagen en una sola pasada (de ahí el nombre)
- Creado por Joseph Redmon en 2015, YOLOv8 es la versión más reciente

¿Por qué YOLOv8?
- Balance óptimo entre precisión y velocidad
- Modelo preentrenado en COCO dataset (80 clases de objetos comunes)
- Fácil de usar con la librería ultralytics
- Puede correr en CPU (sin GPU) con rendimiento aceptable

Clases detectables (COCO dataset):
- Personas, vehículos (carro, bicicleta, moto)
- Animales (perro, gato, pájaro)
- Objetos cotidianos (silla, mesa, teléfono, laptop)
- Alimentos, utensilios, y más

Limitaciones:
- Solo detecta las 80 clases del dataset COCO
- Objetos pequeños o parcialmente ocultos tienen menor precisión
- No identifica texto ni lee contenido
============================================================================
"""

from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

from app.core.config import settings
from app.models.schemas import DetectedObject, BoundingBox

# Configurar logging
logger = logging.getLogger(__name__)


# Diccionario de traducción: inglés → (español, género)
# género: "m" = masculino (un), "f" = femenino (una)
COCO_CLASSES_ES = {
    "person": ("persona", "f"),
    "bicycle": ("bicicleta", "f"),
    "car": ("carro", "m"),
    "motorcycle": ("motocicleta", "f"),
    "airplane": ("avión", "m"),
    "bus": ("autobús", "m"),
    "train": ("tren", "m"),
    "truck": ("camión", "m"),
    "boat": ("bote", "m"),
    "traffic light": ("semáforo", "m"),
    "fire hydrant": ("hidrante", "m"),
    "stop sign": ("señal de pare", "f"),
    "parking meter": ("parquímetro", "m"),
    "bench": ("banco", "m"),
    "bird": ("pájaro", "m"),
    "cat": ("gato", "m"),
    "dog": ("perro", "m"),
    "horse": ("caballo", "m"),
    "sheep": ("oveja", "f"),
    "cow": ("vaca", "f"),
    "elephant": ("elefante", "m"),
    "bear": ("oso", "m"),
    "zebra": ("cebra", "f"),
    "giraffe": ("jirafa", "f"),
    "backpack": ("mochila", "f"),
    "umbrella": ("paraguas", "m"),
    "handbag": ("bolso", "m"),
    "tie": ("corbata", "f"),
    "suitcase": ("maleta", "f"),
    "frisbee": ("frisbee", "m"),
    "skis": ("esquís", "m"),
    "snowboard": ("tabla de nieve", "f"),
    "sports ball": ("pelota", "f"),
    "kite": ("cometa", "f"),
    "baseball bat": ("bate de béisbol", "m"),
    "baseball glove": ("guante de béisbol", "m"),
    "skateboard": ("patineta", "f"),
    "surfboard": ("tabla de surf", "f"),
    "tennis racket": ("raqueta de tenis", "f"),
    "bottle": ("botella", "f"),
    "wine glass": ("copa de vino", "f"),
    "cup": ("taza", "f"),
    "fork": ("tenedor", "m"),
    "knife": ("cuchillo", "m"),
    "spoon": ("cuchara", "f"),
    "bowl": ("tazón", "m"),
    "banana": ("banana", "f"),
    "apple": ("manzana", "f"),
    "sandwich": ("sándwich", "m"),
    "orange": ("naranja", "f"),
    "broccoli": ("brócoli", "m"),
    "carrot": ("zanahoria", "f"),
    "hot dog": ("perro caliente", "m"),
    "pizza": ("pizza", "f"),
    "donut": ("dona", "f"),
    "cake": ("pastel", "m"),
    "chair": ("silla", "f"),
    "couch": ("sofá", "m"),
    "potted plant": ("planta en maceta", "f"),
    "bed": ("cama", "f"),
    "dining table": ("mesa de comedor", "f"),
    "toilet": ("inodoro", "m"),
    "tv": ("televisor", "m"),
    "laptop": ("computadora portátil", "f"),
    "mouse": ("ratón de computadora", "m"),
    "remote": ("control remoto", "m"),
    "keyboard": ("teclado", "m"),
    "cell phone": ("teléfono celular", "m"),
    "microwave": ("microondas", "m"),
    "oven": ("horno", "m"),
    "toaster": ("tostadora", "f"),
    "sink": ("lavabo", "m"),
    "refrigerator": ("refrigerador", "m"),
    "book": ("libro", "m"),
    "clock": ("reloj", "m"),
    "vase": ("jarrón", "m"),
    "scissors": ("tijeras", "f"),
    "teddy bear": ("oso de peluche", "m"),
    "hair drier": ("secador de pelo", "m"),
    "toothbrush": ("cepillo de dientes", "m"),
}

# Lookup rápido de género por nombre en español
GENDER_MAP = {name_es: gender for (name_es, gender) in COCO_CLASSES_ES.values()}


class ObjectDetectionService:
    """
    Servicio para detección de objetos en imágenes.

    Utiliza YOLOv8 para identificar y localizar objetos comunes
    en imágenes, proporcionando nombres y ubicaciones.

    Uso:
        service = ObjectDetectionService()
        resultados = service.detect_objects(imagen_cv2)
    """

    def __init__(self):
        """
        Inicializa el servicio cargando el modelo YOLO.

        El modelo se descarga automáticamente la primera vez.
        Versiones disponibles:
        - yolov8n: Nano (más rápido, menos preciso)
        - yolov8s: Small
        - yolov8m: Medium
        - yolov8l: Large
        - yolov8x: Extra Large (más preciso, más lento)
        """
        self.model = None
        self.confidence_threshold = settings.YOLO_CONFIDENCE_THRESHOLD
        self._load_model()

    def _load_model(self) -> None:
        """
        Carga el modelo YOLO.

        El modelo se descarga de internet la primera vez (~6MB para nano).
        Subsecuentes ejecuciones usan el modelo cacheado.
        """
        try:
            model_name = settings.YOLO_MODEL
            logger.info(f"Cargando modelo YOLO: {model_name}")

            # YOLO descarga automáticamente el modelo si no existe
            self.model = YOLO(model_name)

            logger.info(f"Modelo YOLO cargado exitosamente")

        except Exception as e:
            logger.error(f"Error cargando modelo YOLO: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo YOLO: {e}")

    def detect_objects(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        depth_map: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Detecta objetos en una imagen con estimación de profundidad.

        Proceso:
        1. Ejecutar inferencia YOLO
        2. Filtrar detecciones por confianza
        3. Traducir nombres a español
        4. Estimar profundidad (Depth Anything o heurística bbox)
        5. Clasificar en zonas: muy_cerca / cerca / lejos
        6. Generar resumen textual

        Args:
            image: Imagen en formato OpenCV (numpy array BGR)
            confidence_threshold: Umbral de confianza (0.0-1.0)
            depth_map: Mapa de profundidad pre-calculado (opcional).
                       Si no se provee, se calcula internamente.

        Returns:
            Diccionario con objects, object_count, summary, raw_depths
        """
        if self.model is None:
            return {
                "objects": [],
                "object_count": 0,
                "summary": "Error: Modelo no cargado",
                "error": "Modelo YOLO no inicializado"
            }

        threshold = confidence_threshold or self.confidence_threshold

        try:
            img_height, img_width = image.shape[:2]
            img_area = img_width * img_height
            img_shape = (img_height, img_width)

            # Obtener mapa de profundidad si no fue proporcionado
            if depth_map is None:
                try:
                    from app.services.depth_estimation_service import (
                        get_depth_estimation_service,
                    )
                    depth_service = get_depth_estimation_service()
                    depth_map = depth_service.estimate_depth_map(image)
                except Exception as e:
                    logger.debug(f"Depth estimation no disponible: {e}")

            # Ejecutar inferencia YOLO
            results = self.model(image, verbose=False)

            # Procesar resultados con depth map
            detected_objects, raw_depths = self._process_results(
                results, threshold, img_area, depth_map, img_shape
            )

            # Generar resumen en español
            summary = self._generate_summary(detected_objects)

            return {
                "objects": detected_objects,
                "object_count": len(detected_objects),
                "summary": summary,
                "raw_depths": raw_depths,
            }

        except Exception as e:
            logger.error(f"Error en detección de objetos: {e}")
            return {
                "objects": [],
                "object_count": 0,
                "summary": "Error durante la detección",
                "error": str(e)
            }

    def _process_results(
        self,
        results,
        threshold: float,
        img_area: float,
        depth_map: Optional[np.ndarray],
        img_shape: tuple
    ) -> tuple:
        """
        Procesa los resultados crudos de YOLO con estimación de profundidad.

        Args:
            results: Resultados de YOLO
            threshold: Umbral de confianza mínima
            img_area: Área de la imagen en píxeles
            depth_map: Mapa de profundidad [H,W] normalizado, o None
            img_shape: (height, width) de la imagen original

        Returns:
            Tupla (detected_objects, raw_depths)
            raw_depths: {name_es: max_depth_value} para smoothing en WebSocket
        """
        from app.services.depth_estimation_service import (
            DepthEstimationService,
            ZONE_LABELS,
        )

        detected_objects = []
        # raw_depths guarda el depth más alto (más cercano) por clase
        raw_depths: Dict[str, float] = {}

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])
                if confidence < threshold:
                    continue

                class_id = int(boxes.cls[i])
                class_name = result.names[class_id]

                bbox = boxes.xyxy[i].cpu().numpy()
                x_min, y_min, x_max, y_max = (
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                )

                # Traducir nombre al español
                class_info = COCO_CLASSES_ES.get(class_name)
                name_es = class_info[0] if class_info else class_name

                # Estimar profundidad → zona
                if depth_map is not None:
                    depth_value = DepthEstimationService.get_object_depth(
                        depth_map,
                        (x_min, y_min, x_max, y_max),
                        img_shape
                    )
                    zone = DepthEstimationService.depth_to_zone(depth_value)
                else:
                    # Fallback: heurística de bounding box
                    bbox_area = (x_max - x_min) * (y_max - y_min)
                    zone = DepthEstimationService.bbox_heuristic_zone(
                        bbox_area, img_area
                    )
                    depth_value = {
                        "muy_cerca": 0.85, "cerca": 0.5, "lejos": 0.15
                    }.get(zone, 0.15)

                label = ZONE_LABELS.get(zone, zone)

                # Guardar depth más alto (más cercano) por clase
                if name_es not in raw_depths or depth_value > raw_depths[name_es]:
                    raw_depths[name_es] = depth_value

                detected_obj = DetectedObject(
                    name=class_name,
                    name_es=name_es,
                    confidence=round(confidence, 3),
                    bounding_box=BoundingBox(
                        x_min=x_min, y_min=y_min,
                        x_max=x_max, y_max=y_max
                    ),
                    distance_zone=zone,
                    distance_estimate=label,
                )
                detected_objects.append(detected_obj)

        detected_objects.sort(key=lambda x: x.confidence, reverse=True)
        return detected_objects, raw_depths

    @staticmethod
    def _get_article(name_es: str, count: int) -> str:
        """
        Retorna el artículo correcto según género y número.

        Args:
            name_es: Nombre del objeto en español
            count: Cantidad de objetos

        Returns:
            Artículo apropiado ("un", "una", etc.)
        """
        gender = GENDER_MAP.get(name_es, "m")
        if count == 1:
            return "una" if gender == "f" else "un"
        return ""

    @staticmethod
    def _pluralize(name: str) -> str:
        """Pluraliza un nombre en español."""
        if name.endswith("a"):
            return name[:-1] + "as"
        elif name.endswith("o"):
            return name[:-1] + "os"
        elif name.endswith(("ón", "or", "és")):
            return name + "es"
        elif name.endswith("z"):
            return name[:-1] + "ces"
        elif name.endswith("s"):
            return name  # ya es plural (paraguas, microondas, tijeras)
        else:
            return name + "s"

    def _generate_summary(self, objects: List[DetectedObject]) -> str:
        """
        Genera un resumen textual con zonas de distancia.

        Formato: "Se detectó un gato muy cerca y una silla lejos.
                  Precaución, gato muy cerca."

        Args:
            objects: Lista de objetos detectados

        Returns:
            Resumen en español en lenguaje natural
        """
        if not objects:
            return "No se detectaron objetos en la imagen."

        ZONE_PRIORITY = {"muy_cerca": 3, "cerca": 2, "lejos": 1}

        # Contar objetos y trackear la zona más cercana por tipo
        object_counts: Dict[str, int] = {}
        closest_zone: Dict[str, str] = {}
        for obj in objects:
            name = obj.name_es
            object_counts[name] = object_counts.get(name, 0) + 1
            zone = obj.distance_zone or "lejos"
            if name not in closest_zone:
                closest_zone[name] = zone
            elif ZONE_PRIORITY.get(zone, 0) > ZONE_PRIORITY.get(closest_zone[name], 0):
                closest_zone[name] = zone

        # Construir descripciones: "un gato muy cerca", "2 sillas lejos"
        from app.services.depth_estimation_service import ZONE_LABELS
        descriptions = []
        for name, count in object_counts.items():
            article = self._get_article(name, count)
            zone = closest_zone.get(name, "lejos")
            zone_label = ZONE_LABELS.get(zone, "")

            if count == 1:
                desc = f"{article} {name} {zone_label}".strip()
            else:
                plural = self._pluralize(name)
                desc = f"{count} {plural} {zone_label}".strip()
            descriptions.append(desc)

        if len(descriptions) == 1:
            summary = f"Se detectó {descriptions[0]}."
        elif len(descriptions) == 2:
            summary = f"Se detectaron {descriptions[0]} y {descriptions[1]}."
        else:
            last = descriptions.pop()
            summary = f"Se detectaron {', '.join(descriptions)} y {last}."

        # Alerta de peligro para objetos muy cerca
        danger_objects = [
            name for name, zone in closest_zone.items() if zone == "muy_cerca"
        ]
        if danger_objects:
            if len(danger_objects) == 1:
                summary += f" Precaución, {danger_objects[0]} muy cerca."
            else:
                summary += " Precaución, objetos muy cerca."

        return summary

    def get_model_info(self) -> Dict:
        """
        Obtiene información sobre el modelo cargado.

        Returns:
            Diccionario con información del modelo
        """
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_name": settings.YOLO_MODEL,
            "num_classes": len(self.model.names),
            "confidence_threshold": self.confidence_threshold
        }


# Instancia global del servicio
object_detection_service: Optional[ObjectDetectionService] = None


def get_object_detection_service() -> ObjectDetectionService:
    """
    Factory function para obtener el servicio de detección.

    Implementa lazy loading: el modelo solo se carga cuando
    se necesita por primera vez.

    Returns:
        Instancia del servicio de detección de objetos
    """
    global object_detection_service
    if object_detection_service is None:
        object_detection_service = ObjectDetectionService()
    return object_detection_service
