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


# Diccionario de traducción: inglés → español
# Las 80 clases del dataset COCO traducidas
COCO_CLASSES_ES = {
    "person": "persona",
    "bicycle": "bicicleta",
    "car": "carro",
    "motorcycle": "motocicleta",
    "airplane": "avión",
    "bus": "autobús",
    "train": "tren",
    "truck": "camión",
    "boat": "bote",
    "traffic light": "semáforo",
    "fire hydrant": "hidrante",
    "stop sign": "señal de pare",
    "parking meter": "parquímetro",
    "bench": "banco",
    "bird": "pájaro",
    "cat": "gato",
    "dog": "perro",
    "horse": "caballo",
    "sheep": "oveja",
    "cow": "vaca",
    "elephant": "elefante",
    "bear": "oso",
    "zebra": "cebra",
    "giraffe": "jirafa",
    "backpack": "mochila",
    "umbrella": "paraguas",
    "handbag": "bolso",
    "tie": "corbata",
    "suitcase": "maleta",
    "frisbee": "frisbee",
    "skis": "esquís",
    "snowboard": "tabla de nieve",
    "sports ball": "pelota",
    "kite": "cometa",
    "baseball bat": "bate de béisbol",
    "baseball glove": "guante de béisbol",
    "skateboard": "patineta",
    "surfboard": "tabla de surf",
    "tennis racket": "raqueta de tenis",
    "bottle": "botella",
    "wine glass": "copa de vino",
    "cup": "taza",
    "fork": "tenedor",
    "knife": "cuchillo",
    "spoon": "cuchara",
    "bowl": "tazón",
    "banana": "banana",
    "apple": "manzana",
    "sandwich": "sándwich",
    "orange": "naranja",
    "broccoli": "brócoli",
    "carrot": "zanahoria",
    "hot dog": "perro caliente",
    "pizza": "pizza",
    "donut": "dona",
    "cake": "pastel",
    "chair": "silla",
    "couch": "sofá",
    "potted plant": "planta en maceta",
    "bed": "cama",
    "dining table": "mesa de comedor",
    "toilet": "inodoro",
    "tv": "televisor",
    "laptop": "computadora portátil",
    "mouse": "ratón de computadora",
    "remote": "control remoto",
    "keyboard": "teclado",
    "cell phone": "teléfono celular",
    "microwave": "microondas",
    "oven": "horno",
    "toaster": "tostadora",
    "sink": "lavabo",
    "refrigerator": "refrigerador",
    "book": "libro",
    "clock": "reloj",
    "vase": "jarrón",
    "scissors": "tijeras",
    "teddy bear": "oso de peluche",
    "hair drier": "secador de pelo",
    "toothbrush": "cepillo de dientes",
}


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
        confidence_threshold: Optional[float] = None
    ) -> Dict:
        """
        Detecta objetos en una imagen.

        Proceso:
        1. Ejecutar inferencia del modelo
        2. Filtrar detecciones por confianza
        3. Traducir nombres a español
        4. Generar resumen textual

        Args:
            image: Imagen en formato OpenCV (numpy array BGR)
            confidence_threshold: Umbral de confianza (0.0-1.0)
                                  Si no se especifica, usa el default

        Returns:
            Diccionario con:
            - objects: Lista de objetos detectados
            - object_count: Número de objetos
            - summary: Resumen en español
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
            # Ejecutar inferencia
            # verbose=False evita que YOLO imprima en consola
            results = self.model(image, verbose=False)

            # Procesar resultados
            detected_objects = self._process_results(results, threshold)

            # Generar resumen en español
            summary = self._generate_summary(detected_objects)

            return {
                "objects": detected_objects,
                "object_count": len(detected_objects),
                "summary": summary
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
        threshold: float
    ) -> List[DetectedObject]:
        """
        Procesa los resultados crudos de YOLO.

        Args:
            results: Resultados de YOLO (objeto Results)
            threshold: Umbral de confianza mínima

        Returns:
            Lista de objetos DetectedObject
        """
        detected_objects = []

        # YOLO puede devolver múltiples resultados (uno por imagen)
        # Como procesamos una imagen a la vez, tomamos el primero
        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            # Iterar sobre cada detección
            for i in range(len(boxes)):
                # Obtener confianza
                confidence = float(boxes.conf[i])

                # Filtrar por umbral
                if confidence < threshold:
                    continue

                # Obtener clase
                class_id = int(boxes.cls[i])
                class_name = result.names[class_id]

                # Obtener bounding box
                # xyxy = [x_min, y_min, x_max, y_max]
                bbox = boxes.xyxy[i].cpu().numpy()

                # Traducir nombre al español
                name_es = COCO_CLASSES_ES.get(class_name, class_name)

                # Crear objeto de detección
                detected_obj = DetectedObject(
                    name=class_name,
                    name_es=name_es,
                    confidence=round(confidence, 3),
                    bounding_box=BoundingBox(
                        x_min=int(bbox[0]),
                        y_min=int(bbox[1]),
                        x_max=int(bbox[2]),
                        y_max=int(bbox[3])
                    )
                )
                detected_objects.append(detected_obj)

        # Ordenar por confianza (mayor primero)
        detected_objects.sort(key=lambda x: x.confidence, reverse=True)

        return detected_objects

    def _generate_summary(self, objects: List[DetectedObject]) -> str:
        """
        Genera un resumen textual de los objetos detectados.

        Diseñado para ser leído por Text-to-Speech, por lo que
        usa lenguaje natural y evita tecnicismos.

        Args:
            objects: Lista de objetos detectados

        Returns:
            Resumen en español en lenguaje natural
        """
        if not objects:
            return "No se detectaron objetos en la imagen."

        # Contar objetos por tipo
        object_counts = {}
        for obj in objects:
            name = obj.name_es
            object_counts[name] = object_counts.get(name, 0) + 1

        # Construir descripción
        descriptions = []
        for name, count in object_counts.items():
            if count == 1:
                descriptions.append(f"una {name}")
            else:
                # Pluralización simple (no perfecta para todos los casos)
                if name.endswith("a"):
                    plural = name[:-1] + "as"
                elif name.endswith("o"):
                    plural = name[:-1] + "os"
                elif name.endswith(("ón", "or")):
                    plural = name + "es"
                else:
                    plural = name + "s"
                descriptions.append(f"{count} {plural}")

        # Formatear como oración
        if len(descriptions) == 1:
            summary = f"Se detectó {descriptions[0]}."
        elif len(descriptions) == 2:
            summary = f"Se detectaron {descriptions[0]} y {descriptions[1]}."
        else:
            # Última coma antes de "y"
            last = descriptions.pop()
            summary = f"Se detectaron {', '.join(descriptions)} y {last}."

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
