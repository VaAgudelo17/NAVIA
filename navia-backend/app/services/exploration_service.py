"""
============================================================================
NAVIA Backend - Servicio de Exploración Mejorado
============================================================================
Servicio dedicado para el modo EXPLORACIÓN con las siguientes mejoras:

1. Filtrado semántico de objetos (ignorar clases irrelevantes)
2. Priorización inteligente (score = confianza × tamaño × cercanía)
3. Estimación de distancia basada en bounding box
4. Posición espacial mejorada (horizontal + vertical)
5. Extracción de atributos visuales (color dominante)
6. Filtro inteligente de OCR
7. Narrativa estructurada y natural
8. Limitación de sobrecarga cognitiva (máx 3 objetos, 1 texto)

Pipeline:
  YOLO detections
  → Filtrado semántico
  → Ranking por importancia
  → Estimación espacial (horizontal + vertical + distancia)
  → Extracción de atributos básicos
  → OCR condicional
  → Generación de narrativa estructurada
============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import cv2

from app.models.schemas import DetectedObject, ExplorationResponse, BoundingBox
from app.services.object_detection_service import get_object_detection_service, GENDER_MAP

logger = logging.getLogger(__name__)


# ============================================================================
# CLASES IRRELEVANTES PARA EXPLORACIÓN
# Objetos de fondo que no aportan valor al usuario
# ============================================================================
IGNORED_CLASSES = {
    # Fondo/ambiente (si YOLO-World las detectara)
    "grass", "sky", "floor", "wall", "background", "ground", "ceiling",
    "road", "pavement", "sidewalk", "dirt", "sand", "water", "snow",
    # Texturas y superficies
    "texture", "pattern", "tile", "carpet", "wood", "concrete", "brick",
}

# Clases prioritarias para exploración (personas, animales, obstáculos)
HIGH_PRIORITY_CLASSES = {
    # Personas
    "persona", "niño", "bebé", "persona en silla de ruedas",
    # Animales
    "perro", "gato", "pájaro",
    # Vehículos cercanos (peligro)
    "carro", "motocicleta", "bicicleta", "scooter eléctrico",
    # Obstáculos importantes
    "silla de ruedas", "cochecito de bebé", "escaleras", "puerta",
}

MEDIUM_PRIORITY_CLASSES = {
    # Muebles grandes
    "silla", "mesa", "sofá", "cama", "escritorio", "estante",
    # Electrodomésticos grandes
    "refrigerador", "televisor", "lavadora",
    # Objetos de calle
    "semáforo", "señal de pare", "poste", "hidrante",
}

# Sinónimos: etiquetas distintas de YOLO que representan el mismo objeto
# Se normalizan a la etiqueta canónica antes de deduplicar
SYNONYM_MAP: dict[str, str] = {
    'gatito': 'gato',
    'perrito': 'perro',
    'cachorro': 'perro',
    'niña': 'niño',
    'mujer': 'persona',
    'hombre': 'persona',
    'adulto': 'persona',
    'chico': 'niño',
    'chica': 'niño',
    'bebé': 'bebé',
    'infante': 'bebé',
    'camiseta': 'camisa',
    'peluche': 'oso de peluche',
    'mochila': 'bolso',
    'cartera': 'bolso',
    'computadora portátil': 'computadora portátil',  # evita duplicados por variantes
}

# Accesorios que pertenecen a una persona
# Si hay una persona + estos objetos en posición similar, no narrarlos por separado
PERSON_ACCESSORIES = {
    "lentes", "lentes de sol", "sombrero", "gorra", "reloj de pulsera",
    "billetera", "bolso", "cartera", "mochila", "chaqueta", "abrigo",
    "zapato", "tenis", "tacón", "botas", "sandalia", "pantufla",
    "collar", "anillo", "pendientes", "perfume",
}


class ExplorationService:
    """
    Servicio mejorado para el modo Exploración.
    
    Genera descripciones claras, útiles y naturales orientadas a accesibilidad,
    priorizando objetos cercanos y su relación espacial con el usuario.
    """

    def __init__(self):
        self.detection_service = None
        self.ocr_service = None
        self._initialized = False
        
        # Configuración
        self.max_objects = 6  # Más objetos para no cortar personas en escenas grupales
        self.min_ocr_area_ratio = 0.02  # 2% del área de imagen
        self.min_ocr_confidence = 60  # Confianza mínima OCR (subido de 40 a 60)
        self.min_ocr_chars = 6  # Mínimo caracteres válidos (subido de 4 a 6)
        self.min_word_length_avg = 2.5  # Longitud promedio mínima de palabras
        self.max_uppercase_ratio = 0.7  # Máximo ratio de mayúsculas (ruido suele ser TODO MAYÚSCULAS)
        self.enable_color_detection = True  # Activado: usa recorte central para mayor precisión

    def _ensure_initialized(self) -> None:
        """Inicializa servicios de forma lazy."""
        if not self._initialized:
            logger.info("Inicializando ExplorationService...")
            self.detection_service = get_object_detection_service()
            from app.services.ocr_service import get_ocr_service
            self.ocr_service = get_ocr_service()
            self._initialized = True

    def analyze(self, image: np.ndarray) -> ExplorationResponse:
        """
        Analiza una imagen para el modo Exploración.
        
        Pipeline:
        1. Detección de objetos con YOLO
        2. Filtrado semántico
        3. Ranking por importancia
        4. Estimación espacial
        5. Extracción de color dominante
        6. OCR condicional
        7. Generación de narrativa
        
        Args:
            image: Imagen en formato OpenCV (BGR)
            
        Returns:
            ExplorationResponse con descripción estructurada
        """
        self._ensure_initialized()
        
        try:
            img_height, img_width = image.shape[:2]
            img_area = img_width * img_height
            
            # 1. Detectar objetos con umbral más alto para exploración (40%)
            from app.core.config import settings
            detection_result = self.detection_service.detect_objects(
                image, 
                confidence_threshold=settings.YOLO_EXPLORATION_CONFIDENCE
            )
            all_objects = detection_result.get("objects", [])
            
            # LOG: Ver qué detectó YOLO
            if all_objects:
                objs_info = [(o.name_es, f"{o.confidence:.0%}") for o in all_objects[:10]]
                logger.info(f"[Exploración] YOLO detectó {len(all_objects)} objetos (umbral {settings.YOLO_EXPLORATION_CONFIDENCE:.0%}): {objs_info}")
            else:
                logger.info("[Exploración] YOLO no detectó ningún objeto")
            
            # 2. Filtrado semántico
            filtered_objects = self._filter_semantic(all_objects)
            
            # 2.5. Combinar persona con accesorios (si hay persona, no narrar accesorios por separado)
            merge_result = self._merge_person_with_accessories(
                filtered_objects, img_width, img_height
            )
            filtered_objects = merge_result["objects"]
            person_accessories = merge_result["person_accessories"]
            
            # 3. Ranking por importancia (score combinado)
            ranked_objects = self._rank_by_importance(
                filtered_objects, img_width, img_height, img_area
            )
            
            # 4. Limitar a máximo 3 objetos
            top_objects = ranked_objects[:self.max_objects]
            
            # 5. Enriquecer con posición espacial y atributos
            enriched_objects = []
            for obj in top_objects:
                enriched = self._enrich_object(obj, image, img_width, img_height, img_area)
                # Agregar accesorios si es una persona con accesorios
                obj_name = obj.name_es if obj.name_es else ""
                if obj_name in person_accessories:
                    enriched["accessories"] = person_accessories[obj_name]
                enriched_objects.append(enriched)
            
            # 6. OCR condicional
            ocr_result = self._conditional_ocr(image, img_area)
            
            # 7. Generar narrativa estructurada
            description = self._generate_narrative(enriched_objects, ocr_result, all_objects)
            
            return ExplorationResponse(
                success=True,
                message="Entorno explorado correctamente",
                description=description,
                detected_text=ocr_result.get("text", "") if ocr_result else "",
                has_text=ocr_result.get("has_text", False) if ocr_result else False,
                objects=[obj["detected_object"] for obj in enriched_objects],
                object_count=len(enriched_objects),
            )
            
        except Exception as e:
            logger.error(f"Error en exploración: {e}")
            return ExplorationResponse(
                success=False,
                message=f"Error durante la exploración: {str(e)}",
                description="No fue posible explorar el entorno.",
                detected_text="",
                has_text=False,
                objects=[],
                object_count=0,
            )

    # =========================================================================
    # 1. FILTRADO SEMÁNTICO Y DEDUPLICACIÓN
    # =========================================================================
    def _filter_semantic(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        """
        Filtra objetos irrelevantes y normaliza nombres.

        - Elimina clases irrelevantes (fondo, texturas, etc.)
        - Normaliza sinónimos en el nombre mostrado (gatito → gato)
        - NO deduplica por nombre: si hay 3 personas en posiciones distintas,
          las 3 deben pasar. La deduplicación real la hace _deduplicate_by_overlap
          usando IoU: solo elimina detecciones del mismo objeto físico.
        """
        result: List[DetectedObject] = []

        for obj in objects:
            name_en = obj.name.lower() if obj.name else ""
            name_es = obj.name_es.lower() if obj.name_es else ""

            # Ignorar clases irrelevantes
            if name_en in IGNORED_CLASSES or name_es in IGNORED_CLASSES:
                continue

            # Normalizar nombre para mostrar (no afecta deduplicación)
            canonical = SYNONYM_MAP.get(name_es, name_es)
            if canonical != name_es:
                obj.name_es = canonical

            result.append(obj)

        # Deduplicar solo por solapamiento físico (IoU)
        return self._deduplicate_by_overlap(result)

    def _iou(self, b1, b2) -> float:
        """Calcula Intersection over Union entre dos bounding boxes."""
        if not b1 or not b2:
            return 0.0
        ix1 = max(b1.x_min, b2.x_min)
        iy1 = max(b1.y_min, b2.y_min)
        ix2 = min(b1.x_max, b2.x_max)
        iy2 = min(b1.y_max, b2.y_max)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = (b1.x_max - b1.x_min) * (b1.y_max - b1.y_min)
        area2 = (b2.x_max - b2.x_min) * (b2.y_max - b2.y_min)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def _deduplicate_by_overlap(
        self,
        objects: List[DetectedObject],
        iou_threshold: float = 0.45,
    ) -> List[DetectedObject]:
        """
        Elimina detecciones distintas que cubren el mismo objeto físico.

        Personas usan umbral más alto (0.75): en fotos grupales o selfies
        dos personas pueden tener sus bboxes solapados hasta un 50-60%,
        pero siguen siendo individuos distintos. Solo se elimina si el
        solapamiento es casi total (>75%), lo que indica misma detección.

        Objetos inanimados usan el umbral estándar (0.45).
        """
        PERSON_NAMES = {"persona", "niño", "bebé", "persona en silla de ruedas"}
        candidates = sorted(objects, key=lambda o: o.confidence, reverse=True)
        kept: List[DetectedObject] = []

        for obj in candidates:
            duplicate = False
            # Umbral adaptativo según si es persona o no
            is_person = obj.name_es in PERSON_NAMES
            threshold = 0.75 if is_person else iou_threshold

            for accepted in kept:
                if self._iou(obj.bounding_box, accepted.bounding_box) > threshold:
                    logger.debug(
                        f"[Exploración] Descartado '{obj.name_es}' ({obj.confidence:.0%}) "
                        f"por solapamiento con '{accepted.name_es}' ({accepted.confidence:.0%})"
                    )
                    duplicate = True
                    break
            if not duplicate:
                kept.append(obj)

        return kept

    # =========================================================================
    # 1.5. COMBINAR PERSONA CON ACCESORIOS
    # =========================================================================
    def _merge_person_with_accessories(
        self,
        objects: List[DetectedObject],
        img_width: int,
        img_height: int
    ) -> Dict:
        """
        Combina una persona con sus accesorios detectados.
        
        Si hay una persona Y accesorios (lentes, sombrero, etc.) en posición similar,
        NO narra los accesorios por separado - solo narra la persona con los accesorios.
        
        Esto evita descripciones como:
        "Una persona frente a ti. Un sombrero frente a ti."
        
        Y las cambia a:
        "Una persona con sombrero frente a ti."
        
        Returns:
            Lista de objetos con accesorios合并ados a la persona
        """
        if not objects:
            return {"objects": objects, "person_accessories": {}}
        
        # Buscar personas en la imagen
        person_names = {"persona", "niño", "bebé", "persona en silla de ruedas"}
        persons = [obj for obj in objects if obj.name_es.lower() in person_names]
        
        if not persons:
            return {"objects": objects, "person_accessories": {}}  # No hay personas, retornar normal
        
        # Encontrar accesorios de cada persona
        person = persons[0]  # Usar la primera persona (la más importante)
        person_bbox = person.bounding_box
        
        if not person_bbox:
            return {"objects": objects, "person_accessories": {}}
        
        # Buscar accesorios en posición similar a la persona
        accessories_found = []
        other_objects = []
        
        for obj in objects:
            if obj.name_es.lower() in person_names:
                continue
            
            # Verificar si es un accesorio conocido
            is_accessory = any(
                acc in obj.name_es.lower() 
                for acc in PERSON_ACCESSORIES
            )
            
            if not is_accessory:
                other_objects.append(obj)
                continue
            
            # Verificar si está en posición similar a la persona
            obj_bbox = obj.bounding_box
            if not obj_bbox:
                other_objects.append(obj)
                continue
            
            # Calcular overlap (qué tanto el accesorio está dentro del bounding box de la persona)
            # Si el centro del accesorio está dentro del bbox de la persona, considerarlo accesorio
            obj_center_x = (obj_bbox.x_min + obj_bbox.x_max) / 2
            obj_center_y = (obj_bbox.y_min + obj_bbox.y_max) / 2
            
            if (person_bbox.x_min <= obj_center_x <= person_bbox.x_max and
                person_bbox.y_min <= obj_center_y <= person_bbox.y_max):
                # El accesorio está dentro de la persona - es parte de ella
                accessories_found.append(obj.name_es)
            else:
                # Está fuera, narrarlo por separado
                other_objects.append(obj)
        
        # Retornar: TODAS las personas + objetos no-accesorios
        if accessories_found and person:
            person_name = person.name_es
            return {
                "objects": persons + other_objects,  # todas las personas, no solo la primera
                "person_accessories": {person_name: accessories_found}
            }

        return {
            "objects": objects,
            "person_accessories": {}
        }

    # =========================================================================
    # 2. RANKING POR IMPORTANCIA
    # =========================================================================
    def _rank_by_importance(
        self,
        objects: List[DetectedObject],
        img_width: int,
        img_height: int,
        img_area: int
    ) -> List[DetectedObject]:
        """
        Ordena objetos por score de importancia combinado.
        
        PRIORIDAD FUERTE al tamaño y posición central:
        - Objetos grandes y centrados = el usuario los está mirando directamente
        - La confianza de YOLO es secundaria
        
        Score = (tamaño^1.5) × (cercanía_centro^2) × confianza × prioridad_semántica
        """
        scored_objects = []
        
        center_x = img_width / 2
        center_y = img_height / 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for obj in objects:
            if not hasattr(obj, 'bounding_box') or obj.bounding_box is None:
                continue
            
            bbox = obj.bounding_box
            
            # Calcular área relativa del bbox (0 a 1)
            bbox_area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
            size_ratio = bbox_area / img_area if img_area > 0 else 0
            
            # FUERTE bonus por tamaño: objetos grandes son MUY importantes
            # size_ratio^1.5 hace que objetos grandes dominen
            size_score = size_ratio ** 1.5
            
            # Calcular cercanía al centro (1.0 = centro, 0.0 = esquina)
            obj_center_x = (bbox.x_min + bbox.x_max) / 2
            obj_center_y = (bbox.y_min + bbox.y_max) / 2
            dist_to_center = np.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)
            center_ratio = 1.0 - (dist_to_center / max_dist) if max_dist > 0 else 0.5
            
            # FUERTE bonus por estar centrado: el usuario apunta la cámara a lo importante
            center_score = center_ratio ** 2
            
            # Prioridad semántica
            name_es = obj.name_es.lower() if obj.name_es else ""
            if name_es in HIGH_PRIORITY_CLASSES or any(p in name_es for p in HIGH_PRIORITY_CLASSES):
                semantic_priority = 2.0
            elif name_es in MEDIUM_PRIORITY_CLASSES or any(p in name_es for p in MEDIUM_PRIORITY_CLASSES):
                semantic_priority = 1.3
            else:
                semantic_priority = 1.0
            
            # Score final: tamaño y centro son MÁS importantes que confianza
            score = (
                size_score *          # Tamaño (peso fuerte)
                center_score *        # Cercanía al centro (peso fuerte)
                (obj.confidence ** 0.5) *  # Confianza (peso reducido)
                semantic_priority     # Bonus semántico
            )
            
            scored_objects.append((score, obj))
        
        # Ordenar por score descendente
        scored_objects.sort(key=lambda x: x[0], reverse=True)
        
        return [obj for _, obj in scored_objects]

    # =========================================================================
    # 3. ESTIMACIÓN DE DISTANCIA Y POSICIÓN ESPACIAL
    # =========================================================================
    def _estimate_distance(self, bbox_area: int, img_area: int) -> Tuple[str, str]:
        """
        Estima distancia basada en tamaño del bounding box.
        
        bbox grande → "muy cerca"
        bbox mediano → "a unos metros"
        bbox pequeño → "lejos"
        
        Returns:
            Tuple (zone, label) ej: ("muy_cerca", "muy cerca")
        """
        ratio = bbox_area / img_area if img_area > 0 else 0
        
        if ratio > 0.15:  # >15% del área = muy cerca
            return "muy_cerca", "muy cerca"
        elif ratio > 0.05:  # 5-15% = a unos metros
            return "cerca", "a unos metros"
        else:  # <5% = lejos
            return "lejos", "lejos"

    def _get_spatial_position(
        self,
        bbox: BoundingBox,
        img_width: int,
        img_height: int
    ) -> Dict[str, str]:
        """
        Calcula posición espacial completa (horizontal + vertical).
        
        Horizontal: izquierda / frente / derecha
        Vertical: arriba / a tu altura / a nivel del suelo
        
        Returns:
            Dict con 'horizontal', 'vertical', 'combined'
        """
        # Centro del objeto
        center_x = (bbox.x_min + bbox.x_max) / 2
        center_y = (bbox.y_min + bbox.y_max) / 2
        
        # Posición horizontal
        ratio_x = center_x / img_width if img_width > 0 else 0.5
        if ratio_x < 0.33:
            horizontal = "a tu izquierda"
        elif ratio_x > 0.66:
            horizontal = "a tu derecha"
        else:
            horizontal = "frente a ti"
        
        # Posición vertical
        ratio_y = center_y / img_height if img_height > 0 else 0.5
        if ratio_y < 0.33:
            vertical = "arriba"
        elif ratio_y > 0.66:
            vertical = "a nivel del suelo"
        else:
            vertical = "a tu altura"
        
        # Combinación natural
        if horizontal == "frente a ti":
            if vertical == "a tu altura":
                combined = "frente a ti"
            else:
                combined = f"frente a ti, {vertical}"
        else:
            if vertical == "a tu altura":
                combined = horizontal
            else:
                combined = f"{horizontal}, {vertical}"
        
        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "combined": combined
        }

    # =========================================================================
    # 4. EXTRACCIÓN DE ATRIBUTOS VISUALES
    # =========================================================================
    def _get_dominant_color(self, image: np.ndarray, bbox: BoundingBox) -> Optional[str]:
        """
        Extrae el color dominante del objeto.
        
        Usa K-means simplificado para encontrar el color principal
        dentro del bounding box.
        
        Returns:
            Nombre del color en español o None
        """
        try:
            # Extraer región central del bbox (40% interior)
            # Evita incluir fondo y bordes donde el color del objeto es menos representativo
            x1 = max(0, bbox.x_min)
            y1 = max(0, bbox.y_min)
            x2 = min(image.shape[1], bbox.x_max)
            y2 = min(image.shape[0], bbox.y_max)

            if x2 <= x1 or y2 <= y1:
                return None

            bw = x2 - x1
            bh = y2 - y1
            margin_x = int(bw * 0.3)
            margin_y = int(bh * 0.3)
            cx1 = x1 + margin_x
            cy1 = y1 + margin_y
            cx2 = x2 - margin_x
            cy2 = y2 - margin_y

            if cx2 <= cx1 or cy2 <= cy1:
                roi = image[y1:y2, x1:x2]
            else:
                roi = image[cy1:cy2, cx1:cx2]
            
            if roi.size == 0:
                return None
            
            # Redimensionar para eficiencia
            roi_small = cv2.resize(roi, (50, 50), interpolation=cv2.INTER_AREA)
            
            # Convertir a HSV para mejor clasificación de color
            hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
            
            # Calcular histograma de Hue
            h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180])
            dominant_hue_idx = np.argmax(h_hist)
            
            # Calcular saturación y valor promedio
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])
            
            # Clasificar color
            # Si baja saturación o valor → gris/blanco/negro
            if avg_saturation < 30:
                if avg_value < 50:
                    return "negro"
                elif avg_value > 200:
                    return "blanco"
                else:
                    return "gris"
            
            # Mapeo de hue a nombre de color (HSV tiene hue 0-180)
            # 0-15: rojo, 15-25: naranja, 25-35: amarillo, 35-85: verde
            # 85-125: azul, 125-145: morado, 145-180: rojo
            hue_ranges = [
                (0, 15, "rojo"),
                (15, 25, "naranja"),
                (25, 40, "amarillo"),
                (40, 85, "verde"),
                (85, 125, "azul"),
                (125, 145, "morado"),
                (145, 180, "rojo"),
            ]
            
            hue_value = dominant_hue_idx * 15  # Convertir índice a valor de hue
            
            for low, high, color_name in hue_ranges:
                if low <= hue_value < high:
                    return color_name
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extrayendo color: {e}")
            return None

    # =========================================================================
    # 5. ENRIQUECER OBJETO CON TODOS LOS ATRIBUTOS
    # =========================================================================
    def _enrich_object(
        self,
        obj: DetectedObject,
        image: np.ndarray,
        img_width: int,
        img_height: int,
        img_area: int
    ) -> Dict:
        """
        Enriquece un objeto detectado con toda la información espacial y visual.
        
        Returns:
            Dict con detected_object, position, distance, color, etc.
        """
        bbox = obj.bounding_box
        bbox_area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
        
        # Distancia
        distance_zone, distance_label = self._estimate_distance(bbox_area, img_area)
        
        # Posición espacial
        position = self._get_spatial_position(bbox, img_width, img_height)
        
        # Color dominante (desactivado por defecto - da resultados incorrectos)
        color = None
        if self.enable_color_detection:
            color = self._get_dominant_color(image, bbox)
        
        # Actualizar el objeto con la zona de distancia
        obj.distance_zone = distance_zone
        obj.distance_estimate = distance_label
        
        return {
            "detected_object": obj,
            "name_es": obj.name_es,
            "position": position,
            "distance_zone": distance_zone,
            "distance_label": distance_label,
            "color": color,
            "confidence": obj.confidence,
        }

    # =========================================================================
    # 6. OCR CONDICIONAL - FILTRO ESTRICTO PARA EXPLORACIÓN
    # =========================================================================
    def _conditional_ocr(self, image: np.ndarray, img_area: int) -> Optional[Dict]:
        """
        Ejecuta OCR solo si tiene sentido y el texto es coherente.
        
        En modo EXPLORACIÓN, el OCR debe ser muy selectivo porque:
        - Fotos de exteriores tienen texturas que se confunden con texto
        - Hojas, vegetación, superficies rugosas generan falsos positivos
        - El usuario necesita información útil, no ruido
        
        Filtros aplicados:
        1. Confianza OCR >= 60%
        2. Mínimo 6 caracteres
        3. Ratio alfanumérico > 70%
        4. Longitud promedio de palabra >= 2.5 caracteres
        5. No más de 70% mayúsculas (ruido suele ser TODO MAYÚSCULAS)
        6. Detectar patrones de ruido comunes
        7. Al menos 50% de palabras deben ser "reales" (>= 2 chars)
        
        Returns:
            Resultado OCR filtrado o None si no pasa
        """
        try:
            result = self.ocr_service.extract_text(image)
            
            if not result.get("has_text"):
                return None
            
            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0)
            word_count = result.get("word_count", 0)
            
            # Filtro 1: Confianza mínima (60%)
            if confidence < self.min_ocr_confidence:
                logger.info(f"[Exploración OCR] Descartado: confianza {confidence:.0f}% < {self.min_ocr_confidence}%")
                return None
            
            # Filtro 2: Longitud mínima
            if len(text) < self.min_ocr_chars:
                logger.info(f"[Exploración OCR] Descartado: texto muy corto ({len(text)} chars)")
                return None
            
            # Filtro 3: Ratio alfanumérico (debe ser > 70%)
            alpha_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text) if text else 0
            if alpha_ratio < 0.7:
                logger.info(f"[Exploración OCR] Descartado: muchos símbolos (alpha_ratio={alpha_ratio:.2f})")
                return None
            
            # Filtro 4: Longitud promedio de palabras
            words = text.split()
            if words:
                avg_word_len = sum(len(w) for w in words) / len(words)
                if avg_word_len < self.min_word_length_avg:
                    logger.info(f"[Exploración OCR] Descartado: palabras muy cortas (avg={avg_word_len:.1f})")
                    return None
            
            # Filtro 5: Demasiadas mayúsculas (ruido visual suele ser TODO MAYÚSCULAS)
            letters = [c for c in text if c.isalpha()]
            if letters:
                uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
                if uppercase_ratio > self.max_uppercase_ratio:
                    logger.info(f"[Exploración OCR] Descartado: demasiadas mayúsculas ({uppercase_ratio:.0%})")
                    return None
            
            # Filtro 6: Detectar patrones de ruido comunes
            # Si hay muchas secuencias de consonantes sin vocales = ruido
            noise_patterns = self._detect_noise_patterns(text)
            if noise_patterns:
                logger.info(f"[Exploración OCR] Descartado: patrones de ruido detectados")
                return None
            
            # Filtro 7: Al menos 50% de palabras deben ser "reales" (>= 3 chars)
            real_words = [w for w in words if len(w) >= 3]
            if words and len(real_words) / len(words) < 0.5:
                logger.info(f"[Exploración OCR] Descartado: pocas palabras reales ({len(real_words)}/{len(words)})")
                return None
            
            logger.info(f"[Exploración OCR] Aceptado: {word_count} palabras, {confidence:.0f}% confianza")
            
            return {
                "has_text": True,
                "text": text,
                "confidence": confidence,
                "word_count": word_count,
            }
            
        except Exception as e:
            logger.debug(f"Error en OCR condicional: {e}")
            return None

    def _detect_noise_patterns(self, text: str) -> bool:
        """
        Detecta patrones comunes de ruido visual en OCR.
        
        Patrones de ruido:
        - Muchas secuencias de 3+ consonantes sin vocales (ej: "BSNR", "PTRK")
        - Palabras que son solo 1-2 caracteres repetidos
        - Secuencias que alternan mayúsculas/minúsculas sin sentido
        
        Returns:
            True si se detecta ruido, False si parece texto real
        """
        words = text.split()
        
        if not words:
            return True
        
        noise_word_count = 0
        vowels = set('aeiouAEIOUáéíóúÁÉÍÓÚ')
        
        for word in words:
            # Solo analizar palabras de 3+ caracteres
            if len(word) < 3:
                continue
            
            # Contar vocales
            vowel_count = sum(1 for c in word if c in vowels)
            letter_count = sum(1 for c in word if c.isalpha())
            
            if letter_count == 0:
                continue
            
            vowel_ratio = vowel_count / letter_count
            
            # Si una palabra de 4+ letras no tiene vocales = ruido
            if letter_count >= 4 and vowel_count == 0:
                noise_word_count += 1
                continue
            
            # Si ratio de vocales es muy bajo (< 15%) = probablemente ruido
            if letter_count >= 4 and vowel_ratio < 0.15:
                noise_word_count += 1
                continue
            
            # Detectar secuencias de consonantes muy largas (4+)
            consonant_streak = 0
            max_consonant_streak = 0
            for c in word:
                if c.isalpha() and c not in vowels:
                    consonant_streak += 1
                    max_consonant_streak = max(max_consonant_streak, consonant_streak)
                else:
                    consonant_streak = 0
            
            if max_consonant_streak >= 4:
                noise_word_count += 1
        
        # Si más del 40% de las palabras analizables son ruido = descartar
        analyzable_words = [w for w in words if len(w) >= 3]
        if analyzable_words and noise_word_count / len(analyzable_words) > 0.4:
            return True
        
        return False

    # =========================================================================
    # 6.5 CONTEXTO AMBIENTAL
    # =========================================================================
    def _detect_environment_context(self, objects: List[DetectedObject]) -> Optional[str]:
        """
        Infiere si el entorno es interior o exterior a partir de los objetos detectados.

        Retorna "interior", "exterior" o None si no hay suficientes señales.
        Se requieren al menos 2 señales del mismo tipo para evitar falsos positivos.
        """
        interior_signals = {
            "silla", "mesa", "sofá", "cama", "escritorio", "estante",
            "televisor", "refrigerador", "estufa", "lavadora", "secadora",
            "inodoro", "lavamanos", "bañera", "puerta", "puerta de vidrio",
            "puerta cerrada", "lámpara", "ventana", "espejo", "cortina",
            "alfombra", "armario", "closet", "microondas", "tostadora",
        }
        exterior_signals = {
            "carro", "autobús", "camión", "motocicleta", "bicicleta",
            "scooter eléctrico", "taxi", "semáforo", "señal de pare",
            "señal de tráfico", "poste", "farola", "árbol", "borde de acera",
            "tapa de alcantarilla", "reductor de velocidad", "andamio",
            "contenedor de basura", "cono de tráfico", "barrera vial",
        }

        names = {o.name_es.lower() for o in objects if o.name_es}
        interior_count = len(names & interior_signals)
        exterior_count = len(names & exterior_signals)

        if interior_count >= 2 and interior_count > exterior_count:
            return "interior"
        if exterior_count >= 2 and exterior_count > interior_count:
            return "exterior"
        return None

    # =========================================================================
    # 7. GENERACIÓN DE NARRATIVA ESTRUCTURADA
    # =========================================================================
    def _generate_narrative(
        self,
        enriched_objects: List[Dict],
        ocr_result: Optional[Dict],
        all_objects: Optional[List[DetectedObject]] = None,
    ) -> str:
        """
        Genera narrativa natural y concisa.

        Formato mejorado:
        - Prefijo de contexto ambiental (interior/exterior) si hay suficientes señales
        - Objeto principal: descripción completa con color si está disponible
        - Objetos secundarios: más breve, evita redundancia
        - Agrupa por posición similar

        Ejemplo:
        "Parece un entorno interior. Un ventilador de color blanco frente a ti, muy cerca.
         También hay una cómoda y una mesa a la derecha."
        """
        if not enriched_objects:
            return "No se detectan objetos relevantes en el entorno."

        # Detectar contexto ambiental
        context_prefix = ""
        if all_objects:
            env = self._detect_environment_context(all_objects)
            if env == "interior":
                context_prefix = "Parece un entorno interior. "
            elif env == "exterior":
                context_prefix = "Parece un entorno exterior. "
        
        parts = []

        person_names = {"persona", "niño", "bebé", "persona en silla de ruedas"}
        persons = [o for o in enriched_objects if o["name_es"] in person_names]
        has_persons = len(persons) > 0

        # Si hay personas, suprimir accesorios para no generar ruido
        # ("lentes a tu derecha lejos" junto a "una persona frente a ti" no tiene sentido)
        if has_persons:
            non_persons = [
                o for o in enriched_objects
                if o["name_es"] not in person_names
                and o["name_es"] not in PERSON_ACCESSORIES
            ]
        else:
            non_persons = [o for o in enriched_objects if o["name_es"] not in person_names]

        # Narrar personas
        if len(persons) > 1:
            positions = [o["position"]["horizontal"] for o in persons]
            unique_positions = list(dict.fromkeys(positions))
            if len(unique_positions) == 1:
                parts.append(f"Hay {len(persons)} personas {unique_positions[0]}.")
            else:
                pos_str = " y ".join(unique_positions)
                parts.append(f"Hay {len(persons)} personas: {pos_str}.")
        elif len(persons) == 1:
            parts.append(self._describe_object(persons[0], is_main=(not non_persons)))

        # Narrar objetos no-persona (máx 2 para no sobrecargar)
        if non_persons:
            parts.append(self._describe_object(non_persons[0], is_main=(not has_persons)))
            if len(non_persons) > 1:
                secondary_phrase = self._describe_secondary_objects(non_persons[1:3])
                if secondary_phrase:
                    parts.append(secondary_phrase)

        # Texto detectado
        if ocr_result and ocr_result.get("has_text"):
            text = ocr_result["text"]
            word_count = ocr_result.get("word_count", 0)
            if word_count <= 10:
                parts.append(f"Se lee: {text}")
            else:
                preview = ' '.join(text.split()[:10])
                parts.append(f"Hay un texto que dice: {preview}...")

        narrative = " ".join(parts) if parts else "No se detectan objetos relevantes en el entorno."
        return context_prefix + narrative

    def _describe_object(self, obj_data: Dict, is_main: bool = False) -> str:
        """
        Genera descripción de un objeto individual.
        
        Args:
            obj_data: Datos enriquecidos del objeto
            is_main: Si es el objeto principal (descripción más completa)
        """
        name_es = obj_data["name_es"]
        position = obj_data["position"]["horizontal"]  # Solo horizontal para ser conciso
        distance = obj_data["distance_label"]
        
        # Determinar artículo
        gender = GENDER_MAP.get(name_es, "m")
        article = "Una" if gender == "f" else "Un"
        
        # Formato conciso: "Un ventilador frente a ti, muy cerca."
        color = obj_data.get("color")
        if color and name_es not in {"persona", "niño", "bebé", "persona en silla de ruedas"}:
            phrase = f"{article} {name_es} de color {color} {position}, {distance}."
        else:
            phrase = f"{article} {name_es} {position}, {distance}."

        # Verificar si tiene accesorios (persona + lentes/sombrero/etc)
        accessories = obj_data.get("accessories")
        if accessories and len(accessories) > 0:
            accessory = accessories[0]
            acc_gender = GENDER_MAP.get(accessory, "m")
            acc_article = "unos" if accessory.endswith("s") else ("una" if acc_gender == "f" else "un")
            phrase = phrase.rstrip(".") + f". Tiene puesto {acc_article} {accessory}."

        return phrase

    def _describe_secondary_objects(self, objects: List[Dict]) -> str:
        """
        Describe objetos secundarios de forma concisa.
        
        Agrupa objetos en la misma posición para evitar repetición.
        Ejemplo: "También hay una cómoda y una mesa a la derecha."
        """
        if not objects:
            return ""
        
        # Agrupar por posición horizontal
        by_position = {}
        for obj in objects:
            pos = obj["position"]["horizontal"]
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(obj["name_es"])
        
        # Generar frases agrupadas
        phrases = []
        for position, names in by_position.items():
            # Crear lista con artículos
            items = []
            for name in names:
                gender = GENDER_MAP.get(name, "m")
                article = "una" if gender == "f" else "un"
                items.append(f"{article} {name}")
            
            if len(items) == 1:
                phrases.append(f"{items[0]} {position}")
            else:
                # "una cómoda y una mesa"
                items_str = " y ".join(items)
                phrases.append(f"{items_str} {position}")
        
        if not phrases:
            return ""
        
        # Unir todas las frases
        if len(phrases) == 1:
            return f"También hay {phrases[0]}."
        else:
            return f"También hay {', '.join(phrases[:-1])} y {phrases[-1]}."


# ============================================================================
# INSTANCIA GLOBAL
# ============================================================================
_exploration_service: Optional[ExplorationService] = None


def get_exploration_service() -> ExplorationService:
    """Factory function para obtener el servicio de exploración."""
    global _exploration_service
    if _exploration_service is None:
        _exploration_service = ExplorationService()
    return _exploration_service
