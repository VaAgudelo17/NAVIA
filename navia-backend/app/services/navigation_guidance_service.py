"""
============================================================================
NAVIA Backend - Servicio Unificado de Navegación y Guía
============================================================================
Servicio central que unifica navegación + evaluación de riesgo en un solo
pipeline optimizado para movilidad peatonal de personas ciegas.

Pipeline completo:
  1. Filtrado de clases relevantes para caminata
  2. Cálculo de proximidad (depth + bbox heurística)
  3. Clasificación de altura del obstáculo (suelo/cuerpo/cabeza)
  4. Análisis de movimiento entre frames (se acerca/aleja/estático)
  5. Scoring de riesgo combinado (tipo + proximidad + altura + movimiento)
  6. Generación de instrucciones priorizadas para TTS

Ejemplo de salida:
  "Cuidado: una bicicleta se acerca por la derecha.
   Hay un bordillo muy cerca frente a ti.
   Una persona está a unos metros a la izquierda."

Principios:
  - Máximo 3 instrucciones por frame (evitar sobrecarga cognitiva)
  - Instrucciones cortas, claras y orientadas a acción
  - Priorizar peligros reales para movilidad peatonal
  - Ignorar objetos decorativos e irrelevantes para caminar
============================================================================
"""

import logging
from typing import List, Dict, Optional, Tuple

from app.core.config import settings
from app.models.schemas import DetectedObject, BoundingBox

logger = logging.getLogger(__name__)


# ============================================================================
# CLASES RELEVANTES PARA MOVILIDAD PEATONAL
# ============================================================================
# Solo estos objetos se consideran para navegación.
# Todo lo demás (comida, decoración, electrónica, ropa, etc.) se ignora.

PEDESTRIAN_RELEVANT_CLASSES = {
    # --- PERSONAS (impredecibles, requieren atención) ---
    "persona", "niño", "bebé", "persona en silla de ruedas",

    # --- ANIMALES que obstruyen el paso ---
    "perro", "gato", "cachorro",

    # --- VEHÍCULOS (peligro crítico) ---
    "carro", "autobús", "camión", "motocicleta", "bicicleta",
    "scooter eléctrico", "taxi", "ambulancia", "patrulla",

    # --- OBSTÁCULOS DE SUELO (riesgo de tropiezo/caída) ---
    "escaleras", "escalera mecánica", "borde de acera",
    "tapa de alcantarilla", "reductor de velocidad",
    "alfombra", "tapete", "maleta", "mochila", "patineta",
    "pelota", "caja de cartón", "bolsa plástica",

    # --- OBSTÁCULOS A NIVEL CORPORAL ---
    "silla", "mesa", "escritorio", "banco", "mostrador",
    "mesa de centro", "mesa de comedor", "taburete",
    "mecedora", "sofá", "sillón", "cama",
    "carrito de compras", "cochecito de bebé", "silla de ruedas",
    "contenedor de basura", "contenedor de reciclaje",
    "aspiradora", "tabla de planchar",

    # --- INFRAESTRUCTURA DE CALLE ---
    "poste", "farola", "hidrante", "semáforo",
    "señal de pare", "cono de tráfico", "barrera vial",
    "señal de construcción", "señal de estacionamiento",
    "parquímetro", "buzón",

    # --- BARRERAS Y ESTRUCTURAS ---
    "cerca", "portón", "árbol", "arbusto",

    # --- MUROS Y PAREDES ---
    "pared", "pared de ladrillo", "pared de vidrio",
    "muro de piedra", "pilar", "columna",

    # --- BALCONES Y TERRAZAS (peligro de caída) ---
    "balcón", "barandal de balcón", "terraza",
    "cornisa", "barandal",

    # --- PUERTAS Y ACCESOS ---
    "puerta", "puerta corrediza", "ascensor", "pasamanos",
    "puerta abierta", "puerta cerrada", "puerta de vidrio",
    "puerta giratoria", "reja", "puerta de garaje",
    "salida de emergencia", "marco de puerta",

    # --- PELIGROS DOMÉSTICOS ---
    "cuchillo", "tijeras", "estufa", "horno",

    # --- SEÑALES DE SEGURIDAD ---
    "señal de advertencia", "señal de piso mojado",
    "señal de salida",

    # --- ELECTRODOMÉSTICOS GRANDES EN EL PASO ---
    "refrigerador", "lavadora", "secadora", "lavavajillas",

    # --- OBJETOS DE BAÑO (resbaladizos) ---
    "inodoro", "lavamanos", "bañera", "ducha", "báscula",
}


# ============================================================================
# PESO SEMÁNTICO POR TIPO DE OBJETO
# ============================================================================
# Cuánto "peligro base" representa cada tipo de objeto para un peatón ciego.
# Escala 0.0 - 1.0 donde 1.0 = máximo peligro.

DANGER_WEIGHT: Dict[str, float] = {
    # --- Vehículos: máximo peligro ---
    "carro": 1.0, "autobús": 1.0, "camión": 1.0,
    "motocicleta": 0.95, "bicicleta": 0.85,
    "scooter eléctrico": 0.85, "taxi": 1.0,
    "ambulancia": 1.0, "patrulla": 1.0,

    # --- Obstáculos de suelo: alto riesgo de caída ---
    "escaleras": 0.9, "escalera mecánica": 0.9,
    "borde de acera": 0.85, "tapa de alcantarilla": 0.8,
    "reductor de velocidad": 0.7,
    "alfombra": 0.3, "tapete": 0.3,
    "maleta": 0.5, "mochila": 0.4,
    "patineta": 0.6, "pelota": 0.4,
    "caja de cartón": 0.4, "bolsa plástica": 0.3,

    # --- Personas y animales: impredecibles ---
    "persona": 0.6, "niño": 0.65, "bebé": 0.5,
    "persona en silla de ruedas": 0.6,
    "perro": 0.7, "gato": 0.5, "cachorro": 0.5,

    # --- Infraestructura de calle ---
    "poste": 0.75, "farola": 0.7, "hidrante": 0.65,
    "semáforo": 0.5, "señal de pare": 0.4,
    "cono de tráfico": 0.6, "barrera vial": 0.7,
    "señal de construcción": 0.55,
    "señal de estacionamiento": 0.3,
    "parquímetro": 0.5, "buzón": 0.45,

    # --- Muebles y objetos grandes ---
    "silla": 0.55, "mesa": 0.6, "escritorio": 0.55,
    "banco": 0.5, "mostrador": 0.55,
    "mesa de centro": 0.55, "mesa de comedor": 0.6,
    "taburete": 0.5, "mecedora": 0.5,
    "sofá": 0.5, "sillón": 0.5, "cama": 0.45,
    "carrito de compras": 0.6, "cochecito de bebé": 0.6,
    "silla de ruedas": 0.6,
    "contenedor de basura": 0.55, "contenedor de reciclaje": 0.5,
    "aspiradora": 0.4, "tabla de planchar": 0.45,

    # --- Barreras y estructuras ---
    "cerca": 0.5, "portón": 0.55,
    "árbol": 0.55, "arbusto": 0.4,

    # --- Muros y paredes (obstáculo sólido, no se puede atravesar) ---
    "pared": 0.6, "pared de ladrillo": 0.6,
    "pared de vidrio": 0.7,  # Más peligroso: invisible
    "muro de piedra": 0.6,
    "pilar": 0.7, "columna": 0.7,  # Fácil de chocar de frente

    # --- Balcones y terrazas (peligro de caída) ---
    "balcón": 0.85, "barandal de balcón": 0.8,
    "terraza": 0.75, "cornisa": 0.9,  # Máximo peligro: caída
    "barandal": 0.7,

    # --- Puertas y accesos ---
    "puerta": 0.4, "puerta corrediza": 0.4,
    "ascensor": 0.35, "pasamanos": 0.2,
    "puerta abierta": 0.35, "puerta cerrada": 0.5,  # Cerrada = obstáculo
    "puerta de vidrio": 0.65,  # Peligro: invisible, fácil de chocar
    "puerta giratoria": 0.6,  # Mecanismo en movimiento
    "reja": 0.5, "puerta de garaje": 0.55,
    "salida de emergencia": 0.2,  # Referencia útil, bajo peligro
    "marco de puerta": 0.35,

    # --- Peligros domésticos ---
    "cuchillo": 0.7, "tijeras": 0.5,
    "estufa": 0.6, "horno": 0.55,

    # --- Señales de seguridad ---
    "señal de advertencia": 0.5, "señal de piso mojado": 0.6,
    "señal de salida": 0.2,

    # --- Electrodomésticos grandes ---
    "refrigerador": 0.45, "lavadora": 0.4,
    "secadora": 0.4, "lavavajillas": 0.35,

    # --- Baño ---
    "inodoro": 0.35, "lavamanos": 0.35,
    "bañera": 0.45, "ducha": 0.35, "báscula": 0.35,
}

# Peso por defecto para objetos relevantes sin peso específico
_DEFAULT_DANGER_WEIGHT = 0.4


# ============================================================================
# CLASIFICACIÓN DE ALTURA
# ============================================================================
# Tipos de riesgo según altura del obstáculo:
#   - suelo: riesgo de tropiezo (y_max en zona inferior de la imagen)
#   - cuerpo: riesgo de colisión frontal (zona media)
#   - cabeza: riesgo de golpe en cabeza (zona superior)

HEIGHT_ZONE_LABELS = {
    "suelo": "a nivel del suelo",
    "cuerpo": "a nivel del cuerpo",
    "cabeza": "a nivel de la cabeza",
}

# Multiplicadores de peligro por altura
# Cabeza es más crítico porque es menos esperado y más dañino
HEIGHT_RISK_MULTIPLIER = {
    "suelo": 1.0,
    "cuerpo": 1.1,
    "cabeza": 1.3,
}


# ============================================================================
# ETIQUETAS DE PROXIMIDAD PARA TTS
# ============================================================================
PROXIMITY_LABELS = {
    "muy_cerca": "muy cerca",
    "cerca": "a unos metros",
    "lejos": "a lo lejos",
}


class NavigationGuidanceService:
    """
    Servicio unificado de navegación y riesgo para movilidad peatonal.

    Pipeline por frame:
      1. Filtrar clases relevantes para caminata
      2. Calcular posición espacial (izquierda/frente/derecha)
      3. Clasificar altura (suelo/cuerpo/cabeza)
      4. Analizar movimiento entre frames (se acerca/aleja/estático)
      5. Calcular score de riesgo combinado
      6. Generar instrucciones priorizadas para TTS

    Diseñado para usarse tanto en endpoints HTTP como WebSocket.
    """

    def __init__(self):
        # Estado de movimiento: almacena bboxes del frame anterior
        # Clave: name_es, Valor: (center_x, center_y, area)
        self._previous_bboxes: Dict[str, Tuple[float, float, float]] = {}
        self._frame_count: int = 0

    # ====================================================================
    # PIPELINE PRINCIPAL
    # ====================================================================

    def generate_guidance(
        self,
        objects: List[DetectedObject],
        img_width: int,
        img_height: int,
        track_movement: bool = True,
    ) -> dict:
        """
        Pipeline completo de navegación + riesgo.

        Args:
            objects: Objetos detectados por YOLO con zonas de distancia
            img_width: Ancho de la imagen en píxeles
            img_height: Alto de la imagen en píxeles
            track_movement: Si True, analiza movimiento entre frames

        Returns:
            dict con:
              - instruction: str (texto para TTS)
              - obstacles: list (obstáculos procesados)
              - path_clear: bool (si el camino central está libre)
              - has_danger: bool (si hay peligro real)
              - priority: str (none/medium/high/critical)
              - obstacle_details: list (detalles completos para UI)
        """
        self._frame_count += 1

        if not objects:
            self._previous_bboxes.clear()
            return self._empty_result()

        # --- PASO 1: Filtrar clases relevantes para caminata ---
        relevant = self._filter_pedestrian_relevant(objects)

        if not relevant:
            self._previous_bboxes.clear()
            return self._empty_result()

        # --- PASO 2: Enriquecer cada objeto con análisis completo ---
        analyzed = []
        current_bboxes: Dict[str, Tuple[float, float, float]] = {}

        for obj in relevant:
            bbox = obj.bounding_box
            # Posición horizontal
            position = self._get_spatial_position(bbox, img_width)
            # Altura del obstáculo
            height_zone = self._classify_height(bbox, img_height)
            # Proximidad (usar distance_zone de depth o heurística)
            proximity = obj.distance_zone or "lejos"

            # Movimiento (comparar con frame anterior)
            movement = "estatico"
            if track_movement:
                center_x = (bbox.x_min + bbox.x_max) / 2
                center_y = (bbox.y_min + bbox.y_max) / 2
                area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
                current_bboxes[obj.name_es] = (center_x, center_y, area)
                movement = self._analyze_movement(
                    obj.name_es, center_x, center_y, area,
                    img_width, img_height
                )

            # Score de riesgo combinado
            risk_score = self._calculate_risk_score(
                obj.name_es, proximity, height_zone, movement
            )

            analyzed.append({
                "object": obj,
                "name_es": obj.name_es,
                "position": position,
                "height_zone": height_zone,
                "proximity": proximity,
                "movement": movement,
                "risk_score": risk_score,
                "confidence": obj.confidence,
            })

        # Actualizar estado de movimiento
        if track_movement:
            self._previous_bboxes = current_bboxes

        # --- PASO 3: Ordenar por riesgo (mayor primero) ---
        analyzed.sort(key=lambda x: (-x["risk_score"], -x["confidence"]))

        # --- PASO 4: Determinar estado del camino ---
        center_obstacles = [
            a for a in analyzed
            if a["position"] == "frente a ti"
            and a["proximity"] in ("muy_cerca", "cerca")
        ]
        path_clear = len(center_obstacles) == 0

        # --- PASO 5: Determinar prioridad global ---
        max_score = analyzed[0]["risk_score"] if analyzed else 0
        priority = self._score_to_priority(max_score)
        has_danger = priority in ("critical", "high")

        # --- PASO 6: Generar instrucciones ---
        instruction = self._generate_instructions(analyzed, path_clear)

        return {
            "instruction": instruction,
            "obstacles": [a["object"] for a in analyzed if a["proximity"] != "lejos"],
            "path_clear": path_clear,
            "has_danger": has_danger,
            "priority": priority,
            "obstacle_details": [
                {
                    "name": a["name_es"],
                    "position": a["position"],
                    "proximity": a["proximity"],
                    "height_zone": a["height_zone"],
                    "movement": a["movement"],
                    "risk_score": round(a["risk_score"], 2),
                }
                for a in analyzed[:6]
            ],
        }

    # ====================================================================
    # PASO 1: FILTRADO DE CLASES RELEVANTES
    # ====================================================================

    def _filter_pedestrian_relevant(
        self, objects: List[DetectedObject]
    ) -> List[DetectedObject]:
        """
        Filtra solo objetos relevantes para movilidad peatonal.

        Ignora automáticamente: cielo, césped, paredes, comida,
        decoración, electrónica pequeña, ropa, maquillaje, joyería, etc.
        """
        return [
            obj for obj in objects
            if obj.name_es in PEDESTRIAN_RELEVANT_CLASSES
        ]

    # ====================================================================
    # PASO 2: POSICIÓN ESPACIAL
    # ====================================================================

    @staticmethod
    def _get_spatial_position(bbox: BoundingBox, img_width: int) -> str:
        """
        Posición horizontal del objeto respecto al usuario.

        Divide la imagen en 3 columnas:
          - Izquierda: 0-33%
          - Frente: 33-66%
          - Derecha: 66-100%
        """
        center_x = (bbox.x_min + bbox.x_max) / 2
        ratio = center_x / img_width if img_width > 0 else 0.5

        if ratio < 0.33:
            return "a tu izquierda"
        elif ratio > 0.66:
            return "a tu derecha"
        else:
            return "frente a ti"

    # ====================================================================
    # PASO 3: CLASIFICACIÓN DE ALTURA
    # ====================================================================

    @staticmethod
    def _classify_height(bbox: BoundingBox, img_height: int) -> str:
        """
        Clasifica la altura del obstáculo usando la posición vertical
        del centro del bounding box.

        En una cámara frontal (a nivel del pecho/cara):
          - Zona inferior (>66% de la imagen): obstáculo a nivel del suelo
          - Zona media (33-66%): obstáculo a nivel del cuerpo
          - Zona superior (<33%): obstáculo a nivel de cabeza

        Args:
            bbox: Bounding box del objeto
            img_height: Alto de la imagen

        Returns:
            "suelo", "cuerpo", o "cabeza"
        """
        center_y = (bbox.y_min + bbox.y_max) / 2
        ratio = center_y / img_height if img_height > 0 else 0.5

        if ratio > 0.66:
            return "suelo"
        elif ratio < 0.33:
            return "cabeza"
        else:
            return "cuerpo"

    # ====================================================================
    # PASO 4: ANÁLISIS DE MOVIMIENTO
    # ====================================================================

    def _analyze_movement(
        self,
        name_es: str,
        center_x: float,
        center_y: float,
        area: float,
        img_width: int,
        img_height: int,
    ) -> str:
        """
        Analiza si un objeto se acerca, se aleja o está estático
        comparando su bounding box con el frame anterior.

        Criterios:
          - Si el área crece > 10%: se acerca
          - Si el área decrece > 10%: se aleja
          - Si se desplaza hacia el centro de la imagen: trayectoria de colisión
          - De lo contrario: estático

        Args:
            name_es: Nombre del objeto en español
            center_x, center_y: Centro actual del bbox
            area: Área actual del bbox
            img_width, img_height: Dimensiones de la imagen

        Returns:
            "acercandose", "alejandose", o "estatico"
        """
        prev = self._previous_bboxes.get(name_es)
        if prev is None:
            return "estatico"

        prev_cx, prev_cy, prev_area = prev

        # Evitar división por cero
        if prev_area <= 0:
            return "estatico"

        # Cambio de área (si crece, el objeto se acerca)
        area_change = (area - prev_area) / prev_area

        # Cambio de posición hacia el centro de la imagen
        img_center_x = img_width / 2
        prev_dist_to_center = abs(prev_cx - img_center_x)
        curr_dist_to_center = abs(center_x - img_center_x)
        moving_toward_center = curr_dist_to_center < prev_dist_to_center * 0.85

        # Decidir movimiento
        if area_change > 0.10:
            return "acercandose"
        elif area_change < -0.10:
            return "alejandose"
        elif moving_toward_center and area_change > 0.03:
            # Se mueve hacia el centro Y crece un poco: trayectoria de colisión
            return "acercandose"
        else:
            return "estatico"

    # ====================================================================
    # PASO 5: SCORING DE RIESGO
    # ====================================================================

    def _calculate_risk_score(
        self,
        name_es: str,
        proximity: str,
        height_zone: str,
        movement: str,
    ) -> float:
        """
        Calcula un puntaje de riesgo combinado (0.0 - 1.0).

        Fórmula:
          score = peso_objeto × mult_proximidad × mult_altura × mult_movimiento

        Todos los multiplicadores están normalizados para que el score
        máximo sea ~1.0 para el caso más peligroso.

        Args:
            name_es: Nombre del objeto
            proximity: "muy_cerca", "cerca", "lejos"
            height_zone: "suelo", "cuerpo", "cabeza"
            movement: "acercandose", "alejandose", "estatico"

        Returns:
            Score de riesgo [0.0, ~1.0]
        """
        # Peso base del objeto
        base_weight = DANGER_WEIGHT.get(name_es, _DEFAULT_DANGER_WEIGHT)

        # Multiplicador de proximidad
        proximity_mult = {
            "muy_cerca": 1.0,
            "cerca": 0.6,
            "lejos": 0.2,
        }.get(proximity, 0.2)

        # Multiplicador de altura
        height_mult = HEIGHT_RISK_MULTIPLIER.get(height_zone, 1.0)

        # Multiplicador de movimiento
        movement_mult = {
            "acercandose": 1.4,
            "estatico": 1.0,
            "alejandose": 0.6,
        }.get(movement, 1.0)

        score = base_weight * proximity_mult * height_mult * movement_mult

        # Clamp a [0, 1]
        return min(1.0, max(0.0, score))

    @staticmethod
    def _score_to_priority(score: float) -> str:
        """
        Convierte score numérico a nivel de prioridad.

        Umbrales:
          - critical: >= 0.75 (peligro inmediato)
          - high: >= 0.50 (requiere atención)
          - medium: >= 0.25 (precaución)
          - none: < 0.25 (sin peligro significativo)
        """
        if score >= 0.75:
            return "critical"
        elif score >= 0.50:
            return "high"
        elif score >= 0.25:
            return "medium"
        return "none"

    # ====================================================================
    # PASO 6: GENERACIÓN DE INSTRUCCIONES
    # ====================================================================

    def _generate_instructions(
        self,
        analyzed: list,
        path_clear: bool,
    ) -> str:
        """
        Genera instrucciones de navegación priorizadas para TTS.

        Reglas:
          1. Máximo 3 instrucciones (evitar sobrecarga cognitiva)
          2. Primero alertas críticas (objetos cercanos, en movimiento, peligrosos)
          3. Luego obstáculos secundarios
          4. Formato: claro, corto, orientado a acción

        Args:
            analyzed: Lista de objetos analizados ordenados por riesgo
            path_clear: Si el camino central está libre

        Returns:
            Texto de instrucciones para TTS
        """
        if not analyzed:
            return "Camino libre."

        phrases = []
        max_phrases = 3

        # Filtrar solo obstáculos que merecen mención
        # (no mencionar objetos lejanos sin movimiento)
        mentionable = [
            a for a in analyzed
            if a["proximity"] in ("muy_cerca", "cerca")
            or a["movement"] == "acercandose"
        ]

        if not mentionable:
            return "Camino libre."

        for item in mentionable:
            if len(phrases) >= max_phrases:
                break

            name = item["name_es"]
            position = item["position"]
            proximity = item["proximity"]
            movement = item["movement"]
            height = item["height_zone"]
            score = item["risk_score"]

            phrase = self._build_phrase(
                name, position, proximity, movement, height, score
            )
            if phrase:
                phrases.append(phrase)

        # Añadir indicación de camino libre si aplica
        if path_clear and phrases:
            # Solo si hay obstáculos laterales pero el centro está libre
            has_center_obstacle = any(
                a["position"] == "frente a ti" and a["proximity"] != "lejos"
                for a in mentionable
            )
            if not has_center_obstacle:
                phrases.append("El camino al frente está libre")

        if not phrases:
            return "Camino libre."

        return ". ".join(phrases) + "."

    @staticmethod
    def _build_phrase(
        name: str,
        position: str,
        proximity: str,
        movement: str,
        height: str,
        score: float,
    ) -> str:
        """
        Construye una frase individual de alerta.

        Formato según nivel de peligro:
          - Critical: "Cuidado: [objeto] [movimiento] [posición]"
          - High: "Atención: [objeto] [proximidad] [posición]"
          - Medium: "[objeto] [proximidad] [posición]"

        Args:
            name: Nombre del objeto
            position: "a tu izquierda", "frente a ti", "a tu derecha"
            proximity: "muy_cerca", "cerca", "lejos"
            movement: "acercandose", "alejandose", "estatico"
            height: "suelo", "cuerpo", "cabeza"
            score: Score de riesgo

        Returns:
            Frase de alerta o string vacío si no merece mención
        """
        name_cap = name.capitalize()
        prox_label = PROXIMITY_LABELS.get(proximity, proximity)

        # === FRASES ESPECÍFICAS PARA ESTRUCTURAS ===

        # Balcones y cornisas: peligro de caída, siempre alertar
        _BALCONY_CLASSES = {"balcón", "barandal de balcón", "terraza", "cornisa"}
        if name in _BALCONY_CLASSES:
            if proximity == "muy_cerca":
                return f"Peligro: {name_cap} {prox_label} {position}, riesgo de caída"
            elif proximity == "cerca":
                return f"Precaución: {name_cap} {prox_label} {position}"
            return ""

        # Paredes y muros: indicar dirección para esquivar
        _WALL_CLASSES = {"pared", "pared de ladrillo", "pared de vidrio", "muro de piedra", "pilar", "columna"}
        if name in _WALL_CLASSES:
            if proximity == "muy_cerca":
                return f"Cuidado: {name_cap} {prox_label} {position}"
            elif proximity == "cerca":
                return f"{name_cap} {prox_label} {position}"
            return ""

        # Puertas: indicar si está abierta/cerrada
        _DOOR_CLASSES = {"puerta abierta", "puerta cerrada", "puerta de vidrio", "puerta giratoria", "reja", "puerta de garaje", "marco de puerta"}
        if name in _DOOR_CLASSES:
            if proximity in ("muy_cerca", "cerca"):
                return f"{name_cap} {prox_label} {position}"
            return ""

        # === FRASES GENÉRICAS ===

        # Objetos que se acercan: alerta de movimiento
        if movement == "acercandose":
            if score >= 0.75:
                return f"Cuidado: {name_cap} se acerca {position}"
            else:
                return f"Atención: {name_cap} se acerca {position}"

        # Objetos muy cerca: alerta de proximidad
        if proximity == "muy_cerca":
            # Añadir info de altura solo si es relevante
            height_info = ""
            if height == "suelo":
                height_info = " a nivel del suelo"
            elif height == "cabeza":
                height_info = " a nivel de la cabeza"

            if score >= 0.75:
                return f"Cuidado: {name_cap} {prox_label} {position}{height_info}"
            else:
                return f"{name_cap} {prox_label} {position}{height_info}"

        # Objetos cerca
        if proximity == "cerca":
            return f"{name_cap} {prox_label} {position}"

        # Objetos lejanos que se acercan (ya cubierto arriba)
        return ""

    # ====================================================================
    # UTILIDADES
    # ====================================================================

    @staticmethod
    def _empty_result() -> dict:
        """Resultado cuando no hay obstáculos."""
        return {
            "instruction": "Camino libre.",
            "obstacles": [],
            "path_clear": True,
            "has_danger": False,
            "priority": "none",
            "obstacle_details": [],
        }

    def reset_movement_state(self) -> None:
        """Reinicia el estado de movimiento (nueva sesión o reset)."""
        self._previous_bboxes.clear()
        self._frame_count = 0


# ============================================================================
# SINGLETON
# ============================================================================

_navigation_guidance_service: Optional[NavigationGuidanceService] = None


def get_navigation_guidance_service() -> NavigationGuidanceService:
    """Factory function para obtener el servicio de navegación y guía."""
    global _navigation_guidance_service
    if _navigation_guidance_service is None:
        _navigation_guidance_service = NavigationGuidanceService()
    return _navigation_guidance_service
