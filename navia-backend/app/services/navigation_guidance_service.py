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
    # ── PERSONAS ──────────────────────────────────────────────────────────────
    # Siempre relevantes: se mueven, son impredecibles
    "persona", "niño", "bebé", "persona en silla de ruedas",

    # ── ANIMALES ──────────────────────────────────────────────────────────────
    # Domésticos (comunes en interiores y calle)
    "perro", "cachorro", "gato",
    # Grandes (rurales o eventos, pueden bloquear o atacar)
    "caballo", "vaca", "cerdo", "oveja", "burro",

    # ── VEHÍCULOS ─────────────────────────────────────────────────────────────
    # Peligro máximo: matan
    "carro", "autobús", "camión", "motocicleta", "bicicleta",
    "scooter eléctrico", "taxi", "ambulancia", "patrulla",

    # ── PELIGROS DE SUELO (tropiezo / caída) ──────────────────────────────────
    "escaleras", "escalera mecánica",
    "escalón",                  # un solo peldaño: más peligroso porque no se espera
    "borde de acera",           # desnivel, fácil de tropezar
    "tapa de alcantarilla",     # hueco o superficie resbaladiza
    "reductor de velocidad",    # resalto en la vía
    "maleta", "mochila", "patineta", "caja de cartón",  # objetos inesperados en el piso

    # ── OBSTÁCULOS A NIVEL CORPORAL (colisión frontal) ────────────────────────
    # Muebles principales (suficientes — no necesitamos cada variante)
    "silla", "mesa", "sofá", "cama", "escritorio",
    "banco", "mostrador", "mesa de comedor",
    # Vehículos de paso lento
    "carrito de compras", "cochecito de bebé", "silla de ruedas",
    # Contenedores grandes en la calle
    "contenedor de basura",

    # ── INFRAESTRUCTURA DE CALLE ──────────────────────────────────────────────
    "poste", "farola",          # columnas en la acera, fácil de chocar
    "árbol",                    # tronco puede estar en el camino
    "andamio",                  # zona de construcción, muy peligroso
    "cono de tráfico",          # señal de zona de peligro/obras
    "barrera vial",             # cierra el paso
    "cerca", "portón",          # delimitan el camino
    "semáforo",                 # referencia y obstáculo físico
    "hidrante",                 # obstáculo bajo a nivel de rodilla
    "señal de piso mojado",     # piso resbaladizo

    # ── PAREDES Y ESTRUCTURAS SÓLIDAS ───────────────────────────────────────
    "pared", "pared de ladrillo", "pared de vidrio",
    "pilar", "columna",

    # ── PELIGROS DE CAÍDA ────────────────────────────────────────────────────
    "balcón", "cornisa",
    "barandal de balcón", "barandal",

    # ── PUERTAS Y ACCESOS ────────────────────────────────────────────────────
    "puerta", "puerta abierta", "puerta cerrada",
    "puerta de vidrio", "puerta giratoria", "puerta corrediza",
    "reja", "ascensor",

    # ── COCINA / LAVANDERÍA (interior) ───────────────────────────────────────
    "estufa", "horno",
    "refrigerador", "lavadora", "secadora",

    # ── BAÑO (navegación interior) ────────────────────────────────────────────
    "inodoro", "lavamanos", "bañera",
}


# ============================================================================
# PESO SEMÁNTICO POR TIPO DE OBJETO
# ============================================================================
# Cuánto "peligro base" representa cada tipo de objeto para un peatón ciego.
# Escala 0.0 - 1.0 donde 1.0 = máximo peligro.

DANGER_WEIGHT: Dict[str, float] = {
    # ── VEHÍCULOS: máximo peligro ─────────────────────────────────────────────
    "carro": 1.0, "autobús": 1.0, "camión": 1.0, "taxi": 1.0,
    "ambulancia": 1.0, "patrulla": 1.0,
    "motocicleta": 0.95,
    "bicicleta": 0.85, "scooter eléctrico": 0.85,

    # ── PERSONAS: impredecibles, siempre relevantes ───────────────────────────
    "niño": 0.75,                   # más impredecible que adulto
    "persona": 0.70,
    "persona en silla de ruedas": 0.65,
    "bebé": 0.60,

    # ── ANIMALES ─────────────────────────────────────────────────────────────
    "caballo": 0.90,                # grande y puede asustar/golpear
    "vaca": 0.85, "cerdo": 0.75, "burro": 0.80, "oveja": 0.65,
    "perro": 0.75,                  # puede atacar o cruzarse
    "cachorro": 0.55, "gato": 0.50,

    # ── PELIGROS DE SUELO: caída / tropiezo ───────────────────────────────────
    "escaleras": 0.90, "escalera mecánica": 0.90,
    "cornisa": 0.90,                # caída al vacío
    "balcón": 0.88, "barandal de balcón": 0.82,
    "borde de acera": 0.85,
    "tapa de alcantarilla": 0.80,
    "barandal": 0.72,
    "reductor de velocidad": 0.65,
    "escalón": 0.80,                # un solo peldaño: muy peligroso porque no se espera
    "patineta": 0.65,
    "maleta": 0.55,
    "mochila": 0.45,
    "caja de cartón": 0.40,

    # ── ESTRUCTURAS SÓLIDAS: colisión frontal ─────────────────────────────────
    "pared de vidrio": 0.80,        # invisible, muy peligrosa
    "pared": 0.65, "pared de ladrillo": 0.65,
    "pilar": 0.75, "columna": 0.75,
    "poste": 0.75, "farola": 0.70,
    "árbol": 0.60,

    # ── BARRERAS Y CIERRES ────────────────────────────────────────────────────
    "andamio": 0.80,                # zona de obras, muy peligroso
    "barrera vial": 0.75,
    "cerca": 0.60, "portón": 0.60, "reja": 0.60,
    "cono de tráfico": 0.60,
    "contenedor de basura": 0.55,

    # ── PUERTAS: barrera o peligro de colisión ────────────────────────────────
    "puerta de vidrio": 0.75,       # invisible
    "puerta giratoria": 0.65,       # mecanismo en movimiento
    "puerta cerrada": 0.55,         # obstáculo directo
    "puerta": 0.50, "puerta corrediza": 0.50,
    "puerta abierta": 0.35,         # menos peligro pero delimita el espacio
    "reja": 0.60, "ascensor": 0.40,

    # ── COCINA: peligros en interiores ───────────────────────────────────────
    "estufa": 0.70,                 # quemadura + obstáculo
    "horno": 0.65,
    "refrigerador": 0.50, "lavadora": 0.45, "secadora": 0.45,

    # ── MUEBLES: obstáculos físicos en el paso ────────────────────────────────
    "mesa de comedor": 0.65, "mesa": 0.62,
    "escritorio": 0.60, "mostrador": 0.60,
    "silla": 0.58, "banco": 0.55,
    "cama": 0.55, "sofá": 0.52,
    "carrito de compras": 0.65, "cochecito de bebé": 0.65,
    "silla de ruedas": 0.60,

    # ── SEÑALES / REFERENCIAS ─────────────────────────────────────────────────
    "señal de piso mojado": 0.65,   # piso resbaladizo
    "semáforo": 0.50,               # referencia de cruce + obstáculo físico
    "hidrante": 0.60,               # bajo, fácil de chocar

    # ── BAÑO (navegación interior) ────────────────────────────────────────────
    "bañera": 0.55,
    "inodoro": 0.45, "lavamanos": 0.40,
}

# Peso por defecto para objetos relevantes sin peso específico
_DEFAULT_DANGER_WEIGHT = 0.4

# Objetos intrínsecamente peligrosos → alert_type = "peligro" sin importar distancia.
# Incluye: vehículos, riesgos de caída, peligros térmicos, superficies invisibles.
PELIGRO_CLASSES: set = {
    # Vehículos
    "carro", "autobús", "camión", "taxi", "ambulancia", "patrulla",
    "motocicleta", "bicicleta", "scooter eléctrico",
    # Animales que pueden golpear o asustar
    "caballo", "vaca", "burro", "cerdo",
    # Caídas
    "escaleras", "escalera mecánica", "escalón", "balcón", "cornisa",
    "borde de acera", "tapa de alcantarilla", "barandal de balcón",
    # Peligros térmicos
    "estufa", "horno",
    # Superficies invisibles / peligro de colisión oculta
    "pared de vidrio", "puerta de vidrio",
    # Andamios y obras
    "andamio",
}

# ============================================================================
# LISTA NEGRA — NUNCA reportar como obstáculo
# ============================================================================
# Objetos que YOLO detecta con frecuencia pero que NO bloquean el paso de una
# persona caminando. Incluirlos causaría alertas falsas e innecesarias.
#
# Categorías:
#  • Elementos decorativos de pared/techo (espejo, cortina, ventana)
#  • Calzado y ropa (pequeños, en el piso no obstruyen el paso)
#  • Artículos de higiene personal (diminutos)
#  • Electrónica (no en la trayectoria de caminar)

IGNORE_CLASSES: set = {
    # ── Elementos de pared / techo ────────────────────────────────────────────
    "espejo", "espejo de baño", "espejo compacto",
    "cortina", "cortina de baño", "persiana",
    "ventana",                      # superficie vertical transparente en la pared
    "candelabro", "lámpara", "foco",
    # ── Calzado y ropa ────────────────────────────────────────────────────────
    "zapato", "tenis", "tacón", "bota", "sandalia", "pantufla",
    "ropa", "camisa", "pantalón", "sombrero",
    # ── Higiene personal (objetos pequeños) ───────────────────────────────────
    "toalla de mano", "toalla de baño",
    "champú", "jabón", "pasta dental", "cepillo de dientes",
    "papel higiénico", "rollo de papel",
    # ── Vajilla y utensilios ──────────────────────────────────────────────────
    "taza", "vaso", "plato", "tazón",
    "tenedor", "cuchillo", "cuchara",
    "botella", "lata",
    # ── Electrónica ──────────────────────────────────────────────────────────
    "teléfono", "computadora portátil", "teclado", "mouse",
    "televisión", "control remoto", "tablet",
    "cámara",
    # ── Decoración y arte ─────────────────────────────────────────────────────
    "cuadro", "planta", "florero", "jarrón",
    "almohada", "cojín",
}

# Altura máxima (fracción del alto de imagen, desde arriba) para que un objeto
# en zona superior sea considerado obstáculo válido.
# Por encima de este umbral (objetos en el techo / parte alta de la pared)
# se ignoran, EXCEPTO si son clases de peligro crítico (escaleras, balcón, etc.).
# 0.25 = ignorar si el centro del bbox está en el 25% superior de la imagen.
_TOP_ZONE_THRESHOLD = 0.25


# ============================================================================
# FILTROS DE CALIDAD PARA NAVEGACIÓN
# ============================================================================

# Objetos críticos: se aceptan con umbral de confianza más bajo
# porque un falso negativo (no detectar un carro) es peor que un falso positivo
CRITICAL_OBJECTS = {
    # Vehículos
    "carro", "autobús", "camión", "motocicleta", "bicicleta",
    "taxi", "ambulancia", "patrulla", "scooter eléctrico",
    # Personas
    "persona", "niño", "bebé",
    # Animales grandes (pueden ser peligrosos o bloquear el paso)
    "perro", "caballo", "vaca", "cerdo", "burro",
}

# Confianza mínima para objetos críticos (más permisivo)
MIN_CONFIDENCE_CRITICAL = 0.22
# Confianza mínima para el resto de objetos relevantes
MIN_CONFIDENCE_GENERAL = 0.28
# Área mínima del bounding box como fracción del área total de la imagen
# Objetos que ocupan menos del 0.3% de la imagen son probablemente ruido
MIN_BBOX_AREA_FRACTION = 0.003

# Objetos con alta tasa de confusión visual → requieren mayor confianza
# Clave: nombre en español, Valor: umbral mínimo de confianza
HIGH_CONFUSION_OBJECTS: Dict[str, float] = {
    # Baño: muy frecuentemente confundidos — umbral alto
    "inodoro": 0.70,        # confundido con silla (muy común)
    "bañera": 0.70,         # confundida con cama/sofá (muy común)
    "lavamanos": 0.65,      # confundido con mesa/mostrador
    # Muebles confundidos con objetos sobre ellos o similares
    "cama": 0.62,           # confundida con cobija/ropa de cama encima
    "mesa": 0.60,           # confundida con juego de mesa u objetos planos encima
    "sofá": 0.58,           # confundido con cojines/cobija encima
    "escritorio": 0.62,     # confundido con armario/closet vertical
    # Objetos de suelo: difíciles de distinguir de texturas
    "tapa de alcantarilla": 0.60,
    "reductor de velocidad": 0.60,
    "escalón": 0.60,        # confundido con sombra o desnivel del piso
    # Objetos portátiles: se confunden entre sí y con otros objetos
    "mochila": 0.55,        # confundida con bolso/caja
    "caja de cartón": 0.60, # confundida con contenedor/maleta/cajón
    "maleta": 0.55,
    # Electrodomésticos similares entre sí
    "secadora": 0.58,       # muy similar a lavadora
    # Estructuras que se confunden con paredes o fondos
    "andamio": 0.55,        # confundido con cerca/estructura metálica
    # Balcones/cornisas: críticos pero a veces confundidos con ventanas/paredes
    "balcón": 0.55,
    "cornisa": 0.60,
    "barandal de balcón": 0.55,
}


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
        relevant = self._filter_pedestrian_relevant(objects, img_width, img_height)

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
        # Los objetos al frente tienen prioridad sobre los laterales.
        # Un sofá muy cerca a la derecha no debe eclipsar una mesa al frente.
        def _sort_key(a):
            front_bonus = 0.3 if a["position"] == "frente a ti" else 0.0
            return -(a["risk_score"] + front_bonus)

        analyzed.sort(key=_sort_key)

        # --- PASO 4: Determinar estado del camino ---
        center_obstacles = [
            a for a in analyzed
            if a["position"] == "frente a ti"
            and a["proximity"] in ("muy_cerca", "cerca")
        ]
        # También considerar objetos en el centro clasificados como "lejos" pero con
        # peso de peligro alto: el modelo de profundidad puede subestimar distancias
        # para superficies planas (puertas cerradas, paredes, vidrios).
        # Se usa umbral de confianza más bajo (0.35) para objetos bloqueantes como
        # puertas cerradas, que YOLO no siempre detecta con alta confianza.
        lejos_front_high_conf = [
            a for a in analyzed
            if a["position"] == "frente a ti"
            and a["proximity"] == "lejos"
            and a["confidence"] >= (0.35 if DANGER_WEIGHT.get(a["name_es"], 0) >= 0.55 else 0.45)
            and DANGER_WEIGHT.get(a["name_es"], 0) >= 0.55
        ]
        path_clear = len(center_obstacles) == 0 and len(lejos_front_high_conf) == 0

        # --- PASO 5: Determinar prioridad global ---
        max_score = analyzed[0]["risk_score"] if analyzed else 0
        priority = self._score_to_priority(max_score)
        has_danger = priority in ("critical", "high")

        # --- PASO 5b: Determinar tipo de alerta visual ---
        # "peligro"  → objeto intrínsecamente peligroso (vehículo, escalera, balcón,
        #              estufa...) sin importar posición ni distancia.
        # "atencion" → obstáculo FRENTE al usuario que puede impedir su paso
        #              (cerca/muy_cerca o acercándose), pero no es clase peligrosa.
        #              Objetos laterales NUNCA activan "atencion" — no bloquean el paso.
        # None       → camino libre o solo objetos lejanos/laterales sin riesgo.
        alert_type: Optional[str] = None
        if analyzed:
            has_peligro_class = any(
                a["name_es"] in PELIGRO_CLASSES for a in analyzed
            )
            # Solo obstáculos frontales CERCANOS cuentan para "atencion"
            # (no activar para objetos lejanos aunque bloqueen técnicamente el path)
            has_front_blocking = any(
                a for a in analyzed
                if a["position"] == "frente a ti"
                and (
                    a["proximity"] in ("muy_cerca", "cerca")
                    or a["movement"] == "acercandose"
                )
            )
            if has_peligro_class:
                alert_type = "peligro"
            elif has_front_blocking:
                alert_type = "atencion"

        # --- PASO 6: Generar instrucciones ---
        instruction = self._generate_instructions(analyzed, path_clear)

        return {
            "instruction": instruction,
            "obstacles": [a["object"] for a in analyzed if a["proximity"] != "lejos"],
            "path_clear": path_clear,
            "has_danger": has_danger,
            "priority": priority,
            "alert_type": alert_type,
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
        self, objects: List[DetectedObject], img_width: int, img_height: int
    ) -> List[DetectedObject]:
        """
        Filtra objetos relevantes para movilidad peatonal aplicando tres criterios:
        1. Clase relevante (lista blanca de obstáculos peatonales)
        2. Confianza mínima (más permisivo para objetos críticos como carros/personas)
        3. Área mínima del bounding box (descarta detecciones diminutas = ruido)
        """
        img_area = img_width * img_height
        filtered = []

        for obj in objects:
            # Criterio 0: lista negra — nunca reportar estos objetos
            if obj.name_es in IGNORE_CLASSES:
                logger.debug(f"[Nav/Filter] '{obj.name_es}' en lista negra → ignorado")
                continue

            # Criterio 1: clase relevante para caminata
            if obj.name_es not in PEDESTRIAN_RELEVANT_CLASSES:
                continue

            # Criterio 2: confianza mínima según criticidad del objeto
            if obj.name_es in CRITICAL_OBJECTS:
                min_conf = MIN_CONFIDENCE_CRITICAL
            elif obj.name_es in HIGH_CONFUSION_OBJECTS:
                min_conf = HIGH_CONFUSION_OBJECTS[obj.name_es]
            else:
                min_conf = MIN_CONFIDENCE_GENERAL

            if obj.confidence < min_conf:
                logger.debug(
                    f"[Nav/Filter] '{obj.name_es}' descartado por confianza baja "
                    f"({obj.confidence:.2f} < {min_conf})"
                )
                continue

            # Criterio 3: área mínima del bounding box
            if obj.bounding_box and img_area > 0:
                bbox = obj.bounding_box
                bbox_area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
                if bbox_area / img_area < MIN_BBOX_AREA_FRACTION:
                    logger.debug(
                        f"[Nav/Filter] '{obj.name_es}' descartado por bbox diminuto "
                        f"({bbox_area/img_area*100:.2f}% < {MIN_BBOX_AREA_FRACTION*100}%)"
                    )
                    continue

            # Criterio 4: zona de altura — ignorar objetos en la parte superior de la
            # imagen (cortinas, lámparas, elementos colgantes) a menos que sean
            # clases de peligro crítico (balcón, escaleras, etc.) que pueden aparecer
            # legítimamente arriba.
            if obj.bounding_box and img_height > 0:
                center_y = (obj.bounding_box.y_min + obj.bounding_box.y_max) / 2
                top_ratio = center_y / img_height
                if top_ratio < _TOP_ZONE_THRESHOLD and obj.name_es not in PELIGRO_CLASSES:
                    logger.debug(
                        f"[Nav/Filter] '{obj.name_es}' descartado por zona superior "
                        f"(center_y={top_ratio:.2f} < {_TOP_ZONE_THRESHOLD})"
                    )
                    continue

            filtered.append(obj)

        return filtered

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

        if ratio < 0.28:
            return "a tu izquierda"
        elif ratio > 0.72:
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
        Genera instrucción combinada frente + lateral en una sola frase.

        Formato:
          - Obstáculo al frente + lateral notable → "Atención: silla al frente, cama a la izquierda."
          - Solo frente bloqueado                → "Atención: silla muy cerca al frente."
          - Frente libre + lateral notable        → "Camino libre al frente, cama a la izquierda."
          - Nada relevante                        → "Camino libre."
        """
        if not analyzed:
            return "Camino libre."

        # Filtrar obstáculos que merecen mención
        mentionable = [
            a for a in analyzed
            if a["proximity"] in ("muy_cerca", "cerca")
            or a["movement"] == "acercandose"
            or (
                a["proximity"] == "lejos"
                and a["position"] == "frente a ti"
                and a["confidence"] >= 0.45
                and DANGER_WEIGHT.get(a["name_es"], 0) >= 0.55
            )
        ]

        if not mentionable:
            return "Camino libre."

        # Separar frente y laterales
        front_obs = [a for a in mentionable if a["position"] == "frente a ti"]
        lateral_obs = [
            a for a in mentionable
            if a["position"] != "frente a ti"
            and (a["proximity"] in ("muy_cerca", "cerca") or a["movement"] == "acercandose")
        ]

        # --- Parte del frente ---
        if front_obs:
            top = front_obs[0]
            # Objetos lejanos pero peligrosos (escaleras, balcón, etc.)
            if top["proximity"] == "lejos":
                name_cap = top["name_es"].capitalize()
                if top["name_es"] in PELIGRO_CLASSES:
                    front_phrase = f"¡Cuidado! {name_cap} al frente"
                else:
                    front_phrase = f"Atención: {name_cap} al frente"
            else:
                front_phrase = self._build_phrase(
                    top["name_es"], top["position"], top["proximity"],
                    top["movement"], top["height_zone"], top["risk_score"],
                )
            # Quitar punto final si lo tuviera (lo añadimos al combinar)
            front_phrase = front_phrase.rstrip(".")
        else:
            front_phrase = "Camino libre al frente"

        # --- Parte lateral (máximo 1 objeto) ---
        lateral_phrase = ""
        if lateral_obs:
            best = lateral_obs[0]
            lat = self._build_phrase(
                best["name_es"], best["position"], best["proximity"],
                best["movement"], best["height_zone"], best["risk_score"],
            ).rstrip(".")
            if lat:
                # Lateral siempre en minúscula al combinarse
                lateral_phrase = lat[0].lower() + lat[1:]

        # --- Combinar ---
        # Si el frente está libre: lateral primero, luego confirmación del frente.
        #   "Cama a tu derecha, el camino al frente está libre."
        # Si hay obstáculo al frente: el aviso del frente va primero (es lo prioritario).
        #   "Atención: silla muy cerca al frente, cama a tu izquierda."
        if front_obs and lateral_phrase:
            return f"{front_phrase}, {lateral_phrase}."
        elif not front_obs and lateral_phrase:
            lat_cap = lateral_phrase[0].upper() + lateral_phrase[1:]
            return f"{lat_cap}, el camino al frente está libre."
        elif front_phrase:
            return front_phrase + "."
        return "Camino libre."

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
        is_front = position == "frente a ti"

        # ── Objetos que merecen "¡Cuidado!" — usa PELIGRO_CLASSES (fuente única) ──
        _CUIDADO_CLASSES = PELIGRO_CLASSES

        # ── Objetos que merecen "Precaución" (solo al frente) ────────────────
        _PRECAUCION_CLASSES = {
            "persona", "niño", "bebé", "persona en silla de ruedas",
            "perro", "caballo", "vaca", "cerdo", "burro",
            "mesa", "silla", "sofá", "cama", "escritorio",
            "mesa de comedor", "mostrador", "banco",
            "carrito de compras", "cochecito de bebé", "silla de ruedas",
            "pared", "pared de ladrillo", "pared de vidrio", "pilar", "columna",
            "andamio", "barrera vial", "contenedor de basura",
            "estufa", "horno", "refrigerador",
        }

        # Objeto acercándose: alerta solo si viene al frente
        if movement == "acercandose":
            if is_front:
                if name in _CUIDADO_CLASSES:
                    return f"¡Cuidado! {name_cap} se acerca {position}"
                return f"Atención: {name_cap} se acerca {position}"
            # Lateral acercándose: informativo sin prefijo
            return f"{name_cap} se acerca {position}"

        # Objeto muy cerca
        if proximity == "muy_cerca":
            height_info = ""
            if height == "suelo":
                height_info = " a nivel del suelo"
            elif height == "cabeza":
                height_info = " a nivel de la cabeza"

            if is_front:
                # Al frente: usar prefijo según peligro
                if name in _CUIDADO_CLASSES:
                    return f"¡Cuidado! {name_cap} {prox_label} {position}{height_info}"
                if name in _PRECAUCION_CLASSES:
                    return f"Atención: {name_cap} {prox_label} {position}{height_info}"
                return f"{name_cap} {prox_label} {position}{height_info}"
            else:
                # Lateral muy cerca: solo informativo, sin prefijo
                return f"{name_cap} {prox_label} {position}{height_info}"

        # Objeto a unos metros: solo informativo sin prefijo (frente o lateral)
        if proximity == "cerca":
            return f"{name_cap} {prox_label} {position}"

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
            "alert_type": None,
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
