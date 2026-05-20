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
    # ── OBSTÁCULO DESCONOCIDO (depth-only) ────────────────────────────────────
    # Inyectado cuando YOLO no clasifica nada pero el depth muestra masa cerca.
    # Captura paredes, árboles, postes, obstáculos no entrenados.
    "obstáculo",

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
    "hueco", "agujero",         # huecos en el piso
    "reductor de velocidad",    # resalto en la vía
    # NOTA: maleta, mochila, patineta, caja de cartón se eliminaron — YOLO los
    # confunde con texturas/sombras y generan demasiados falsos positivos.

    # ── OBSTÁCULOS A NIVEL CORPORAL (colisión frontal) ────────────────────────
    # Muebles principales (suficientes — no necesitamos cada variante)
    "silla", "mesa", "sofá", "cama", "escritorio",
    "banco", "mostrador", "mesa de comedor",
    "armario",
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
    "letrero", "señal",         # señalización útil para orientarse
    "señal de pare", "señal de tráfico",

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
    # ── VENTANAS Y SUPERFICIES ALTAS ─────────────────────────────────────────
    "ventana",                  # superficie de vidrio, peligro de colisión
    "espejo",                   # superficie reflectante grande: colisión oculta
    "ventilador",               # peligro a altura de cabeza, riesgo de golpe

    # ── COCINA / LAVANDERÍA (interior) ───────────────────────────────────────
    "estufa", "horno",
    "refrigerador", "lavadora", "secadora",

    # ── BAÑO (navegación interior) ────────────────────────────────────────────
    "inodoro", "lavamanos", "bañera",

    # ── EQUIPAMIENTO MÉDICO / GYM ─────────────────────────────────────────────
    "camilla",      # camilla hospitalaria con ruedas, bloquea el paso
    "caminadora",   # cinta de correr de gym, grande y puede estar en movimiento
}


# ============================================================================
# PESO SEMÁNTICO POR TIPO DE OBJETO
# ============================================================================
# Cuánto "peligro base" representa cada tipo de objeto para un peatón ciego.
# Escala 0.0 - 1.0 donde 1.0 = máximo peligro.

DANGER_WEIGHT: Dict[str, float] = {
    # ── OBSTÁCULO DESCONOCIDO (detectado por depth, no por YOLO) ────────────
    # Peso alto: si depth dice que algo grande está cerca, hay que detenerse
    # aunque no sepamos qué es.
    "obstáculo": 0.85,

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
    "hueco": 0.90, "agujero": 0.90, # huecos en el piso, peligro de caída

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
    "armario": 0.65,                 # mueble grande que bloquea el paso
    "espejo": 0.65,                  # superficie reflectante: colisión oculta
    "puerta de vidrio": 0.75,       # invisible
    "puerta giratoria": 0.65,       # mecanismo en movimiento
    "puerta cerrada": 0.65,         # obstáculo directo — subido de 0.55
    "puerta": 0.50, "puerta corrediza": 0.50,
    "puerta abierta": 0.35,         # menos peligro pero delimita el espacio
    "reja": 0.60, "ascensor": 0.40,
    "ventana": 0.70,                # vidrio invisible, colisión

    # ── ALTURA DE CABEZA: golpes de cabeza ────────────────────────────────────
    "ventilador": 0.65,             # aspas en movimiento, riesgo de golpe

    # ── SEÑALIZACIÓN: orientación e información ───────────────────────────────
    "letrero": 0.30, "señal": 0.30,
    "señal de pare": 0.50, "señal de tráfico": 0.40,

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

    # ── EQUIPAMIENTO MÉDICO / GYM ────────────────────────────────────────────
    "camilla": 0.62,      # objeto grande con ruedas, bloquea el paso
    "caminadora": 0.65,   # máquina grande, puede estar en movimiento
}

# Peso por defecto para objetos relevantes sin peso específico
_DEFAULT_DANGER_WEIGHT = 0.4

# Objetos intrínsecamente peligrosos → alert_type = "peligro" sin importar distancia.
# Incluye: vehículos, riesgos de caída, peligros térmicos, superficies invisibles.
PELIGRO_CLASSES: set = {
    # Obstáculo desconocido detectado por depth → se trata como peligro
    "obstáculo",
    # Vehículos
    "carro", "autobús", "camión", "taxi", "ambulancia", "patrulla",
    "motocicleta", "bicicleta", "scooter eléctrico",
    # Animales que pueden golpear o asustar
    "caballo", "vaca", "burro", "cerdo",
    # Caídas
    "escaleras", "escalera mecánica", "escalón", "balcón", "cornisa",
    "borde de acera", "tapa de alcantarilla", "barandal de balcón",
    "hueco", "agujero",
    # Peligros térmicos
    "estufa", "horno",
    # Superficies invisibles / peligro de colisión oculta
    "pared de vidrio", "puerta de vidrio", "ventana",
    # Altura de cabeza con piezas en movimiento
    "ventilador",
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
    # "espejo" movido a PEDESTRIAN_RELEVANT_CLASSES — superficie reflectante
    # grande puede no identificarse como vidrio y causar colisión.
    "espejo de baño", "espejo compacto",
    "cortina", "cortina de baño", "persiana",
    # NOTA: "ventana" se quitó de la lista negra. Es vidrio = peligro de colisión.
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
    "televisión", "televisor",   # ambas formas; "televisor" viene de prompts de navegación
    "control remoto", "tablet", "cámara",
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
# Objetos que ocupan menos del 1% de la imagen son probablemente ruido o
# están demasiado lejos para preocupar al usuario peatón.
# Objetos críticos (carros, personas) tienen su propia constante más permisiva.
MIN_BBOX_AREA_FRACTION = 0.01
# Para objetos críticos (vehículos, personas, animales grandes) basta con 0.4%
# porque importan aunque estén lejos.
MIN_BBOX_AREA_FRACTION_CRITICAL = 0.004

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
    "caja de pañuelos": 0.60, # confundida con cualquier objeto cúbico
    "maleta": 0.55,
    # Electrodomésticos similares entre sí
    "secadora": 0.58,       # muy similar a lavadora
    # Estructuras que se confunden con paredes o fondos
    "andamio": 0.55,        # confundido con cerca/estructura metálica
    # Balcones/cornisas: críticos pero a veces confundidos con ventanas/paredes
    "balcón": 0.55,
    "cornisa": 0.60,
    "barandal de balcón": 0.55,
    # Animales similares entre sí
    "gato": 0.55,           # confundido con perro pequeño y peluches
    "perro": 0.50,           # confundido con gato grande
    "cachorro": 0.55,
    "gatito": 0.55,
    # Superficies reflectantes: confundidas con TV apagado o pared oscura
    "espejo": 0.70,          # requiere alta confianza para no confundir con televisor
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
        # Estabilidad por objeto: cuántos frames consecutivos lleva el mismo objeto
        # en posición/proximidad iguales. Usado para suprimir alertas repetidas
        # cuando el usuario está parado frente al mismo obstáculo.
        # Clave: name_es, Valor: (position, proximity, frames_count)
        self._stable_state: Dict[str, Tuple[str, str, int]] = {}
        # Fingerprint de la escena: conjunto de (nombre, proximidad) ordenados.
        # Si el fingerprint no cambia varios frames, el usuario está parado y
        # ya escuchó la advertencia — supresión completa de TTS.
        self._last_scene_fingerprint: str = ""
        self._scene_unchanged_frames: int = 0

    # ====================================================================
    # PIPELINE PRINCIPAL
    # ====================================================================

    def generate_guidance(
        self,
        objects: List[DetectedObject],
        img_width: int,
        img_height: int,
        track_movement: bool = True,
        directional_depth: Optional[Dict[str, float]] = None,
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

        # --- PASO 5a: Detectar estabilidad de la escena (modelo Waze) ---
        # Fingerprint: nombre + posición horizontal + zona gruesa (cerca vs lejos).
        # Se usa zona gruesa ("cerca" agrupa muy_cerca y cerca) porque el sensor
        # de profundidad fluctúa entre esas dos zonas frame a frame aunque el
        # usuario esté parado — con proximidad exacta el fingerprint nunca
        # estabilizaba y el cooldown largo de 15s nunca aplicaba.
        mentionable_for_fingerprint = [
            a for a in analyzed
            if a["proximity"] in ("muy_cerca", "cerca")
            or a["movement"] == "acercandose"
        ]
        fingerprint_items = sorted(
            f"{a['name_es']}|{a['position']}|{'cerca' if a['proximity'] in ('muy_cerca', 'cerca') else 'lejos'}"
            for a in mentionable_for_fingerprint
        )
        fingerprint = "::".join(fingerprint_items)

        if fingerprint == self._last_scene_fingerprint and fingerprint:
            self._scene_unchanged_frames += 1
        else:
            self._scene_unchanged_frames = 0
            self._last_scene_fingerprint = fingerprint

        # 5 frames a ~3fps ≈ ~1.7 segundos de inmovilidad para confirmar escena estable
        scene_stable = self._scene_unchanged_frames >= 5

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
        instruction = self._generate_instructions(analyzed, path_clear, directional_depth)

        return {
            "instruction": instruction,
            "obstacles": [a["object"] for a in analyzed if a["proximity"] != "lejos"],
            "path_clear": path_clear,
            "has_danger": has_danger,
            "priority": priority,
            "alert_type": alert_type,
            "scene_stable": scene_stable,
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

            # Criterio 0.5: descartar reflejos del usuario.
            # Si una "persona" cubre >45% del frame y está centrada, casi seguro
            # es el reflejo del usuario en un espejo o la propia mano/cuerpo
            # frente al lente. Avisar de "persona muy cerca" en estos casos
            # confunde al usuario sobre obstáculos reales.
            if (
                obj.name_es in {"persona", "niño"}
                and obj.bounding_box
                and img_area > 0
            ):
                bbox = obj.bounding_box
                bbox_area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
                area_ratio = bbox_area / img_area
                cx = (bbox.x_min + bbox.x_max) / 2
                centered = abs(cx / img_width - 0.5) < 0.2 if img_width > 0 else False
                if area_ratio > 0.45 and centered:
                    logger.debug(
                        f"[Nav/Filter] '{obj.name_es}' descartado: probable reflejo "
                        f"(área={area_ratio:.0%}, centrada)"
                    )
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

            # Criterio 3: área mínima del bounding box.
            # Críticos (vehículos, personas, animales grandes) usan umbral más
            # bajo porque importan aunque estén lejos. El resto debe ocupar al
            # menos 1% del frame para evitar reportar texturas/sombras como
            # obstáculos diminutos (caja de cartón, mochila confundidas).
            if obj.bounding_box and img_area > 0:
                bbox = obj.bounding_box
                bbox_area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
                min_area = (
                    MIN_BBOX_AREA_FRACTION_CRITICAL
                    if obj.name_es in CRITICAL_OBJECTS
                    else MIN_BBOX_AREA_FRACTION
                )
                if bbox_area / img_area < min_area:
                    logger.debug(
                        f"[Nav/Filter] '{obj.name_es}' descartado por bbox diminuto "
                        f"({bbox_area/img_area*100:.2f}% < {min_area*100}%)"
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

        Cono de atención estrecho: solo el 30% central (35%-65%) cuenta
        como "frente a ti". Antes era 28%-72% (44% del ancho), lo que
        provocaba alertas innecesarias por objetos casi laterales que
        en realidad no estaban en el camino del usuario.
        """
        center_x = (bbox.x_min + bbox.x_max) / 2
        ratio = center_x / img_width if img_width > 0 else 0.5

        if ratio < 0.35:
            return "a tu izquierda"
        elif ratio > 0.65:
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
        directional_depth: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Genera instrucción tipo Waze: identifica varios obstáculos pero
        siempre termina con UNA sugerencia de acción.

        Formato:
          - Frente bloqueado + lado libre claro:
              "Silla y mesa al frente. Muévete a la izquierda."
          - Frente bloqueado sin escapatoria (ambos lados ocupados):
              "Silla y mesa al frente. Detente."
          - Solo laterales (frente libre):
              "Mesa a tu derecha. Al frente está libre."
          - Peligro intrínseco al frente (carro, escaleras, vidrio):
              "¡Cuidado! Carro al frente. Detente."
          - Nada relevante:
              "Camino libre."
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
            # Hay objetos pero todos están lejos y sin amenaza inmediata.
            # Devolver vacío: la UI visual ya los muestra; decir "Camino libre."
            # mientras el usuario ve objetos detectados confunde.
            # La reaseguración periódica ("Camino libre.") la maneja el móvil.
            return ""

        # Separar por posición
        front_obs = [a for a in mentionable if a["position"] == "frente a ti"]
        left_obs = [a for a in mentionable if a["position"] == "a tu izquierda"]
        right_obs = [a for a in mentionable if a["position"] == "a tu derecha"]

        # --- Decidir lado libre ---
        free_side = None
        if directional_depth:
            left_load = directional_depth.get("izquierda", 0.5)
            right_load = directional_depth.get("derecha", 0.5)
            if abs(left_load - right_load) >= 0.15:
                free_side = "izquierda" if left_load < right_load else "derecha"

        if free_side is None:
            left_blocked = any(a["proximity"] in ("muy_cerca", "cerca") for a in left_obs)
            right_blocked = any(a["proximity"] in ("muy_cerca", "cerca") for a in right_obs)
            if left_blocked and not right_blocked:
                free_side = "derecha"
            elif right_blocked and not left_blocked:
                free_side = "izquierda"
            elif not left_blocked and not right_blocked:
                # Ningún lateral bloqueado: cualquiera sirve, prefiere derecha
                free_side = "derecha"

        PERSON_NAMES = {"persona", "niño", "bebé", "persona en silla de ruedas"}

        # --- Caso 1: hay obstáculo(s) al frente ---
        if front_obs:
            # Separar personas de obstáculos físicos
            front_persons = [a for a in front_obs if a["name_es"] in PERSON_NAMES]
            front_things  = [a for a in front_obs if a["name_es"] not in PERSON_NAMES]

            # Si solo hay personas al frente: aviso informativo, sin dirección.
            # Las personas se mueven solas — no tiene sentido decir "muévete a la derecha".
            if front_persons and not front_things:
                count = len(front_persons)
                if count == 1:
                    phrase = "Hay una persona al frente."
                elif count == 2:
                    phrase = "Hay dos personas al frente."
                elif count == 3:
                    phrase = "Hay tres personas al frente."
                else:
                    phrase = f"Hay {count} personas al frente."
                return phrase

            # Si hay obstáculos físicos (con o sin personas), usar el obstáculo top.
            top_front = front_things[0] if front_things else front_obs[0]
            obj_phrase = top_front["name_es"].capitalize()

            # ¿Hay clase intrínseca de peligro al frente?
            has_danger = top_front["name_es"] in PELIGRO_CLASSES
            prefix = "¡Cuidado! " if has_danger else ""

            # Decidir acción
            if free_side:
                action = f"Muévete a la {free_side}"
            else:
                action = "Detente"

            return f"{prefix}{obj_phrase} al frente. {action}."

        # --- Caso 2: frente libre, hay laterales notables ---
        # Mencionar el lateral más cercano y confirmar frente libre
        all_laterals = sorted(
            left_obs + right_obs,
            key=lambda a: 0 if a["proximity"] == "muy_cerca" else 1
        )
        if all_laterals:
            lat = all_laterals[0]
            lat_phrase = self._build_phrase(
                lat["name_es"], lat["position"], lat["proximity"],
                lat["movement"], lat["height_zone"], lat["risk_score"],
            ).rstrip(".")
            return f"{lat_phrase}. Al frente está libre."

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
        Construye una frase individual de alerta — versión concisa.

        Diseño: en navegación cada palabra cuenta. Frases largas como
        "Atención: Cama muy cerca frente a ti a nivel del suelo." (12 palabras)
        se sustituyen por "Cama al frente." (3 palabras). El nivel de peligro
        ya viaja en `priority` y la móvil lo refleja con vibración / cadencia.

        Solo se mantiene "¡Cuidado!" para PELIGRO_CLASSES intrínsecas
        (vehículos, escaleras, balcones, vidrio…) porque sí necesitan
        prefijo de alarma para reaccionar inmediatamente.

        Altura solo se menciona cuando es relevante para no chocar
        ("a nivel de la cabeza" sí, "a nivel del suelo" no — es lo normal
        de un mueble).
        """
        name_cap = name.capitalize()
        is_front = position == "frente a ti"
        is_danger_class = name in PELIGRO_CLASSES

        # Posición concisa: "frente a ti" → "al frente"
        pos_short = "al frente" if is_front else position

        # Altura solo se menciona si es a nivel de cabeza (peligro de golpe)
        height_info = " a nivel de la cabeza" if height == "cabeza" else ""

        # Objeto acercándose: caso más urgente
        if movement == "acercandose":
            if is_danger_class and is_front:
                return f"¡Cuidado! {name_cap} se acerca."
            if is_front:
                return f"{name_cap} se acerca."
            return f"{name_cap} se acerca {position}."

        # Objeto muy cerca
        if proximity == "muy_cerca":
            if is_danger_class and is_front:
                return f"¡Cuidado! {name_cap} {pos_short}{height_info}."
            if is_front:
                return f"{name_cap} {pos_short}{height_info}."
            # Lateral muy cerca: informativo, sin prefijo
            return f"{name_cap} {position}{height_info}."

        # Objeto a unos metros: informativo, frase corta
        if proximity == "cerca":
            if is_front:
                return f"{name_cap} {pos_short}."
            return f"{name_cap} {position}."

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
            "scene_stable": False,
            "obstacle_details": [],
        }

    def reset_movement_state(self) -> None:
        """Reinicia el estado de movimiento (nueva sesión o reset)."""
        self._previous_bboxes.clear()
        self._stable_state.clear()
        self._last_scene_fingerprint = ""
        self._scene_unchanged_frames = 0
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
