"""
============================================================================
NAVIA Backend - Servicio de Detección de Objetos (YOLO-World)
============================================================================
Este módulo implementa detección de objetos usando YOLO-World, un detector
open-vocabulary que puede identificar cualquier objeto especificado por texto.

¿Por qué YOLO-World en vez de YOLOv8 estándar?
- YOLOv8 estándar solo detecta 80 clases del dataset COCO
- YOLO-World puede detectar CUALQUIER objeto que se le indique
- Ideal para asistencia visual: lámparas, juguetes, zapatos, espejos, etc.
- Misma librería ultralytics, misma interfaz de resultados

¿Cómo funciona?
1. Se define una lista de objetos a detectar (set_classes)
2. YOLO-World codifica los nombres como embeddings de texto
3. Durante inferencia, compara regiones de la imagen contra esos embeddings
4. Retorna detecciones con la misma estructura que YOLO estándar
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


# ============================================================================
# Diccionario de clases: inglés → (español, género)
# género: "m" = masculino (un), "f" = femenino (una)
#
# Organizado por categorías relevantes para personas con discapacidad visual.
# YOLO-World detectará SOLO los objetos listados aquí.
# ============================================================================

WORLD_CLASSES_ES = {
    # ===================== PERSONAS =====================
    "person": ("persona", "f"),
    "human": ("persona", "f"),
    "man": ("persona", "f"),
    "woman": ("persona", "f"),
    "adult": ("persona", "f"),
    "standing person": ("persona", "f"),
    "sitting person": ("persona", "f"),
    "walking person": ("persona", "f"),
    "child": ("niño", "m"),
    "kid": ("niño", "m"),
    "baby": ("bebé", "m"),
    "infant": ("bebé", "m"),
    "wheelchair user": ("persona en silla de ruedas", "f"),

    # ===================== ANIMALES =====================
    # Perro (múltiples variantes para mejor detección)
    "dog": ("perro", "m"),
    "puppy": ("cachorro", "m"),
    "guide dog": ("perro guía", "m"),
    "service dog": ("perro de servicio", "m"),
    # Gato
    "cat": ("gato", "m"),
    "kitten": ("gatito", "m"),
    # Pájaros
    "bird": ("pájaro", "m"),
    "parrot": ("loro", "m"),
    "canary": ("canario", "m"),
    "chicken": ("gallina", "f"),
    "rooster": ("gallo", "m"),
    "duck": ("pato", "m"),
    "pigeon": ("paloma", "f"),
    "swan": ("cisne", "m"),
    "duck": ("pato", "m"),
    # Animales de carga
    "horse": ("caballo", "m"),
    "cow": ("vaca", "f"),
    "pig": ("cerdo", "m"),
    "sheep": ("oveja", "f"),
    "goat": ("cabra", "f"),
    "donkey": ("burro", "m"),
    # Mascotas pequeñas
    "fish in tank": ("pez", "m"),
    "rabbit": ("conejo", "m"),
    "hamster": ("hámster", "m"),
    "turtle": ("tortuga", "f"),
    "guinea pig": ("cuy", "m"),
    # Insectos y arácnidos
    "insect": ("insecto", "m"),
    "spider": ("araña", "f"),
    "butterfly": ("mariposa", "f"),
    "bee": ("abeja", "f"),
    "wasp": ("avispa", "f"),
    "ant": ("hormiga", "f"),
    "cockroach": ("cucaracha", "f"),
    "mosquito": ("mosquito", "m"),
    "fly": ("mosca", "f"),
    "dragonfly": ("libélula", "f"),
    "grasshopper": ("saltamontes", "m"),
    "cricket": ("grillo", "m"),
    # Otros animales salvajes
    "deer": ("ciervo", "m"),
    "squirrel": ("ardilla", "f"),
    "raccoon": ("mapache", "m"),
    "fox": ("zorro", "m"),
    "rabbit": ("conejo", "m"),
    "bear": ("oso", "m"),
    "lion": ("león", "m"),
    "tiger": ("tigre", "m"),
    "monkey": ("mono", "m"),
    "snake": ("serpiente", "f"),
    "frog": ("rana", "f"),

    # ===================== MUEBLES =====================
    "chair with four legs": ("silla", "f"),
    "wooden table": ("mesa", "f"),
    "office desk": ("escritorio", "m"),
    "sofa couch with cushions": ("sofá", "m"),
    "upholstered armchair": ("sillón", "m"),
    "bed with mattress": ("cama", "f"),
    "bunk bed": ("litera", "f"),
    "crib": ("cuna", "f"),
    "bookshelf": ("estante", "m"),
    "wooden cabinet": ("gabinete", "m"),
    "chest of drawers": ("cómoda", "f"),
    "wardrobe closet": ("armario", "m"),
    "nightstand": ("mesa de noche", "f"),
    "bar stool with legs": ("taburete", "m"),
    "park bench outdoor seat": ("banco", "m"),
    "kitchen counter": ("mostrador", "m"),
    "coffee table": ("mesa de centro", "f"),
    "dining table": ("mesa de comedor", "f"),
    "rocking chair with curved legs": ("mecedora", "f"),
    "ottoman footrest": ("puf", "m"),
    "tv stand": ("mueble de televisor", "m"),

    # ===================== MUROS, PAREDES Y ESTRUCTURAS =====================
    "concrete wall": ("pared", "f"),
    "brick wall": ("pared de ladrillo", "f"),
    "glass wall": ("pared de vidrio", "f"),
    "stone wall": ("muro de piedra", "m"),
    "pillar": ("pilar", "m"),
    "concrete column": ("columna", "f"),

    # ===================== BALCONES Y TERRAZAS =====================
    "balcony": ("balcón", "m"),
    "balcony railing": ("barandal de balcón", "m"),
    "terrace": ("terraza", "f"),
    "ledge": ("cornisa", "f"),
    "railing": ("barandal", "m"),

    # ===================== PUERTAS ESPECÍFICAS =====================
    # Prompts genéricos: mejor recall que prompts hiper-específicos
    "door open wide": ("puerta abierta", "f"),
    "door closed shut blocking passage": ("puerta cerrada", "f"),
    "closed door flat surface": ("puerta cerrada", "f"),
    "glass door transparent": ("puerta de vidrio", "f"),
    "revolving door": ("puerta giratoria", "f"),
    "metal gate fence": ("reja", "f"),
    "garage door large": ("puerta de garaje", "f"),
    "emergency exit door": ("salida de emergencia", "f"),
    "doorway opening": ("marco de puerta", "m"),

    # ===================== HOGAR / NAVEGACIÓN INTERIOR =====================
    # Prompts específicos para evitar confusiones entre objetos similares
    "table lamp": ("lámpara de mesa", "f"),
    "floor lamp": ("lámpara de pie", "f"),
    "ceiling light fixture": ("luz de techo", "f"),
    "chandelier": ("candelabro", "m"),
    "wall mirror": ("espejo", "m"),         # "wall mirror" no se confunde con ventana
    "window with glass": ("ventana", "f"),   # "window with glass" no se confunde con espejo
    "door panel": ("puerta", "f"),           # más genérico que "wooden door"
    "sliding door panel": ("puerta corrediza", "f"),
    "curtain on window": ("cortina", "f"),   # "curtain on window" evita confusión con toalla
    "window blind": ("persiana", "f"),
    "ceiling fan": ("ventilador de techo", "m"),
    "standing fan": ("ventilador", "m"),
    "pedestal fan": ("ventilador", "m"),
    "electric fan": ("ventilador", "m"),
    "oscillating fan": ("ventilador", "m"),
    "tower fan": ("ventilador de torre", "m"),
    "desk fan": ("ventilador de escritorio", "m"),
    "air conditioner unit": ("aire acondicionado", "m"),
    "radiator heater": ("calentador", "m"),
    "area rug on floor": ("alfombra", "f"),  # "on floor" lo diferencia de toalla
    "doormat": ("tapete", "m"),
    "trash can with lid": ("papelera", "f"),
    "recycling bin": ("contenedor de reciclaje", "m"),
    "wicker basket": ("cesta", "f"),
    "cardboard box": ("caja de cartón", "f"),
    "storage container": ("contenedor", "m"),
    "bed pillow": ("almohada", "f"),
    "pillow": ("almohada", "f"),
    "sleeping pillow": ("almohada", "f"),
    "decorative pillow": ("almohada", "f"),
    "sofa cushion": ("cojín", "m"),
    "blanket on bed": ("cobija", "f"),       # "on bed" evita confusión con toalla
    "bedsheet": ("sábana", "f"),
    "candle": ("vela", "f"),
    "framed picture on wall": ("cuadro", "m"),
    "painting on wall": ("pintura", "f"),
    "wall clock": ("reloj de pared", "m"),
    "alarm clock on table": ("despertador", "m"),
    "flower vase": ("jarrón", "m"),
    "potted plant": ("planta en maceta", "f"),
    "flower bouquet": ("ramo de flores", "m"),
    "broom": ("escoba", "f"),
    "mop": ("trapeador", "m"),
    "vacuum cleaner": ("aspiradora", "f"),
    "clothes hanger": ("gancho de ropa", "m"),
    "clothes rack": ("perchero", "m"),
    "ironing board": ("tabla de planchar", "f"),
    "iron appliance": ("plancha", "f"),
    "staircase": ("escaleras", "f"),
    "single step on floor": ("escalón", "m"),
    "metal construction scaffolding on building": ("andamio", "m"),
    "stair handrail": ("pasamanos", "m"),
    "elevator door": ("ascensor", "m"),
    "escalator": ("escalera mecánica", "f"),
    "light switch on wall": ("interruptor de luz", "m"),
    "electrical outlet": ("toma de corriente", "f"),
    "tissue box": ("caja de pañuelos", "f"),
    "paper towel roll": ("rollo de papel", "m"),
    "plastic bag": ("bolsa plástica", "f"),
    "fire alarm on wall": ("alarma de incendio", "f"),

    # ===================== COCINA =====================
    "dinner plate": ("plato", "m"),
    "drinking glass": ("vaso", "m"),
    "coffee cup": ("taza", "f"),
    "coffee mug": ("taza de café", "f"),
    "cooking pot": ("olla", "f"),
    "frying pan": ("sartén", "f"),
    "tea kettle": ("tetera", "f"),
    "kitchen blender": ("licuadora", "f"),
    "cutting board": ("tabla de cortar", "f"),
    "serving tray": ("bandeja", "f"),
    "glass jar": ("frasco", "m"),
    "tin can": ("lata", "f"),
    "water bottle": ("botella de agua", "f"),
    "wine bottle": ("botella de vino", "f"),
    "soup bowl": ("tazón", "m"),
    "dinner fork": ("tenedor", "m"),
    "kitchen knife": ("cuchillo", "m"),
    "eating spoon": ("cuchara", "f"),
    "napkin": ("servilleta", "f"),
    "salt shaker": ("salero", "m"),
    "coffee maker machine": ("cafetera", "f"),
    "food container": ("recipiente de comida", "m"),
    "plastic cup": ("vaso plástico", "m"),
    "wine glass": ("copa de vino", "f"),
    "thermos bottle": ("termo", "m"),
    "kitchen sink": ("fregadero", "m"),
    "dish sponge": ("esponja", "f"),
    "kitchen towel hanging": ("paño de cocina", "m"),  # "hanging" diferencia de toalla de baño

    # ===================== ELECTRÓNICA =====================
    "television screen": ("televisor", "m"),
    "remote control": ("control remoto", "m"),
    "cell phone": ("teléfono celular", "m"),
    "laptop computer": ("computadora portátil", "f"),
    "laptop back": ("computadora portátil", "f"),
    "closed laptop": ("computadora portátil", "f"),
    "notebook computer": ("computadora portátil", "f"),
    "desktop computer monitor": ("monitor de computadora", "m"),
    "computer tower": ("torre de computadora", "f"),
    "tablet device": ("tableta", "f"),
    "over ear headphones": ("audífonos", "m"),
    "wireless earbuds": ("auriculares", "m"),
    "bluetooth speaker": ("parlante", "m"),
    "phone charger cable": ("cargador", "m"),
    "computer keyboard": ("teclado", "m"),
    "computer mouse": ("ratón de computadora", "m"),
    "power strip with plugs": ("regleta", "f"),
    "wifi router": ("router", "m"),
    "printer machine": ("impresora", "f"),
    "camera": ("cámara", "f"),
    "usb flash drive": ("memoria USB", "f"),
    "power bank": ("batería portátil", "f"),
    "smart watch": ("reloj inteligente", "m"),
    "video game controller": ("control de videojuegos", "m"),

    # ===================== ROPA Y OBJETOS PERSONALES =====================
    "shoe": ("zapato", "m"),
    "sneaker": ("tenis", "m"),
    "high heel shoe": ("tacón", "m"),
    "boot": ("bota", "f"),
    "sandal": ("sandalia", "f"),
    "slipper": ("pantufla", "f"),
    "hat": ("sombrero", "m"),
    "baseball cap": ("gorra", "f"),
    "eyeglasses": ("lentes", "m"),
    "sunglasses": ("lentes de sol", "m"),
    "wrist watch": ("reloj de pulsera", "m"),
    "leather wallet": ("billetera", "f"),
    "metal key": ("llave", "f"),
    "key chain": ("llavero", "m"),
    "handbag": ("bolso", "m"),
    "shoulder bag": ("bolso", "m"),
    "tote bag": ("bolso", "m"),
    "backpack bag on floor or back": ("mochila", "f"),
    "backpack": ("mochila", "f"),
    "purse": ("cartera", "f"),
    "umbrella": ("paraguas", "m"),
    "travel suitcase": ("maleta", "f"),
    "jacket": ("chaqueta", "f"),
    "winter coat": ("abrigo", "m"),
    "scarf": ("bufanda", "f"),
    "gloves": ("guantes", "m"),
    "belt": ("cinturón", "m"),
    "necktie": ("corbata", "f"),
    "dress": ("vestido", "m"),
    "shirt on hanger": ("camisa", "f"),
    "shirt": ("camisa", "f"),
    "t-shirt": ("camiseta", "f"),
    "folded shirt": ("camisa", "f"),
    "pair of pants": ("pantalón", "m"),
    "skirt": ("falda", "f"),
    "hair brush": ("cepillo de pelo", "m"),
    "comb": ("peine", "m"),
    
    # ===================== MAQUILLAJE Y BELLEZA =====================
    "makeup bag": ("estuche de maquillaje", "m"),
    "perfume bottle": ("perfume", "m"),
    # "lipstick" eliminado: confunde los labios de cualquier persona con un labial
    "mascara tube": ("rímel", "m"),
    "eyeliner pencil": ("delineador", "m"),
    "eyeshadow palette": ("paleta de sombras", "f"),
    "blush compact": ("rubor", "m"),
    "foundation bottle": ("base de maquillaje", "f"),
    "makeup brush": ("brocha de maquillaje", "f"),
    "nail polish bottle": ("esmalte de uñas", "m"),
    "face cream jar": ("crema facial", "f"),
    "body lotion bottle": ("loción corporal", "f"),
    "sunscreen bottle": ("protector solar", "m"),
    "hair straightener": ("plancha de cabello", "f"),
    "curling iron": ("rizador de cabello", "m"),
    "hair clip": ("pinza de cabello", "f"),
    "hair tie": ("liga de cabello", "f"),
    "cotton pads": ("algodones", "m"),
    "q-tips cotton swabs": ("hisopos", "m"),
    "tweezers": ("pinzas", "f"),
    "eyelash curler": ("rizador de pestañas", "m"),
    "compact mirror": ("espejo compacto", "m"),
    
    # ===================== JOYERÍA Y ACCESORIOS =====================
    "jewelry box": ("joyero", "m"),
    "necklace": ("collar", "m"),
    "ring": ("anillo", "m"),
    "earrings": ("aretes", "m"),
    "bracelet": ("pulsera", "f"),
    "watch": ("reloj", "m"),
    "brooch": ("broche", "m"),
    "hair band": ("diadema", "f"),
    "tiara": ("tiara", "f"),
    "pendant": ("colgante", "m"),
    "anklet": ("tobillera", "f"),
    "cufflinks": ("gemelos", "m"),

    # ===================== VEHÍCULOS Y CALLE =====================
    "car": ("carro", "m"),
    "bus": ("autobús", "m"),
    "truck": ("camión", "m"),
    "motorcycle": ("motocicleta", "f"),
    "bicycle": ("bicicleta", "f"),
    "electric scooter": ("scooter eléctrico", "m"),
    "taxi cab": ("taxi", "m"),
    "ambulance vehicle": ("ambulancia", "f"),
    "police car": ("patrulla", "f"),
    "traffic light": ("semáforo", "m"),
    "stop sign": ("señal de pare", "f"),
    "crosswalk on road": ("paso peatonal", "m"),
    "street pole": ("poste", "m"),
    "metal fence": ("cerca", "f"),
    "entrance gate": ("portón", "m"),
    "tree": ("árbol", "m"),
    "bush": ("arbusto", "m"),
    "fire hydrant": ("hidrante", "m"),
    "parking meter": ("parquímetro", "m"),
    "street lamp post": ("farola", "f"),
    "traffic cone": ("cono de tráfico", "m"),
    "road barrier": ("barrera vial", "f"),
    "speed bump": ("reductor de velocidad", "m"),
    "manhole cover": ("tapa de alcantarilla", "f"),
    "sidewalk curb": ("borde de acera", "m"),
    "mailbox": ("buzón", "m"),
    "garbage dumpster": ("contenedor de basura", "m"),
    "shopping cart": ("carrito de compras", "m"),
    "baby stroller": ("cochecito de bebé", "m"),
    "wheelchair": ("silla de ruedas", "f"),
    "parking sign": ("señal de estacionamiento", "f"),
    "construction sign": ("señal de construcción", "f"),

    # ===================== COMIDA =====================
    "fruit": ("fruta", "f"),
    "banana": ("banana", "f"),
    "apple": ("manzana", "f"),
    "orange fruit": ("naranja", "f"),
    "sandwich": ("sándwich", "m"),
    "pizza": ("pizza", "f"),
    "birthday cake": ("pastel", "m"),
    "loaf of bread": ("pan", "m"),
    "rice on plate": ("arroz", "m"),
    "egg": ("huevo", "m"),
    "cheese": ("queso", "m"),
    "chicken meat": ("pollo", "m"),
    "salad bowl": ("ensalada", "f"),
    "candy": ("dulce", "m"),
    "ice cream": ("helado", "m"),
    "cereal box": ("caja de cereal", "f"),
    "milk carton": ("cartón de leche", "m"),
    "juice box": ("jugo", "m"),
    "snack bag": ("bolsa de snacks", "f"),
    "cookie": ("galleta", "f"),

    # ===================== BAÑO =====================
    "bathroom toilet with tank and seat": ("inodoro", "m"),
    "bathroom sink with faucet": ("lavamanos", "m"),
    "bathtub in bathroom": ("bañera", "f"),
    "shower head fixture on wall": ("ducha", "f"),
    "bar of soap": ("jabón", "m"),
    "toothbrush": ("cepillo de dientes", "m"),
    "toothpaste tube": ("pasta dental", "f"),
    "toilet paper roll": ("papel higiénico", "m"),
    "bath towel on rack": ("toalla de baño", "f"),
    "hand towel": ("toalla de mano", "f"),
    "shampoo bottle": ("champú", "m"),
    "hair dryer": ("secador de pelo", "m"),
    "bathroom weighing scale": ("báscula", "f"),
    "shower curtain": ("cortina de baño", "f"),
    "soap dispenser": ("dispensador de jabón", "m"),
    "bathroom mirror on wall": ("espejo de baño", "m"),

    # ===================== ELECTRODOMÉSTICOS =====================
    "refrigerator": ("refrigerador", "m"),
    "microwave oven": ("microondas", "m"),
    "kitchen oven": ("horno", "m"),
    "kitchen stove": ("estufa", "f"),
    "toaster": ("tostadora", "f"),
    "washing machine": ("lavadora", "f"),
    "clothes dryer machine": ("secadora", "f"),
    "dishwasher machine": ("lavavajillas", "m"),
    "water heater": ("calentador de agua", "m"),
    "air purifier": ("purificador de aire", "m"),

    # ===================== JUGUETES Y ENTRETENIMIENTO =====================
    "toy": ("juguete", "m"),
    "baby doll": ("muñeca", "f"),
    "teddy bear": ("oso de peluche", "m"),
    "stuffed animal toy": ("peluche", "m"),
    "stuffed animal": ("peluche", "m"),
    "plush toy": ("peluche", "m"),
    "rubber ball": ("pelota", "f"),
    "jigsaw puzzle": ("rompecabezas", "m"),
    "board game box": ("juego de mesa", "m"),
    "playing cards": ("cartas de juego", "f"),
    "building blocks": ("bloques de construcción", "m"),
    "action figure": ("figura de acción", "f"),
    "toy car": ("carrito de juguete", "m"),
    "bicycle helmet": ("casco de bicicleta", "m"),
    "skateboard": ("patineta", "f"),
    "jump rope": ("cuerda de saltar", "f"),
    "kite": ("cometa", "f"),
    "coloring book": ("libro de colorear", "m"),
    "crayon": ("crayón", "m"),

    # ===================== OFICINA Y ESCUELA =====================
    "pen": ("bolígrafo", "m"),
    "pencil": ("lápiz", "m"),
    "notebook": ("cuaderno", "m"),
    "sheet of paper": ("hoja de papel", "f"),
    "folder": ("carpeta", "f"),
    "stapler": ("grapadora", "f"),
    "scissors": ("tijeras", "f"),
    "ruler": ("regla", "f"),
    "calculator": ("calculadora", "f"),
    "whiteboard": ("pizarra", "f"),
    "book": ("libro", "m"),
    "magazine": ("revista", "f"),
    "newspaper": ("periódico", "m"),
    "envelope": ("sobre", "m"),
    "tape dispenser": ("dispensador de cinta", "m"),
    "desk organizer": ("organizador de escritorio", "m"),
    "globe model": ("globo terráqueo", "m"),
    "backpack bag": ("morral", "m"),

    # ===================== SEGURIDAD Y SALUD =====================
    "fire extinguisher": ("extintor", "m"),
    "smoke detector on ceiling": ("detector de humo", "m"),
    "exit sign": ("señal de salida", "f"),
    "first aid kit": ("botiquín", "m"),
    "medicine bottle": ("frasco de medicina", "m"),
    "face mask": ("mascarilla", "f"),
    "hand sanitizer bottle": ("gel antibacterial", "m"),
    "safety helmet": ("casco de seguridad", "m"),
    "safety goggles": ("gafas de seguridad", "f"),
    "warning sign": ("señal de advertencia", "f"),
    "wet floor sign": ("señal de piso mojado", "f"),
    "security camera": ("cámara de seguridad", "f"),
    "fire alarm pull": ("alarma contra incendios", "f"),
    "thermometer": ("termómetro", "m"),
}

# Lookup rápido de género por nombre en español (se actualiza tras definir NAVIGATION_WORLD_CLASSES_ES)
GENDER_MAP = {name_es: gender for (name_es, gender) in WORLD_CLASSES_ES.values()}


# ============================================================================
# CLASES COMPACTAS PARA MODO NAVEGACIÓN
# ============================================================================
# Solo obstáculos físicos relevantes para movilidad peatonal (~70 clases).
# Menos clases → menos confusión entre objetos visualmente similares.
# Con 391 clases YOLO confunde puerta↔espejo, cobija↔cama, etc.
# Con ~70 clases de obstáculos esas confusiones desaparecen.

NAVIGATION_WORLD_CLASSES_ES: dict = {
    # Personas
    "person": ("persona", "f"),
    "child": ("niño", "m"),
    "baby": ("bebé", "m"),
    "wheelchair user": ("persona en silla de ruedas", "f"),
    # Animales
    "dog": ("perro", "m"),
    "puppy": ("cachorro", "m"),
    "cat": ("gato", "m"),
    "horse": ("caballo", "m"),
    "cow": ("vaca", "f"),
    "pig": ("cerdo", "m"),
    "sheep": ("oveja", "f"),
    "donkey": ("burro", "m"),
    # Vehículos
    "car": ("carro", "m"),
    "bus": ("autobús", "m"),
    "truck": ("camión", "m"),
    "motorcycle": ("motocicleta", "f"),
    "bicycle": ("bicicleta", "f"),
    "electric scooter": ("scooter eléctrico", "m"),
    "taxi cab": ("taxi", "m"),
    "ambulance vehicle": ("ambulancia", "f"),
    "police car": ("patrulla", "f"),
    # Muebles obstáculo
    "sitting chair with backrest and four legs": ("silla", "f"),
    "large wooden dining table": ("mesa", "f"),
    "flat horizontal desk with keyboard and monitor": ("escritorio", "m"),
    "upholstered sofa couch furniture": ("sofá", "m"),
    "bed frame with mattress headboard": ("cama", "f"),
    "outdoor park bench seat": ("banco", "m"),
    "kitchen counter top": ("mostrador", "m"),
    "large dining room table": ("mesa de comedor", "f"),
    "shopping cart supermarket": ("carrito de compras", "m"),
    "baby stroller pram": ("cochecito de bebé", "m"),
    "manual wheelchair": ("silla de ruedas", "f"),
    # Puertas y accesos
    "door closed shut blocking passage": ("puerta cerrada", "f"),
    "closed door flat surface": ("puerta cerrada", "f"),
    "door open wide": ("puerta abierta", "f"),
    "glass door transparent": ("puerta de vidrio", "f"),
    "revolving door": ("puerta giratoria", "f"),
    "metal gate fence": ("reja", "f"),
    "door panel": ("puerta", "f"),
    "sliding door panel": ("puerta corrediza", "f"),
    "elevator door": ("ascensor", "m"),
    # Paredes y estructuras
    "concrete wall": ("pared", "f"),
    "brick wall": ("pared de ladrillo", "f"),
    "glass wall": ("pared de vidrio", "f"),
    "pillar": ("pilar", "m"),
    "concrete column": ("columna", "f"),
    # Escaleras y desniveles
    "staircase": ("escaleras", "f"),
    "single step on floor": ("escalón", "m"),
    "escalator": ("escalera mecánica", "f"),
    "sidewalk curb": ("borde de acera", "m"),
    "manhole cover": ("tapa de alcantarilla", "f"),
    "speed bump": ("reductor de velocidad", "m"),
    # Balcones y caídas
    "balcony": ("balcón", "m"),
    "balcony railing": ("barandal de balcón", "m"),
    "ledge": ("cornisa", "f"),
    "railing": ("barandal", "m"),
    # Infraestructura de calle
    "street pole": ("poste", "m"),
    "street lamp post": ("farola", "f"),
    "tree": ("árbol", "m"),
    "fire hydrant": ("hidrante", "m"),
    "traffic cone": ("cono de tráfico", "m"),
    "road barrier": ("barrera vial", "f"),
    "metal fence": ("cerca", "f"),
    "entrance gate": ("portón", "m"),
    "traffic light": ("semáforo", "m"),
    "garbage dumpster": ("contenedor de basura", "m"),
    "metal construction scaffolding on building": ("andamio", "m"),
    # Electrodomésticos peligrosos (solo fuente de calor — los otros causan demasiados falsos positivos)
    "kitchen stove with burners cooking": ("estufa", "f"),
    "kitchen oven appliance door": ("horno", "m"),
    # Objetos portátiles obstáculo
    "backpack bag on floor or back": ("mochila", "f"),
    "travel suitcase": ("maleta", "f"),
    "cardboard box": ("caja de cartón", "f"),
    "skateboard": ("patineta", "f"),
    # Señales de peligro
    "wet floor sign": ("señal de piso mojado", "f"),
}

# Lookup combinado: cubre tanto WORLD_CLASSES_ES como NAVIGATION_WORLD_CLASSES_ES.
# Necesario porque los prompts de navegación son distintos a los completos
# (más específicos para reducir confusión), y el traductor busca por prompt inglés.
ALL_CLASSES_LOOKUP: dict = {**WORLD_CLASSES_ES, **NAVIGATION_WORLD_CLASSES_ES}


class ObjectDetectionService:
    """
    Servicio para detección de objetos en imágenes.

    Utiliza YOLO-World para identificar y localizar objetos en imágenes,
    con vocabulario abierto (~298 clases relevantes para asistencia visual).

    Uso:
        service = ObjectDetectionService()
        resultados = service.detect_objects(imagen_cv2)
    """

    def __init__(self):
        """
        Inicializa el servicio cargando el modelo YOLO-World.

        El modelo se descarga automáticamente la primera vez (~30MB).
        Después de cargar, se configuran las clases a detectar.
        """
        self.model = None
        self.confidence_threshold = settings.YOLO_CONFIDENCE_THRESHOLD
        self._current_class_mode: str = "full"
        self._load_model()

    def configure_for_navigation(self) -> None:
        """No-op: YOLOv8 estándar usa clases COCO fijas, no requiere set_classes."""
        self._current_class_mode = "navigation"

    def configure_for_full(self) -> None:
        """No-op: YOLOv8 estándar usa clases COCO fijas, no requiere set_classes."""
        self._current_class_mode = "full"

    def _load_model(self) -> None:
        """
        Carga el modelo YOLO-World y configura las clases a detectar.

        YOLO-World requiere set_classes() para definir qué objetos buscar.
        Los nombres en inglés se usan como prompts de texto para el modelo.
        """
        try:
            import ssl
            # Evitar error SSL en macOS al descargar componentes del modelo
            ssl._create_default_https_context = ssl._create_unverified_context

            model_name = settings.YOLO_MODEL
            logger.info(f"Cargando modelo YOLO-World: {model_name}")

            # Cargar modelo YOLOv8 estándar (80 clases COCO, sin CLIP)
            self.model = YOLO(model_name)

            logger.info(f"Modelo YOLO cargado: {model_name}")

        except Exception as e:
            logger.error(f"Error cargando modelo YOLO-World: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo YOLO-World: {e}")

    def detect_objects(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        depth_map: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Detecta objetos en una imagen con estimación de profundidad.

        Proceso:
        1. Ejecutar inferencia YOLO-World
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
                "error": "Modelo YOLO-World no inicializado"
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

            # Ejecutar inferencia YOLO-World
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
                "depth_map": depth_map,  # incluido para análisis de pared en WebSocket
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
        Procesa los resultados crudos de YOLO-World con estimación de profundidad.

        Args:
            results: Resultados de YOLO-World
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
                class_info = ALL_CLASSES_LOOKUP.get(class_name)
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
            "num_classes": len(WORLD_CLASSES_ES),
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
