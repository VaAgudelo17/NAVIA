"""
============================================================================
NAVIA Backend - Clasificador Jerárquico de Documentos v3
============================================================================
Clasificador rediseñado con taxonomía mejorada para lectura accesible.

Arquitectura de 2 fases + regla de texto insuficiente:

  Pre-check: Si < 10 palabras válidas O densidad extremadamente baja
    → IMAGEN_VISUAL + visual_description (sin ejecutar clasificación)

  Fase 1 (MACRO): Señales estructurales del layout OCR
    → DOCUMENTO_FORMAL | INTERFAZ_DIGITAL | TEXTO_CONVERSACIONAL | IMAGEN_VISUAL

  Fase 2 (SUBTIPO): Keywords + patrones semánticos sobre texto limpio
    → factura, chat, red_social, login, etc.

  Post: Asignación de reading_mode recomendado
    → dialogue | structured_fields | list_items | paragraph_text | visual_description

Señales de layout (Fase 1):
  - Densidad de texto (% del área cubierta por palabras)
  - Distribución vertical (palabras concentradas arriba/abajo/uniformes)
  - Cantidad de bloques y líneas de Tesseract
  - Varianza de alturas de palabra (detecta encabezados grandes)
  - Alineación horizontal (columnas, centrado, irregular)
  - Ratio de aspecto del área de texto
  - Señales UI: palabras cortas, bloques aislados, texto centrado

Mecanismos adicionales:
  - Scoring probabilístico normalizado 0-1
  - Gap mínimo entre top-2 candidatos → "mixto" si gap < 0.15
  - No forzar clasificación cuando confianza < 0.6
  - Estabilización temporal entre frames (memoria de corto plazo)
  - Explicación interna de la decisión (reasoning trace)
  - reading_mode recomendado para NarrativeGenerator
============================================================================
"""

import re
import time
import logging
import statistics
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LayoutSignals:
    """Señales estructurales extraídas del layout OCR."""
    # Densidad
    text_density: float = 0.0       # % del área imagen cubierta por texto
    word_count: int = 0
    num_blocks: int = 0
    num_lines: int = 0
    words_per_line: float = 0.0

    # Distribución vertical (0.0=arriba, 1.0=abajo)
    vertical_center: float = 0.5    # centro de masa vertical del texto
    vertical_spread: float = 0.0    # desviación estándar vertical normalizada
    top_heavy: bool = False         # >60% del texto en la mitad superior
    bottom_heavy: bool = False      # >60% del texto en la mitad inferior

    # Tamaños de texto
    avg_word_height: float = 0.0    # altura promedio de palabras (px)
    max_word_height: float = 0.0    # altura máxima detectada
    height_variance: float = 0.0    # varianza normalizada de alturas
    has_large_text: bool = False    # algún texto >2x el promedio

    # Alineación horizontal
    left_aligned_ratio: float = 0.0   # % de palabras alineadas a la izquierda
    center_aligned_ratio: float = 0.0 # % de palabras centradas
    has_columns: bool = False         # detecta 2+ columnas verticales

    # Espaciado
    avg_line_gap: float = 0.0      # separación promedio entre líneas
    has_dense_region: bool = False  # alguna zona con muchas palabras juntas
    has_sparse_region: bool = False # alguna zona con pocas palabras

    # Área de texto
    text_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (left, top, right, bottom)
    text_aspect_ratio: float = 1.0  # width/height del bbox de texto

    # --- NUEVAS señales UI (v3) ---
    short_word_ratio: float = 0.0   # % de palabras con <=4 chars (típico UI)
    avg_word_length: float = 0.0    # longitud promedio de palabras
    block_isolation: float = 0.0    # separación promedio entre bloques (normalizada)
    has_ui_pattern: bool = False    # detecta patrón UI (bloques aislados + palabras cortas)


@dataclass
class ClassificationExplanation:
    """Explicación interna de por qué se eligió un tipo."""
    macro_type: str = ""
    macro_reasons: List[str] = field(default_factory=list)
    macro_scores: Dict[str, float] = field(default_factory=dict)

    subtype: str = ""
    subtype_reasons: List[str] = field(default_factory=list)
    subtype_scores: Dict[str, float] = field(default_factory=dict)

    final_type: str = ""
    final_confidence: float = 0.0
    is_ambiguous: bool = False
    ambiguity_note: str = ""
    was_stabilized: bool = False
    stabilization_note: str = ""


@dataclass
class ClassificationResult:
    """Resultado completo de la clasificación."""
    doc_type: str
    confidence: float
    label: str
    macro_type: str
    reading_mode: str                # NUEVO: dialogue | structured_fields | list_items | paragraph_text | visual_description
    explanation: ClassificationExplanation
    layout_signals: LayoutSignals


# ============================================================================
# TAXONOMÍA v3
# ============================================================================

_MACRO_TYPES = (
    "DOCUMENTO_FORMAL",
    "INTERFAZ_DIGITAL",
    "TEXTO_CONVERSACIONAL",
    "IMAGEN_VISUAL",
)

_MACRO_TO_SUBTYPES: Dict[str, List[str]] = {
    "DOCUMENTO_FORMAL": [
        "factura", "recibo", "carta", "formulario", "contrato",
        "hoja_de_vida", "informe", "noticia", "correo", "presentacion",
        "etiqueta", "tarjeta", "documento_informativo",
        "receta_medica", "boleto", "identificacion", "horario",
        "instrucciones", "resultado_lab", "tabla_nutricional",
        "calendario", "factura_servicio", "ticket_transporte", "credencial",
        "tarjeta_felicitacion",
    ],
    "INTERFAZ_DIGITAL": [
        "app_menu", "app_settings", "app_login", "app_form",
        "app_social", "app_service", "app_banking", "notificacion", "mapa",
    ],
    "TEXTO_CONVERSACIONAL": [
        "chat", "comentario",
    ],
    "IMAGEN_VISUAL": [
        "imagen_visual",
    ],
}

# Mapeo subtipo → reading_mode recomendado
_SUBTYPE_TO_READING_MODE: Dict[str, str] = {
    # DOCUMENTO_FORMAL
    "factura": "structured_fields",
    "recibo": "structured_fields",
    "formulario": "structured_fields",
    "contrato": "structured_fields",
    "hoja_de_vida": "paragraph_text",
    "carta": "paragraph_text",
    "informe": "paragraph_text",
    "noticia": "paragraph_text",
    "correo": "paragraph_text",
    "presentacion": "paragraph_text",
    "documento_informativo": "paragraph_text",
    "etiqueta": "list_items",
    "tarjeta": "structured_fields",
    "receta_medica": "structured_fields",
    "boleto": "structured_fields",
    "identificacion": "structured_fields",
    "horario": "list_items",
    "instrucciones": "list_items",
    "resultado_lab": "structured_fields",
    "tabla_nutricional": "structured_fields",
    "calendario": "list_items",
    "factura_servicio": "structured_fields",
    "ticket_transporte": "structured_fields",
    "credencial": "structured_fields",

    # INTERFAZ_DIGITAL
    "app_menu": "list_items",
    "app_settings": "list_items",
    "app_login": "structured_fields",
    "app_form": "structured_fields",
    "app_social": "dialogue",
    "app_service": "structured_fields",
    "app_banking": "structured_fields",
    "tarjeta_felicitacion": "paragraph_text",
    "notificacion": "list_items",
    "mapa": "visual_description",

    # TEXTO_CONVERSACIONAL
    "chat": "dialogue",
    "comentario": "dialogue",

    # IMAGEN_VISUAL
    "imagen_visual": "visual_description",

    # Fallbacks
    "desconocido": "paragraph_text",
    "mixto": "paragraph_text",
}

# Mapeo macro-tipo → reading_mode por defecto (cuando subtipo es desconocido)
_MACRO_TO_DEFAULT_READING_MODE: Dict[str, str] = {
    "DOCUMENTO_FORMAL": "paragraph_text",
    "INTERFAZ_DIGITAL": "list_items",
    "TEXTO_CONVERSACIONAL": "dialogue",
    "IMAGEN_VISUAL": "visual_description",
}


# ============================================================================
# LAYOUT ANALYZER - Extrae señales estructurales
# ============================================================================

class LayoutAnalyzer:
    """
    Extrae señales estructurales del layout OCR usando los bounding boxes
    de Tesseract. Incluye señales específicas para detectar UI de apps.
    """

    def analyze(self, layout_data: Dict, img_width: int, img_height: int) -> LayoutSignals:
        """
        Analiza la estructura espacial del texto detectado por Tesseract.

        Args:
            layout_data: Dict con 'word_boxes', 'num_blocks', 'num_lines'
            img_width: Ancho de la imagen procesada
            img_height: Alto de la imagen procesada

        Returns:
            LayoutSignals con todas las métricas estructurales
        """
        signals = LayoutSignals()
        boxes = layout_data.get("word_boxes", [])

        if not boxes or img_width == 0 or img_height == 0:
            signals.num_blocks = layout_data.get("num_blocks", 0)
            signals.num_lines = layout_data.get("num_lines", 0)
            return signals

        signals.word_count = len(boxes)
        signals.num_blocks = layout_data.get("num_blocks", 0)
        signals.num_lines = layout_data.get("num_lines", 0)
        signals.words_per_line = (
            len(boxes) / signals.num_lines if signals.num_lines > 0 else 0.0
        )

        # --- Densidad de texto ---
        total_text_area = sum(b["width"] * b["height"] for b in boxes)
        img_area = img_width * img_height
        signals.text_density = total_text_area / img_area if img_area > 0 else 0.0

        # --- Distribución vertical ---
        y_centers = [(b["top"] + b["height"] / 2) / img_height for b in boxes]
        signals.vertical_center = statistics.mean(y_centers) if y_centers else 0.5
        signals.vertical_spread = (
            statistics.stdev(y_centers) if len(y_centers) > 1 else 0.0
        )
        top_count = sum(1 for y in y_centers if y < 0.5)
        signals.top_heavy = top_count > len(y_centers) * 0.6
        signals.bottom_heavy = top_count < len(y_centers) * 0.4

        # --- Tamaños de texto ---
        heights = [b["height"] for b in boxes if b["height"] > 0]
        if heights:
            signals.avg_word_height = statistics.mean(heights)
            signals.max_word_height = max(heights)
            signals.height_variance = (
                statistics.stdev(heights) / signals.avg_word_height
                if signals.avg_word_height > 0 and len(heights) > 1
                else 0.0
            )
            signals.has_large_text = signals.max_word_height > signals.avg_word_height * 2.0

        # --- Alineación horizontal ---
        lefts = [b["left"] for b in boxes]
        if lefts:
            left_margin_threshold = img_width * 0.15
            center_band = (img_width * 0.3, img_width * 0.7)

            left_count = sum(1 for l in lefts if l < left_margin_threshold)
            center_count = sum(
                1 for i, b in enumerate(boxes)
                if center_band[0] < (b["left"] + b["width"] / 2) < center_band[1]
            )

            signals.left_aligned_ratio = left_count / len(lefts)
            signals.center_aligned_ratio = center_count / len(lefts)

            # Detectar columnas: si hay 2+ clusters horizontales
            signals.has_columns = self._detect_columns(boxes, img_width)

        # --- Espaciado entre líneas ---
        signals.avg_line_gap = self._compute_line_gaps(boxes, img_height)

        # --- Regiones densas y dispersas ---
        signals.has_dense_region, signals.has_sparse_region = (
            self._detect_density_regions(boxes, img_width, img_height)
        )

        # --- Bounding box del texto completo ---
        all_left = min(b["left"] for b in boxes)
        all_top = min(b["top"] for b in boxes)
        all_right = max(b["left"] + b["width"] for b in boxes)
        all_bottom = max(b["top"] + b["height"] for b in boxes)
        signals.text_bbox = (all_left, all_top, all_right, all_bottom)

        text_w = all_right - all_left
        text_h = all_bottom - all_top
        signals.text_aspect_ratio = text_w / text_h if text_h > 0 else 1.0

        # --- NUEVAS señales UI (v3) ---
        self._compute_ui_signals(boxes, signals, img_width, img_height)

        return signals

    def _compute_ui_signals(
        self, boxes: List[Dict], signals: LayoutSignals,
        img_width: int, img_height: int
    ) -> None:
        """Calcula señales específicas para detectar interfaces digitales."""
        # Longitud promedio de palabras y ratio de palabras cortas
        word_texts = [b.get("text", "") for b in boxes if b.get("text")]
        if word_texts:
            lengths = [len(w) for w in word_texts]
            signals.avg_word_length = statistics.mean(lengths) if lengths else 0.0
            short_count = sum(1 for l in lengths if l <= 4)
            signals.short_word_ratio = short_count / len(lengths) if lengths else 0.0
        else:
            # Estimar desde el ancho de los bboxes si no hay texto
            widths = [b["width"] for b in boxes if b["width"] > 0]
            if widths and signals.avg_word_height > 0:
                avg_chars_est = statistics.mean(widths) / (signals.avg_word_height * 0.6)
                signals.avg_word_length = avg_chars_est
                short_count = sum(1 for w in widths if w / (signals.avg_word_height * 0.6) <= 4)
                signals.short_word_ratio = short_count / len(widths)

        # Aislamiento entre bloques: distancia vertical promedio entre bloques
        if signals.num_blocks >= 2:
            block_centers: Dict[int, List[float]] = {}
            for b in boxes:
                blk = b.get("block", 0)
                if blk not in block_centers:
                    block_centers[blk] = []
                block_centers[blk].append((b["top"] + b["height"] / 2) / img_height)

            if len(block_centers) >= 2:
                centers_sorted = sorted(
                    statistics.mean(ys) for ys in block_centers.values()
                )
                gaps = [
                    centers_sorted[i + 1] - centers_sorted[i]
                    for i in range(len(centers_sorted) - 1)
                ]
                signals.block_isolation = statistics.mean(gaps) if gaps else 0.0

        # Patrón UI: bloques aislados + palabras cortas + densidad moderada
        signals.has_ui_pattern = (
            signals.num_blocks >= 2
            and signals.short_word_ratio > 0.4
            and signals.text_density < 0.08
            and signals.block_isolation > 0.05
        )

    def _detect_columns(self, boxes: List[Dict], img_width: int) -> bool:
        """Detecta si el texto está organizado en columnas."""
        if len(boxes) < 10:
            return False

        mid = img_width / 2
        left_words = [b for b in boxes if (b["left"] + b["width"] / 2) < mid * 0.8]
        right_words = [b for b in boxes if (b["left"] + b["width"] / 2) > mid * 1.2]

        if left_words and right_words:
            left_ratio = len(left_words) / len(boxes)
            right_ratio = len(right_words) / len(boxes)
            return left_ratio > 0.2 and right_ratio > 0.2

        return False

    def _compute_line_gaps(self, boxes: List[Dict], img_height: int) -> float:
        """Calcula el espaciado promedio entre líneas."""
        line_ys: Dict[Tuple[int, int], List[float]] = {}
        for b in boxes:
            key = (b["block"], b["line"])
            if key not in line_ys:
                line_ys[key] = []
            line_ys[key].append(b["top"] + b["height"] / 2)

        sorted_ys = sorted(
            statistics.mean(ys) for ys in line_ys.values()
        )

        if len(sorted_ys) < 2:
            return 0.0

        gaps = [sorted_ys[i + 1] - sorted_ys[i] for i in range(len(sorted_ys) - 1)]
        avg_gap = statistics.mean(gaps) / img_height if img_height > 0 else 0.0
        return avg_gap

    def _detect_density_regions(
        self, boxes: List[Dict], img_width: int, img_height: int
    ) -> Tuple[bool, bool]:
        """Detecta si hay regiones con alta y baja densidad de texto."""
        if len(boxes) < 5:
            return False, False

        quadrant_counts = [0, 0, 0, 0]  # TL, TR, BL, BR
        mid_x, mid_y = img_width / 2, img_height / 2
        for b in boxes:
            cx = b["left"] + b["width"] / 2
            cy = b["top"] + b["height"] / 2
            idx = (0 if cy < mid_y else 2) + (0 if cx < mid_x else 1)
            quadrant_counts[idx] += 1

        total = sum(quadrant_counts)
        if total == 0:
            return False, False

        ratios = [c / total for c in quadrant_counts]
        has_dense = any(r > 0.5 for r in ratios)
        has_sparse = any(r < 0.05 for r in ratios)
        return has_dense, has_sparse


# ============================================================================
# MACRO CLASSIFIER - Fase 1 (solo layout, sin keywords)
# ============================================================================
# Scoring probabilístico normalizado 0-1

class MacroClassifier:
    """
    Fase 1: Clasifica en macro-tipo usando señales estructurales del layout
    + boost semántico ligero para corregir ambigüedades.

    El boost semántico NO reemplaza la fase 2 (subtipos). Solo interviene
    cuando el layout genera un macro-tipo incorrecto (ej: factura con pocas
    palabras por línea → el layout dice TEXTO_CONVERSACIONAL, pero las
    keywords TOTAL/IVA/SUBTOTAL dicen DOCUMENTO_FORMAL).

    Scoring normalizado 0-1:
      - Si gap entre top-1 y top-2 < 0.15 → marca como ambiguo
      - Si confianza < 0.6 → no forzar clasificación
    """

    # Gap mínimo entre top-1 y top-2 para clasificación confiable
    MIN_GAP = 0.15

    # Confianza mínima para aceptar clasificación
    MIN_CONFIDENCE = 0.6

    # Keywords semánticas que indican DOCUMENTO_FORMAL (facturas/recibos)
    # cuando el layout es ambiguo. Estas NO reemplazan la fase de subtipos,
    # solo ayudan a que el macro-tipo no se equivoque.
    _FORMAL_KEYWORDS = re.compile(
        r'\b(?:'
        r'total|subtotal|sub\s*t(?:ot|tl)'
        r'|i\.?v\.?a\.?|impuesto'
        r'|factura|invoice|recibo|receipt'
        r'|n\.?i\.?t\.?|r\.?i\.?f\.?|r\.?u\.?c\.?'
        r'|precio|monto|pago|descuento'
        r'|cantidad|unidad|producto'
        r'|tarjeta\s+d[eé]bito|tarjeta\s+cr[eé]dito|efectivo|cambio'
        r'|base\s+imp|discriminacion\s+tarifa'
        r')\b',
        re.I
    )

    # Keywords que indican TEXTO_CONVERSACIONAL real (chat/mensajes)
    _CHAT_KEYWORDS = re.compile(
        r'\b(?:'
        r'jaja|hola|ok[ií]?\b|xd|lol|jejej|ajaj'
        r'|buenos?\s+d[ií]as?|buenas?\s+(?:tardes?|noches?)'
        r'|gracias|dale|va[le]e|sip?|nop?'
        r')\b',
        re.I
    )

    # Patrón de timestamps de chat (8:35 PM, 10:21 a.m.)
    _CHAT_TIMESTAMP = re.compile(
        r'\d{1,2}:\d{2}\s*(?:p\.?m\.?|a\.?m\.?)',
        re.I
    )

    def classify(self, signals: LayoutSignals, ocr_confidence: float,
                 raw_text: str = ""
                 ) -> Tuple[str, float, List[str]]:
        """
        Clasifica el macro-tipo basado en señales de layout + boost semántico.

        El boost semántico NO reemplaza la fase 2 (subtipos), solo corrige
        casos donde el layout engaña (ej: factura con pocas palabras/línea
        → parece chat).

        Args:
            signals: LayoutSignals del analyzer
            ocr_confidence: Confianza promedio del OCR
            raw_text: Texto OCR crudo (para boost semántico, opcional)

        Returns:
            (macro_type, confidence, reasons)
        """
        scores: Dict[str, float] = {m: 0.0 for m in _MACRO_TYPES}
        reasons: Dict[str, List[str]] = {m: [] for m in _MACRO_TYPES}

        # =============================================================
        # IMAGEN_VISUAL: poco o nada de texto
        # =============================================================
        if signals.word_count < 3:
            scores["IMAGEN_VISUAL"] += 30
            reasons["IMAGEN_VISUAL"].append(f"Muy pocas palabras ({signals.word_count})")
        if signals.text_density < 0.005:
            scores["IMAGEN_VISUAL"] += 20
            reasons["IMAGEN_VISUAL"].append(
                f"Densidad de texto muy baja ({signals.text_density:.4f})"
            )
        if signals.word_count == 0:
            scores["IMAGEN_VISUAL"] += 20
            reasons["IMAGEN_VISUAL"].append("Sin texto detectado")

        # =============================================================
        # DOCUMENTO_FORMAL: texto denso, estructurado, muchas líneas
        # =============================================================
        if signals.text_density > 0.04:
            scores["DOCUMENTO_FORMAL"] += 15
            reasons["DOCUMENTO_FORMAL"].append(
                f"Alta densidad de texto ({signals.text_density:.3f})"
            )
        if signals.num_lines > 10:
            scores["DOCUMENTO_FORMAL"] += 10
            reasons["DOCUMENTO_FORMAL"].append(f"Muchas líneas ({signals.num_lines})")
        if signals.num_blocks >= 3:
            scores["DOCUMENTO_FORMAL"] += 8
            reasons["DOCUMENTO_FORMAL"].append(f"Múltiples bloques ({signals.num_blocks})")
        if signals.has_columns:
            scores["DOCUMENTO_FORMAL"] += 12
            reasons["DOCUMENTO_FORMAL"].append("Detecta columnas (layout tabular)")
        if signals.left_aligned_ratio > 0.6:
            scores["DOCUMENTO_FORMAL"] += 8
            reasons["DOCUMENTO_FORMAL"].append(
                f"Texto alineado a la izquierda ({signals.left_aligned_ratio:.0%})"
            )
        if signals.has_large_text and signals.height_variance > 0.3:
            scores["DOCUMENTO_FORMAL"] += 6
            reasons["DOCUMENTO_FORMAL"].append(
                "Variación de tamaños (encabezados + cuerpo)"
            )
        if signals.vertical_spread > 0.25:
            scores["DOCUMENTO_FORMAL"] += 5
            reasons["DOCUMENTO_FORMAL"].append(
                f"Texto distribuido verticalmente ({signals.vertical_spread:.2f})"
            )
        # Penalizar documento si tiene patrón UI
        if signals.has_ui_pattern:
            scores["DOCUMENTO_FORMAL"] -= 10
            reasons["DOCUMENTO_FORMAL"].append(
                "Penalización: patrón UI detectado (bloques aislados + palabras cortas)"
            )

        # =============================================================
        # INTERFAZ_DIGITAL: bloques aislados, palabras cortas, UI
        # =============================================================
        if signals.has_ui_pattern:
            scores["INTERFAZ_DIGITAL"] += 18
            reasons["INTERFAZ_DIGITAL"].append(
                "Patrón UI detectado (bloques aislados + palabras cortas + densidad moderada)"
            )
        if signals.num_blocks >= 2 and signals.text_density < 0.06:
            scores["INTERFAZ_DIGITAL"] += 10
            reasons["INTERFAZ_DIGITAL"].append(
                f"Bloques separados con baja densidad ({signals.num_blocks} bloques)"
            )
        if signals.has_sparse_region and signals.has_dense_region:
            scores["INTERFAZ_DIGITAL"] += 10
            reasons["INTERFAZ_DIGITAL"].append(
                "Mezcla de zonas densas y vacías (layout de app)"
            )
        if signals.center_aligned_ratio > 0.4:
            scores["INTERFAZ_DIGITAL"] += 8
            reasons["INTERFAZ_DIGITAL"].append(
                f"Texto centrado ({signals.center_aligned_ratio:.0%})"
            )
        if signals.top_heavy and signals.word_count < 40:
            scores["INTERFAZ_DIGITAL"] += 6
            reasons["INTERFAZ_DIGITAL"].append(
                "Texto concentrado arriba (barra/header de app)"
            )
        if 5 <= signals.word_count <= 50 and signals.num_blocks >= 2:
            scores["INTERFAZ_DIGITAL"] += 5
            reasons["INTERFAZ_DIGITAL"].append(
                f"Cantidad moderada de texto ({signals.word_count} palabras)"
            )
        if signals.short_word_ratio > 0.5:
            scores["INTERFAZ_DIGITAL"] += 8
            reasons["INTERFAZ_DIGITAL"].append(
                f"Alto ratio de palabras cortas ({signals.short_word_ratio:.0%})"
            )
        if signals.block_isolation > 0.08:
            scores["INTERFAZ_DIGITAL"] += 6
            reasons["INTERFAZ_DIGITAL"].append(
                f"Bloques bien separados (isolation={signals.block_isolation:.2f})"
            )

        # =============================================================
        # TEXTO_CONVERSACIONAL: burbujas, bloques alternados, muchas
        # líneas cortas, distribución vertical amplia
        # =============================================================
        if signals.words_per_line < 6 and signals.num_lines > 5:
            scores["TEXTO_CONVERSACIONAL"] += 12
            reasons["TEXTO_CONVERSACIONAL"].append(
                f"Líneas cortas ({signals.words_per_line:.1f} palabras/línea) + muchas líneas"
            )
        if signals.num_blocks >= 3 and signals.vertical_spread > 0.2:
            scores["TEXTO_CONVERSACIONAL"] += 10
            reasons["TEXTO_CONVERSACIONAL"].append(
                f"Múltiples bloques distribuidos verticalmente ({signals.num_blocks} bloques)"
            )
        if signals.has_dense_region and signals.has_sparse_region:
            scores["TEXTO_CONVERSACIONAL"] += 6
            reasons["TEXTO_CONVERSACIONAL"].append(
                "Mezcla de zonas densas y vacías (burbujas de chat)"
            )
        if 0.02 < signals.text_density < 0.08 and signals.num_lines > 8:
            scores["TEXTO_CONVERSACIONAL"] += 8
            reasons["TEXTO_CONVERSACIONAL"].append(
                f"Densidad media con muchas líneas (chat-like)"
            )
        # Alternancia izquierda/derecha en bloques (burbujas de chat)
        if signals.left_aligned_ratio < 0.5 and signals.center_aligned_ratio < 0.5:
            if signals.num_blocks >= 3:
                scores["TEXTO_CONVERSACIONAL"] += 5
                reasons["TEXTO_CONVERSACIONAL"].append(
                    "Texto ni alineado a izquierda ni centrado (posibles burbujas alternadas)"
                )

        # =============================================================
        # BOOST SEMÁNTICO: keywords que corrigen macro cuando layout
        # es ambiguo (ej: factura con pocas palabras/línea → parece chat)
        # Solo aplica cuando hay texto disponible.
        # =============================================================
        if raw_text and len(raw_text) > 20:
            text_upper = raw_text.upper()

            # --- Boost DOCUMENTO_FORMAL por keywords de factura/recibo ---
            formal_hits = self._FORMAL_KEYWORDS.findall(text_upper)
            n_formal = len(formal_hits)
            if n_formal >= 3:
                boost = min(n_formal * 5, 25)  # 5 pts por hit, tope 25
                scores["DOCUMENTO_FORMAL"] += boost
                reasons["DOCUMENTO_FORMAL"].append(
                    f"Semantic boost: {n_formal} keywords financieras "
                    f"({', '.join(set(h.strip() for h in formal_hits[:5]))})"
                )
                # Penalizar TEXTO_CONVERSACIONAL y INTERFAZ_DIGITAL
                # cuando hay señales claras de factura
                if n_formal >= 4:
                    scores["TEXTO_CONVERSACIONAL"] -= 10
                    reasons["TEXTO_CONVERSACIONAL"].append(
                        f"Penalización: {n_formal} keywords financieras → no es chat"
                    )
                    scores["INTERFAZ_DIGITAL"] -= 8
                    reasons["INTERFAZ_DIGITAL"].append(
                        f"Penalización: {n_formal} keywords financieras → no es app"
                    )
            elif n_formal >= 2:
                # Boost menor con solo 2 hits
                scores["DOCUMENTO_FORMAL"] += 8
                reasons["DOCUMENTO_FORMAL"].append(
                    f"Semantic boost (leve): {n_formal} keywords financieras"
                )

            # --- Boost TEXTO_CONVERSACIONAL por keywords de chat ---
            chat_hits = self._CHAT_KEYWORDS.findall(raw_text)
            ts_hits = self._CHAT_TIMESTAMP.findall(raw_text)
            n_chat = len(chat_hits) + len(ts_hits)
            if n_chat >= 3:
                boost = min(n_chat * 4, 20)
                scores["TEXTO_CONVERSACIONAL"] += boost
                reasons["TEXTO_CONVERSACIONAL"].append(
                    f"Semantic boost: {n_chat} señales de chat/mensajes"
                )

        # =============================================================
        # Normalizar scores a 0-1
        # =============================================================
        max_score = max(scores.values())
        if max_score == 0:
            return "IMAGEN_VISUAL", 0.3, ["Sin señales de layout claras"]

        normalized: Dict[str, float] = {}
        for m in _MACRO_TYPES:
            normalized[m] = max(scores[m], 0.0) / max_score if max_score > 0 else 0.0

        # Ordenar por score
        sorted_types = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = sorted_types[0]
        second_type, second_score = sorted_types[1] if len(sorted_types) > 1 else ("", 0.0)

        # Gap analysis
        gap = best_score - second_score  # ya normalizado 0-1

        if gap < self.MIN_GAP and best_score > 0:
            # Ambiguo: muy cerca entre top-1 y top-2
            reasons[best_type].append(
                f"AMBIGUO: gap={gap:.2f} < {self.MIN_GAP} vs {second_type}"
            )
            # Usar confianza reducida
            confidence = best_score * 0.6
        else:
            confidence = best_score

        # Guardar scores normalizados en reasons para debug
        return best_type, round(confidence, 3), reasons[best_type]


# ============================================================================
# SUBTYPE CLASSIFIER - Fase 2 (keywords sobre texto limpio)
# ============================================================================

# Acrónimos cortos (<=3 chars, uppercase) que NO deben eliminarse
# en normalize_for_classification. Sin esto, "IVA", "NIT", "RIF" etc.
# se eliminan por la regla anti-basura y los subtipos factura/recibo pierden señales.
_IMPORTANT_SHORT_TOKENS = frozenset({
    'iva', 'nit', 'rif', 'ruc', 'rfc',     # Fiscal
    'dni', 'dui',                            # Identificación
    'cc', 'bcc',                             # Correo
    'rx',                                    # Receta médica
    'qr',                                    # Boleto/código
    'pnr', 'eta',                            # Viaje
    'cv',                                    # Hoja de vida
    'pdf', 'url',                            # Documento digital
    'atm', 'pin', 'otp',                     # Bancario
    'usd', 'eur', 'cop', 'mxn', 'ars',      # Monedas
    'pen', 'bob', 'brl', 'clp',             # Monedas LATAM
})

# Palabras cortas reales en español que no son basura OCR
_REAL_SHORT_WORDS = frozenset({
    'y', 'o', 'a', 'e', 'u', 'el', 'la', 'de', 'no', 'es',
    'lo', 'en', 'se', 'me', 'te', 'le', 'al', 'mi', 'tu',
    'si', 'ya', 'un', 'yo', 'ni', 'he', 'su', 'os', 'do',
    'que', 'por', 'con', 'del', 'las', 'los', 'una', 'son',
    'fue', 'ser', 'hay', 'van', 'mas', 'más', 'sin', 'nos',
    'hoy', 'muy', 'día', 'dia', 'ver', 'dar', 'mal', 'bien',
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
    'can', 'had', 'her', 'was', 'one', 'our', 'out',
})


def normalize_for_classification(raw_text: str) -> str:
    """
    Limpia texto OCR ANTES de clasificar.
    Más agresivo que la limpieza para narrativa — elimina toda la basura
    que puede causar falsos positivos en las keywords.
    """
    if not raw_text:
        return ""

    words = raw_text.split()
    clean = []
    for w in words:
        # Preservar "Tú:" y "Yo:" antes de strip (importante para chat)
        if re.match(r'^(?:T[uú]|Yo):$', w, re.I):
            clean.append(w)
            continue

        stripped = w.strip('.,;:!?()[]{}"\'`')
        wl = stripped.lower()

        if not wl:
            continue

        # Preservar palabras cortas reales
        if wl in _REAL_SHORT_WORDS:
            clean.append(stripped)
            continue

        # Eliminar tokens de 1-2 chars no reconocidos
        if len(wl) <= 2:
            continue

        # Preservar números con contexto (5min, 20%, 151K)
        if re.match(r'^\d+[%KkMm]?$', wl):
            clean.append(stripped)
            continue

        # Preservar timestamps de chat (8:35PM, 8:57 p.m., 2:36pmM, s:32pm)
        if re.match(r'^\d{1,2}:\d{2}\s*(?:p\.?m\.?|a\.?m\.?|pm|am)?[mM]?$', wl, re.I):
            clean.append(stripped)
            continue

        # Preservar "p.m." y "a.m." sueltos
        if re.match(r'^[ap]\.?m\.?$', wl, re.I):
            clean.append(stripped)
            continue

        # Eliminar tokens puramente simbólicos
        if re.match(r'^[\d\W]+$', wl):
            continue

        # Eliminar tokens con >50% basura (no-letras)
        letter_count = sum(1 for c in wl if c.isalpha())
        if letter_count < len(wl) * 0.5:
            continue

        # Eliminar tokens cortos all-caps (basura UI: "GD", "ED")
        # EXCEPTO acrónimos importantes para clasificación
        if len(stripped) <= 3 and stripped.isupper():
            if wl not in _IMPORTANT_SHORT_TOKENS:
                continue

        clean.append(stripped)

    return " ".join(clean)


# Reglas de subtipo organizadas por la nueva taxonomía.
# Cada subtipo tiene: [(regex_compilado, peso), ...]

_SUBTYPE_RULES: Dict[str, List[Tuple[re.Pattern, int]]] = {
    # ===== DOCUMENTO_FORMAL =====
    "factura": [
        # Palabra clave directa
        (re.compile(r'\bfactura\b', re.I), 10),
        (re.compile(r'\bfactura\s+(?:electr[oó]nica|de\s+venta|de\s+compra|comercial)\b', re.I), 12),
        (re.compile(r'\bn[úu]mero\s+de\s+factura\b', re.I), 10),
        (re.compile(r'\bn[úu]mero\s+de\s+control\b', re.I), 12),
        (re.compile(r'\bfact\.?\s*(?:No\.?|N[úu]m\.?|#)\s*\d+', re.I), 10),
        (re.compile(r'\bcontrol\s+fiscal\b', re.I), 10),
        (re.compile(r'\btimbre\b', re.I), 8),
        # Identificación fiscal
        (re.compile(r'\bR\.?I\.?F\.?\b', re.I), 7),
        (re.compile(r'\bN\.?I\.?T\.?\b', re.I), 7),
        (re.compile(r'\bC\.?U\.?I\.?T\.?\b', re.I), 7),
        (re.compile(r'\bR\.?U\.?C\.?\b', re.I), 6),
        (re.compile(r'\bR\.?F\.?C\.?\b', re.I), 6),
        # Totales y montos
        (re.compile(r'\btotal\s*:?\s*[\$\€Bs\.]*\s*[\d,.]+', re.I), 7),
        (re.compile(r'\bsubtotal\b', re.I), 8),
        (re.compile(r'\bi\.?v\.?a\.?\b', re.I), 7),
        (re.compile(r'\bimpuesto\b', re.I), 6),
        (re.compile(r'\bdescuento\b', re.I), 5),
        # Estructura de factura
        (re.compile(r'\bcantidad\b.*\bprecio\b', re.I | re.S), 6),
        (re.compile(r'\bprecio\s+unitario\b', re.I), 7),
        (re.compile(r'\bbase\s+imponible\b', re.I), 12),
        (re.compile(r'\bforma\s+de\s+pago\b', re.I), 8),
        (re.compile(r'\bcondiciones?\s+de\s+pago\b', re.I), 6),
        (re.compile(r'\bfecha\s+de\s+(?:emisi[oó]n|vencimiento|pago)\b', re.I), 5),
        # Proveedor / cliente
        (re.compile(r'\b(?:proveedor|vendedor|emisor|cliente|comprador|receptor)\b', re.I), 5),
        (re.compile(r'\braz[oó]n\s+social\b', re.I), 7),
        # Keywords negativos (rechaza si aparecen)
        (re.compile(r'\b(?:cancelado|pagado|en\s+concepto\s+de)\b', re.I), -5),
    ],
    "recibo": [
        (re.compile(r'\brecibo\b', re.I), 10),
        (re.compile(r'\brecib[oí]\s+de\b', re.I), 10),
        (re.compile(r'\brecibo\s+de\s+(?:caja|pago|cobro|arrendamiento|alquiler|n[oó]mina|sueldo)\b', re.I), 12),
        (re.compile(r'\bpagado\b', re.I), 10),
        (re.compile(r'\bcancelado\b', re.I), 10),
        (re.compile(r'\bmonto\s+pagado\b', re.I), 8),
        (re.compile(r'\bcomprobante\b', re.I), 7),
        (re.compile(r'\bcomprobante\s+de\s+(?:pago|dep[oó]uito|transferencia)\b', re.I), 10),
        (re.compile(r'\bpagadero\s+a\b', re.I), 12),
        (re.compile(r'\bfolio\b', re.I), 8),
        (re.compile(r'\breferencia\s+de\s+pago\b', re.I), 10),
        (re.compile(r'\ben\s+concepto\s+de\b', re.I), 10),
        # Pagos
        (re.compile(r'\b(?:abonado|saldo)\b', re.I), 6),
        (re.compile(r'\b(?:efectivo|tarjeta|transferencia|cheque|dep[oó]uito)\b', re.I), 5),
        (re.compile(r'\b(?:n[oó]mina|sueldo|salario|quincena)\b', re.I), 7),
        (re.compile(r'\brecib[ií]\s+(?:conforme|de\s+conformidad)\b', re.I), 9),
        # Keywords negativos (rechaza si aparecen)
        (re.compile(r'\bn[úu]mero\s+de\s+control\b', re.I), -8),
        (re.compile(r'\bbase\s+imponible\b', re.I), -8),
    ],
    "carta": [
        (re.compile(r'\bestimat[oa]\b', re.I), 8),
        (re.compile(r'\batentamente\b', re.I), 9),
        (re.compile(r'\bcordialmente\b', re.I), 9),
        (re.compile(r'\ba\s+quien\s+corresponda\b', re.I), 10),
        (re.compile(r'\bme\s+dirijo\b', re.I), 8),
        (re.compile(r'\bla\s+presente\b', re.I), 7),
        # Saludos formales
        (re.compile(r'\b(?:respetad[oa]|apreciad[oa]|distinguid[oa])\b', re.I), 8),
        (re.compile(r'\b(?:se[ñn]or(?:a|es)?|Sr\.?a?|Sres?\.?)\b', re.I), 5),
        # Despedidas formales
        (re.compile(r'\b(?:sin\s+(?:otro|m[aá]s)\s+particular|quedo\s+(?:de\s+usted|atent[oa]))\b', re.I), 9),
        (re.compile(r'\b(?:agradezco\s+(?:su|de\s+antemano)|agradeciendo)\b', re.I), 7),
        (re.compile(r'\b(?:reciba\s+un\s+(?:cordial|atento)\s+saludo)\b', re.I), 9),
        # Estructura de carta
        (re.compile(r'\b(?:por\s+medio\s+de\s+la\s+presente|mediante\s+la\s+presente)\b', re.I), 10),
        (re.compile(r'\b(?:hago\s+constar|certifico|constancia)\b', re.I), 8),
        (re.compile(r'\b(?:remitente|destinatario)\b', re.I), 7),
        (re.compile(r'\b(?:carta\s+de\s+(?:renuncia|recomendaci[oó]n|presentaci[oó]n|poder|aceptaci[oó]n))\b', re.I), 10),
    ],
    "formulario": [
        (re.compile(r'\bformulario\b', re.I), 10),
        (re.compile(r'\bsolicitante\b', re.I), 7),
        (re.compile(r'\bc[eé]dula\s+de\s+identidad\b', re.I), 8),
        (re.compile(r'\bfecha\s+de\s+nacimiento\b', re.I), 7),
        (re.compile(r'\bmarque\s+con\b', re.I), 7),
        (re.compile(r'\bsolicitud\b', re.I), 6),
        (re.compile(r'\bcampo\s+obligatorio\b', re.I), 8),
        (re.compile(r'\brellene?\b', re.I), 6),
        (re.compile(r'\bnombre\s+completo\b', re.I), 5),
        (re.compile(r'\bdirecci[oó]n\b.*\btel[eé]fono\b', re.I | re.S), 6),
        # Campos típicos de formulario
        (re.compile(r'\b(?:estado\s+civil|profesi[oó]n|ocupaci[oó]n)\b', re.I), 7),
        (re.compile(r'\b(?:firma\s+del\s+(?:solicitante|interesado|titular))\b', re.I), 8),
        (re.compile(r'\b(?:adjuntar|anexar|documentos?\s+requeridos?)\b', re.I), 6),
        (re.compile(r'\b(?:uso\s+(?:oficial|interno)|no\s+escribir?\s+aqu[ií])\b', re.I), 8),
        (re.compile(r'\b(?:llenar|completar|diligenciar)\b', re.I), 6),
        (re.compile(r'\b(?:en\s+letra\s+(?:clara|imprenta|legible))\b', re.I), 7),
        # Campos de formularios digitales/empresariales (alta especificidad)
        (re.compile(r'\b(?:enviar|submit)\b', re.I), 8),
        (re.compile(r'\bnombre\s+de\s+la\s+compa[ñn][ií]a\b', re.I), 10),
        (re.compile(r'\bpersona\s+de\s+contacto\b', re.I), 9),
        (re.compile(r'\bc[oó]digo\s+postal\b', re.I), 8),
        (re.compile(r'\bestado\s*/\s*provincia\b|\bprovincia\b', re.I), 6),
        # Pares típicos de formulario (nombre + apellido como etiquetas separadas)
        (re.compile(r'\bnombre\b.*\bapellido\b', re.I | re.S), 7),
        # Placeholders de ejemplo → formulario digital vacío
        (re.compile(r'ejemplo@\w+\.\w+|example@', re.I), 7),
        (re.compile(r'\(000\)\s*000[-\s]?0000|\(0+\)', re.I), 5),
        # Instrucción explícita de validación de campo
        (re.compile(r'\bintroducid?\s+un\s+n[uú]mero\s+v[aá]lido\b', re.I), 8),
        (re.compile(r'\bpor\s+favor\s+(?:complete|llene|ingrese|introduzca)\b', re.I), 7),
        # Negativos: contenido conversacional/emocional → no es formulario
        (re.compile(r'\b(?:créeme|te\s+quiero|hablemos|infidelidad|ansiosos?|iniciativa)\b', re.I), -15),
        (re.compile(r'\b(?:demasiado\s+(?:viejo|vieja|joven)|seminueva|preanciana|chismes)\b', re.I), -15),
        (re.compile(r'["«»""].{10,}["«»""]', re.I), -10),  # texto entre comillas largas → cita
        (re.compile(r'\d{1,2}:\d{2}\s*(?:AM|PM|a\.m\.|p\.m\.)', re.I), -12),  # timestamp de chat
    ],
    "contrato": [
        (re.compile(r'\bcontrato\b', re.I), 10),
        (re.compile(r'\bcontrato\s+de\s+(?:trabajo|arrendamiento|servicios|compraventa|prestaci[oó]n)\b', re.I), 12),
        (re.compile(r'\bcl[aá]usula\b', re.I), 9),
        (re.compile(r'\blas\s+partes\b', re.I), 7),
        (re.compile(r'\bfirmante\b', re.I), 7),
        (re.compile(r'\bvigencia\b', re.I), 6),
        (re.compile(r'\bcompromiso\b', re.I), 5),
        # Legal
        (re.compile(r'\b(?:primera|segunda|tercera|cuarta|quinta)\s*[:\.]\s', re.I), 6),
        (re.compile(r'\b(?:arrendador|arrendatario|empleador|empleado|contratante|contratista)\b', re.I), 8),
        (re.compile(r'\b(?:obligaciones|derechos|responsabilidades)\b', re.I), 6),
        (re.compile(r'\b(?:penalidad|indemnizaci[oó]n|rescisi[oó]n|terminaci[oó]n)\b', re.I), 7),
        (re.compile(r'\b(?:de\s+com[uú]n\s+acuerdo|mutuo\s+consentimiento)\b', re.I), 8),
        (re.compile(r'\b(?:conste\s+por\s+el\s+presente|celebran\s+el\s+siguiente)\b', re.I), 9),
        (re.compile(r'\b(?:testigo|notario|fedatario|fe\s+p[uú]blica)\b', re.I), 7),
    ],
    "hoja_de_vida": [
        (re.compile(r'\b(?:curr[ií]cul[ou]m?\s*(?:vitae)?|hoja\s+de\s+vida|resume|CV)\b', re.I), 10),
        (re.compile(r'\bexperiencia\s+(?:laboral|profesional)\b', re.I), 10),
        (re.compile(r'\bhist(?:orial)?\s+(?:laboral|profesional)\b', re.I), 14),  # "HISTORIAL LABORAL"
        (re.compile(r'\bresumen\s+profesional\b', re.I), 14),                      # "RESUMEN PROFESIONAL"
        (re.compile(r'\beducaci[oó]n\b|\bformaci[oó]n\s+acad[eé]mica\b', re.I), 7),
        (re.compile(r'\bhabilidades\b|\bcompetencias\b', re.I), 6),
        (re.compile(r'\baptitudes?\b', re.I), 7),
        (re.compile(r'\breferencias\b', re.I), 5),
        # Secciones típicas de CV
        (re.compile(r'\b(?:datos?\s+personales?|informaci[oó]n\s+personal)\b', re.I), 7),
        (re.compile(r'\b(?:idiomas?|languages?)\b', re.I), 5),
        (re.compile(r'\b(?:objetivo\s+(?:profesional|laboral)|perfil\s+profesional)\b', re.I), 9),
        (re.compile(r'\b(?:logros?\s+(?:profesionales?|acad[eé]micos?))\b', re.I), 8),
        (re.compile(r'\b(?:certificaciones?|cursos?\s+(?:adicionales?|complementarios?))\b', re.I), 6),
        (re.compile(r'\b(?:disponibilidad|pretensiones?\s+salariales?|pretensi[oó]n)\b', re.I), 7),
        (re.compile(r'\binformaci[oó]n\s+adicional\b', re.I), 8),
        # Patrones de experiencia con fechas (empresa 2020-2025)
        (re.compile(r'\b(?:cargo|puesto|posici[oó]n)\s*:', re.I), 6),
        (re.compile(r'\b(?:empresa|compa[ñn][ií]a|organizaci[oó]n)\s*:', re.I), 5),
        (re.compile(r'\b\d{4}\s*[-–]\s*(?:\d{4}|actual|presente)\b', re.I), 6),
        # Combinación de 2+ secciones típicas de CV en el mismo texto = muy probable CV
        (re.compile(r'(?=.*\b(?:aptitudes?|habilidades?)\b)(?=.*\b(?:historial|experiencia)\b)', re.I | re.S), 12),
        (re.compile(r'(?=.*\b(?:resumen\s+profesional|perfil\s+profesional)\b)(?=.*\b(?:contacto|tel[eé]fono)\b)', re.I | re.S), 10),
    ],
    "informe": [
        (re.compile(r'\binforme\b', re.I), 10),
        (re.compile(r'\binforme\s+(?:de\s+)?(?:gesti[oó]n|t[eé]cnico|financiero|auditor[ií]a|avance|final)\b', re.I), 12),
        (re.compile(r'\bconclusiones\b', re.I), 7),
        (re.compile(r'\brecomendaciones\b', re.I), 7),
        (re.compile(r'\bintroducci[oó]n\b', re.I), 5),
        (re.compile(r'\bmetodolog[ií]a\b', re.I), 7),
        (re.compile(r'\bresultados\b', re.I), 5),
        # Estructura académica / técnica
        (re.compile(r'\b(?:resumen\s+ejecutivo|abstract|summary)\b', re.I), 8),
        (re.compile(r'\b(?:marco\s+te[oó]rico|antecedentes|justificaci[oó]n)\b', re.I), 7),
        (re.compile(r'\b(?:objetivos?\s+(?:general(?:es)?|espec[ií]ficos?))\b', re.I), 8),
        (re.compile(r'\b(?:bibliograf[ií]a|referencias\s+bibliogr[aá]ficas|fuentes)\b', re.I), 6),
        (re.compile(r'\b(?:anexos?|ap[eé]ndices?)\b', re.I), 5),
        (re.compile(r'\b(?:tabla\s+\d+|figura\s+\d+|gr[aá]fico\s+\d+)\b', re.I), 6),
        (re.compile(r'\b(?:elaborado\s+por|preparado\s+por|autor)\b', re.I), 6),
    ],
    "documento_informativo": [
        (re.compile(r'\bart[íi]culo\s+\d+\b', re.I), 7),
        (re.compile(r'\bcomunicado\b', re.I), 8),
        (re.compile(r'\bcomunicado\s+(?:de\s+prensa|oficial|p[uú]blico)\b', re.I), 10),
        (re.compile(r'\bresoluci[óo]n\b', re.I), 6),
        (re.compile(r'\bregulaci[óo]n\b|\bnorma\b', re.I), 5),
        # Documentos oficiales / legales
        (re.compile(r'\b(?:decreto|ley|acuerdo|circular|memorando|memo)\b', re.I), 8),
        (re.compile(r'\b(?:disposici[oó]n|reglamento|estatuto|ordenanza)\b', re.I), 7),
        (re.compile(r'\b(?:bolet[ií]n|gaceta|diario\s+oficial)\b', re.I), 8),
        (re.compile(r'\b(?:publicado|emitido|expedido|promulgado)\b', re.I), 5),
        (re.compile(r'\b(?:considerando|por\s+(?:cuanto|tanto)|decreta|resuelve)\b', re.I), 8),
    ],
    "noticia": [
        (re.compile(r'\b(?:noticias?|news|breaking)\b', re.I), 9),
        (re.compile(r'\b(?:redacci[oó]n|periodista|reporter|corresponsal)\b', re.I), 7),
        (re.compile(r'\b(?:fuente|Reuters|AP|AFP|EFE|CNN|BBC|El\s+Pa[ií]s|El\s+Tiempo)\b', re.I), 8),
        (re.compile(r'\b(?:leer\s+m[aá]s|read\s+more|ver\s+m[aá]s)\b', re.I), 8),
        (re.compile(r'\b(?:exclusiv[oa]|[uú]ltima\s+hora|urgente|breaking\s+news)\b', re.I), 9),
        # Medios / publicaciones
        (re.compile(r'\b(?:editorial|columna|opini[oó]n|cr[oó]nica|reportaje)\b', re.I), 7),
        (re.compile(r'\b(?:foto\s*:?\s*\w+|imagen\s*:?\s*\w+|cr[eé]dito)\b', re.I), 5),
        (re.compile(r'\b(?:publicado\s+(?:el|hace)|actualizado\s+(?:el|hace))\b', re.I), 6),
        (re.compile(r'\b(?:compartir|enviar|guardar|imprimir)\b.*(?:noticia|art[ií]culo)', re.I | re.S), 6),
        (re.compile(r'\b(?:suscr[ií]b[ae](?:se|te)?|premium|paywall)\b', re.I), 6),
    ],
    "correo": [
        (re.compile(r'\b(?:bandeja\s+de\s+entrada|inbox)\b', re.I), 10),
        (re.compile(r'\b(?:asunto|subject)\s*:', re.I), 9),
        (re.compile(r'\b(?:de|from)\s*:\s*\S+@\S+', re.I), 10),
        (re.compile(r'\b(?:responder|reply|reenviar|forward)\b', re.I), 7),
        (re.compile(r'\b(?:Gmail|Outlook|Yahoo\s*Mail|ProtonMail|Thunderbird)\b', re.I), 8),
        # Campos de correo
        (re.compile(r'\b(?:para|to)\s*:\s*\S+@\S+', re.I), 9),
        (re.compile(r'\b(?:CC|CCO|BCC)\s*:', re.I), 8),
        (re.compile(r'\b(?:adjunto|attachment|archivo\s+adjunto)\b', re.I), 7),
        (re.compile(r'\b(?:responder\s+a\s+todos|reply\s+all)\b', re.I), 8),
        (re.compile(r'\b(?:no\s+le[ií]do|unread|importante|starred|destacado)\b', re.I), 6),
        (re.compile(r'\b(?:spam|correo\s+no\s+deseado|junk)\b', re.I), 7),
        (re.compile(r'\b(?:redactar|compose|nuevo\s+(?:correo|mensaje))\b', re.I), 7),
    ],
    "presentacion": [
        (re.compile(r'\b(?:diapositiva|slide)\s*\d*', re.I), 10),
        (re.compile(r'\b(?:presentaci[oó]n|presentation)\b', re.I), 9),
        (re.compile(r'\b\d+\s*/\s*\d+\b', re.I), 4),
        (re.compile(r'\b(?:PowerPoint|Google\s+Slides|Keynote|Canva|Prezi)\b', re.I), 8),
        # Estructura de presentación
        (re.compile(r'\b(?:agenda|contenido|[ií]ndice|temario)\b', re.I), 5),
        (re.compile(r'\b(?:ponente|expositor|conferencista|speaker)\b', re.I), 7),
        (re.compile(r'\b(?:preguntas|Q&A|gracias\s+por\s+su\s+atenci[oó]n)\b', re.I), 7),
    ],
    "etiqueta": [
        # Marcas de productos de cuidado personal / cosméticos
        (re.compile(r'\b(?:La\s+Roche[- ]Posay|Cetaphil|Neutrogena|Av[eè]ne|Vichy|CeraVe|Bioderma|Nivea|Eucerin|Garnier|L\'Or[eé]al)\b', re.I), 14),
        (re.compile(r'\b(?:[Áa]cido\s+[Ss]alic[íi]lico|[Áa]cido\s+[Gg]lic[oó]lico|[Rr]etinol|[Nn]iacinamida|[Hh]ialur[oó]nico)\b', re.I), 12),
        (re.compile(r'\b(?:oil\s+control|matificante|hidratante\s+facial|antimanchas|anti[\s-]imperfec|triple\s+correction)\b', re.I), 11),
        (re.compile(r'\b(?:pieles?\s+(?:mixtas?|grasas?|secas?|sensibles?)|barros|espinillas|acné)\b', re.I), 8),
        (re.compile(r'\b(?:dermatol[oó]gico|probado\s+(?:dermatol[oó]gica|cl[íi]nicamente)|sin\s+parabenos)\b', re.I), 9),
        (re.compile(r'\b(?:EFFACLAR|EFFIDERM|TOLERIANE|ANTHELIOS|CICAPLAST)\b', re.I), 12),
        (re.compile(r'\b\d+\s*(?:ml|mL|g|mg|oz|fl\.?\s*oz)\b', re.I), 7),
        (re.compile(r'\b(?:uso\s+externo|solo\s+para\s+uso\s+externo|external\s+use\s+only)\b', re.I), 10),
        (re.compile(r'\bingredientes\b', re.I), 10),
        (re.compile(r'\bvencimiento\b', re.I), 8),
        (re.compile(r'\bpeso\b.*\bneto\b', re.I), 9),
        (re.compile(r'\bfabricado\b', re.I), 7),
        (re.compile(r'\b(?:lote|batch|registro\s+sanitario|INVIMA|FDA|ANMAT|COFEPRIS)\b', re.I), 9),
        (re.compile(r'\b(?:contiene|puede\s+contener|may\s+contain|al[eé]rgenos?|allergens?)\b', re.I), 8),
        (re.compile(r'\b(?:conservar|almacenar|store|refrigerar)\b', re.I), 6),
        # Datos de fabricación
        (re.compile(r'\b(?:fabricado\s+(?:en|por)|hecho\s+en|made\s+in|elaborado\s+(?:en|por))\b', re.I), 8),
        (re.compile(r'\b(?:distribuido\s+por|importado\s+por|envasado\s+(?:en|por))\b', re.I), 7),
        (re.compile(r'\b(?:fecha\s+de\s+(?:elaboraci[oó]n|fabricaci[oó]n|producci[oó]n))\b', re.I), 7),
        (re.compile(r'\b(?:contenido\s+neto|net\s+(?:weight|content|wt))\b', re.I), 8),
        # Advertencias de etiqueta
        (re.compile(r'\b(?:mantener\s+(?:refrigerado|en\s+lugar\s+(?:fresco|seco)))\b', re.I), 7),
        (re.compile(r'\b(?:una\s+vez\s+abierto|consumir\s+(?:antes|preferentemente))\b', re.I), 7),
        (re.compile(r'\b(?:libre\s+de\s+(?:gluten|lactosa|az[uú]car)|sin\s+(?:gluten|TACC|conservantes))\b', re.I), 7),
        (re.compile(r'\b(?:org[aá]nico|natural|vegano|kosher|halal)\b', re.I), 5),
    ],
    "tarjeta": [
        (re.compile(r'\b(?:director|gerente|coordinador|especialista|jefe|presidente|CEO|CTO|CFO)\b', re.I), 7),
        (re.compile(r'\bdepartamento\b', re.I), 6),
        (re.compile(r'\bsucursal\b', re.I), 5),
        # Datos de contacto (combo típico de tarjeta de presentación)
        (re.compile(r'\b(?:tel[eé]fono|phone|cel(?:ular)?|mobile|fax)\s*:?\s*[\+\d\(\)]', re.I), 7),
        (re.compile(r'\b\w+@\w+\.\w+\b', re.I), 5),
        (re.compile(r'\b(?:www\.|http|\.com|\.org|\.net)\b', re.I), 5),
        (re.compile(r'\b(?:tarjeta\s+de\s+presentaci[oó]n|business\s+card)\b', re.I), 10),
        (re.compile(r'\b(?:asesor|consultor|representante|vendedor|ejecutivo|analista)\b', re.I), 6),
        # Penalizaciones: secciones de CV que NO aparecen en tarjetas de presentación
        (re.compile(r'\b(?:resumen\s+profesional|historial\s+laboral|experiencia\s+(?:laboral|profesional))\b', re.I), -18),
        (re.compile(r'\b(?:aptitudes?|habilidades?|competencias?)\b', re.I), -10),
        (re.compile(r'\binformaci[oó]n\s+adicional\b', re.I), -8),
        (re.compile(r'\bformaci[oó]n\b|\beducaci[oó]n\b', re.I), -8),
        (re.compile(r'\b\d{4}\s*[-–]\s*(?:\d{4}|actual|presente)\b', re.I), -8),  # fechas de empleo
    ],
    "receta_medica": [
        # Palabra clave directa
        (re.compile(r'\b(?:receta|prescripci[oó]n|prescription|Rx)\b', re.I), 10),
        (re.compile(r'\b(?:receta\s+m[eé]dica|orden\s+m[eé]dica)\b', re.I), 12),
        # Medicamentos y formas farmacéuticas
        (re.compile(r'\b(?:medicamento|medicina|medication|medicine|f[aá]rmaco)\b', re.I), 8),
        (re.compile(r'\b(?:tabletas?|c[aá]psulas?|comprimidos?|grageas?|pastillas?)\b', re.I), 9),
        (re.compile(r'\b(?:gotas?|jarabe|susp(?:ensi[oó]n)?|soluci[oó]n|crema|ung[üu]ento|pomada|gel)\b', re.I), 8),
        (re.compile(r'\b(?:ampolla|inyecci[oó]n|supositorio|parche|inhalador|spray\s+nasal)\b', re.I), 8),
        # Dosis y posología
        (re.compile(r'\b(?:dosis|dosage|posolog[ií]a)\b', re.I), 9),
        (re.compile(r'\b\d+\s*mg\b', re.I), 8),
        (re.compile(r'\b(?:cada\s+\d+\s+horas?|every\s+\d+\s+hours?)\b', re.I), 10),
        (re.compile(r'\b(?:cada\s+\d+\s+d[ií]as?)\b', re.I), 8),
        (re.compile(r'\b(?:una\s+vez\s+al\s+d[ií]a|dos\s+veces|tres\s+veces)\b', re.I), 8),
        (re.compile(r'\b(?:en\s+ayunas|antes\s+de\s+(?:comer|dormir)|despu[eé]s\s+de\s+(?:comer|las\s+comidas))\b', re.I), 8),
        (re.compile(r'\b(?:durante\s+\d+\s+d[ií]as?|por\s+\d+\s+d[ií]as?)\b', re.I), 7),
        # Vía de administración
        (re.compile(r'\b(?:v[ií]a\s+(?:oral|t[oó]pica|intramuscular|intravenosa|subling[uü]al|rectal|nasal))\b', re.I), 8),
        (re.compile(r'\b(?:oral|t[oó]pico|inyectable|subling[uü]al)\b', re.I), 6),
        # Profesional médico
        (re.compile(r'\b(?:Dr\.?|Dra\.?|m[eé]dico|doctor|doctora)\b', re.I), 6),
        (re.compile(r'\b(?:M\.?P\.?\s*\d+|matr[ií]cula|colegiado|registro\s+m[eé]dico)\b', re.I), 8),
        (re.compile(r'\b(?:especialista\s+en|especialidad)\b', re.I), 5),
        # Farmacia / despacho
        (re.compile(r'\b(?:farmacia|pharmacy|drogu[eé]r[ií]a|botica)\b', re.I), 7),
        (re.compile(r'\b(?:despachar|dispensar|surtir)\b', re.I), 7),
        (re.compile(r'\b(?:paciente|patient)\b', re.I), 5),
        # Diagnóstico
        (re.compile(r'\b(?:diagn[oó]stico|diagnosis|CIE[-\s]?\d+|ICD)\b', re.I), 8),
        # Medicamentos comunes (Latinoamérica)
        (re.compile(r'\b(?:acetaminof[eé]n|ibuprofeno|amoxicilina|omeprazol|metformina|losart[aá]n)\b', re.I), 8),
        (re.compile(r'\b(?:diclofenaco|naproxeno|azitromicina|ciprofloxacin[oa]|prednisona)\b', re.I), 8),
        (re.compile(r'\b(?:loratadina|cetirizina|atorvastatina|amlodipino|metoprolol)\b', re.I), 8),
    ],
    "boleto": [
        # Palabra clave directa
        (re.compile(r'\b(?:boleto|ticket|boarding\s+pass|pase\s+de\s+abordar|tarjeta\s+de\s+embarque)\b', re.I), 10),
        (re.compile(r'\b(?:e-?ticket|boleto\s+electr[oó]nico|pase\s+electr[oó]nico)\b', re.I), 10),
        # Transporte aéreo
        (re.compile(r'\b(?:vuelo|flight)\s*(?:No\.?|#)?\s*[A-Z]{0,2}\d+', re.I), 10),
        (re.compile(r'\b(?:asiento|seat)\s*:?\s*\d+[A-F]?\b', re.I), 9),
        (re.compile(r'\b(?:puerta|gate)\s*:?\s*[A-Z]?\d+', re.I), 8),
        (re.compile(r'\b(?:terminal)\s*:?\s*\d', re.I), 7),
        (re.compile(r'\b(?:clase\s+(?:econ[oó]mica|ejecutiva|primera|business|economy))\b', re.I), 8),
        (re.compile(r'\b(?:equipaje|baggage|maleta|bulto)\b', re.I), 6),
        # Transporte terrestre
        (re.compile(r'\b(?:tren|train|bus|aut[oó]bus|metro|ferrocarril)\b', re.I), 7),
        (re.compile(r'\b(?:ruta|route|and[eé]n|plataforma|platform|parada|estaci[oó]n)\b', re.I), 7),
        (re.compile(r'\b(?:vag[oó]n|coche|wagon|car)\b', re.I), 6),
        # Datos de viaje
        (re.compile(r'\b(?:salida|departure|partida)\b', re.I), 7),
        (re.compile(r'\b(?:llegada|arrival|destino|destination)\b', re.I), 7),
        (re.compile(r'\b(?:pasajero|passenger|viajero|titular)\b', re.I), 8),
        (re.compile(r'\b(?:origen|procedencia)\b', re.I), 5),
        (re.compile(r'\b(?:hora\s+de\s+(?:embarque|abordaje|salida))\b', re.I), 9),
        # Eventos / entretenimiento
        (re.compile(r'\b(?:evento|concierto|cine|teatro|funci[oó]n|espect[aá]culo|show)\b', re.I), 8),
        (re.compile(r'\b(?:fila|row|secci[oó]n|section|zona|localidad|tribuna|palco)\b', re.I), 6),
        (re.compile(r'\b(?:entrada\s+(?:general|VIP|preferencial|adulto|ni[ñn]o))\b', re.I), 8),
        # Códigos
        (re.compile(r'\b(?:QR|c[oó]digo\s+de\s+barras|barcode)\b', re.I), 6),
        (re.compile(r'\b(?:PNR|localizador|confirmaci[oó]n|booking\s+ref)\b', re.I), 9),
        # Aerolíneas
        (re.compile(r'\b(?:Avianca|LATAM|Copa|Wingo|JetBlue|American|Delta|United|Volaris|VivaAerobus)\b', re.I), 8),
    ],
    "identificacion": [
        # Tipo de documento
        (re.compile(r'\b(?:c[eé]dula|licencia|DNI|pasaporte|passport)\b', re.I), 10),
        (re.compile(r'\b(?:c[eé]dula\s+de\s+(?:ciudadan[ií]a|identidad|extranjer[ií]a))\b', re.I), 12),
        (re.compile(r'\b(?:documento\s+(?:nacional\s+)?de\s+identidad|identification|ID\s+card)\b', re.I), 10),
        (re.compile(r'\b(?:tarjeta\s+de\s+identidad|carnet\s+de\s+identidad)\b', re.I), 10),
        (re.compile(r'\b(?:licencia\s+de\s+(?:conducir|conducci[oó]n)|driver\'?s?\s+licen[cs]e)\b', re.I), 10),
        (re.compile(r'\b(?:pasaporte\s+(?:electr[oó]nico|biom[eé]trico))\b', re.I), 10),
        (re.compile(r'\b(?:permiso\s+de\s+(?:residencia|trabajo)|visa|green\s+card)\b', re.I), 9),
        (re.compile(r'\b(?:credencial\s+(?:de\s+elector|INE|para\s+votar))\b', re.I), 10),
        # Datos personales del documento
        (re.compile(r'\b(?:fecha\s+de\s+nacimiento|date\s+of\s+birth|F\.?\s*(?:de\s+)?Nac\.?)\b', re.I), 8),
        (re.compile(r'\b(?:lugar\s+de\s+nacimiento|place\s+of\s+birth)\b', re.I), 8),
        (re.compile(r'\b(?:nacionalidad|nationality|ciudadan[ií]a|citizenship)\b', re.I), 7),
        (re.compile(r'\b(?:sexo|sex|g[eé]nero|gender)\s*:?\s*(?:M|F|masculino|femenino|male|female)\b', re.I), 8),
        (re.compile(r'\b(?:estatura|altura|height|peso|weight)\s*:?\s*\d', re.I), 6),
        (re.compile(r'\b(?:tipo\s+de\s+sangre|blood\s+type|grupo\s+sangu[ií]neo|RH)\b', re.I), 7),
        (re.compile(r'\b(?:color\s+de\s+(?:ojos|pelo|cabello)|eye\s+color|hair)\b', re.I), 6),
        # Fechas del documento
        (re.compile(r'\b(?:fecha\s+de\s+(?:expedici[oó]n|emisi[oó]n|issue))\b', re.I), 7),
        (re.compile(r'\b(?:fecha\s+de\s+(?:expiraci[oó]n|vencimiento|caducidad|expiry))\b', re.I), 7),
        (re.compile(r'\b(?:v[aá]lido\s+hasta|valid\s+(?:until|thru|through))\b', re.I), 7),
        # Autoridad emisora
        (re.compile(r'\b(?:rep[uú]blica|estado|gobierno|ministerio)\b', re.I), 5),
        (re.compile(r'\b(?:registradur[ií]a|RENIEC|INE|ONPE|TSE|CNE|SERECI)\b', re.I), 9),
        (re.compile(r'\b(?:autoridad\s+(?:emisora|competente)|issuing\s+authority)\b', re.I), 7),
        # Seguridad
        (re.compile(r'\b(?:huella|fingerprint|firma\s+digital|biom[eé]tric[oa])\b', re.I), 6),
        (re.compile(r'\b(?:MRZ|zona\s+de\s+lectura\s+mec[aá]nica)\b', re.I), 8),
    ],
    "horario": [
        # Palabra clave directa
        (re.compile(r'\b(?:horario|schedule|agenda|itinerario|cronograma|planeaci[oó]n)\b', re.I), 10),
        (re.compile(r'\b(?:horario\s+de\s+(?:clases|trabajo|atenci[oó]n|servicio|vuelos|trenes|buses))\b', re.I), 12),
        # Días de la semana (múltiples matches = señal fuerte)
        (re.compile(r'\b(?:lunes|monday|Mon)\b', re.I), 7),
        (re.compile(r'\b(?:martes|tuesday|Tue)\b', re.I), 7),
        (re.compile(r'\b(?:mi[eé]rcoles|wednesday|Wed)\b', re.I), 7),
        (re.compile(r'\b(?:jueves|thursday|Thu)\b', re.I), 7),
        (re.compile(r'\b(?:viernes|friday|Fri)\b', re.I), 7),
        (re.compile(r'\b(?:s[aá]bado|saturday|Sat)\b', re.I), 7),
        (re.compile(r'\b(?:domingo|sunday|Sun)\b', re.I), 7),
        # Abreviaciones de días
        (re.compile(r'\b(?:Lun|Mar|Mi[eé]|Jue|Vie|S[aá]b|Dom)\b', re.I), 6),
        (re.compile(r'\b(?:L|M|X|J|V|S|D)\s*[-–/]\s*(?:L|M|X|J|V|S|D)\b', re.I), 6),
        # Rangos de hora
        (re.compile(r'\b\d{1,2}:\d{2}\s*[-–a]\s*\d{1,2}:\d{2}\b', re.I), 9),
        (re.compile(r'\b(?:de\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\s*(?:a|[-–])\s*\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b', re.I), 9),
        # Citas / reuniones
        (re.compile(r'\b(?:cita|appointment|reuni[oó]n|meeting|sesi[oó]n|consulta)\b', re.I), 7),
        (re.compile(r'\b(?:cita\s+(?:m[eé]dica|odontol[oó]gica|legal|con\s+\w+))\b', re.I), 9),
        # Académico
        (re.compile(r'\b(?:clase|materia|asignatura|curso|taller|laboratorio|pr[aá]ctica)\b', re.I), 6),
        (re.compile(r'\b(?:per[ií]odo|trimestre|semestre|bimestre|cuatrimestre)\b', re.I), 6),
        (re.compile(r'\b(?:profesor|teacher|docente|instructor)\b', re.I), 5),
        # Espacio
        (re.compile(r'\b(?:aula|sal[oó]n|room|bloque|edificio|piso|oficina)\b', re.I), 5),
        # Temporal
        (re.compile(r'\b(?:semana|week|mensual|monthly|diario|daily|semanal|weekly)\b', re.I), 6),
        (re.compile(r'\b(?:turno\s+(?:ma[ñn]ana|tarde|noche)|jornada)\b', re.I), 7),
        (re.compile(r'\b(?:hora\s+de\s+(?:entrada|salida|almuerzo|descanso|inicio|fin))\b', re.I), 8),
        # Negativos: productos de skincare/cosmética → etiqueta, no horario
        (re.compile(r'\b(?:La\s+Roche[- ]Posay|Cetaphil|Neutrogena|Av[eè]ne|Vichy|CeraVe|Bioderma|Nivea|Eucerin)\b', re.I), -20),
        (re.compile(r'\b(?:[Áa]cido\s+[Ss]alic[íi]lico|hidratante\s+facial|oil\s+control|matificante|antimanchas)\b', re.I), -18),
        (re.compile(r'\b(?:uso\s+externo|pieles?\s+(?:mixtas?|grasas?|secas?)|barros|espinillas)\b', re.I), -15),
    ],
    "instrucciones": [
        # Palabra clave directa
        (re.compile(r'\b(?:instrucciones|instructions|modo\s+de\s+(?:uso|empleo)|forma\s+de\s+uso)\b', re.I), 10),
        (re.compile(r'\b(?:instrucciones\s+de\s+(?:uso|montaje|instalaci[oó]n|lavado|cuidado))\b', re.I), 12),
        (re.compile(r'\b(?:paso\s+\d+|step\s+\d+)\b', re.I), 10),
        # Secuencia: "Primero", "Luego", "Después", "Finalmente"
        (re.compile(r'\b(?:primero|luego|despu[eé]s|seguidamente|finalmente|por\s+[uú]ltimo)\b', re.I), 8),
        (re.compile(r'\b(?:a\s+continuaci[oó]n|next|then|finally|afterwards)\b', re.I), 6),
        (re.compile(r'\b(?:en\s+(?:primer|segundo|tercer)\s+lugar)\b', re.I), 8),
        # Advertencias y precauciones
        (re.compile(r'\b(?:precauci[oó]n|advertencia|warning|caution|peligro|danger)\b', re.I), 8),
        (re.compile(r'\b(?:no\s+(?:usar|ingerir|mezclar|exponer|tocar|sumergir)|do\s+not)\b', re.I), 8),
        (re.compile(r'\b(?:antes\s+de\s+(?:usar|comenzar|iniciar|empezar)|before\s+use)\b', re.I), 7),
        (re.compile(r'\b(?:mantener\s+fuera|keep\s+(?:away|out))\b', re.I), 7),
        (re.compile(r'\b(?:solo\s+para\s+uso\s+(?:externo|profesional)|for\s+external\s+use)\b', re.I), 7),
        # Verbos infinitivos de instrucciones
        (re.compile(r'\b(?:aplicar|mezclar|agitar|disolver|calentar|refrigerar|limpiar|secar|lavar)\b', re.I), 6),
        (re.compile(r'\b(?:abrir|conectar|presionar|pulsar|encender|apagar|insertar|retirar|colocar)\b', re.I), 5),
        (re.compile(r'\b(?:ensamblar|armar|montar|instalar|desmontar|atornillar|ajustar|nivelar)\b', re.I), 7),
        (re.compile(r'\b(?:esperar|aguardar|dejar\s+reposar|dejar\s+secar|dejar\s+enfriar)\b', re.I), 5),
        # Verbos imperativos CONJUGADOS (2da persona singular: tú)
        (re.compile(r'\b(?:aplica|mezcla|agita|disuelve|calienta|vierte|remueve|enjuaga|peina|reserva)\b', re.I), 7),
        (re.compile(r'\b(?:coloca|retira|inserta|presiona|conecta|abre|cierra|enciende|apaga)\b', re.I), 6),
        (re.compile(r'\b(?:deja|espera|repite|verifica|aseg[uú]rate|mide|corta|dobla|pega|seca|limpia|lava)\b', re.I), 6),
        (re.compile(r'\b(?:gira|empuja|jala|sujeta|sostiene?|levanta|baja|enrolla|desenrolla)\b', re.I), 6),
        # Verbos imperativos CONJUGADOS (usted: formal)
        (re.compile(r'\b(?:aplique|mezcle|agite|caliente|vierta|remueva|enjuague|coloque|retire)\b', re.I), 7),
        (re.compile(r'\b(?:inserte|presione|conecte|abra|cierre|encienda|apague|verifique)\b', re.I), 6),
        # Guías / tutoriales
        (re.compile(r'\b(?:c[oó]mo\s+(?:usar|hacer|preparar|instalar|configurar|armar|montar|limpiar|reparar))\b', re.I), 10),
        (re.compile(r'\b(?:gu[ií]a|guide|manual|tutorial)\b', re.I), 9),
        (re.compile(r'\b(?:gu[ií]a\s+(?:r[aá]pida|de\s+inicio|de\s+usuario|de\s+instalaci[oó]n))\b', re.I), 12),
        # Contexto de instrucciones
        (re.compile(r'\b(?:importante|nota|atenci[oó]n|tip|consejo|sugerencia|recomendaci[oó]n)\b', re.I), 5),
        (re.compile(r'\b(?:con\s+abundante|en\s+un\s+recipiente|hasta\s+obtener|seg[uú]n\s+(?:la|el|tu|las))\b', re.I), 6),
        (re.compile(r'\b\d+\s*(?:ml|gr?|oz|litros?|partes?|gotas?|cucharadas?|tazas?)\b', re.I), 6),
        (re.compile(r'\b(?:aplicaciones?\s+posteriores?|reci[eé]n\s+lavado|de\s+costumbre)\b', re.I), 5),
        # Recetas de cocina (subtipo de instrucciones)
        (re.compile(r'\b(?:preparaci[oó]n|elaboraci[oó]n|procedimiento)\s*:', re.I), 7),
        (re.compile(r'\b(?:hornear|hervir|freír|saltear|batir|amasar|rallar|picar|licuar)\b', re.I), 6),
        (re.compile(r'\b(?:hornea|hierve|fr[ií]e|saltea|bate|amasa|ralla|pica|licua)\b', re.I), 7),
    ],
    "resultado_lab": [
        # Palabra clave directa
        (re.compile(r'\b(?:resultado(?:s)?\s+de\s+(?:laboratorio|ex[aá]men(?:es)?|sangre|orina|heces))\b', re.I), 12),
        (re.compile(r'\b(?:laboratorio|laboratory|lab)\b', re.I), 8),
        (re.compile(r'\b(?:laboratorio\s+cl[ií]nico|clinical\s+lab(?:oratory)?)\b', re.I), 10),
        # Hemograma / hematología
        (re.compile(r'\b(?:hemograma|hematolog[ií]a|blood\s+count|CBC)\b', re.I), 12),
        (re.compile(r'\b(?:hemoglobina|hemoglobin|Hb|Hgb)\b', re.I), 10),
        (re.compile(r'\b(?:hematocrito|hematocrit|Hct)\b', re.I), 10),
        (re.compile(r'\b(?:plaquetas|platelets|thrombocytes)\b', re.I), 10),
        (re.compile(r'\b(?:leucocitos|leukocytes|WBC|gl[oó]bulos\s+blancos)\b', re.I), 10),
        (re.compile(r'\b(?:eritrocitos|erythrocytes|RBC|gl[oó]bulos\s+rojos)\b', re.I), 10),
        (re.compile(r'\b(?:neutr[oó]filos|linfocitos|monocitos|eosin[oó]filos|bas[oó]filos)\b', re.I), 9),
        # Química sanguínea
        (re.compile(r'\b(?:glucosa|glucose|glicemia|blood\s+sugar)\b', re.I), 9),
        (re.compile(r'\b(?:colesterol|cholesterol)\b', re.I), 9),
        (re.compile(r'\b(?:triglic[eé]ridos|triglycerides)\b', re.I), 9),
        (re.compile(r'\b(?:creatinina|creatinine|urea|BUN|[aá]cido\s+[uú]rico)\b', re.I), 9),
        (re.compile(r'\b(?:bilirrubina|bilirubin|transaminasas?|ALT|AST|TGO|TGP)\b', re.I), 9),
        (re.compile(r'\b(?:fosfatasa\s+alcalina|albumina|prote[ií]nas?\s+totales?)\b', re.I), 8),
        (re.compile(r'\b(?:HDL|LDL|VLDL)\b', re.I), 8),
        (re.compile(r'\b(?:TSH|T3|T4|hormona\s+tiroide?a?|tiroides?)\b', re.I), 9),
        (re.compile(r'\b(?:PSA|HbA1c|hemoglobina\s+glicosilada|glucosilada)\b', re.I), 9),
        # Orina
        (re.compile(r'\b(?:parcial\s+de\s+orina|uroan[aá]lisis|urinalysis)\b', re.I), 10),
        (re.compile(r'\b(?:pH|densidad|aspecto|color)\s*:?\s*\d', re.I), 5),
        # Valores de referencia
        (re.compile(r'\b(?:valor(?:es)?\s+de\s+referencia|reference\s+value|rango\s+(?:normal|de\s+referencia))\b', re.I), 10),
        (re.compile(r'\b(?:resultado|value|valor)\s*:?\s*\d', re.I), 4),
        (re.compile(r'\b(?:alto|bajo|normal|anormal|elevado|disminuido|positivo|negativo)\b', re.I), 4),
        # Unidades de laboratorio
        (re.compile(r'\b(?:mg/dL|g/dL|mmol/L|UI/L|mL|mm3|cel/uL|mEq/L|ng/mL|pg/mL|U/L)\b', re.I), 9),
        (re.compile(r'\b(?:x10[³3]/[uμ]L|x10[⁶6]/[uμ]L|fl|fL)\b', re.I), 8),
        # Datos del paciente/muestra
        (re.compile(r'\b(?:muestra|sample|tipo\s+de\s+muestra|specimen)\b', re.I), 6),
        (re.compile(r'\b(?:paciente|patient)\b', re.I), 5),
        (re.compile(r'\b(?:fecha\s+de\s+(?:toma|recolecci[oó]n|procesamiento))\b', re.I), 7),
        (re.compile(r'\b(?:bioqu[ií]mic[oa]|bacteri[oó]log[oa]|microbiol[oó]g[oa])\b', re.I), 7),
    ],
    "tabla_nutricional": [
        # Título (señal MUY fuerte)
        (re.compile(r'\b(?:tabla\s+nutricional|informaci[oó]n\s+nutricional|nutrition(?:al)?\s+facts?)\b', re.I), 14),
        (re.compile(r'\bvalor\s+nutricional\b', re.I), 12),
        (re.compile(r'\b(?:datos\s+nutricionales|nutrientes?)\b', re.I), 9),
        # Macronutrientes
        (re.compile(r'\b(?:calor[ií]as|calories|energ[ií]a\s+total|valor\s+energ[eé]tico)\b', re.I), 9),
        (re.compile(r'\b(?:grasa(?:s)?\s+total(?:es)?|total\s+fat)\b', re.I), 9),
        (re.compile(r'\b(?:grasa(?:s)?\s+saturada(?:s)?|saturated\s+fat)\b', re.I), 9),
        (re.compile(r'\b(?:grasa(?:s)?\s+trans|trans\s+fat)\b', re.I), 9),
        (re.compile(r'\b(?:colesterol|cholesterol)\s*:?\s*\d+\s*mg\b', re.I), 9),
        (re.compile(r'\b(?:sodio|sodium)\s*:?\s*\d+\s*mg\b', re.I), 9),
        (re.compile(r'\b(?:carbohidratos?\s+totales?|total\s+carb(?:ohydrate)?s?)\b', re.I), 9),
        (re.compile(r'\b(?:fibra\s+(?:diet[eé]tica|alimentaria)|dietary\s+fiber)\b', re.I), 8),
        (re.compile(r'\b(?:az[úu]cares?\s+totales?|total\s+sugars?|az[úu]cares?\s+a[ñn]adidos?|added\s+sugars?)\b', re.I), 9),
        (re.compile(r'\b(?:prote[ií]nas?|protein)\s*:?\s*\d+\s*g?\b', re.I), 8),
        # Valor diario
        (re.compile(r'%\s*(?:valor\s+diario|VD|DV|daily\s+value)', re.I), 10),
        (re.compile(r'\b(?:basado\s+en\s+(?:una\s+)?dieta\s+de\s+\d)', re.I), 9),
        (re.compile(r'\b(?:dieta\s+de\s+2[\.,]?000\s*(?:cal(?:or[ií]as)?|kcal)?)\b', re.I), 9),
        # Porción
        (re.compile(r'\b(?:porci[oó]n|serving)\s*(?:size|por)?\s*:?\s*\d+', re.I), 8),
        (re.compile(r'\b(?:porciones|servings)\s+por\s+(?:envase|paquete|contenedor|container)\b', re.I), 9),
        (re.compile(r'\b(?:tama[ñn]o\s+de\s+(?:la\s+)?porci[oó]n|serving\s+size)\b', re.I), 10),
        # Micronutrientes
        (re.compile(r'\b(?:vitamina\s+[ABCDEK]\d?|vitamin\s+[ABCDEK]\d?)\b', re.I), 7),
        (re.compile(r'\b(?:hierro|iron|calcio|calcium|potasio|potassium|zinc|magnesio|f[oó]sforo)\b', re.I), 7),
        (re.compile(r'\b(?:[aá]cido\s+f[oó]lico|folic\s+acid|niacina|riboflavina|tiamina)\b', re.I), 7),
        # Unidades nutricionales
        (re.compile(r'\b\d+\s*(?:g|mg|mcg|kcal|kJ)\b', re.I), 6),
    ],
    "calendario": [
        # Palabra clave directa
        (re.compile(r'\b(?:calendario|calendar)\b', re.I), 12),
        (re.compile(r'\b(?:calendario\s+(?:mensual|semanal|anual|acad[eé]mico|escolar|laboral|de\s+eventos?))\b', re.I), 14),
        # Meses del año (múltiples matches = señal muy fuerte)
        (re.compile(r'\b(?:enero|january|Jan)\b', re.I), 7),
        (re.compile(r'\b(?:febrero|february|Feb)\b', re.I), 7),
        (re.compile(r'\b(?:marzo|march|Mar)\b', re.I), 7),
        (re.compile(r'\b(?:abril|april|Apr)\b', re.I), 7),
        (re.compile(r'\b(?:mayo|may)\b', re.I), 6),
        (re.compile(r'\b(?:junio|june|Jun)\b', re.I), 7),
        (re.compile(r'\b(?:julio|july|Jul)\b', re.I), 7),
        (re.compile(r'\b(?:agosto|august|Aug)\b', re.I), 7),
        (re.compile(r'\b(?:septiembre|september|Sep|Sept)\b', re.I), 7),
        (re.compile(r'\b(?:octubre|october|Oct)\b', re.I), 7),
        (re.compile(r'\b(?:noviembre|november|Nov)\b', re.I), 7),
        (re.compile(r'\b(?:diciembre|december|Dec)\b', re.I), 7),
        # Días de la semana (en contexto de calendario, no horario)
        (re.compile(r'\b(?:Dom|Lun|Mar|Mi[eé]|Jue|Vie|S[aá]b)\b', re.I), 5),
        (re.compile(r'\b(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat)\b', re.I), 5),
        # Grid de números (típico de vista mensual: 1-31)
        (re.compile(r'\b(?:1[0-9]|2[0-9]|3[01])\b', re.I), 2),
        # Eventos / fechas señaladas
        (re.compile(r'\b(?:feriado|festivo|holiday|d[ií]a\s+(?:libre|festivo|feriado))\b', re.I), 8),
        (re.compile(r'\b(?:fecha\s+l[ií]mite|deadline|vencimiento|recordatorio)\b', re.I), 6),
        (re.compile(r'\b(?:cumplea[ñn]os|birthday|aniversario|anniversary)\b', re.I), 7),
        # Apps de calendario (señal MUY fuerte — exclusiva)
        (re.compile(r'\b(?:Google\s+Calendar|Outlook\s+Calendar|Apple\s+Calendar|iCal|Samsung\s+Calendar)\b', re.I), 14),
        # Navegación temporal (exclusivo de calendario)
        (re.compile(r'\b(?:mes\s+(?:anterior|siguiente)|(?:previous|next)\s+month)\b', re.I), 9),
        (re.compile(r'\b(?:hoy|today|esta\s+semana|this\s+week|este\s+mes|this\s+month)\b', re.I), 5),
        (re.compile(r'\b(?:a[ñn]o\s+\d{4}|year\s+\d{4})\b', re.I), 7),
        # Vista de calendario (exclusivo)
        (re.compile(r'\b(?:vista\s+(?:mensual|semanal|diaria|anual)|(?:month|week|day|year)\s+view)\b', re.I), 10),
        (re.compile(r'\b(?:semana\s+\d+|week\s+\d+)\b', re.I), 8),
        (re.compile(r'\b(?:nuevo\s+evento|add\s+event|crear\s+evento|new\s+event)\b', re.I), 9),
        (re.compile(r'\b(?:evento|event)\b', re.I), 5),
    ],

    # ===== INTERFAZ_DIGITAL =====
    "app_menu": [
        (re.compile(r'\bmen[úu]\b', re.I), 10),
        (re.compile(r'\b(?:men[úu]\s+(?:del\s+d[ií]a|principal|de\s+opciones|de\s+navegaci[oó]n))\b', re.I), 12),
        (re.compile(r'\bentrantes\b', re.I), 8),
        (re.compile(r'\bpostres\b', re.I), 8),
        (re.compile(r'\bbebidas\b', re.I), 7),
        (re.compile(r'\bpromoci[óo]n(?:es)?\b', re.I), 7),
        (re.compile(r'\bcombo\b', re.I), 6),
        # Comida / restaurante
        (re.compile(r'\b(?:hamburguesa|pizza|ensalada|sopa|pasta|taco|burrito|wrap|s[aá]ndwich)\b', re.I), 8),
        (re.compile(r'\b(?:perro\s+caliente|hot\s+dog)\b', re.I), 8),
        (re.compile(r'\b(?:pollo|carne|pescado|cerdo|res|camar[oó]n|at[uú]n|salm[oó]n)\b', re.I), 5),
        (re.compile(r'\b(?:arroz|frijoles?|papas?\s+(?:fritas?|a\s+la)|yuca|pl[aá]tano)\b', re.I), 5),
        (re.compile(r'\b(?:porci[oó]n|plato|raci[oó]n|unidad)\b', re.I), 6),
        (re.compile(r'\b(?:acompa[ñn]amiento|guarnici[oó]n|adicional)\b', re.I), 6),
        # Patrón: item + precio
        (re.compile(r'\b\w+\b\s+\$\s*\d+', re.I), 7),
        # Menús de app
        (re.compile(r'\b(?:opciones|options|herramientas|tools|inicio|home)\b', re.I), 4),
        (re.compile(r'\b(?:favoritos|recientes|historial|categor[ií]as?)\b', re.I), 4),
    ],
    "app_settings": [
        (re.compile(r'\b(?:configuraci[oó]n|ajustes?|settings?)\b', re.I), 12),
        (re.compile(r'\b(?:configuraci[oó]n\s+(?:general|avanzada|del\s+sistema|de\s+red))\b', re.I), 12),
        (re.compile(r'\b(?:preferencias|personalizar|customize)\b', re.I), 10),
        (re.compile(r'\b(?:Wi-?Fi|Bluetooth|datos?\s+m[oó]viles|NFC|GPS)\b', re.I), 10),
        (re.compile(r'\b(?:brillo|brightness|volumen|volume|sonido|sound)\b', re.I), 10),
        (re.compile(r'\b(?:modo\s+(?:oscuro|claro|avi[oó]n|nocturno|no\s+molestar)|dark\s+mode)\b', re.I), 10),
        (re.compile(r'\b(?:activar|desactivar|enable|disable|toggle|switch)\b', re.I), 8),
        (re.compile(r'\b(?:cuenta|account|privacidad|privacy|seguridad|security)\b', re.I), 7),
        # Ajustes específicos
        (re.compile(r'\b(?:notificaciones|almacenamiento|storage|bater[ií]a|battery)\b', re.I), 8),
        (re.compile(r'\b(?:idioma|language|regi[oó]n|zona\s+horaria|time\s+zone)\b', re.I), 8),
        (re.compile(r'\b(?:pantalla|display|fondo\s+de\s+pantalla|wallpaper)\b', re.I), 7),
        (re.compile(r'\b(?:actualizaci[oó]n|update|versi[oó]n|version|acerca\s+de|about)\b', re.I), 7),
        (re.compile(r'\b(?:accesibilidad|accessibility|talkback|voiceover)\b', re.I), 9),
        (re.compile(r'\b(?:respaldo|backup|restaurar|restore|restablecer|reset)\b', re.I), 8),
        # Keywords negativos
        (re.compile(r'\b(?:iniciar\s+sesi[oó]n|registrar|log\s*in|sign\s*in)\b', re.I), -10),
        # Lista de verbos/frases en inglés → es un libro/glosario, no pantalla de ajustes
        (re.compile(r'\b(?:TURN\s+(?:UP|OFF|ON|DOWN)|CALL\s+ON|GET\s+OVER|GO\s+OVER|GO\s+THROUGH|LOOK\s+AFTER|RUN\s+INTO|CATCH\s+UP|BREAK\s+IN)\b', re.I), -20),
        (re.compile(r'\b(?:SWITCH\s+(?:OFF|ON)|RECOVER\s+FROM|TAKE\s+CARE\s+OF|FIND\s+BY\s+CHANCE|KEEP\s+ABREAST)\b', re.I), -20),
    ],
    "app_login": [
        (re.compile(r'\b(?:iniciar?\s+sesi[oó]n|log\s*in|sign\s*in)\b', re.I), 12),
        (re.compile(r'\b(?:registr(?:ar(?:se)?|o)|sign\s*up|create\s+account|crear\s+cuenta|nuevo\s+usuario)\b', re.I), 12),
        (re.compile(r'\b(?:contrase[ñn]a|password|clave)\b', re.I), 8),
        (re.compile(r'\b(?:olvidaste?\s+(?:tu\s+)?contrase[ñn]a|forgot\s+(?:your\s+)?password|recuperar\s+(?:contrase[ñn]a|cuenta))\b', re.I), 12),
        (re.compile(r'\b(?:bienvenid[oa]|welcome)\b', re.I), 5),
        (re.compile(r'\b(?:usuario|user(?:name)?|correo|email)\b', re.I), 5),
        # Autenticación
        (re.compile(r'\b(?:iniciar\s+con\s+(?:Google|Facebook|Apple|Twitter|GitHub|Microsoft))\b', re.I), 10),
        (re.compile(r'\b(?:sign\s+in\s+with\s+(?:Google|Facebook|Apple|Microsoft))\b', re.I), 10),
        (re.compile(r'\b(?:verificaci[oó]n|c[oó]digo\s+de\s+verificaci[oó]n|OTP|c[oó]digo\s+SMS)\b', re.I), 10),
        (re.compile(r'\b(?:autenticaci[oó]n|dos\s+pasos|two[\s-]factor|2FA|MFA)\b', re.I), 10),
        (re.compile(r'\b(?:recordar(?:me)?|remember\s+me|mantener\s+sesi[oó]n)\b', re.I), 7),
        (re.compile(r'\b(?:aceptar?\s+t[eé]rminos|terms\s+(?:of\s+service|and\s+conditions))\b', re.I), 6),
        (re.compile(r'\b(?:pol[ií]tica\s+de\s+privacidad|privacy\s+policy)\b', re.I), 5),
        # Keywords negativos
        (re.compile(r'\b(?:Wi-?Fi|Bluetooth|bater[ií]a|brillo|volumen)\b', re.I), -8),
    ],
    "app_form": [
        (re.compile(r'\b(?:enviar|submit|guardar|save|cancelar|cancel)\b', re.I), 6),
        (re.compile(r'\b(?:seleccionar|select|elegir|choose)\b', re.I), 5),
        (re.compile(r'\b(?:campo|field|obligatorio|required)\b', re.I), 7),
        (re.compile(r'\b(?:siguiente|next|anterior|back|continuar|continue|atr[aá]s)\b', re.I), 5),
        # Elementos de formulario digital
        (re.compile(r'\b(?:checkbox|radio\s+button|dropdown|men[uú]\s+desplegable)\b', re.I), 7),
        (re.compile(r'\b(?:cargar|upload|adjuntar|attach)\b', re.I), 6),
        (re.compile(r'\b(?:validar|validaci[oó]n|error|campo\s+inv[aá]lido|invalid)\b', re.I), 6),
        (re.compile(r'\b(?:confirmar|confirmaci[oó]n|confirm)\b', re.I), 5),
        (re.compile(r'\b(?:aceptar|accept|rechazar|reject|decline)\b', re.I), 5),
    ],
    "app_social": [
        # Plataformas
        (re.compile(r'\b(?:Instagram|Twitter|TikTok|Facebook|LinkedIn|YouTube|GitHub|Snapchat|Pinterest|Reddit|Threads)\b', re.I), 9),
        (re.compile(r'\b(?:seguir|follow|following|followers|seguidores)\b', re.I), 8),
        (re.compile(r'\b(?:me\s+gusta|like|likes|heart)\b', re.I), 7),
        (re.compile(r'\b(?:compartir|share|retweet|repost|reblog)\b', re.I), 7),
        (re.compile(r'@\w{2,}', re.I), 6),
        (re.compile(r'#\w{2,}', re.I), 6),
        # Perfil
        (re.compile(r'\b(?:perfil|profile|bio|about)\b', re.I), 7),
        (re.compile(r'\b(?:Repositories|contribution|Overview|Stars|Forks)\b', re.I), 7),
        (re.compile(r'\b(?:publicaci[oó]n|post|feed|timeline|stories|reel|short)\b', re.I), 6),
        (re.compile(r'\b(?:seguidores|following|followers)\b.*\d+', re.I), 8),
        # Interacciones sociales
        (re.compile(r'\b(?:comentar|comment|responder\s+a|reply\s+to)\b', re.I), 5),
        (re.compile(r'\b(?:mencionar|mention|etiquetar|tag)\b', re.I), 5),
        (re.compile(r'\b(?:mensaje\s+directo|DM|direct\s+message)\b', re.I), 6),
        (re.compile(r'\b(?:notificaciones?\s+de\s+(?:likes?|comentarios?|seguidores?))\b', re.I), 7),
        (re.compile(r'\b(?:trending|tendencia|viral|popular)\b', re.I), 5),
    ],
    "app_service": [
        # Apps de transporte
        (re.compile(r'\b(?:Uber|Cabify|DiDi|InDriver|Beat|Lyft|Bolt|Grab|Gojek)\b', re.I), 10),
        (re.compile(r'\b(?:Economy|Comfort|Premium|UberX|UberXL|Black|Pool|Share)\b', re.I), 8),
        (re.compile(r'\b(?:sug(?:erido|\.)?)\s*\.?\s*(?:COP|USD|MXN|EUR|BRL|ARS|PEN|BOB|Bs)\b', re.I), 9),
        (re.compile(r'\b(?:COP|USD|MXN|EUR|BRL|ARS|PEN|BOB)\s*[\d.,]+', re.I), 7),
        (re.compile(r'\b\d+\s*min\b', re.I), 5),
        (re.compile(r'\b(?:Limited\s+Availability|disponibilidad\s+limitada)\b', re.I), 7),
        (re.compile(r'\b(?:conductor|driver|chofer|piloto)\b', re.I), 6),
        (re.compile(r'\b(?:solicitar\s+(?:viaje|carro|auto)|request\s+(?:ride|trip))\b', re.I), 9),
        (re.compile(r'\b(?:punto\s+de\s+(?:recogida|encuentro)|pickup\s+point)\b', re.I), 7),
        # Apps de delivery
        (re.compile(r'\b(?:Rappi|PedidosYa|iFood|Uber\s*Eats|Domicilios|DoorDash|Glovo|Didi\s*Food)\b', re.I), 10),
        (re.compile(r'\b(?:delivery|domicilio|entrega|env[ií]o)\b', re.I), 6),
        (re.compile(r'\b(?:a[ñn]adir\s+al\s+carrito|add\s+to\s+cart|agregar)\b', re.I), 7),
        (re.compile(r'\b(?:pedido|order|tu\s+pedido|your\s+order)\b', re.I), 6),
        (re.compile(r'\b(?:tiempo\s+(?:estimado|de\s+entrega)|ETA|estimated\s+(?:time|delivery))\b', re.I), 7),
        # Tiendas online / e-commerce
        (re.compile(r'\b(?:SHEIN|AliExpress|Amazon|Mercado\s*Libre|Falabella|Ripley|Linio|Temu|Wish|eBay|Zara|H&M)\b', re.I), 12),
        (re.compile(r'\b(?:venta\s+flash|flash\s+sale|oferta\s+del\s+d[ií]a|liquidaci[oó]n)\b', re.I), 9),
        (re.compile(r'\b(?:env[ií]o\s+gratuito|env[ií]o\s+gratis|free\s+shipping)\b', re.I), 9),
        (re.compile(r'\b(?:productos?\s+(?:m[aá]s\s+vendidos?|destacados?|nuevos?)|best\s+sellers?)\b', re.I), 8),
        (re.compile(r'\b(?:carrito\s+de\s+compras?|bolsa\s+de\s+compras?|shopping\s+(?:cart|bag))\b', re.I), 8),
        (re.compile(r'\b(?:comprar\s+ahora|buy\s+now|a[ñn]adir\s+al\s+carrito|agregar\s+al\s+carrito)\b', re.I), 9),
        (re.compile(r'\b(?:descuento|descuentos?|ahorra|off|sale|promo(?:ci[oó]n)?)\b', re.I), 5),
        (re.compile(r'\b(?:talla|tallas?|sizes?|colores?|variantes?)\b', re.I), 6),
        (re.compile(r'\b(?:por\s+tiempo\s+limitado|tiempo\s+limitado|limited\s+time)\b', re.I), 7),
        # Servicios
        (re.compile(r'\b(?:tarifa|fare)\b', re.I), 6),
        (re.compile(r'\b(?:suscripci[oó]n|subscription|plan\s+(?:mensual|anual|premium))\b', re.I), 6),
        (re.compile(r'\b(?:reservar|book|agendar|schedule)\b', re.I), 5),
        # Pagos en apps
        (re.compile(r'\b(?:m[eé]todo\s+de\s+pago|payment\s+method|pagar\s+con)\b', re.I), 6),
        (re.compile(r'\b(?:cupón|cupon|promo\s*code|c[oó]digo\s+(?:de\s+)?descuento)\b', re.I), 6),
    ],
    "notificacion": [
        (re.compile(r'\bnotificaci[oó]n(?:es)?\b', re.I), 12),
        (re.compile(r'\b(?:centro\s+de\s+notificaciones|notification\s+center)\b', re.I), 12),
        (re.compile(r'\b(?:permitir|bloquear|dismiss|allow|deny|descartar)\b', re.I), 8),
        (re.compile(r'\bhace\s+\d+\s+(?:min(?:utos?)?|horas?|seg(?:undos?)?|d[ií]as?)', re.I), 8),
        # Tipos de notificación
        (re.compile(r'\b(?:alerta|alert|aviso|recordatorio|reminder)\b', re.I), 8),
        (re.compile(r'\b(?:nueva\s+(?:notificaci[oó]n|alerta|actualizaci[oó]n))\b', re.I), 10),
        (re.compile(r'\b(?:push|banner|badge|popup|emergente)\b', re.I), 5),
        (re.compile(r'\b(?:silenciar|mute|no\s+molestar|do\s+not\s+disturb)\b', re.I), 6),
        (re.compile(r'\b(?:marcar\s+como\s+le[ií]d[oa]|mark\s+as\s+read|borrar\s+todo|clear\s+all)\b', re.I), 7),
        (re.compile(r'\b(?:ahora|mismo|justo\s+ahora|right\s+now)\b', re.I), 6),
        # Keywords negativos
        (re.compile(r'\b(?:WhatsApp|Telegram|en\s+l[ií]na|typing|escribiendo)\b', re.I), -10),
    ],
    "mapa": [
        # Negativos fuertes: los mapas nunca tienen estas señales
        (re.compile(r'\d{1,2}:\d{2}\s*(?:p\.?m\.?|a\.?m\.?|PM|AM)', re.I), -20),   # timestamp de chat (8:13 AM)
        (re.compile(r'\b(?:WhatsApp|Telegram|Messenger|Signal|iMessage)\b', re.I), -25),
        (re.compile(r'\b(?:jaja[ja]*|jeje[je]*|haha|lol|omg|wtf)\b', re.I), -15),
        (re.compile(r'\b(?:publica|publicar|estados|historia|story|stories)\b', re.I), -10),
        (re.compile(r'\b(?:salmo|vers[ií]culo|bíblico|biblia|dios|señor|amen)\b', re.I), -12),
        # Palabra clave directa
        (re.compile(r'\b(?:mapa|map)\b', re.I), 8),
        (re.compile(r'\b(?:mapa\s+(?:de\s+)?(?:calles|carreteras|tr[aá]nsito|tr[aá]fico|sat[eé]lite))\b', re.I), 12),
        # Apps de mapas / navegación
        (re.compile(r'\b(?:Google\s+Maps|Waze|Apple\s+Maps|Maps|Mapbox|OpenStreetMap)\b', re.I), 12),
        (re.compile(r'\b(?:Moovit|Citymapper|HERE\s+Maps|Maps\.me)\b', re.I), 10),
        # Navegación / direcciones
        (re.compile(r'\b(?:navegar|navigate|navegaci[oó]n|navigation|direcciones|directions)\b', re.I), 9),
        (re.compile(r'\b(?:c[oó]mo\s+llegar|get\s+directions|ir\s+a|go\s+to)\b', re.I), 9),
        (re.compile(r'\b(?:ruta|route|recorrido|trayecto|camino)\b', re.I), 7),
        (re.compile(r'\b(?:ruta\s+m[aá]s\s+(?:r[aá]pida|corta)|fastest\s+route|shortest\s+route)\b', re.I), 10),
        (re.compile(r'\b(?:indicaciones|turn[\s-]by[\s-]turn|paso\s+a\s+paso)\b', re.I), 7),
        # Tráfico / tiempo de viaje
        (re.compile(r'\b(?:tr[aá]fico|traffic|tr[aá]nsito|congesti[oó]n)\b', re.I), 8),
        (re.compile(r'\b(?:tiempo\s+(?:estimado|de\s+(?:viaje|llegada))|ETA|estimated\s+(?:time|arrival))\b', re.I), 8),
        (re.compile(r'\b\d+\s*(?:min|h)\s+(?:en\s+)?(?:auto|carro|coche|car|a\s+pie|walking|bici|bike)\b', re.I), 9),
        (re.compile(r'\b\d+\s*(?:km|mi|metros|m)\b', re.I), 5),
        # Ubicación / puntos de interés
        (re.compile(r'\b(?:ubicaci[oó]n|location|mi\s+ubicaci[oó]n|my\s+location)\b', re.I), 7),
        (re.compile(r'\b(?:buscar\s+(?:lugar|direcci[oó]n|sitio)|search\s+(?:place|location|address))\b', re.I), 8),
        (re.compile(r'\b(?:cerca\s+de\s+(?:m[ií]|aqu[ií])|near(?:by)?|nearby\s+places?)\b', re.I), 7),
        (re.compile(r'\b(?:restaurantes?|gasolineras?|farmacias?|hospitales?|cajeros?)\s+cerca', re.I), 7),
        (re.compile(r'\b(?:punto\s+de\s+inter[eé]s|POI|landmark|lugar\s+(?:guardado|favorito))\b', re.I), 6),
        # Elementos de mapa
        (re.compile(r'\b(?:calle|avenida|carrera|autopista|boulevard|highway|freeway)\b', re.I), 5),
        (re.compile(r'\b(?:Av\.?|Cra\.?|Cl\.?|Cll\.?|Blvd\.?|Km\.?)\s*\d', re.I), 6),
        (re.compile(r'\b(?:zoom|acercar|alejar|sat[eé]lite|street\s+view|vista\s+(?:de\s+)?calle)\b', re.I), 7),
        (re.compile(r'\b(?:norte|sur|este|oeste|north|south|east|west|NE|NW|SE|SW)\b', re.I), 4),
        # Transporte en mapa
        (re.compile(r'\b(?:en\s+auto|en\s+carro|driving|a\s+pie|walking|en\s+bici(?:cleta)?|cycling|transporte\s+p[uú]blico|transit)\b', re.I), 7),
        (re.compile(r'\b(?:parada|estaci[oó]n|terminal|aeropuerto|puerto)\b', re.I), 5),
        # Compartir ubicación
        (re.compile(r'\b(?:compartir\s+(?:ubicaci[oó]n|mi\s+ubicaci[oó]n)|share\s+(?:location|my\s+location))\b', re.I), 8),
        (re.compile(r'\b(?:enviar\s+ubicaci[oó]n|ubicaci[oó]n\s+en\s+tiempo\s+real|live\s+location)\b', re.I), 8),
    ],

    # ===== TEXTO_CONVERSACIONAL =====
    "chat": [
        # Nombres de apps de mensajería
        (re.compile(r'\b(?:WhatsApp|Telegram|Messenger|Signal|iMessage|Viber|Line|WeChat)\b', re.I), 12),
        (re.compile(r'\b(?:escribir?\s+(?:un\s+)?mensaje|type\s+a\s+message)\b', re.I), 10),
        # Header de WhatsApp (lista de chats)
        (re.compile(r'\bChats\b.*\b(?:Archivados|Buscar|buscar)\b', re.I | re.S), 12),
        (re.compile(r'\bArchivados\s+\d+', re.I), 10),
        (re.compile(r'\b(?:llamada\s+de\s+voz|videoLLamada|voice\s+call)\b', re.I), 10),
        # Timestamps de mensajes (señal MUY fuerte de chat)
        (re.compile(r'\d{1,2}:\d{2}\s*(?:p\.?m\.?|a\.?m\.?|PM|AM)', re.I), 8),
        # Estado de conexión
        (re.compile(r'\b(?:en\s+l[ií]na|online)\b', re.I), 10),
        (re.compile(r'\b(?:[uú]lt(?:ima)?\s*vez|last\s+seen)\b', re.I), 8),
        (re.compile(r'\b(?:visto\s+a\s+las?\b|visto\s+\d)', re.I), 10),
        (re.compile(r'\b(?:escribiendo|typing)\b', re.I), 12),
        (re.compile(r'\b(?:conectad[oa]|desconectad[oa]|ausente|ocupad[oa]|away)\b', re.I), 5),
        # Expresiones informales (cada match suma)
        (re.compile(r'\b(?:jaja[ja]*|jeje[je]*|haha[ha]*|jiji[ji]*)\b', re.I), 6),
        (re.compile(r'\b[xX][dD]+\b', re.I), 5),
        (re.compile(r'\b(?:lol|lmao|omg|wtf|tbh|imo|btw)\b', re.I), 5),
        # Saludos y jerga
        (re.compile(r'\b(?:hola|hey|pana|bro|mano|vale|parce|wey|marica|compa)\b', re.I), 4),
        (re.compile(r'\b(?:qu[eé]\s+(?:m[aá]s|hay|onda|tal)|como?\s+(?:estas?|est[aá]s?|andas?|vas?))\b', re.I), 5),
        # Palabras de chat
        (re.compile(r'\b(?:grupo|chat|conversaci[oó]n|Chats)\b', re.I), 6),
        (re.compile(r'\b(?:audio|nota\s+de\s+voz|voice\s+note)\b', re.I), 6),
        (re.compile(r'\b(?:Sticker|Foto|Video|GIF|imagen|foto|documento)\b', re.I), 4),
        # Patrones de preview de chat
        (re.compile(r'\b(?:T[uú]|Yo)\s*:', re.I), 7),
        (re.compile(r'\bp\.?\s*m\.?\b', re.I), 3),
        # Emojis descritos por OCR
        (re.compile(r'\b(?:enviado|entregado|le[ií]do|visto|delivered|read|sent)\b', re.I), 5),
        # Patrones de mensajería
        (re.compile(r'\b(?:responder|reenviar|eliminar\s+(?:para\s+(?:m[ií]|todos))|forward|delete)\b', re.I), 5),
        (re.compile(r'\b(?:mensaje\s+(?:de\s+voz|eliminado|reenviado)|missed\s+call|llamada\s+perdida)\b', re.I), 7),
        (re.compile(r'\b(?:cifrado\s+de\s+extremo|end[\s-]to[\s-]end\s+encrypt)\b', re.I), 6),
        # Keywords negativos (rechaza si aparecen) -区分chat vs notificacion
        (re.compile(r'\b(?:permitir|bloquear|centro\s+de\s+notificaciones)\b', re.I), -8),
    ],
    "comentario": [
        (re.compile(r'\b(?:comentar(?:io)?|comment|responder|reply|respuesta)\b', re.I), 7),
        (re.compile(r'\b(?:hace\s+\d+\s+(?:min(?:utos?)?|horas?|d[ií]as?|mes(?:es)?|a[ñn]os?|semanas?))', re.I), 7),
        (re.compile(r'\b(?:me\s+gusta|like|likes|dislike)\b', re.I), 5),
        (re.compile(r'@\w{2,}', re.I), 5),
        # Señal muy fuerte: contador de comentarios visible
        (re.compile(r'\b\d+\s+comentarios?\b', re.I), 14),
        (re.compile(r'\b\d+\s+(?:comments?|respuestas?|replies)\b', re.I), 12),
        (re.compile(r'\b(?:ver\s+(?:m[aá]s\s+)?(?:comentarios?|respuestas?)|show\s+(?:more\s+)?(?:comments?|replies))\b', re.I), 8),
        (re.compile(r'\b(?:mejor\s+comentario|top\s+comment|m[aá]s\s+(?:reciente|antiguo)|newest|oldest)\b', re.I), 7),
        (re.compile(r'\b(?:reportar|denunciar|report|flag)\b', re.I), 4),
        (re.compile(r'\b(?:editar|edit|borrar|delete|eliminar)\b', re.I), 4),
        # Patrón típico de hilo: "Usuario Hace X tiempo texto Responder"
        (re.compile(r'\bResponder\b', re.I), 8),
        # Penalización: señales de correo real que NO aparecen en secciones de comentarios
        (re.compile(r'\b(?:asunto|subject|de:|from:|para:|to:|cc:|bcc:)\b', re.I), -10),
    ],

    # ===== IMAGEN_VISUAL =====
    "imagen_visual": [
        # No tiene keywords — se clasifica por exclusión/regla de texto insuficiente
    ],

    # ===== TIPOS ADICIONALES =====
    "factura_servicio": [
        # Palabra clave directa
        (re.compile(r'\b(?:factura\s+de\s+servicio|factura\s+de\s+(?:agua|luz|gas|electricidad|internet|tel[eé]fono|m[oó]vil))\b', re.I), 14),
        (re.compile(r'\b(?:servicio\s+de\s+(?:agua|luz|gas|electricidad|internet|tel[eé]fono|m[oó]vil))\b', re.I), 12),
        (re.compile(r'\b(?:servicios?\s+p[uú]blicos?|servicios?\s+domiciliarios?)\b', re.I), 12),
        (re.compile(r'\b(?:per[ií]odo\s+de\s+facturaci[oó]n|consumo\s+del\s+per[ií]odo)\b', re.I), 12),
        (re.compile(r'\b(?:lectura\s+anterior|lectura\s+actual)\b', re.I), 10),
        (re.compile(r'\b(?:cargo\s+fijo|cargo\s+variable|tarifa\s+(?:basic[oa]|social|residencial|comercial))\b', re.I), 10),
        (re.compile(r'\b(?:subtotal|total\s+a\s+pagar|monto\s+total)\b', re.I), 8),
        (re.compile(r'\b(?:fecha\s+l[ií]mite\s+de\s+pago|vence\s+el|fecha\s+de\s+vencimiento)\b', re.I), 10),
        (re.compile(r'\b(?:kWh|m3|metros\s+c[uú]bicos|gigabytes?|GB|MB)\b', re.I), 10),
        (re.compile(r'\b(?:n[úu]mero\s+de\s+cliente|c[oó]digo\s+de\s+servicio|referencia\s+bancaria)\b', re.I), 10),
        (re.compile(r'\b(?:medidor|n[úu]mero\s+de\s+medidor|lectura\s+del\s+medidor)\b', re.I), 10),
        (re.compile(r'\b(?:consumo\s+(?:m3|kWh)|metros?\s+c[uú]bicos?|kilovatios?)\b', re.I), 10),
    ],
    "ticket_transporte": [
        # Palabra clave directa
        (re.compile(r'\b(?:boleto\s+(?:de\s+)?(?:transporte|bus|metro)|pasaje\s+(?:simple|ida|ida\s+y\s+vuelta))\b', re.I), 14),
        (re.compile(r'\b(?:tarjeta\s+de\s+transporte|recarga\s+de\s+saldo|tarjet[eá]s?\s+magn[eé]tica)\b', re.I), 12),
        (re.compile(r'\b(?:parada|de\s+bajada|estaci[oó]n\s+de\s+(?:subida|bajada))\b', re.I), 10),
        (re.compile(r'\b(?:ruta\s+\d+|l[ií]nea\s+\d+|circuito\s+\d+|tramo)\b', re.I), 12),
        (re.compile(r'\b(?:valsol|pasaje|subsecretar[ií]a\s+de\s+transporte|autoridad\s+metropolitana)\b', re.I), 12),
        (re.compile(r'\b(?:tiempo\s+de\s+validez|vigencia\s+del\s+pasaje)\b', re.I), 10),
        (re.compile(r'\b(?:viaje\s+(?:sencillo|de\s+ida|redondo)|single\s+trip|round\s+trip)\b', re.I), 12),
        (re.compile(r'\b(?:tarifa\s+(?:general|estudiantil|adulto\s+mayor|discapacitado))\b', re.I), 10),
        (re.compile(r'\b(?:saldo\s+actual|saldo\s+restante|recarga)\b', re.I), 8),
    ],
    "credencial": [
        # Palabra clave directa — máximo peso
        (re.compile(r'\b(?:credencial|carnet|gafete|badge|identification\s+card|access\s+card|employee\s+ID|ID\s+emplead[oa])\b', re.I), 16),
        (re.compile(r'\b(?:empleado|trabajador|funcionario|colaborador|staff|personal)\b', re.I), 12),
        # Cargo / puesto con dos puntos (muy específico de credencial)
        (re.compile(r'\b(?:cargo|puesto|posici[oó]n)\s*:', re.I), 14),
        (re.compile(r'\b(?:cargo|puesto|posici[oó]n)\b', re.I), 9),
        # Departamento con dos puntos (muy específico de credencial)
        (re.compile(r'\b(?:departamento|[aá]rea|secci[oó]n|divisi[oó]n)\s*:', re.I), 12),
        (re.compile(r'\b(?:departamento|[aá]rea\s+de\s+trabajo)\b', re.I), 8),
        # Validez — común en credenciales
        (re.compile(r'\b(?:v[aá]lido\s+hasta|vence\s+el|validez|vencimiento|expira|exp\.?\s*:)\b', re.I), 10),
        # Empresa / institución emisora
        (re.compile(r'\b(?:empresa|compa[ñn][ií]a|organizaci[oó]n|corporaci[oó]n|instituci[oó]n)\b', re.I), 8),
        # Identificador único de credencial
        (re.compile(r'\b(?:identificador|c[oó]digo\s+(?:de\s+)?emplead[oa]|n[úu]mero\s+(?:de\s+)?emplead[oa]|ficha|legajo)\b', re.I), 14),
        # Frases de uso personal / acceso
        (re.compile(r'\b(?:uso\s+personal|intransferible|portar\s+(?:este|el)\s+(?:carnet|gafete|documento))\b', re.I), 12),
        (re.compile(r'\b(?:acceso\s+(?:de\s+)?(?:personal|visitantes?|veh[ií]culos?)|nivel\s+de\s+acceso)\b', re.I), 10),
        # Visita / contratista
        (re.compile(r'\b(?:visita|visitante|contratista|proveedor)\b', re.I), 10),
        # Seguridad / recepción
        (re.compile(r'\b(?:vigilante|seguridad|recepci[oó]n|porter[ií]a)\b', re.I), 6),
        # Keywords negativos: términos de CV/hoja de vida que no aparecen en credenciales
        (re.compile(r'\b(?:experiencia\s+(?:laboral|profesional)|hoja\s+de\s+vida|curr[ií]cul[ou]m)\b', re.I), -15),
        (re.compile(r'\b(?:aptitudes?|habilidades?|competencias?)\b', re.I), -8),
        (re.compile(r'\b(?:educaci[oó]n|formaci[oó]n\s+acad[eé]mica|estudios)\b', re.I), -8),
        (re.compile(r'\b(?:idiomas?|lenguas?)\b', re.I), -8),
        (re.compile(r'\b(?:objetivo\s+(?:profesional|laboral)|perfil\s+profesional)\b', re.I), -12),
        # Keywords negativos: términos de formulario que NUNCA aparecen en credenciales
        (re.compile(r'\bformulario\b', re.I), -20),
        (re.compile(r'\b(?:enviar|submit)\b', re.I), -12),
        (re.compile(r'\bc[oó]digo\s+postal\b', re.I), -12),
        (re.compile(r'\bpersona\s+de\s+contacto\b', re.I), -10),
        (re.compile(r'ejemplo@\w+\.\w+|\(000\)\s*000', re.I), -10),
        (re.compile(r'\bintroducid?\s+un\s+n[uú]mero\s+v[aá]lido\b', re.I), -10),
        (re.compile(r'\bnombre\s+de\s+la\s+compa[ñn][ií]a\b', re.I), -8),
    ],
    "app_banking": [
        # Palabras clave directas de banca
        (re.compile(r'\b(?:banco|bancolombia|nequi|daviplata|davivienda|bbva|scotiabank|itaú|itau|popular|occidente|bogot[aá]|ban[ck]\w*)\b', re.I), 10),
        (re.compile(r'\b(?:cupo\s+(?:disponible|total|usado|de\s+cr[eé]dito)|cupo\s*[\$:]\s*[\d,.]+)\b', re.I), 14),
        (re.compile(r'\b(?:pago\s+m[ií]nimo|pago\s+total|pago\s+oportuno|fecha\s+de\s+corte|fecha\s+de\s+pago)\b', re.I), 12),
        (re.compile(r'\b(?:saldo\s+(?:disponible|actual|total|a\s+favor|en\s+cuenta|de\s+ahorros|corriente))\b', re.I), 12),
        (re.compile(r'\b(?:transferencia|consignaci[oó]n|retiro|dep[oó]sito|tr[aá]nsacci[oó]n|movimiento)\b', re.I), 8),
        (re.compile(r'\b(?:tarjeta\s+de\s+cr[eé]dito|tarjeta\s+d[eé]bito|cuenta\s+de\s+(?:ahorros|corriente))\b', re.I), 10),
        (re.compile(r'\b(?:extracto|estado\s+de\s+cuenta|resumen\s+de\s+cuenta)\b', re.I), 12),
        (re.compile(r'\b(?:intereses?|tasa\s+de\s+inter[eé]s|mora)\b', re.I), 7),
        (re.compile(r'\b(?:cuota|abono|cr[eé]dito\s+(?:aprobado|disponible|utilizado|total))\b', re.I), 7),
        (re.compile(r'\b(?:n[úu]mero\s+de\s+cuenta|n[úu]mero\s+de\s+tarjeta|\*{3,}\d{4})\b', re.I), 8),
        (re.compile(r'\b(?:COP|USD)\s*[\d,.]+|\d[\d,.]*\s*(?:COP|USD)\b', re.I), 5),
        (re.compile(r'\b(?:próximo\s+pago|vence?\s+el|fecha\s+l[ií]mite)\b', re.I), 8),
        # Apps específicas de banca digital
        (re.compile(r'\b(?:mi\s+banco|home\s+banking|banca\s+m[oó]vil|banca\s+digital|billetera\s+(?:digital|virtual))\b', re.I), 10),
    ],
    "tarjeta_felicitacion": [
        # Frases características de tarjetas de felicitación
        (re.compile(r'\b(?:felicitaciones?|congratulations?|enhorabuena)\b', re.I), 14),
        (re.compile(r'\b(?:feliz\s+(?:cumplea[ñn]os?|navidad|a[ñn]o\s+nuevo|d[ií]a\s+de|graduaci[oó]n))\b', re.I), 12),
        (re.compile(r'\b(?:cumplea[ñn]os?|birthday)\b', re.I), 8),
        (re.compile(r'\b(?:graduaci[oó]n|te\s+graduaste|se\s+gradu[oó]|eres\s+graduad[oa]|egresad[oa])\b', re.I), 10),
        (re.compile(r'\b(?:te\s+quer(?:emos|emos)|con\s+(?:todo\s+nuestro|mucho)\s+(?:amor|cari[ñn]o)|con\s+afecto)\b', re.I), 9),
        (re.compile(r'\b(?:qu(?:e\s+)?(?:dios\s+te|siempre)\s+(?:bendiga|acompa[ñn]e)|bendiciones)\b', re.I), 8),
        (re.compile(r'\b(?:tu\s+[eé]xito|mucho\s+[eé]xito|[eé]xitos|que\s+lo\s+(?:disfrutes?|celebres?|pases?\s+bien))\b', re.I), 9),
        (re.compile(r'\b(?:este\s+(?:nuevo\s+)?(?:logro|paso|etapa|cap[ií]tulo)|nuevo\s+inicio|nuevo\s+camino)\b', re.I), 8),
        (re.compile(r'\b(?:con\s+cari[ñn]o|con\s+amor|con\s+todo\s+mi\s+(?:amor|cari[ñn]o|coraz[oó]n))\b', re.I), 8),
        (re.compile(r'\b(?:para\s+(?:ti|vos|usted)|te\s+deseo|les?\s+desea(?:mos?)?)\b', re.I), 5),
        (re.compile(r'\b(?:eres\s+(?:increíble|espectacular|maravillos[oa]|lo\s+mejor)|tan\s+orgullos[oa])\b', re.I), 9),
        # Frases de graduación específicas
        (re.compile(r'\b(?:doctor(?:a)?|licenciad[oa]|ingenier[oa]|abogad[oa]|médic[oa])\b.*\b(?:graduaci[oó]n|titulaci[oó]n)\b', re.I | re.S), 12),
        (re.compile(r'\b(?:universidad|facultad|carrera|promoci[oó]n)\b.*\b(?:graduaci[oó]n|grado|t[ií]tulo)\b', re.I | re.S), 10),
        # Frases del cuerpo del mensaje (cuando el título decorativo no fue leído por OCR)
        (re.compile(r'\bnuevo\s+a[ñn]o\s+de\s+vida\b', re.I), 12),
        (re.compile(r'\ba[ñn]o\s+(?:de\s+)?vida\b', re.I), 9),
        (re.compile(r'\blleno\s+(?:de\s+)?alegr[ií]a\b', re.I), 9),
        (re.compile(r'\bsorpresas?\s+agradables?\b', re.I), 9),
        (re.compile(r'\bluz\s+y\s+alegr[ií]a\b|\balegr[ií]a\s+y\s+amor\b|\bamor\s+y\s+(?:alegr[ií]a|paz)\b', re.I), 8),
        (re.compile(r'\b(?:que\s+(?:todos?\s+tus|este)\s+(?:sue[ñn]os?|deseos?|metas?)\s+(?:se\s+)?cumplan?)\b', re.I), 10),
        (re.compile(r'\b(?:pases?\s+(?:un\s+)?(?:lindo|hermoso|bonito|genial|especial)\s+d[ií]a)\b', re.I), 9),
        # Señales de tarjeta de graduación detectables desde el cuerpo del texto
        # (cuando el título "Felicitaciones por tu grado" está en cursiva y no lo lee el OCR)
        (re.compile(r'\b(?:viaje\s+lleno\s+de\s+esfuerzo|lleno\s+de\s+esfuerzo)\b', re.I), 12),
        (re.compile(r'\b(?:esfuerzo\W{0,5}trabajo\W{0,5}perseverancia|trabajo\W{0,5}perseverancia)\b', re.I), 12),
        (re.compile(r'\bperseverancia\b', re.I), 7),
        (re.compile(r'\b(?:dado\s+alas?|te\s+ha\s+dado\s+alas?|dar(?:te)?\s+alas?)\b', re.I), 11),
        (re.compile(r'\bhacer\s+grandes\s+cosas\b', re.I), 10),
        (re.compile(r'\b(?:lo\s+lograste|lo\s+conseguiste|lograste\s+tu\s+(?:meta|sue[ñn]o|objetivo))\b', re.I), 11),
        (re.compile(r'\b(?:lograrlo|haberlo\s+logrado)\b', re.I), 8),
        (re.compile(r'\b(?:saber\s+que\s+puedes|puedes\s+hacer\s+grandes)\b', re.I), 9),
        (re.compile(r'\b(?:felicitaciones?|felicidades)\b', re.I), 10),
        # Keywords negativos (distinguir de carta formal)
        (re.compile(r'\b(?:atentamente|cordialmente|a\s+quien\s+corresponda|me\s+dirijo)\b', re.I), -8),
    ],
}


class SubtypeClassifier:
    """
    Fase 2: Clasifica el subtipo usando keywords sobre texto LIMPIO.

    Evalúa TODOS los subtipos (no solo los del macro-tipo).
    El macro-tipo da un bonus a sus subtipos, pero no excluye otros.

    Scoring probabilístico:
      - Scores normalizados 0-1
      - Gap < 0.15 entre top-1 y top-2 → "mixto"
      - Confianza < 0.6 → no forzar clasificación
    """

    MIN_GAP_RATIO = 0.15    # Gap mínimo normalizado para clasificación confiable
    MIN_SUBTYPE_SCORE = 5    # Score mínimo absoluto (antes de normalizar)
    MIN_CONFIDENCE = 0.6     # Confianza mínima para aceptar

    def classify(
        self,
        clean_text: str,
        macro_type: str,
        word_count: int,
    ) -> Tuple[str, float, Dict[str, float], List[str], bool, str]:
        """
        Clasifica el subtipo usando keywords sobre texto limpio.

        Returns:
            (subtype, confidence, all_scores, reasons, is_ambiguous, ambiguity_note)
        """
        preferred_subtypes = set(_MACRO_TO_SUBTYPES.get(macro_type, []))
        all_candidate_subtypes = set(_SUBTYPE_RULES.keys())

        # Calcular scores para cada subtipo
        raw_scores: Dict[str, float] = {}
        reasons: Dict[str, List[str]] = {}

        for subtype in all_candidate_subtypes:
            rules = _SUBTYPE_RULES.get(subtype, [])
            if not rules:
                raw_scores[subtype] = 0.0
                reasons[subtype] = []
                continue

            total = 0.0
            sub_reasons = []
            for pattern, weight in rules:
                matches = len(pattern.findall(clean_text))
                if matches > 0:
                    total += matches * weight
                    sub_reasons.append(
                        f"'{pattern.pattern[:40]}' ({matches}x, peso {weight})"
                    )

            # Bonus del 20% si pertenece al macro-tipo predicho
            if subtype in preferred_subtypes and total > 0:
                bonus = total * 0.2
                total += bonus
                sub_reasons.append(
                    f"Bonus macro {macro_type} (+{bonus:.0f})"
                )

            raw_scores[subtype] = total
            reasons[subtype] = sub_reasons

        # --- Ajuste por longitud del documento ---
        # Documentos cortos (<60 palabras) suelen ser credenciales, etiquetas o tarjetas,
        # no formularios ni contratos. Penalizar subtipos de documentos largos si el
        # texto es muy corto, y dar bonus a subtipos de documentos compactos.
        SHORT_DOC_THRESHOLD = 60
        if word_count < SHORT_DOC_THRESHOLD and word_count > 0:
            short_boost_types = {"credencial", "etiqueta", "tarjeta_presentacion", "ticket_transporte", "boleto"}
            short_penalize_types = {"formulario", "contrato", "acta", "certificado"}
            boost_factor = max(0.1, (SHORT_DOC_THRESHOLD - word_count) / SHORT_DOC_THRESHOLD)
            for st in short_boost_types:
                if st in raw_scores and raw_scores[st] > 0:
                    extra = raw_scores[st] * boost_factor * 0.4
                    raw_scores[st] += extra
                    reasons[st].append(f"Boost doc-corto ({word_count} words, +{extra:.0f})")
            for st in short_penalize_types:
                if st in raw_scores and raw_scores[st] > 0:
                    penalty = raw_scores[st] * boost_factor * 0.3
                    raw_scores[st] = max(0, raw_scores[st] - penalty)
                    reasons[st].append(f"Penalización doc-corto ({word_count} words, -{penalty:.0f})")

        if not raw_scores:
            return "desconocido", 0.3, {}, ["Sin subtipos candidatos"], False, ""

        # --- Normalizar scores a 0-1 ---
        max_raw = max(raw_scores.values())
        if max_raw < self.MIN_SUBTYPE_SCORE:
            return (
                "desconocido", 0.3, raw_scores,
                ["Score insuficiente para cualquier subtipo"],
                False, ""
            )

        scores: Dict[str, float] = {}
        for st, raw in raw_scores.items():
            scores[st] = raw / max_raw if max_raw > 0 else 0.0

        # Ordenar
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = sorted_types[0]
        second_type, second_score = sorted_types[1] if len(sorted_types) > 1 else ("", 0.0)

        # --- Gap analysis ---
        is_ambiguous = False
        ambiguity_note = ""
        gap = best_score - second_score

        if gap < self.MIN_GAP_RATIO and second_score > 0:
            is_ambiguous = True
            ambiguity_note = (
                f"Ambiguo: {best_type} ({best_score:.2f}) vs "
                f"{second_type} ({second_score:.2f}), gap={gap:.2f}"
            )
            logger.info(f"[Classifier/Subtype] {ambiguity_note}")

        # --- Confianza ---
        confidence = best_score
        if is_ambiguous:
            confidence *= 0.7

        # Si confianza < MIN_CONFIDENCE, no forzar
        if confidence < self.MIN_CONFIDENCE:
            return (
                "desconocido", confidence, scores,
                [f"Confianza insuficiente ({confidence:.2f} < {self.MIN_CONFIDENCE})"],
                is_ambiguous, ambiguity_note
            )

        return (
            best_type, round(confidence, 3), scores,
            reasons.get(best_type, []), is_ambiguous, ambiguity_note
        )


# ============================================================================
# TEMPORAL STABILIZER - Memoria entre frames
# ============================================================================

@dataclass
class _FrameEntry:
    """Una entrada en la memoria temporal de frames."""
    timestamp: float
    macro_type: str
    subtype: str
    confidence: float
    text_hash: str


class TemporalStabilizer:
    """
    Estabiliza la clasificación entre frames consecutivos.

    Cuando la cámara se mueve ligeramente, el OCR puede producir
    clasificaciones diferentes frame a frame. Este módulo:
    1. Mantiene una ventana de los últimos N resultados (3s)
    2. Si el tipo cambia pero el contenido es similar, mantiene el anterior
    3. Requiere M frames consistentes para cambiar de tipo
    """

    def __init__(self, window_seconds: float = 3.0, min_consistency: int = 2):
        self._entries: List[_FrameEntry] = []
        self._window = window_seconds
        self._min_consistency = min_consistency

    def stabilize(
        self,
        proposed_type: str,
        proposed_macro: str,
        confidence: float,
        text_hash: str,
    ) -> Tuple[str, str, bool, str]:
        """
        Estabiliza el tipo propuesto comparando con frames recientes.

        Returns:
            (final_type, final_macro, was_stabilized, note)
        """
        now = time.time()

        # Purgar entradas viejas
        self._entries = [
            e for e in self._entries
            if (now - e.timestamp) < self._window
        ]

        # Si no hay historial, aceptar el propuesto
        if not self._entries:
            self._entries.append(_FrameEntry(
                timestamp=now, macro_type=proposed_macro,
                subtype=proposed_type, confidence=confidence,
                text_hash=text_hash,
            ))
            return proposed_type, proposed_macro, False, ""

        # Contar tipos recientes
        recent_types = [e.subtype for e in self._entries]
        last_type = self._entries[-1].subtype
        last_hash = self._entries[-1].text_hash

        # Si el contenido es muy diferente, aceptar el cambio inmediatamente
        if text_hash != last_hash:
            self._entries.append(_FrameEntry(
                timestamp=now, macro_type=proposed_macro,
                subtype=proposed_type, confidence=confidence,
                text_hash=text_hash,
            ))
            return proposed_type, proposed_macro, False, ""

        # Contenido similar — estabilizar tipo
        if proposed_type != last_type:
            proposed_count = sum(1 for t in recent_types if t == proposed_type)

            if proposed_count < self._min_consistency:
                note = (
                    f"Estabilizado: propuesto={proposed_type}, "
                    f"mantenido={last_type} ({proposed_count}/{self._min_consistency})"
                )
                logger.info(f"[Classifier/Stabilizer] {note}")
                self._entries.append(_FrameEntry(
                    timestamp=now, macro_type=proposed_macro,
                    subtype=proposed_type, confidence=confidence,
                    text_hash=text_hash,
                ))
                last_macro = self._entries[-2].macro_type
                return last_type, last_macro, True, note

        # Aceptar el tipo propuesto
        self._entries.append(_FrameEntry(
            timestamp=now, macro_type=proposed_macro,
            subtype=proposed_type, confidence=confidence,
            text_hash=text_hash,
        ))
        return proposed_type, proposed_macro, False, ""


# ============================================================================
# HIERARCHICAL DOCUMENT CLASSIFIER - Orquestador
# ============================================================================

_LABELS: Dict[str, str] = {
    # DOCUMENTO_FORMAL
    "factura": "Factura",
    "recibo": "Recibo de pago",
    "carta": "Carta",
    "formulario": "Formulario",
    "contrato": "Contrato",
    "hoja_de_vida": "Hoja de vida",
    "informe": "Informe",
    "documento_informativo": "Documento informativo",
    "noticia": "Noticia o artículo",
    "correo": "Correo electrónico",
    "presentacion": "Presentación",
    "etiqueta": "Etiqueta de producto",
    "tarjeta": "Tarjeta de presentación",
    "receta_medica": "Receta médica",
    "boleto": "Boleto o ticket",
    "identificacion": "Documento de identificación",
    "horario": "Horario o agenda",
    "instrucciones": "Instrucciones de uso",
    "resultado_lab": "Resultado de laboratorio",
    "tabla_nutricional": "Tabla nutricional",
    "calendario": "Calendario",
    "factura_servicio": "Factura de servicio",
    "ticket_transporte": "Ticket de transporte",
    "credencial": "Credencial de empleado",

    # INTERFAZ_DIGITAL
    "app_menu": "Menú de aplicación",
    "app_settings": "Pantalla de configuración",
    "app_login": "Pantalla de inicio de sesión",
    "app_form": "Formulario digital",
    "app_social": "Red social",
    "app_service": "Servicio o compra",
    "app_banking": "Aplicación bancaria",
    "tarjeta_felicitacion": "Tarjeta de felicitación",
    "notificacion": "Notificación",
    "mapa": "Mapa",

    # TEXTO_CONVERSACIONAL
    "chat": "Conversación de chat",
    "comentario": "Comentarios",

    # IMAGEN_VISUAL
    "imagen_visual": "Imagen",

    # Fallbacks
    "desconocido": "Documento",
    "mixto": "Contenido mixto",

    # Legacy compatibility
    "red_social": "Red social",
    "login": "Pantalla de inicio de sesión",
    "configuracion": "Pantalla de configuración",
    "menu": "Menú",
}

# Umbral de texto insuficiente
_MIN_WORDS_FOR_CLASSIFICATION = 10
_MIN_DENSITY_FOR_CLASSIFICATION = 0.003


class HierarchicalDocumentClassifier:
    """
    Orquestador del clasificador jerárquico v3.

    Pipeline:
      OCR result (text + layout)
        → Pre-check: texto insuficiente? → IMAGEN_VISUAL + visual_description
        → normalize_for_classification(text)
        → LayoutAnalyzer.analyze(layout, dimensions)
        → Fase 1: MacroClassifier(layout_signals) → macro_type
        → Fase 2: SubtypeClassifier(clean_text, macro_type) → subtype
        → TemporalStabilizer(subtype) → final_type
        → Asignar reading_mode
        → ClassificationResult con explicación
    """

    def __init__(self):
        self._layout_analyzer = LayoutAnalyzer()
        self._macro_classifier = MacroClassifier()
        self._subtype_classifier = SubtypeClassifier()
        self._stabilizer = TemporalStabilizer()

    def classify(
        self,
        raw_text: str,
        word_count: int,
        layout_data: Dict,
        img_width: int,
        img_height: int,
        ocr_confidence: float,
    ) -> ClassificationResult:
        """
        Pipeline completo de clasificación jerárquica.

        Args:
            raw_text: Texto OCR crudo (sin limpiar)
            word_count: Número de palabras del OCR
            layout_data: Datos espaciales de Tesseract (word_boxes, etc.)
            img_width: Ancho de la imagen normalizada
            img_height: Alto de la imagen normalizada
            ocr_confidence: Confianza promedio del OCR

        Returns:
            ClassificationResult con tipo, confianza, reading_mode y explicación
        """
        explanation = ClassificationExplanation()

        # --- Paso 0: Analizar layout ---
        layout_signals = self._layout_analyzer.analyze(
            layout_data, img_width, img_height
        )

        # --- REGLA DE TEXTO INSUFICIENTE ---
        if self._is_insufficient_text(word_count, layout_signals.text_density):
            explanation.macro_type = "IMAGEN_VISUAL"
            explanation.macro_reasons = [
                f"Texto insuficiente: {word_count} palabras, "
                f"densidad={layout_signals.text_density:.4f}"
            ]
            explanation.final_type = "imagen_visual"
            explanation.final_confidence = 0.9

            logger.info(
                f"[Classifier/PreCheck] Texto insuficiente → IMAGEN_VISUAL "
                f"({word_count} palabras, densidad={layout_signals.text_density:.4f})"
            )

            return ClassificationResult(
                doc_type="imagen_visual",
                confidence=0.9,
                label=_LABELS["imagen_visual"],
                macro_type="IMAGEN_VISUAL",
                reading_mode="visual_description",
                explanation=explanation,
                layout_signals=layout_signals,
            )

        # --- Paso 1: Macro-clasificación (layout + boost semántico) ---
        macro_type, macro_conf, macro_reasons = (
            self._macro_classifier.classify(layout_signals, ocr_confidence, raw_text)
        )
        explanation.macro_type = macro_type
        explanation.macro_reasons = macro_reasons
        logger.info(
            f"[Classifier/Macro] {macro_type} (conf={macro_conf:.2f}): "
            f"{'; '.join(macro_reasons[:3])}"
        )

        # --- Paso 2: Limpiar texto ANTES de buscar keywords ---
        clean_text = normalize_for_classification(raw_text)

        # --- Paso 3: Sub-clasificación (keywords sobre texto limpio) ---
        (subtype, sub_conf, sub_scores, sub_reasons,
         is_ambiguous, ambiguity_note) = self._subtype_classifier.classify(
            clean_text, macro_type, word_count
        )
        explanation.subtype = subtype
        explanation.subtype_reasons = sub_reasons
        explanation.subtype_scores = sub_scores
        explanation.is_ambiguous = is_ambiguous
        explanation.ambiguity_note = ambiguity_note

        logger.info(
            f"[Classifier/Subtype] {subtype} (conf={sub_conf:.2f})"
            + (f" AMBIGUO: {ambiguity_note}" if is_ambiguous else "")
        )

        # --- Paso 4: Resolución de conflictos macro <-> subtipo ---
        final_type = self._resolve_conflicts(
            macro_type, macro_conf, subtype, sub_conf, sub_scores, explanation
        )

        # --- Paso 5: Estabilización temporal ---
        text_hash = _fast_hash(clean_text)
        final_type, final_macro, was_stabilized, stab_note = (
            self._stabilizer.stabilize(
                final_type, macro_type, sub_conf, text_hash
            )
        )
        if was_stabilized:
            explanation.was_stabilized = True
            explanation.stabilization_note = stab_note

        # --- Confianza final ---
        final_confidence = self._compute_final_confidence(
            macro_conf, sub_conf, is_ambiguous, was_stabilized
        )

        explanation.final_type = final_type
        explanation.final_confidence = final_confidence

        label = _LABELS.get(final_type, "Documento")

        # --- Asignar reading_mode ---
        reading_mode = _SUBTYPE_TO_READING_MODE.get(
            final_type,
            _MACRO_TO_DEFAULT_READING_MODE.get(final_macro, "paragraph_text")
        )

        logger.info(
            f"[Classifier/Final] {final_type} ({label}), "
            f"conf={final_confidence:.2f}, macro={final_macro}, "
            f"reading_mode={reading_mode}"
        )

        return ClassificationResult(
            doc_type=final_type,
            confidence=final_confidence,
            label=label,
            macro_type=final_macro,
            reading_mode=reading_mode,
            explanation=explanation,
            layout_signals=layout_signals,
        )

    def _is_insufficient_text(self, word_count: int, text_density: float) -> bool:
        """
        Determina si el texto es insuficiente para clasificación completa.
        Si es así, se clasifica directamente como IMAGEN_VISUAL.
        """
        return (
            word_count < _MIN_WORDS_FOR_CLASSIFICATION
            or text_density < _MIN_DENSITY_FOR_CLASSIFICATION
        )

    def _resolve_conflicts(
        self,
        macro_type: str,
        macro_conf: float,
        subtype: str,
        sub_conf: float,
        sub_scores: Dict[str, float],
        explanation: ClassificationExplanation,
    ) -> str:
        """
        Resuelve conflictos entre macro-tipo y subtipo.

        Estrategia:
          1. Sin conflicto → usar subtipo directamente
          2. Conflicto → buscar el mejor subtipo DENTRO del macro-tipo
          3. Si el mejor dentro del macro tiene score competitivo
             (>= 70% del score del subtipo ganador), preferir el macro
          4. Solo hacer override del macro cuando el subtipo foráneo
             es SIGNIFICATIVAMENTE mejor que cualquier subtipo del macro
        """
        # Si el subtipo pertenece al macro-tipo, no hay conflicto
        expected_subtypes = _MACRO_TO_SUBTYPES.get(macro_type, [])
        if subtype in expected_subtypes or subtype == "desconocido":
            return subtype

        # Conflicto: el subtipo no pertenece al macro-tipo
        subtype_score = sub_scores.get(subtype, 0)

        # Buscar el mejor subtipo DENTRO del macro-tipo esperado
        best_in_macro = "desconocido"
        best_in_macro_score = 0.0
        for st in expected_subtypes:
            sc = sub_scores.get(st, 0.0)
            if sc > best_in_macro_score:
                best_in_macro = st
                best_in_macro_score = sc

        # Si hay un buen candidato dentro del macro, preferirlo
        # a menos que el foráneo sea MUY superior (>30% más)
        if best_in_macro_score > 0.3:
            if best_in_macro_score >= subtype_score * 0.7:
                # El candidato del macro es competitivo → usar el del macro
                explanation.macro_reasons.append(
                    f"Conflicto resuelto: {best_in_macro} "
                    f"(score={best_in_macro_score:.2f}) dentro de {macro_type} "
                    f"preferido sobre {subtype} (score={subtype_score:.2f})"
                )
                return best_in_macro

        # El subtipo foráneo es significativamente mejor
        if subtype_score > 0.7 and subtype_score > best_in_macro_score * 1.5:
            explanation.macro_reasons.append(
                f"OVERRIDE: subtipo {subtype} (score={subtype_score:.2f}) "
                f"prevalece sobre macro {macro_type} "
                f"(muy superior al mejor del macro: {best_in_macro}={best_in_macro_score:.2f})"
            )
            return subtype

        # Fallback: usar el mejor del macro si tiene algo
        if best_in_macro_score > 0.3:
            return best_in_macro

        return "desconocido"

    def _compute_final_confidence(
        self,
        macro_conf: float,
        sub_conf: float,
        is_ambiguous: bool,
        was_stabilized: bool,
    ) -> float:
        """Combina las confianzas de ambas fases en una confianza final."""
        combined = macro_conf * 0.35 + sub_conf * 0.65

        if is_ambiguous:
            combined *= 0.75
        if was_stabilized:
            combined *= 0.9

        return round(min(max(combined, 0.1), 0.99), 2)

    @staticmethod
    def get_label(doc_type: str) -> str:
        """Retorna la etiqueta legible para un tipo de documento."""
        return _LABELS.get(doc_type, "Documento")


# ============================================================================
# HELPERS
# ============================================================================

def _fast_hash(text: str) -> str:
    """Hash rápido para comparar textos entre frames."""
    import hashlib
    normalized = re.sub(r'[^\w]', '', text.lower())
    return hashlib.md5(normalized.encode()).hexdigest()


# ============================================================================
# SINGLETON
# ============================================================================

_classifier_instance: Optional[HierarchicalDocumentClassifier] = None


def get_document_classifier() -> HierarchicalDocumentClassifier:
    """Factory function para obtener el clasificador (singleton)."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = HierarchicalDocumentClassifier()
    return _classifier_instance
