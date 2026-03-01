"""
============================================================================
NAVIA Backend - Optimizador de Lectura para TTS
============================================================================
Módulo de post-procesamiento que optimiza la narrativa generada para
una experiencia auditiva superior.

4 sistemas:
1. Lectura Jerárquica: contenido principal primero, metadatos después
2. Memoria Contextual: evita repetir lo mismo si la escena no cambia
3. Manejo de Incertidumbre: comunica al usuario cuando el OCR no es claro
4. Optimización TTS: frases cortas, pausas naturales, sin sobrecarga

Se aplica DESPUÉS del NarrativeGenerator y ANTES del ProsodyEnhancer.
============================================================================
"""

import re
import time
import hashlib
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# 1. LECTURA JERÁRQUICA
# ============================================================================

# Prioridad de campos por tipo de documento.
# Cada tipo define qué información es ESENCIAL (se lee siempre)
# y qué es SECUNDARIA (solo en modo detallado).
_CONTENT_PRIORITY: Dict[str, Dict[str, List[str]]] = {
    "factura": {
        "esencial": ["totals", "amounts"],
        "secundario": ["dates", "ids", "headers", "emails", "phones"],
    },
    "recibo": {
        "esencial": ["amounts", "totals"],
        "secundario": ["dates", "ids", "phones"],
    },
    "carta": {
        "esencial": ["body"],
        "secundario": ["dates", "emails", "phones", "headers"],
    },
    "formulario": {
        "esencial": ["headers"],
        "secundario": ["dates", "ids"],
    },
    "documento_informativo": {
        "esencial": ["headers", "body"],
        "secundario": ["dates"],
    },
    "etiqueta": {
        "esencial": ["body"],
        "secundario": ["dates", "amounts"],
    },
    "menu": {
        "esencial": ["body"],
        "secundario": ["amounts"],
    },
    "tarjeta": {
        "esencial": ["headers", "phones", "emails"],
        "secundario": [],
    },
    "chat": {
        "esencial": ["body"],
        "secundario": [],
    },
    "notificacion": {
        "esencial": ["body"],
        "secundario": [],
    },
    "login": {
        "esencial": ["body"],
        "secundario": [],
    },
    "red_social": {
        "esencial": ["body"],
        "secundario": ["amounts"],
    },
    "noticia": {
        "esencial": ["headers", "body"],
        "secundario": ["dates"],
    },
    "correo": {
        "esencial": ["headers", "body"],
        "secundario": ["dates", "emails"],
    },
    "presentacion": {
        "esencial": ["headers", "body"],
        "secundario": [],
    },
    "configuracion": {
        "esencial": ["body"],
        "secundario": [],
    },
}


class HierarchicalReader:
    """
    Reordena la narrativa para que el contenido principal se lea primero.

    Estrategia:
    - Detecta secciones de la narrativa (intro, datos, cuerpo)
    - Mueve el cuerpo principal al frente, metadatos al final
    - En modo resumen, omite metadatos secundarios
    """

    # Patrones que indican metadatos (no contenido principal)
    _METADATA_PATTERNS = [
        re.compile(r'^(?:Fecha|Fechada?)\s*(?:de[l]?\s+\w+)?:', re.IGNORECASE),
        re.compile(r'^(?:Correo|Email|Tel[ée]fono|Identificaci[oó]n)\s*:', re.IGNORECASE),
        re.compile(r'^(?:N[uú]mero|Comprobante)\s*:', re.IGNORECASE),
        re.compile(r'^Fuente:', re.IGNORECASE),
        re.compile(r'^Diapositiva\s+\d+', re.IGNORECASE),
    ]

    # Patrones que indican contenido principal
    _CONTENT_PATTERNS = [
        re.compile(r'^(?:Dice|El (?:texto|contenido|mensaje)\s+dice|Los mensajes dicen|Contenido)', re.IGNORECASE),
        re.compile(r'^(?:Título|T[ií]tulo principal)', re.IGNORECASE),
        re.compile(r'^(?:Opciones visibles|Tiene campos|Los últimos mensajes)', re.IGNORECASE),
        re.compile(r'^(?:Monto|Total|El monto)', re.IGNORECASE),
        re.compile(r'^Hashtags:', re.IGNORECASE),
    ]

    def reorder(self, narrative: str, doc_type: str, reading_mode: str) -> str:
        """
        Reordena la narrativa para priorizar contenido principal.

        En modo 'resumen': intro + contenido esencial (máx ~40 palabras)
        En modo 'detallado'/'financiero': intro + contenido + metadatos
        """
        if not narrative:
            return narrative

        # Separar en oraciones
        sentences = self._split_sentences(narrative)
        if len(sentences) <= 2:
            return narrative  # Muy corta, no reordenar

        intro = []       # "Esto es una factura."
        content = []     # Contenido principal (montos, texto, mensajes)
        metadata = []    # Fechas, correos, teléfonos, IDs

        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue

            if self._is_intro(s, doc_type):
                intro.append(s)
            elif self._is_metadata(s):
                metadata.append(s)
            else:
                content.append(s)

        # En resumen: solo intro + contenido principal
        if reading_mode == "resumen":
            parts = intro + content
            # Limitar a ~50 palabras total
            result = self._limit_words(" ".join(parts), max_words=50)
            return result

        # Detallado/financiero: intro + contenido + metadatos
        parts = intro + content + metadata
        return " ".join(parts)

    def _split_sentences(self, text: str) -> List[str]:
        """Divide texto en oraciones respetando abreviaciones."""
        # Split por punto seguido de espacio y mayúscula, o punto final
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def _is_intro(self, sentence: str, doc_type: str) -> bool:
        """Detecta si una oración es la intro del tipo de documento."""
        intro_patterns = [
            r'^(?:Esto|Este|Esta)\s+(?:es|documento)',
            r'^(?:Se\s+detect[oó]|Es\s+una?)',
            r'^(?:No\s+se\s+detect[oó]|Imagen)',
        ]
        return any(re.match(p, sentence, re.IGNORECASE) for p in intro_patterns)

    def _is_metadata(self, sentence: str) -> bool:
        """Detecta si una oración contiene metadatos secundarios."""
        return any(p.match(sentence) for p in self._METADATA_PATTERNS)

    def _limit_words(self, text: str, max_words: int) -> str:
        """Limita el texto a max_words, cortando en límite de oración."""
        words = text.split()
        if len(words) <= max_words:
            return text

        # Cortar en la oración más cercana al límite
        truncated = " ".join(words[:max_words])
        # Buscar el último punto antes del corte
        last_period = truncated.rfind('.')
        if last_period > len(truncated) * 0.5:
            return truncated[:last_period + 1]
        return truncated + "."


# ============================================================================
# 2. MEMORIA CONTEXTUAL DE CORTO PLAZO
# ============================================================================

@dataclass
class ReadingMemoryEntry:
    """Una entrada en la memoria de lecturas recientes."""
    timestamp: float
    doc_type: str
    text_hash: str
    narrative_hash: str
    word_count: int
    key_content: str  # Primeras 100 palabras del texto limpio


class ContextualMemory:
    """
    Memoria de corto plazo que evita repeticiones innecesarias.

    Guarda las últimas N lecturas y compara la nueva con las anteriores.
    Si el contenido es muy similar (misma imagen o escena sin cambios),
    informa al usuario en lugar de repetir todo.

    Ventana: últimas 5 lecturas en los últimos 60 segundos.
    """

    def __init__(self, max_entries: int = 5, ttl_seconds: float = 60.0):
        self._entries: List[ReadingMemoryEntry] = []
        self._max_entries = max_entries
        self._ttl = ttl_seconds

    def check_and_update(
        self,
        raw_text: str,
        doc_type: str,
        word_count: int,
        narrative: str,
    ) -> Optional[str]:
        """
        Verifica si esta lectura es una repetición y actualiza la memoria.

        Returns:
            None si es contenido nuevo (proceder normalmente).
            str con mensaje alternativo si es repetición.
        """
        now = time.time()

        # Limpiar entradas expiradas
        self._entries = [
            e for e in self._entries
            if (now - e.timestamp) < self._ttl
        ]

        # Calcular hashes del contenido actual
        text_hash = self._hash_text(raw_text)
        narrative_hash = self._hash_text(narrative)
        key_content = " ".join(raw_text.split()[:100]) if raw_text else ""

        # Buscar coincidencias en memoria
        for entry in reversed(self._entries):
            similarity = self._compute_similarity(text_hash, entry.text_hash,
                                                   key_content, entry.key_content)

            if similarity > 0.85:
                # Contenido prácticamente idéntico
                elapsed = now - entry.timestamp
                if elapsed < 5.0:
                    # Menos de 5 segundos → probablemente el mismo frame
                    msg = "Es el mismo contenido que acabo de leer."
                    logger.info(f"[Memory] Repetición exacta ({elapsed:.1f}s)")
                    self._add_entry(now, doc_type, text_hash, narrative_hash,
                                    word_count, key_content)
                    return msg
                elif elapsed < 30.0:
                    # Misma escena en menos de 30s
                    msg = "El contenido no ha cambiado desde la última lectura."
                    logger.info(f"[Memory] Sin cambios ({elapsed:.1f}s)")
                    self._add_entry(now, doc_type, text_hash, narrative_hash,
                                    word_count, key_content)
                    return msg

            elif similarity > 0.60:
                # Parcialmente similar → contenido cambió un poco
                elapsed = now - entry.timestamp
                if elapsed < 15.0 and entry.doc_type == doc_type:
                    logger.info(f"[Memory] Contenido similar ({similarity:.0%})")
                    # No bloquear, pero no repetir intro
                    # (esto se maneja en el caller)

        # Contenido nuevo → guardar en memoria
        self._add_entry(now, doc_type, text_hash, narrative_hash,
                        word_count, key_content)
        return None

    def get_last_doc_type(self) -> Optional[str]:
        """Retorna el tipo de documento de la última lectura."""
        if self._entries:
            return self._entries[-1].doc_type
        return None

    def is_same_type_as_last(self, doc_type: str) -> bool:
        """Verifica si el tipo actual es igual al de la última lectura."""
        last = self.get_last_doc_type()
        return last is not None and last == doc_type

    def _add_entry(self, timestamp: float, doc_type: str, text_hash: str,
                   narrative_hash: str, word_count: int, key_content: str):
        self._entries.append(ReadingMemoryEntry(
            timestamp=timestamp,
            doc_type=doc_type,
            text_hash=text_hash,
            narrative_hash=narrative_hash,
            word_count=word_count,
            key_content=key_content,
        ))
        # Mantener solo las últimas N entradas
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    @staticmethod
    def _hash_text(text: str) -> str:
        """Hash normalizado del texto (ignora espacios, mayúsculas, puntuación)."""
        if not text:
            return ""
        normalized = re.sub(r'[^\w]', '', text.lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    @staticmethod
    def _compute_similarity(hash_a: str, hash_b: str,
                            text_a: str, text_b: str) -> float:
        """
        Calcula similitud entre dos textos.
        Usa hash exacto + similitud de palabras como fallback.
        """
        if not hash_a or not hash_b:
            return 0.0

        # Hash exacto = 100% similar
        if hash_a == hash_b:
            return 1.0

        # Similitud por palabras compartidas (Jaccard)
        if not text_a or not text_b:
            return 0.0

        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union) if union else 0.0


# ============================================================================
# 3. MANEJO DE INCERTIDUMBRE
# ============================================================================

class UncertaintyHandler:
    """
    Detecta baja confianza del OCR y comunica incertidumbre al usuario.

    Niveles:
    - Alta confianza (>=75%): Lee normalmente
    - Media confianza (50-74%): Prefijo sutil "Parece decir..."
    - Baja confianza (30-49%): "No se lee muy bien, pero parece decir..."
    - Muy baja (<30%): "No se pudo leer el texto claramente."
    """

    # Umbrales de confianza
    HIGH_CONFIDENCE = 75.0
    MEDIUM_CONFIDENCE = 50.0
    LOW_CONFIDENCE = 30.0

    def apply(
        self,
        narrative: str,
        ocr_confidence: Optional[float],
        word_count: int,
        doc_type: str,
    ) -> str:
        """
        Modifica la narrativa según el nivel de confianza del OCR.

        No toca la intro ("Esto es una factura.") ni el tipo de documento.
        Solo modifica la parte del contenido leído.
        """
        if not narrative or ocr_confidence is None:
            return narrative

        # Muy baja confianza + pocas palabras = no vale la pena leer
        if ocr_confidence < self.LOW_CONFIDENCE and word_count < 10:
            return self._replace_content_with_uncertainty(narrative, doc_type,
                                                          "very_low")

        # Baja confianza
        if ocr_confidence < self.MEDIUM_CONFIDENCE:
            return self._add_uncertainty_prefix(narrative, "low")

        # Media confianza
        if ocr_confidence < self.HIGH_CONFIDENCE:
            return self._add_uncertainty_prefix(narrative, "medium")

        # Alta confianza: no modificar
        return narrative

    def _add_uncertainty_prefix(self, narrative: str, level: str) -> str:
        """Agrega indicador de incertidumbre al contenido (no a la intro)."""
        # Separar intro del contenido
        sentences = re.split(r'(?<=[.!?])\s+', narrative, maxsplit=1)

        if len(sentences) < 2:
            # Solo una oración
            if level == "low":
                return f"No se lee muy bien, pero: {narrative}"
            return narrative

        intro = sentences[0]
        content = sentences[1]

        # Eliminar "Dice:" redundante del contenido si existe
        # (ya que vamos a agregar "Parece decir:")
        content = re.sub(r'(?:^|\.\s*)(?:Dice|El texto dice|El contenido dice)\s*:\s*',
                         '. ', content, flags=re.IGNORECASE).strip()
        content = re.sub(r'^\.\s*', '', content)  # Limpiar punto inicial

        if level == "low":
            prefix = "No se lee con total claridad, pero parece decir:"
        else:  # medium
            prefix = "Parece decir:"

        return f"{intro} {prefix} {content}"

    def _replace_content_with_uncertainty(self, narrative: str,
                                          doc_type: str, level: str) -> str:
        """Reemplaza el contenido con mensaje de incertidumbre."""
        # Mantener la intro si existe
        sentences = re.split(r'(?<=[.!?])\s+', narrative, maxsplit=1)
        intro = sentences[0] if sentences else ""

        msg = ("No se pudo leer el texto con claridad. "
               "Intenta con mejor iluminación o acercando más la cámara.")

        if intro and self._is_intro(intro):
            return f"{intro} {msg}"
        return msg

    @staticmethod
    def _is_intro(sentence: str) -> bool:
        return bool(re.match(
            r'^(?:Esto|Este|Esta|Es\s+una?|Se\s+detect[oó])',
            sentence, re.IGNORECASE
        ))


# ============================================================================
# 4. OPTIMIZADOR TTS
# ============================================================================

class TTSOptimizer:
    """
    Optimiza el texto para síntesis de voz natural con Piper TTS.

    Principios:
    - Frases cortas (máx 15-20 palabras por oración)
    - Pausas lógicas entre secciones (punto + espacio)
    - Evitar listas largas (máx 4 elementos, luego "y más")
    - Evitar texto redundante
    - Controlar ritmo: info importante = más lenta (más puntos)
    """

    MAX_SENTENCE_WORDS = 20
    MAX_LIST_ITEMS = 4

    def optimize(self, narrative: str, doc_type: str) -> str:
        """Pipeline de optimización TTS."""
        if not narrative:
            return narrative

        text = narrative

        # 1. Eliminar redundancias
        text = self._remove_redundancy(text)

        # 2. Acortar oraciones largas
        text = self._shorten_sentences(text)

        # 3. Truncar listas largas
        text = self._truncate_lists(text)

        # 4. Agregar pausas lógicas entre secciones
        text = self._add_section_pauses(text, doc_type)

        # 5. Limpieza final
        text = self._final_cleanup(text)

        return text

    def _remove_redundancy(self, text: str) -> str:
        """Elimina frases que repiten información ya dicha."""
        # "Esto es una factura. Este documento es una factura." → quitar la segunda
        text = re.sub(
            r'(?<=\.)\s+(?:Este\s+documento\s+es|Esto\s+es)\s+(?:una?\s+)?\w+\.',
            '',
            text,
            count=1
        )

        # Eliminar "Se detectó texto." si después viene contenido útil
        if re.search(r'Se\s+detect[oó]\s+(?:texto|un\s+documento)\.\s+\w', text):
            # Solo si hay más contenido después
            parts = re.split(r'Se\s+detect[oó]\s+(?:texto|un\s+documento)\.\s*', text, maxsplit=1)
            if len(parts) == 2 and len(parts[1].split()) > 5:
                text = parts[1]

        return text

    def _shorten_sentences(self, text: str) -> str:
        """Divide oraciones de más de MAX_SENTENCE_WORDS palabras."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) <= self.MAX_SENTENCE_WORDS:
                result.append(sentence)
                continue

            # Buscar punto de corte natural
            cut_points = [
                # Conjunciones
                (r'\s+(y|pero|sin embargo|además|también|aunque|porque|donde|cuando)\s+',
                 r'. \1 '),
                # Dos puntos o punto y coma
                (r'\s*[:;]\s*', '. '),
                # Coma después de ~10 palabras
                (r',\s+', '. '),
            ]

            shortened = sentence
            for pattern, replacement in cut_points:
                # Solo cortar si el resultado tendría oraciones más cortas
                candidate = re.sub(pattern, replacement, shortened, count=1,
                                   flags=re.IGNORECASE)
                parts = candidate.split('. ')
                if all(len(p.split()) <= self.MAX_SENTENCE_WORDS + 5 for p in parts):
                    shortened = candidate
                    break

            result.append(shortened)

        return " ".join(result)

    def _truncate_lists(self, text: str) -> str:
        """Si hay una enumeración larga, la trunca a MAX_LIST_ITEMS."""
        # Detectar patrones como "Opciones visibles: A, B, C, D, E, F, G."
        def truncate_list(match):
            prefix = match.group(1)
            items_str = match.group(2)
            items = [i.strip() for i in items_str.split(',') if i.strip()]

            if len(items) <= self.MAX_LIST_ITEMS:
                return match.group(0)

            kept = items[:self.MAX_LIST_ITEMS]
            remaining = len(items) - self.MAX_LIST_ITEMS
            return f"{prefix}{', '.join(kept)}, y {remaining} más."

        # Listas después de dos puntos
        text = re.sub(
            r'((?:Opciones visibles|Montos?(?:\s+encontrados)?|Hashtags?|Otros montos)\s*:\s*)'
            r'([^.]+\.)',
            truncate_list,
            text
        )

        return text

    def _add_section_pauses(self, text: str, doc_type: str) -> str:
        """Agrega pausas entre secciones semánticas para mejor prosodia."""
        # Antes de "El texto dice:" agregar pausa
        text = re.sub(
            r'(?<=[.])\s+((?:El\s+)?(?:texto|contenido|mensaje)\s+dice:)',
            r'... \1',
            text,
            flags=re.IGNORECASE
        )

        # Antes de "Parece decir:" agregar pausa
        text = re.sub(
            r'(?<=[.])\s+((?:Parece|No se lee)\s)',
            r'... \1',
            text,
            flags=re.IGNORECASE
        )

        # Para facturas: pausa antes del total
        if doc_type in ("factura", "recibo"):
            text = re.sub(
                r'(?<=[.])\s+((?:El\s+)?(?:monto|total)\s)',
                r'... \1',
                text,
                flags=re.IGNORECASE
            )

        return text

    def _final_cleanup(self, text: str) -> str:
        """Limpieza final del texto optimizado."""
        # Múltiples puntos seguidos → máximo 3
        text = re.sub(r'\.{4,}', '...', text)
        # Espacio antes de punto
        text = re.sub(r'\s+\.', '.', text)
        # Múltiples espacios
        text = re.sub(r'\s{2,}', ' ', text)
        # Punto doble
        text = re.sub(r'\.\.(?!\.)', '.', text)
        # Asegurar que termina en punto
        text = text.strip()
        if text and text[-1] not in '.!?':
            text += '.'
        return text


# ============================================================================
# ORQUESTADOR: ReadingOptimizer
# ============================================================================

class ReadingOptimizer:
    """
    Orquestador que aplica los 4 sistemas de optimización en orden.

    Pipeline:
      narrative (raw) → UncertaintyHandler → HierarchicalReader
                      → TTSOptimizer → resultado final

    La ContextualMemory se consulta ANTES de generar narrativa
    para evitar procesamiento innecesario.
    """

    def __init__(self):
        self.hierarchy = HierarchicalReader()
        self.memory = ContextualMemory()
        self.uncertainty = UncertaintyHandler()
        self.tts = TTSOptimizer()

    def check_memory(
        self,
        raw_text: str,
        doc_type: str,
        word_count: int,
        narrative: str,
    ) -> Optional[str]:
        """
        Consulta la memoria contextual ANTES de generar.
        Retorna mensaje de repetición o None si es contenido nuevo.
        """
        return self.memory.check_and_update(
            raw_text, doc_type, word_count, narrative
        )

    def optimize(
        self,
        narrative: str,
        doc_type: str,
        reading_mode: str,
        ocr_confidence: Optional[float],
        word_count: int,
    ) -> str:
        """
        Aplica todos los optimizadores a la narrativa.

        Args:
            narrative: Texto generado por NarrativeGenerator
            doc_type: Tipo de documento clasificado
            reading_mode: "resumen", "detallado", "financiero"
            ocr_confidence: Confianza del OCR (0-100)
            word_count: Cantidad de palabras detectadas

        Returns:
            Narrativa optimizada para TTS
        """
        if not narrative:
            return narrative

        # 1. Manejo de incertidumbre (agregar prefijos si OCR bajo)
        text = self.uncertainty.apply(narrative, ocr_confidence,
                                      word_count, doc_type)

        # 2. Lectura jerárquica (reordenar: contenido primero)
        text = self.hierarchy.reorder(text, doc_type, reading_mode)

        # 3. Optimización TTS (frases cortas, pausas, sin redundancia)
        text = self.tts.optimize(text, doc_type)

        return text


# ============================================================================
# SINGLETON
# ============================================================================

_reading_optimizer: Optional[ReadingOptimizer] = None


def get_reading_optimizer() -> ReadingOptimizer:
    """Factory function para obtener el optimizador (singleton)."""
    global _reading_optimizer
    if _reading_optimizer is None:
        _reading_optimizer = ReadingOptimizer()
    return _reading_optimizer
