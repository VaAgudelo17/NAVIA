"""
============================================================================
NAVIA Backend - Servicio de Lectura Inteligente
============================================================================
Transforma OCR crudo en narrativa natural para usuarios ciegos.

Pipeline:
  Imagen → Tesseract OCR → DocumentClassifier → StructureExtractor
         → NarrativeGenerator → ProsodyEnhancer → SmartReadingResponse

Clasificación basada en scoring de keywords (sin LLMs).
Narrativas generadas con templates por tipo de documento × modo de lectura.
Prosodia optimizada para Piper TTS (sin SSML, usa puntuación).
============================================================================
"""

import re
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# DOCUMENT CLASSIFIER
# ============================================================================

_CLASSIFICATION_RULES: Dict[str, List[Tuple[re.Pattern, int]]] = {
    "factura": [
        (re.compile(r'\bfactura\b', re.IGNORECASE), 10),
        (re.compile(r'\bR\.?I\.?F\.?\b', re.IGNORECASE), 6),
        (re.compile(r'\bN\.?I\.?T\.?\b', re.IGNORECASE), 6),
        (re.compile(r'\bN\.?I\.?F\.?\b', re.IGNORECASE), 5),
        (re.compile(r'\btotal\s*:?\s*[\$\€Bs\.]*\s*[\d,.]+', re.IGNORECASE), 7),
        (re.compile(r'\bsubtotal\b', re.IGNORECASE), 5),
        (re.compile(r'\bi\.?v\.?a\.?\b', re.IGNORECASE), 6),
        (re.compile(r'\bimpuesto\b', re.IGNORECASE), 4),
        (re.compile(r'\bn[úu]mero\s+de\s+factura\b', re.IGNORECASE), 8),
        (re.compile(r'\bcantidad\b.*\bprecio\b', re.IGNORECASE | re.DOTALL), 5),
        (re.compile(r'\bdescripci[óo]n\b.*\bunitario\b', re.IGNORECASE | re.DOTALL), 5),
        (re.compile(r'\bemitido\s+por\b', re.IGNORECASE), 3),
    ],
    "recibo": [
        (re.compile(r'\brecibo\b', re.IGNORECASE), 10),
        (re.compile(r'\brecib[oí]\s+de\b', re.IGNORECASE), 10),
        (re.compile(r'\bpagado\b', re.IGNORECASE), 6),
        (re.compile(r'\bconcepto\b', re.IGNORECASE), 4),
        (re.compile(r'\bmonto\s+pagado\b', re.IGNORECASE), 7),
        (re.compile(r'\bfecha\s+de\s+pago\b', re.IGNORECASE), 5),
        (re.compile(r'\bcomprobante\b', re.IGNORECASE), 6),
        (re.compile(r'\brecibo\s+de\s+caja\b', re.IGNORECASE), 9),
        (re.compile(r'\bcancel[oó]\b', re.IGNORECASE), 4),
        (re.compile(r'\bpago\b', re.IGNORECASE), 4),
    ],
    "carta": [
        (re.compile(r'\bestimat[oa]\b', re.IGNORECASE), 8),
        (re.compile(r'\bse[ñn]or[ae]?\b', re.IGNORECASE), 6),
        (re.compile(r'\batentamente\b', re.IGNORECASE), 9),
        (re.compile(r'\bcordialmente\b', re.IGNORECASE), 9),
        (re.compile(r'\bme\s+dirijo\b', re.IGNORECASE), 8),
        (re.compile(r'\bpor\s+medio\s+de\b', re.IGNORECASE), 5),
        (re.compile(r'\bla\s+presente\b', re.IGNORECASE), 6),
        (re.compile(r'\ba\s+quien\s+corresponda\b', re.IGNORECASE), 9),
        (re.compile(r'\bsaludo[s]?\b', re.IGNORECASE), 5),
        (re.compile(r'\bfirm[aó]\b', re.IGNORECASE), 3),
    ],
    "formulario": [
        (re.compile(r'\bformulario\b', re.IGNORECASE), 10),
        (re.compile(r'\bsolicitante\b', re.IGNORECASE), 7),
        (re.compile(r'\bapellido\b.*\bnombre\b', re.IGNORECASE | re.DOTALL), 6),
        (re.compile(r'\bc[eé]dula\s+de\s+identidad\b', re.IGNORECASE), 8),
        (re.compile(r'\bfecha\s+de\s+nacimiento\b', re.IGNORECASE), 7),
        (re.compile(r'\bmarque\s+con\b', re.IGNORECASE), 6),
        (re.compile(r'\bllene\b', re.IGNORECASE), 6),
        (re.compile(r'\bsolicitud\b', re.IGNORECASE), 5),
        (re.compile(r'\bfirma\s+del?\s*(solicitante|titular)\b', re.IGNORECASE), 7),
    ],
    "documento_informativo": [
        (re.compile(r'\bart[íi]culo\s+\d+\b', re.IGNORECASE), 7),
        (re.compile(r'\bsecci[óo]n\s+\d+\b|\bcap[íi]tulo\b', re.IGNORECASE), 6),
        (re.compile(r'\binformaci[óo]n\b|\baviso\b', re.IGNORECASE), 5),
        (re.compile(r'\bcomunicado\b', re.IGNORECASE), 8),
        (re.compile(r'\bresoluci[óo]n\b', re.IGNORECASE), 6),
        (re.compile(r'\bregulaci[óo]n\b|\bnorma\b', re.IGNORECASE), 5),
        (re.compile(r'\bpublicado\b', re.IGNORECASE), 4),
        (re.compile(r'\boficial\b', re.IGNORECASE), 3),
    ],
}

_IMAGE_INDICATORS = [
    re.compile(r'\bwww\.\w+\.\w+\b', re.IGNORECASE),
]

_SCORE_THRESHOLD = 5
_LOW_TEXT_THRESHOLD = 5


class DocumentClassifier:
    """Clasificador de documentos basado en scoring de keywords."""

    def classify(self, text: str, word_count: int) -> Tuple[str, float]:
        if not text or word_count == 0:
            return "desconocido", 0.0

        # Solo clasificar como imagen_visual si hay MUY pocas palabras
        # y ninguna señal de documento
        if word_count < _LOW_TEXT_THRESHOLD:
            image_score = sum(2 for pat in _IMAGE_INDICATORS if pat.search(text))
            if image_score > 0 or word_count < 3:
                return "imagen_visual", min(0.5 + image_score * 0.1, 0.9)

        scores: Dict[str, float] = {}
        for doc_type, rules in _CLASSIFICATION_RULES.items():
            total = 0.0
            for pattern, weight in rules:
                matches = len(pattern.findall(text))
                total += matches * weight
            scores[doc_type] = total

        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]

        # Log scores para debug
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"[Classifier] Scores: {top_scores}")

        if best_score < _SCORE_THRESHOLD:
            return "desconocido", 0.3

        confidence = min(0.5 + (best_score / 100.0), 0.99)
        return best_type, confidence

    @staticmethod
    def get_label(doc_type: str) -> str:
        labels = {
            "factura": "Factura",
            "recibo": "Recibo de pago",
            "carta": "Carta",
            "formulario": "Formulario",
            "documento_informativo": "Documento informativo",
            "imagen_visual": "Imagen con texto",
            "desconocido": "Documento",
        }
        return labels.get(doc_type, "Documento")


# ============================================================================
# STRUCTURE EXTRACTOR
# ============================================================================

@dataclass
class ExtractedData:
    dates: List[str] = field(default_factory=list)
    amounts: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    totals: List[str] = field(default_factory=list)


_PAT_DATE = re.compile(
    r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
    r'|\b(\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|'
    r'septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?\d{4})\b',
    re.IGNORECASE
)

_PAT_AMOUNT = re.compile(
    r'(?:[\$\€£]+\s*[\d]{1,3}(?:[,\.\s]\d{3})*(?:[,\.]\d{1,2})?)'
    r'|(?:Bs\.?\s*[\d]{1,3}(?:[,\.\s]\d{3})*(?:[,\.]\d{1,2})?)'
    r'|(?:(?:USD|EUR|VES|COP|MXN)\s*[\d,\.]+)',
    re.IGNORECASE
)

_PAT_EMAIL = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')

_PAT_PHONE = re.compile(
    r'(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}'
)

_PAT_ID = re.compile(
    r'\b(?:R\.?I\.?F\.?|N\.?I\.?T\.?|N\.?I\.?F\.?|C\.?I\.?|[Cc][ée]dula)\s*[:\-]?\s*'
    r'[VEJGvejg]?\-?\s*\d[\d\-]{5,12}\b',
    re.IGNORECASE
)

_PAT_HEADER = re.compile(
    r'^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9\s\-\.]{2,}$',
    re.MULTILINE
)

_PAT_TOTAL = re.compile(
    r'^.*\b(?:total|subtotal|gran\s+total|monto\s+total|neto|bruto)\b.*$',
    re.IGNORECASE | re.MULTILINE
)


class StructureExtractor:
    """Extrae campos estructurados del texto OCR usando regex."""

    def extract(self, text: str) -> ExtractedData:
        data = ExtractedData()
        if not text:
            return data

        # Fechas
        data.dates = list(dict.fromkeys(
            m.group(0).strip()
            for m in _PAT_DATE.finditer(text)
            if m.group(0) and m.group(0).strip()
        ))[:5]

        # Montos
        data.amounts = list(dict.fromkeys(
            m.group(0).strip()
            for m in _PAT_AMOUNT.finditer(text)
            if m.group(0) and m.group(0).strip()
        ))[:10]

        # Emails
        data.emails = list(dict.fromkeys(_PAT_EMAIL.findall(text)))[:3]

        # IDs (RIF, NIT, cédula) - ANTES de phones para filtrar falsos positivos
        data.ids = list(dict.fromkeys(
            m.group(0).strip()
            for m in _PAT_ID.finditer(text)
        ))[:3]

        # Teléfonos (mínimo 7 dígitos, excluir números que aparecen en IDs)
        raw_phones = _PAT_PHONE.findall(text)
        id_digits = {re.sub(r'\D', '', id_val) for id_val in data.ids}
        date_digits = {re.sub(r'\D', '', d) for d in data.dates}
        data.phones = [
            p.strip() for p in raw_phones
            if len(re.sub(r'\D', '', p)) >= 7
            and re.sub(r'\D', '', p) not in id_digits
            and not any(re.sub(r'\D', '', p) in id_d for id_d in id_digits)
            and re.sub(r'\D', '', p) not in date_digits
        ][:3]

        # Headers (líneas en mayúsculas con >= 3 chars)
        all_headers = _PAT_HEADER.findall(text)
        data.headers = [
            h.strip() for h in all_headers
            if len(h.strip()) >= 3 and not h.strip().isdigit()
        ][:5]

        # Líneas de totales
        data.totals = [
            line.strip()
            for line in _PAT_TOTAL.findall(text)
            if line.strip()
        ][:3]

        return data


# ============================================================================
# NARRATIVE GENERATOR
# ============================================================================

class NarrativeGenerator:
    """Genera narrativas naturales en español por tipo de documento y modo."""

    def generate(
        self,
        doc_type: str,
        reading_mode: str,
        raw_text: str,
        extracted: ExtractedData,
        visual_caption: Optional[str] = None,
    ) -> str:
        generators = {
            "factura": self._gen_factura,
            "recibo": self._gen_recibo,
            "carta": self._gen_carta,
            "formulario": self._gen_formulario,
            "documento_informativo": self._gen_documento_info,
            "imagen_visual": self._gen_imagen_visual,
            "desconocido": self._gen_desconocido,
        }
        fn = generators.get(doc_type, self._gen_desconocido)
        try:
            return fn(reading_mode, raw_text, extracted, visual_caption)
        except Exception as e:
            logger.warning(f"Error generando narrativa para {doc_type}: {e}")
            return self._fallback(raw_text)

    # --- FACTURA ---

    @staticmethod
    def _clean_total(line: str) -> str:
        """Limpia una línea de total para lectura natural."""
        # Quitar espacios múltiples
        clean = re.sub(r'\s{2,}', ' ', line.strip())
        # Convertir a formato más natural: "TOTAL: Bs. 50,46" → "Total: Bs. 50,46"
        clean = re.sub(r'^(SUBTOTAL|TOTAL|GRAN TOTAL|MONTO TOTAL|NETO|BRUTO)',
                       lambda m: m.group(0).capitalize(), clean, flags=re.IGNORECASE)
        return clean

    def _gen_factura(self, mode: str, text: str, ex: ExtractedData,
                     caption: Optional[str]) -> str:
        intro = "Este documento es una factura."

        if mode == "resumen":
            parts = [intro]
            if ex.totals:
                # Extraer solo el monto del total (sin la palabra "total")
                total_clean = self._clean_total(ex.totals[-1])
                monto = re.sub(r'^(?:sub)?total\s*:?\s*', '', total_clean, flags=re.IGNORECASE).strip()
                parts.append(f"El monto total es {monto}.")
            elif ex.amounts:
                parts.append(f"Monto principal: {ex.amounts[0]}.")
            if ex.dates:
                parts.append(f"Fechada el {ex.dates[0]}.")
            return " ".join(parts)

        if mode == "financiero":
            parts = [intro]
            if ex.totals:
                for t in ex.totals:
                    parts.append(f"{self._clean_total(t)}.")
            if ex.amounts:
                unique = [a for a in ex.amounts if not any(a in t for t in ex.totals)][:5]
                if unique:
                    parts.append("Otros montos: " + ", ".join(unique) + ".")
            if ex.dates:
                parts.append(f"Fecha de la factura: {ex.dates[0]}.")
            if ex.ids:
                parts.append(f"Identificación: {ex.ids[0]}.")
            return " ".join(parts)

        # detallado
        parts = [intro]
        if ex.ids:
            parts.append(f"Identificación del emisor: {ex.ids[0]}.")
        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")
        if ex.headers:
            parts.append(f"Empresa: {ex.headers[0]}.")
        if ex.totals:
            for t in ex.totals:
                parts.append(f"{self._clean_total(t)}.")
        elif ex.amounts:
            parts.append("Montos detectados: " + ", ".join(ex.amounts[:4]) + ".")
        if ex.emails:
            parts.append(f"Correo de contacto: {ex.emails[0]}.")
        if ex.phones:
            parts.append(f"Teléfono: {ex.phones[0]}.")
        return " ".join(parts)

    # --- RECIBO ---

    def _gen_recibo(self, mode: str, text: str, ex: ExtractedData,
                    caption: Optional[str]) -> str:
        intro = "Este documento es un recibo de pago."

        if mode == "resumen":
            parts = [intro]
            if ex.amounts:
                parts.append(f"Monto: {ex.amounts[0]}.")
            if ex.dates:
                parts.append(f"Fecha de pago: {ex.dates[0]}.")
            return " ".join(parts)

        if mode == "financiero":
            parts = [intro]
            if ex.totals:
                parts.extend([f"{self._clean_total(t)}." for t in ex.totals])
            elif ex.amounts:
                parts.append(f"Monto pagado: {ex.amounts[0]}.")
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            return " ".join(parts)

        parts = [intro]
        if ex.dates:
            parts.append(f"Fecha del pago: {ex.dates[0]}.")
        if ex.amounts:
            parts.append(f"Monto pagado: {ex.amounts[0]}.")
        if ex.ids:
            parts.append(f"Número de comprobante: {ex.ids[0]}.")
        if ex.emails:
            parts.append(f"Correo: {ex.emails[0]}.")
        if ex.phones:
            parts.append(f"Teléfono: {ex.phones[0]}.")
        return " ".join(parts)

    # --- CARTA ---

    def _gen_carta(self, mode: str, text: str, ex: ExtractedData,
                   caption: Optional[str]) -> str:
        intro = "Este documento es una carta."

        if mode == "resumen":
            parts = [intro]
            if ex.dates:
                parts.append(f"Fechada el {ex.dates[0]}.")
            first = self._extract_first_sentence(text, skip_lines=3)
            if first:
                parts.append(f"Comienza diciendo: {first}")
            return " ".join(parts)

        if mode == "financiero":
            parts = [intro]
            if ex.amounts:
                parts.append("Montos mencionados: " + ", ".join(ex.amounts[:3]) + ".")
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            return " ".join(parts)

        parts = [intro]
        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")
        if ex.headers:
            parts.append(f"Asunto: {ex.headers[0]}.")
        body = self._extract_body_preview(text, max_words=40)
        if body:
            parts.append(f"Contenido: {body}.")
        if ex.emails:
            parts.append(f"Correo: {ex.emails[0]}.")
        if ex.phones:
            parts.append(f"Teléfono: {ex.phones[0]}.")
        return " ".join(parts)

    # --- FORMULARIO ---

    def _gen_formulario(self, mode: str, text: str, ex: ExtractedData,
                        caption: Optional[str]) -> str:
        intro = "Este documento es un formulario."

        if mode == "resumen":
            parts = [intro]
            if ex.headers:
                parts.append(f"Título: {ex.headers[0]}.")
            parts.append("Contiene campos para completar.")
            return " ".join(parts)

        parts = [intro]
        if ex.headers:
            parts.append(f"Título del formulario: {ex.headers[0]}.")
        field_count = text.count(':')
        if field_count > 0:
            parts.append(f"Tiene aproximadamente {field_count} campos para llenar.")
        if ex.ids:
            parts.append(f"Solicita identificación: {ex.ids[0]}.")
        if ex.dates:
            parts.append(f"Incluye fecha: {ex.dates[0]}.")
        if ex.phones:
            parts.append(f"Teléfono de contacto: {ex.phones[0]}.")
        return " ".join(parts)

    # --- DOCUMENTO INFORMATIVO ---

    def _gen_documento_info(self, mode: str, text: str, ex: ExtractedData,
                            caption: Optional[str]) -> str:
        intro = "Este es un documento informativo."

        if mode == "resumen":
            parts = [intro]
            if ex.headers:
                parts.append(f"Título: {ex.headers[0]}.")
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            return " ".join(parts)

        parts = [intro]
        if ex.headers:
            parts.append(f"Título principal: {ex.headers[0]}.")
        if ex.dates:
            parts.append(f"Fecha de emisión: {ex.dates[0]}.")
        body = self._extract_body_preview(text, max_words=50)
        if body:
            parts.append(f"El contenido dice: {body}.")
        return " ".join(parts)

    # --- IMAGEN VISUAL ---

    def _gen_imagen_visual(self, mode: str, text: str, ex: ExtractedData,
                           caption: Optional[str]) -> str:
        parts = []
        if caption:
            parts.append(f"La imagen muestra: {caption}.")
        if text and text.strip():
            clean = " ".join(text.split()[:30])
            parts.append(f"El texto visible dice: {clean}.")
        elif not caption:
            parts.append("No se detectó texto claro en la imagen.")
        return " ".join(parts) if parts else "Imagen sin texto legible detectado."

    # --- DESCONOCIDO ---

    def _gen_desconocido(self, mode: str, text: str, ex: ExtractedData,
                         caption: Optional[str]) -> str:
        if not text or not text.strip():
            return "No se detectó texto en la imagen."

        intro = "Se detectó texto."

        if mode == "resumen":
            parts = [intro]
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            if ex.amounts:
                parts.append(f"Monto: {ex.amounts[0]}.")
            preview = " ".join(text.split()[:30])
            parts.append(f"{preview}.")
            return " ".join(parts)

        if mode == "financiero":
            parts = [intro]
            if ex.totals:
                for t in ex.totals:
                    parts.append(f"{t}.")
            elif ex.amounts:
                parts.append("Montos encontrados: " + ", ".join(ex.amounts[:5]) + ".")
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            if not ex.amounts and not ex.totals:
                preview = " ".join(text.split()[:30])
                parts.append(f"El texto dice: {preview}.")
            return " ".join(parts)

        # detallado: leer todo el contenido limpio
        parts = [intro]
        if ex.headers:
            parts.append(f"Título: {ex.headers[0]}.")
        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")
        if ex.amounts:
            parts.append("Montos: " + ", ".join(ex.amounts[:5]) + ".")
        # Leer el contenido principal (hasta 80 palabras)
        body = self._extract_body_preview(text, max_words=80)
        if body:
            parts.append(f"El texto dice: {body}.")
        return " ".join(parts)

    # --- HELPERS ---

    @staticmethod
    def _fallback(raw_text: str) -> str:
        words = raw_text.split()[:60]
        return " ".join(words) if words else "No se pudo leer el documento."

    @staticmethod
    def _extract_first_sentence(text: str, skip_lines: int = 0) -> str:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines[skip_lines:]:
            if len(line.split()) >= 5:
                words = line.split()[:20]
                return " ".join(words)
        return ""

    @staticmethod
    def _extract_body_preview(text: str, max_words: int = 40) -> str:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        body_words: List[str] = []
        for line in lines:
            if len(line.split()) < 3:
                continue
            body_words.extend(line.split())
            if len(body_words) >= max_words:
                break
        return " ".join(body_words[:max_words])


# ============================================================================
# PROSODY ENHANCER
# ============================================================================

class ProsodyEnhancer:
    """Mejora texto para prosodia natural en Piper TTS."""

    _ABBREV = [
        (re.compile(r'\bRIF\b'), 'Erre-I-F'),
        (re.compile(r'\bNIT\b'), 'Ene-I-T'),
        (re.compile(r'\bIVA\b'), 'I-V-A'),
        (re.compile(r'\bC\.I\.'), 'cédula'),
        (re.compile(r'\bS\.A\.'), 'Sociedad Anónima'),
        (re.compile(r'\bC\.A\.'), 'Compañía Anónima'),
        (re.compile(r'\bBs\.\s'), 'bolívares '),
        (re.compile(r'\bBs\s'), 'bolívares '),
        (re.compile(r'\bUSD\b'), 'dólares'),
        (re.compile(r'\bEUR\b'), 'euros'),
        (re.compile(r'\bCOP\b'), 'pesos colombianos'),
        (re.compile(r'\bMXN\b'), 'pesos mexicanos'),
        (re.compile(r'\bVES\b'), 'bolívares'),
        (re.compile(r'\bTelf?\.\s', re.IGNORECASE), 'teléfono '),
        (re.compile(r'\bAv\.\s'), 'avenida '),
        (re.compile(r'\bNo\.\s'), 'número '),
        (re.compile(r'\bn[úu]m\.\s', re.IGNORECASE), 'número '),
        (re.compile(r'\bNº\s'), 'número '),
    ]

    _AMOUNT_REPEAT = re.compile(
        r'(?:total|monto|pago|valor)\s*[:.]?\s*'
        r'((?:[\$\€£]|bol[ií]vares|d[oó]lares|pesos)?\s*[\d]{1,3}(?:[,\.\s]\d{3})*(?:[,\.]\d{1,2})?'
        r'(?:\s*(?:bol[ií]vares|d[oó]lares|pesos|euros))?)',
        re.IGNORECASE
    )

    def enhance(self, text: str, doc_type: str) -> str:
        if not text:
            return text

        text = self._expand_abbreviations(text)
        text = self._normalize_sentence_endings(text)

        if doc_type in ("factura", "recibo"):
            text = self._repeat_key_amounts(text)

        text = self._break_long_sentences(text)

        # Limpiar puntuación doble
        text = re.sub(r'\.{2,}', '...', text)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\s{2,}', ' ', text)

        return text.strip()

    def _expand_abbreviations(self, text: str) -> str:
        for pattern, replacement in self._ABBREV:
            text = pattern.sub(replacement, text)
        return text

    def _normalize_sentence_endings(self, text: str) -> str:
        # Solo agregar punto cuando hay un salto de línea claro (párrafo)
        # entre una línea que no termina en puntuación y la siguiente
        text = re.sub(r'([a-záéíóúñ0-9])\s*\n\s*([A-ZÁÉÍÓÚÑ])', r'\1. \2', text)
        return text

    def _repeat_key_amounts(self, text: str) -> str:
        match = self._AMOUNT_REPEAT.search(text)
        if match:
            amount = match.group(1).strip()
            if amount and len(amount) >= 2:
                end = match.end()
                text = text[:end] + f"... {amount}" + text[end:]
        return text

    def _break_long_sentences(self, text: str, max_words: int = 40) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) <= max_words:
                result.append(sentence)
            else:
                # Insertar una pausa natural con coma en conjunciones
                broken = re.sub(
                    r'\s+(y|pero|sin embargo|además|también|aunque)\s+',
                    r', \1 ',
                    sentence,
                    count=1,
                    flags=re.IGNORECASE
                )
                result.append(broken)
        return ' '.join(result)


# ============================================================================
# SMART READING SERVICE (ORQUESTADOR)
# ============================================================================

class SmartReadingService:
    """
    Orquestador principal del modo lectura inteligente.

    Pipeline: OCR → Classify → Extract → Narrate → Enhance → Response
    """

    def __init__(self):
        self._ocr_service = None
        self._classifier = DocumentClassifier()
        self._extractor = StructureExtractor()
        self._generator = NarrativeGenerator()
        self._enhancer = ProsodyEnhancer()
        self._captioning = None

    def _get_ocr(self):
        if self._ocr_service is None:
            from app.services.ocr_service import get_ocr_service
            self._ocr_service = get_ocr_service()
        return self._ocr_service

    def _get_captioning(self):
        if self._captioning is None and settings.CAPTIONING_ENABLED:
            try:
                from app.services.captioning_service import get_captioning_service
                self._captioning = get_captioning_service()
            except Exception as e:
                logger.warning(f"Captioning no disponible: {e}")
        return self._captioning

    def analyze(self, image: np.ndarray, reading_mode: str = "detallado") -> dict:
        """
        Pipeline completo de lectura inteligente.

        Args:
            image: Imagen BGR (OpenCV)
            reading_mode: "resumen" | "detallado" | "financiero"

        Returns:
            dict compatible con SmartReadingResponse
        """
        # 1. OCR
        ocr_result = self._get_ocr().extract_text(image)
        raw_text = ocr_result.get("text", "")
        has_text = ocr_result.get("has_text", False)
        confidence = ocr_result.get("confidence")
        word_count = ocr_result.get("word_count", 0)

        logger.info(f"[SmartReading] OCR: {word_count} palabras, confianza={confidence}, has_text={has_text}")
        if raw_text:
            logger.info(f"[SmartReading] Texto OCR (primeras 200 chars): {raw_text[:200]}")

        # 2. Clasificar
        doc_type, cls_confidence = self._classifier.classify(raw_text, word_count)
        logger.info(f"[SmartReading] Clasificación: {doc_type} (conf={cls_confidence:.2f})")

        # 3. Extraer campos
        extracted = self._extractor.extract(raw_text)
        logger.info(f"[SmartReading] Campos: fechas={len(extracted.dates)}, montos={len(extracted.amounts)}, ids={len(extracted.ids)}, totales={len(extracted.totals)}")

        # 4. Florence-2 para imagen_visual
        visual_caption = None
        if doc_type == "imagen_visual":
            captioning = self._get_captioning()
            if captioning and captioning.is_available:
                try:
                    caption_en = captioning.generate_caption(image, detailed=False)
                    if caption_en:
                        visual_caption = captioning._translate_caption(caption_en)
                except Exception as e:
                    logger.debug(f"Florence-2 caption falló: {e}")

        # 5. Generar narrativa
        narrative = self._generator.generate(
            doc_type=doc_type,
            reading_mode=reading_mode,
            raw_text=raw_text,
            extracted=extracted,
            visual_caption=visual_caption,
        )

        # 6. Mejorar prosodia
        narrative = self._enhancer.enhance(narrative, doc_type)

        logger.info(f"[SmartReading] Narrativa ({reading_mode}): {narrative[:150]}...")

        # 7. Construir respuesta
        return {
            "success": True,
            "message": "Documento analizado correctamente",
            "narrative": narrative,
            "document_type": doc_type,
            "document_type_label": DocumentClassifier.get_label(doc_type),
            "reading_mode": reading_mode,
            "raw_text": raw_text,
            "has_text": has_text,
            "ocr_confidence": confidence,
            "word_count": word_count,
            "extracted_fields": {
                "dates": extracted.dates,
                "amounts": extracted.amounts,
                "emails": extracted.emails,
                "phones": extracted.phones,
                "ids": extracted.ids,
                "headers": extracted.headers,
                "totals": extracted.totals,
            },
            "visual_caption": visual_caption,
        }


# ============================================================================
# SINGLETON
# ============================================================================

_smart_reading_service: Optional[SmartReadingService] = None


def get_smart_reading_service() -> SmartReadingService:
    global _smart_reading_service
    if _smart_reading_service is None:
        _smart_reading_service = SmartReadingService()
    return _smart_reading_service
