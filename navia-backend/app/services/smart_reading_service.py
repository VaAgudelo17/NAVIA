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
from app.services.barcode_service import get_barcode_reader

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
    # Etiquetas de productos (supermercado, medicamentos, etc.)
    "etiqueta": [
        (re.compile(r'\bingredientes\b', re.IGNORECASE), 10),
        (re.compile(r'\bcontenido\b', re.IGNORECASE), 8),
        (re.compile(r'\bpeso\b.*\bneto\b', re.IGNORECASE), 7),
        (re.compile(r'\bvalor\s+nutricional\b', re.IGNORECASE), 10),
        (re.compile(r'\bcalor[íi]as\b', re.IGNORECASE), 8),
        (re.compile(r'\bprote[íi]nas\b', re.IGNORECASE), 7),
        (re.compile(r'\bgrasas\b', re.IGNORECASE), 6),
        (re.compile(r'\bcarbohidratos\b', re.IGNORECASE), 6),
        (re.compile(r'\bfabricado\b', re.IGNORECASE), 6),
        (re.compile(r'\bvencimiento\b', re.IGNORECASE), 8),
        (re.compile(r'\blote\b', re.IGNORECASE), 5),
        (re.compile(r'\bc[óo]digo\s+de\s+barras\b', re.IGNORECASE), 7),
        (re.compile(r'\bprecio\b', re.IGNORECASE), 5),
        (re.compile(r'\bdescuento\b', re.IGNORECASE), 5),
    ],
    # Menú de restaurante
    "menu": [
        (re.compile(r'\bmen[úu]\b', re.IGNORECASE), 10),
        (re.compile(r'\bcarta\b', re.IGNORECASE), 8),
        (re.compile(r'\bentrantes\b', re.IGNORECASE), 8),
        (re.compile(r'\bplatos\b', re.IGNORECASE), 7),
        (re.compile(r'\bpostres\b', re.IGNORECASE), 8),
        (re.compile(r'\bbebidas\b', re.IGNORECASE), 7),
        (re.compile(r'\bprecio\b.*\b\$\b', re.IGNORECASE), 6),
        (re.compile(r'\b\$ ?\d+\b', re.IGNORECASE), 5),
        (re.compile(r'\bpromoci[óo]n\b', re.IGNORECASE), 7),
        (re.compile(r'\bcombo\b', re.IGNORECASE), 6),
        (re.compile(r'\bincluye\b', re.IGNORECASE), 5),
    ],
    # Tarjetas de presentación
    "tarjeta": [
        (re.compile(r'\btarjeta\b', re.IGNORECASE), 8),
        (re.compile(r'\bdirector\b', re.IGNORECASE), 7),
        (re.compile(r'\bgerente\b', re.IGNORECASE), 6),
        (re.compile(r'\bcoordinador\b', re.IGNORECASE), 6),
        (re.compile(r'\bconsultor\b', re.IGNORECASE), 6),
        (re.compile(r'\basespecialista\b', re.IGNORECASE), 7),
        (re.compile(r'\bdepartamento\b', re.IGNORECASE), 6),
        (re.compile(r'\bsucursal\b', re.IGNORECASE), 5),
    ],
    # Capturas de chat (WhatsApp, Telegram, Instagram, etc.)
    "chat": [
        (re.compile(r'\b\d{1,2}:\d{2}\s*(?:am|pm|a\.m\.|p\.m\.)?\b', re.IGNORECASE), 6),
        (re.compile(r'\b(?:en\s+l[ií]nea|online|ult(?:ima|\.)?\s*vez|last\s+seen)\b', re.IGNORECASE), 8),
        (re.compile(r'\b(?:escribir?|escribe)\s+(?:un\s+)?mensaje\b', re.IGNORECASE), 10),
        (re.compile(r'\b(?:type\s+a\s+message|mensaje)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:hoy|ayer|today|yesterday)\b', re.IGNORECASE), 4),
        (re.compile(r'\b(?:jaja|jeje|haha|hola|hey|mano|bro|pana)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:WhatsApp|Telegram|Messenger|Instagram|Signal)\b', re.IGNORECASE), 10),
        (re.compile(r'\b(?:audio|foto|imagen|video|sticker|enviado|recibido)\b', re.IGNORECASE), 4),
        (re.compile(r'\b(?:grupo|chat|conversaci[oó]n)\b', re.IGNORECASE), 6),
        (re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\s*,?\s*\d{1,2}:\d{2}\b', re.IGNORECASE), 7),
    ],
    # Notificaciones (push, alertas del sistema, avisos de apps)
    "notificacion": [
        (re.compile(r'\bnotificaci[oó]n(?:es)?\b', re.IGNORECASE), 10),
        (re.compile(r'\balerta\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:hace|hace)\s+\d+\s+(?:min|hora|seg|segundo|minuto)', re.IGNORECASE), 8),
        (re.compile(r'\b\d+\s*(?:min|h|hrs?)\s+(?:ago|atr[aá]s)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:permitir|bloquear|aceptar|rechazar|dismiss|allow|deny)\b', re.IGNORECASE), 6),
        (re.compile(r'\b(?:nueva?\s+)?(?:mensaje|correo|actualizaci[oó]n|recordatorio)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:silenciar|borrar|marcar|leer)\b', re.IGNORECASE), 4),
        (re.compile(r'\b(?:centro\s+de\s+notificaciones|notification)\b', re.IGNORECASE), 9),
        (re.compile(r'\b(?:ahora|just\s+now|now)\b', re.IGNORECASE), 3),
    ],
    # Pantallas de login / registro
    "login": [
        (re.compile(r'\b(?:iniciar?\s+sesi[oó]n|log\s*in|sign\s*in)\b', re.IGNORECASE), 10),
        (re.compile(r'\b(?:registr(?:ar(?:se|te)?|o)|sign\s*up|create\s+account)\b', re.IGNORECASE), 9),
        (re.compile(r'\b(?:contrase[ñn]a|password|clave)\b', re.IGNORECASE), 8),
        (re.compile(r'\b(?:usuario|user(?:name)?|correo\s+electr[oó]nico|email)\b', re.IGNORECASE), 6),
        (re.compile(r'\b(?:olvidaste?\s+(?:tu\s+)?contrase[ñn]a|forgot\s+password|recuperar)\b', re.IGNORECASE), 9),
        (re.compile(r'\b(?:iniciar?\s+con|continuar?\s+con|sign\s+in\s+with)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:Google|Facebook|Apple|GitHub)\b', re.IGNORECASE), 4),
        (re.compile(r'\b(?:verificaci[oó]n|c[oó]digo|OTP|token)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:bienvenid[oa]|welcome)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:entrar|acceder|ingresar)\b', re.IGNORECASE), 6),
    ],
    # Redes sociales (posts, perfiles, feeds)
    "red_social": [
        (re.compile(r'\b(?:me\s+gusta|like|likes|liked)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:comentario|comment|comments)\b', re.IGNORECASE), 6),
        (re.compile(r'\b(?:compartir|share|shared|retweet|repost)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:seguir|follow|following|followers|seguidores)\b', re.IGNORECASE), 8),
        (re.compile(r'\b(?:publicaci[oó]n|post|tweet|reel|story|stories)\b', re.IGNORECASE), 7),
        (re.compile(r'@\w{2,}', re.IGNORECASE), 6),
        (re.compile(r'#\w{2,}', re.IGNORECASE), 6),
        (re.compile(r'\b(?:Instagram|Twitter|TikTok|Facebook|X|Threads|LinkedIn|YouTube)\b', re.IGNORECASE), 9),
        (re.compile(r'\b(?:perfil|profile|bio|feed|timeline)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:suscri(?:bir|ptores)|subscribe|subscribers)\b', re.IGNORECASE), 6),
        (re.compile(r'\b\d+[KkMm]?\s*(?:views|vistas|reproducciones|likes)\b', re.IGNORECASE), 6),
    ],
    # Noticias, artículos, blogs, portadas de revista
    "noticia": [
        (re.compile(r'\b(?:noticias?|news|breaking)\b', re.IGNORECASE), 9),
        (re.compile(r'\b(?:art[ií]culo|article)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:publicado|published|por|by)\s+\w+', re.IGNORECASE), 5),
        (re.compile(r'\b(?:redacci[oó]n|editor(?:ial)?|periodista|reporter|autor)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:opini[oó]n|cr[oó]nica|reportaje|entrevista|columna)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:fuente|source|Reuters|AP|AFP|EFE|CNN|BBC)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:blog|post|entrada)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:portada|cover|revista|magazine)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:leer\s+m[aá]s|read\s+more|continuar\s+leyendo|ver\s+m[aá]s)\b', re.IGNORECASE), 8),
        (re.compile(r'\b(?:exclusiv[oa]|[uú]ltima\s+hora|urgente|breaking)\b', re.IGNORECASE), 8),
        (re.compile(r'\b(?:categor[ií]a|secci[oó]n|deportes|pol[ií]tica|econom[ií]a|tecnolog[ií]a|salud|cultura)\b', re.IGNORECASE), 4),
    ],
    # Correos electrónicos (Gmail, Outlook, bandeja de entrada)
    "correo": [
        (re.compile(r'\b(?:bandeja\s+de\s+entrada|inbox|recibidos)\b', re.IGNORECASE), 10),
        (re.compile(r'\b(?:asunto|subject)\s*:', re.IGNORECASE), 9),
        (re.compile(r'\b(?:de|from)\s*:\s*\S+@\S+', re.IGNORECASE), 10),
        (re.compile(r'\b(?:para|to)\s*:\s*\S+@\S+', re.IGNORECASE), 8),
        (re.compile(r'\b(?:responder|reply|reenviar|forward|archivar|archive)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:borrador(?:es)?|draft|spam|papelera|trash|enviados|sent)\b', re.IGNORECASE), 8),
        (re.compile(r'\b(?:Gmail|Outlook|Yahoo\s*Mail|correo|mail)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:adjunto|attachment|archivo\s+adjunto)\b', re.IGNORECASE), 6),
        (re.compile(r'\b(?:CC|BCC|CCO)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:no\s+le[ií]do|unread|le[ií]do|read)\b', re.IGNORECASE), 5),
    ],
    # Presentaciones (PowerPoint, slides, diapositivas)
    "presentacion": [
        (re.compile(r'\b(?:presentaci[oó]n|presentation)\b', re.IGNORECASE), 9),
        (re.compile(r'\b(?:diapositiva|slide)\s*\d*', re.IGNORECASE), 10),
        (re.compile(r'\b(?:agenda|objetivos?|conclusi[oó]n|conclusiones)\b', re.IGNORECASE), 6),
        (re.compile(r'\b(?:introducci[oó]n|overview|resumen\s+ejecutivo)\b', re.IGNORECASE), 6),
        (re.compile(r'\b(?:pregunt[ao]s|questions|gracias|thanks|thank\s+you)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:siguiente|next|anterior|previous|p[aá]gina)\b', re.IGNORECASE), 4),
        (re.compile(r'\b\d+\s*/\s*\d+\b', re.IGNORECASE), 4),  # "3/10" paginación
        (re.compile(r'\b(?:PowerPoint|Google\s+Slides|Keynote|Canva)\b', re.IGNORECASE), 8),
        (re.compile(r'\b(?:equipo|team|proyecto|project|plan|estrategia|strategy)\b', re.IGNORECASE), 3),
    ],
    # Pantallas de configuración / ajustes
    "configuracion": [
        (re.compile(r'\b(?:configuraci[oó]n|ajustes?|settings?)\b', re.IGNORECASE), 10),
        (re.compile(r'\b(?:activar|desactivar|enable|disable|on|off)\b', re.IGNORECASE), 6),
        (re.compile(r'\b(?:Wi-?Fi|Bluetooth|datos?\s+m[oó]viles|mobile\s+data|NFC)\b', re.IGNORECASE), 8),
        (re.compile(r'\b(?:brillo|brightness|volumen|volume|sonido|sound)\b', re.IGNORECASE), 7),
        (re.compile(r'\b(?:bater[ií]a|battery|almacenamiento|storage)\b', re.IGNORECASE), 6),
        (re.compile(r'\b(?:cuenta|account|privacidad|privacy|seguridad|security)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:notificaciones|pantalla|display|idioma|language)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:acerca\s+de|about|versi[oó]n|version|actualizar|update)\b', re.IGNORECASE), 5),
        (re.compile(r'\b(?:modo\s+(?:oscuro|claro|avi[oó]n|no\s+molestar)|dark\s+mode|airplane)\b', re.IGNORECASE), 8),
        (re.compile(r'\b(?:general|accesibilidad|accessibility)\b', re.IGNORECASE), 5),
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
            "etiqueta": "Etiqueta de producto",
            "menu": "Menú de restaurante",
            "tarjeta": "Tarjeta de presentación",
            "chat": "Conversación de chat",
            "notificacion": "Notificación",
            "login": "Pantalla de inicio de sesión",
            "red_social": "Publicación de red social",
            "noticia": "Noticia o artículo",
            "correo": "Correo electrónico",
            "presentacion": "Presentación",
            "configuracion": "Pantalla de configuración",
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
    r'(?:[\$€£]+\s*[\d]{1,3}(?:[,\.\s]\d{3})*(?:[,\.]\d{1,2})?)'
    r'|(?:Bs\.?\s*[\d]{1,3}(?:[\s,\.]+\d{3})*(?:[,\.]\d{1,2})?)'
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
    r'\b((?:sub\s*)?total|gran\s+total|monto\s+total|neto|bruto)'
    r'\s*:?\s*'
    r'((?:Bs\.?|[\$€£]|USD|COP|VES|MXN)?\s*[\d]+[\d\s,\.]*[\d])',
    re.IGNORECASE
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

        # Líneas de totales: captura (label, monto) como "Total: Bs 2.549.614"
        data.totals = []
        for m in _PAT_TOTAL.finditer(text):
            label = m.group(1).strip().capitalize()
            amount = m.group(2).strip()
            # Limpiar espacios dentro del número: "2. 549. 614" → "2.549.614"
            amount = re.sub(r'(\d)\s+(\d)', r'\1\2', amount)
            amount = re.sub(r'(\d)\.\s+', r'\1.', amount)
            data.totals.append(f"{label}: {amount}")
        data.totals = data.totals[:3]

        return data


# ============================================================================
# OCR TEXT RECONSTRUCTOR
# ============================================================================

# Basura típica de status bar / chrome de apps en capturas de pantalla
_STATUS_BAR_PATTERNS = re.compile(
    r'\b(?:'
    r'\d{1,2}:\d{2}\s*(?:[AaPp]\.?[Mm]\.?){0,2}'  # timestamps: 3:03, 8:36PM
    r'|\d{1,3}%'                                     # battery: 85%
    r'|all\s+[A-Z]{1,3}'                             # "all ED", "all GD"
    r'|[A-Z]{1,2}\s+\d{1,2}:\d{2}'                  # "ED 3:03"
    r')\b',
    re.IGNORECASE
)

# Palabras cortas reales en español/inglés (no basura OCR)
_REAL_SHORT_WORDS_SET = frozenset({
    'y', 'o', 'a', 'e', 'u', 'el', 'la', 'de', 'no', 'es',
    'lo', 'en', 'se', 'me', 'te', 'le', 'al', 'mi', 'tu',
    'si', 'ya', 'un', 'yo', 'ni', 'he', 'su', 'os', 'do',
    'que', 'por', 'con', 'del', 'las', 'los', 'una', 'son',
    'fue', 'ser', 'hay', 'van', 'mas', 'más', 'sin', 'nos',
    'hoy', 'muy', 'día', 'dia', 'ver', 'dar', 'mal', 'bien',
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
    'can', 'had', 'her', 'was', 'one', 'our', 'out', 'new',
    'fui', 'eso', 'esa', 'voy', 'mis', 'tus', 'sus', 'vez',
    'sol', 'mar', 'luz', 'pan', 'sal', 'red', 'fin', 'dos',
    'web', 'app', 'pdf', 'url',
})

# Palabras de 3 chars válidas (superset para no tener que duplicar)
_VALID_3CHAR = frozenset({
    'que', 'por', 'con', 'del', 'las', 'los', 'una', 'son',
    'fue', 'ser', 'hay', 'van', 'mas', 'más', 'sin', 'nos',
    'hoy', 'muy', 'día', 'dia', 'ver', 'dar', 'mal', 'vez',
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
    'can', 'had', 'her', 'was', 'one', 'our', 'out', 'new',
    'fui', 'eso', 'esa', 'voy', 'mis', 'tus', 'sus',
    'sol', 'mar', 'luz', 'pan', 'sal', 'red', 'fin', 'dos',
    'web', 'app', 'pdf', 'url',
})

# Patrones de texto de la propia UI de NAVIA (feedback loop)
_NAVIA_UI_PATTERNS = re.compile(
    r'\b(?:Lectura|Conversaci[oó]n\s+de\s+chat|Nueva\s+Imagen|Repetir|'
    r'\d+\s+palabras|confianza|auto|La\s+imagen\s+est[aá]\s+muy\s+oscura|'
    r'Busca\s+mejor\s+iluminaci[oó]n|enciende\s+la\s+linterna|'
    r'Esto\s+es\s+una?\s+(?:conversaci[oó]n|notificaci[oó]n|pantalla|'
    r'publicaci[oó]n|noticia|correo|diapositiva|documento)|'
    r'Los\s+mensajes\s+dicen|Se\s+detect[oó]\s+(?:texto|un\s+documento)|'
    r'No\s+se\s+(?:pudo|detect[oó])\s+leer|Documento\s+analizado|'
    r'Opciones\s+visibles|El\s+texto\s+dice|Dice:|Tiene\s+campos|'
    r'Pantalla\s+de\s+(?:inicio|configuraci[oó]n)|'
    r'Factura|Recibo\s+de\s+pago|Etiqueta\s+de\s+producto|'
    r'Men[uú]\s+de\s+restaurante|Tarjeta\s+de\s+presentaci[oó]n|'
    r'Noticia\s+o\s+art[ií]culo|Correo\s+electr[oó]nico|'
    r'Publicaci[oó]n\s+de\s+red\s+social|Presentaci[oó]n|'
    r'Es\s+un\s+perfil\s+de|Es\s+un\s+chat|'
    r'Intenta\s+(?:mantener|acercar|con\s+mejor)|'
    r'borrosa|enfocar\s+bien|iluminaci[oó]n)\b',
    re.IGNORECASE
)


class OCRTextReconstructor:
    """
    Fase obligatoria de reconstrucción semántica del texto OCR.

    Se ejecuta UNA VEZ antes de cualquier generación de narrativa.
    Transforma el texto OCR crudo en texto limpio y coherente optimizado
    para lectura por voz.

    Pipeline de reconstrucción:
      1. Eliminar texto de la propia UI de NAVIA (anti-feedback-loop)
      2. Eliminar basura de status bar (timestamps, batería, fragmentos)
      3. Limpiar tokens individuales (basura OCR, símbolos sueltos)
      4. Separar palabras pegadas por OCR
      5. Reconstruir líneas coherentes agrupando tokens relacionados
      6. Detectar y etiquetar estructuras semánticas (encabezados, montos)

    El texto reconstruido es lo que reciben TODOS los generadores.
    Ningún generador debe usar jamás el texto OCR crudo.
    """

    def reconstruct(self, raw_text: str, doc_type: str = "") -> str:
        """
        Reconstruye texto OCR crudo en texto limpio para TTS.

        Args:
            raw_text: Texto OCR crudo de Tesseract
            doc_type: Tipo de documento (para heurísticas específicas)

        Returns:
            Texto reconstruido, limpio y coherente para narrativa
        """
        if not raw_text or not raw_text.strip():
            return ""

        text = raw_text

        # 1. Anti-feedback-loop: eliminar texto de la propia UI de NAVIA
        text = self._strip_navia_ui(text)
        if not text.strip():
            return ""

        # 2. Eliminar basura de status bar y chrome de apps
        text = self._strip_status_bar(text)

        # 3. Separar palabras pegadas por OCR (ej: "VerbosHttp" → "Verbos Http")
        text = self._split_glued_words(text)

        # 4. Limpiar tokens individuales (la limpieza más agresiva)
        text = self._clean_tokens(text)

        # 5. Reconstruir líneas coherentes
        text = self._rebuild_lines(text)

        # 6. Limpieza final para TTS
        text = self._final_tts_cleanup(text)

        return text.strip()

    def clean_fragment(self, fragment: str) -> str:
        """
        Limpia un fragmento corto de texto OCR (para regex captures).

        Menos agresivo que reconstruct() — preserva la estructura del
        fragmento pero elimina basura obvia. Para usar en regex group()
        captures de los generadores (sender, subject, ingredient, etc.)
        """
        if not fragment:
            return ""
        # Strip puntuación envolvente
        fragment = fragment.strip('.,;:!?()[]{}"\'`\t ')
        # Separar palabras pegadas
        fragment = self._split_glued_words(fragment)
        # Limpiar tokens pero ser menos agresivo (no eliminar 3-char)
        words = fragment.split()
        clean = []
        for w in words:
            stripped = w.strip('.,;:!?()[]{}"\'`')
            if not stripped:
                continue
            wl = stripped.lower()
            # Eliminar tokens puramente basura
            if len(wl) == 1 and wl not in 'aeiouáéíóú0123456789':
                continue
            # Eliminar tokens solo símbolos
            if re.match(r'^[^\w]+$', stripped):
                continue
            # Eliminar tokens con >60% no-letras
            letters = sum(1 for c in wl if c.isalpha())
            if len(wl) > 2 and letters < len(wl) * 0.4:
                continue
            clean.append(stripped)
        return " ".join(clean)

    # --- Etapas internas ---

    @staticmethod
    def _strip_navia_ui(text: str) -> str:
        """Elimina texto de la propia UI de NAVIA (anti-feedback-loop)."""
        match = _NAVIA_UI_PATTERNS.search(text)
        if match:
            text = text[:match.start()].strip()
        return text

    @staticmethod
    def _strip_status_bar(text: str) -> str:
        """Elimina basura típica del status bar de capturas de pantalla."""
        # Eliminar timestamps sueltos, porcentajes de batería, etc.
        text = _STATUS_BAR_PATTERNS.sub(' ', text)
        # Eliminar fragmentos cortos all-caps de 1-2 chars aislados
        text = re.sub(r'(?<!\w)[A-Z]{1,2}(?!\w)', ' ', text)
        return re.sub(r'\s{2,}', ' ', text).strip()

    @staticmethod
    def _split_glued_words(text: str) -> str:
        """
        Separa palabras pegadas por errores de OCR.

        Ej: "VerbosHttp" → "Verbos Http"
            "ServicioPan" → "Servicio Pan"
            "8:36pmM" → "8:36pm M"

        Regla: camelCase split (minúscula seguida de mayúscula)
        """
        # Split camelCase: "VerbosHttp" → "Verbos Http"
        text = re.sub(r'([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])', r'\1 \2', text)
        # Split dígitos pegados a letras: "8:36pmM" → "8:36pm M"
        text = re.sub(r'(\d)([A-ZÁÉÍÓÚÑ])', r'\1 \2', text)
        return text

    @staticmethod
    def _clean_tokens(text: str) -> str:
        """
        Limpia tokens individuales del texto OCR.

        Más completo que _clean_ocr_words — también:
        - Normaliza Unicode (NFC)
        - Detecta y preserva números con contexto
        - Es consistente con la lista de palabras cortas reales
        """
        import unicodedata
        text = unicodedata.normalize('NFC', text)

        words = text.split()
        clean = []
        prev_is_word = False  # Para contexto de números

        for i, w in enumerate(words):
            stripped = w.strip('.,;:!?()[]{}"\'`')
            wl = stripped.lower()

            if not wl:
                continue

            # Preservar palabras cortas reales
            if wl in _REAL_SHORT_WORDS_SET:
                clean.append(stripped)
                prev_is_word = True
                continue

            # Eliminar tokens de 1-2 chars no reconocidos
            if len(wl) <= 2:
                prev_is_word = False
                continue

            # Números y montos: preservar si parecen métricas/cantidades/precios
            if re.match(r'^[\d\W]+$', wl):
                # Preservar: "123", "45%", "10K", "$25", "$35.50", "€100"
                if re.match(r'^[\$€£¥₡₲₺₽]?\d[\d,.]*[%KkMm]?$', stripped):
                    clean.append(stripped)
                    prev_is_word = False
                continue

            # Eliminar tokens con >50% basura (no-letras)
            letter_count = sum(1 for c in wl if c.isalpha())
            if letter_count < len(wl) * 0.5:
                prev_is_word = False
                continue

            # Eliminar tokens cortos all-caps (basura UI: "GD", "ED")
            if len(stripped) <= 3 and stripped.isupper():
                prev_is_word = False
                continue

            # Tokens de 3 chars: solo aceptar los conocidos o con vocal
            if len(wl) == 3 and wl not in _VALID_3CHAR:
                if not re.search(r'[aeiouáéíóú]', wl):
                    prev_is_word = False
                    continue

            clean.append(stripped)
            prev_is_word = True

        return " ".join(clean)

    @staticmethod
    def _rebuild_lines(text: str) -> str:
        """
        Reconstruye líneas coherentes agrupando tokens.

        - Elimina líneas con menos de 2 palabras reales
        - Colapsa líneas vacías
        """
        if not text:
            return ""

        lines = text.split('\n')
        rebuilt = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            # Contar palabras reales (>2 chars con letras)
            real = [w for w in words
                    if len(w) > 2 and re.search(r'[a-záéíóúñ]', w, re.IGNORECASE)]
            # Aceptar línea si tiene al menos 2 palabras reales,
            # O si es corta pero toda es real (ej: nombre propio)
            if len(real) >= 2 or (len(words) <= 3 and len(real) == len(words)):
                rebuilt.append(" ".join(words))

        return "\n".join(rebuilt)

    @staticmethod
    def _final_tts_cleanup(text: str) -> str:
        """Limpieza final optimizada para lectura por voz."""
        # Colapsar espacios múltiples
        text = re.sub(r'\s{2,}', ' ', text)
        # Eliminar puntuación repetida
        text = re.sub(r'([.!?,;:])\1+', r'\1', text)
        # Eliminar guiones sueltos
        text = re.sub(r'\s-\s', ' ', text)
        text = re.sub(r'^-\s', '', text, flags=re.MULTILINE)
        # Eliminar paréntesis vacíos o con basura
        text = re.sub(r'\([^)]{0,2}\)', '', text)
        text = re.sub(r'\[[^\]]{0,2}\]', '', text)
        return text.strip()


# Singleton
_reconstructor_instance: Optional[OCRTextReconstructor] = None

def get_text_reconstructor() -> OCRTextReconstructor:
    global _reconstructor_instance
    if _reconstructor_instance is None:
        _reconstructor_instance = OCRTextReconstructor()
    return _reconstructor_instance


def _cf(fragment: str) -> str:
    """Atajo módulo-level para clean_fragment (para regex captures en generators)."""
    return get_text_reconstructor().clean_fragment(fragment)


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
            # DOCUMENTO_FORMAL
            "factura": self._gen_factura,
            "recibo": self._gen_recibo,
            "carta": self._gen_carta,
            "formulario": self._gen_formulario,
            "documento_informativo": self._gen_documento_info,
            "etiqueta": self._gen_etiqueta,
            "tarjeta": self._gen_tarjeta,
            "contrato": self._gen_documento_info,
            "hoja_de_vida": self._gen_documento_info,
            "informe": self._gen_documento_info,
            "noticia": self._gen_noticia,
            "correo": self._gen_correo,
            "presentacion": self._gen_presentacion,
            "receta_medica": self._gen_receta_medica,
            "boleto": self._gen_boleto,
            "identificacion": self._gen_identificacion,
            "horario": self._gen_horario,
            "instrucciones": self._gen_instrucciones,
            "resultado_lab": self._gen_resultado_lab,
            "tabla_nutricional": self._gen_tabla_nutricional,
            "calendario": self._gen_calendario,

            # NUEVOS TIPOS
            "factura_servicio": self._gen_factura,
            "ticket_transporte": self._gen_boleto,
            "credencial": self._gen_identificacion,

            # INTERFAZ_DIGITAL (nuevos subtipos → generadores existentes)
            "app_menu": self._gen_menu,
            "app_settings": self._gen_configuracion,
            "app_login": self._gen_login,
            "app_form": self._gen_formulario,
            "app_social": self._gen_red_social,
            "app_service": self._make_app_gen("un servicio o compra"),
            "notificacion": self._gen_notificacion,
            "mapa": self._gen_mapa,

            # TEXTO_CONVERSACIONAL
            "chat": self._gen_chat,
            "comentario": self._gen_chat,

            # IMAGEN_VISUAL
            "imagen_visual": self._gen_imagen_visual,

            # Legacy compatibility
            "menu": self._gen_menu,
            "login": self._gen_login,
            "red_social": self._gen_red_social,
            "configuracion": self._gen_configuracion,

            # Fallbacks
            "desconocido": self._gen_desconocido,
            "mixto": self._gen_desconocido,
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
        """Narrativa natural para facturas y tickets de compra.

        Extrae directamente del texto OCR (crudo o limpio):
        - Total a pagar (busca TOTAL + monto más grande cercano)
        - Productos/items (cuenta líneas con precio)
        - Fecha
        - Método de pago
        - Empresa/establecimiento
        """
        # Usar texto crudo que tiene los montos (el clean los puede eliminar)
        src = text if text else ""

        # --- 1. Buscar TOTAL (el dato más importante) ---
        total_amount = None

        def _clean_bs_amount(raw_amt: str) -> str:
            """Limpia un monto en Bs: 'Bs 2.533. 818,60' → 'Bs 2.533.818,60'"""
            clean = re.sub(r'(\d)[\s]+(?=\d)', r'\1', raw_amt)
            clean = re.sub(r'\.\s+', '.', clean)
            clean = re.sub(r',\s+', ',', clean)
            return clean.strip()

        def _parse_amount_value(amt_str: str) -> int:
            """Extrae valor numérico de un monto para comparación."""
            digits = re.sub(r'[^\d]', '', amt_str)
            try:
                return int(digits) if digits else 0
            except ValueError:
                return 0

        # Estrategia 1: buscar SUBTOTAL primero (más confiable, OCR garbles TOTAL)
        # Variantes OCR: SUBTOTAL, SUBTTL, SUBTTOL, SUBTL, SUB TOTAL
        sub_m = re.search(
            r'\bSUB\s*T(?:OT|T)?(?:AL|L)?\b[^\n]{0,15}?'
            r'((?:Bs\.?|[\$€£])\s*[\d]+(?:[\s,\.]+\d+)*)',
            src, re.IGNORECASE)
        subtotal_amount = None
        if sub_m:
            subtotal_amount = _clean_bs_amount(sub_m.group(1))

        # Estrategia 2: buscar "TOTAL" (no SUBTOTAL) + Bs + número
        total_m = re.search(
            r'(?<!\bSUB)\bTOTAL\b[^\n]{0,25}?'
            r'((?:Bs\.?|[\$€£]|USD|COP|VES|MXN)\s*'
            r'[\d]+(?:[\s,\.]+\d+)*)',
            src, re.IGNORECASE)
        if total_m:
            candidate = _clean_bs_amount(total_m.group(1))
            total_val = _parse_amount_value(candidate)
            sub_val = _parse_amount_value(subtotal_amount) if subtotal_amount else 0
            # Usar TOTAL si es >= SUBTOTAL (lógico) o si no hay SUBTOTAL
            if total_val >= sub_val and total_val >= 1000:
                total_amount = candidate
            elif sub_val > 0:
                # TOTAL garbled, usar SUBTOTAL
                total_amount = subtotal_amount
        elif subtotal_amount:
            total_amount = subtotal_amount

        # Estrategia 3: tomar el monto más grande de ex.amounts
        if not total_amount and ex.amounts:
            best = None
            best_val = 0
            for a in ex.amounts:
                val = _parse_amount_value(a)
                # Filtrar montos absurdamente grandes (>100 dígitos = regex greedy)
                if val > best_val and len(re.sub(r'[^\d]', '', a)) <= 15:
                    best_val = val
                    best = a
            if best:
                total_amount = best

        # Estrategia 4: buscar el monto más grande en el texto directamente
        if not total_amount:
            all_bs = re.findall(
                r'(?:Bs\.?|[\$€£])\s*([\d]{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
                src, re.I)
            best_raw = None
            best_val = 0
            for a in all_bs:
                val = _parse_amount_value(a)
                if val > best_val:
                    best_val = val
                    best_raw = a
            if best_raw:
                total_amount = f"Bs {best_raw}"

        # --- 2. Contar productos/items ---
        # Productos: texto seguido de (E) o (G) y luego Bs/$/€ + número
        product_pattern = re.compile(
            r'([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑa-záéíóúñ\s\d]{2,45}?)'
            r'\s*(?:pk\.?|pK\.?)?\s*'
            r'\([EGeg]\)\s*'
            r'(?:Bs\.?|[\$€£])',
            re.IGNORECASE)
        products = product_pattern.findall(src)
        # Limpiar nombres de producto
        _SKIP_NAMES = frozenset({
            'TOTAL', 'SUBTOTAL', 'EXENTO', 'SUBTTL', 'EFECTIVO',
            'TARJETA', 'DEBITO', 'CREDITO', 'IVA', 'FACTURA',
            'NETO', 'BRUTO',
        })
        product_names = []
        for p in products:
            name = p.strip()
            # Eliminar peso/volumen del final: "500G", "1KG", "60ML", "900ML"
            name = re.sub(r'\s*\d+\s*(?:KG|G|ML|LT|OZ|LB|CC)\s*$', '', name, flags=re.I)
            # Eliminar multiplicadores: "0,79xBs"
            name = re.sub(r'\s*\d+[,\.]\d+x.*$', '', name, flags=re.I)
            name = name.strip()
            if len(name) > 2 and name.upper() not in _SKIP_NAMES:
                # Capitalizar: "ARROZ DONA BLANCA" → "Arroz Doña Blanca"
                product_names.append(name.title())
        num_items = len(product_names)

        # --- 3. Fecha ---
        fecha = None
        if ex.dates:
            fecha = ex.dates[0]
        else:
            # Buscar FECHA: dd/mm/yyyy o dd-mm-yyyy
            fecha_m = re.search(r'FECHA\s*:?\s*(\d{1,2}[\s/\-]\s*\d{1,2}[\s/\-]\s*\d{2,4})',
                                src, re.IGNORECASE)
            if fecha_m:
                fecha = re.sub(r'\s+', '', fecha_m.group(1))

        # --- 4. Método de pago ---
        metodo_pago = None
        if re.search(r'\bTARJETA\s+DEBITO\b', src, re.I):
            metodo_pago = "tarjeta débito"
        elif re.search(r'\bTARJETA\s+(?:DE\s+)?CR[ÉE]DITO\b', src, re.I):
            metodo_pago = "tarjeta de crédito"
        elif re.search(r'\bEFECTIVO\b', src, re.I):
            # Si hay tarjeta Y efectivo, es pago mixto
            if re.search(r'\bTARJETA\b', src, re.I):
                metodo_pago = "tarjeta y efectivo"
            else:
                metodo_pago = "efectivo"

        # --- 5. Número de factura ---
        num_factura = None
        nf_m = re.search(r'FACTURA\s*:?\s*#?\s*(\d{4,})', src, re.I)
        if nf_m:
            num_factura = nf_m.group(1)
        elif ex.ids:
            num_factura = ex.ids[0]

        # === CONSTRUIR NARRATIVA ===
        parts = []

        # Intro
        parts.append("Tienes una factura de compra.")

        # Total (lo más importante)
        if total_amount:
            parts.append(f"El total a pagar es {total_amount}.")

        # Resumen de productos
        if num_items > 0:
            if num_items <= 3:
                nombres = ", ".join(product_names[:3])
                parts.append(f"Compraste {nombres}.")
            elif num_items <= 8:
                top3 = ", ".join(product_names[:3])
                parts.append(f"Tiene {num_items} productos. Entre ellos: {top3}.")
            else:
                top3 = ", ".join(product_names[:3])
                parts.append(f"Tiene {num_items} productos. Los primeros son: {top3}, entre otros.")

        # Método de pago
        if metodo_pago:
            parts.append(f"Pagado con {metodo_pago}.")

        # Fecha
        if fecha:
            parts.append(f"Fecha: {fecha}.")

        # Número de factura (solo en detallado)
        if mode == "detallado" or mode == "financiero":
            if num_factura:
                parts.append(f"Factura número {num_factura}.")

        # Si no encontramos nada útil, dar resumen mínimo
        if len(parts) == 1:
            if ex.amounts:
                parts.append(f"Se ven montos como {ex.amounts[0]}.")
            else:
                parts.append("No pude leer los detalles con claridad. Intenta con mejor iluminación.")

        return " ".join(parts)

    # --- RECIBO ---

    def _gen_recibo(self, mode: str, text: str, ex: ExtractedData,
                    caption: Optional[str]) -> str:
        """Narrativa natural para recibos de pago."""
        # Detectar método de pago (heurística por keywords)
        metodo = None
        if re.search(r'\btarjeta\s+(?:de\s+)?cr[eé]dito\b', text, re.IGNORECASE):
            metodo = "tarjeta de crédito"
        elif re.search(r'\btarjeta\s+(?:de\s+)?d[eé]bito\b', text, re.IGNORECASE):
            metodo = "tarjeta débito"
        elif re.search(r'\btarjeta\b', text, re.IGNORECASE):
            metodo = "tarjeta"
        elif re.search(r'\befectivo\b', text, re.IGNORECASE):
            metodo = "efectivo"
        elif re.search(r'\btransferencia\b', text, re.IGNORECASE):
            metodo = "transferencia"
        elif re.search(r'\b(?:nequi|daviplata|pse|paypal)\b', text, re.IGNORECASE):
            m = re.search(r'\b(nequi|daviplata|pse|paypal)\b', text, re.IGNORECASE)
            metodo = m.group(1) if m else None

        if ex.amounts:
            intro = f"Tienes un recibo de pago por {ex.amounts[0]}."
        else:
            intro = "Tienes un recibo de pago."

        if mode == "resumen":
            parts = [intro]
            if metodo:
                parts.append(f"Pagado con {metodo}.")
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            return " ".join(parts)

        if mode == "financiero":
            parts = [intro]
            if ex.totals:
                parts.extend([f"{self._clean_total(t)}." for t in ex.totals])
            if metodo:
                parts.append(f"Método de pago: {metodo}.")
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            return " ".join(parts)

        # detallado
        parts = [intro]
        if metodo:
            parts.append(f"Pagado con {metodo}.")
        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")
        if ex.ids:
            parts.append(f"Comprobante número {ex.ids[0]}.")
        if ex.emails:
            parts.append(f"Correo de contacto: {ex.emails[0]}.")
        if ex.phones:
            parts.append(f"Teléfono: {ex.phones[0]}.")
        parts.append("Guarda este recibo como comprobante.")
        return " ".join(parts)

    # --- CARTA ---

    def _gen_carta(self, mode: str, text: str, ex: ExtractedData,
                   caption: Optional[str]) -> str:
        """Narrativa natural para cartas y oficios."""
        # Buscar remitente/destinatario
        remitente = None
        rem_match = re.search(
            r'(?:atentamente|cordialmente|firma)\s*:?\s*,?\s*([^\n]{3,40})',
            text, re.IGNORECASE)
        if rem_match:
            remitente = _cf(rem_match.group(1))

        if remitente:
            intro = f"Tienes una carta firmada por {remitente}."
        elif ex.headers:
            intro = f"Tienes una carta de {ex.headers[0]}."
        else:
            intro = "Tienes una carta."

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
                parts.append("Menciona estos montos: " + ", ".join(ex.amounts[:3]) + ".")
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            return " ".join(parts)

        # detallado
        parts = [intro]
        if ex.dates:
            parts.append(f"Con fecha {ex.dates[0]}.")
        if ex.headers and not remitente:
            parts.append(f"Asunto: {ex.headers[0]}.")
        body = self._extract_body_preview(text, max_words=40)
        if body:
            parts.append(f"Dice: {body}.")
        if ex.emails:
            parts.append(f"Correo de contacto: {ex.emails[0]}.")
        if ex.phones:
            parts.append(f"Teléfono: {ex.phones[0]}.")
        return " ".join(parts)

    # --- FORMULARIO ---

    def _gen_formulario(self, mode: str, text: str, ex: ExtractedData,
                        caption: Optional[str]) -> str:
        """Narrativa natural para formularios."""
        if ex.headers:
            intro = f"Tienes un formulario de {ex.headers[0]}."
        else:
            intro = "Tienes un formulario para llenar."

        # Detectar campos comunes
        campos = []
        campo_patterns = [
            (r'\bnombre\s+completo\b', 'nombre completo'),
            (r'\bcorreo|e-?mail\b', 'correo electrónico'),
            (r'\btel[eé]fono|celular\b', 'teléfono'),
            (r'\bdirecci[oó]n\b', 'dirección'),
            (r'\bc[eé]dula|documento\b', 'documento de identidad'),
            (r'\bfirma\b', 'firma'),
            (r'\bfecha\s+de\s+nacimiento\b', 'fecha de nacimiento'),
        ]
        for pat, label in campo_patterns:
            if re.search(pat, text, re.IGNORECASE):
                campos.append(label)

        if mode == "resumen":
            parts = [intro]
            if campos:
                parts.append(f"Te pide: {', '.join(campos[:3])}.")
            else:
                parts.append("Tiene varios campos para completar.")
            return " ".join(parts)

        # detallado
        parts = [intro]
        field_count = text.count(':')
        if campos:
            parts.append(f"Te pide datos como {', '.join(campos[:5])}.")
        elif field_count > 2:
            parts.append(f"Tiene alrededor de {field_count} campos para llenar.")
        if ex.ids:
            parts.append(f"Pide tu identificación: {ex.ids[0]}.")
        if ex.dates:
            parts.append(f"Fecha incluida: {ex.dates[0]}.")
        if ex.phones:
            parts.append(f"Teléfono de contacto: {ex.phones[0]}.")
        return " ".join(parts)

    # --- DOCUMENTO INFORMATIVO ---

    def _gen_documento_info(self, mode: str, text: str, ex: ExtractedData,
                            caption: Optional[str]) -> str:
        """Narrativa natural para documentos informativos genéricos."""
        if ex.headers:
            intro = f"Tienes un documento sobre {ex.headers[0]}."
        else:
            intro = "Tienes un documento informativo."

        if mode == "resumen":
            parts = [intro]
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            first = self._extract_first_sentence(text, skip_lines=2)
            if first and (not ex.headers or first.lower() != ex.headers[0].lower()):
                parts.append(f"Comienza diciendo: {first}")
            return " ".join(parts)

        # detallado
        parts = [intro]
        if ex.dates:
            parts.append(f"Con fecha {ex.dates[0]}.")
        body = self._extract_body_preview(text, max_words=50)
        if body:
            parts.append(f"Dice: {body}.")
        return " ".join(parts)

    # --- IMAGEN VISUAL ---

    def _gen_imagen_visual(self, mode: str, text: str, ex: ExtractedData,
                           caption: Optional[str]) -> str:
        parts = []
        if caption:
            parts.append(f"La imagen muestra: {caption}.")
        if text and text.strip():
            preview = self._extract_body_preview(text, max_words=30)
            if preview:
                parts.append(f"El texto visible dice: {preview}.")
        elif not caption:
            parts.append("No se detectó texto claro en la imagen.")
        return " ".join(parts) if parts else "Imagen sin texto legible detectado."

    # --- ETIQUETA DE PRODUCTO ---

    def _gen_etiqueta(self, mode: str, text: str, ex: ExtractedData,
                      caption: Optional[str]) -> str:
        """Genera narrativa natural para etiquetas de productos."""
        if not text or not text.strip():
            return "Es una etiqueta de producto, pero no se pudo leer."

        # Detectar nombre del producto (primera línea larga o header)
        product_name = None
        if ex.headers:
            product_name = ex.headers[0]
        else:
            first = self._extract_first_sentence(text, skip_lines=0)
            if first and len(first.split()) <= 8:
                product_name = first

        if product_name:
            parts = [f"Es la etiqueta de {product_name}."]
        else:
            parts = ["Es una etiqueta de producto."]

        # Buscar precio
        price_match = re.search(
            r'(?:precio|price)\s*:?\s*([\$\€£Bs\.]*\s*[\d,.]+)',
            text, re.IGNORECASE
        )
        if price_match:
            parts.append(f"Precio: {price_match.group(1).strip()}.")
        elif ex.amounts:
            parts.append(f"Precio: {ex.amounts[0]}.")

        # Buscar vencimiento
        venc_match = re.search(
            r'(?:venc(?:imiento|e)?|exp(?:iry|iration)?|caduc(?:idad|a)?)\s*:?\s*([^\n]{3,25})',
            text, re.IGNORECASE
        )
        if venc_match:
            venc_val = venc_match.group(1).strip()
            if venc_val:
                parts.append(f"Vence en {venc_val}.")
        elif ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")

        # Buscar peso neto
        peso_match = re.search(
            r'(?:peso\s+neto|net\s+w(?:eight|t)?)\s*:?\s*([\d,.]+\s*(?:g|kg|ml|l|oz|lb)\b)',
            text, re.IGNORECASE
        )
        if peso_match:
            parts.append(f"Peso neto: {peso_match.group(1).strip()}.")

        # Buscar calorías / valor nutricional
        cal_match = re.search(
            r'(?:calor[ií]as?|energ[ií]a|kcal)\s*:?\s*([\d,.]+\s*(?:kcal|cal)?)',
            text, re.IGNORECASE
        )
        if cal_match:
            parts.append(f"Calorías: {cal_match.group(1).strip()}.")

        if mode == "resumen":
            return " ".join(parts)

        # Detallado: buscar ingredientes
        ingr_match = re.search(
            r'ingredientes\s*:?\s*([^\n]{5,200})',
            text, re.IGNORECASE
        )
        if ingr_match:
            ingredientes = _cf(ingr_match.group(1))
            # Limitar largo
            words = ingredientes.split()
            if len(words) > 25:
                ingredientes = " ".join(words[:25]) + "..."
            parts.append(f"Ingredientes: {ingredientes}.")

        # Buscar fabricante
        fab_match = re.search(
            r'(?:fabricado|elaborado|producido|hecho)\s+(?:por|en)\s*:?\s*([^\n]{3,50})',
            text, re.IGNORECASE
        )
        if fab_match:
            parts.append(f"Fabricado por: {_cf(fab_match.group(1))}.")

        # Lote
        lote_match = re.search(
            r'(?:lote|lot|batch)\s*:?\s*([A-Za-z0-9\-]{2,20})',
            text, re.IGNORECASE
        )
        if lote_match:
            parts.append(f"Lote: {lote_match.group(1).strip()}.")

        return " ".join(parts)

    # --- MENÚ DE RESTAURANTE ---

    def _gen_menu(self, mode: str, text: str, ex: ExtractedData,
                  caption: Optional[str]) -> str:
        """Genera narrativa natural para menús de restaurante."""
        if not text or not text.strip():
            return "Es un menú, pero no se pudo leer."

        # Detectar nombre del restaurante (header o primera línea prominente)
        restaurant = None
        if ex.headers:
            restaurant = ex.headers[0]
        if restaurant:
            parts = [f"Es el menú de {restaurant}."]
        else:
            parts = ["Tienes un menú de restaurante."]

        # Detectar secciones del menú
        sections_found = []
        section_patterns = [
            (r'\bentrantes?\b|\bstarters?\b|\baperitivs?\b', 'Entrantes'),
            (r'\bplatos?\s+(?:principal(?:es)?|fuert(?:es?|e))\b|\bmain\b', 'Platos principales'),
            (r'\bpostres?\b|\bdesserts?\b', 'Postres'),
            (r'\bbebidas?\b|\bdrinks?\b|\brefrescos?\b', 'Bebidas'),
            (r'\bensaladas?\b|\bsalads?\b', 'Ensaladas'),
            (r'\bsopas?\b|\bcremas?\b', 'Sopas'),
            (r'\bcombos?\b|\bpromoci[oó]n(?:es)?\b', 'Combos/Promociones'),
            (r'\bpizzas?\b', 'Pizzas'),
            (r'\bhamburguesas?\b|\bburgers?\b', 'Hamburguesas'),
            (r'\bpastas?\b', 'Pastas'),
        ]
        for pattern, label in section_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                sections_found.append(label)

        if sections_found:
            sects = ", ".join(sections_found[:5])
            parts.append(f"Secciones: {sects}.")

        # Extraer items con precios: "Ensalada César $12.99" o "Pasta ... Bs. 250"
        items = []
        item_pattern = re.compile(
            r'([A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ\s]{2,30})\s+'
            r'(?:[\$\€£]|Bs\.?\s*)\s*([\d,.]+)',
            re.IGNORECASE
        )
        for m in item_pattern.finditer(text):
            item_name = _cf(m.group(1))
            item_price = m.group(2).strip()
            # Filtrar basura (no items de 1-2 words que sean secciones)
            if len(item_name.split()) >= 1 and len(item_name) > 3:
                items.append((item_name, item_price))

        if mode == "resumen":
            if items:
                count = len(items)
                parts.append(f"Tiene {count} platos con precio visible.")
                # Mostrar rango de precios
                try:
                    prices = [float(p.replace(',', '.')) for _, p in items]
                    min_p = min(prices)
                    max_p = max(prices)
                    if min_p != max_p:
                        parts.append(f"Precios desde {min_p:.0f} hasta {max_p:.0f}.")
                    else:
                        parts.append(f"Precio: {min_p:.0f}.")
                except ValueError:
                    pass
            elif ex.amounts:
                parts.append(f"Se ven precios como {ex.amounts[0]}.")
            return " ".join(parts)

        # Detallado: listar items (hasta 8)
        if items:
            parts.append(f"Se encontraron {len(items)} platos:")
            for name, price in items[:8]:
                parts.append(f"{name}, {price}.")
            if len(items) > 8:
                parts.append(f"Y {len(items) - 8} platos más.")
        else:
            # No se detectaron items con precio, leer contenido general
            body = self._extract_body_preview(text, max_words=50)
            if body:
                parts.append(f"Dice: {body}.")

        return " ".join(parts)

    # --- CHAT ---

    # Patrones de UI del teléfono / status bar / chrome de chat apps
    _PHONE_UI_PATTERNS = re.compile(
        r'^\d{1,2}:\d{2}\s*$|'                     # solo timestamp (3:03)
        r'^all\s*$|^ED\s*$|^You\s*$|'              # fragmentos de status bar
        r'^\d+%\s*$|'                               # porcentaje de batería
        r'^[A-Z]{1,3}\s*$',                         # fragmentos cortos en mayúscula
        re.IGNORECASE
    )

    def _gen_chat(self, mode: str, text: str, ex: ExtractedData,
                  caption: Optional[str]) -> str:
        """Genera narrativa natural para capturas de chat."""
        if not text or not text.strip():
            return "Es un chat, pero no se pudo leer."

        # Detectar contacto/grupo
        contact = None
        # El nombre del contacto aparece en la primera línea (header del chat)
        # Típicamente: "Brayan" o "Mamá" antes de los timestamps/mensajes
        first_line = text.split('\n')[0] if '\n' in text else text[:40]
        header = first_line[:40]
        # Limpiar timestamps, basura de status bar, indicadores de estado
        header_clean = re.sub(r'\d{1,2}:\d{2}\s*(?:[AaPp]\.?[Mm]\.?)*', '', header)
        header_clean = re.sub(r'\b(?:all|ED|GD|You|Fe|PM|AM|en\s+l[ií]nea|online|'
                              r'ult(?:ima|\.)?\s*vez|escribir?\s+mensaje|'
                              r'WhatsApp|Telegram|Messenger|Instagram|Signal)\b',
                              '', header_clean, flags=re.IGNORECASE).strip()

        # Buscar nombre propio (1-3 palabras capitalizadas, >3 chars total)
        _chat_ui_words = {
            'Lectura', 'Implementar', 'Nueva', 'Imagen', 'Repetir',
            'Overview', 'Publicación', 'Notificación', 'Vale', 'Que',
            'No', 'Si', 'Los', 'Las', 'Hay', 'Hoy', 'Ayer',
            'Escribe', 'Escribir', 'Mensaje', 'Chat', 'Grupo',
            'Audio', 'Foto', 'Video', 'Sticker', 'Archivo',
        }
        contact_match = re.search(
            r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})',
            header_clean
        )
        if contact_match:
            name = contact_match.group(1)
            name_words = name.split()
            # Filter: all words must NOT be UI words, and name must be > 3 chars
            if (all(w not in _chat_ui_words for w in name_words)
                    and len(name) > 3
                    and len(name_words) <= 3):
                contact = name

        # 3. Detectar plataforma
        platform = None
        for p in ['WhatsApp', 'Telegram', 'Messenger', 'Instagram', 'Signal', 'iMessage']:
            if re.search(rf'\b{p}\b', text, re.IGNORECASE):
                platform = p
                break

        # 4. Limpiar y extraer mensajes
        # Quitar timestamps, status bar, UI
        clean = re.sub(r'\d{1,2}:\d{2}\s*(?:[AaPp]\.?[Mm]\.?){1,2}\b', ' ', text)
        clean = re.sub(r'\b(?:all|ED|You|PM|AM|GD)\b', ' ', clean)
        clean = self._clean_ocr_words(clean)

        # Segmentar en mensajes
        segments = re.split(r'(?<=[a-záéíóúñ])\s+(?=[A-ZÁÉÍÓÚÑ][a-záéíóúñ])', clean)
        messages = []
        for seg in segments:
            seg = seg.strip()
            real = [w for w in seg.split()
                    if re.search(r'[a-záéíóúñ]', w, re.IGNORECASE) and len(w) > 1]
            if len(real) >= 2:
                # Limitar largo de cada mensaje
                messages.append(" ".join(seg.split()[:20]))

        # 5. Construir narrativa natural
        parts = []

        # Intro contextual
        if platform and contact:
            parts.append(f"Es un chat de {platform} con {contact}.")
        elif contact:
            parts.append(f"Es un chat con {contact}.")
        elif platform:
            parts.append(f"Es un chat de {platform}.")
        else:
            parts.append("Es una conversación de chat.")

        # Mensajes
        if not messages:
            parts.append("No se pudieron leer los mensajes.")
        elif len(messages) == 1:
            parts.append(f"El mensaje dice: {messages[0]}.")
        elif len(messages) <= 3:
            for msg in messages:
                parts.append(f"{msg}.")
        else:
            count = len(messages)
            parts.append(f"Hay {count} mensajes.")
            # Leer los últimos 3
            for msg in messages[-3:]:
                parts.append(f"{msg}.")

        return " ".join(parts)

    # --- NOTIFICACIÓN ---

    def _gen_notificacion(self, mode: str, text: str, ex: ExtractedData,
                          caption: Optional[str]) -> str:
        """Narrativa natural para notificaciones y alertas."""
        if not text or not text.strip():
            return "Tienes una notificación, pero no pude leer el contenido."

        # Detectar app de la notificación
        app = None
        app_match = re.search(
            r'\b(WhatsApp|Gmail|Instagram|Facebook|Twitter|Uber|Rappi|'
            r'YouTube|Spotify|Netflix|Telegram|TikTok|LinkedIn|Slack|'
            r'Teams|Outlook|Calendar|Maps|Waze)\b', text, re.IGNORECASE)
        if app_match:
            app = app_match.group(1)

        if app:
            parts = [f"Tienes una notificación de {app}."]
        else:
            parts = ["Tienes una notificación."]

        body = self._extract_body_preview(text, max_words=40)
        if body:
            parts.append(f"Dice: {body}.")
        return " ".join(parts)

    # --- LOGIN ---

    def _gen_login(self, mode: str, text: str, ex: ExtractedData,
                   caption: Optional[str]) -> str:
        """Narrativa natural para pantallas de inicio de sesión."""
        if not text or not text.strip():
            return "Estás en una pantalla de inicio de sesión."

        # Detectar qué servicio/app es
        service = None
        service_match = re.search(
            r'\b(Google|Facebook|Apple|Instagram|Twitter|X|GitHub|Microsoft|'
            r'Outlook|Netflix|Spotify|Amazon|WhatsApp|Telegram|LinkedIn|'
            r'TikTok|Snapchat|Discord|Uber|PayPal)\b', text, re.IGNORECASE)
        if service_match:
            service = service_match.group(1)

        if service:
            parts = [f"Estás en la pantalla de inicio de sesión de {service}."]
        else:
            parts = ["Estás en una pantalla de inicio de sesión."]

        # Detectar campos visibles
        has_password = bool(re.search(r'\b(?:contrase[ñn]a|password|clave)\b', text, re.IGNORECASE))
        has_user = bool(re.search(r'\b(?:usuario|email|correo|user)\b', text, re.IGNORECASE))
        has_register = bool(re.search(r'\b(?:registr|sign\s*up|crear?\s+cuenta|create)\b', text, re.IGNORECASE))
        has_forgot = bool(re.search(r'\b(?:olvid|forgot|recuperar)\b', text, re.IGNORECASE))

        if has_user and has_password:
            parts.append("Te pide usuario y contraseña.")
        elif has_user:
            parts.append("Te pide ingresar tu usuario o correo.")
        if has_register:
            parts.append("También puedes crear una cuenta nueva.")
        if has_forgot:
            parts.append("Si olvidaste tu contraseña, hay opción para recuperarla.")

        return " ".join(parts)

    # --- RED SOCIAL ---

    def _gen_red_social(self, mode: str, text: str, ex: ExtractedData,
                        caption: Optional[str]) -> str:
        """Genera narrativa natural para perfiles y publicaciones de redes sociales."""
        if not text or not text.strip():
            return "Es una red social, pero no se pudo leer el contenido."

        parts = []

        # 1. Detectar plataforma
        platform_patterns = {
            'GitHub': r'\b(?:GitHub|Repositories|repository|repos|contribution|follower|following|Overview)\b',
            'Instagram': r'\b(?:Instagram|Reels?|Stories)\b',
            'Twitter': r'\b(?:Twitter|tweet|retweet)\b',
            'X': r'\b(?:^X$|x\.com)\b',
            'TikTok': r'\b(?:TikTok|For\s+You)\b',
            'Facebook': r'\b(?:Facebook|fb\.com)\b',
            'YouTube': r'\b(?:YouTube|suscri|subscribers?|views)\b',
            'LinkedIn': r'\b(?:LinkedIn|connections?|experience)\b',
            'Reddit': r'\b(?:Reddit|subreddit|upvote)\b',
        }
        platform = None
        for name, pattern in platform_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                platform = name
                break

        # 2. Detectar si es PERFIL o PUBLICACIÓN
        is_profile = bool(re.search(
            r'\b(?:follower|following|seguidores|contribution|bio|'
            r'repos|repositories|Overview|connections)\b',
            text, re.IGNORECASE
        ))

        # 3. Extraer nombre de usuario/persona
        # Filtrar palabras de UI que no son nombres
        _UI_WORDS = {
            'Overview', 'Repositories', 'Repository', 'Projects', 'Packages',
            'Popular', 'Pinned', 'Contribution', 'Contributions', 'Following',
            'Followers', 'Follower', 'Settings', 'Profile', 'Posts', 'Feed',
            'Timeline', 'Stories', 'Reels', 'Home', 'Search', 'Explore',
            'Activity', 'Notifications', 'Messages', 'Share', 'Comment',
            'Servicios', 'Java', 'Python',
        }
        # Buscar patrones de nombre (2+ palabras capitalizadas consecutivas, no UI)
        name_match = None
        for m in re.finditer(
            r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})',
            text
        ):
            words_in_name = m.group(1).split()
            if not any(w in _UI_WORDS for w in words_in_name):
                name_match = m
                break
        username_match = re.search(r'@(\w{2,30})', text)

        # 4. Extraer métricas numéricas
        metrics = {}
        metric_patterns = [
            (r'(\d[\d,.]*)\s*(?:follower|seguidore)s?', 'seguidores'),
            (r'(\d[\d,.]*)\s*(?:following|siguiendo)', 'siguiendo'),
            (r'(\d[\d,.]*)\s*(?:contribution|contribuci[oó]n)', 'contribuciones'),
            (r'(?:Repositories?|repos)\s*(\d+)', 'repositorios'),
            (r'(\d[\d,.]*[KkMm]?)\s*(?:likes?|me\s+gusta)', 'me gusta'),
            (r'(\d[\d,.]*[KkMm]?)\s*(?:comments?|comentarios?)', 'comentarios'),
            (r'(\d[\d,.]*[KkMm]?)\s*(?:views?|vistas|reproducciones)', 'vistas'),
            (r'(\d[\d,.]*[KkMm]?)\s*(?:subscribers?|suscriptores?)', 'suscriptores'),
            (r'(\d[\d,.]*)\s*(?:posts?|publicaciones?)', 'publicaciones'),
        ]
        for pattern, label in metric_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                metrics[label] = m.group(1)

        # 5. Construir narrativa natural
        if is_profile:
            if platform:
                parts.append(f"Es un perfil de {platform}.")
            else:
                parts.append("Es un perfil de red social.")

            if name_match:
                parts.append(f"El nombre es {name_match.group(1)}.")
            elif username_match:
                parts.append(f"El usuario es {username_match.group(0)}.")

            if metrics:
                metric_parts = []
                for label, value in list(metrics.items())[:4]:
                    metric_parts.append(f"{value} {label}")
                parts.append("Tiene " + ", ".join(metric_parts) + ".")
        else:
            # Publicación
            if platform:
                parts.append(f"Es una publicación de {platform}.")
            else:
                parts.append("Es una publicación de red social.")

            if username_match:
                parts.append(f"De {username_match.group(0)}.")
            elif name_match:
                parts.append(f"De {name_match.group(1)}.")

            # Hashtags
            hashtags = re.findall(r'#(\w{2,30})', text)
            if hashtags:
                tags = ", ".join(hashtags[:3])
                parts.append(f"Etiquetas: {tags}.")

            if metrics:
                metric_parts = [f"{v} {k}" for k, v in list(metrics.items())[:3]]
                parts.append(", ".join(metric_parts) + ".")

            # Contenido del post (solo frases significativas)
            phrases = self._extract_meaningful_phrases(text, max_phrases=3, min_words=3)
            if phrases:
                best = phrases[0]
                parts.append(f"Dice: {best}.")

        return " ".join(parts) if parts else "Red social sin contenido legible."

    # --- NOTICIA / ARTÍCULO ---

    def _gen_noticia(self, mode: str, text: str, ex: ExtractedData,
                     caption: Optional[str]) -> str:
        """Narrativa natural para noticias, artículos, blogs, portadas."""
        if not text or not text.strip():
            return "Hay una noticia, pero no pude leer el contenido."

        # Buscar fuente/medio
        source_name = None
        source_match = re.search(
            r'\b(Reuters|AP|AFP|EFE|CNN|BBC|El\s+Pa[ií]s|El\s+Nacional|'
            r'El\s+Universal|El\s+Tiempo|New\s+York\s+Times|The\s+Guardian|'
            r'Washington\s+Post|Forbes|Bloomberg|Semana|Portafolio|'
            r'La\s+Rep[uú]blica|Infobae)\b', text, re.IGNORECASE)
        if source_match:
            source_name = source_match.group(1)

        # Extraer título
        titulo = None
        first = self._extract_first_sentence(text, skip_lines=0)
        if first and source_name and first.strip().lower() == source_name.strip().lower():
            first = self._extract_first_sentence(text, skip_lines=1)
        if first:
            titulo = first

        # Intro contextual
        if source_name and titulo:
            parts = [f"Hay una noticia de {source_name}. El titular dice: {titulo}."]
        elif source_name:
            parts = [f"Hay una noticia de {source_name}."]
        elif titulo:
            parts = [f"Hay una noticia. El titular dice: {titulo}."]
        else:
            parts = ["Hay un artículo o noticia."]

        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")

        if mode != "resumen":
            body_text = text
            if source_name:
                body_text = re.sub(
                    r'\b' + re.escape(source_name) + r'\b',
                    '', body_text, count=1, flags=re.IGNORECASE
                ).strip()
            body = self._extract_body_preview(body_text, max_words=60)
            if body and (not titulo or titulo.lower() not in body[:50].lower()):
                parts.append(f"Dice: {body}.")

        return " ".join(parts)

    # --- CORREO ELECTRÓNICO ---

    def _gen_correo(self, mode: str, text: str, ex: ExtractedData,
                    caption: Optional[str]) -> str:
        """Narrativa natural para correos electrónicos."""
        if not text or not text.strip():
            return "Tienes un correo, pero no pude leer el contenido."

        # Extraer remitente
        remitente = None
        from_match = re.search(r'(?:de|from)\s*:?\s*([^\n]{3,40})', text, re.IGNORECASE)
        if from_match:
            remitente = _cf(from_match.group(1))

        # Extraer asunto
        subject_text = None
        subject_match = re.search(r'(?:asunto|subject)\s*:?\s*([^\n]{3,60})', text, re.IGNORECASE)
        if subject_match:
            subject_text = _cf(subject_match.group(1))

        # Intro contextual
        if remitente and subject_text:
            parts = [f"Tienes un correo de {remitente}. Asunto: {subject_text}."]
        elif remitente:
            parts = [f"Tienes un correo de {remitente}."]
        elif subject_text:
            parts = [f"Tienes un correo con asunto: {subject_text}."]
        else:
            parts = ["Tienes un correo electrónico."]

        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")

        if mode == "detallado":
            body_text = text
            body_text = re.sub(r'(?:de|from)\s*:?\s*[^\n]{3,40}', '', body_text, count=1, flags=re.IGNORECASE)
            body_text = re.sub(r'(?:asunto|subject)\s*:?\s*[^\n]{3,60}', '', body_text, count=1, flags=re.IGNORECASE)
            body = self._extract_body_preview(body_text, max_words=50)
            if body and (not subject_text or subject_text.lower() not in body.lower()):
                parts.append(f"El mensaje dice: {body}.")

        return " ".join(parts)

    # --- PRESENTACIÓN ---

    def _gen_presentacion(self, mode: str, text: str, ex: ExtractedData,
                          caption: Optional[str]) -> str:
        """Narrativa natural para diapositivas/presentaciones."""
        if not text or not text.strip():
            return "Es una diapositiva, pero no pude leer el contenido."

        parts = []

        # Detectar número de slide
        slide_match = re.search(r'(\d+)\s*/\s*(\d+)', text)

        # El título suele ser el texto más prominente (primera línea larga)
        first = self._extract_first_sentence(text, skip_lines=0)

        if slide_match and first:
            parts.append(f"Es la diapositiva {slide_match.group(1)} de {slide_match.group(2)}. Título: {first}.")
        elif slide_match:
            parts.append(f"Es la diapositiva {slide_match.group(1)} de {slide_match.group(2)}.")
        elif first:
            parts.append(f"Es una diapositiva titulada: {first}.")
        else:
            parts.append("Es una diapositiva de presentación.")

        body = self._extract_body_preview(text, max_words=40)
        if body and body != first:
            parts.append(f"Dice: {body}.")

        return " ".join(parts)

    # --- CONFIGURACIÓN ---

    def _gen_configuracion(self, mode: str, text: str, ex: ExtractedData,
                           caption: Optional[str]) -> str:
        """Narrativa natural para pantallas de configuración/ajustes."""
        if not text or not text.strip():
            return "Estás en una pantalla de ajustes."

        # Detectar opciones visibles
        options = []
        option_patterns = [
            (r'\bWi-?Fi\b', 'Wi-Fi'),
            (r'\bBluetooth\b', 'Bluetooth'),
            (r'\b(?:datos?\s+m[oó]viles|mobile\s+data)\b', 'datos móviles'),
            (r'\b(?:brillo|brightness)\b', 'brillo'),
            (r'\b(?:volumen|volume|sonido|sound)\b', 'sonido'),
            (r'\b(?:bater[ií]a|battery)\b', 'batería'),
            (r'\b(?:almacenamiento|storage)\b', 'almacenamiento'),
            (r'\b(?:modo\s+oscuro|dark\s+mode)\b', 'modo oscuro'),
            (r'\b(?:modo\s+avi[oó]n|airplane)\b', 'modo avión'),
            (r'\b(?:notificaciones|notifications)\b', 'notificaciones'),
            (r'\b(?:privacidad|privacy)\b', 'privacidad'),
            (r'\b(?:accesibilidad|accessibility)\b', 'accesibilidad'),
            (r'\b(?:pantalla|display)\b', 'pantalla'),
            (r'\b(?:idioma|language)\b', 'idioma'),
            (r'\b(?:cuenta|account)\b', 'cuenta'),
        ]
        for pattern, label in option_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                options.append(label)

        if options:
            visible = ", ".join(options[:6])
            parts = [f"Estás en los ajustes del teléfono. Se ven las opciones de {visible}."]
        else:
            parts = ["Estás en una pantalla de configuración."]
            body = self._extract_body_preview(text, max_words=30)
            if body:
                parts.append(f"Dice: {body}.")

        return " ".join(parts)

    # --- TARJETA DE PRESENTACIÓN ---

    def _gen_tarjeta(self, mode: str, text: str, ex: ExtractedData,
                     caption: Optional[str]) -> str:
        """Narrativa natural para tarjetas de presentación / business cards."""
        if not text or not text.strip():
            return "Es una tarjeta de presentación, pero no pude leerla."

        # Nombre: típicamente primera línea prominente o header
        name = None
        if ex.headers:
            name = ex.headers[0]
        else:
            first = self._extract_first_sentence(text, skip_lines=0)
            if first and len(first.split()) <= 5:
                name = first

        if name:
            parts = [f"Es la tarjeta de presentación de {name}."]
        else:
            parts = ["Es una tarjeta de presentación."]

        # Cargo/título profesional
        cargo_match = re.search(
            r'(?:cargo|puesto|título|posición|position|title)\s*:?\s*([^\n]{3,40})',
            text, re.IGNORECASE
        )
        if cargo_match:
            parts.append(f"Cargo: {_cf(cargo_match.group(1))}.")
        else:
            # Heurística: segunda línea corta suele ser el cargo
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if len(lines) >= 2 and len(lines[1].split()) <= 6:
                parts.append(f"Cargo: {_cf(lines[1])}.")

        # Empresa/organización
        empresa_match = re.search(
            r'(?:empresa|company|organización|compañía)\s*:?\s*([^\n]{3,40})',
            text, re.IGNORECASE
        )
        if empresa_match:
            parts.append(f"Empresa: {_cf(empresa_match.group(1))}.")

        if mode == "resumen":
            # En resumen, solo nombre + cargo + contacto principal
            if ex.phones:
                parts.append(f"Teléfono: {ex.phones[0]}.")
            elif ex.emails:
                parts.append(f"Correo: {ex.emails[0]}.")
            return " ".join(parts)

        # Detallado: todos los datos de contacto
        if ex.phones:
            if len(ex.phones) == 1:
                parts.append(f"Teléfono: {ex.phones[0]}.")
            else:
                parts.append("Teléfonos: " + ", ".join(ex.phones[:3]) + ".")

        if ex.emails:
            if len(ex.emails) == 1:
                parts.append(f"Correo: {ex.emails[0]}.")
            else:
                parts.append("Correos: " + ", ".join(ex.emails[:2]) + ".")

        # Dirección (heurística: línea larga con números)
        addr_match = re.search(
            r'(?:direcci[oó]n|address|ubicaci[oó]n)\s*:?\s*([^\n]{5,60})',
            text, re.IGNORECASE
        )
        if addr_match:
            parts.append(f"Dirección: {_cf(addr_match.group(1))}.")

        # Sitio web (excluir emails)
        web_match = re.search(
            r'(?:www\.\S+|https?://\S+)',
            text, re.IGNORECASE
        )
        if web_match:
            parts.append(f"Web: {web_match.group(0).strip()}.")

        return " ".join(parts)

    # --- RECETA MÉDICA ---

    def _gen_receta_medica(self, mode: str, text: str, ex: ExtractedData,
                           caption: Optional[str]) -> str:
        """Narrativa natural para recetas médicas. Prioriza: medicamento, dosis, frecuencia."""
        if not text or not text.strip():
            return "Es una receta médica, pero no pude leer el contenido. Intenta acercar más la cámara."

        parts = []

        # Doctor
        doc_match = re.search(
            r'\b(?:Dr\.?|Dra\.?|Doctor[a]?)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})',
            text, re.IGNORECASE)
        if doc_match:
            parts.append(f"Tienes una receta del doctor {doc_match.group(1)}.")
        else:
            parts.append("Tienes una receta médica.")

        # Medicamentos (buscar nombres + dosis)
        med_patterns = [
            re.compile(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ]?[a-záéíóúñ]+)?)\s+(\d+)\s*(mg|ml|g|mcg|UI)\b', re.I),
            re.compile(r'\b(acetaminof[eé]n|ibuprofeno|amoxicilina|omeprazol|metformina|losart[aá]n|diclofenaco|naproxeno|azitromicina|ciprofloxacin[oa]|prednisona|loratadina|cetirizina|atorvastatina|amlodipino|metoprolol)\s*(\d+)?\s*(mg|ml|g)?\b', re.I),
        ]
        meds_found = []
        for pat in med_patterns:
            for m in pat.finditer(text):
                name = m.group(1).strip()
                dose = m.group(2) if m.group(2) else ""
                unit = m.group(3) if m.group(3) else ""
                med_str = name
                if dose:
                    med_str += f" de {dose}"
                if unit:
                    med_str += f" {unit}"
                if med_str not in meds_found:
                    meds_found.append(med_str)

        if meds_found:
            if len(meds_found) == 1:
                parts.append(f"Te recetaron {meds_found[0]}.")
            else:
                parts.append("Te recetaron los siguientes medicamentos.")
                for med in meds_found[:4]:
                    parts.append(f"{med}.")
        else:
            body = self._extract_body_preview(text, max_words=15)
            if body:
                parts.append(f"Contiene: {body}.")

        # Frecuencia
        freq_match = re.search(r'\bcada\s+(\d+)\s*(horas?|d[ií]as?)\b', text, re.I)
        if freq_match:
            num = freq_match.group(1)
            unit = freq_match.group(2).lower()
            if 'hora' in unit:
                veces = 24 // int(num) if int(num) > 0 else 0
                parts.append(f"Tómalo cada {num} horas, es decir, {veces} veces al día.")
            else:
                parts.append(f"Tómalo cada {num} {unit}.")

        # Duración
        dur_match = re.search(r'\b(?:durante|por)\s+(\d+)\s*(d[ií]as?|semanas?)\b', text, re.I)
        if dur_match:
            parts.append(f"El tratamiento dura {dur_match.group(1)} {dur_match.group(2)}.")

        # Vía de administración
        via_match = re.search(r'\bv[ií]a\s+(oral|t[oó]pica|intramuscular|intravenosa|subling[uü]al|nasal|rectal)\b', text, re.I)
        if via_match:
            parts.append(f"Se administra por vía {via_match.group(1)}.")

        # Indicaciones especiales
        if re.search(r'\b(?:en\s+ayunas|antes\s+de\s+comer)\b', text, re.I):
            parts.append("Tómalo en ayunas, antes de comer.")
        elif re.search(r'\b(?:despu[eé]s\s+de\s+(?:comer|las\s+comidas)|con\s+(?:las\s+)?comidas?)\b', text, re.I):
            parts.append("Tómalo después de comer para evitar malestar estomacal.")

        if mode != "resumen":
            parts.append("Recuerda completar todo el tratamiento aunque te sientas mejor.")

        return " ".join(parts)

    # --- BOLETO / TICKET ---

    def _gen_boleto(self, mode: str, text: str, ex: ExtractedData,
                    caption: Optional[str]) -> str:
        """Narrativa natural para boletos de avión, bus, tren o eventos."""
        if not text or not text.strip():
            return "Es un boleto o ticket, pero no pude leer los detalles. Intenta acercar más la cámara."

        parts = []

        # Detectar tipo de boleto
        is_flight = bool(re.search(r'\b(?:vuelo|flight|boarding|embarque|aerol[ií]nea|airline)\b', text, re.I))
        is_event = bool(re.search(r'\b(?:concierto|cine|teatro|evento|funci[oó]n|espect[aá]culo|show|entrada)\b', text, re.I))
        is_bus = bool(re.search(r'\b(?:bus|aut[oó]bus|terminal\s+(?:de\s+)?(?:buses|transporte))\b', text, re.I))
        is_train = bool(re.search(r'\b(?:tren|train|ferrocarril|and[eé]n)\b', text, re.I))

        if is_flight:
            parts.append("Tienes un boleto de avión.")
        elif is_event:
            parts.append("Tienes una entrada para un evento.")
        elif is_bus:
            parts.append("Tienes un boleto de bus.")
        elif is_train:
            parts.append("Tienes un boleto de tren.")
        else:
            parts.append("Tienes un boleto o ticket.")

        # Aerolínea / compañía
        airline_match = re.search(r'\b(Avianca|LATAM|Copa|Wingo|JetBlue|American|Delta|United|Volaris|VivaAerobus|Aeroméxico|Spirit)\b', text, re.I)
        if airline_match:
            parts.append(f"Con {airline_match.group(1)}.")

        # Número de vuelo/ruta
        flight_match = re.search(r'\b(?:vuelo|flight)\s*(?:No\.?|#)?\s*([A-Z]{0,2}\d{2,5})\b', text, re.I)
        if flight_match:
            parts.append(f"Vuelo número {flight_match.group(1)}.")

        # Origen y destino
        origin_match = re.search(r'\b(?:origen|from|de|salida)\s*:?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-Za-záéíóúñ]+)?)', text, re.I)
        dest_match = re.search(r'\b(?:destino|to|hacia|llegada)\s*:?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-Za-záéíóúñ]+)?)', text, re.I)
        if origin_match and dest_match:
            parts.append(f"Desde {origin_match.group(1)} hasta {dest_match.group(1)}.")
        elif dest_match:
            parts.append(f"Con destino a {dest_match.group(1)}.")

        # Fecha y hora
        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")
        hora_match = re.search(r'\b(?:salida|departure|hora)\s*:?\s*(\d{1,2}:\d{2})\b', text, re.I)
        if hora_match:
            parts.append(f"Hora de salida: {hora_match.group(1)}.")

        # Asiento
        seat_match = re.search(r'\b(?:asiento|seat)\s*:?\s*(\d{1,3}[A-F]?)\b', text, re.I)
        if seat_match:
            seat = seat_match.group(1)
            parts.append(f"Tu asiento es el {seat}.")

        # Puerta
        gate_match = re.search(r'\b(?:puerta|gate)\s*:?\s*([A-Z]?\d{1,3})\b', text, re.I)
        if gate_match:
            parts.append(f"Puerta de embarque: {gate_match.group(1)}.")

        # Pasajero
        pax_match = re.search(r'\b(?:pasajero|passenger|nombre|name)\s*:?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-Za-záéíóúñ]+){0,3})', text, re.I)
        if pax_match:
            parts.append(f"A nombre de {pax_match.group(1)}.")

        # PNR / Localizador
        pnr_match = re.search(r'\b(?:PNR|localizador|confirmaci[oó]n|booking)\s*:?\s*([A-Z0-9]{4,8})\b', text, re.I)
        if pnr_match:
            parts.append(f"Código de reserva: {pnr_match.group(1)}.")

        # Evento específico
        if is_event:
            fila_match = re.search(r'\b(?:fila|row)\s*:?\s*(\d+)\b', text, re.I)
            sec_match = re.search(r'\b(?:secci[oó]n|section|zona)\s*:?\s*([A-Za-z0-9]+)\b', text, re.I)
            if fila_match:
                parts.append(f"Fila {fila_match.group(1)}.")
            if sec_match:
                parts.append(f"Sección {sec_match.group(1)}.")

        if mode != "resumen" and is_flight:
            parts.append("Recuerda llegar al aeropuerto con anticipación y tener tu documento de identidad a la mano.")

        return " ".join(parts)

    # --- IDENTIFICACIÓN ---

    def _gen_identificacion(self, mode: str, text: str, ex: ExtractedData,
                            caption: Optional[str]) -> str:
        """Narrativa natural para documentos de identidad y credenciales."""
        if not text or not text.strip():
            return "Es un documento de identificación, pero no pude leer los datos."

        parts = []

        # Detectar credencial de empleado primero (prioridad sobre otros tipos)
        is_credencial = bool(re.search(
            r'\b(?:credencial|carnet|gafete|badge|empleado|trabajador|funcionario|colaborador|cargo\s*:|puesto\s*:|departamento\s*:)\b',
            text, re.I))

        if is_credencial:
            # Nombre de empresa
            empresa_match = re.search(
                r'\b(?:empresa|compa[ñn][ií]a|organizaci[oó]n|corporaci[oó]n|instituci[oó]n)\s*:?\s*([^\n]{2,40})',
                text, re.I)
            # También buscar empresa en las primeras líneas (suele estar arriba)
            first_lines = text.strip().split('\n')[:3]
            empresa_candidate = None
            for line in first_lines:
                line = line.strip()
                if len(line) > 3 and not re.search(r'\b(?:nombre|cargo|departamento|empleado|credencial|válido|fecha)\b', line, re.I):
                    empresa_candidate = line
                    break

            if empresa_match:
                parts.append(f"Es una credencial de empleado de {_cf(empresa_match.group(1).strip())}.")
            elif empresa_candidate:
                parts.append(f"Es una credencial de empleado de {_cf(empresa_candidate)}.")
            else:
                parts.append("Es una credencial de empleado.")

            # Nombre del titular
            name_match = re.search(
                r'\b(?:nombre|nombres?|name)\s*:?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-Za-záéíóúñ]+){0,4})',
                text, re.I)
            apellido_match = re.search(
                r'\b(?:apellido|apellidos?|surname)\s*:?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-Za-záéíóúñ]+){0,2})',
                text, re.I)
            if name_match and apellido_match:
                parts.append(f"Pertenece a {name_match.group(1)} {apellido_match.group(1)}.")
            elif name_match:
                parts.append(f"Pertenece a {name_match.group(1)}.")

            # Cargo / puesto
            cargo_match = re.search(r'\b(?:cargo|puesto|posici[oó]n)\s*:?\s*([^\n]{2,40})', text, re.I)
            if cargo_match:
                parts.append(f"Cargo: {_cf(cargo_match.group(1).strip())}.")

            # Departamento / área
            depto_match = re.search(r'\b(?:departamento|[aá]rea|secci[oó]n|divisi[oó]n)\s*:?\s*([^\n]{2,40})', text, re.I)
            if depto_match:
                parts.append(f"Departamento: {_cf(depto_match.group(1).strip())}.")

            # Número de empleado / código
            cod_match = re.search(
                r'\b(?:c[oó]digo|n[úu]mero|no\.?|ficha|legajo|ID)\s*(?:de\s+)?(?:emplead[oa])?\s*:?\s*([\w\d\-]{3,15})\b',
                text, re.I)
            if cod_match:
                parts.append(f"Código de empleado: {cod_match.group(1)}.")
            elif ex.ids:
                parts.append(f"Código: {ex.ids[0]}.")

            # Vigencia
            vig_match = re.search(r'\b(?:v[aá]lido\s+hasta|vigencia|vencimiento|expira|exp\.?)\s*:?\s*([^\n]{4,15})', text, re.I)
            if vig_match:
                parts.append(f"Válida hasta: {_cf(vig_match.group(1).strip())}.")

            if mode != "resumen":
                parts.append("No olvides portar siempre tu credencial dentro de las instalaciones.")

            return " ".join(parts)

        # --- Documentos de identidad personal ---
        is_cedula = bool(re.search(r'\bc[eé]dula\b', text, re.I))
        is_passport = bool(re.search(r'\bpasaporte|passport\b', text, re.I))
        is_license = bool(re.search(r'\blicencia\s+de\s+(?:conducir|conducci[oó]n)\b', text, re.I))
        is_dni = bool(re.search(r'\bDNI\b', text, re.I))

        if is_cedula:
            parts.append("Es una cédula de ciudadanía.")
        elif is_passport:
            parts.append("Es un pasaporte.")
        elif is_license:
            parts.append("Es una licencia de conducir.")
        elif is_dni:
            parts.append("Es un documento nacional de identidad.")
        else:
            parts.append("Es un documento de identificación.")

        # Nombre
        name_match = re.search(
            r'\b(?:nombre|nombres?|name)\s*:?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-Za-záéíóúñ]+){0,4})',
            text, re.I)
        apellido_match = re.search(
            r'\b(?:apellido|apellidos?|surname|last\s+name)\s*:?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-Za-záéíóúñ]+){0,2})',
            text, re.I)
        if name_match and apellido_match:
            parts.append(f"A nombre de {name_match.group(1)} {apellido_match.group(1)}.")
        elif name_match:
            parts.append(f"A nombre de {name_match.group(1)}.")

        # Número de documento
        num_match = re.search(r'\b(?:n[úu]mero|no\.?|num\.?|#)\s*:?\s*([\d\.\-]{5,15})\b', text, re.I)
        if num_match:
            parts.append(f"Número de documento: {num_match.group(1)}.")
        elif ex.ids:
            parts.append(f"Número: {ex.ids[0]}.")

        # Fecha de nacimiento
        nac_match = re.search(
            r'\b(?:fecha\s+de\s+nacimiento|nacimiento|f\.?\s*(?:de\s+)?nac\.?|date\s+of\s+birth)\s*:?\s*([^\n]{5,20})',
            text, re.I)
        if nac_match:
            parts.append(f"Fecha de nacimiento: {_cf(nac_match.group(1))}.")

        # Nacionalidad
        nac_match2 = re.search(r'\b(?:nacionalidad|nationality)\s*:?\s*([A-Za-záéíóúñ]+)', text, re.I)
        if nac_match2:
            parts.append(f"Nacionalidad: {nac_match2.group(1)}.")

        # Sexo
        sex_match = re.search(r'\b(?:sexo|sex|g[eé]nero)\s*:?\s*(M|F|masculino|femenino|male|female)\b', text, re.I)
        if sex_match:
            val = sex_match.group(1).upper()
            sexo = "Masculino" if val in ('M', 'MASCULINO', 'MALE') else "Femenino"
            parts.append(f"Sexo: {sexo}.")

        # Tipo de sangre
        blood_match = re.search(r'\b(?:tipo\s+de\s+sangre|RH|grupo\s+sangu[ií]neo)\s*:?\s*([ABO]+[\+\-]?)\b', text, re.I)
        if blood_match:
            parts.append(f"Tipo de sangre: {blood_match.group(1)}.")

        # Vigencia
        vig_match = re.search(r'\b(?:v[aá]lido\s+hasta|vigencia|vencimiento|expira)\s*:?\s*([^\n]{4,15})', text, re.I)
        if vig_match:
            parts.append(f"Vigente hasta: {_cf(vig_match.group(1))}.")

        if mode != "resumen":
            parts.append("Protege siempre tu información personal. No compartas fotos de tu documento con desconocidos.")

        return " ".join(parts)

    # --- HORARIO ---

    def _gen_horario(self, mode: str, text: str, ex: ExtractedData,
                     caption: Optional[str]) -> str:
        """Narrativa natural para horarios de clase, trabajo o citas.

        Maneja dos formatos:
        1. Tipo lista: "Lunes 7:00-8:30 Matemáticas Aula 301"
        2. Tipo tabla/grid: "LUNES MARTES ... ASEO ASEO ... TAREAS TAREAS"
        """
        if not text or not text.strip():
            return "Es un horario, pero no pude leer el contenido."

        src = text
        parts = []

        # Detectar tipo de horario
        is_class = bool(re.search(r'\b(?:clase|materia|asignatura|curso|profesor|aula)\b', src, re.I))
        is_work = bool(re.search(r'\b(?:turno|jornada|oficina|trabajo)\b', src, re.I))

        if is_class:
            parts.append("Es tu horario de clases.")
        elif is_work:
            parts.append("Es tu horario de trabajo.")
        else:
            parts.append("Tienes un horario semanal.")

        # --- Extraer días presentes ---
        _DAY_MAP = [
            ('lunes', 'lunes'), ('martes', 'martes'),
            ('mi[eé]rcoles', 'miércoles'), ('jueves', 'jueves'),
            ('viernes', 'viernes'), ('s[aá]bado', 'sábado'),
            ('domingo', 'domingo'),
        ]
        days_found = []
        for pat, name in _DAY_MAP:
            if re.search(rf'\b{pat}\b', src, re.I):
                days_found.append(name)

        # --- Estrategia 1: Rangos de hora (formato lista) ---
        time_ranges = re.findall(r'(\d{1,2}:\d{2})\s*[-–a]\s*(\d{1,2}:\d{2})', src)
        activities_after_time = re.findall(
            r'\d{1,2}:\d{2}\s*[-–a]\s*\d{1,2}:\d{2}\s+'
            r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-Za-záéíóúñ]+){0,3})',
            src)

        if time_ranges and activities_after_time:
            # Formato lista con horas → leer actividades con su hora
            n_days = len(days_found)
            if n_days > 0:
                parts.append(f"Abarca {n_days} días: {', '.join(days_found)}.")

            count = min(len(time_ranges), len(activities_after_time))
            limit = 4 if mode == "resumen" else 8
            for i in range(min(count, limit)):
                start, end = time_ranges[i]
                act = activities_after_time[i]
                parts.append(f"De {start} a {end}, {act}.")

            # Aulas
            aulas = re.findall(r'\b(?:aula|sal[oó]n|lab)\s*:?\s*([A-Za-z0-9\-]+)', src, re.I)
            if aulas and mode != "resumen":
                parts.append(f"En las aulas: {', '.join(aulas[:3])}.")

            return " ".join(parts)

        # --- Estrategia 2: Formato tabla/grid (días como encabezados) ---
        # Extraer actividades únicas (palabras repetidas = actividades en la grid)
        # Excluir días, números, y basura
        _SKIP_WORDS = frozenset({
            'HORARIO', 'HORA', 'HORAS', 'LUNES', 'MARTES', 'MIERCOLES',
            'MIÉRCOLES', 'JUEVES', 'VIERNES', 'SABADO', 'SÁBADO', 'DOMINGO',
            'AM', 'PM', 'SCHEDULE',
        })
        words = re.findall(r'\b([A-ZÁÉÍÓÚÑ]{3,})\b', src)
        # Contar frecuencia de cada palabra (en grid, actividades se repiten por día)
        from collections import Counter
        word_freq = Counter(w.upper() for w in words if w.upper() not in _SKIP_WORDS)

        # Actividades = palabras que aparecen >= 2 veces (se repiten por día)
        activities_grid = [w.title() for w, count in word_freq.most_common(15)
                          if count >= 2 and len(w) >= 3]

        # Capturar pares de palabras consecutivas DIFERENTES:
        # "TAREAS EN" (ok), pero NO "ASEO ASEO" (duplicado)
        two_word = re.findall(r'\b([A-ZÁÉÍÓÚÑ]{3,})\s+([A-ZÁÉÍÓÚÑ]{2,})\b', src)
        two_word_freq = Counter(
            f"{a} {b}".title() for a, b in two_word
            if a.upper() not in _SKIP_WORDS
            and b.upper() not in _SKIP_WORDS
            and a.upper() != b.upper()  # NO duplicados como "ASEO ASEO"
        )
        compound_activities = [w for w, c in two_word_freq.most_common(10) if c >= 2]

        # Combinar: preferir compound si existe
        final_activities = []
        used_singles = set()
        for comp in compound_activities:
            final_activities.append(comp)
            for part in comp.split():
                used_singles.add(part.title())
        for act in activities_grid:
            if act not in used_singles:
                final_activities.append(act)

        # Quitar duplicados y limitar
        seen = set()
        unique_activities = []
        for a in final_activities:
            al = a.lower()
            if al not in seen:
                seen.add(al)
                unique_activities.append(a)

        n_days = len(days_found)
        if n_days > 0:
            parts.append(f"Va de {days_found[0]} a {days_found[-1]}.")

        if unique_activities:
            act_list = ", ".join(unique_activities[:6])
            parts.append(f"Las actividades incluyen: {act_list}.")
            if len(unique_activities) > 6:
                parts.append(f"Y {len(unique_activities) - 6} actividades más.")
        else:
            # No se pudieron extraer actividades
            parts.append("No pude identificar las actividades con claridad.")

        # Horas sueltas (sin rango, ej: "7", "8", "10")
        horas = re.findall(r'\b(\d{1,2})\s*(?:am|pm|:00|hrs?)?\b', src, re.I)
        horas_validas = [h for h in horas if 5 <= int(h) <= 23]
        if horas_validas and mode != "resumen":
            parts.append(f"Los bloques empiezan alrededor de las {horas_validas[0]}.")

        return " ".join(parts)

    # --- INSTRUCCIONES ---

    def _gen_instrucciones(self, mode: str, text: str, ex: ExtractedData,
                           caption: Optional[str]) -> str:
        """Narrativa natural para instrucciones paso a paso."""
        if not text or not text.strip():
            return "Son instrucciones, pero no pude leer el contenido."

        parts = ["Estas son instrucciones paso a paso."]

        # Intentar extraer pasos numerados
        steps = re.findall(
            r'\b(?:paso|step)\s*(\d+)\s*:?\s*(.{5,80}?)(?=\b(?:paso|step)\s*\d|\Z)',
            text, re.I | re.S)

        if steps:
            for num, content in steps[:8]:
                clean_step = _cf(content.strip())
                parts.append(f"Paso {num}: {clean_step}.")
        else:
            # Buscar verbos imperativos como inicio de frase (pasos implícitos)
            imp_pattern = re.compile(
                r'\b((?:aplica|mezcla|agita|vierte|remueve|enjuaga|coloca|retira|'
                r'deja|espera|conecta|presiona|abre|cierra|hornea|hierve|corta|'
                r'aplique|mezcle|agite|vierta|coloque|retire|abra|cierre|'
                r'limpie|seque|lave)\b[^.!?\n]{3,60})',
                re.I)
            imp_steps = imp_pattern.findall(text)

            if imp_steps:
                step_num = 1
                for step_text in imp_steps[:8]:
                    clean_step = _cf(step_text.strip())
                    parts.append(f"Paso {step_num}: {clean_step}.")
                    step_num += 1
            else:
                body = self._extract_body_preview(text, max_words=60)
                if body:
                    parts.append(body + ".")

        # Advertencias
        warn_match = re.search(
            r'\b(?:precauci[oó]n|advertencia|importante|warning|cuidado|peligro)\s*:?\s*(.{5,80})',
            text, re.I)
        if warn_match:
            parts.append(f"Importante: {_cf(warn_match.group(1))}.")

        # Medidas
        measures = re.findall(r'\b(\d+)\s*(ml|g|oz|litros?|cucharadas?|tazas?|gotas?|minutos?)\b', text, re.I)
        if measures and mode != "resumen":
            meas_strs = [f"{qty} {unit}" for qty, unit in measures[:4]]
            parts.append("Medidas mencionadas: " + ", ".join(meas_strs) + ".")

        return " ".join(parts)

    # --- RESULTADO DE LABORATORIO ---

    def _gen_resultado_lab(self, mode: str, text: str, ex: ExtractedData,
                           caption: Optional[str]) -> str:
        """Narrativa natural para resultados de laboratorio clínico.

        Prioriza: valores fuera de rango → normales → sin rango.
        No lista rangos numéricos, solo dice si es normal/alto/bajo.
        """
        if not text or not text.strip():
            return "Son resultados de laboratorio, pero no pude leer los valores."

        src = text
        parts = []

        # Paciente (evitar capturar palabras clave como "Orden", "Doctor")
        _LAB_SKIP = {'Orden', 'Doctor', 'Médico', 'Medico', 'Fecha', 'Reporte',
                     'Laboratorio', 'Resultado', 'Referencia', 'Unidad'}
        pac_match = re.search(
            r'\b(?:paciente|patient|nombre)\s*:?\s*'
            r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-Za-záéíóúñ]+){1,3})',
            src, re.I)

        paciente = None
        if pac_match:
            candidate = pac_match.group(1).strip()
            if candidate.split()[0] not in _LAB_SKIP:
                paciente = candidate

        if paciente:
            parts.append(f"Tienes los resultados de laboratorio de {paciente}.")
        else:
            parts.append("Tienes unos resultados de laboratorio.")

        if ex.dates:
            parts.append(f"Del {ex.dates[0]}.")

        # Laboratorio / Orden
        lab_match = re.search(r'\b(?:laboratorio|lab)\s*:?\s*([A-Za-záéíóúñ\s]{3,30})', src, re.I)
        orden_match = re.search(r'\b(?:orden)\s*:?\s*(\d+)', src, re.I)

        # Extraer valores: nombre + valor + unidad
        lab_pattern = re.compile(
            r'\b(hemoglobina|hematocrito|plaquetas|leucocitos|eritrocitos|'
            r'gl[oó]bulos?\s*(?:blancos?|rojos?)|'
            r'neutr[oó]filos|linfocitos|monocitos|eosin[oó]filos|bas[oó]filos|'
            r'glucosa|glicemia|colesterol|triglic[eé]ridos|'
            r'creatinina|urea|[aá]cido\s+[uú]rico|bilirrubina|'
            r'ALT|AST|TGO|TGP|TSH|T3|T4|'
            r'HDL|LDL|VLDL|PSA|HbA1c|'
            r'hemoglobina\s+glicosilada|'
            r'VCM|HCM|CHCM|RDW|VSG|PCR)\s*:?\s*'
            r'([\d,.]+)\s*'
            r'(mg/dL|g/dL|%|mmol/L|UI/L|mL|mm3|cel/uL|mEq/L|ng/mL|U/L|fL|pg|pg/mL|x10\^?\d)?',
            re.I)

        # Rango de referencia: "Ref: 12.0-16.0" o "12.0 - 16.0" o "400-540"
        ref_pattern = re.compile(
            r'(?:ref(?:erencia)?|rango|normal|valores?\s+(?:de\s+)?ref)\s*:?\s*'
            r'([\d,.]+)\s*[-–a]\s*([\d,.]+)',
            re.I)

        results = []
        for m in lab_pattern.finditer(src):
            test_name = m.group(1).strip()
            value_str = m.group(2).strip().replace(',', '.')
            unit = m.group(3) if m.group(3) else ""

            try:
                value = float(value_str)
            except ValueError:
                continue

            # Buscar rango de referencia DESPUÉS de este match
            after_text = src[m.end():m.end()+80]
            status = "sin rango"
            ref_m = ref_pattern.search(after_text)
            if ref_m:
                try:
                    low = float(ref_m.group(1).replace(',', '.'))
                    high = float(ref_m.group(2).replace(',', '.'))
                    if value < low:
                        status = "bajo"
                    elif value > high:
                        status = "alto"
                    else:
                        status = "normal"
                except ValueError:
                    pass

            results.append((test_name.capitalize(), value_str, unit, status))

        if results:
            # Separar anormales vs normales
            abnormal = [(n, v, u, s) for n, v, u, s in results if s in ("alto", "bajo")]
            normal = [(n, v, u, s) for n, v, u, s in results if s == "normal"]
            unknown = [(n, v, u, s) for n, v, u, s in results if s == "sin rango"]

            total = len(results)
            if abnormal:
                parts.append(f"Atención, hay {len(abnormal)} valor{'es' if len(abnormal) > 1 else ''} fuera de lo normal.")
                for name, val, unit, status in abnormal:
                    u = f" {unit}" if unit else ""
                    parts.append(f"{name}: {val}{u}, está {status}.")

            if normal:
                if len(normal) <= 3:
                    for name, val, unit, _ in normal:
                        u = f" {unit}" if unit else ""
                        parts.append(f"{name}: {val}{u}, normal.")
                else:
                    nombres = ", ".join(n for n, _, _, _ in normal[:5])
                    parts.append(f"Los demás valores están normales: {nombres}.")

            if unknown and mode != "resumen":
                for name, val, unit, _ in unknown[:3]:
                    u = f" {unit}" if unit else ""
                    parts.append(f"{name}: {val}{u}.")
        else:
            # No se pudieron parsear valores, dar resumen general
            # Contar cuántos números hay (probable que haya datos)
            nums = re.findall(r'\b\d+[.,]?\d*\b', src)
            if len(nums) > 5:
                parts.append(f"Veo {len(nums)} valores numéricos pero no pude identificar los nombres con claridad.")
                parts.append("Intenta con mejor iluminación o acercando más la cámara.")
            else:
                parts.append("No pude leer los valores con claridad.")

        parts.append("Consulta con tu médico para interpretar estos resultados.")

        return " ".join(parts)

    # --- TABLA NUTRICIONAL ---

    def _gen_tabla_nutricional(self, mode: str, text: str, ex: ExtractedData,
                               caption: Optional[str]) -> str:
        """Narrativa natural para tablas nutricionales."""
        if not text or not text.strip():
            return "Es una tabla nutricional, pero no pude leer los valores."

        parts = ["Esta es la información nutricional del producto."]

        # Porción
        porc_match = re.search(
            r'\b(?:porci[oó]n|serving\s+size|tama[ñn]o\s+de\s+(?:la\s+)?porci[oó]n)\s*:?\s*([\d,.]+\s*(?:g|ml|oz|unidad(?:es)?)?)',
            text, re.I)
        if porc_match:
            parts.append(f"Tamaño de porción: {porc_match.group(1)}.")

        # Calorías
        cal_match = re.search(r'\b(?:calor[ií]as|calories|energ[ií]a)\s*:?\s*([\d,.]+)\s*(?:kcal|cal|kJ)?', text, re.I)
        if cal_match:
            parts.append(f"Calorías por porción: {cal_match.group(1)}.")

        # Macronutrientes principales
        nutrients = [
            (r'\b(?:grasa(?:s)?\s+total(?:es)?|total\s+fat)\s*:?\s*([\d,.]+)\s*g?', 'Grasas totales'),
            (r'\b(?:grasa(?:s)?\s+saturada(?:s)?|saturated)\s*:?\s*([\d,.]+)\s*g?', 'Grasas saturadas'),
            (r'\b(?:sodio|sodium)\s*:?\s*([\d,.]+)\s*mg?', 'Sodio'),
            (r'\b(?:carbohidratos?\s+totales?|total\s+carb)\s*:?\s*([\d,.]+)\s*g?', 'Carbohidratos'),
            (r'\b(?:fibra|fiber)\s*:?\s*([\d,.]+)\s*g?', 'Fibra'),
            (r'\b(?:az[úu]cares?\s+totales?|total\s+sugars?)\s*:?\s*([\d,.]+)\s*g?', 'Azúcares'),
            (r'\b(?:prote[ií]nas?|protein)\s*:?\s*([\d,.]+)\s*g?', 'Proteínas'),
            (r'\b(?:colesterol|cholesterol)\s*:?\s*([\d,.]+)\s*mg?', 'Colesterol'),
        ]

        found_nutrients = []
        for pattern, name in nutrients:
            m = re.search(pattern, text, re.I)
            if m:
                found_nutrients.append((name, m.group(1)))

        if found_nutrients:
            for name, value in found_nutrients[:6 if mode != "resumen" else 4]:
                unit = "mg" if name in ('Sodio', 'Colesterol') else "g"
                parts.append(f"{name}: {value} {unit}.")
        else:
            body = self._extract_body_preview(text, max_words=40)
            if body:
                parts.append(body + ".")

        return " ".join(parts)

    # --- CALENDARIO ---

    def _gen_calendario(self, mode: str, text: str, ex: ExtractedData,
                        caption: Optional[str]) -> str:
        """Narrativa natural para calendarios."""
        if not text or not text.strip():
            return "Es un calendario, pero no pude leer el contenido."

        parts = []

        # Detectar app
        app_match = re.search(r'\b(Google\s+Calendar|Outlook|Apple\s+Calendar|Samsung\s+Calendar)\b', text, re.I)
        if app_match:
            parts.append(f"Es tu calendario de {app_match.group(1)}.")
        else:
            parts.append("Es un calendario.")

        # Mes y año
        months_es = {
            'enero': 'enero', 'febrero': 'febrero', 'marzo': 'marzo',
            'abril': 'abril', 'mayo': 'mayo', 'junio': 'junio',
            'julio': 'julio', 'agosto': 'agosto', 'septiembre': 'septiembre',
            'octubre': 'octubre', 'noviembre': 'noviembre', 'diciembre': 'diciembre',
            'january': 'enero', 'february': 'febrero', 'march': 'marzo',
            'april': 'abril', 'may': 'mayo', 'june': 'junio',
            'july': 'julio', 'august': 'agosto', 'september': 'septiembre',
            'october': 'octubre', 'november': 'noviembre', 'december': 'diciembre',
        }
        month_found = None
        for pat, name in months_es.items():
            if re.search(rf'\b{pat}\b', text, re.I):
                month_found = name
                break

        year_match = re.search(r'\b(20\d{2})\b', text)
        if month_found and year_match:
            parts.append(f"Estás viendo {month_found} de {year_match.group(1)}.")
        elif month_found:
            parts.append(f"Estás viendo el mes de {month_found}.")

        # Eventos
        events = re.findall(
            r'\b(?:evento|cita|reuni[oó]n|meeting|appointment|cumplea[ñn]os|birthday)\s*:?\s*(.{3,50})',
            text, re.I)
        if events:
            parts.append("Eventos encontrados:")
            for ev in events[:4]:
                parts.append(f"{_cf(ev.strip())}.")

        # Hoy / día señalado
        if re.search(r'\b(?:hoy|today)\b', text, re.I):
            parts.append("El día de hoy está marcado.")

        # Feriados
        if re.search(r'\b(?:feriado|festivo|holiday)\b', text, re.I):
            parts.append("Hay días festivos marcados en el calendario.")

        # Vista
        view_match = re.search(r'\b(?:vista\s+(mensual|semanal|diaria|anual))\b', text, re.I)
        if view_match:
            parts.append(f"Está en vista {view_match.group(1)}.")

        if not events and mode != "resumen":
            body = self._extract_body_preview(text, max_words=30)
            if body:
                parts.append(body + ".")

        return " ".join(parts)

    # --- MAPA / NAVEGACIÓN ---

    def _gen_mapa(self, mode: str, text: str, ex: ExtractedData,
                  caption: Optional[str]) -> str:
        """Narrativa natural para mapas y navegación."""
        if not text or not text.strip():
            if caption:
                return f"Es un mapa. La imagen muestra: {caption}."
            return "Es un mapa, pero no pude leer los detalles."

        parts = []

        # App
        app_match = re.search(r'\b(Google\s+Maps|Waze|Apple\s+Maps|Moovit|Citymapper)\b', text, re.I)
        if app_match:
            parts.append(f"Estás usando {app_match.group(1)}.")
        else:
            parts.append("Es un mapa de navegación.")

        # Destino
        dest_match = re.search(
            r'\b(?:hacia|to|destino|destination|ir\s+a|go\s+to|navegar?\s+(?:a|hacia))\s*:?\s*(.{3,40})',
            text, re.I)
        if dest_match:
            parts.append(f"Destino: {_cf(dest_match.group(1))}.")

        # Tiempo estimado
        time_match = re.search(r'(\d+)\s*(?:min(?:utos?)?)\b', text, re.I)
        if time_match:
            mins = int(time_match.group(1))
            if mins >= 60:
                hrs = mins // 60
                rem = mins % 60
                if rem > 0:
                    parts.append(f"Tiempo estimado: {hrs} hora{'s' if hrs > 1 else ''} y {rem} minutos.")
                else:
                    parts.append(f"Tiempo estimado: {hrs} hora{'s' if hrs > 1 else ''}.")
            else:
                parts.append(f"Tiempo estimado: {mins} minutos.")

        # Distancia
        dist_match = re.search(r'(\d+[\.,]?\d*)\s*(km|mi|metros|m)\b', text, re.I)
        if dist_match:
            parts.append(f"Distancia: {dist_match.group(1)} {dist_match.group(2)}.")

        # Tráfico
        if re.search(r'\b(?:tr[aá]fico\s+(?:pesado|denso|congestionado)|congesti[oó]n)\b', text, re.I):
            parts.append("Hay tráfico pesado en la ruta.")
        elif re.search(r'\b(?:tr[aá]fico\s+(?:moderado|regular))\b', text, re.I):
            parts.append("El tráfico está moderado.")
        elif re.search(r'\b(?:tr[aá]fico\s+(?:fluido|libre)|sin\s+tr[aá]fico)\b', text, re.I):
            parts.append("El tráfico está fluido, buen momento para salir.")

        # Modo de transporte
        if re.search(r'\b(?:en\s+auto|driving|en\s+carro|en\s+coche)\b', text, re.I):
            parts.append("Ruta en auto.")
        elif re.search(r'\b(?:a\s+pie|walking|caminando)\b', text, re.I):
            parts.append("Ruta a pie.")
        elif re.search(r'\b(?:en\s+bici|cycling|bicicleta)\b', text, re.I):
            parts.append("Ruta en bicicleta.")
        elif re.search(r'\b(?:transporte\s+p[uú]blico|transit|bus|metro)\b', text, re.I):
            parts.append("Ruta en transporte público.")

        # Indicaciones de giro
        turn_match = re.search(r'\b(?:gir[ae]\s+a\s+la\s+(derecha|izquierda)|turn\s+(left|right))\b', text, re.I)
        if turn_match:
            direction = turn_match.group(1) or turn_match.group(2)
            parts.append(f"Próxima indicación: gira a la {direction}.")

        # Dirección/calle
        street_match = re.search(r'\b(?:Av\.?|Avenida|Calle|Carrera|Cra\.?|Cl\.?|Boulevard|Autopista)\s+[^\n]{3,30}', text, re.I)
        if street_match and mode != "resumen":
            parts.append(f"Sobre: {_cf(street_match.group(0))}.")

        # Caption visual como complemento
        if caption and mode != "resumen":
            parts.append(f"En la imagen se ve: {caption}.")

        return " ".join(parts)

    # --- GENÉRICO INTERFAZ DE APP (para subtipos nuevos sin generador propio) ---

    def _gen_interfaz_app(self, mode: str, text: str, ex: ExtractedData,
                          caption: Optional[str],
                          label: str = "una pantalla de aplicación") -> str:
        """Generador genérico para interfaces de app sin generador especializado.

        Usa el label del subtipo para dar contexto, y lee el contenido
        de forma inteligente según el reading_mode.
        """
        if not text or not text.strip():
            if caption:
                return f"Es {label}. La imagen muestra: {caption}."
            return f"Es {label}, pero no se pudo leer el contenido."

        parts = [f"Es {label}."]

        if mode == "resumen":
            if ex.amounts:
                parts.append(f"Monto: {ex.amounts[0]}.")
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            preview = self._extract_body_preview(text, max_words=25)
            if preview:
                parts.append(f"Dice: {preview}.")
            return " ".join(parts)

        if mode == "financiero":
            if ex.totals:
                for t in ex.totals[:3]:
                    parts.append(f"{self._clean_total(t)}.")
            elif ex.amounts:
                parts.append("Montos: " + ", ".join(ex.amounts[:5]) + ".")
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            return " ".join(parts)

        # Detallado
        if ex.headers:
            parts.append(f"Título: {ex.headers[0]}.")
        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")
        if ex.amounts:
            parts.append("Montos: " + ", ".join(ex.amounts[:5]) + ".")
        if ex.emails:
            parts.append(f"Correo: {ex.emails[0]}.")
        if ex.phones:
            parts.append(f"Teléfono: {ex.phones[0]}.")
        body = self._extract_body_preview(text, max_words=60)
        if body:
            parts.append(f"Contenido: {body}.")
        return " ".join(parts)

    # Fábricas para crear generadores con label específico
    def _make_app_gen(self, label: str):
        """Crea un generador específico de interfaz de app con el label dado."""
        def gen(mode, text, ex, caption):
            return self._gen_interfaz_app(mode, text, ex, caption, label=label)
        return gen

    # --- DESCONOCIDO (fallback universal) ---

    def _gen_desconocido(self, mode: str, text: str, ex: ExtractedData,
                         caption: Optional[str]) -> str:
        if not text or not text.strip():
            if caption:
                return f"La imagen muestra: {caption}."
            return "No se detectó texto en la imagen."

        # Texto ya viene reconstruido del OCRTextReconstructor.
        # Verificar que hay contenido legible.
        preview_check = self._extract_body_preview(text, max_words=5)
        if not preview_check:
            if caption:
                return f"La imagen muestra: {caption}. No se pudo leer texto claramente."
            return "Se detectó algo de texto, pero no se pudo leer claramente. Intenta acercar más la cámara al texto."

        intro = "Se detectó un documento."

        if mode == "resumen":
            parts = [intro]
            if ex.dates:
                parts.append(f"Fecha: {ex.dates[0]}.")
            if ex.amounts:
                parts.append(f"Monto: {ex.amounts[0]}.")
            preview = self._extract_body_preview(text, max_words=30)
            if preview:
                parts.append(f"Dice: {preview}.")
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
                preview = self._extract_body_preview(text, max_words=30)
                if preview:
                    parts.append(f"Dice: {preview}.")
            return " ".join(parts)

        # detallado: leer contenido limpio
        parts = [intro]
        if ex.headers:
            parts.append(f"Título: {ex.headers[0]}.")
        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")
        if ex.amounts:
            parts.append("Montos: " + ", ".join(ex.amounts[:5]) + ".")
        if ex.emails:
            parts.append(f"Correo: {ex.emails[0]}.")
        if ex.phones:
            parts.append(f"Teléfono: {ex.phones[0]}.")
        # Leer el contenido principal (hasta 80 palabras)
        body = self._extract_body_preview(text, max_words=80)
        if body:
            parts.append(f"Dice: {body}.")
        return " ".join(parts)

    # --- HELPERS ---

    # Palabras cortas reales en español (no basura OCR)
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

    @staticmethod
    def _clean_ocr_words(text: str) -> str:
        """
        Limpia basura OCR del texto, preservando solo palabras reales.

        Elimina:
        - Tokens de 1-2 chars que no son palabras reales
        - Tokens que son solo símbolos/números sueltos
        - Tokens con >40% caracteres no-alfabéticos
        - Tokens que parecen artefactos de UI (corchetes, comillas sueltas)
        - Tokens en mayúsculas de 1-2 chars (basura de status bar)
        """
        if not text:
            return ""
        words = text.split()
        clean = []
        for w in words:
            # Limpiar puntuación y comillas envolventes
            stripped = w.strip('.,;:!?()[]{}"\'`')
            wl = stripped.lower()

            if not wl:
                continue

            # Preservar palabras cortas reales
            if wl in NarrativeGenerator._REAL_SHORT_WORDS:
                clean.append(stripped)
                continue

            # Eliminar tokens de 1-2 chars no reconocidos
            if len(wl) <= 2:
                continue

            # Eliminar tokens puramente simbólicos (no letras, no dígitos)
            # PERO preservar números que podrían ser métricas/cantidades
            # como "5", "20", "151", "3%", "5min"
            if re.match(r'^[\d\W]+$', wl):
                # Preservar si es un número puro y hay contexto (palabras antes/después)
                # Números sueltos entre palabras reales son métricas: "tiene 20 repositorios"
                if re.match(r'^\d+[%KkMm]?$', wl):
                    clean.append(stripped)
                continue

            # Eliminar tokens con >40% basura (no-letras)
            letter_count = sum(1 for c in wl if c.isalpha())
            if letter_count < len(wl) * 0.6:
                continue

            # Eliminar tokens cortos all-caps (basura UI: "GD", "ED", "Ss")
            if len(stripped) <= 3 and stripped.isupper():
                continue

            # Eliminar tokens de 3 chars que no parecen palabras reales
            if len(wl) == 3 and wl not in {
                'que', 'por', 'con', 'del', 'las', 'los', 'una', 'son',
                'fue', 'ser', 'hay', 'van', 'mas', 'más', 'sin', 'nos',
                'hoy', 'muy', 'día', 'dia', 'ver', 'dar', 'mal', 'vez',
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                'can', 'had', 'her', 'was', 'one', 'our', 'out', 'new',
                'fui', 'eso', 'esa', 'eso', 'voy', 'mis', 'tus', 'sus',
                'sol', 'mar', 'luz', 'pan', 'sal', 'red', 'fin', 'dos',
                'web', 'app', 'pdf', 'url',
            }:
                # Solo aceptar si tiene al menos una vocal (no es acronimo basura)
                if not re.search(r'[aeiouáéíóú]', wl):
                    continue

            clean.append(stripped)
        return " ".join(clean)

    @staticmethod
    def _extract_meaningful_phrases(text: str, max_phrases: int = 5,
                                     min_words: int = 3) -> List[str]:
        """
        Extrae frases significativas del texto OCR.

        En lugar de devolver todo el texto crudo, identifica segmentos
        que parecen frases reales (>= min_words palabras con letras).
        """
        if not text:
            return []

        # Limpiar primero
        clean = NarrativeGenerator._clean_ocr_words(text)
        if not clean:
            return []

        # Intentar segmentar por cambios mayúscula/minúscula
        segments = re.split(r'(?<=[a-záéíóúñ])\s+(?=[A-ZÁÉÍÓÚÑ][a-záéíóúñ])', clean)
        if len(segments) <= 1:
            # Segmentar por saltos de línea o doble espacio
            segments = re.split(r'\n+|\s{2,}', clean)

        phrases = []
        for seg in segments:
            seg = seg.strip()
            words = seg.split()
            # Contar palabras con letras reales
            real = [w for w in words if re.search(r'[a-záéíóúñ]', w, re.IGNORECASE)
                    and len(w) > 1]
            if len(real) >= min_words:
                phrases.append(seg)
                if len(phrases) >= max_phrases:
                    break

        return phrases

    @staticmethod
    def _fallback(raw_text: str) -> str:
        """Fallback universal: nunca dumpar texto OCR crudo."""
        if not raw_text or not raw_text.strip():
            return "No se pudo leer el documento."
        # Contar palabras para dar contexto
        wc = len(raw_text.split())
        if wc < 10:
            return "Se detectó poco texto. Intenta acercar más la cámara."
        return ("Se detectó un documento pero no pude interpretar bien el contenido. "
                "Intenta con mejor iluminación o acercando más la cámara.")

    @staticmethod
    def _extract_first_sentence(text: str, skip_lines: int = 0) -> str:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines[skip_lines:]:
            clean = NarrativeGenerator._clean_ocr_words(line)
            if len(clean.split()) >= 4:
                words = clean.split()[:20]
                return " ".join(words)
        return ""

    @staticmethod
    def _extract_body_preview(text: str, max_words: int = 40) -> str:
        clean = NarrativeGenerator._clean_ocr_words(text)
        if not clean:
            return ""
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        body_words: List[str] = []
        for line in lines:
            if len(line.split()) < 2:
                continue
            body_words.extend(line.split())
            if len(body_words) >= max_words:
                break
        return " ".join(body_words[:max_words])


# ============================================================================
# PROSODY ENHANCER
# ============================================================================

class ProsodyEnhancer:
    """
    Mejora texto para prosodia natural en Piper TTS.

    Normaliza:
    - Abreviaturas → texto expandido
    - Teléfonos → dígito por dígito
    - Fechas → "15 de marzo de 2026"
    - Montos → "ciento cincuenta dólares"
    - Emails → deletreo natural
    - Oraciones largas → pausas en conjunciones
    """

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

    # --- Patrones para normalización avanzada ---

    # Teléfonos: +58-212-555-1234, 0212-555-1234, (0212) 555-1234, etc.
    _PAT_PHONE_TTS = re.compile(
        r'(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{3,4}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}'
    )

    # Fechas numéricas: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
    _PAT_DATE_NUMERIC = re.compile(
        r'\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b'
    )

    # Montos: $150.00, Bs. 1.234,56, 150,00 dólares, etc.
    _PAT_AMOUNT_TTS = re.compile(
        r'([\$\€£])\s*([\d]{1,3}(?:[,\.\s]\d{3})*(?:[,\.]\d{1,2})?)'
        r'|(\d{1,3}(?:[,\.\s]\d{3})*(?:[,\.]\d{1,2})?)\s*'
        r'(bol[ií]vares|d[oó]lares|pesos(?:\s+colombianos|\s+mexicanos)?|euros)'
    )

    # Emails
    _PAT_EMAIL_TTS = re.compile(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    )

    # --- Tablas de conversión ---

    _MONTHS = {
        1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
        5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
        9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre',
    }

    _DIGIT_NAMES = {
        '0': 'cero', '1': 'uno', '2': 'dos', '3': 'tres', '4': 'cuatro',
        '5': 'cinco', '6': 'seis', '7': 'siete', '8': 'ocho', '9': 'nueve',
    }

    _CURRENCY_SYMBOL = {
        '$': 'pesos', '€': 'euros', '£': 'libras',
    }

    # Unidades, decenas, centenas para conversión a palabras
    _UNITS = ['', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve']
    _TEENS = [
        'diez', 'once', 'doce', 'trece', 'catorce', 'quince',
        'dieciséis', 'diecisiete', 'dieciocho', 'diecinueve',
    ]
    _TENS = ['', 'diez', 'veinte', 'treinta', 'cuarenta', 'cincuenta',
             'sesenta', 'setenta', 'ochenta', 'noventa']
    _HUNDREDS = [
        '', 'ciento', 'doscientos', 'trescientos', 'cuatrocientos',
        'quinientos', 'seiscientos', 'setecientos', 'ochocientos', 'novecientos',
    ]

    def enhance(self, text: str, doc_type: str) -> str:
        if not text:
            return text

        text = self._expand_abbreviations(text)
        text = self._normalize_phones(text)
        text = self._normalize_dates(text)
        text = self._normalize_amounts(text)
        text = self._normalize_emails(text)
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

    # ---- TELÉFONOS ----

    # Patrones que NO deben tratarse como teléfonos (IDs, RIF, NIT, etc.)
    _PAT_ID_CONTEXT = re.compile(
        r'(?:R\.?I\.?F\.?|N\.?I\.?T\.?|C\.?I\.?|[Cc][ée]dula|Erre-I-F|Ene-I-T)\s*[:\-]?\s*'
        r'[VEJGvejg]?\-?\s*\d[\d\-]{5,15}',
        re.IGNORECASE
    )

    def _normalize_phones(self, text: str) -> str:
        """Convierte teléfonos a lectura dígito por dígito con pausas.
        Evita tocar números que son IDs (RIF, NIT, cédula)."""

        # Primero, proteger IDs reemplazándolos temporalmente
        id_placeholders = {}
        protected = text
        for i, match in enumerate(self._PAT_ID_CONTEXT.finditer(text)):
            placeholder = f"__ID_PLACEHOLDER_{i}__"
            id_placeholders[placeholder] = match.group(0)
            protected = protected.replace(match.group(0), placeholder, 1)

        def phone_to_speech(match: re.Match) -> str:
            raw = match.group(0)
            # Verificar que no estamos dentro de un placeholder
            if '__ID_PLACEHOLDER_' in raw:
                return raw
            digits = re.sub(r'\D', '', raw)
            if len(digits) < 7 or len(digits) > 15:
                return raw  # No es teléfono

            # Agrupar dígitos en bloques para lectura natural
            named = [self._DIGIT_NAMES.get(d, d) for d in digits]

            groups = []
            i = 0
            while i < len(named):
                chunk_size = 4 if i == 0 and len(named) > 10 else 3 if len(named) - i > 4 else len(named) - i
                groups.append(' '.join(named[i:i + chunk_size]))
                i += chunk_size

            return ', '.join(groups)

        result = self._PAT_PHONE_TTS.sub(phone_to_speech, protected)

        # Restaurar IDs protegidos
        for placeholder, original in id_placeholders.items():
            result = result.replace(placeholder, original)

        return result

    # ---- FECHAS ----

    def _normalize_dates(self, text: str) -> str:
        """Convierte 15/03/2026 → 'quince de marzo de dos mil veintiséis'."""
        def date_to_speech(match: re.Match) -> str:
            day_s, month_s, year_s = match.group(1), match.group(2), match.group(3)
            try:
                day = int(day_s)
                month = int(month_s)
                year = int(year_s)

                # Año de 2 dígitos
                if year < 100:
                    year += 2000 if year < 50 else 1900

                month_name = self._MONTHS.get(month)
                if not month_name or day < 1 or day > 31:
                    return match.group(0)

                day_word = self._number_to_words(day)
                year_word = self._year_to_words(year)

                return f"{day_word} de {month_name} de {year_word}"
            except (ValueError, KeyError):
                return match.group(0)

        return self._PAT_DATE_NUMERIC.sub(date_to_speech, text)

    # ---- MONTOS ----

    def _normalize_amounts(self, text: str) -> str:
        """Convierte $150.00 → 'ciento cincuenta dólares'."""
        def amount_to_speech(match: re.Match) -> str:
            try:
                symbol = match.group(1)
                if symbol:
                    # Formato: $150.00
                    amount_str = match.group(2)
                    currency = self._CURRENCY_SYMBOL.get(symbol, symbol)
                else:
                    # Formato: 150,00 bolívares
                    amount_str = match.group(3)
                    currency = match.group(4)

                # Parsear número: manejar formato latino (1.234,56) y anglosajón (1,234.56)
                clean = amount_str.replace(' ', '')

                # Detectar formato: si el último separador es coma, es decimal latino
                if re.search(r',\d{1,2}$', clean):
                    # Latino: 1.234,56
                    integer_part = clean.rsplit(',', 1)[0].replace('.', '')
                    decimal_part = clean.rsplit(',', 1)[1] if ',' in clean else ''
                elif re.search(r'\.\d{1,2}$', clean):
                    # Anglosajón: 1,234.56
                    integer_part = clean.rsplit('.', 1)[0].replace(',', '')
                    decimal_part = clean.rsplit('.', 1)[1] if '.' in clean else ''
                else:
                    # Sin decimales
                    integer_part = clean.replace('.', '').replace(',', '')
                    decimal_part = ''

                num = int(integer_part) if integer_part else 0

                # Solo convertir montos razonables (hasta 999,999,999)
                if num > 999_999_999:
                    return match.group(0)

                words = self._number_to_words(num)

                if decimal_part and int(decimal_part) > 0:
                    dec_words = self._number_to_words(int(decimal_part))
                    words += f" con {dec_words}"

                return f"{words} {currency}"
            except (ValueError, IndexError, AttributeError):
                return match.group(0)

        return self._PAT_AMOUNT_TTS.sub(amount_to_speech, text)

    # ---- EMAILS ----

    def _normalize_emails(self, text: str) -> str:
        """Convierte emails a lectura natural: user@domain.com → 'user, arroba, domain, punto, com'."""
        def email_to_speech(match: re.Match) -> str:
            email = match.group(0)
            parts = email.split('@')
            if len(parts) != 2:
                return email

            local = parts[0]
            domain = parts[1]

            # Deletrear local part con separadores naturales
            local_spoken = local.replace('.', ', punto, ').replace('_', ', guion bajo, ').replace('-', ', guion, ')

            # Dominio: separar en partes
            domain_parts = domain.split('.')
            domain_spoken = ', punto, '.join(domain_parts)

            return f"{local_spoken}, arroba, {domain_spoken}"

        return self._PAT_EMAIL_TTS.sub(email_to_speech, text)

    # ---- CONVERSIÓN NÚMEROS A PALABRAS ----

    def _number_to_words(self, n: int) -> str:
        """Convierte un entero (0-999,999,999) a palabras en español."""
        if n == 0:
            return 'cero'
        if n == 100:
            return 'cien'
        if n < 0:
            return f'menos {self._number_to_words(-n)}'

        parts = []

        # Millones
        if n >= 1_000_000:
            millions = n // 1_000_000
            if millions == 1:
                parts.append('un millón')
            else:
                parts.append(f'{self._number_to_words(millions)} millones')
            n %= 1_000_000

        # Miles
        if n >= 1000:
            thousands = n // 1000
            if thousands == 1:
                parts.append('mil')
            else:
                parts.append(f'{self._number_to_words(thousands)} mil')
            n %= 1000

        # Centenas
        if n >= 100:
            parts.append(self._HUNDREDS[n // 100])
            n %= 100

        # Decenas y unidades
        if n >= 20:
            ten = n // 10
            unit = n % 10
            if unit == 0:
                parts.append(self._TENS[ten])
            elif ten == 2:
                # veinti-: veintiuno, veintidós, etc.
                special = ['veintiuno', 'veintidós', 'veintitrés', 'veinticuatro',
                           'veinticinco', 'veintiséis', 'veintisiete', 'veintiocho', 'veintinueve']
                parts.append(special[unit - 1])
            else:
                parts.append(f'{self._TENS[ten]} y {self._UNITS[unit]}')
        elif n >= 10:
            parts.append(self._TEENS[n - 10])
        elif n > 0:
            parts.append(self._UNITS[n])

        return ' '.join(parts)

    def _year_to_words(self, year: int) -> str:
        """Convierte año a palabras: 2026 → 'dos mil veintiséis'."""
        if 2000 <= year <= 2099:
            remainder = year - 2000
            if remainder == 0:
                return 'dos mil'
            return f'dos mil {self._number_to_words(remainder)}'
        if 1900 <= year <= 1999:
            remainder = year - 1900
            if remainder == 0:
                return 'mil novecientos'
            return f'mil novecientos {self._number_to_words(remainder)}'
        return self._number_to_words(year)

    # ---- UTILIDADES EXISTENTES (mejoradas) ----

    def _normalize_sentence_endings(self, text: str) -> str:
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

    Pipeline PRIMARIO (con internet):
      Imagen → Gemini Vision → narrativa + OCR + campos → ProsodyEnhancer → Response

    Pipeline FALLBACK (sin internet o si Gemini falla):
      Imagen → Tesseract OCR (con layout) → HierarchicalDocumentClassifier
             → StructureExtractor → NarrativeGenerator → ProsodyEnhancer → Response

    Clasificación jerárquica (v3):
      Pre-check: texto insuficiente → IMAGEN_VISUAL + visual_description
      Fase 1 (layout): macro-tipo (DOCUMENTO_FORMAL, INTERFAZ_DIGITAL, TEXTO_CONVERSACIONAL, IMAGEN_VISUAL)
      Fase 2 (keywords): subtipo (factura, chat, app_login, etc.) sobre texto limpio
      + scoring probabilístico 0-1 + gap analysis + reading_mode recomendado
      + estabilización temporal + detección de ambigüedad
    """

    def __init__(self):
        self._ocr_service = None
        self._gemini_service = None
        self._classifier_v2 = None       # Nuevo clasificador jerárquico
        self._classifier_legacy = DocumentClassifier()  # Para get_label() en Gemini path
        self._extractor = StructureExtractor()
        self._generator = NarrativeGenerator()
        self._enhancer = ProsodyEnhancer()
        self._optimizer = None
        self._captioning = None

    def _get_classifier(self):
        if self._classifier_v2 is None:
            from app.services.document_classifier import get_document_classifier
            self._classifier_v2 = get_document_classifier()
        return self._classifier_v2

    def _get_ocr(self):
        if self._ocr_service is None:
            from app.services.ocr_service import get_ocr_service
            self._ocr_service = get_ocr_service()
        return self._ocr_service

    def _get_optimizer(self):
        if self._optimizer is None:
            from app.services.reading_optimizer import get_reading_optimizer
            self._optimizer = get_reading_optimizer()
        return self._optimizer

    def _get_gemini(self):
        if self._gemini_service is None:
            try:
                from app.services.gemini_service import get_gemini_service
                self._gemini_service = get_gemini_service()
            except Exception as e:
                logger.warning(f"[SmartReading] Gemini no disponible: {e}")
        return self._gemini_service

    def _get_captioning(self):
        if self._captioning is None and settings.CAPTIONING_ENABLED:
            try:
                from app.services.captioning_service import get_captioning_service
                self._captioning = get_captioning_service()
            except Exception as e:
                logger.warning(f"Captioning no disponible: {e}")
        return self._captioning

    def _get_quality_analyzer(self):
        if not hasattr(self, '_quality_analyzer') or self._quality_analyzer is None:
            from app.utils.image_utils import get_image_quality_analyzer
            self._quality_analyzer = get_image_quality_analyzer()
        return self._quality_analyzer

    def _auto_select_reading_mode(self, doc_type: str, extracted, word_count: int,
                                   classifier_reading_mode: Optional[str] = None) -> str:
        """
        Selecciona automáticamente el modo de lectura óptimo según el tipo de documento.

        Mapea el reading_mode del clasificador v3 (dialogue, structured_fields,
        list_items, paragraph_text, visual_description) al modo de lectura
        del NarrativeGenerator (financiero, detallado, resumen).

        Si el clasificador provee reading_mode, se usa como base.
        Si no, se usa la lógica legacy por doc_type.
        """
        # --- Mapeo desde classifier reading_mode (v3) ---
        if classifier_reading_mode:
            _READING_MODE_MAP = {
                "structured_fields": "financiero",
                "paragraph_text": "detallado",
                "dialogue": "resumen",
                "list_items": "resumen",
                "visual_description": "resumen",
            }
            mapped = _READING_MODE_MAP.get(classifier_reading_mode)
            if mapped:
                return mapped

        # --- Fallback: lógica legacy por doc_type ---
        if doc_type in ("factura", "recibo", "app_service"):
            return "financiero"
        if doc_type in ("carta", "formulario", "documento_informativo", "correo",
                        "contrato", "hoja_de_vida", "informe"):
            return "detallado"
        if doc_type in ("etiqueta", "menu", "tarjeta", "chat", "notificacion",
                        "login", "red_social", "configuracion", "presentacion",
                        "app_menu", "app_settings", "app_login", "app_form",
                        "app_social", "comentario"):
            return "resumen"
        if doc_type == "noticia":
            return "detallado"
        if doc_type == "imagen_visual":
            return "resumen"
        if doc_type == "desconocido":
            if word_count < 30:
                return "resumen"
            elif extracted and (
                (hasattr(extracted, 'amounts') and len(extracted.amounts) > 0) or
                (hasattr(extracted, 'totals') and len(extracted.totals) > 0) or
                (isinstance(extracted, dict) and (len(extracted.get("amounts", [])) > 0 or len(extracted.get("totals", [])) > 0))
            ):
                return "financiero"
            else:
                return "detallado"
        return "detallado"

    def analyze(self, image: np.ndarray, reading_mode: Optional[str] = None) -> dict:
        """
        Pipeline completo de lectura inteligente.

        Estrategia:
        0. Primero intentar detectar códigos QR/barras (más rápido)
        1. Analizar calidad de imagen (siempre, local)
        2. Pipeline Tesseract local + Clasificador jerárquico v3

        Args:
            image: Imagen BGR (OpenCV)
            reading_mode: Opcional. Si no se especifica, se selecciona automáticamente.

        Returns:
            dict compatible con SmartReadingResponse
        """
        # 0. PRIMERO: Intentar detectar códigos QR/barras
        logger.info("[SmartReading] Intentando detectar códigos QR/barras...")
        barcode_reader = get_barcode_reader()
        barcode_result = barcode_reader.read(image)
        
        if barcode_result.has_codes:
            logger.info(f"[SmartReading] Código detectado: {barcode_result.summary}")
            return self._build_barcode_response(barcode_result, image)
        
        logger.info("[SmartReading] No se detectaron códigos QR/barras, usando OCR...")

        # 1. Analizar calidad de imagen ANTES de todo
        quality_analyzer = self._get_quality_analyzer()
        quality_report = quality_analyzer.analyze(image)

        # 2. Pipeline Tesseract local + Clasificador v3
        logger.info("[SmartReading] Usando Tesseract + Clasificador v3")
        return self._analyze_with_tesseract(image, quality_report, reading_mode)

    def _try_gemini(self, image: np.ndarray) -> Optional[dict]:
        """
        Intenta analizar con Gemini Vision.
        Retorna None si falla (sin internet, rate limit, etc.)
        """
        gemini = self._get_gemini()
        if gemini is None or not gemini.is_available:
            return None

        try:
            result = gemini.analyze_document(image)
            if result and result.get("narrative"):
                logger.info("[SmartReading] Gemini respondió correctamente")
                return result
            else:
                logger.info("[SmartReading] Gemini no devolvió narrativa, usando Tesseract")
                return None
        except Exception as e:
            logger.warning(f"[SmartReading] Gemini falló: {e}")
            return None

    def _build_response_from_gemini(
        self,
        gemini_result: dict,
        quality_report,
        reading_mode: Optional[str] = None,
    ) -> dict:
        """Construye la respuesta usando el resultado de Gemini."""
        doc_type = gemini_result.get("document_type", "desconocido")
        raw_text = gemini_result.get("raw_text", "")
        narrative = gemini_result.get("narrative", "")
        word_count = gemini_result.get("word_count", 0)
        confidence = gemini_result.get("confidence", 0)
        extracted_fields = gemini_result.get("extracted_fields", {})
        has_text = bool(raw_text and raw_text.strip())

        # Auto-select reading mode (para el label en la respuesta)
        if reading_mode is None:
            reading_mode = self._auto_select_reading_mode(
                doc_type, extracted_fields, word_count
            )

        # Optimizar lectura (jerarquía, incertidumbre, TTS)
        optimizer = self._get_optimizer()

        memory_msg = optimizer.check_memory(raw_text, doc_type, word_count, narrative)
        if memory_msg:
            logger.info(f"[SmartReading/Gemini/Memory] {memory_msg}")
            narrative = memory_msg
        else:
            narrative = optimizer.optimize(
                narrative=narrative,
                doc_type=doc_type,
                reading_mode=reading_mode,
                ocr_confidence=float(confidence),
                word_count=word_count,
            )

        # Aplicar ProsodyEnhancer a la narrativa
        # (normaliza teléfonos, fechas, montos, emails para TTS natural)
        narrative = self._enhancer.enhance(narrative, doc_type)

        logger.info(f"[SmartReading/Gemini] {doc_type}, {word_count} palabras, narrativa: {narrative[:150]}...")

        # Feedback de calidad solo si imagen ilegible (critical)
        if quality_report.feedback_text and not quality_report.is_acceptable:
            narrative = quality_report.feedback_text + " " + narrative

        quality_data = {
            "overall_score": quality_report.overall_score,
            "is_acceptable": quality_report.is_acceptable,
            "issues": [
                {"code": i.code, "severity": i.severity, "message": i.message_es}
                for i in quality_report.issues
            ],
            "feedback_text": quality_report.feedback_text,
        }

        return {
            "success": True,
            "message": "Documento analizado con Gemini Vision",
            "narrative": narrative,
            "document_type": doc_type,
            "document_type_label": self._get_classifier().get_label(doc_type),
            "reading_mode": reading_mode,
            "raw_text": raw_text,
            "has_text": has_text,
            "ocr_confidence": float(confidence),
            "word_count": word_count,
            "extracted_fields": extracted_fields,
            "visual_caption": None,
            "image_quality": quality_data,
        }

    def _build_barcode_response(self, barcode_result, image: np.ndarray) -> dict:
        """
        Construye la respuesta cuando se detecta un código QR o de barras.
        También corre OCR sobre la imagen completa para capturar texto circundante.
        """
        # Obtener información del código
        codes_data = []
        for code in barcode_result.codes:
            codes_data.append({
                "type": code.type,
                "data": code.data,
                "format": code.format_name,
            })

        # Determinar el tipo de documento
        if barcode_result.codes[0].type == "QR_CODE":
            doc_type = "codigo_qr"
            doc_label = "Código QR"
        else:
            doc_type = "codigo_barras"
            doc_label = "Código de barras"

        # Construir narrative base desde el resumen del QR
        narrative = barcode_result.summary

        # Correr OCR para detectar texto circundante al código
        surrounding_text = ""
        ocr_word_count = 0
        OCR_CONFIDENCE_THRESHOLD = 60.0  # % mínimo para considerar texto válido
        try:
            ocr_result = self._get_ocr().extract_text(image)
            surrounding_text = ocr_result.get("text", "").strip()
            ocr_word_count = ocr_result.get("word_count", 0)
            ocr_confidence = ocr_result.get("confidence") or 0.0
            logger.info(f"[SmartReading/Barcode] OCR: {ocr_word_count} palabras, {ocr_confidence:.0f}% confianza")

            # Descartar texto con baja confianza (es ruido/basura)
            if ocr_confidence < OCR_CONFIDENCE_THRESHOLD:
                logger.info(f"[SmartReading/Barcode] Texto descartado por baja confianza ({ocr_confidence:.0f}% < {OCR_CONFIDENCE_THRESHOLD}%)")
                surrounding_text = ""
                ocr_word_count = 0
        except Exception as e:
            logger.warning(f"[SmartReading/Barcode] OCR falló: {e}")

        # Combinar narrativa del QR con texto circundante si es significativo
        # Mínimo 4 palabras y confianza aceptable
        if ocr_word_count >= 4 and surrounding_text:
            preview = surrounding_text[:200] + ("..." if len(surrounding_text) > 200 else "")
            narrative = f"{narrative}. Alrededor del código hay texto que dice: {preview}"

        # Construir raw_text para compatibilidad
        raw_text = "; ".join([f"{c['type']}: {c['data']}" for c in codes_data])
        if surrounding_text:
            raw_text += f"\n{surrounding_text}"

        return {
            "success": True,
            "message": f"Se detectó {len(barcode_result.codes)} código(s)",
            "narrative": narrative,
            "document_type": doc_type,
            "document_type_label": doc_label,
            "confidence": 0.99,
            "reading_mode": "detallado",
            "raw_text": raw_text,
            "has_text": True,
            "ocr_confidence": 0.99,
            "word_count": len(barcode_result.codes) + ocr_word_count,
            "extracted_fields": {
                "codes": codes_data,
                "product_info": barcode_result.product_info,
            },
            "visual_caption": None,
            "image_quality": None,
            "classification": None,
        }

    def _analyze_with_tesseract(
        self,
        image: np.ndarray,
        quality_report,
        reading_mode: Optional[str] = None,
    ) -> dict:
        """
        Pipeline completo con Tesseract (fallback offline).

        Pipeline v2 con clasificador jerárquico:
          1. OCR con datos estructurales (bounding boxes, bloques, líneas)
          2. Clasificación jerárquica:
             Fase 1: MacroClassifier (solo layout) → macro-tipo
             Fase 2: SubtypeClassifier (keywords sobre texto limpio) → subtipo
             + estabilización temporal + detección de ambigüedad
          3. Extraer campos estructurados (regex)
          4. Generar narrativa por tipo
          5. Optimizar para TTS
        """
        # 1. OCR (ahora devuelve datos estructurales en 'layout')
        ocr_result = self._get_ocr().extract_text(image)
        raw_text = ocr_result.get("text", "")
        has_text = ocr_result.get("has_text", False)
        confidence = ocr_result.get("confidence")
        word_count = ocr_result.get("word_count", 0)
        layout_data = ocr_result.get("layout", {
            "word_boxes": [], "num_blocks": 0, "num_lines": 0
        })
        img_width = ocr_result.get("image_width", 800)
        img_height = ocr_result.get("image_height", 600)

        logger.info(f"[SmartReading/Tesseract] OCR: {word_count} palabras, "
                     f"confianza={confidence}, "
                     f"bloques={layout_data.get('num_blocks', 0)}, "
                     f"líneas={layout_data.get('num_lines', 0)}")
        if raw_text:
            logger.info(f"[SmartReading/Tesseract] Texto (200 chars): {raw_text[:200]}")

        # 2. Clasificación jerárquica (Fase 1: layout → Fase 2: keywords)
        classifier = self._get_classifier()
        cls_result = classifier.classify(
            raw_text=raw_text,
            word_count=word_count,
            layout_data=layout_data,
            img_width=img_width,
            img_height=img_height,
            ocr_confidence=confidence or 0.0,
        )
        doc_type = cls_result.doc_type
        cls_confidence = cls_result.confidence

        # Log de explicación interna
        expl = cls_result.explanation
        logger.info(
            f"[SmartReading/Classifier] "
            f"macro={expl.macro_type} → subtipo={doc_type} "
            f"(conf={cls_confidence:.2f})"
            + (f" AMBIGUO: {expl.ambiguity_note}" if expl.is_ambiguous else "")
            + (f" ESTABILIZADO: {expl.stabilization_note}" if expl.was_stabilized else "")
        )

        # 3. Extraer campos (usa texto crudo — los regex de StructureExtractor
        #    necesitan el texto original para detectar patrones como "Total: $150")
        extracted = self._extractor.extract(raw_text)

        # 3.5 RECONSTRUCCIÓN SEMÁNTICA: limpiar texto ANTES de generar narrativa
        #     Ningún generador debe usar jamás el texto OCR crudo.
        reconstructor = get_text_reconstructor()
        clean_text = reconstructor.reconstruct(raw_text, doc_type)
        logger.info(
            f"[SmartReading/Reconstruct] "
            f"{len(raw_text)} chars → {len(clean_text)} chars "
            f"({len(clean_text.split())} palabras limpias)"
        )
        if clean_text:
            logger.info(f"[SmartReading/Reconstruct] Texto limpio (200 chars): {clean_text[:200]}")

        # 4. Auto-select reading mode (usa reading_mode del clasificador v3 si disponible)
        classifier_reading_mode = getattr(cls_result, 'reading_mode', None)
        if reading_mode is None:
            reading_mode = self._auto_select_reading_mode(
                doc_type, extracted, word_count,
                classifier_reading_mode=classifier_reading_mode
            )
            logger.info(f"[SmartReading/Tesseract] Modo automático: {reading_mode} "
                        f"(classifier sugirió: {classifier_reading_mode})")

        # 5. Florence-2 para imagen_visual
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

        # 6. Generar narrativa
        # Los generadores reciben el texto crudo para extracción con regex
        # (el clean text puede eliminar montos, fechas, etc.)
        # Si clean_text está vacío, usar raw_text directamente.
        gen_text = clean_text if clean_text.strip() else raw_text
        narrative = self._generator.generate(
            doc_type=doc_type,
            reading_mode=reading_mode,
            raw_text=gen_text,
            extracted=extracted,
            visual_caption=visual_caption,
        )

        # 6.5 Optimizar lectura (jerarquía, incertidumbre, TTS)
        optimizer = self._get_optimizer()

        # Memoria contextual: verificar si es repetición
        memory_msg = optimizer.check_memory(raw_text, doc_type, word_count, narrative)
        if memory_msg:
            logger.info(f"[SmartReading/Memory] {memory_msg}")
            narrative = memory_msg

        # Aplicar optimización completa si no es repetición
        if not memory_msg:
            narrative = optimizer.optimize(
                narrative=narrative,
                doc_type=doc_type,
                reading_mode=reading_mode,
                ocr_confidence=confidence,
                word_count=word_count,
            )

        # 7. Mejorar prosodia
        narrative = self._enhancer.enhance(narrative, doc_type)

        logger.info(f"[SmartReading/Tesseract] Narrativa ({reading_mode}): {narrative[:150]}...")

        # 8. Feedback de calidad solo si imagen ilegible (critical)
        #    No agregar si el OCR logró leer — el frontend ya maneja el aviso.
        if quality_report.feedback_text and not quality_report.is_acceptable:
            narrative = quality_report.feedback_text + " " + narrative

        # 9. Construir respuesta
        quality_data = {
            "overall_score": quality_report.overall_score,
            "is_acceptable": quality_report.is_acceptable,
            "issues": [
                {"code": i.code, "severity": i.severity, "message": i.message_es}
                for i in quality_report.issues
            ],
            "feedback_text": quality_report.feedback_text,
        }

        return {
            "success": True,
            "message": "Documento analizado con Tesseract (offline)",
            "narrative": narrative,
            "document_type": doc_type,
            "document_type_label": cls_result.label,
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
            "image_quality": quality_data,
            "classification": {
                "macro_type": cls_result.macro_type,
                "classifier_reading_mode": classifier_reading_mode,
                "is_ambiguous": expl.is_ambiguous,
                "ambiguity_note": expl.ambiguity_note,
                "was_stabilized": expl.was_stabilized,
                "macro_reasons": expl.macro_reasons[:3],
                "subtype_reasons": expl.subtype_reasons[:3],
            },
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
