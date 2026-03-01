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
            "etiqueta": self._gen_etiqueta,
            "menu": self._gen_menu,
            "chat": self._gen_chat,
            "notificacion": self._gen_notificacion,
            "login": self._gen_login,
            "red_social": self._gen_red_social,
            "noticia": self._gen_noticia,
            "correo": self._gen_correo,
            "presentacion": self._gen_presentacion,
            "configuracion": self._gen_configuracion,
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
            parts.append(f"Dice: {body}.")
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

    # --- ETIQUETA DE PRODUCTO ---

    def _gen_etiqueta(self, mode: str, text: str, ex: ExtractedData,
                      caption: Optional[str]) -> str:
        """Genera narrativa natural para etiquetas de productos."""
        text = self._strip_navia_ui(text)
        if not text or not text.strip():
            return "Es una etiqueta de producto, pero no se pudo leer."

        parts = ["Esto es una etiqueta de producto."]

        # Detectar nombre del producto (primera línea larga o header)
        product_name = None
        if ex.headers:
            product_name = ex.headers[0]
        else:
            first = self._extract_first_sentence(text, skip_lines=0)
            if first and len(first.split()) <= 8:
                product_name = first

        if product_name:
            parts.append(f"Producto: {product_name}.")

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
            parts.append(f"Vencimiento: {venc_match.group(1).strip()}.")
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
            ingredientes = ingr_match.group(1).strip()
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
            parts.append(f"Fabricado por: {fab_match.group(1).strip()}.")

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
        text = self._strip_navia_ui(text)
        if not text or not text.strip():
            return "Es un menú, pero no se pudo leer."

        parts = ["Esto es un menú."]

        # Detectar nombre del restaurante (header o primera línea prominente)
        restaurant = None
        if ex.headers:
            restaurant = ex.headers[0]
        if restaurant:
            parts.append(f"Restaurante: {restaurant}.")

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
            item_name = m.group(1).strip()
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

    # Patrones de texto que pertenecen a la UI de NAVIA (feedback loop)
    # Detecta narrativas generadas por NAVIA para evitar leer su propia salida
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

        # 1. Anti-feedback-loop
        navia_match = self._NAVIA_UI_PATTERNS.search(text)
        if navia_match:
            text = text[:navia_match.start()].strip()
            if not text:
                return "Es un chat, pero no se pudo leer el contenido."

        # 2. Detectar contacto/grupo
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
        """Genera narrativa para notificaciones y alertas."""
        text = self._strip_navia_ui(text)
        if not text or not text.strip():
            return "Es una notificación, pero no se pudo leer el contenido."

        parts = ["Esto es una notificación."]
        body = self._extract_body_preview(text, max_words=40)
        if body:
            parts.append(f"Dice: {body}.")
        return " ".join(parts)

    # --- LOGIN ---

    def _gen_login(self, mode: str, text: str, ex: ExtractedData,
                   caption: Optional[str]) -> str:
        """Genera narrativa para pantallas de inicio de sesión."""
        text = self._strip_navia_ui(text)
        if not text or not text.strip():
            return "Es una pantalla de inicio de sesión."

        parts = ["Esto es una pantalla de inicio de sesión."]

        # Detectar qué servicio/app es
        service_match = re.search(
            r'\b(Google|Facebook|Apple|Instagram|Twitter|X|GitHub|Microsoft|'
            r'Outlook|Netflix|Spotify|Amazon|WhatsApp|Telegram|LinkedIn|'
            r'TikTok|Snapchat|Discord|Uber|PayPal)\b', text, re.IGNORECASE)
        if service_match:
            parts.append(f"Es de {service_match.group(1)}.")

        # Detectar campos visibles
        has_password = bool(re.search(r'\b(?:contrase[ñn]a|password|clave)\b', text, re.IGNORECASE))
        has_user = bool(re.search(r'\b(?:usuario|email|correo|user)\b', text, re.IGNORECASE))
        has_register = bool(re.search(r'\b(?:registr|sign\s*up|crear?\s+cuenta|create)\b', text, re.IGNORECASE))
        has_forgot = bool(re.search(r'\b(?:olvid|forgot|recuperar)\b', text, re.IGNORECASE))

        if has_user and has_password:
            parts.append("Tiene campos para usuario y contraseña.")
        elif has_user:
            parts.append("Pide ingresar el usuario o correo.")
        if has_register:
            parts.append("También tiene opción de registrarse.")
        if has_forgot:
            parts.append("Hay opción para recuperar la contraseña.")

        return " ".join(parts)

    # --- RED SOCIAL ---

    def _gen_red_social(self, mode: str, text: str, ex: ExtractedData,
                        caption: Optional[str]) -> str:
        """Genera narrativa natural para perfiles y publicaciones de redes sociales."""
        text = self._strip_navia_ui(text)
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
        """Genera narrativa para noticias, artículos, blogs, portadas."""
        text = self._strip_navia_ui(text)
        if not text or not text.strip():
            return "Es una noticia o artículo, pero no se pudo leer el contenido."

        parts = ["Esto es una noticia o artículo."]

        # Buscar fuente/medio
        source_name = None
        source_match = re.search(
            r'\b(Reuters|AP|AFP|EFE|CNN|BBC|El\s+Pa[ií]s|El\s+Nacional|'
            r'El\s+Universal|El\s+Tiempo|New\s+York\s+Times|The\s+Guardian|'
            r'Washington\s+Post|Forbes|Bloomberg)\b', text, re.IGNORECASE)
        if source_match:
            source_name = source_match.group(1)
            parts.append(f"Fuente: {source_name}.")

        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")

        if mode == "resumen":
            # Solo título/primera oración, saltando la línea de la fuente
            first = self._extract_first_sentence(text, skip_lines=0)
            # Evitar que el título sea simplemente el nombre de la fuente
            if first and source_name and first.strip().lower() == source_name.strip().lower():
                first = self._extract_first_sentence(text, skip_lines=1)
            if first:
                parts.append(f"Título: {first}.")
        else:
            # Detallado: más contenido, pero quitar fuente del body para no repetir
            body_text = text
            if source_name:
                # Quitar primera aparición del nombre de la fuente del body
                body_text = re.sub(
                    r'\b' + re.escape(source_name) + r'\b',
                    '', body_text, count=1, flags=re.IGNORECASE
                ).strip()
            body = self._extract_body_preview(body_text, max_words=60)
            if body:
                parts.append(f"Dice: {body}.")

        return " ".join(parts)

    # --- CORREO ELECTRÓNICO ---

    def _gen_correo(self, mode: str, text: str, ex: ExtractedData,
                    caption: Optional[str]) -> str:
        """Genera narrativa para correos electrónicos."""
        text = self._strip_navia_ui(text)
        if not text or not text.strip():
            return "Es un correo electrónico, pero no se pudo leer el contenido."

        parts = ["Esto es un correo electrónico."]

        # Extraer remitente
        from_match = re.search(r'(?:de|from)\s*:?\s*([^\n]{3,40})', text, re.IGNORECASE)
        if from_match:
            parts.append(f"De: {from_match.group(1).strip()}.")

        # Extraer asunto
        subject_text = None
        subject_match = re.search(r'(?:asunto|subject)\s*:?\s*([^\n]{3,60})', text, re.IGNORECASE)
        if subject_match:
            subject_text = subject_match.group(1).strip()
            parts.append(f"Asunto: {subject_text}.")

        if ex.dates:
            parts.append(f"Fecha: {ex.dates[0]}.")

        if ex.emails:
            parts.append(f"Correo: {ex.emails[0]}.")

        if mode == "detallado":
            # Quitar líneas de De:/Asunto: del texto antes de extraer body
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
        """Genera narrativa para diapositivas/presentaciones."""
        text = self._strip_navia_ui(text)
        if not text or not text.strip():
            return "Es una diapositiva, pero no se pudo leer el contenido."

        parts = ["Esto es una diapositiva de presentación."]

        # Detectar número de slide
        slide_match = re.search(r'(\d+)\s*/\s*(\d+)', text)
        if slide_match:
            parts.append(f"Diapositiva {slide_match.group(1)} de {slide_match.group(2)}.")

        # El título suele ser el texto más prominente (primera línea larga)
        first = self._extract_first_sentence(text, skip_lines=0)
        if first:
            parts.append(f"Título: {first}.")

        body = self._extract_body_preview(text, max_words=40)
        if body and body != first:
            parts.append(f"Contenido: {body}.")

        return " ".join(parts)

    # --- CONFIGURACIÓN ---

    def _gen_configuracion(self, mode: str, text: str, ex: ExtractedData,
                           caption: Optional[str]) -> str:
        """Genera narrativa para pantallas de configuración/ajustes."""
        text = self._strip_navia_ui(text)
        if not text or not text.strip():
            return "Es una pantalla de configuración."

        parts = ["Esto es una pantalla de configuración o ajustes."]

        # Detectar opciones visibles
        options = []
        option_patterns = [
            (r'\bWi-?Fi\b', 'Wi-Fi'),
            (r'\bBluetooth\b', 'Bluetooth'),
            (r'\b(?:datos?\s+m[oó]viles|mobile\s+data)\b', 'Datos móviles'),
            (r'\b(?:brillo|brightness)\b', 'Brillo'),
            (r'\b(?:volumen|volume|sonido|sound)\b', 'Sonido'),
            (r'\b(?:bater[ií]a|battery)\b', 'Batería'),
            (r'\b(?:almacenamiento|storage)\b', 'Almacenamiento'),
            (r'\b(?:modo\s+oscuro|dark\s+mode)\b', 'Modo oscuro'),
            (r'\b(?:modo\s+avi[oó]n|airplane)\b', 'Modo avión'),
            (r'\b(?:notificaciones|notifications)\b', 'Notificaciones'),
            (r'\b(?:privacidad|privacy)\b', 'Privacidad'),
            (r'\b(?:accesibilidad|accessibility)\b', 'Accesibilidad'),
            (r'\b(?:pantalla|display)\b', 'Pantalla'),
            (r'\b(?:idioma|language)\b', 'Idioma'),
            (r'\b(?:cuenta|account)\b', 'Cuenta'),
        ]
        for pattern, label in option_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                options.append(label)

        if options:
            visible = ", ".join(options[:6])
            parts.append(f"Opciones visibles: {visible}.")
        else:
            body = self._extract_body_preview(text, max_words=30)
            if body:
                parts.append(f"Dice: {body}.")

        return " ".join(parts)

    # --- HELPER: Strip NAVIA UI text (reutilizable) ---

    def _strip_navia_ui(self, text: str) -> str:
        """Elimina texto de la propia UI de NAVIA para evitar feedback loop."""
        if not text:
            return text
        navia_match = self._NAVIA_UI_PATTERNS.search(text)
        if navia_match:
            text = text[:navia_match.start()].strip()
        return text

    # --- DESCONOCIDO (fallback universal) ---

    def _gen_desconocido(self, mode: str, text: str, ex: ExtractedData,
                         caption: Optional[str]) -> str:
        # Primero limpiar texto de la UI de NAVIA
        text = self._strip_navia_ui(text)

        if not text or not text.strip():
            if caption:
                return f"La imagen muestra: {caption}."
            return "No se detectó texto en la imagen."

        # Filtrar texto: solo líneas con palabras reales (> 2 words, no basura)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        clean_lines = []
        for line in lines:
            words = line.split()
            if len(words) < 2:
                continue
            # Calcular ratio de palabras "reales" (> 2 chars con letras) vs basura
            real_words = [w for w in words if len(w) > 2 and re.search(r'[a-záéíóúñ]', w, re.IGNORECASE)]
            if len(real_words) >= len(words) * 0.4:
                clean_lines.append(" ".join(words))

        clean_text = "\n".join(clean_lines)
        if not clean_text.strip():
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
            preview = " ".join(clean_text.split()[:30])
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
                preview = " ".join(clean_text.split()[:30])
                parts.append(f"Dice: {preview}.")
            return " ".join(parts)

        # detallado: leer contenido limpio (no basura)
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
        # Leer el contenido principal (hasta 80 palabras del texto limpio)
        body = self._extract_body_preview(clean_text, max_words=80)
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
        clean = NarrativeGenerator._clean_ocr_words(raw_text)
        words = clean.split()[:40]
        return " ".join(words) if words else "No se pudo leer el documento."

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
        '$': 'dólares', '€': 'euros', '£': 'libras',
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
      Imagen → Tesseract OCR → DocumentClassifier → StructureExtractor
             → NarrativeGenerator → ProsodyEnhancer → Response
    """

    def __init__(self):
        self._ocr_service = None
        self._gemini_service = None
        self._classifier = DocumentClassifier()
        self._extractor = StructureExtractor()
        self._generator = NarrativeGenerator()
        self._enhancer = ProsodyEnhancer()
        self._optimizer = None
        self._captioning = None

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

    def _auto_select_reading_mode(self, doc_type: str, extracted, word_count: int) -> str:
        """
        Selecciona automáticamente el modo de lectura óptimo según el tipo de documento.
        """
        if doc_type in ("factura", "recibo"):
            return "financiero"
        if doc_type in ("carta", "formulario", "documento_informativo", "correo"):
            return "detallado"
        if doc_type in ("etiqueta", "menu", "tarjeta", "chat", "notificacion",
                        "login", "red_social", "configuracion", "presentacion"):
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
        1. Analizar calidad de imagen (siempre, local)
        2. Intentar Gemini Vision (si está disponible)
        3. Si Gemini falla → Tesseract local (fallback)
        4. Aplicar ProsodyEnhancer a la narrativa
        5. Construir respuesta

        Args:
            image: Imagen BGR (OpenCV)
            reading_mode: Opcional. Si no se especifica, se selecciona automáticamente.

        Returns:
            dict compatible con SmartReadingResponse
        """
        # 0. Analizar calidad de imagen ANTES de todo
        quality_analyzer = self._get_quality_analyzer()
        quality_report = quality_analyzer.analyze(image)

        # 1. Intentar Gemini Vision (pipeline primario)
        gemini_result = self._try_gemini(image)

        if gemini_result is not None:
            return self._build_response_from_gemini(
                gemini_result, quality_report, reading_mode
            )

        # 2. Fallback: Pipeline Tesseract local
        logger.info("[SmartReading] Usando Tesseract (fallback local)")
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

        # Si calidad es mala y poco texto, agregar feedback
        if quality_report.feedback_text and (
            not has_text or confidence < 50
        ):
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
            "document_type_label": DocumentClassifier.get_label(doc_type),
            "reading_mode": reading_mode,
            "raw_text": raw_text,
            "has_text": has_text,
            "ocr_confidence": float(confidence),
            "word_count": word_count,
            "extracted_fields": extracted_fields,
            "visual_caption": None,
            "image_quality": quality_data,
        }

    def _analyze_with_tesseract(
        self,
        image: np.ndarray,
        quality_report,
        reading_mode: Optional[str] = None,
    ) -> dict:
        """
        Pipeline completo con Tesseract (fallback offline).
        Este es el pipeline original que funciona sin internet.
        """
        # 1. OCR
        ocr_result = self._get_ocr().extract_text(image)
        raw_text = ocr_result.get("text", "")
        has_text = ocr_result.get("has_text", False)
        confidence = ocr_result.get("confidence")
        word_count = ocr_result.get("word_count", 0)

        logger.info(f"[SmartReading/Tesseract] OCR: {word_count} palabras, confianza={confidence}")
        if raw_text:
            logger.info(f"[SmartReading/Tesseract] Texto (200 chars): {raw_text[:200]}")

        # 2. Clasificar
        doc_type, cls_confidence = self._classifier.classify(raw_text, word_count)
        logger.info(f"[SmartReading/Tesseract] Clasificación: {doc_type} (conf={cls_confidence:.2f})")

        # 3. Extraer campos
        extracted = self._extractor.extract(raw_text)

        # 4. Auto-select reading mode
        if reading_mode is None:
            reading_mode = self._auto_select_reading_mode(doc_type, extracted, word_count)
            logger.info(f"[SmartReading/Tesseract] Modo automático: {reading_mode}")

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
        narrative = self._generator.generate(
            doc_type=doc_type,
            reading_mode=reading_mode,
            raw_text=raw_text,
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

        # 8. Feedback de calidad si OCR pobre
        if quality_report.feedback_text and (
            not has_text or (confidence is not None and confidence < 50)
        ):
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
            "image_quality": quality_data,
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
