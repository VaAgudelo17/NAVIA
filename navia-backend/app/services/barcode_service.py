"""
========================================================================
NAVIA Backend - Servicio de Lectura de Códigos QR y Barras
========================================================================
Detecta y decodifica códigos QR, códigos de barras (EAN, UPC, Code128, etc.)
y proporciona información adicional basada en el tipo de código.

Funcionalidades:
  - Detección de códigos QR
  - Detección de códigos de barras (EAN-13, EAN-8, UPC-A, UPC-E, Code128, Code39)
  - Identificación de tipo de código
  - Para códigos de producto: intento de obtener información del producto

Dependencias: opencv-python-headless (ya incluido en requirements.txt)
========================================================================
"""

import re
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class DetectedCode:
    """Representa un código detectado en la imagen."""
    type: str              # QR_CODE, EAN_13, UPC_A, etc.
    data: str              # Contenido decodificado
    format_name: str       # Nombre legible del formato
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    confidence: float = 1.0


@dataclass
class BarcodeResult:
    """Resultado completo del escaneo de códigos."""
    codes: List[DetectedCode]
    has_codes: bool
    summary: str
    product_info: Optional[Dict] = None


# Mapeo de tipos de códigos a nombres legibles (valores basados en OpenCV)
# Los valores pueden variar según la versión de OpenCV
# Servicios conocidos por dominio para generar narrativas más naturales
_KNOWN_SERVICES = {
    "youtube.com": "un video de YouTube",
    "youtu.be": "un video de YouTube",
    "instagram.com": "una publicación o perfil de Instagram",
    "facebook.com": "una página de Facebook",
    "fb.me": "una página de Facebook",
    "twitter.com": "un perfil de Twitter",
    "x.com": "un perfil de X (Twitter)",
    "tiktok.com": "un video de TikTok",
    "vm.tiktok.com": "un video de TikTok",
    "spotify.com": "contenido de Spotify",
    "open.spotify.com": "contenido de Spotify",
    "wa.me": "un chat de WhatsApp",
    "whatsapp.com": "un chat de WhatsApp",
    "maps.google.com": "una ubicación en Google Maps",
    "t.me": "un canal o contacto de Telegram",
    "linktr.ee": "un árbol de enlaces",
    "paypal.com": "un pago de PayPal",
    "paypal.me": "un pago de PayPal",
}

_BARCODE_TYPE_MAP = {
    0: "Unknown",
    1: "EAN-13",
    2: "EAN-8", 
    3: "UPC-A",
    4: "UPC-E",
    5: "EAN-8",
    6: "Code 128",
    7: "Code 39",
    8: "Code 93",
    9: "ITF",
    10: "Codabar",
    11: "Unknown",
    12: "Unknown",
    13: "Unknown",
    14: "Unknown",
    15: "Código QR",
}

def _get_barcode_type_name(code_type) -> str:
    """Obtiene el nombre del tipo de código de barras."""
    if code_type in _BARCODE_TYPE_MAP:
        return _BARCODE_TYPE_MAP[code_type]
    return f"Código ({code_type})"

# Códigos de país para EAN-13 (primeiros 3 dígitos)
EAN13_COUNTRY_CODES = {
    "000": "EE.UU. y Canadá",
    "001": "EE.UU. y Canadá",
    "030": "EE.UU. (UC)", 
    "050": "EE.UU. (coupons)",
    "400": "Alemania",
    "500": "Reino Unido",
    "600": "Sudáfrica",
    "750": "México",
    "760": "Suiza",
    "770": "Colombia",
    "789": "Brasil",
    "800": "Italia",
    "840": "España",
    "850": "Cuba",
    "859": "República Checa/Eslovaquia",
    "890": "India",
    "900": "Austria",
    "910": "Eslovenia",
    "920": "Alemania",
    "930": "Australia",
    "940": "Nueva Zelanda",
}


class BarcodeReader:
    """
    Servicio de lectura de códigos QR y barras.
    
    Usa OpenCV内置 (versión 4.x+) para detección y decodificación.
    No requiere dependencias adicionales.
    """

    def __init__(self):
        self._qr_detector = cv2.QRCodeDetector()
        self._barcode_detector = cv2.barcode.BarcodeDetector()

    def read(self, image: np.ndarray) -> BarcodeResult:
        """
        Lee todos los códigos QR y de barras en una imagen.
        
        Args:
            image: Imagen en formato BGR (OpenCV)
            
        Returns:
            BarcodeResult con los códigos detectados
        """
        if image is None or image.size == 0:
            return BarcodeResult(
                codes=[],
                has_codes=False,
                summary="No se detectó ninguna imagen válida"
            )

        codes = []
        
        # 1. Detectar códigos QR
        qr_codes = self._read_qr_codes(image)
        codes.extend(qr_codes)
        
        # 2. Detectar códigos de barras
        barcode_codes = self._read_barcodes(image)
        codes.extend(barcode_codes)
        
        # 3. Generar resumen
        if not codes:
            return BarcodeResult(
                codes=[],
                has_codes=False,
                summary="No se detectó ningún código QR ni de barras"
            )
        
        # 4. Intentar obtener información del producto
        product_info = None
        if codes:
            product_info = self._get_product_info(codes)
        
        summary = self._generate_summary(codes)
        
        return BarcodeResult(
            codes=codes,
            has_codes=True,
            summary=summary,
            product_info=product_info
        )

    def _read_qr_codes(self, image: np.ndarray) -> List[DetectedCode]:
        """Detecta y decodifica códigos QR."""
        codes = []
        
        try:
            # Decodificar códigos QR
            retval, decoded_info, points, straight_qrcode = self._qr_detector.detectAndDecodeMulti(image)
            
            if retval and decoded_info:
                for i, data in enumerate(decoded_info):
                    if data:  # Skip empty results
                        bbox = None
                        if points is not None and i < len(points):
                            pts = points[i]
                            x = int(pts[0][0])
                            y = int(pts[0][1])
                            w = int(pts[2][0] - pts[0][0])
                            h = int(pts[2][1] - pts[0][1])
                            bbox = (x, y, w, h)
                        
                        codes.append(DetectedCode(
                            type="QR_CODE",
                            data=data,
                            format_name="Código QR",
                            bounding_box=bbox,
                            confidence=1.0
                        ))
                        
        except Exception as e:
            logger.warning(f"Error detectando códigos QR: {e}")
        
        return codes

    def _read_barcodes(self, image: np.ndarray) -> List[DetectedCode]:
        """Detecta y decodifica códigos de barras."""
        codes = []
        
        try:
            # El detector de códigos de barras necesita una imagen en escala de grises o a color
            # y puede detectar múltiples códigos
            retval, decoded_info, decoded_type, points = self._barcode_detector.detectAndDecodeMulti(image)
            
            if retval and decoded_info:
                for i, data in enumerate(decoded_info):
                    if data:  # Skip empty results
                        barcode_type = decoded_type[i] if i < len(decoded_type) else cv2.barcode.BARCODE_CODE_128
                        
                        bbox = None
                        if points is not None and i < len(points):
                            pts = points[i]
                            if pts is not None and len(pts) >= 4:
                                x = int(pts[0][0])
                                y = int(pts[0][1])
                                w = int(pts[2][0] - pts[0][0])
                                h = int(pts[2][1] - pts[0][1])
                                bbox = (x, y, w, h)
                        
                        format_name = _get_barcode_type_name(barcode_type)
                        
                        codes.append(DetectedCode(
                            type=barcode_type,
                            data=data,
                            format_name=format_name,
                            bounding_box=bbox,
                            confidence=1.0
                        ))
                        
        except Exception as e:
            # El detector puede fallar si no hay códigos, no es necesariamente un error
            logger.debug(f"Detectando códigos de barras: {e}")
        
        return codes

    def _get_product_info(self, codes: List[DetectedCode]) -> Optional[Dict]:
        """
        Intenta obtener información del producto para códigos de barras de productos.
        
        Por ahora retorna información básica. En el futuro podría integrar:
        - OpenFoodFacts API para alimentos
        - UPCItemDB para productos generales
        """
        for code in codes:
            # Solo intentar para códigos de producto comunes
            if code.type in ["QR_CODE", "EAN_13", "EAN_8", "UPC_A", "UPC_E"]:
                # Limpiar el código
                numeric_code = re.sub(r'[^\d]', '', code.data)
                
                info = {
                    "raw_data": code.data,
                    "numeric_only": numeric_code,
                    "length": len(numeric_code),
                }
                
                # Agregar información según el tipo
                if code.type == "EAN_13" and len(numeric_code) >= 3:
                    country_code = numeric_code[:3]
                    info["country"] = EAN13_COUNTRY_CODES.get(country_code, "Desconocido")
                
                # Para QR, intentar detectar si es URL
                if code.type == "QR_CODE":
                    if code.data.startswith("http://") or code.data.startswith("https://"):
                        info["is_url"] = True
                        info["domain"] = self._extract_domain(code.data)
                    elif code.data.startswith("WIFI:"):
                        info["is_wifi"] = True
                        info["wifi_ssid"] = self._extract_wifi_ssid(code.data)
                
                return info
        
        return None

    def _resolve_redirect(self, url: str, timeout: float = 3.0) -> tuple:
        """
        Sigue redirecciones y lee el título de la página en una sola petición.
        Lee solo los primeros 8 KB (suficiente para el <head>).
        Retorna (final_url, page_title). Ambos caen al valor original/None si falla.
        """
        try:
            import requests
            response = requests.get(
                url,
                allow_redirects=True,
                timeout=timeout,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; NAVIA/1.0)"},
            )
            final_url = str(response.url)

            # Leer solo hasta </title> o los primeros 8 KB
            content = b""
            for chunk in response.iter_content(chunk_size=1024):
                content += chunk
                if len(content) >= 8192 or b"</title>" in content.lower():
                    break
            response.close()

            if final_url != url:
                logger.info(f"[Barcode] Redirect resuelto: {url} → {final_url}")

            page_title = self._extract_page_title(content.decode("utf-8", errors="ignore"))
            if page_title:
                logger.info(f"[Barcode] Título: {page_title}")

            return final_url, page_title

        except Exception as e:
            logger.debug(f"[Barcode] No se pudo resolver {url}: {e}")
            return url, None

    def _extract_page_title(self, html: str) -> Optional[str]:
        """Extrae el título de la página y lo limpia para TTS."""
        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if not match:
            return None

        title = match.group(1).strip()

        # Decodificar entidades HTML básicas
        title = (title
                 .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                 .replace("&#39;", "'").replace("&quot;", '"').replace("&nbsp;", " "))

        # Eliminar sufijos de sitio conocidos: "Título - YouTube", "Título | Instagram"
        title = re.sub(
            r'\s*[-–—|]\s*(YouTube|Instagram|Facebook|Twitter|X|TikTok|Spotify|WhatsApp)\s*$',
            '', title, flags=re.IGNORECASE,
        )

        title = title.strip()

        # Limitar longitud para TTS
        if len(title) > 80:
            title = title[:77] + "..."

        return title if title else None

    def _extract_domain(self, url: str) -> str:
        """Extrae el dominio de una URL."""
        try:
            match = re.search(r'https?://([^/]+)', url)
            return match.group(1) if match else url
        except:
            return url

    def _extract_wifi(self, qr_data: str) -> Dict:
        """Extrae información WiFi de un código QR."""
        wifi = {}
        try:
            if qr_data.startswith("WIFI:"):
                ssid_match = re.search(r'S:([^;]+)', qr_data)
                pass_match = re.search(r'P:([^;]+)', qr_data)
                if ssid_match:
                    wifi["ssid"] = ssid_match.group(1)
                if pass_match:
                    wifi["has_password"] = True
        except:
            pass
        return wifi

    def _generate_summary(self, codes: List[DetectedCode]) -> str:
        """Genera un resumen legible de los códigos detectados."""
        if not codes:
            return "No se detectó ningún código"
        
        if len(codes) == 1:
            code = codes[0]
            if code.type == "QR_CODE":
                return self._summary_qr(code)
            else:
                return self._summary_barcode(code)
        else:
            # Múltiples códigos
            parts = []
            for code in codes[:3]:  # Max 3 para no saturar
                if code.type == "QR_CODE":
                    parts.append(f"QR: {code.data[:30]}{'...' if len(code.data) > 30 else ''}")
                else:
                    parts.append(f"{code.format_name}: {code.data}")
            
            summary = ". ".join(parts)
            if len(codes) > 3:
                summary += f" y {len(codes) - 3} más"
            return summary

    def _summary_qr(self, code: DetectedCode) -> str:
        """Genera resumen para código QR."""
        data = code.data

        # Detectar tipo de contenido
        if data.startswith("http://") or data.startswith("https://"):
            # Seguir redirecciones y obtener título de la página en una sola petición
            resolved, page_title = self._resolve_redirect(data)
            domain = self._extract_domain(resolved)
            # Buscar servicio conocido por dominio (sin www.)
            clean_domain = domain.lower().lstrip("www.")
            service_label = None
            for known_domain, label in _KNOWN_SERVICES.items():
                if clean_domain == known_domain or clean_domain.endswith("." + known_domain):
                    service_label = label
                    break

            base = (f"Es un código QR que lleva a {service_label}"
                    if service_label
                    else f"Es un código QR con un enlace al sitio web {clean_domain}")

            if page_title:
                return f"{base}: {page_title}"
            return base

        if data.startswith("WIFI:"):
            ssid = self._extract_wifi_ssid(data)
            return f"Es un código QR con la contraseña de la red WiFi llamada {ssid}"

        if data.startswith("BEGIN:VCARD") or data.startswith("MECARD:"):
            # Intentar extraer nombre de la vCard/MECARD
            name_match = re.search(r'(?:FN:|N:)([^\n;]+)', data)
            if name_match:
                name = name_match.group(1).strip()
                return f"Es un código QR con la tarjeta de contacto de {name}"
            return "Es un código QR con una tarjeta de contacto"

        if data.startswith("geo:"):
            coords_match = re.search(r'geo:([\d.\-]+),([\d.\-]+)', data)
            if coords_match:
                return f"Es un código QR con una ubicación geográfica"
            return "Es un código QR con una ubicación geográfica"

        if data.startswith("mailto:"):
            email = data.replace("mailto:", "").split("?")[0]
            return f"Es un código QR para enviar un correo a {email}"

        if data.startswith("tel:"):
            number = data.replace("tel:", "")
            return f"Es un código QR con el número de teléfono {number}"

        if data.startswith("smsto:") or data.startswith("sms:"):
            return "Es un código QR para enviar un mensaje de texto"

        if re.match(r'^\d+$', data):
            return f"Es un código QR con el número {data}"

        # Texto genérico
        preview = data[:80] + "..." if len(data) > 80 else data
        return f"Es un código QR que dice: {preview}"

    def _summary_barcode(self, code: DetectedCode) -> str:
        """Genera resumen para código de barras."""
        format_name = code.format_name
        data = code.data
        
        # Para códigos de producto
        if code.type in ["EAN_13", "EAN_8", "UPC_A", "UPC_E"]:
            numeric = re.sub(r'[^\d]', '', data)
            country_info = ""
            if code.type == "EAN_13" and len(numeric) >= 3:
                country_info = f". Origen probable: {EAN13_COUNTRY_CODES.get(numeric[:3], 'desconocido')}"
            
            return f"{format_name}. Código de producto: {data}{country_info}"
        
        # Para otros tipos
        return f"{format_name}. Código: {data}"

    def _extract_wifi_ssid(self, qr_data: str) -> str:
        """Extrae el SSID de un código QR WiFi."""
        match = re.search(r'S:([^;]+)', qr_data)
        return match.group(1) if match else "desconocida"


# Singleton
_barcode_reader: Optional[BarcodeReader] = None


def get_barcode_reader() -> BarcodeReader:
    """Obtiene la instancia singleton del lector de códigos."""
    global _barcode_reader
    if _barcode_reader is None:
        _barcode_reader = BarcodeReader()
    return _barcode_reader


# Funciones de conveniencia
def read_barcodes(image: np.ndarray) -> BarcodeResult:
    """Función de conveniencia para leer códigos de una imagen."""
    return get_barcode_reader().read(image)
