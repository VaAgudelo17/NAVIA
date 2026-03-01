"""
============================================================================
NAVIA Backend - Utilidades para PDFs
============================================================================
Funciones para convertir PDFs a imágenes para OCR.
============================================================================
"""

import io
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def convert_pdf_first_page_to_image(pdf_bytes: bytes) -> np.ndarray:
    """
    Convierte la primera página de un PDF a imagen para OCR.
    
    Usa fitz (PyMuPDF) si está disponible, sino pdf2image.
    
    Args:
        pdf_bytes: Contenido del archivo PDF en bytes
        
    Returns:
        Imagen en formato OpenCV (numpy array BGR)
        
    Raises:
        ImportError: Si ninguna librería PDF está disponible
        ValueError: Si el PDF no tiene páginas
    """
    try:
        import fitz  # PyMuPDF
        return _convert_with_fitz(pdf_bytes)
    except ImportError:
        pass
    
    # Fallback a pdf2image
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=1)
        if not images:
            raise ValueError("PDF no tiene páginas")
        
        # Convertir PIL Image a OpenCV
        pil_image = images[0]
        return _pil_to_cv2(pil_image)
    except ImportError:
        raise ImportError(
            "Se requiere 'pymupdf' o 'pdf2image' para procesar PDFs. "
            "Instalar con: pip install pymupdf"
        )


def _convert_with_fitz(pdf_bytes: bytes) -> np.ndarray:
    """Convierte PDF a imagen usando PyMuPDF (fitz)."""
    import fitz
    
    # Abrir PDF desde bytes
    pdf_stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    
    if doc.page_count == 0:
        doc.close()
        raise ValueError("PDF no tiene páginas")
    
    # Obtener primera página
    page = doc[0]
    
    # Renderizar a imagen (200 DPI para buen OCR)
    zoom = 200 / 72  # 72 es el zoom base de fitz
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    # Convertir a numpy array (RGBA)
    img_bytes = pix.tobytes("png")
    import cv2
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convertir RGBA a BGR si es necesario
    if img is None:
        # Como fallback, cargar con PIL
        from PIL import Image
        pil_img = Image.open(io.BytesIO(img_bytes))
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        img = _pil_to_cv2(pil_img)
    
    doc.close()
    logger.info(f"[PDF] Convertida página 1 a imagen: {img.shape[1]}x{img.shape[0]}")
    
    return img


def _pil_to_cv2(pil_image) -> np.ndarray:
    """Convierte PIL Image a OpenCV numpy array (RGB -> BGR)."""
    import cv2
    np_img = np.array(pil_image)
    # RGB a BGR
    return np_img[:, :, ::-1]


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """
    Obtiene el número de páginas en un PDF.
    
    Args:
        pdf_bytes: Contenido del archivo PDF en bytes
        
    Returns:
        Número de páginas
    """
    try:
        import fitz
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        count = doc.page_count
        doc.close()
        return count
    except ImportError:
        try:
            from pdf2image import convert_from_bytes
            # pdf2image es lento para solo contar páginas
            # Asumimos 1 por defecto
            return 1
        except:
            return 1


def extract_pdf_metadata(pdf_bytes: bytes) -> dict:
    """
    Extrae metadatos básicos de un PDF.
    
    Args:
        pdf_bytes: Contenido del archivo PDF en bytes
        
    Returns:
        Diccionario con metadatos (pages, title, author, etc.)
    """
    metadata = {
        "page_count": 0,
        "title": None,
        "author": None,
    }
    
    try:
        import fitz
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        metadata["page_count"] = doc.page_count
        
        # Metadatos del documento
        doc_meta = doc.metadata
        if doc_meta:
            metadata["title"] = doc_meta.get("title") or None
            metadata["author"] = doc_meta.get("author") or None
            
        doc.close()
    except ImportError:
        metadata["page_count"] = get_pdf_page_count(pdf_bytes)
    
    return metadata
