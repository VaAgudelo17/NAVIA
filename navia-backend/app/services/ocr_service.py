"""
============================================================================
NAVIA Backend - Servicio de OCR (Reconocimiento Óptico de Caracteres)
============================================================================
Este módulo implementa la extracción de texto de imágenes usando Tesseract.

¿Qué es Tesseract?
- Motor OCR de código abierto desarrollado originalmente por HP (1985-1995)
- Mantenido por Google desde 2006
- Uno de los motores OCR más precisos disponibles gratuitamente
- Soporta más de 100 idiomas

¿Por qué pytesseract?
- Wrapper de Python para Tesseract
- API simple y fácil de usar
- Permite acceso a datos detallados (confianza, posiciones)

Limitaciones del OCR:
- Requiere texto legible y con buen contraste
- Texto manuscrito tiene menor precisión
- Fuentes decorativas pueden ser problemáticas
- La orientación afecta los resultados
============================================================================
"""

import pytesseract
from PIL import Image
import numpy as np
from typing import Dict, Optional
import cv2
import logging

from app.core.config import settings
from app.utils.image_utils import (
    preprocess_for_ocr,
    cv2_image_to_pil,
    resize_image_if_needed
)

# Configurar logging
logger = logging.getLogger(__name__)

# Configurar ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


class OCRService:
    """
    Servicio para extracción de texto de imágenes.

    Uso:
        service = OCRService()
        resultado = service.extract_text(imagen_cv2)
    """

    def __init__(self):
        """Inicializa el servicio de OCR."""
        self.language = settings.TESSERACT_LANG
        self._verify_tesseract_installation()

    def _verify_tesseract_installation(self) -> None:
        """
        Verifica que Tesseract esté instalado correctamente.

        Raises:
            RuntimeError: Si Tesseract no está disponible
        """
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR versión {version} detectado")
        except Exception as e:
            logger.error(f"Tesseract no encontrado: {e}")
            raise RuntimeError(
                "Tesseract OCR no está instalado o no se encuentra en el PATH. "
                f"Ruta configurada: {settings.TESSERACT_CMD}. "
                "Instalar con: brew install tesseract tesseract-lang (macOS)"
            )

    def extract_text(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> Dict:
        """
        Extrae texto de una imagen.

        Proceso:
        1. Redimensionar si es muy grande
        2. Preprocesar (opcional pero recomendado)
        3. Ejecutar OCR
        4. Limpiar y estructurar resultados

        Args:
            image: Imagen en formato OpenCV (numpy array BGR)
            preprocess: Si True, aplica preprocesamiento para mejorar OCR

        Returns:
            Diccionario con:
            - text: Texto extraído
            - confidence: Confianza promedio
            - word_count: Número de palabras
            - has_text: Si se encontró texto
        """
        try:
            # Paso 1: Redimensionar si es necesario
            image = resize_image_if_needed(image)

            # Paso 2: Preprocesar para mejorar OCR
            if preprocess:
                processed_image = preprocess_for_ocr(image)
                # Para OCR, convertimos la imagen binaria a PIL
                pil_image = Image.fromarray(processed_image)
            else:
                pil_image = cv2_image_to_pil(image)

            # Paso 3: Ejecutar OCR con datos detallados
            # psm 3 = Automatic page segmentation (sin OSD)
            # Configuración optimizada para texto mixto
            custom_config = r'--oem 3 --psm 3'

            # Obtener texto y datos de confianza
            data = pytesseract.image_to_data(
                pil_image,
                lang=self.language,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )

            # Paso 4: Procesar resultados
            return self._process_ocr_results(data)

        except Exception as e:
            logger.error(f"Error en OCR: {str(e)}")
            return {
                "text": "",
                "confidence": 0.0,
                "word_count": 0,
                "has_text": False,
                "error": str(e)
            }

    def _process_ocr_results(self, data: Dict) -> Dict:
        """
        Procesa y limpia los resultados del OCR.

        Tesseract devuelve datos que incluyen:
        - Palabras individuales
        - Nivel de confianza por palabra (-1 si no hay texto)
        - Posiciones de cada palabra

        Args:
            data: Diccionario de datos de Tesseract

        Returns:
            Resultados procesados y estructurados
        """
        words = []
        confidences = []

        # Iterar sobre cada detección
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])

            # Solo incluir texto con confianza > 0
            # -1 significa que no hay texto en esa región
            if conf > 0 and text:
                words.append(text)
                confidences.append(conf)

        # Construir texto completo
        full_text = ' '.join(words)

        # Limpiar texto (remover espacios múltiples, caracteres extraños)
        full_text = self._clean_text(full_text)

        # Calcular confianza promedio
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "text": full_text,
            "confidence": round(avg_confidence, 2),
            "word_count": len(words),
            "has_text": len(words) > 0
        }

    def _clean_text(self, text: str) -> str:
        """
        Limpia el texto extraído.

        Operaciones:
        - Remover espacios múltiples
        - Remover caracteres no imprimibles
        - Normalizar saltos de línea

        Args:
            text: Texto a limpiar

        Returns:
            Texto limpio
        """
        import re

        # Remover caracteres no imprimibles excepto espacios y saltos de línea
        text = re.sub(r'[^\x20-\x7E\xA0-\xFF\n]', '', text)

        # Normalizar espacios múltiples a uno solo
        text = re.sub(r' +', ' ', text)

        # Normalizar múltiples saltos de línea
        text = re.sub(r'\n+', '\n', text)

        # Eliminar espacios al inicio/final de cada línea
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()

    def extract_text_simple(self, image: np.ndarray) -> str:
        """
        Versión simplificada que solo devuelve el texto.

        Útil cuando no se necesitan métricas adicionales.

        Args:
            image: Imagen en formato OpenCV

        Returns:
            Texto extraído (string)
        """
        result = self.extract_text(image)
        return result.get("text", "")


# Instancia global del servicio (Singleton pattern)
ocr_service = OCRService() if settings.DEBUG_MODE else None


def get_ocr_service() -> OCRService:
    """
    Factory function para obtener el servicio OCR.

    Permite lazy initialization (inicializar solo cuando se necesita).

    Returns:
        Instancia del servicio OCR
    """
    global ocr_service
    if ocr_service is None:
        ocr_service = OCRService()
    return ocr_service
