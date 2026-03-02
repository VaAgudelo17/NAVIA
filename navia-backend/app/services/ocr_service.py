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
    resize_image_if_needed,
    normalize_image_for_ocr,
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
        Extrae texto de una imagen, adaptándose a cualquier tamaño.

        Pipeline adaptativo multi-estrategia:
        1. Normalizar tamaño al rango óptimo para OCR (upscale o downscale)
        2. Preprocesar → obtener versión grayscale mejorada Y binaria
        3. Probar OCR en este orden (escoger el mejor resultado):
           a) Grayscale mejorada con PSM 3 (auto page segmentation)
           b) Grayscale mejorada con PSM 6 (uniform text block)
           c) Binaria con PSM 3
           d) Binaria con PSM 6
           e) Imagen a color original (sin preprocesamiento)

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
            h, w = image.shape[:2]
            logger.info(f"[OCR] Imagen original: {w}x{h}")

            # Paso 1: Normalizar tamaño (upscale pequeñas, downscale grandes)
            normalized = normalize_image_for_ocr(image)

            best_result = None

            def _try_ocr(pil_img: Image.Image, psm: int, label: str) -> Optional[Dict]:
                """Intenta OCR con una configuración dada."""
                nonlocal best_result
                config = f'--oem 3 --psm {psm}'
                try:
                    data = pytesseract.image_to_data(
                        pil_img,
                        lang=self.language,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    result = self._process_ocr_results(data)
                    wc = result.get("word_count", 0)
                    conf = result.get("confidence", 0)

                    if wc > 0:
                        logger.info(f"[OCR] {label}: {wc} palabras, {conf:.0f}% conf")

                    if best_result is None:
                        best_result = result
                    elif self._is_better(result, best_result):
                        best_result = result
                        logger.info(f"[OCR] {label} es el mejor resultado hasta ahora")

                    return result
                except Exception as e:
                    logger.debug(f"[OCR] {label} falló: {e}")
                    return None

            # Paso 2: Preprocesar → grayscale mejorada + binaria
            if preprocess:
                grayscale_enhanced, binary = preprocess_for_ocr(normalized)
                pil_gray = Image.fromarray(grayscale_enhanced)
                pil_binary = Image.fromarray(binary)

                # Estrategia A: Grayscale mejorada (mejor para fotos de celular)
                _try_ocr(pil_gray, 3, "Grayscale PSM3")
                _try_ocr(pil_gray, 6, "Grayscale PSM6")

                # Estrategia B: Binaria (mejor para documentos escaneados)
                _try_ocr(pil_binary, 3, "Binary PSM3")
                _try_ocr(pil_binary, 6, "Binary PSM6")

            # Estrategia C: Imagen a color original (fallback)
            # Probar si no hubo preprocesamiento o si resultados son pobres
            if not preprocess or (best_result and best_result.get("word_count", 0) < 3):
                logger.info("[OCR] Intentando con imagen a color...")
                pil_color = cv2_image_to_pil(normalized)
                _try_ocr(pil_color, 3, "Color PSM3")

            # Agregar dimensiones de la imagen normalizada al resultado
            nh, nw = normalized.shape[:2]
            if best_result:
                best_result["image_width"] = nw
                best_result["image_height"] = nh
            return best_result or {
                "text": "",
                "confidence": 0.0,
                "word_count": 0,
                "has_text": False,
                "layout": {"word_boxes": [], "num_blocks": 0, "num_lines": 0},
                "image_width": nw,
                "image_height": nh,
            }

        except Exception as e:
            logger.error(f"Error en OCR: {str(e)}")
            return {
                "text": "",
                "confidence": 0.0,
                "word_count": 0,
                "has_text": False,
                "error": str(e)
            }

    def _is_better(self, candidate: Dict, current_best: Dict) -> bool:
        """
        Compara dos resultados OCR y decide si el candidato es mejor.
        Prioriza CONFIANZA sobre cantidad de palabras para evitar basura.
        """
        c_wc = candidate.get("word_count", 0)
        b_wc = current_best.get("word_count", 0)
        c_conf = candidate.get("confidence", 0)
        b_conf = current_best.get("confidence", 0)

        # Calcular un score combinado: confianza pesa más que cantidad
        # Esto evita que 262 palabras al 55% gane sobre 100 palabras al 85%
        c_score = c_wc * (c_conf / 100.0) ** 2
        b_score = b_wc * (b_conf / 100.0) ** 2

        # El candidato necesita un score claramente mejor
        if c_score > b_score * 1.1 and c_conf > 30:
            return True

        # Si misma cantidad, mayor confianza gana
        if c_wc == b_wc and c_conf > b_conf:
            return True

        return False

    def _process_ocr_results(self, data: Dict) -> Dict:
        """
        Procesa y limpia los resultados del OCR.

        Tesseract devuelve datos que incluyen:
        - Palabras individuales con nivel de confianza
        - Posiciones (bounding boxes) de cada palabra
        - Estructura jerárquica: page → block → paragraph → line → word

        Preserva datos espaciales para análisis de layout posterior.

        Args:
            data: Diccionario de datos de Tesseract (image_to_data)

        Returns:
            Resultados procesados con texto, confianza, y datos estructurales
        """
        import re

        words = []
        confidences = []

        # Datos estructurales preservados para el clasificador
        word_boxes = []      # {text, conf, left, top, width, height, block, line}
        block_set = set()
        line_set = set()

        # Iterar sobre cada detección
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])

            # Filtrar basura de Tesseract:
            # - confianza mínima 30% (< 30% es casi siempre basura de iconos/UI)
            # - ignorar "palabras" de 1 caracter suelto (excepto números y letras comunes)
            # - ignorar secuencias de solo símbolos/puntuación
            if conf < 30 or not text:
                continue

            # Permitir palabras de 1 char solo si son significativas
            if len(text) == 1 and text not in 'aAeEiIoOuUyY0123456789':
                continue

            # Ignorar texto que es solo puntuación/símbolos
            if re.match(r'^[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑüÜ]+$', text):
                continue

            words.append(text)
            confidences.append(conf)

            # Preservar datos espaciales
            block_num = int(data['block_num'][i])
            line_num = int(data['line_num'][i])
            block_set.add(block_num)
            line_set.add((block_num, line_num))

            word_boxes.append({
                "text": text,
                "conf": conf,
                "left": int(data['left'][i]),
                "top": int(data['top'][i]),
                "width": int(data['width'][i]),
                "height": int(data['height'][i]),
                "block": block_num,
                "par": int(data['par_num'][i]),
                "line": line_num,
            })

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
            "has_text": len(words) > 0,
            # Datos estructurales para el clasificador jerárquico
            "layout": {
                "word_boxes": word_boxes,
                "num_blocks": len(block_set),
                "num_lines": len(line_set),
            },
        }

    def _clean_text(self, text: str) -> str:
        """
        Limpia el texto extraído.

        Operaciones:
        - Remover caracteres de control (excepto newlines y tabs)
        - Normalizar espacios múltiples
        - Normalizar saltos de línea

        Args:
            text: Texto a limpiar

        Returns:
            Texto limpio
        """
        import re
        import unicodedata

        # Remover caracteres de control excepto \n y \t
        # Preservar TODOS los caracteres imprimibles Unicode (acentos, ñ, ü, etc.)
        cleaned = []
        for ch in text:
            if ch in ('\n', '\t'):
                cleaned.append(ch)
            elif unicodedata.category(ch).startswith('C'):
                # Control character - skip
                continue
            else:
                cleaned.append(ch)
        text = ''.join(cleaned)

        # Normalizar tabs a espacios
        text = text.replace('\t', ' ')

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
