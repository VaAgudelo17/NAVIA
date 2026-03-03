"""
============================================================================
NAVIA Backend - Utilidades de Procesamiento de Imágenes
============================================================================
Este módulo contiene funciones auxiliares para el manejo de imágenes.

Funciones principales:
- Validación de imágenes (formato, tamaño)
- Conversión entre formatos (bytes → numpy array)
- Preprocesamiento para mejorar resultados de OCR
- Análisis de calidad de imagen (blur, brillo, contraste, inclinación)
- Guardado y carga de imágenes
============================================================================
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field
from fastapi import UploadFile, HTTPException
import uuid
import logging
import aiofiles

from app.core.config import settings

logger = logging.getLogger(__name__)


async def validate_image(file: UploadFile) -> bool:
    """
    Valida que el archivo subido sea válido para procesamiento.
    Acepta cualquier tipo de imagen o PDF.
    """
    # Verificar tipo MIME
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido: {file.content_type}. "
                   f"Tipos permitidos: {settings.ALLOWED_IMAGE_TYPES}"
        )

    # Leer contenido para verificar tamaño
    content = await file.read()

    # Verificar tamaño
    if len(content) > settings.MAX_IMAGE_SIZE:
        max_mb = settings.MAX_IMAGE_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"Archivo demasiado grande. Tamaño máximo: {max_mb} MB"
        )

    # Resetear posición del archivo para lecturas posteriores
    await file.seek(0)

    return True

    # Para imágenes: verificar que el contenido sea realmente una imagen
    try:
        image = Image.open(BytesIO(content))
        image.verify()  # Verifica integridad sin cargar completamente
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="El archivo no es una imagen válida"
        )

    return True


async def save_uploaded_image(file: UploadFile) -> Tuple[str, Path]:
    """
    Guarda una imagen subida en el sistema de archivos.

    Genera un identificador único (UUID) para evitar colisiones
    y mantener la privacidad del nombre original.

    Args:
        file: Archivo de imagen subido

    Returns:
        Tuple con (image_id, ruta_completa_del_archivo)
    """
    # Generar ID único
    image_id = str(uuid.uuid4())

    # Obtener extensión original
    extension = Path(file.filename).suffix.lower()
    if not extension:
        extension = ".jpg"  # Extensión por defecto

    # Construir nombre de archivo
    filename = f"{image_id}{extension}"
    filepath = settings.UPLOAD_DIR / filename

    # Guardar archivo de forma asíncrona
    content = await file.read()
    async with aiofiles.open(filepath, 'wb') as f:
        await f.write(content)

    return image_id, filepath


def bytes_to_cv2_image(image_bytes: bytes) -> np.ndarray:
    """
    Convierte bytes de imagen a formato OpenCV (numpy array).

    OpenCV usa el formato BGR (Blue-Green-Red) internamente,
    que es diferente al RGB estándar.

    Args:
        image_bytes: Imagen en formato bytes

    Returns:
        Imagen como numpy array en formato BGR
    """
    # Convertir bytes a array numpy
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decodificar imagen usando OpenCV
    # cv2.IMREAD_COLOR carga la imagen a color (3 canales)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("No se pudo decodificar la imagen")

    return image


def cv2_image_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convierte imagen de OpenCV a formato PIL.

    Necesario porque pytesseract trabaja mejor con imágenes PIL.

    Args:
        cv2_image: Imagen en formato OpenCV (numpy array BGR)

    Returns:
        Imagen en formato PIL (RGB)
    """
    # Convertir de BGR a RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Convertir a PIL
    return Image.fromarray(rgb_image)


def preprocess_for_ocr(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesa una imagen para mejorar los resultados del OCR.
    Se adapta automáticamente al tamaño y condiciones de la imagen.

    Retorna DOS versiones:
    - Grayscale mejorada (CLAHE + denoising + deskew): mejor para fotos con
      fondos de color, texto sobre fondos variados, fotos de celular
    - Binaria (binarización adaptativa): mejor para documentos escaneados
      con fondo blanco y texto negro

    Pipeline adaptativo:
    1. Escala de grises
    2. CLAHE (ecualización adaptativa de histograma) para iluminación desigual
    3. Reducción de ruido adaptativa al tamaño
    4. Corrección de inclinación (deskew) automática
    5. Binarización adaptativa con blockSize proporcional (versión binaria)

    Args:
        image: Imagen en formato OpenCV (BGR)

    Returns:
        Tupla (grayscale_enhanced, binary):
        - grayscale_enhanced: imagen en escala de grises mejorada
        - binary: imagen binarizada (blanco y negro puro)
    """
    height, width = image.shape[:2]

    # Paso 1: Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Paso 2: CLAHE - corrige iluminación desigual (sombras parciales, flash)
    # clipLimit y tileGridSize se adaptan al tamaño de la imagen
    tile_size = max(4, min(16, min(height, width) // 100))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_size, tile_size))
    equalized = clahe.apply(gray)

    # Paso 3: Reducción de ruido adaptativa
    # Imágenes pequeñas necesitan filtro más suave para no perder detalle
    # Imágenes grandes toleran filtro más agresivo
    min_dim = min(height, width)
    if min_dim < 500:
        # Imagen pequeña: filtro suave
        denoised = cv2.bilateralFilter(equalized, 5, 10, 10)
    elif min_dim < 1200:
        # Imagen mediana: filtro estándar
        denoised = cv2.bilateralFilter(equalized, 7, 12, 12)
    else:
        # Imagen grande: filtro moderado (no agresivo, preservar detalle)
        denoised = cv2.bilateralFilter(equalized, 9, 15, 15)

    # Paso 4: Deskew - corrección automática de inclinación
    denoised = _auto_deskew(denoised)

    # --- Versión grayscale mejorada (para fotos de celular) ---
    grayscale_enhanced = denoised

    # --- Versión binaria (para documentos escaneados) ---
    # blockSize debe ser impar y proporcional a la resolución de la imagen
    block_size = max(11, min(51, min_dim // 30))
    if block_size % 2 == 0:
        block_size += 1  # Debe ser impar

    # C (constante): más alta para imágenes con bajo contraste
    contrast = float(np.std(denoised))
    c_value = 6 if contrast < 40 else 3

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value
    )

    return grayscale_enhanced, binary


def _auto_deskew(gray: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Corrige automáticamente la inclinación de un documento.
    Detecta el ángulo predominante de las líneas de texto y rota la imagen.

    Args:
        gray: Imagen en escala de grises
        max_angle: Ángulo máximo a corregir (grados). Si la inclinación
                   es mayor, asumimos que no es un documento rotado.

    Returns:
        Imagen corregida (o la original si no se detecta inclinación significativa)
    """
    try:
        # Detectar bordes
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detectar líneas con Hough
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=80,
            minLineLength=gray.shape[1] // 8,  # Mínimo 1/8 del ancho
            maxLineGap=10
        )

        if lines is None or len(lines) < 3:
            return gray  # No suficientes líneas para estimar inclinación

        # Calcular ángulos de líneas casi horizontales
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, dx))
            if -max_angle <= angle <= max_angle:
                angles.append(angle)

        if len(angles) < 3:
            return gray

        # Mediana (robusta a outliers)
        median_angle = float(np.median(angles))

        # Solo corregir si la inclinación es significativa (> 0.5°)
        if abs(median_angle) < 0.5:
            return gray

        logger.info(f"[Deskew] Corrigiendo inclinación de {median_angle:.1f}°")

        # Rotar la imagen
        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # Calcular nuevo tamaño para no recortar
        cos_a = abs(rotation_matrix[0, 0])
        sin_a = abs(rotation_matrix[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)

        # Ajustar la traslación
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(
            gray, rotation_matrix, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    except Exception as e:
        logger.debug(f"[Deskew] No se pudo corregir inclinación: {e}")
        return gray


def normalize_image_for_ocr(
    image: np.ndarray,
    min_dimension: int = 800,
    max_dimension: int = 3000,
    optimal_dpi_range: Tuple[int, int] = (250, 350),
) -> np.ndarray:
    """
    Normaliza el tamaño de la imagen al rango óptimo para OCR.

    Tesseract funciona mejor con texto a ~300 DPI equivalente.
    Imágenes muy pequeñas se escalan hacia arriba.
    Imágenes muy grandes se reducen.

    A diferencia de resize_image_if_needed, esta función:
    - TAMBIÉN escala imágenes pequeñas hacia arriba (upscale)
    - Apunta a un rango óptimo, no solo a un máximo
    - Usa interpolación apropiada para cada dirección

    Args:
        image: Imagen en formato OpenCV
        min_dimension: Dimensión mínima deseada (lado menor)
        max_dimension: Dimensión máxima permitida (lado mayor)

    Returns:
        Imagen en el rango óptimo para OCR
    """
    height, width = image.shape[:2]
    current_min = min(height, width)
    current_max = max(height, width)

    scale = 1.0

    if current_min < min_dimension:
        # Imagen pequeña: escalar hacia arriba
        scale = min_dimension / current_min
        # Pero no exceder el máximo
        if current_max * scale > max_dimension:
            scale = max_dimension / current_max
        logger.info(f"[OCR Resize] Upscaling {width}x{height} → scale={scale:.2f}")
    elif current_max > max_dimension:
        # Imagen grande: reducir
        scale = max_dimension / current_max
        logger.info(f"[OCR Resize] Downscaling {width}x{height} → scale={scale:.2f}")

    if abs(scale - 1.0) < 0.05:
        return image  # No necesita cambio (<5% diferencia)

    new_width = int(width * scale)
    new_height = int(height * scale)

    # Interpolación: CUBIC para upscale (suaviza), AREA para downscale (preserva)
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA

    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    logger.info(f"[OCR Resize] Resultado: {new_width}x{new_height}")
    return resized


def resize_image_if_needed(
    image: np.ndarray,
    max_dimension: int = 1920
) -> np.ndarray:
    """
    Redimensiona la imagen si excede las dimensiones máximas.
    Para uso general (no OCR). Para OCR usar normalize_image_for_ocr().

    Args:
        image: Imagen en formato OpenCV
        max_dimension: Dimensión máxima permitida (ancho o alto)

    Returns:
        Imagen redimensionada (o la original si no excede el límite)
    """
    height, width = image.shape[:2]

    if max(height, width) <= max_dimension:
        return image

    if width > height:
        scale = max_dimension / width
    else:
        scale = max_dimension / height

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


def get_image_info(image: np.ndarray) -> dict:
    """
    Obtiene información básica de una imagen.

    Args:
        image: Imagen en formato OpenCV

    Returns:
        Diccionario con dimensiones y otras propiedades
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "total_pixels": width * height,
        "aspect_ratio": round(width / height, 2)
    }


# ============================================================================
# ANÁLISIS DE CALIDAD DE IMAGEN
# ============================================================================

@dataclass
class ImageQualityIssue:
    """Un problema detectado en la calidad de la imagen."""
    code: str               # blur, dark, bright, low_contrast, skewed, low_res
    severity: str           # warning, critical
    message_es: str         # Mensaje para el usuario en español (TTS-friendly)
    score: float            # 0.0 (pésimo) - 1.0 (perfecto) para este aspecto


@dataclass
class ImageQualityReport:
    """Resultado completo del análisis de calidad."""
    overall_score: float = 1.0          # 0.0 - 1.0
    is_acceptable: bool = True          # Si la imagen es usable para OCR
    issues: List[ImageQualityIssue] = field(default_factory=list)
    feedback_text: str = ""             # Texto combinado para TTS
    metrics: Dict[str, float] = field(default_factory=dict)  # Valores numéricos

    @property
    def has_critical(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)


class ImageQualityAnalyzer:
    """
    Analiza la calidad de una imagen para OCR y genera feedback
    hablado para usuarios ciegos.

    Métricas analizadas:
    - Blur (nitidez): varianza del Laplaciano
    - Brillo: media del canal de luminosidad
    - Contraste: desviación estándar del canal de luminosidad
    - Inclinación: ángulo de rotación detectado via Hough lines
    - Resolución: píxeles totales
    """

    # --- Umbrales calibrados para fotos de celular ---
    BLUR_THRESHOLD_CRITICAL = 15.0    # Laplacian variance < 15 = completamente ilegible
    BLUR_THRESHOLD_WARNING = 30.0     # No se usa (solo critical)
    DARK_THRESHOLD_CRITICAL = 25.0    # Media de brillo < 25 = completamente oscuro
    DARK_THRESHOLD_WARNING = 50.0     # No se usa (solo critical)
    BRIGHT_THRESHOLD_CRITICAL = 248.0 # Media de brillo > 248 = sobreexpuesto (muy permisivo para screenshots/fondos blancos)
    BRIGHT_THRESHOLD_WARNING = 240.0  # No se usa (solo critical)
    CONTRAST_THRESHOLD_CRITICAL = 10.0  # StdDev < 10 = completamente sin contraste
    CONTRAST_THRESHOLD_WARNING = 20.0   # No se usa (solo critical)
    SKEW_THRESHOLD_WARNING = 5.0      # Ángulo > 5° = algo inclinado
    SKEW_THRESHOLD_CRITICAL = 15.0    # > 15° = muy inclinado
    MIN_RESOLUTION = 640 * 480        # Mínimo para OCR decente

    def analyze(self, image: np.ndarray) -> ImageQualityReport:
        """
        Ejecuta análisis de calidad simplificado para usuarios ciegos.

        Solo reporta problemas CRÍTICOS (imagen ilegible).
        No reporta warnings ni inclinación — un usuario ciego
        no puede corregir esos detalles.
        """
        report = ImageQualityReport()
        issues: List[ImageQualityIssue] = []

        # Convertir a gris una sola vez
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # --- 1. Blur (nitidez) — solo critical ---
        blur_score, blur_value = self._check_blur(gray)
        report.metrics["blur_variance"] = blur_value
        report.metrics["blur_score"] = blur_score
        if blur_value < self.BLUR_THRESHOLD_CRITICAL:
            issues.append(ImageQualityIssue(
                code="blur", severity="critical", score=blur_score,
                message_es="No pude leer. Intenta tomar otra foto."
            ))

        # --- 2. Brillo — solo critical (muy oscuro o sobreexpuesto) ---
        brightness = float(np.mean(gray))
        report.metrics["brightness"] = brightness
        if brightness < self.DARK_THRESHOLD_CRITICAL:
            bright_score = brightness / self.DARK_THRESHOLD_WARNING
            issues.append(ImageQualityIssue(
                code="dark", severity="critical", score=max(bright_score, 0.0),
                message_es="No pude leer, la imagen está muy oscura. Intenta tomar otra foto con más luz."
            ))
        elif brightness > self.BRIGHT_THRESHOLD_CRITICAL:
            bright_score = max(1.0 - (brightness - self.BRIGHT_THRESHOLD_WARNING) / 45.0, 0.0)
            issues.append(ImageQualityIssue(
                code="bright", severity="critical", score=bright_score,
                message_es="No pude leer, hay demasiada luz. Intenta tomar otra foto evitando reflejos."
            ))

        # --- 3. Contraste — solo critical ---
        contrast = float(np.std(gray))
        report.metrics["contrast"] = contrast
        if contrast < self.CONTRAST_THRESHOLD_CRITICAL:
            contrast_score = contrast / self.CONTRAST_THRESHOLD_WARNING
            issues.append(ImageQualityIssue(
                code="low_contrast", severity="critical", score=max(contrast_score, 0.0),
                message_es="No pude leer, no se distingue el texto. Intenta tomar otra foto."
            ))

        # --- 4. Resolución — solo si es extremadamente baja ---
        height, width = gray.shape[:2]
        total_pixels = width * height
        report.metrics["resolution"] = float(total_pixels)
        report.metrics["width"] = float(width)
        report.metrics["height"] = float(height)

        # --- Calcular score global ---
        report.issues = issues
        if issues:
            scores = [i.score for i in issues]
            report.overall_score = round(sum(scores) / len(scores), 2)
        else:
            report.overall_score = 1.0

        report.is_acceptable = not report.has_critical
        report.feedback_text = self._build_feedback(issues)

        logger.info(
            f"[ImageQuality] score={report.overall_score:.2f}, "
            f"acceptable={report.is_acceptable}, "
            f"issues={[i.code for i in issues]}, "
            f"blur={report.metrics.get('blur_variance', 0):.1f}, "
            f"brightness={brightness:.1f}, contrast={contrast:.1f}"
        )

        return report

    def _check_blur(self, gray: np.ndarray) -> Tuple[float, float]:
        """
        Detecta blur usando la varianza del Laplaciano.
        Mayor varianza = imagen más nítida.
        """
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalizar a 0-1 (100 = excelente nitidez)
        score = min(laplacian_var / 150.0, 1.0)
        return round(score, 2), round(laplacian_var, 2)

    def _detect_skew(self, gray: np.ndarray) -> float:
        """
        Detecta inclinación del documento usando transformada de Hough.
        Retorna ángulo en grados (-45 a +45).
        """
        try:
            # Detectar bordes
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detectar líneas con Hough
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=80,
                minLineLength=gray.shape[1] // 6,  # Mínimo 1/6 del ancho
                maxLineGap=10
            )

            if lines is None or len(lines) == 0:
                return 0.0

            # Calcular ángulos de todas las líneas
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Solo considerar líneas casi horizontales (-45° a 45°)
                if -45 <= angle <= 45:
                    angles.append(angle)

            if not angles:
                return 0.0

            # Mediana de ángulos (robusta a outliers)
            return round(float(np.median(angles)), 1)

        except Exception as e:
            logger.debug(f"Skew detection falló: {e}")
            return 0.0

    def _build_feedback(self, issues: List[ImageQualityIssue]) -> str:
        """
        Construye texto de feedback para TTS.
        Para usuarios ciegos: un solo mensaje claro si hay problema crítico.
        Sin warnings — solo habla cuando la foto no sirve.
        """
        if not issues:
            return ""

        # Solo usar el primer problema crítico (mensaje más relevante)
        critical = [i for i in issues if i.severity == "critical"]
        if critical:
            return critical[0].message_es

        return ""


# Singleton
_image_quality_analyzer: Optional[ImageQualityAnalyzer] = None


def get_image_quality_analyzer() -> ImageQualityAnalyzer:
    global _image_quality_analyzer
    if _image_quality_analyzer is None:
        _image_quality_analyzer = ImageQualityAnalyzer()
    return _image_quality_analyzer
