"""
============================================================================
NAVIA Backend - Utilidades de Procesamiento de Imágenes
============================================================================
Este módulo contiene funciones auxiliares para el manejo de imágenes.

Funciones principales:
- Validación de imágenes (formato, tamaño)
- Conversión entre formatos (bytes → numpy array)
- Preprocesamiento para mejorar resultados de OCR
- Guardado y carga de imágenes
============================================================================
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException
import uuid
import aiofiles

from app.core.config import settings


async def validate_image(file: UploadFile) -> bool:
    """
    Valida que el archivo subido sea una imagen válida.

    Verificaciones:
    1. Tipo MIME permitido (jpeg, png, webp)
    2. Tamaño dentro del límite (10 MB por defecto)
    3. Contenido realmente es una imagen (no solo extensión)

    Args:
        file: Archivo subido via FastAPI

    Returns:
        True si la imagen es válida

    Raises:
        HTTPException: Si la validación falla
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
            detail=f"Imagen demasiado grande. Tamaño máximo: {max_mb} MB"
        )

    # Resetear posición del archivo para lecturas posteriores
    await file.seek(0)

    # Verificar que el contenido sea realmente una imagen
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


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocesa una imagen para mejorar los resultados del OCR.

    Técnicas aplicadas:
    1. Conversión a escala de grises (reduce complejidad)
    2. Reducción de ruido (filtro bilateral)
    3. Binarización adaptativa (mejora contraste de texto)
    4. Corrección de inclinación (opcional, si está muy inclinada)

    Justificación técnica:
    - El texto es fundamentalmente información de contraste
    - Eliminar color reduce ruido sin perder información textual
    - La binarización crea separación clara entre texto y fondo

    Args:
        image: Imagen en formato OpenCV (BGR)

    Returns:
        Imagen preprocesada optimizada para OCR
    """
    # Paso 1: Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Paso 2: Reducción de ruido
    # El filtro bilateral preserva bordes mientras suaviza áreas uniformes
    # Parámetros: d=11 (diámetro), sigmaColor=17, sigmaSpace=17
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)

    # Paso 3: Binarización adaptativa
    # ADAPTIVE_THRESH_GAUSSIAN_C: umbral basado en suma ponderada gaussiana
    # THRESH_BINARY: píxeles sobre umbral = blanco, bajo = negro
    # blockSize=11: tamaño del vecindario para calcular umbral
    # C=2: constante restada del umbral calculado
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return binary


def resize_image_if_needed(
    image: np.ndarray,
    max_dimension: int = 1920
) -> np.ndarray:
    """
    Redimensiona la imagen si excede las dimensiones máximas.

    Imágenes muy grandes:
    - Aumentan tiempo de procesamiento
    - Consumen más memoria
    - No mejoran significativamente los resultados

    Args:
        image: Imagen en formato OpenCV
        max_dimension: Dimensión máxima permitida (ancho o alto)

    Returns:
        Imagen redimensionada (o la original si no excede el límite)
    """
    height, width = image.shape[:2]

    # Verificar si necesita redimensionamiento
    if max(height, width) <= max_dimension:
        return image

    # Calcular factor de escala
    if width > height:
        scale = max_dimension / width
    else:
        scale = max_dimension / height

    # Calcular nuevas dimensiones
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Redimensionar usando interpolación de área (mejor para reducción)
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
