"""
============================================================================
NAVIA Backend - Endpoints de Análisis de Imágenes
============================================================================
Este módulo contiene todos los endpoints relacionados con el procesamiento
de imágenes: OCR, detección de objetos y descripción de escenas.

Endpoints disponibles:
- POST /analyze/upload: Subir imagen
- POST /analyze/ocr: Extraer texto de imagen
- POST /analyze/objects: Detectar objetos en imagen
- POST /analyze/scene: Análisis completo de la escena

Todas las respuestas son JSON y están diseñadas para ser
procesadas por la aplicación móvil cliente.
============================================================================
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
import numpy as np
import logging

from app.models.schemas import (
    ImageUploadResponse,
    OCRResponse,
    ObjectDetectionResponse,
    SceneDescriptionResponse,
    NavigationResponse,
    ExplorationResponse,
    RiskResponse,
    ErrorResponse,
    DetectedObject
)
from app.utils.image_utils import (
    validate_image,
    save_uploaded_image,
    bytes_to_cv2_image,
    resize_image_if_needed
)
from app.services.ocr_service import get_ocr_service
from app.services.object_detection_service import get_object_detection_service
from app.services.scene_description_service import get_scene_description_service
from app.services.navigation_service import get_navigation_service
from app.services.risk_service import get_risk_service

# Configurar logging
logger = logging.getLogger(__name__)

# Crear router
router = APIRouter(
    prefix="/analyze",
    tags=["Análisis de Imágenes"]
)


# ============================================================================
# ENDPOINT: SUBIR IMAGEN
# ============================================================================

@router.post(
    "/upload",
    response_model=ImageUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subir imagen para análisis",
    description="""
    Sube una imagen al servidor para procesamiento posterior.

    **Formatos soportados:** JPEG, PNG, WebP
    **Tamaño máximo:** 10 MB

    La imagen se almacena temporalmente y se retorna un ID único
    para referencia en operaciones posteriores.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Imagen inválida"},
        413: {"model": ErrorResponse, "description": "Imagen muy grande"}
    }
)
async def upload_image(
    image: UploadFile = File(
        ...,
        description="Archivo de imagen a procesar"
    )
) -> ImageUploadResponse:
    """
    Sube una imagen al servidor.

    Args:
        image: Archivo de imagen (multipart/form-data)

    Returns:
        ImageUploadResponse con ID de imagen y nombre de archivo
    """
    # Validar imagen
    await validate_image(image)

    # Resetear posición después de validación
    await image.seek(0)

    # Guardar imagen
    image_id, filepath = await save_uploaded_image(image)

    logger.info(f"Imagen subida: {image_id} -> {filepath}")

    return ImageUploadResponse(
        success=True,
        message="Imagen subida correctamente",
        image_id=image_id,
        filename=filepath.name
    )


# ============================================================================
# ENDPOINT: OCR (Extracción de Texto)
# ============================================================================

@router.post(
    "/ocr",
    response_model=OCRResponse,
    summary="Extraer texto de imagen (OCR)",
    description="""
    Analiza una imagen y extrae cualquier texto visible.

    **Tecnología:** Tesseract OCR
    **Idiomas soportados:** Español e Inglés

    El texto extraído es ideal para:
    - Leer carteles y señales
    - Leer documentos y etiquetas
    - Identificar nombres de productos

    **Nota:** La precisión depende de la calidad de la imagen
    y legibilidad del texto.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Imagen inválida"}
    }
)
async def extract_text(
    image: UploadFile = File(..., description="Imagen con texto a extraer")
) -> OCRResponse:
    """
    Extrae texto de una imagen usando OCR.

    Args:
        image: Archivo de imagen con texto

    Returns:
        OCRResponse con texto extraído y métricas
    """
    try:
        # Validar imagen
        await validate_image(image)
        await image.seek(0)

        # Leer imagen
        content = await image.read()
        cv2_image = bytes_to_cv2_image(content)

        # Redimensionar si es necesario
        cv2_image = resize_image_if_needed(cv2_image)

        # Ejecutar OCR
        ocr_service = get_ocr_service()
        result = ocr_service.extract_text(cv2_image)

        # Verificar errores
        if "error" in result:
            return OCRResponse(
                success=False,
                message=f"Error en OCR: {result['error']}",
                text="",
                has_text=False
            )

        return OCRResponse(
            success=True,
            message="Texto extraído correctamente" if result["has_text"]
                    else "No se detectó texto en la imagen",
            text=result["text"],
            confidence=result["confidence"],
            word_count=result["word_count"],
            has_text=result["has_text"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en OCR: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )


# ============================================================================
# ENDPOINT: DETECCIÓN DE OBJETOS
# ============================================================================

@router.post(
    "/objects",
    response_model=ObjectDetectionResponse,
    summary="Detectar objetos en imagen",
    description="""
    Identifica y localiza objetos en la imagen.

    **Tecnología:** YOLOv8 (modelo preentrenado)
    **Objetos detectables:** 80 clases (COCO dataset)

    Incluye:
    - Personas, animales, vehículos
    - Muebles, electrónicos, utensilios
    - Alimentos, accesorios, deportes

    Cada objeto incluye:
    - Nombre en español e inglés
    - Nivel de confianza
    - Ubicación en la imagen (bounding box)
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Imagen inválida"}
    }
)
async def detect_objects(
    image: UploadFile = File(..., description="Imagen para detectar objetos")
) -> ObjectDetectionResponse:
    """
    Detecta objetos en una imagen.

    Args:
        image: Archivo de imagen

    Returns:
        ObjectDetectionResponse con lista de objetos detectados
    """
    try:
        # Validar imagen
        await validate_image(image)
        await image.seek(0)

        # Leer imagen
        content = await image.read()
        cv2_image = bytes_to_cv2_image(content)

        # Redimensionar si es necesario
        cv2_image = resize_image_if_needed(cv2_image)

        # Ejecutar detección
        detector = get_object_detection_service()
        result = detector.detect_objects(cv2_image)

        # Verificar errores
        if "error" in result:
            return ObjectDetectionResponse(
                success=False,
                message=f"Error en detección: {result['error']}",
                objects=[],
                object_count=0,
                summary="Error durante la detección de objetos"
            )

        return ObjectDetectionResponse(
            success=True,
            message=f"Se detectaron {result['object_count']} objeto(s)"
                    if result['object_count'] > 0
                    else "No se detectaron objetos",
            objects=result["objects"],
            object_count=result["object_count"],
            summary=result["summary"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en detección: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )


# ============================================================================
# ENDPOINT: DESCRIPCIÓN DE ESCENA (Análisis Completo)
# ============================================================================

@router.post(
    "/scene",
    response_model=SceneDescriptionResponse,
    summary="Análisis completo de escena",
    description="""
    **Endpoint principal para la aplicación móvil.**

    Realiza un análisis completo de la imagen combinando:
    1. **OCR:** Extracción de texto visible
    2. **Detección de objetos:** Identificación de elementos
    3. **Descripción:** Generación de texto natural para TTS

    La respuesta incluye:
    - `description`: Texto listo para conversión a audio (TTS)
    - `detected_text`: Texto encontrado en la imagen
    - `objects`: Lista detallada de objetos detectados

    **Uso recomendado:** Este es el endpoint que debe usar la app
    móvil para obtener una descripción completa que pueda
    convertir a audio para el usuario.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Imagen inválida"},
        500: {"model": ErrorResponse, "description": "Error de procesamiento"}
    }
)
async def analyze_scene(
    image: UploadFile = File(..., description="Imagen a analizar")
) -> SceneDescriptionResponse:
    """
    Realiza un análisis completo de la escena.

    Combina OCR y detección de objetos para generar una
    descripción textual completa, optimizada para TTS.

    Args:
        image: Archivo de imagen a analizar

    Returns:
        SceneDescriptionResponse con descripción completa
    """
    try:
        # Validar imagen
        await validate_image(image)
        await image.seek(0)

        # Leer imagen
        content = await image.read()
        cv2_image = bytes_to_cv2_image(content)

        # Redimensionar si es necesario
        cv2_image = resize_image_if_needed(cv2_image)

        # Obtener servicio de descripción
        scene_service = get_scene_description_service()

        # Analizar escena
        result = scene_service.describe_scene(cv2_image)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analizando escena: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analizando imagen: {str(e)}"
        )


# ============================================================================
# ENDPOINT: ANÁLISIS RÁPIDO (Solo objetos, sin OCR)
# ============================================================================

@router.post(
    "/quick",
    summary="Análisis rápido de objetos",
    description="""
    Versión simplificada que solo detecta objetos (sin OCR).

    **Más rápido** que el análisis completo.
    Útil cuando no se espera texto en la imagen.
    """
)
async def quick_analysis(
    image: UploadFile = File(..., description="Imagen a analizar")
) -> dict:
    """
    Análisis rápido sin OCR.

    Args:
        image: Archivo de imagen

    Returns:
        Diccionario con descripción simplificada
    """
    try:
        # Validar imagen
        await validate_image(image)
        await image.seek(0)

        # Leer imagen
        content = await image.read()
        cv2_image = bytes_to_cv2_image(content)
        cv2_image = resize_image_if_needed(cv2_image)

        # Solo detección de objetos
        detector = get_object_detection_service()
        result = detector.detect_objects(cv2_image)

        return {
            "success": True,
            "description": result["summary"],
            "object_count": result["object_count"],
            "objects": [
                {
                    "name": obj.name_es,
                    "confidence": obj.confidence
                }
                for obj in result["objects"]
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en análisis rápido: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )


# ============================================================================
# NUEVOS MODOS DE NAVIA
# ============================================================================

@router.post(
    "/navegacion",
    response_model=NavigationResponse,
    summary="Modo Navegación - instrucciones de navegación",
    description="""
    Detecta obstáculos y genera instrucciones cortas de navegación.
    Ejemplo: "Camino libre al frente. Silla cerca a la izquierda."
    """
)
async def analyze_navigation(
    image: UploadFile = File(..., description="Imagen para navegación")
) -> NavigationResponse:
    """Modo Navegación: detección de obstáculos + instrucciones espaciales."""
    try:
        await validate_image(image)
        await image.seek(0)

        content = await image.read()
        cv2_image = bytes_to_cv2_image(content)
        cv2_image = resize_image_if_needed(cv2_image)

        # Detectar objetos
        detector = get_object_detection_service()
        result = detector.detect_objects(cv2_image)

        if "error" in result:
            return NavigationResponse(
                success=False,
                message=f"Error: {result['error']}",
                instruction="Error al analizar la imagen.",
            )

        # Generar instrucciones de navegación
        h, w = cv2_image.shape[:2]
        nav_service = get_navigation_service()
        nav_result = nav_service.generate_navigation_summary(
            result["objects"], w, h
        )

        return NavigationResponse(
            success=True,
            message="Navegación analizada",
            instruction=nav_result["instruction"],
            obstacles=nav_result["obstacles"],
            path_clear=nav_result["path_clear"],
            object_count=result["object_count"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en navegación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/exploracion",
    response_model=ExplorationResponse,
    summary="Modo Exploración - descripción del entorno",
    description="""
    Genera una descripción estructurada del entorno.
    Ejemplo: "Hay una mesa frente a ti y una silla a la izquierda."
    """
)
async def analyze_exploration(
    image: UploadFile = File(..., description="Imagen para exploración")
) -> ExplorationResponse:
    """Modo Exploración: descripción estructurada del entorno."""
    try:
        await validate_image(image)
        await image.seek(0)

        content = await image.read()
        cv2_image = bytes_to_cv2_image(content)
        cv2_image = resize_image_if_needed(cv2_image)

        scene_service = get_scene_description_service()
        return scene_service.describe_exploration(cv2_image)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en exploración: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/lectura",
    response_model=OCRResponse,
    summary="Modo Lectura - extracción de texto",
    description="""
    OCR puro para leer documentos, etiquetas y señales.
    """
)
async def analyze_reading(
    image: UploadFile = File(..., description="Imagen con texto a leer")
) -> OCRResponse:
    """Modo Lectura: OCR puro."""
    return await extract_text(image)


@router.post(
    "/riesgo",
    response_model=RiskResponse,
    summary="Modo Riesgo - detección de peligros",
    description="""
    Evalúa peligros en la imagen y genera alertas.
    Ejemplo: "Peligro. Vehículo muy cerca a la izquierda."
    """
)
async def analyze_risk(
    image: UploadFile = File(..., description="Imagen para evaluar riesgo")
) -> RiskResponse:
    """Modo Riesgo: clasificación de peligros y alertas."""
    try:
        await validate_image(image)
        await image.seek(0)

        content = await image.read()
        cv2_image = bytes_to_cv2_image(content)
        cv2_image = resize_image_if_needed(cv2_image)

        # Detectar objetos
        detector = get_object_detection_service()
        result = detector.detect_objects(cv2_image)

        if "error" in result:
            return RiskResponse(
                success=False,
                message=f"Error: {result['error']}",
            )

        # Evaluar riesgo
        h, w = cv2_image.shape[:2]
        risk_service = get_risk_service()
        risk_result = risk_service.evaluate_risk(result["objects"], w)

        return RiskResponse(
            success=True,
            message="Riesgo evaluado",
            has_danger=risk_result["has_danger"],
            priority=risk_result["priority"],
            alert_text=risk_result["alert_text"],
            dangers=risk_result["dangers"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en evaluación de riesgo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
