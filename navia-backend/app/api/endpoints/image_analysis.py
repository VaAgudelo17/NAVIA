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

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
import numpy as np
import time
import logging

from app.core.config import settings

from app.models.schemas import (
    ImageUploadResponse,
    OCRResponse,
    ObjectDetectionResponse,
    SceneDescriptionResponse,
    NavigationResponse,
    ExplorationResponse,
    RiskResponse,
    SmartReadingResponse,
    ErrorResponse,
    DetectedObject,
    BarcodeResponse,
    DetectedCode
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
from app.services.navigation_guidance_service import get_navigation_guidance_service
from app.services.smart_reading_service import get_smart_reading_service
from app.services.exploration_service import get_exploration_service
from app.services.history_service import save_to_history
from app.services.barcode_service import get_barcode_reader

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
    summary="Modo Navegación - navegación asistida con evaluación de riesgo",
    description="""
    Pipeline unificado de navegación + riesgo para movilidad peatonal.
    
    **Pipeline completo:**
    1. Detección de objetos (YOLO-World + Depth Anything)
    2. Filtrado de clases relevantes para caminata
    3. Clasificación de altura (suelo/cuerpo/cabeza)
    4. Análisis de movimiento (se acerca/aleja/estático)
    5. Scoring de riesgo combinado
    6. Instrucciones priorizadas para TTS
    
    **Ejemplo:** "Cuidado: una bicicleta se acerca por la derecha.
    Hay un bordillo muy cerca frente a ti."
    """
)
async def analyze_navigation(
    image: UploadFile = File(..., description="Imagen para navegación"),
    background_tasks: BackgroundTasks = None,
) -> NavigationResponse:
    """Modo Navegación: pipeline unificado de navegación + riesgo."""
    try:
        start_time = time.time()

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

        # Pipeline unificado: navegación + riesgo
        h, w = cv2_image.shape[:2]
        guidance_service = get_navigation_guidance_service()

        # Calcular profundidad direccional si el mapa está disponible
        directional_depth = None
        depth_map = result.get("depth_map")
        if depth_map is not None:
            from app.services.depth_estimation_service import get_depth_estimation_service
            directional_depth = get_depth_estimation_service().compute_directional_depth(depth_map)

        guidance = guidance_service.generate_guidance(
            result["objects"], w, h,
            track_movement=False,  # HTTP no tiene frames consecutivos
            directional_depth=directional_depth,
        )

        processing_time = (time.time() - start_time) * 1000

        response = NavigationResponse(
            success=True,
            message="Navegación analizada",
            instruction=guidance["instruction"],
            obstacles=guidance["obstacles"],
            path_clear=guidance["path_clear"],
            has_danger=guidance["has_danger"],
            priority=guidance["priority"],
            obstacle_details=guidance["obstacle_details"],
            object_count=result["object_count"],
        )

        # Guardar en historial (background, no bloquea la respuesta)
        if background_tasks:
            background_tasks.add_task(
                save_to_history,
                mode="navegacion",
                result_data=response.model_dump(),
                result_summary=guidance["instruction"][:200],
                processing_time_ms=processing_time,
                object_count=result["object_count"],
                has_danger=guidance["has_danger"],
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en navegación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/exploracion",
    response_model=ExplorationResponse,
    summary="Modo Exploración - descripción del entorno (mejorado)",
    description="""
    Genera una descripción estructurada del entorno optimizada para accesibilidad.
    
    **Mejoras implementadas:**
    - Filtrado semántico (ignora objetos irrelevantes como piso, cielo, paredes)
    - Priorización inteligente (score = confianza × tamaño × cercanía al centro)
    - Estimación de distancia basada en tamaño del objeto
    - Posición espacial completa (horizontal + vertical)
    - Extracción de color dominante del objeto principal
    - Filtro inteligente de OCR (evita ruido visual)
    - Máximo 3 objetos + 1 bloque de texto (evita sobrecarga cognitiva)
    
    **Ejemplo de salida:**
    "Un gato está frente a ti, muy cerca, a nivel del suelo. Es de color naranja."
    """
)
async def analyze_exploration(
    image: UploadFile = File(..., description="Imagen para exploración"),
    background_tasks: BackgroundTasks = None,
) -> ExplorationResponse:
    """Modo Exploración mejorado: descripción estructurada con priorización inteligente."""
    try:
        start_time = time.time()

        await validate_image(image)
        await image.seek(0)

        content = await image.read()
        cv2_image = bytes_to_cv2_image(content)
        cv2_image = resize_image_if_needed(cv2_image)

        # Usar el nuevo servicio de exploración mejorado
        exploration_service = get_exploration_service()
        response = exploration_service.analyze(cv2_image)

        processing_time = (time.time() - start_time) * 1000

        # Guardar en historial
        if background_tasks:
            background_tasks.add_task(
                save_to_history,
                mode="exploracion",
                result_data=response.model_dump() if hasattr(response, 'model_dump') else response,
                result_summary=response.description[:200] if hasattr(response, 'description') else "",
                processing_time_ms=processing_time,
                object_count=response.object_count if hasattr(response, 'object_count') else 0,
                has_text=response.has_text if hasattr(response, 'has_text') else False,
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en exploración: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/lectura",
    response_model=SmartReadingResponse,
    summary="Modo Lectura Inteligente",
    description="""
    Lectura inteligente de documentos e imágenes con texto.
    Detecta automáticamente el tipo de documento y genera la narrativa óptima.
    
    **Formatos soportados:** JPEG, PNG, WebP, PDF (primera página)
    
    **Tipos detectados:** factura, recibo, carta, formulario, documento informativo, etiqueta, menú, tarjeta, imagen con texto, desconocido
    """
)
async def analyze_reading(
    image: UploadFile = File(..., description="Imagen o PDF con texto a leer"),
    background_tasks: BackgroundTasks = None,
) -> SmartReadingResponse:
    """Modo Lectura Inteligente: OCR + clasificación + narrativa automática."""
    try:
        start_time = time.time()

        content = await image.read()
        
        # Detectar tipo de archivo y procesar
        if image.content_type == "application/pdf":
            # Convertir PDF a imagen
            from app.utils.pdf_utils import convert_pdf_first_page_to_image
            cv2_image = convert_pdf_first_page_to_image(content)
        else:
            # Es imagen
            await validate_image(image)
            await image.seek(0)
            cv2_image = bytes_to_cv2_image(content)
        
        # NO redimensionar aquí: el OCR service normaliza internamente

        # Ejecutar lectura inteligente (modo automático basado en tipo de documento)
        smart_service = get_smart_reading_service()
        result = smart_service.analyze(cv2_image)

        processing_time = (time.time() - start_time) * 1000

        # Generar audio TTS inline para eliminar el segundo round-trip en la app
        audio_b64: Optional[str] = None
        audio_fmt: Optional[str] = None
        narrative_text = result["narrative"]
        if narrative_text and settings.TTS_ENABLED:
            try:
                import base64
                from app.services.tts_service import get_tts_service
                tts = get_tts_service()
                audio_bytes = tts.synthesize(narrative_text)
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                audio_fmt = "mp3" if settings.TTS_BACKEND == "gtts" else "wav"
            except Exception as tts_err:
                logger.warning(f"[Lectura] TTS inline falló, la app usará endpoint TTS: {tts_err}")

        response = SmartReadingResponse(
            success=True,
            message="Lectura inteligente completada",
            narrative=narrative_text,
            document_type=result["document_type"],
            document_type_label=result["document_type_label"],
            reading_mode="auto",
            raw_text=result["raw_text"],
            has_text=result["has_text"],
            ocr_confidence=result["ocr_confidence"],
            word_count=result["word_count"],
            extracted_fields=result["extracted_fields"],
            visual_caption=result.get("visual_caption"),
            image_quality=result.get("image_quality"),
            classification=result.get("classification"),
            audio_base64=audio_b64,
            audio_format=audio_fmt,
        )

        # Guardar en historial
        if background_tasks:
            background_tasks.add_task(
                save_to_history,
                mode="lectura",
                result_data=response.model_dump(),
                result_summary=result["narrative"][:200],
                reading_mode="auto",
                processing_time_ms=processing_time,
                has_text=result["has_text"],
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en lectura inteligente: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/riesgo",
    response_model=RiskResponse,
    summary="Modo Riesgo - detección de peligros (unificado con navegación)",
    description="""
    Evalúa peligros en la imagen usando el pipeline unificado de navegación.
    
    **Nota:** Este endpoint ahora usa el mismo pipeline que /navegacion.
    Se mantiene por compatibilidad. La respuesta incluye los mismos campos
    de navegación + riesgo combinados.
    
    Ejemplo: "Cuidado: Carro muy cerca frente a ti."
    """
)
async def analyze_risk(
    image: UploadFile = File(..., description="Imagen para evaluar riesgo"),
    background_tasks: BackgroundTasks = None,
) -> RiskResponse:
    """Modo Riesgo: usa pipeline unificado de navegación + riesgo."""
    try:
        start_time = time.time()

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

        # Pipeline unificado: navegación + riesgo
        h, w = cv2_image.shape[:2]
        guidance_service = get_navigation_guidance_service()
        guidance = guidance_service.generate_guidance(
            result["objects"], w, h,
            track_movement=False,
        )

        processing_time = (time.time() - start_time) * 1000

        response = RiskResponse(
            success=True,
            message="Riesgo evaluado",
            instruction=guidance["instruction"],
            obstacles=guidance["obstacles"],
            path_clear=guidance["path_clear"],
            has_danger=guidance["has_danger"],
            priority=guidance["priority"],
            obstacle_details=guidance["obstacle_details"],
            object_count=result["object_count"],
        )

        # Guardar en historial
        if background_tasks:
            background_tasks.add_task(
                save_to_history,
                mode="navegacion",
                result_data=response.model_dump(),
                result_summary=guidance["instruction"][:200],
                processing_time_ms=processing_time,
                has_danger=guidance["has_danger"],
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en evaluación de riesgo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINT: CÓDIGOS DE BARRAS / QR
# ============================================================================

@router.post(
    "/barcode",
    response_model=BarcodeResponse,
    summary="Detectar códigos QR y de barras",
    description="""
    Detecta y decodifica códigos QR y códigos de barras en una imagen.

    **Códigos soportados:**
    - Códigos QR
    - EAN-13, EAN-8
    - UPC-A, UPC-E
    - Code 128, Code 39, Code 93
    - ITF, Codabar

    **Para códigos QR detecta:**
    - URLs (indica el dominio)
    - Redes WiFi (SSID)
    - Contactos (vCard)
    - Geolocalización
    - Textos genéricos

    **Para códigos de barras:**
    - Códigos de producto (EAN/UPC)
    - Indica el país de origen según el código
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Imagen inválida"}
    }
)
async def read_barcodes(
    image: UploadFile = File(..., description="Imagen con códigos QR o de barras")
) -> BarcodeResponse:
    """
    Lee códigos QR y de barras de una imagen.
    
    Args:
        image: Archivo de imagen con códigos
        
    Returns:
        BarcodeResponse con los códigos detectados y resumen
    """
    try:
        # Validar imagen
        await validate_image(image)
        await image.seek(0)
        
        # Leer imagen
        content = await image.read()
        cv2_image = bytes_to_cv2_image(content)
        
        if cv2_image is None:
            return BarcodeResponse(
                success=False,
                message="No se pudo procesar la imagen",
                codes=[],
                has_codes=False,
                summary="Error al procesar la imagen"
            )
        
        # Redimensionar si es necesario
        cv2_image = resize_image_if_needed(cv2_image)
        
        # Leer códigos
        barcode_reader = get_barcode_reader()
        result = barcode_reader.read(cv2_image)
        
        # Convertir a schema
        codes = []
        for code in result.codes:
            codes.append(DetectedCode(
                type=code.type,
                data=code.data,
                format_name=code.format_name,
                bounding_box=dict(zip(["x", "y", "w", "h"], code.bounding_box)) if code.bounding_box else None,
                confidence=code.confidence
            ))
        
        return BarcodeResponse(
            success=result.has_codes,
            message=f"Se detectaron {len(codes)} código(s)" if result.has_codes else "No se detectaron códigos",
            codes=codes,
            has_codes=result.has_codes,
            summary=result.summary,
            count=len(codes),
            product_info=result.product_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error leyendo códigos de barras: {e}")
        return BarcodeResponse(
            success=False,
            message=f"Error: {str(e)}",
            codes=[],
            has_codes=False,
            summary="Error al procesar la imagen"
        )
