"""
============================================================================
NAVIA Backend - Endpoint de Health Check
============================================================================
Este módulo implementa el endpoint de verificación de estado del servicio.

¿Por qué un health check?
- Permite verificar que el servidor está funcionando
- Útil para monitoreo y balanceadores de carga
- Proporciona información de versión y estado
- Facilita debugging en producción
============================================================================
"""

from fastapi import APIRouter, status
from app.models.schemas import HealthResponse
from app.core.config import settings

# Crear router para este grupo de endpoints
# prefix: todas las rutas empezarán con /health
# tags: agrupa endpoints en la documentación Swagger
router = APIRouter(
    prefix="/health",
    tags=["Health Check"]
)


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Verificar estado del servicio",
    description="Retorna el estado actual del servicio y su versión"
)
async def health_check() -> HealthResponse:
    """
    Endpoint de verificación de estado.

    Retorna información básica sobre el estado del servicio:
    - status: "healthy" si todo funciona correctamente
    - version: versión actual de la API
    - timestamp: fecha/hora de la respuesta

    Returns:
        HealthResponse: Estado del servicio
    """
    return HealthResponse(
        success=True,
        message="Servicio operativo",
        status="healthy",
        version=settings.PROJECT_VERSION
    )


@router.get(
    "/detailed",
    summary="Estado detallado del servicio",
    description="Retorna información detallada incluyendo servicios de IA"
)
async def health_check_detailed() -> dict:
    """
    Verificación detallada del estado.

    Incluye información sobre:
    - Estado general del servicio
    - Disponibilidad de servicios de OCR y detección
    - Configuración activa

    Returns:
        dict: Estado detallado del servicio
    """
    from app.services.ocr_service import get_ocr_service
    from app.services.object_detection_service import get_object_detection_service

    # Verificar servicios
    services_status = {
        "ocr": "unknown",
        "object_detection": "unknown"
    }

    try:
        ocr = get_ocr_service()
        services_status["ocr"] = "available"
    except Exception as e:
        services_status["ocr"] = f"error: {str(e)}"

    try:
        detector = get_object_detection_service()
        services_status["object_detection"] = "available"
        model_info = detector.get_model_info()
    except Exception as e:
        services_status["object_detection"] = f"error: {str(e)}"
        model_info = {}

    return {
        "status": "healthy",
        "version": settings.PROJECT_VERSION,
        "project": settings.PROJECT_NAME,
        "debug_mode": settings.DEBUG_MODE,
        "services": services_status,
        "yolo_model": model_info,
        "configuration": {
            "max_image_size_mb": settings.MAX_IMAGE_SIZE / (1024 * 1024),
            "allowed_image_types": settings.ALLOWED_IMAGE_TYPES,
            "ocr_languages": settings.TESSERACT_LANG,
            "yolo_confidence": settings.YOLO_CONFIDENCE_THRESHOLD
        }
    }
