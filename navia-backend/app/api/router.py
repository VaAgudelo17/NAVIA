"""
============================================================================
NAVIA Backend - Router Principal de la API
============================================================================
Este módulo agrupa todos los routers de endpoints en un solo router
principal que se monta en la aplicación FastAPI.

Estructura de endpoints:
- /health: Verificación de estado del servicio
- /analyze: Procesamiento de imágenes (OCR, detección, descripción)

Agregar nuevos módulos de endpoints:
1. Crear archivo en app/api/endpoints/
2. Importar el router aquí
3. Incluirlo con api_router.include_router()
============================================================================
"""

from fastapi import APIRouter

# Importar routers de cada módulo de endpoints
from app.api.endpoints.health import router as health_router
from app.api.endpoints.image_analysis import router as image_router
from app.api.endpoints.realtime_ws import router as realtime_router

# Crear router principal de la API
# Todas las rutas tendrán el prefijo /api/v1
api_router = APIRouter(prefix="/api/v1")

# Incluir routers de cada módulo
api_router.include_router(health_router)
api_router.include_router(image_router)
api_router.include_router(realtime_router)
