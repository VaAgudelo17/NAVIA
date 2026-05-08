"""
NAVIA - Backend de Asistencia Visual con Inteligencia Artificial

Proyecto de tesis para el desarrollo de una aplicación de asistencia visual
dirigida a personas con discapacidad visual. Procesa imágenes en tiempo real
para detectar obstáculos, estimar distancias, leer textos y describir entornos.

Autora: Valentina Agudelo Maldonado
Institución: Universidad San Buenaventura Cali
Año: 2025-2026
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.api.router import api_router
from app.db.database import init_db, close_db


logging.basicConfig(
    level=logging.INFO if settings.DEBUG_MODE else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === INICIO DE LA APLICACIÓN ===
    logger.info("=" * 60)
    logger.info(f"Iniciando {settings.PROJECT_NAME}")
    logger.info(f"Versión: {settings.PROJECT_VERSION}")
    logger.info(f"Modo debug: {'Activado' if settings.DEBUG_MODE else 'Desactivado'}")
    logger.info("=" * 60)

    # Inicializar base de datos
    logger.info("Inicializando base de datos...")
    try:
        await init_db()
        logger.info("Base de datos lista")
    except Exception as e:
        logger.error(f"Error inicializando base de datos: {e}")
        raise

    # Precargar modelo YOLO al inicio (evita spike de RAM en primer WebSocket)
    logger.info("Precargando modelo YOLO...")
    try:
        from app.services.object_detection_service import get_object_detection_service
        get_object_detection_service()
        logger.info("Modelo YOLO cargado exitosamente")
    except Exception as e:
        logger.warning(f"No se pudo precargar YOLO: {e}")

    # Precargar modelo Depth Anything V2 en background.
    # Se lanza como tarea asyncio para no bloquear el arranque del servidor.
    # El modelo pesa ~300MB y puede tardar 20-30s en cargarse en CPU.
    # Con esta estrategia el servidor queda listo de inmediato y Depth
    # se calienta en paralelo; cuando el primer usuario abre navegación
    # el modelo ya está listo (o casi listo).
    async def _warmup_depth():
        try:
            import numpy as np
            from app.services.depth_estimation_service import get_depth_estimation_service
            depth_svc = get_depth_estimation_service()
            loop = asyncio.get_event_loop()
            dummy = np.zeros((240, 320, 3), dtype=np.uint8)
            # estimate_depth_map fuerza la carga real del modelo en el thread pool
            await loop.run_in_executor(None, depth_svc.estimate_depth_map, dummy)
            logger.info("Modelo Depth Anything V2 cargado en background")
        except Exception as e:
            logger.warning(f"Depth pre-warm en background falló: {e}")

    asyncio.create_task(_warmup_depth())

    logger.info(f"Servidor listo en http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info("Documentación disponible en /docs")

    yield  # La aplicación está corriendo

    # === CIERRE DE LA APLICACIÓN ===
    logger.info("Cerrando aplicación...")
    await close_db()
    logger.info("Limpieza completada")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/info", tags=["Información"])
async def api_info():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "description": "Backend de asistencia visual con IA",
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "ocr": "/api/v1/analyze/ocr",
            "objects": "/api/v1/analyze/objects",
            "scene": "/api/v1/analyze/scene"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE
    )
