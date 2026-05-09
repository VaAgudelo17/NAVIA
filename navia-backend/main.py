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

    # Pre-calentar YOLO en modo NAVEGACIÓN: hace CLIP encoding de las clases de nav
    # (~70 clases), lo que tarda 15-30s en CPU la primera vez.
    # Navegación es tiempo real — si el encoding ocurre al abrir el WebSocket,
    # el usuario espera 15-30s antes de recibir el primer frame. Hacerlo aquí
    # elimina ese delay. Exploración es un análisis único (timeout 40s) y puede
    # tolerar el encoding en su primera llamada.
    logger.info("Pre-calentando YOLO para navegación (CLIP encoding)...")
    try:
        from app.services.object_detection_service import get_object_detection_service
        get_object_detection_service().configure_for_navigation()
        logger.info("YOLO en modo navegación listo")
    except Exception as e:
        logger.warning(f"No se pudo pre-calentar navegación: {e}")

    # Verificar disponibilidad de OpenAI (crítico para calidad de exploración)
    logger.info("Verificando OpenAI...")
    try:
        from app.services.openai_service import get_openai_service
        available = get_openai_service()._ensure_initialized()
        if available:
            logger.info("OpenAI disponible — modo exploración usará GPT-4o-mini")
        else:
            logger.warning(
                "OpenAI NO disponible — exploración usará narrativa local (menos precisa). "
                "Verifica que OPENAI_API_KEY esté configurada en las variables de entorno."
            )
    except Exception as e:
        logger.warning(f"No se pudo verificar OpenAI: {e}")

    # Depth Anything V2 se carga lazy al primer uso en WebSocket (navegación).
    # No se pre-carga al inicio para evitar presión de RAM en servidores pequeños.
    # (YOLO + Depth juntos superan 800MB, lo que causa OOM en tier gratuito)

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
