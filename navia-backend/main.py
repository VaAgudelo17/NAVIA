"""
============================================================================
NAVIA - Backend de Asistencia Visual con Inteligencia Artificial
============================================================================

Proyecto de tesis universitaria para el desarrollo de una aplicación
de asistencia visual dirigida a personas con discapacidad visual.

FUNCIONALIDADES:
- Extracción de texto de imágenes (OCR con Tesseract)
- Detección de objetos (YOLOv8)
- Generación de descripciones de escenas
- API REST para comunicación con aplicación móvil

TECNOLOGÍAS:
- Framework: FastAPI (Python)
- Visión por computadora: OpenCV, Tesseract, YOLOv8
- Servidor: Uvicorn (ASGI)

AUTOR: [Tu nombre]
INSTITUCIÓN: Universidad Simón Bolívar
AÑO: 2024

============================================================================
INSTRUCCIONES DE EJECUCIÓN:
============================================================================

1. Crear entorno virtual:
   python -m venv venv

2. Activar entorno virtual:
   - macOS/Linux: source venv/bin/activate
   - Windows: venv\\Scripts\\activate

3. Instalar dependencias:
   pip install -r requirements.txt

4. Instalar Tesseract (macOS):
   brew install tesseract tesseract-lang

5. Ejecutar servidor:
   uvicorn main:app --reload

6. Acceder a documentación:
   http://localhost:8000/docs (Swagger UI)
   http://localhost:8000/redoc (ReDoc)

============================================================================
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.api.router import api_router


# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
# Logging permite registrar eventos importantes durante la ejecución.
# Útil para debugging y monitoreo en producción.

logging.basicConfig(
    level=logging.INFO if settings.DEBUG_MODE else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN: EVENTOS DE INICIO Y CIERRE
# ============================================================================
# El contexto de vida (lifespan) permite ejecutar código al iniciar
# y cerrar la aplicación. Útil para:
# - Precargar modelos de IA
# - Inicializar conexiones
# - Liberar recursos al cerrar

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Contexto de vida de la aplicación.

    El código antes de 'yield' se ejecuta al iniciar.
    El código después de 'yield' se ejecuta al cerrar.
    """
    # === INICIO DE LA APLICACIÓN ===
    logger.info("=" * 60)
    logger.info(f"Iniciando {settings.PROJECT_NAME}")
    logger.info(f"Versión: {settings.PROJECT_VERSION}")
    logger.info(f"Modo debug: {'Activado' if settings.DEBUG_MODE else 'Desactivado'}")
    logger.info("=" * 60)

    # Precargar modelos (opcional, mejora tiempo de primera respuesta)
    if settings.DEBUG_MODE:
        logger.info("Precargando modelos de IA...")
        try:
            from app.services.object_detection_service import get_object_detection_service
            get_object_detection_service()
            logger.info("Modelo YOLO cargado exitosamente")
        except Exception as e:
            logger.warning(f"No se pudo precargar YOLO: {e}")

    logger.info(f"Servidor listo en http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info("Documentación disponible en /docs")

    yield  # La aplicación está corriendo

    # === CIERRE DE LA APLICACIÓN ===
    logger.info("Cerrando aplicación...")
    logger.info("Limpieza completada")


# ============================================================================
# CREACIÓN DE LA APLICACIÓN FASTAPI
# ============================================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
    # Configuración de documentación
    docs_url="/docs",          # Swagger UI
    redoc_url="/redoc",        # ReDoc (alternativa)
    openapi_url="/openapi.json",
    # Información de contacto (ajustar con tus datos)
    contact={
        "name": "Soporte NAVIA",
        "email": "soporte@navia.com"
    },
    license_info={
        "name": "Uso Académico",
        "url": "https://opensource.org/licenses/MIT"
    }
)


# ============================================================================
# CONFIGURACIÓN DE CORS
# ============================================================================
# CORS (Cross-Origin Resource Sharing) permite que la aplicación móvil
# se comunique con el backend desde un dominio diferente.
#
# En desarrollo: permitimos todos los orígenes (*)
# En producción: especificar dominios exactos por seguridad

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # ["*"] en desarrollo
    allow_credentials=True,               # Permitir cookies/auth
    allow_methods=["*"],                  # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],                  # Todos los headers
)


# ============================================================================
# INCLUSIÓN DE RUTAS
# ============================================================================

# Incluir todas las rutas de la API
app.include_router(api_router)


# ============================================================================
# RUTAS ADICIONALES (Raíz)
# ============================================================================

@app.get(
    "/",
    include_in_schema=False,
    summary="Redirección a documentación"
)
async def root():
    """
    Redirige la ruta raíz a la documentación Swagger.

    Esto facilita el acceso a la documentación cuando alguien
    visita la URL base del servidor.
    """
    return RedirectResponse(url="/docs")


@app.get(
    "/info",
    tags=["Información"],
    summary="Información de la API"
)
async def api_info():
    """
    Retorna información básica de la API.

    Útil para verificar conectividad y obtener metadatos.
    """
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


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Ejecutar servidor
    # host="0.0.0.0" permite conexiones desde cualquier IP
    # reload=True reinicia automáticamente cuando cambia el código
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE
    )
