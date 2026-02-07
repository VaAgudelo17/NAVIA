"""
============================================================================
NAVIA Backend - Archivo de Configuración Central
============================================================================
Este módulo centraliza todas las configuraciones del proyecto.
Usar un archivo de configuración central es una buena práctica porque:
1. Facilita cambiar valores sin modificar el código
2. Permite diferentes configuraciones para desarrollo/producción
3. Mantiene las credenciales sensibles separadas del código
============================================================================
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """
    Clase de configuración usando Pydantic.

    Pydantic valida automáticamente los tipos de datos y permite
    cargar valores desde variables de entorno o archivo .env
    """

    # --- INFORMACIÓN DEL PROYECTO ---
    PROJECT_NAME: str = "NAVIA - Backend de Asistencia Visual"
    PROJECT_VERSION: str = "1.0.0"
    PROJECT_DESCRIPTION: str = """
    Backend para aplicación de asistencia visual dirigida a personas
    con discapacidad visual. Procesa imágenes mediante IA para extraer
    texto (OCR), detectar objetos y generar descripciones de escenas.

    Desarrollado como proyecto de tesis universitaria.
    """

    # --- CONFIGURACIÓN DEL SERVIDOR ---
    API_HOST: str = "0.0.0.0"  # Escuchar en todas las interfaces
    API_PORT: int = 8000
    DEBUG_MODE: bool = True  # Cambiar a False en producción

    # --- RUTAS DEL SISTEMA ---
    # Path.resolve() convierte rutas relativas a absolutas
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MODELS_DIR: Path = BASE_DIR / "models"

    # --- CONFIGURACIÓN DE IMÁGENES ---
    # Formatos de imagen permitidos (MIME types)
    ALLOWED_IMAGE_TYPES: List[str] = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    ]
    # Tamaño máximo de imagen: 10 MB (en bytes)
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024

    # --- CONFIGURACIÓN DE OCR (TESSERACT) ---
    # Idiomas para OCR: español e inglés
    # Tesseract usa códigos ISO 639-2 (spa=español, eng=inglés)
    TESSERACT_LANG: str = "spa+eng"
    # Ruta a Tesseract (varía según sistema operativo)
    # macOS con Homebrew: /opt/homebrew/bin/tesseract
    # Linux: /usr/bin/tesseract
    # Windows: C:\Program Files\Tesseract-OCR\tesseract.exe
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "/opt/homebrew/bin/tesseract")

    # --- CONFIGURACIÓN DE YOLO ---
    # Modelo YOLOv8 a utilizar
    # Opciones: yolov8n (nano/rápido), yolov8s (small), yolov8m (medium)
    # Usamos 'nano' para equilibrar velocidad y precisión en desarrollo
    YOLO_MODEL: str = "yolov8n.pt"
    # Umbral de confianza: solo reportar detecciones con >50% confianza
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5

    # --- CONFIGURACIÓN CORS ---
    # CORS permite que la app móvil se comunique con el backend
    # En producción, especificar dominios exactos
    CORS_ORIGINS: List[str] = ["*"]  # Permitir todos en desarrollo

    class Config:
        """Configuración de Pydantic para cargar variables de entorno."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Instancia global de configuración
# Se importa en otros módulos como: from app.core.config import settings
settings = Settings()

# Crear directorio de uploads si no existe
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
