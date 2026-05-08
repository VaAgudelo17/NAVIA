"""Configuración central del backend NAVIA."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):

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
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MODELS_DIR: Path = BASE_DIR / "models"

    # --- CONFIGURACIÓN DE IMÁGENES ---
    # Formatos de imagen y PDF permitidos (MIME types)
    ALLOWED_IMAGE_TYPES: List[str] = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "application/pdf",
    ]
    # Tamaño máximo de imagen: 10 MB (en bytes)
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024

    # --- CONFIGURACIÓN DE OCR (TESSERACT) ---
    TESSERACT_LANG: str = "spa+eng"
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "/opt/homebrew/bin/tesseract")

    # --- CONFIGURACIÓN DE YOLO ---
    YOLO_MODEL: str = "yolov8s-worldv2.pt"   # small: suficiente para detección de obstáculos en tiempo real
    # Umbral de confianza: balance entre detectar objetos y evitar falsos positivos
    YOLO_CONFIDENCE_THRESHOLD: float = 0.35
    # Umbral específico para exploración (bajo para capturar TVs apagados, espejos, etc.)
    YOLO_EXPLORATION_CONFIDENCE: float = 0.40  # subido de 0.28 — reduce confusión gato/perro y cajas

    # --- CONFIGURACIÓN WEBSOCKET TIEMPO REAL ---
    WS_MAX_FRAME_SIZE: int = 5 * 1024 * 1024  # 5 MB máximo por frame
    WS_REALTIME_MAX_DIMENSION: int = 640       # Redimensionar frames para velocidad
    WS_REALTIME_CONFIDENCE_THRESHOLD: float = 0.25  # Umbral para tiempo real (más alto para evitar parpadeo)

    # --- CONFIGURACIÓN DE PROFUNDIDAD (DEPTH ANYTHING V2) ---
    DEPTH_MODEL: str = "depth-anything/Depth-Anything-V2-Small-hf"
    # Umbrales de zona (sobre depth normalizado [0,1], mayor = más cerca)
    DEPTH_ZONE_MUY_CERCA: float = 0.55   # > 0.55 = peligro (antes 0.70, advierte más temprano)
    DEPTH_ZONE_CERCA: float = 0.25       # > 0.25 = cerca  (antes 0.35, detecta desde más lejos)
    # Suavizado exponencial para tiempo real
    DEPTH_SMOOTHING_ALPHA: float = 0.5    # Peso de nueva observación (0-1) — más reactivo
    DEPTH_ZONE_PERSISTENCE: int = 2       # Frames para confirmar cambio de zona (antes 3)

    # --- CONFIGURACIÓN DE CAPTIONING (FLORENCE-2) ---
    # Modelo multimodal para generar descripciones naturales de escenas
    # Solo se usa en endpoints HTTP (/analyze/scene, /analyze/exploracion)
    # NO se usa en WebSocket tiempo real (demasiado lento para realtime)
    CAPTIONING_MODEL: str = "microsoft/Florence-2-base"
    CAPTIONING_ENABLED: bool = True

    # --- CONFIGURACIÓN DE PRIORIDAD SEMÁNTICA ---
    # Prioriza objetos relevantes para asistencia visual
    # high: obstáculos, personas, vehículos, escaleras
    # medium: muebles grandes que obstruyen paso
    # low: decoración, objetos pequeños
    SEMANTIC_PRIORITY_ENABLED: bool = True

    # CLIP para análisis detallado (opcional, desactivado por defecto)
    # Solo se usa en endpoints HTTP, nunca en realtime
    CLIP_ENABLED: bool = False
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"

    # --- CONFIGURACIÓN DE TRACKING (BYTETRACK) ---
    # Tracking espacial para estabilizar detecciones en video
    # Usa IoU + Kalman filter en vez de tracking por nombre
    TRACKING_ENABLED: bool = True
    TRACK_HIGH_THRESH: float = 0.3      # Umbral para primera asociación
    TRACK_LOW_THRESH: float = 0.1       # Umbral para segunda asociación
    TRACK_BUFFER_FRAMES: int = 30       # Frames para mantener track perdido
    TRACK_MATCH_THRESH: float = 0.7     # Umbral IoU para matching

    # --- CONFIGURACIÓN TTS ---
    # TTS_BACKEND: "gtts"  → voz femenina de Google (requiere internet en el servidor)
    #              "piper" → voz local VITS (offline, usar si no hay internet)
    TTS_ENABLED: bool = True
    TTS_BACKEND: str = "gtts"
    TTS_MODEL_NAME: str = "es_ES-davefx-medium"  # Solo se usa si TTS_BACKEND="piper"
    TTS_SAMPLE_RATE: int = 22050
    TTS_MAX_TEXT_LENGTH: int = 5000

    # --- CONFIGURACIÓN DE GEMINI (GOOGLE AI) ---
    # Gemini Vision para OCR avanzado de documentos (primario, con Tesseract fallback)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.5-flash"  # Tier gratuito activo, excelente OCR
    GEMINI_ENABLED: bool = True  # Desactivar para usar solo Tesseract
    GEMINI_TIMEOUT: int = 10  # Timeout en segundos para la API

    # --- CONFIGURACIÓN DE OPENAI ---
    # GPT-4o Vision para validación de detecciones y narrativa empática en modo exploración
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_ENABLED: bool = True
    OPENAI_TIMEOUT: int = 4   # Timeout en segundos para la API

    # --- CONFIGURACIÓN DE LECTURA INTELIGENTE ---
    SMART_READING_ENABLED: bool = True
    SMART_READING_DEFAULT_MODE: str = "detallado"

    # --- CONFIGURACIÓN DE BASE DE DATOS ---
    # SQLite para desarrollo, PostgreSQL para producción
    DATABASE_URL: str = "sqlite+aiosqlite:///./navia.db"
    DB_ECHO: bool = False  # True para ver queries SQL en log

    # --- CONFIGURACIÓN CORS ---
    # CORS permite que la app móvil se comunique con el backend
    # En producción, especificar dominios exactos
    CORS_ORIGINS: List[str] = ["*"]  # Permitir todos en desarrollo

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

# Crear directorio de uploads si no existe
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
