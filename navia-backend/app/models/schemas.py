"""
============================================================================
NAVIA Backend - Schemas de Pydantic (modelos de request/response)
============================================================================
Define los modelos de datos para validación y serialización de la API.
============================================================================
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ============================================================================
# BASE
# ============================================================================

class BaseResponse(BaseModel):
    success: bool
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: Optional[str] = None


# ============================================================================
# HEALTH
# ============================================================================

class HealthResponse(BaseResponse):
    status: str
    version: str


# ============================================================================
# OBJETOS DETECTADOS
# ============================================================================

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DetectedObject(BaseModel):
    name: str
    name_es: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    distance_zone: Optional[str] = None
    distance_estimate: Optional[str] = None
    priority: Optional[str] = None


# ============================================================================
# RIESGO (deprecado, mantenido por compatibilidad)
# ============================================================================

class RiskAlert(BaseModel):
    object_name: str
    risk_level: str
    description: str
    position: Optional[str] = None


# ============================================================================
# IMAGEN / UPLOAD
# ============================================================================

class ImageUploadResponse(BaseResponse):
    image_id: str
    filename: str


# ============================================================================
# OCR
# ============================================================================

class OCRResponse(BaseResponse):
    text: str
    confidence: Optional[float] = None
    word_count: Optional[int] = None
    has_text: bool


# ============================================================================
# DETECCIÓN DE OBJETOS
# ============================================================================

class ObjectDetectionResponse(BaseResponse):
    objects: List[DetectedObject]
    object_count: int
    summary: str


# ============================================================================
# DESCRIPCIÓN DE ESCENA
# ============================================================================

class SceneDescriptionResponse(BaseResponse):
    description: str
    detected_text: Optional[str] = ""
    has_text: bool = False
    caption: Optional[str] = None
    objects: List[DetectedObject] = []
    object_count: int = 0
    processing_details: Optional[Dict[str, Any]] = None


# ============================================================================
# NAVEGACIÓN
# ============================================================================

class NavigationResponse(BaseResponse):
    instruction: str
    obstacles: List[str] = []
    path_clear: bool = True
    has_danger: bool = False
    priority: Optional[str] = None
    obstacle_details: List[Dict[str, Any]] = []
    object_count: int = 0


# ============================================================================
# EXPLORACIÓN
# ============================================================================

class ExplorationResponse(BaseResponse):
    description: str = ""
    detected_text: Optional[str] = ""
    has_text: bool = False
    objects: List[DetectedObject] = []
    object_count: int = 0


# ============================================================================
# RIESGO (endpoint /riesgo, comparte estructura con navegación)
# ============================================================================

class RiskResponse(BaseResponse):
    instruction: Optional[str] = None
    obstacles: List[str] = []
    path_clear: bool = True
    has_danger: bool = False
    priority: Optional[str] = None
    obstacle_details: List[Dict[str, Any]] = []
    object_count: int = 0


# ============================================================================
# LECTURA INTELIGENTE
# ============================================================================

class SmartReadingResponse(BaseResponse):
    narrative: str
    document_type: str
    document_type_label: str
    reading_mode: str
    raw_text: str
    has_text: bool
    ocr_confidence: Optional[float] = None
    word_count: Optional[int] = None
    extracted_fields: Optional[Dict[str, Any]] = None
    visual_caption: Optional[str] = None
    image_quality: Optional[Any] = None
    classification: Optional[Dict[str, Any]] = None


# ============================================================================
# HISTORIAL
# ============================================================================

class HistoryItemResponse(BaseModel):
    id: str
    mode: str
    reading_mode: Optional[str] = None
    result_summary: str
    result_data: Dict[str, Any] = {}
    image_filename: Optional[str] = None
    processing_time_ms: Optional[float] = None
    object_count: Optional[int] = None
    has_text: Optional[bool] = None
    has_danger: Optional[bool] = None
    created_at: Optional[datetime] = None


class HistoryListResponse(BaseResponse):
    items: List[HistoryItemResponse]
    total: int
    page: int
    page_size: int


# ============================================================================
# PREFERENCIAS
# ============================================================================

class PreferenceItem(BaseModel):
    key: str
    value: str


class PreferencesResponse(BaseResponse):
    preferences: Dict[str, str]


class PreferencesUpdateRequest(BaseModel):
    preferences: Dict[str, str]


# ============================================================================
# CÓDIGOS DE BARRAS / QR
# ============================================================================

class DetectedCode(BaseModel):
    """Código detectado en la imagen."""
    type: str
    data: str
    format_name: str
    bounding_box: Optional[Dict[str, int]] = None
    confidence: float = 1.0


class BarcodeResponse(BaseResponse):
    """Respuesta del escaneo de códigos QR y barras."""
    codes: List[DetectedCode] = []
    has_codes: bool = False
    summary: str
    count: int = 0
    product_info: Optional[Dict[str, Any]] = None
