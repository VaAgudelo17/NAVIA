/**
 * Tipos de datos de la API de NAVIA
 * Corresponden a los schemas del backend FastAPI
 */

// Bounding box de un objeto detectado
export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

// Objeto detectado en la imagen
export interface DetectedObject {
  name: string;           // Nombre en inglés
  name_es: string;        // Nombre en español
  confidence: number;     // Confianza (0.0 - 1.0)
  bounding_box: BoundingBox;
}

// Respuesta del health check
export interface HealthResponse {
  success: boolean;
  message: string;
  status: string;
  version: string;
}

// Respuesta del OCR
export interface OCRResponse {
  success: boolean;
  message: string;
  text: string;
  confidence: number | null;
  word_count: number;
  has_text: boolean;
}

// Respuesta de detección de objetos
export interface ObjectDetectionResponse {
  success: boolean;
  message: string;
  objects: DetectedObject[];
  object_count: number;
  summary: string;
}

// Respuesta del análisis de escena (endpoint principal)
export interface SceneDescriptionResponse {
  success: boolean;
  message: string;
  description: string;          // Texto para TTS
  detected_text: string;        // Texto encontrado en la imagen
  has_text: boolean;
  objects: DetectedObject[];
  object_count: number;
  processing_details?: {
    ocr_confidence: number | null;
    ocr_word_count: number;
    image_dimensions: string;
  };
}

// Respuesta de análisis rápido
export interface QuickAnalysisResponse {
  success: boolean;
  description: string;
  object_count: number;
  objects: Array<{
    name: string;
    confidence: number;
  }>;
}

// Error de la API
export interface APIError {
  success: false;
  error_code: string;
  message: string;
  detail?: string;
}
