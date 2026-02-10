/**
 * Tipos de datos de la API de NAVIA
 * Corresponden a los schemas del backend FastAPI
 */

// Modos de NAVIA
export type NaviaMode = 'navegacion' | 'exploracion' | 'lectura' | 'riesgo';

// Bounding box de un objeto detectado
export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

// Zonas de distancia
export type DistanceZone = 'muy_cerca' | 'cerca' | 'lejos';

// Objeto detectado en la imagen
export interface DetectedObject {
  name: string;           // Nombre en inglés
  name_es: string;        // Nombre en español
  confidence: number;     // Confianza (0.0 - 1.0)
  bounding_box: BoundingBox;
  distance_zone?: DistanceZone;    // Zona: muy_cerca, cerca, lejos
  distance_estimate?: string;       // Etiqueta legible: "muy cerca", "cerca", "lejos"
}

// Respuesta del health check
export interface HealthResponse {
  success: boolean;
  message: string;
  status: string;
  version: string;
}

// Respuesta del OCR (modo Lectura)
export interface OCRResponse {
  success: boolean;
  message: string;
  text: string;
  confidence: number | null;
  word_count: number;
  has_text: boolean;
}

// Respuesta de detección de objetos (legacy)
export interface ObjectDetectionResponse {
  success: boolean;
  message: string;
  objects: DetectedObject[];
  object_count: number;
  summary: string;
}

// Respuesta del análisis de escena (legacy)
export interface SceneDescriptionResponse {
  success: boolean;
  message: string;
  description: string;
  detected_text: string;
  has_text: boolean;
  objects: DetectedObject[];
  object_count: number;
  processing_details?: {
    ocr_confidence: number | null;
    ocr_word_count: number;
    image_dimensions: string;
  };
}

// Respuesta de análisis rápido (legacy)
export interface QuickAnalysisResponse {
  success: boolean;
  description: string;
  object_count: number;
  objects: Array<{
    name: string;
    confidence: number;
  }>;
}

// ============================================================================
// NUEVOS MODOS
// ============================================================================

// Respuesta del modo Navegación
export interface NavigationResponse {
  success: boolean;
  message: string;
  instruction: string;        // Texto corto para TTS
  obstacles: DetectedObject[];
  path_clear: boolean;
  object_count: number;
}

// Respuesta del modo Exploración
export interface ExplorationResponse {
  success: boolean;
  message: string;
  description: string;        // Descripción estructurada para TTS
  detected_text: string;
  has_text: boolean;
  objects: DetectedObject[];
  object_count: number;
}

// Alerta de riesgo individual
export interface RiskAlert {
  object_name: string;
  danger_level: 'critical' | 'high' | 'medium';
  distance_zone: string;
  position: string;
}

// Respuesta del modo Riesgo
export interface RiskResponse {
  success: boolean;
  message: string;
  has_danger: boolean;
  priority: 'critical' | 'high' | 'medium' | 'none';
  alert_text: string;
  dangers: RiskAlert[];
}

// Error de la API
export interface APIError {
  success: false;
  error_code: string;
  message: string;
  detail?: string;
}

// Resultado de detección en tiempo real (WebSocket)
export interface RealtimeDetectionResult {
  type: 'detection';
  mode: NaviaMode;
  frame_id: number;
  objects: DetectedObject[];
  object_count: number;
  summary: string;
  processing_time_ms: number;
  timestamp: number;
  changes?: {
    appeared: string[];
    disappeared: string[];
    zone_changes: Array<{ name: string; from_zone: string; to_zone: string }>;
    smoothed_zones: Record<string, DistanceZone>;
    has_significant_change: boolean;
    current_objects: string[];
  };
  // Campos específicos del modo Riesgo
  has_danger?: boolean;
  priority?: string;
}

// Mensaje de estado WebSocket
export interface RealtimeStatusMessage {
  type: 'status';
  state: 'connected' | 'ready' | 'processing';
  message: string;
}
