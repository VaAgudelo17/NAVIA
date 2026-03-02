/**
 * ============================================================================
 * NAVIA Frontend - Servicio de API
 * ============================================================================
 * Este módulo maneja toda la comunicación con el backend FastAPI.
 * Proporciona funciones para enviar imágenes y recibir análisis de IA.
 * ============================================================================
 */

// URL base del backend (configurable via variable de entorno)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/**
 * Tipos de datos que devuelve el backend
 */
export interface BoundingBox {
  x_min: number
  y_min: number
  x_max: number
  y_max: number
}

export type DistanceZone = 'muy_cerca' | 'cerca' | 'lejos'

export type SemanticPriority = 'high' | 'medium' | 'low'

export interface DetectedObject {
  name: string
  name_es: string
  confidence: number
  bounding_box: BoundingBox
  distance_zone?: DistanceZone
  distance_estimate?: string
  priority?: SemanticPriority
}

export interface OCRResponse {
  success: boolean
  message: string
  text: string
  confidence: number | null
  word_count: number
  has_text: boolean
}

export interface ObjectDetectionResponse {
  success: boolean
  message: string
  objects: DetectedObject[]
  object_count: number
  summary: string
}

export interface SceneDescriptionResponse {
  success: boolean
  message: string
  description: string
  detected_text: string
  has_text: boolean
  caption?: string
  objects: DetectedObject[]
  object_count: number
  processing_details?: {
    ocr_confidence: number | null
    ocr_word_count: number
    image_dimensions: string
    captioning_enabled?: boolean
    has_caption?: boolean
  }
}

export interface HealthResponse {
  success: boolean
  message: string
  status: string
  version: string
}

export interface APIError {
  success: false
  error_code: string
  message: string
  detail?: string
}

// ============================================================================
// LECTURA INTELIGENTE
// ============================================================================

export type ReadingMode = 'resumen' | 'detallado' | 'financiero'

export type DocumentType =
  | 'factura'
  | 'recibo'
  | 'carta'
  | 'formulario'
  | 'documento_informativo'
  | 'imagen_visual'
  | 'desconocido'

export interface ExtractedFields {
  dates: string[]
  amounts: string[]
  emails: string[]
  phones: string[]
  ids: string[]
  headers: string[]
  totals: string[]
}

export interface ImageQualityIssue {
  code: string
  severity: string
  message: string
}

export interface ImageQuality {
  overall_score: number
  is_acceptable: boolean
  issues: ImageQualityIssue[]
  feedback_text: string
}

export interface SmartReadingResponse {
  success: boolean
  message: string
  narrative: string
  document_type: DocumentType
  document_type_label: string
  reading_mode: ReadingMode
  raw_text: string
  has_text: boolean
  ocr_confidence: number | null
  word_count: number
  extracted_fields: ExtractedFields
  visual_caption: string | null
  image_quality: ImageQuality | null
}

// ============================================================================
// NUEVOS MODOS
// ============================================================================

export type NaviaMode = 'navegacion' | 'exploracion' | 'lectura'

// Detalle de obstáculo analizado por el pipeline de navegación
export interface ObstacleDetail {
  name: string               // Nombre en español
  position: string           // "a tu izquierda", "frente a ti", "a tu derecha"
  proximity: string          // "muy_cerca", "cerca", "lejos"
  height_zone: string        // "suelo", "cuerpo", "cabeza"
  movement: string           // "acercandose", "alejandose", "estatico"
  risk_score: number         // Puntaje de riesgo combinado (0.0 - 1.0)
}

// Respuesta unificada del modo Navegación (incluye riesgo)
export interface NavigationResponse {
  success: boolean
  message: string
  instruction: string             // Instrucciones priorizadas para TTS
  obstacles: DetectedObject[]     // Obstáculos relevantes
  path_clear: boolean             // Si el camino central está libre
  has_danger: boolean             // Si se detectó peligro real
  priority: 'critical' | 'high' | 'medium' | 'none'
  obstacle_details: ObstacleDetail[]
  object_count: number
}

export interface ExplorationResponse {
  success: boolean
  message: string
  description: string
  detected_text: string
  has_text: boolean
  objects: DetectedObject[]
  object_count: number
}

// Alias de compatibilidad: RiskResponse = NavigationResponse
export type RiskResponse = NavigationResponse

// Resultado de detección en tiempo real (WebSocket)
export interface RealtimeDetectionResult {
  type: 'detection'
  frame_id: number
  objects: DetectedObject[]
  object_count: number
  summary: string               // Instrucciones de navegación para TTS
  processing_time_ms: number
  timestamp: number
  tracked_count?: number
  mode?: string
  // Navegación unificada (siempre presente)
  has_danger?: boolean
  priority?: string
  path_clear?: boolean
  obstacle_details?: ObstacleDetail[]
  changes?: {
    appeared: string[]
    disappeared: string[]
    zone_changes: Array<{ name: string; from_zone: string; to_zone: string }>
    smoothed_zones: Record<string, DistanceZone>
    has_significant_change: boolean
    current_objects: string[]
    tracked_count?: number
  }
}

/**
 * Verifica si el backend está disponible
 */
export async function checkHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/health`)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    return await response.json()
  } catch (error) {
    console.error('Error checking health:', error)
    throw new Error('No se pudo conectar con el servidor. Verifica que el backend esté ejecutándose.')
  }
}

/**
 * Extrae texto de una imagen usando OCR
 * @param imageFile - Archivo de imagen a procesar
 * @returns Resultado del OCR con texto extraído
 */
export async function extractText(imageFile: File): Promise<OCRResponse> {
  const formData = new FormData()
  formData.append('image', imageFile)

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/analyze/ocr`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Error en el procesamiento de OCR')
    }

    return await response.json()
  } catch (error) {
    console.error('Error in OCR:', error)
    throw error
  }
}

/**
 * Detecta objetos en una imagen
 * @param imageFile - Archivo de imagen a procesar
 * @returns Lista de objetos detectados con sus ubicaciones
 */
export async function detectObjects(imageFile: File): Promise<ObjectDetectionResponse> {
  const formData = new FormData()
  formData.append('image', imageFile)

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/analyze/objects`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Error en la detección de objetos')
    }

    return await response.json()
  } catch (error) {
    console.error('Error in object detection:', error)
    throw error
  }
}

/**
 * Analiza una escena completa (OCR + detección de objetos)
 * Este es el endpoint principal para la aplicación.
 * @param imageFile - Archivo de imagen a procesar
 * @returns Descripción completa de la escena para TTS
 */
export async function analyzeScene(imageFile: File): Promise<SceneDescriptionResponse> {
  const formData = new FormData()
  formData.append('image', imageFile)

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/analyze/scene`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Error analizando la escena')
    }

    return await response.json()
  } catch (error) {
    console.error('Error analyzing scene:', error)
    throw error
  }
}

/**
 * Análisis rápido (solo detección de objetos, sin OCR)
 * Más rápido cuando no se espera texto en la imagen.
 * @param imageFile - Archivo de imagen a procesar
 */
export async function quickAnalysis(imageFile: File): Promise<{
  success: boolean
  description: string
  object_count: number
  objects: Array<{ name: string; confidence: number }>
}> {
  const formData = new FormData()
  formData.append('image', imageFile)

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/analyze/quick`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Error en análisis rápido')
    }

    return await response.json()
  } catch (error) {
    console.error('Error in quick analysis:', error)
    throw error
  }
}

// ============================================================================
// NUEVOS MODOS - Funciones API
// ============================================================================

async function postImage<T>(endpoint: string, imageFile: File, errorMsg: string): Promise<T> {
  const formData = new FormData()
  formData.append('image', imageFile)

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || errorMsg)
    }

    return await response.json()
  } catch (error) {
    console.error(`Error: ${errorMsg}:`, error)
    throw error
  }
}

/** Modo Navegación: instrucciones cortas de navegación */
export async function analyzeNavigation(imageFile: File): Promise<NavigationResponse> {
  return postImage<NavigationResponse>('/api/v1/analyze/navegacion', imageFile, 'Error en navegación')
}

/** Modo Exploración: descripción estructurada del entorno */
export async function analyzeExploration(imageFile: File): Promise<ExplorationResponse> {
  return postImage<ExplorationResponse>('/api/v1/analyze/exploracion', imageFile, 'Error en exploración')
}

/** Modo Lectura Inteligente: OCR + clasificación + narrativa automática */
export async function analyzeReading(imageFile: File): Promise<SmartReadingResponse> {
  return postImage<SmartReadingResponse>('/api/v1/analyze/lectura', imageFile, 'Error en lectura')
}

/** Modo Riesgo: redirige al pipeline unificado de navegación */
export async function analyzeRisk(imageFile: File): Promise<RiskResponse> {
  return postImage<RiskResponse>('/api/v1/analyze/navegacion', imageFile, 'Error en evaluación de riesgo')
}

// ============================================================================
// UTILIDADES
// ============================================================================

// ============================================================================
// HISTORIAL Y PREFERENCIAS
// ============================================================================

export interface HistoryItem {
  id: string
  mode: string
  reading_mode?: string
  result_summary: string
  result_data: Record<string, unknown>
  image_filename?: string
  processing_time_ms?: number
  object_count?: number
  has_text?: boolean
  has_danger?: boolean
  created_at: string
}

export interface HistoryListResponse {
  success: boolean
  message: string
  items: HistoryItem[]
  total: number
  page: number
  page_size: number
}

export interface PreferencesResponse {
  success: boolean
  message: string
  preferences: Record<string, string>
}

/**
 * Obtiene el historial de analisis del backend
 */
export async function getHistory(mode?: string, page: number = 1, pageSize: number = 20): Promise<HistoryListResponse> {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  if (mode) params.set('mode', mode)

  const response = await fetch(`${API_BASE_URL}/api/v1/history?${params}`)
  if (!response.ok) throw new Error('Error obteniendo historial')
  return response.json()
}

/**
 * Elimina un registro del historial
 */
export async function deleteHistoryItem(id: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/history/${id}`, { method: 'DELETE' })
  if (!response.ok) throw new Error('Error eliminando registro')
}

/**
 * Limpia todo el historial
 */
export async function clearHistory(): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/history`, { method: 'DELETE' })
  if (!response.ok) throw new Error('Error limpiando historial')
}

/**
 * Obtiene las preferencias del backend
 */
export async function getPreferences(): Promise<PreferencesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/preferences`)
  if (!response.ok) throw new Error('Error obteniendo preferencias')
  return response.json()
}

/**
 * Actualiza preferencias en el backend
 */
export async function updatePreferences(prefs: Record<string, string>): Promise<PreferencesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/preferences`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preferences: prefs }),
  })
  if (!response.ok) throw new Error('Error actualizando preferencias')
  return response.json()
}

/**
 * Convierte un Blob de imagen a File
 * Útil cuando se captura desde la cámara
 */
export function blobToFile(blob: Blob, filename: string = 'capture.jpg'): File {
  return new File([blob], filename, { type: blob.type || 'image/jpeg' })
}

/**
 * Convierte una URL de datos (data URL) a File
 * Útil para imágenes en formato base64
 */
export function dataURLtoFile(dataURL: string, filename: string = 'image.jpg'): File {
  const arr = dataURL.split(',')
  const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/jpeg'
  const bstr = atob(arr[1])
  let n = bstr.length
  const u8arr = new Uint8Array(n)

  while (n--) {
    u8arr[n] = bstr.charCodeAt(n)
  }

  return new File([u8arr], filename, { type: mime })
}
