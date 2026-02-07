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

export interface DetectedObject {
  name: string
  name_es: string
  confidence: number
  bounding_box: BoundingBox
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
  objects: DetectedObject[]
  object_count: number
  processing_details?: {
    ocr_confidence: number | null
    ocr_word_count: number
    image_dimensions: string
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
