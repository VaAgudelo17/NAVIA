/**
 * Servicio de API para conectar con el backend FastAPI
 */

import { API_BASE_URL, API_ENDPOINTS } from '../constants/config';
import {
  HealthResponse,
  OCRResponse,
  ObjectDetectionResponse,
  SceneDescriptionResponse,
  QuickAnalysisResponse,
  NavigationResponse,
  ExplorationResponse,
  RiskResponse,
  SmartReadingResponse,
  ReadingMode,
} from '../types/api';

/**
 * Verifica si el backend está disponible
 */
export async function checkHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.HEALTH}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error checking health:', error);
    throw new Error('No se pudo conectar con el servidor');
  }
}

/**
 * Crea un FormData con la imagen para enviar al backend
 */
function createImageFormData(imageUri: string): FormData {
  const formData = new FormData();

  // Obtener nombre del archivo y tipo
  const filename = imageUri.split('/').pop() || 'photo.jpg';
  const match = /\.(\w+)$/.exec(filename);
  const type = match ? `image/${match[1]}` : 'image/jpeg';

  // Agregar imagen al FormData
  formData.append('image', {
    uri: imageUri,
    name: filename,
    type: type,
  } as any);

  return formData;
}

/**
 * Helper genérico para enviar imagen a un endpoint
 */
async function postImage<T>(endpoint: string, imageUri: string, errorMsg: string): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s máximo

  try {
    const formData = createImageFormData(imageUri);
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
      // NO establecer Content-Type manualmente - fetch auto-genera el boundary correcto
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || errorMsg);
    }

    return await response.json();
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      throw new Error('El servidor tardó demasiado. Inténtalo de nuevo.');
    }
    console.error(`Error: ${errorMsg}:`, error);
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

// ============================================================================
// NUEVOS MODOS
// ============================================================================

/** Modo Navegación: instrucciones cortas de navegación */
export async function analyzeNavigation(imageUri: string): Promise<NavigationResponse> {
  return postImage<NavigationResponse>(API_ENDPOINTS.NAVEGACION, imageUri, 'Error en navegación');
}

/** Modo Exploración: descripción estructurada del entorno */
export async function analyzeExploration(imageUri: string): Promise<ExplorationResponse> {
  return postImage<ExplorationResponse>(API_ENDPOINTS.EXPLORACION, imageUri, 'Error en exploración');
}

/** Modo Lectura Inteligente: OCR + clasificación + narrativa automática */
export async function analyzeReading(
  imageUri: string,
): Promise<SmartReadingResponse> {
  return postImage<SmartReadingResponse>(API_ENDPOINTS.LECTURA, imageUri, 'Error en lectura');
}

/** Modo Riesgo: redirige al pipeline unificado de navegación */
export async function analyzeRisk(imageUri: string): Promise<RiskResponse> {
  return postImage<RiskResponse>(API_ENDPOINTS.NAVEGACION, imageUri, 'Error en evaluación de riesgo');
}

// ============================================================================
// LEGACY (compatibilidad)
// ============================================================================

export async function extractText(imageUri: string): Promise<OCRResponse> {
  return postImage<OCRResponse>(API_ENDPOINTS.OCR, imageUri, 'Error en OCR');
}

export async function detectObjects(imageUri: string): Promise<ObjectDetectionResponse> {
  return postImage<ObjectDetectionResponse>(API_ENDPOINTS.OBJECTS, imageUri, 'Error en detección');
}

export async function analyzeScene(imageUri: string): Promise<SceneDescriptionResponse> {
  return postImage<SceneDescriptionResponse>(API_ENDPOINTS.SCENE, imageUri, 'Error en análisis de escena');
}

export async function quickAnalysis(imageUri: string): Promise<QuickAnalysisResponse> {
  return postImage<QuickAnalysisResponse>(API_ENDPOINTS.QUICK, imageUri, 'Error en análisis rápido');
}

// ============================================================================
// HISTORIAL Y PREFERENCIAS (Backend)
// ============================================================================

export interface HistoryItem {
  id: string;
  mode: string;
  reading_mode?: string;
  result_summary: string;
  result_data: Record<string, unknown>;
  processing_time_ms?: number;
  object_count?: number;
  has_text?: boolean;
  has_danger?: boolean;
  created_at: string;
}

export interface HistoryListResponse {
  success: boolean;
  items: HistoryItem[];
  total: number;
  page: number;
  page_size: number;
}

/** Obtiene el historial de analisis del backend */
export async function getHistory(
  mode?: string,
  page: number = 1,
  pageSize: number = 20,
): Promise<HistoryListResponse> {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
  if (mode) params.set('mode', mode);

  const response = await fetch(`${API_BASE_URL}/api/v1/history?${params}`);
  if (!response.ok) throw new Error('Error obteniendo historial');
  return response.json();
}

/** Obtiene las preferencias del backend */
export async function getBackendPreferences(): Promise<Record<string, string>> {
  const response = await fetch(`${API_BASE_URL}/api/v1/preferences`);
  if (!response.ok) throw new Error('Error obteniendo preferencias');
  const data = await response.json();
  return data.preferences;
}

/** Actualiza preferencias en el backend */
export async function updateBackendPreferences(prefs: Record<string, string>): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/preferences`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preferences: prefs }),
  });
  if (!response.ok) throw new Error('Error actualizando preferencias');
}
