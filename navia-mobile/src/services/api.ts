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
  try {
    const formData = createImageFormData(imageUri);
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || errorMsg);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error: ${errorMsg}:`, error);
    throw error;
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

/** Modo Lectura: OCR puro */
export async function analyzeReading(imageUri: string): Promise<OCRResponse> {
  return postImage<OCRResponse>(API_ENDPOINTS.LECTURA, imageUri, 'Error en lectura');
}

/** Modo Riesgo: detección de peligros */
export async function analyzeRisk(imageUri: string): Promise<RiskResponse> {
  return postImage<RiskResponse>(API_ENDPOINTS.RIESGO, imageUri, 'Error en evaluación de riesgo');
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
