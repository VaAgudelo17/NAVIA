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
 * Extrae texto de una imagen usando OCR
 */
export async function extractText(imageUri: string): Promise<OCRResponse> {
  try {
    const formData = createImageFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.OCR}`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error en el procesamiento de OCR');
    }

    return await response.json();
  } catch (error) {
    console.error('Error in OCR:', error);
    throw error;
  }
}

/**
 * Detecta objetos en una imagen
 */
export async function detectObjects(imageUri: string): Promise<ObjectDetectionResponse> {
  try {
    const formData = createImageFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.OBJECTS}`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error en la detección de objetos');
    }

    return await response.json();
  } catch (error) {
    console.error('Error in object detection:', error);
    throw error;
  }
}

/**
 * Analiza una escena completa (OCR + detección de objetos)
 * Este es el endpoint principal para la aplicación
 */
export async function analyzeScene(imageUri: string): Promise<SceneDescriptionResponse> {
  try {
    const formData = createImageFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.SCENE}`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error analizando la escena');
    }

    return await response.json();
  } catch (error) {
    console.error('Error analyzing scene:', error);
    throw error;
  }
}

/**
 * Análisis rápido (solo detección de objetos, sin OCR)
 */
export async function quickAnalysis(imageUri: string): Promise<QuickAnalysisResponse> {
  try {
    const formData = createImageFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.QUICK}`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error en análisis rápido');
    }

    return await response.json();
  } catch (error) {
    console.error('Error in quick analysis:', error);
    throw error;
  }
}
