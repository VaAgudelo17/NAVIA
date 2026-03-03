/**
 * Configuración de la aplicación NAVIA
 */

import Constants from 'expo-constants';

// URL del backend - se detecta automáticamente.
// Expo inyecta la IP de tu máquina en hostUri (ej. "192.168.1.16:8081").
// Extraemos la IP y apuntamos al puerto del backend (8000).
// Así funciona en cualquier red sin cambiar código.
const BACKEND_PORT = 8000;

function getApiBaseUrl(): string {
  // 1. Override manual (si defines EXPO_PUBLIC_API_URL en .env)
  const envUrl = process.env.EXPO_PUBLIC_API_URL;
  if (envUrl) return envUrl;

  // 2. Auto-detectar IP del dev server de Expo
  const hostUri = Constants.expoConfig?.hostUri; // "192.168.1.16:8081"
  if (hostUri) {
    const host = hostUri.split(':')[0];
    return `http://${host}:${BACKEND_PORT}`;
  }

  // 3. Fallback
  return `http://localhost:${BACKEND_PORT}`;
}

export const API_BASE_URL = getApiBaseUrl();

// Endpoints de la API
export const API_ENDPOINTS = {
  HEALTH: '/api/v1/health',
  TTS: '/api/v1/tts',
  // Nuevos modos
  NAVEGACION: '/api/v1/analyze/navegacion',
  EXPLORACION: '/api/v1/analyze/exploracion',
  LECTURA: '/api/v1/analyze/lectura',
  RIESGO: '/api/v1/analyze/riesgo',
  // Historial y preferencias
  HISTORY: '/api/v1/history',
  PREFERENCES: '/api/v1/preferences',
  // Legacy (compatibilidad)
  OCR: '/api/v1/analyze/ocr',
  OBJECTS: '/api/v1/analyze/objects',
  SCENE: '/api/v1/analyze/scene',
  QUICK: '/api/v1/analyze/quick',
  UPLOAD: '/api/v1/analyze/upload',
};

// Colores de la aplicación (tema oscuro accesible)
export const COLORS = {
  background: '#0a0e14',
  surface: '#141b22',
  primary: '#14b8a6',      // Turquesa
  primaryDark: '#0d9488',
  secondary: '#1e293b',
  text: '#f8fafc',
  textSecondary: '#94a3b8',
  error: '#ef4444',
  success: '#22c55e',
  warning: '#f59e0b',
  border: '#1e293b',
};

// Configuración de Text-to-Speech
export const TTS_CONFIG = {
  language: 'es-ES',
  pitch: 1.0,
  rate: 0.9,
};

// Modos de análisis disponibles
// Nota: "riesgo" fue unificado con "navegacion" en un solo pipeline
export const ANALYSIS_MODES = {
  NAVEGACION: 'navegacion',
  EXPLORACION: 'exploracion',
  LECTURA: 'lectura',
} as const;

export type AnalysisMode = typeof ANALYSIS_MODES[keyof typeof ANALYSIS_MODES];

// Modos que usan WebSocket (tiempo real con cámara)
// Navegación ahora incluye detección de riesgo integrada
export const REALTIME_MODES: AnalysisMode[] = ['navegacion'];

// URL WebSocket (convierte http → ws)
export const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

// Configuración de detección en tiempo real
export const REALTIME_CONFIG = {
  targetFps: 2,           // 2 frames por segundo (mobile)
  ttsMinInterval: 3000,   // 3 segundos entre frases TTS
  imageQuality: 0.3,      // Calidad JPEG baja para velocidad
};
