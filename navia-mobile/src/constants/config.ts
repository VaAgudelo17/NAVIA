/**
 * Configuración de la aplicación NAVIA
 */

// URL del backend - cambiar según el entorno
// Para desarrollo local: usar la IP de tu computadora
// Para producción: usar la URL del servidor
export const API_BASE_URL = 'http://192.168.1.21:8000';

// Endpoints de la API
export const API_ENDPOINTS = {
  HEALTH: '/api/v1/health',
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
export const ANALYSIS_MODES = {
  SCENE: 'scene',      // OCR + Detección de objetos
  TEXT: 'text',        // Solo OCR
  OBJECTS: 'objects',  // Solo detección de objetos
  REALTIME: 'realtime', // Detección en tiempo real
} as const;

export type AnalysisMode = typeof ANALYSIS_MODES[keyof typeof ANALYSIS_MODES];

// URL WebSocket (convierte http → ws)
export const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

// Configuración de detección en tiempo real
export const REALTIME_CONFIG = {
  targetFps: 2,           // 2 frames por segundo (mobile)
  ttsMinInterval: 3000,   // 3 segundos entre frases TTS
  imageQuality: 0.3,      // Calidad JPEG baja para velocidad
};
