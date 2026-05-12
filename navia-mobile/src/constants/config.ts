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

  // 2. Auto-detectar IP del dev server de Expo (solo en nativo)
  const hostUri = Constants.expoConfig?.hostUri; // "192.168.1.16:8081"
  if (hostUri) {
    const host = hostUri.split(':')[0];
    return `http://${host}:${BACKEND_PORT}`;
  }

  // 3. En web (navegador) usar el backend de producción en Railway
  if (typeof document !== 'undefined') {
    return 'https://patient-charisma-production.up.railway.app';
  }

  // 4. Fallback local
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
// language: se usa como fallback si resolveVoice() no encuentra un voice identifier.
// iOS reconoce 'es-ES' (Monica, Siri España).
// Android reconoce 'es-ES', 'es-MX', etc. según lo que el usuario tenga instalado.
import { Platform } from 'react-native';

export const TTS_CONFIG = {
  language: Platform.OS === 'ios' ? 'es-ES' : 'es-MX',
  pitch: 1.0,
  rate: 0.78,
  // Para modo lectura: más lento y con pausas marcadas
  readingRate: 0.60,
  readingPitch: 0.97,
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

// Configuración de detección en tiempo real.
// El backend debe poder procesar cada frame antes del siguiente, sino la
// cola crece y la latencia explota. En HF Spaces (CPU compartida) procesar
// YOLO + Depth tarda ~1.5s; localmente ~300ms.
export const REALTIME_CONFIG = {
  targetFps: 2,            // Bajado 3→2: Railway CPU tarda ~500ms/frame, 3fps saturaba la cola
  ttsMinInterval: 2000,    // 2 segundos entre frases TTS
  imageQuality: 0.35,      // Bajado 0.4→0.35: menos bytes por frame, misma precisión YOLO
};
