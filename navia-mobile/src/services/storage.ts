/**
 * Servicio de persistencia local usando AsyncStorage.
 *
 * Responsabilidades:
 * - Guardar y cargar preferencias del usuario
 * - Guardar historial local de analisis
 *
 * AsyncStorage es un almacenamiento key-value persistente
 * que sobrevive al cierre de la app.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { AnalysisMode } from '../constants/config';
import { ReadingMode } from '../types/api';

// ============================================================================
// CLAVES DE STORAGE
// ============================================================================

const STORAGE_KEYS = {
  PREFERENCES: '@navia_preferences',
  HISTORY: '@navia_history',
} as const;

// ============================================================================
// TIPOS
// ============================================================================

export interface UserPreferences {
  analysisMode: AnalysisMode;
  readingMode: ReadingMode;
  ttsEnabled: boolean;
}

export interface HistoryEntry {
  id: string;
  mode: AnalysisMode;
  readingMode?: ReadingMode;
  resultSummary: string;
  resultData: Record<string, unknown>;
  createdAt: string; // ISO string
}

// Valores por defecto
export const DEFAULT_PREFERENCES: UserPreferences = {
  analysisMode: 'navegacion',
  readingMode: 'detallado',
  ttsEnabled: true,
};

const MAX_HISTORY_ITEMS = 50; // Limite de historial local

// ============================================================================
// PREFERENCIAS
// ============================================================================

/**
 * Guarda las preferencias del usuario en almacenamiento local.
 */
export async function savePreferences(prefs: Partial<UserPreferences>): Promise<void> {
  try {
    // Cargar existentes y mergear
    const current = await loadPreferences();
    const updated = { ...current, ...prefs };
    await AsyncStorage.setItem(STORAGE_KEYS.PREFERENCES, JSON.stringify(updated));
  } catch (error) {
    console.warn('Error guardando preferencias:', error);
  }
}

/**
 * Carga las preferencias del usuario desde almacenamiento local.
 * Si no existen, retorna valores por defecto.
 */
export async function loadPreferences(): Promise<UserPreferences> {
  try {
    const stored = await AsyncStorage.getItem(STORAGE_KEYS.PREFERENCES);
    if (stored) {
      const parsed = JSON.parse(stored);
      // Merge con defaults para cubrir nuevas preferencias
      return { ...DEFAULT_PREFERENCES, ...parsed };
    }
  } catch (error) {
    console.warn('Error cargando preferencias:', error);
  }
  return { ...DEFAULT_PREFERENCES };
}

// ============================================================================
// HISTORIAL LOCAL
// ============================================================================

/**
 * Guarda un resultado de analisis en el historial local.
 * Mantiene maximo MAX_HISTORY_ITEMS entradas (FIFO).
 */
export async function saveToHistory(entry: Omit<HistoryEntry, 'id' | 'createdAt'>): Promise<void> {
  try {
    const history = await getHistory();

    const newEntry: HistoryEntry = {
      ...entry,
      id: generateId(),
      createdAt: new Date().toISOString(),
    };

    // Agregar al inicio (mas reciente primero)
    history.unshift(newEntry);

    // Limitar tamano
    if (history.length > MAX_HISTORY_ITEMS) {
      history.splice(MAX_HISTORY_ITEMS);
    }

    await AsyncStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(history));
  } catch (error) {
    console.warn('Error guardando historial:', error);
  }
}

/**
 * Obtiene el historial local de analisis.
 */
export async function getHistory(limit?: number): Promise<HistoryEntry[]> {
  try {
    const stored = await AsyncStorage.getItem(STORAGE_KEYS.HISTORY);
    if (stored) {
      const history: HistoryEntry[] = JSON.parse(stored);
      return limit ? history.slice(0, limit) : history;
    }
  } catch (error) {
    console.warn('Error cargando historial:', error);
  }
  return [];
}

/**
 * Elimina todo el historial local.
 */
export async function clearHistory(): Promise<void> {
  try {
    await AsyncStorage.removeItem(STORAGE_KEYS.HISTORY);
  } catch (error) {
    console.warn('Error limpiando historial:', error);
  }
}

// ============================================================================
// UTILIDADES
// ============================================================================

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substring(2, 8);
}
