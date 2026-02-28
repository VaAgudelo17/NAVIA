/**
 * Context de Preferencias del Usuario
 *
 * Provee estado global para las preferencias que deben persistir
 * entre reinicios de la app:
 * - analysisMode: modo de analisis seleccionado
 * - readingMode: sub-modo de lectura
 * - ttsEnabled: si el TTS esta activado
 *
 * Al montar, carga las preferencias desde AsyncStorage.
 * Cada cambio se guarda automaticamente en AsyncStorage.
 */

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import {
  loadPreferences,
  savePreferences,
  UserPreferences,
  DEFAULT_PREFERENCES,
} from '../services/storage';
import { AnalysisMode } from '../constants/config';
import { ReadingMode } from '../types/api';

// ============================================================================
// TIPOS DEL CONTEXTO
// ============================================================================

interface PreferencesContextType {
  // Valores
  analysisMode: AnalysisMode;
  readingMode: ReadingMode;
  ttsEnabled: boolean;
  isLoaded: boolean; // true cuando las preferencias se cargaron de storage

  // Setters (guardan automaticamente)
  setAnalysisMode: (mode: AnalysisMode) => void;
  setReadingMode: (mode: ReadingMode) => void;
  setTtsEnabled: (enabled: boolean) => void;
}

const PreferencesContext = createContext<PreferencesContextType | undefined>(undefined);

// ============================================================================
// PROVIDER
// ============================================================================

interface PreferencesProviderProps {
  children: ReactNode;
}

export function PreferencesProvider({ children }: PreferencesProviderProps) {
  const [analysisMode, setAnalysisModeState] = useState<AnalysisMode>(DEFAULT_PREFERENCES.analysisMode);
  const [readingMode, setReadingModeState] = useState<ReadingMode>(DEFAULT_PREFERENCES.readingMode);
  const [ttsEnabled, setTtsEnabledState] = useState<boolean>(DEFAULT_PREFERENCES.ttsEnabled);
  const [isLoaded, setIsLoaded] = useState(false);

  // Cargar preferencias al montar
  useEffect(() => {
    (async () => {
      try {
        const prefs = await loadPreferences();
        setAnalysisModeState(prefs.analysisMode);
        setReadingModeState(prefs.readingMode);
        setTtsEnabledState(prefs.ttsEnabled);
      } catch (error) {
        console.warn('Error cargando preferencias, usando defaults:', error);
      } finally {
        setIsLoaded(true);
      }
    })();
  }, []);

  // Setters que guardan automaticamente
  const setAnalysisMode = useCallback((mode: AnalysisMode) => {
    setAnalysisModeState(mode);
    savePreferences({ analysisMode: mode });
  }, []);

  const setReadingMode = useCallback((mode: ReadingMode) => {
    setReadingModeState(mode);
    savePreferences({ readingMode: mode });
  }, []);

  const setTtsEnabled = useCallback((enabled: boolean) => {
    setTtsEnabledState(enabled);
    savePreferences({ ttsEnabled: enabled });
  }, []);

  const value: PreferencesContextType = {
    analysisMode,
    readingMode,
    ttsEnabled,
    isLoaded,
    setAnalysisMode,
    setReadingMode,
    setTtsEnabled,
  };

  return (
    <PreferencesContext.Provider value={value}>
      {children}
    </PreferencesContext.Provider>
  );
}

// ============================================================================
// HOOK
// ============================================================================

/**
 * Hook para acceder a las preferencias del usuario.
 * Debe usarse dentro de un PreferencesProvider.
 */
export function usePreferences(): PreferencesContextType {
  const context = useContext(PreferencesContext);
  if (!context) {
    throw new Error('usePreferences debe usarse dentro de un PreferencesProvider');
  }
  return context;
}
