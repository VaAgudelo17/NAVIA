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
import {
  ThemeId, FontSizeId, FontFamilyId,
  THEMES, FONT_SIZES, FONT_FAMILIES, ThemeColors,
} from '../constants/themes';

// ============================================================================
// TIPOS DEL CONTEXTO
// ============================================================================

interface PreferencesContextType {
  // Valores
  analysisMode: AnalysisMode;
  readingMode: ReadingMode;
  ttsEnabled: boolean;
  themeId: ThemeId;
  fontSizeId: FontSizeId;
  fontFamilyId: FontFamilyId;
  isLoaded: boolean;

  // Tema activo (colores resueltos)
  theme: ThemeColors;
  fontScale: number;
  /** Familia tipográfica resuelta (string para style.fontFamily, o undefined = sistema) */
  fontFamily: string | undefined;

  // Setters (guardan automaticamente)
  setAnalysisMode: (mode: AnalysisMode) => void;
  setReadingMode: (mode: ReadingMode) => void;
  setTtsEnabled: (enabled: boolean) => void;
  setThemeId: (id: ThemeId) => void;
  setFontSizeId: (id: FontSizeId) => void;
  setFontFamilyId: (id: FontFamilyId) => void;
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
  const [themeId, setThemeIdState] = useState<ThemeId>(DEFAULT_PREFERENCES.themeId);
  const [fontSizeId, setFontSizeIdState] = useState<FontSizeId>(DEFAULT_PREFERENCES.fontSizeId);
  const [fontFamilyId, setFontFamilyIdState] = useState<FontFamilyId>(DEFAULT_PREFERENCES.fontFamilyId);
  const [isLoaded, setIsLoaded] = useState(false);

  // Cargar preferencias al montar
  useEffect(() => {
    (async () => {
      try {
        const prefs = await loadPreferences();
        setAnalysisModeState(prefs.analysisMode);
        setReadingModeState(prefs.readingMode);
        setTtsEnabledState(prefs.ttsEnabled);
        setThemeIdState(prefs.themeId);
        setFontSizeIdState(prefs.fontSizeId);
        setFontFamilyIdState(prefs.fontFamilyId);
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

  const setThemeId = useCallback((id: ThemeId) => {
    setThemeIdState(id);
    savePreferences({ themeId: id });
  }, []);

  const setFontSizeId = useCallback((id: FontSizeId) => {
    setFontSizeIdState(id);
    savePreferences({ fontSizeId: id });
  }, []);

  const setFontFamilyId = useCallback((id: FontFamilyId) => {
    setFontFamilyIdState(id);
    savePreferences({ fontFamilyId: id });
  }, []);

  const value: PreferencesContextType = {
    analysisMode,
    readingMode,
    ttsEnabled,
    themeId,
    fontSizeId,
    fontFamilyId,
    isLoaded,
    theme: THEMES[themeId].colors,
    fontScale: FONT_SIZES[fontSizeId].scale,
    fontFamily: FONT_FAMILIES[fontFamilyId].fontFamily,
    setAnalysisMode,
    setReadingMode,
    setTtsEnabled,
    setThemeId,
    setFontSizeId,
    setFontFamilyId,
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
