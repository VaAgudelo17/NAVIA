"use client"

/**
 * Context de Preferencias del Usuario (Frontend Web)
 *
 * Persiste preferencias en localStorage:
 * - analysisMode: modo de analisis seleccionado
 * - readingMode: sub-modo de lectura
 * - ttsEnabled: si el TTS esta activado
 *
 * Al montar, carga las preferencias desde localStorage.
 * Cada cambio se guarda automaticamente.
 */

import React, { createContext, useContext, useCallback, ReactNode } from "react"
import { useLocalStorage } from "@/hooks/useLocalStorage"
import type { NaviaMode, ReadingMode } from "@/lib/api"

// ============================================================================
// TIPOS
// ============================================================================

interface PreferencesContextType {
  analysisMode: NaviaMode
  readingMode: ReadingMode
  ttsEnabled: boolean
  setAnalysisMode: (mode: NaviaMode) => void
  setReadingMode: (mode: ReadingMode) => void
  setTtsEnabled: (enabled: boolean) => void
}

const PreferencesContext = createContext<PreferencesContextType | undefined>(undefined)

// ============================================================================
// PROVIDER
// ============================================================================

interface PreferencesProviderProps {
  children: ReactNode
}

export function PreferencesProvider({ children }: PreferencesProviderProps) {
  const [analysisMode, setAnalysisModeState] = useLocalStorage<NaviaMode>("navia_analysis_mode", "navegacion")
  const [readingMode, setReadingModeState] = useLocalStorage<ReadingMode>("navia_reading_mode", "detallado")
  const [ttsEnabled, setTtsEnabledState] = useLocalStorage<boolean>("navia_tts_enabled", true)

  const setAnalysisMode = useCallback((mode: NaviaMode) => {
    setAnalysisModeState(mode)
  }, [setAnalysisModeState])

  const setReadingMode = useCallback((mode: ReadingMode) => {
    setReadingModeState(mode)
  }, [setReadingModeState])

  const setTtsEnabled = useCallback((enabled: boolean) => {
    setTtsEnabledState(enabled)
  }, [setTtsEnabledState])

  const value: PreferencesContextType = {
    analysisMode,
    readingMode,
    ttsEnabled,
    setAnalysisMode,
    setReadingMode,
    setTtsEnabled,
  }

  return (
    <PreferencesContext.Provider value={value}>
      {children}
    </PreferencesContext.Provider>
  )
}

// ============================================================================
// HOOK
// ============================================================================

/**
 * Hook para acceder a las preferencias del usuario.
 * Debe usarse dentro de un PreferencesProvider.
 */
export function usePreferences(): PreferencesContextType {
  const context = useContext(PreferencesContext)
  if (!context) {
    throw new Error("usePreferences debe usarse dentro de un PreferencesProvider")
  }
  return context
}
