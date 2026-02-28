"use client"

import { useState, useEffect, useCallback } from "react"

/**
 * Hook generico que sincroniza estado con localStorage.
 *
 * Al montar, carga el valor desde localStorage.
 * Cada cambio se guarda automaticamente en localStorage.
 *
 * @param key - Clave en localStorage
 * @param defaultValue - Valor por defecto si no existe en storage
 */
export function useLocalStorage<T>(key: string, defaultValue: T): [T, (value: T | ((prev: T) => T)) => void] {
  // Inicializar con el valor de localStorage o el default
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === "undefined") return defaultValue
    try {
      const item = window.localStorage.getItem(key)
      return item ? (JSON.parse(item) as T) : defaultValue
    } catch (error) {
      console.warn(`Error leyendo localStorage key "${key}":`, error)
      return defaultValue
    }
  })

  // Guardar en localStorage cada vez que cambia el valor
  useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.localStorage.setItem(key, JSON.stringify(storedValue))
    } catch (error) {
      console.warn(`Error guardando localStorage key "${key}":`, error)
    }
  }, [key, storedValue])

  // Setter que acepta valor directo o funcion updater
  const setValue = useCallback((value: T | ((prev: T) => T)) => {
    setStoredValue((prev) => {
      const newValue = value instanceof Function ? value(prev) : value
      return newValue
    })
  }, [])

  return [storedValue, setValue]
}
