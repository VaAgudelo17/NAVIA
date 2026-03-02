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
  // Siempre inicializar con defaultValue para evitar hydration mismatch.
  // En SSR y en la primera render del cliente usamos el mismo valor.
  const [storedValue, setStoredValue] = useState<T>(defaultValue)
  const [hydrated, setHydrated] = useState(false)

  // Después del mount, leer localStorage y actualizar si difiere
  useEffect(() => {
    try {
      const item = window.localStorage.getItem(key)
      if (item) {
        const parsed = JSON.parse(item) as T
        setStoredValue(parsed)
      }
    } catch (error) {
      console.warn(`Error leyendo localStorage key "${key}":`, error)
    }
    setHydrated(true)
  }, [key])

  // Guardar en localStorage cada vez que cambia el valor (solo después de hidratar)
  useEffect(() => {
    if (!hydrated) return
    try {
      window.localStorage.setItem(key, JSON.stringify(storedValue))
    } catch (error) {
      console.warn(`Error guardando localStorage key "${key}":`, error)
    }
  }, [key, storedValue, hydrated])

  // Setter que acepta valor directo o funcion updater
  const setValue = useCallback((value: T | ((prev: T) => T)) => {
    setStoredValue((prev) => {
      const newValue = value instanceof Function ? value(prev) : value
      return newValue
    })
  }, [])

  return [storedValue, setValue]
}
