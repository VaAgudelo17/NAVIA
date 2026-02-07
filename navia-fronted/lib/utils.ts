import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

/**
 * Combina clases de Tailwind CSS de forma inteligente.
 * Evita conflictos entre clases y las fusiona correctamente.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Formatea un porcentaje de confianza para mostrar al usuario.
 */
export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`
}

/**
 * Genera un ID único para identificar elementos.
 */
export function generateId(): string {
  return Math.random().toString(36).substring(2, 9)
}
