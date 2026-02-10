"use client"

import React, { useState, useEffect, useRef } from "react"

interface AnimatedEyeProps {
  className?: string
}

export function AnimatedEye({ className = "h-8 w-8" }: AnimatedEyeProps) {
  const [isBlinking, setIsBlinking] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    let cancelled = false

    const blink = () => {
      setIsBlinking(true)

      // Mantener cerrado 80ms, luego abrir
      timeoutRef.current = setTimeout(() => {
        if (cancelled) return
        setIsBlinking(false)

        // Esperar intervalo random antes del siguiente parpadeo
        const delay = 3000 + Math.random() * 3000
        timeoutRef.current = setTimeout(() => {
          if (!cancelled) blink()
        }, delay)
      }, 230) // 150ms cerrar + 80ms mantener cerrado
    }

    const initialDelay = 2000 + Math.random() * 2000
    timeoutRef.current = setTimeout(blink, initialDelay)

    return () => {
      cancelled = true
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  return (
    <svg
      viewBox="0 0 64 64"
      className={className}
      aria-hidden="true"
    >
      {/* Pestañas (fuera del grupo animado) */}
      <line x1="22" y1="18" x2="20" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <line x1="32" y1="16" x2="32" y2="9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <line x1="42" y1="18" x2="44" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />

      {/* Cuerpo del ojo (se anima con scaleY) */}
      <g
        style={{
          transformOrigin: "32px 32px",
          transform: `scaleY(${isBlinking ? 0.1 : 1})`,
          transition: isBlinking
            ? "transform 150ms ease-in"
            : "transform 200ms ease-out",
        }}
      >
        {/* Contorno almendrado */}
        <path
          d="M8 32 C8 32, 20 16, 32 16 C44 16, 56 32, 56 32 C56 32, 44 48, 32 48 C20 48, 8 32, 8 32 Z"
          stroke="currentColor"
          strokeWidth="2"
          fill="none"
        />
        {/* Iris */}
        <circle cx="32" cy="32" r="11" className="fill-primary" />
        {/* Pupila */}
        <circle cx="32" cy="32" r="5" className="fill-background" />
        {/* Brillo */}
        <circle cx="28" cy="28" r="1.5" fill="rgba(255,255,255,0.7)" />
      </g>
    </svg>
  )
}
