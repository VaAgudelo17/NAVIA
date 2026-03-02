/**
 * Hook para detección en tiempo real (Web)
 *
 * Extrae frames del video via canvas y los envía al backend
 * por WebSocket. Soporta modo Navegación (con detección de riesgo integrada).
 *
 * Al finalizar la sesión (enabled → false), llama onSessionEnd
 * con un resumen para guardar en historial.
 */

"use client"

import { useState, useRef, useCallback, useEffect } from 'react'
import { RealtimeWebSocket } from '@/lib/websocket'
import { RealtimeTtsManager } from '@/lib/realtimeTts'
import { type RealtimeDetectionResult, type NaviaMode } from '@/lib/api'

export interface RealtimeSessionSummary {
  durationSeconds: number
  framesProcessed: number
  dangerCount: number
  topObstacles: Record<string, number>
  summary: string
}

interface UseRealtimeDetectionOptions {
  videoRef: React.RefObject<HTMLVideoElement | null>
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  enabled: boolean
  ttsEnabled: boolean
  mode: NaviaMode
  targetFps?: number
  onSessionEnd?: (summary: RealtimeSessionSummary) => void
}

export function useRealtimeDetection({
  videoRef,
  canvasRef,
  enabled,
  ttsEnabled,
  mode,
  targetFps = 5,
  onSessionEnd,
}: UseRealtimeDetectionOptions) {
  const [wsStatus, setWsStatus] = useState<string>('disconnected')
  const [latestResult, setLatestResult] = useState<RealtimeDetectionResult | null>(null)

  const wsRef = useRef<RealtimeWebSocket | null>(null)
  const ttsManagerRef = useRef<RealtimeTtsManager>(new RealtimeTtsManager())
  const animFrameRef = useRef<number | null>(null)
  const lastCaptureTime = useRef(0)

  // --- Estadísticas de sesión ---
  const sessionStartRef = useRef<number>(0)
  const framesRef = useRef<number>(0)
  const dangerCountRef = useRef<number>(0)
  const obstacleCountsRef = useRef<Record<string, number>>({})
  const onSessionEndRef = useRef(onSessionEnd)
  onSessionEndRef.current = onSessionEnd

  // Actualizar modo del TTS manager cuando cambie
  useEffect(() => {
    ttsManagerRef.current.setMode(mode)
  }, [mode])

  const handleDetection = useCallback((data: RealtimeDetectionResult) => {
    setLatestResult(data)

    // Acumular estadísticas
    framesRef.current += 1
    if (data.has_danger) {
      dangerCountRef.current += 1
    }
    if (data.obstacle_details) {
      for (const obs of data.obstacle_details) {
        const name = obs.name
        obstacleCountsRef.current[name] = (obstacleCountsRef.current[name] || 0) + 1
      }
    }

    if (ttsEnabled) {
      ttsManagerRef.current.speakResult(
        data.summary,
        data.changes as any,
        {
          has_danger: data.has_danger ?? false,
          priority: data.priority ?? 'none',
          path_clear: data.path_clear ?? true,
        },
      )
    }
  }, [ttsEnabled, mode])

  useEffect(() => {
    if (!enabled) {
      // --- Al desactivar: generar resumen y notificar ---
      const frames = framesRef.current
      if (frames > 0 && onSessionEndRef.current) {
        const durationSeconds = (Date.now() - sessionStartRef.current) / 1000
        const dangers = dangerCountRef.current
        const obstacles = obstacleCountsRef.current

        const topObstacles: Record<string, number> = {}
        Object.entries(obstacles)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 5)
          .forEach(([name, count]) => { topObstacles[name] = count })

        const durationMin = (durationSeconds / 60).toFixed(1)
        const obsText = Object.entries(topObstacles)
          .map(([name, count]) => `${name} (${count}x)`)
          .join(', ')

        const summary = obsText
          ? `Sesión de ${durationMin} min, ${frames} frames. Obstáculos: ${obsText}. Alertas: ${dangers}.`
          : `Sesión de ${durationMin} min, ${frames} frames. Camino libre.`

        onSessionEndRef.current({
          durationSeconds,
          framesProcessed: frames,
          dangerCount: dangers,
          topObstacles,
          summary,
        })
      }

      // Limpieza
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
        animFrameRef.current = null
      }
      wsRef.current?.disconnect()
      wsRef.current = null
      ttsManagerRef.current.reset()
      setLatestResult(null)
      setWsStatus('disconnected')
      return
    }

    // Reiniciar stats de sesión
    sessionStartRef.current = Date.now()
    framesRef.current = 0
    dangerCountRef.current = 0
    obstacleCountsRef.current = {}

    // Conectar WebSocket con modo
    wsRef.current = new RealtimeWebSocket(handleDetection, setWsStatus, mode)
    wsRef.current.connect()

    const intervalMs = 1000 / targetFps

    function captureLoop() {
      const now = performance.now()
      if (now - lastCaptureTime.current >= intervalMs) {
        lastCaptureTime.current = now

        const video = videoRef.current
        const canvas = canvasRef.current
        if (video && canvas && video.readyState >= 2) {
          canvas.width = 640
          canvas.height = Math.round(640 * (video.videoHeight / video.videoWidth))
          const ctx = canvas.getContext('2d')
          if (ctx) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
            const dataUrl = canvas.toDataURL('image/jpeg', 0.5)
            const base64 = dataUrl.split(',')[1]
            wsRef.current?.sendFrame(base64)
          }
        }
      }
      animFrameRef.current = requestAnimationFrame(captureLoop)
    }

    animFrameRef.current = requestAnimationFrame(captureLoop)

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
        animFrameRef.current = null
      }
      wsRef.current?.disconnect()
      wsRef.current = null
      ttsManagerRef.current.stop()
    }
  }, [enabled, targetFps, handleDetection, mode, videoRef, canvasRef])

  return { wsStatus, latestResult }
}
