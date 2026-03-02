/**
 * Hook para detección en tiempo real (Web)
 *
 * Extrae frames del video via canvas y los envía al backend
 * por WebSocket. Soporta modo Navegación (con detección de riesgo integrada).
 */

"use client"

import { useState, useRef, useCallback, useEffect } from 'react'
import { RealtimeWebSocket } from '@/lib/websocket'
import { RealtimeTtsManager } from '@/lib/realtimeTts'
import { type RealtimeDetectionResult, type NaviaMode } from '@/lib/api'

interface UseRealtimeDetectionOptions {
  videoRef: React.RefObject<HTMLVideoElement | null>
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  enabled: boolean
  ttsEnabled: boolean
  mode: NaviaMode
  targetFps?: number
}

export function useRealtimeDetection({
  videoRef,
  canvasRef,
  enabled,
  ttsEnabled,
  mode,
  targetFps = 5,
}: UseRealtimeDetectionOptions) {
  const [wsStatus, setWsStatus] = useState<string>('disconnected')
  const [latestResult, setLatestResult] = useState<RealtimeDetectionResult | null>(null)

  const wsRef = useRef<RealtimeWebSocket | null>(null)
  const ttsManagerRef = useRef<RealtimeTtsManager>(new RealtimeTtsManager())
  const animFrameRef = useRef<number | null>(null)
  const lastCaptureTime = useRef(0)

  // Actualizar modo del TTS manager cuando cambie
  useEffect(() => {
    ttsManagerRef.current.setMode(mode)
  }, [mode])

  const handleDetection = useCallback((data: RealtimeDetectionResult) => {
    setLatestResult(data)

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
