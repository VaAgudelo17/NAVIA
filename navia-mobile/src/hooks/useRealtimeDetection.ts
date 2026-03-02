/**
 * Hook para detección en tiempo real
 *
 * Captura frames periódicos de la cámara y los envía al backend
 * via WebSocket. Soporta modo Navegación (con detección de riesgo integrada).
 *
 * Al finalizar la sesión (enabled → false), llama onSessionEnd
 * con un resumen para guardar en historial local.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { CameraView } from 'expo-camera';
import { RealtimeWebSocket } from '../services/websocket';
import { RealtimeTtsManager } from '../services/realtimeTts';
import { RealtimeDetectionResult, NaviaMode } from '../types/api';
import { REALTIME_CONFIG } from '../constants/config';

export interface RealtimeSessionSummary {
  durationSeconds: number;
  framesProcessed: number;
  dangerCount: number;
  topObstacles: Record<string, number>;
  summary: string;
}

interface UseRealtimeDetectionOptions {
  cameraRef: React.RefObject<CameraView | null>;
  enabled: boolean;
  ttsEnabled: boolean;
  mode: NaviaMode;
  onSessionEnd?: (summary: RealtimeSessionSummary) => void;
}

export function useRealtimeDetection({
  cameraRef,
  enabled,
  ttsEnabled,
  mode,
  onSessionEnd,
}: UseRealtimeDetectionOptions) {
  const [wsStatus, setWsStatus] = useState<string>('disconnected');
  const [latestResult, setLatestResult] = useState<RealtimeDetectionResult | null>(null);

  const wsRef = useRef<RealtimeWebSocket | null>(null);
  const ttsManagerRef = useRef<RealtimeTtsManager>(new RealtimeTtsManager());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isCapturing = useRef(false);

  // --- Estadísticas de sesión ---
  const sessionStartRef = useRef<number>(0);
  const framesRef = useRef<number>(0);
  const dangerCountRef = useRef<number>(0);
  const obstacleCountsRef = useRef<Record<string, number>>({});
  const onSessionEndRef = useRef(onSessionEnd);
  onSessionEndRef.current = onSessionEnd;

  // Actualizar modo del TTS manager cuando cambie
  useEffect(() => {
    ttsManagerRef.current.setMode(mode);
  }, [mode]);

  const handleDetection = useCallback((data: RealtimeDetectionResult) => {
    setLatestResult(data);

    // Acumular estadísticas
    framesRef.current += 1;
    if (data.has_danger) {
      dangerCountRef.current += 1;
    }
    if (data.obstacle_details) {
      for (const obs of data.obstacle_details) {
        const name = obs.name;
        obstacleCountsRef.current[name] = (obstacleCountsRef.current[name] || 0) + 1;
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
      );
    }
  }, [ttsEnabled, mode]);

  useEffect(() => {
    if (enabled) {
      // Reiniciar stats de sesión
      sessionStartRef.current = Date.now();
      framesRef.current = 0;
      dangerCountRef.current = 0;
      obstacleCountsRef.current = {};

      // Conectar WebSocket con modo
      wsRef.current = new RealtimeWebSocket(handleDetection, setWsStatus, mode);
      wsRef.current.connect();

      // Capturar frames periódicamente
      const intervalMs = 1000 / REALTIME_CONFIG.targetFps;
      intervalRef.current = setInterval(async () => {
        if (isCapturing.current || !cameraRef.current) return;
        isCapturing.current = true;

        try {
          const photo = await cameraRef.current.takePictureAsync({
            quality: REALTIME_CONFIG.imageQuality,
            base64: true,
            shutterSound: false,
          });

          if (photo?.base64 && wsRef.current) {
            wsRef.current.sendFrame(photo.base64);
          }
        } catch {
          // Cámara puede no estar lista; ignorar silenciosamente
        } finally {
          isCapturing.current = false;
        }
      }, intervalMs);
    } else {
      // --- Al desactivar: generar resumen y notificar ---
      const frames = framesRef.current;
      if (frames > 0 && onSessionEndRef.current) {
        const durationSeconds = (Date.now() - sessionStartRef.current) / 1000;
        const dangers = dangerCountRef.current;
        const obstacles = obstacleCountsRef.current;

        // Top 5 obstáculos por frecuencia
        const topObstacles: Record<string, number> = {};
        Object.entries(obstacles)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 5)
          .forEach(([name, count]) => { topObstacles[name] = count; });

        const durationMin = (durationSeconds / 60).toFixed(1);
        const obsText = Object.entries(topObstacles)
          .map(([name, count]) => `${name} (${count}x)`)
          .join(', ');

        const summary = obsText
          ? `Sesión de ${durationMin} min, ${frames} frames. Obstáculos: ${obsText}. Alertas: ${dangers}.`
          : `Sesión de ${durationMin} min, ${frames} frames. Camino libre.`;

        onSessionEndRef.current({
          durationSeconds,
          framesProcessed: frames,
          dangerCount: dangers,
          topObstacles,
          summary,
        });
      }

      // Limpieza
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      wsRef.current?.disconnect();
      wsRef.current = null;
      ttsManagerRef.current.reset();
      setLatestResult(null);
      setWsStatus('disconnected');
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      wsRef.current?.disconnect();
      wsRef.current = null;
      ttsManagerRef.current.stop();
    };
  }, [enabled, handleDetection, mode]);

  return { wsStatus, latestResult };
}
