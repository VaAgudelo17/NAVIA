/**
 * Hook para detección en tiempo real
 *
 * Captura frames periódicos de la cámara y los envía al backend
 * via WebSocket para detección con YOLOv8.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { CameraView } from 'expo-camera';
import { RealtimeWebSocket } from '../services/websocket';
import { RealtimeTtsManager } from '../services/realtimeTts';
import { RealtimeDetectionResult } from '../types/api';
import { REALTIME_CONFIG } from '../constants/config';

interface UseRealtimeDetectionOptions {
  cameraRef: React.RefObject<CameraView | null>;
  enabled: boolean;
  ttsEnabled: boolean;
}

export function useRealtimeDetection({
  cameraRef,
  enabled,
  ttsEnabled,
}: UseRealtimeDetectionOptions) {
  const [wsStatus, setWsStatus] = useState<string>('disconnected');
  const [latestResult, setLatestResult] = useState<RealtimeDetectionResult | null>(null);

  const wsRef = useRef<RealtimeWebSocket | null>(null);
  const ttsManagerRef = useRef<RealtimeTtsManager>(new RealtimeTtsManager());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isCapturing = useRef(false);

  const handleDetection = useCallback((data: RealtimeDetectionResult) => {
    setLatestResult(data);

    // Narrar cambios si TTS está habilitado
    if (ttsEnabled && data.changes) {
      ttsManagerRef.current.speakChanges(data.changes);
    }
  }, [ttsEnabled]);

  useEffect(() => {
    if (enabled) {
      // Conectar WebSocket
      wsRef.current = new RealtimeWebSocket(handleDetection, setWsStatus);
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
  }, [enabled, handleDetection]);

  return { wsStatus, latestResult };
}
