/**
 * Cliente WebSocket para detección en tiempo real
 * Usa pipeline unificado de navegación + riesgo
 */

import { API_BASE_URL } from '../constants/config';
import { RealtimeDetectionResult, NaviaMode } from '../types/api';

type DetectionHandler = (data: RealtimeDetectionResult) => void;
type StatusHandler = (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void;
type SceneIntroHandler = (description: string) => void;

export class RealtimeWebSocket {
  private ws: WebSocket | null = null;
  private onDetection: DetectionHandler;
  private onStatus: StatusHandler;
  private onSceneIntro?: SceneIntroHandler;
  private mode: NaviaMode;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private frameId = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    onDetection: DetectionHandler,
    onStatus: StatusHandler,
    mode: NaviaMode = 'navegacion',
    onSceneIntro?: SceneIntroHandler,
  ) {
    this.onDetection = onDetection;
    this.onStatus = onStatus;
    this.mode = mode;
    this.onSceneIntro = onSceneIntro;
  }

  connect(): void {
    const wsUrl = API_BASE_URL.replace(/^http/, 'ws') + '/api/v1/ws/realtime';

    this.onStatus('connecting');
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.onStatus('connected');
      // Enviar configuración de modo al conectar
      try {
        this.ws?.send(JSON.stringify({
          type: 'config',
          mode: this.mode,
        }));
      } catch (e) {
        console.warn('WS send config error:', e);
      }
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'detection') {
          this.onDetection(msg as RealtimeDetectionResult);
        } else if (msg.type === 'scene_intro' && msg.description) {
          this.onSceneIntro?.(msg.description as string);
        }
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    this.ws.onerror = () => {
      this.onStatus('error');
    };

    this.ws.onclose = () => {
      this.onStatus('disconnected');
      this.attemptReconnect();
    };
  }

  sendFrame(base64Data: string): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;

    this.frameId++;
    const message = JSON.stringify({
      type: 'frame',
      data: base64Data,
      frame_id: this.frameId,
      timestamp: Date.now(),
    });
    try {
      this.ws.send(message);
    } catch (e) {
      console.warn('WS sendFrame error:', e);
    }
  }

  disconnect(): void {
    this.maxReconnectAttempts = 0;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    this.reconnectAttempts++;
    // Exponential backoff: 1s, 2s, 4s, 8s, 16s (más agresivo en el primer intento)
    const delayMs = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 16000);
    this.reconnectTimer = setTimeout(
      () => this.connect(),
      delayMs,
    );
  }
}
