/**
 * Cliente WebSocket para detección en tiempo real
 */

import { API_BASE_URL } from '../constants/config';
import { RealtimeDetectionResult } from '../types/api';

type DetectionHandler = (data: RealtimeDetectionResult) => void;
type StatusHandler = (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void;

export class RealtimeWebSocket {
  private ws: WebSocket | null = null;
  private onDetection: DetectionHandler;
  private onStatus: StatusHandler;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private frameId = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(onDetection: DetectionHandler, onStatus: StatusHandler) {
    this.onDetection = onDetection;
    this.onStatus = onStatus;
  }

  connect(): void {
    const wsUrl = API_BASE_URL.replace(/^http/, 'ws') + '/api/v1/ws/realtime';

    this.onStatus('connecting');
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.onStatus('connected');
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'detection') {
          this.onDetection(msg as RealtimeDetectionResult);
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
    this.ws.send(message);
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
    this.reconnectTimer = setTimeout(
      () => this.connect(),
      2000 * this.reconnectAttempts,
    );
  }
}
