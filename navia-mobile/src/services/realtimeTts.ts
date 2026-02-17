/**
 * TTS inteligente para modos tiempo real (Navegación y Riesgo)
 *
 * Delega al ttsManager singleton para evitar conflictos de audio.
 *
 * Modo Navegación:
 * - Usa el summary del backend (ya formateado como instrucción)
 * - Mínimo 3 segundos entre frases
 * - Prioridad LOW (se descarta si algo está sonando)
 *
 * Modo Riesgo:
 * - Solo habla si has_danger === true
 * - Alertas critical usan INTERRUPT (interrumpen todo)
 * - Alertas high usan HIGH (se encolan)
 */

import { ttsManager, TtsPriority } from './ttsManager';
import { REALTIME_CONFIG } from '../constants/config';
import { NaviaMode } from '../types/api';

interface RealtimeChanges {
  appeared: string[];
  disappeared: string[];
  zone_changes: Array<{ name: string; from_zone: string; to_zone: string }>;
  has_significant_change: boolean;
}

interface RiskData {
  has_danger: boolean;
  priority: string;
  alert_text: string;
}

export class RealtimeTtsManager {
  private lastSpeakTime = 0;
  private mode: NaviaMode = 'navegacion';
  private lastSummary = '';

  setMode(mode: NaviaMode): void {
    this.mode = mode;
  }

  /**
   * Procesa resultado de detección según el modo activo.
   */
  speakResult(
    summary: string,
    changes?: RealtimeChanges,
    riskData?: RiskData,
  ): void {
    if (this.mode === 'riesgo') {
      this.speakRiskAlert(riskData);
      return;
    }
    this.speakNavigationSummary(summary, changes);
  }

  private speakNavigationSummary(
    summary: string,
    changes?: RealtimeChanges,
  ): void {
    // Solo hablar si hay cambio significativo o summary cambió
    if (!changes?.has_significant_change && summary === this.lastSummary) return;
    if (!summary) return;

    const now = Date.now();
    if (now - this.lastSpeakTime < REALTIME_CONFIG.ttsMinInterval) return;
    if (ttsManager.isSpeaking()) return;

    this.lastSummary = summary;
    this.lastSpeakTime = now;
    // Prioridad LOW: no interrumpe resultados de análisis en curso
    ttsManager.speak(summary, TtsPriority.LOW);
  }

  private speakRiskAlert(riskData?: RiskData): void {
    if (!riskData?.has_danger || !riskData.alert_text) return;

    const now = Date.now();

    if (riskData.priority === 'critical') {
      // INTERRUPT: peligro crítico debe oírse AHORA
      this.lastSpeakTime = now;
      ttsManager.speak(riskData.alert_text, TtsPriority.INTERRUPT);
    } else {
      // HIGH: respeta cooldown, se encola
      if (now - this.lastSpeakTime < REALTIME_CONFIG.ttsMinInterval) return;
      if (ttsManager.isSpeaking()) return;
      this.lastSpeakTime = now;
      ttsManager.speak(riskData.alert_text, TtsPriority.HIGH);
    }
  }

  stop(): void {
    ttsManager.stop();
  }

  reset(): void {
    ttsManager.stop();
    this.lastSpeakTime = 0;
    this.lastSummary = '';
  }
}
