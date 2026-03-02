/**
 * TTS inteligente para navegación en tiempo real (Mobile)
 *
 * Delega al ttsManager singleton para evitar conflictos de audio.
 *
 * Pipeline unificado (navegación + riesgo):
 * - Usa el summary del backend (instrucciones priorizadas de navegación)
 * - Alertas critical interrumpen el speech actual (INTERRUPT)
 * - Alertas high tienen prioridad alta con cooldown reducido
 * - Instrucciones normales tienen cooldown de 3s
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

interface GuidanceData {
  has_danger: boolean;
  priority: string;
  path_clear: boolean;
}

export class RealtimeTtsManager {
  private lastSpeakTime = 0;
  private mode: NaviaMode = 'navegacion';
  private lastSummary = '';

  setMode(mode: NaviaMode): void {
    this.mode = mode;
  }

  /**
   * Habla el resultado de navegación unificada.
   *
   * Prioridad de speech:
   * 1. Peligro critical → interrumpe inmediatamente
   * 2. Peligro high → prioridad alta, cooldown reducido (1.5s)
   * 3. Navegación normal → prioridad baja, cooldown 3s
   */
  speakResult(
    summary: string,
    changes?: RealtimeChanges,
    guidanceData?: GuidanceData,
  ): void {
    if (!summary) return;

    // Peligro critical: interrumpir speech actual
    if (guidanceData?.has_danger && guidanceData.priority === 'critical') {
      this.lastSummary = summary;
      this.lastSpeakTime = Date.now();
      ttsManager.speak(summary, TtsPriority.INTERRUPT);
      return;
    }

    // Peligro high: prioridad alta con cooldown reducido
    if (guidanceData?.has_danger && guidanceData.priority === 'high') {
      const now = Date.now();
      if (now - this.lastSpeakTime < 1500) return;
      if (ttsManager.isSpeaking()) return;

      this.lastSummary = summary;
      this.lastSpeakTime = Date.now();
      ttsManager.speak(summary, TtsPriority.HIGH);
      return;
    }

    // Navegación normal: cooldown estándar
    this.speakNavigationSummary(summary, changes);
  }

  private speakNavigationSummary(
    summary: string,
    changes?: RealtimeChanges,
  ): void {
    if (!changes?.has_significant_change && summary === this.lastSummary) return;
    if (!summary) return;

    const now = Date.now();
    if (now - this.lastSpeakTime < REALTIME_CONFIG.ttsMinInterval) return;
    if (ttsManager.isSpeaking()) return;

    this.lastSummary = summary;
    this.lastSpeakTime = now;
    ttsManager.speak(summary, TtsPriority.LOW);
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
