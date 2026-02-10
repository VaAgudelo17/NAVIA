/**
 * TTS inteligente para modos tiempo real (Navegación y Riesgo)
 *
 * Modo Navegación:
 * - Usa el summary del backend directamente (ya formateado como instrucción)
 * - Mínimo 3 segundos entre frases
 *
 * Modo Riesgo:
 * - Solo habla si has_danger === true
 * - Alertas critical bypassean el cooldown de 3s
 * - Interrumpe speech actual para alertas critical
 */

import * as Speech from 'expo-speech';
import { TTS_CONFIG, REALTIME_CONFIG } from '../constants/config';
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
  private isSpeaking = false;
  private mode: NaviaMode = 'navegacion';
  private lastSummary = '';

  setMode(mode: NaviaMode): void {
    this.mode = mode;
  }

  /**
   * Procesa resultado de detección según el modo activo.
   * summary: texto del backend (instrucción de navegación o alerta de riesgo)
   */
  async speakResult(
    summary: string,
    changes?: RealtimeChanges,
    riskData?: RiskData,
  ): Promise<void> {
    if (this.mode === 'riesgo') {
      return this.speakRiskAlert(riskData);
    }

    // Modo navegación: usa el summary del backend
    return this.speakNavigationSummary(summary, changes);
  }

  private async speakNavigationSummary(
    summary: string,
    changes?: RealtimeChanges,
  ): Promise<void> {
    // Solo hablar si hay cambio significativo o summary cambió
    if (!changes?.has_significant_change && summary === this.lastSummary) return;
    if (!summary) return;

    const now = Date.now();
    if (now - this.lastSpeakTime < REALTIME_CONFIG.ttsMinInterval) return;
    if (this.isSpeaking) return;

    this.lastSummary = summary;
    await this.doSpeak(summary);
  }

  private async speakRiskAlert(riskData?: RiskData): Promise<void> {
    if (!riskData?.has_danger || !riskData.alert_text) return;

    const now = Date.now();

    if (riskData.priority === 'critical') {
      // Critical: interrumpe todo y habla inmediatamente
      await Speech.stop();
      this.isSpeaking = false;
    } else {
      // High: respeta el cooldown
      if (now - this.lastSpeakTime < REALTIME_CONFIG.ttsMinInterval) return;
      if (this.isSpeaking) return;
    }

    await this.doSpeak(riskData.alert_text);
  }

  private async doSpeak(text: string): Promise<void> {
    this.isSpeaking = true;
    this.lastSpeakTime = Date.now();

    try {
      await Speech.stop();
      await new Promise<void>((resolve) => {
        Speech.speak(text, {
          language: TTS_CONFIG.language,
          pitch: TTS_CONFIG.pitch,
          rate: 1.1,
          onDone: () => resolve(),
          onError: () => resolve(),
        });
      });
    } catch {
      // Ignorar errores TTS en modo tiempo real
    } finally {
      this.isSpeaking = false;
    }
  }

  stop(): void {
    Speech.stop();
    this.isSpeaking = false;
  }

  reset(): void {
    this.stop();
    this.lastSpeakTime = 0;
    this.lastSummary = '';
  }
}
