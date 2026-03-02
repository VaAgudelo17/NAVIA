/**
 * TTS inteligente para navegación en tiempo real (Web)
 *
 * Delega toda la síntesis de voz al ttsManager central.
 *
 * Pipeline unificado (navegación + riesgo):
 * - Usa el summary del backend (instrucciones priorizadas de navegación)
 * - Alertas critical interrumpen el speech actual
 * - Alertas high tienen prioridad alta
 * - Instrucciones normales tienen cooldown de 3s
 */

import { type NaviaMode } from './api'
import { ttsManager, TtsPriority } from './ttsManager'

interface RealtimeChanges {
  appeared: string[]
  disappeared: string[]
  zone_changes: Array<{ name: string; from_zone: string; to_zone: string }>
  has_significant_change: boolean
}

interface GuidanceData {
  has_danger: boolean
  priority: string
  path_clear: boolean
}

export class RealtimeTtsManager {
  private lastSpeakTime = 0
  private minIntervalMs = 3000
  private mode: NaviaMode = 'navegacion'
  private lastSummary = ''

  setMode(mode: NaviaMode): void {
    this.mode = mode
  }

  /**
   * Habla el resultado de navegación unificada.
   *
   * Prioridad de speech:
   * 1. Peligro critical → interrumpe inmediatamente
   * 2. Peligro high → prioridad alta, respeta cooldown corto
   * 3. Navegación normal → prioridad baja, cooldown 3s
   */
  speakResult(
    summary: string,
    changes?: RealtimeChanges,
    guidanceData?: GuidanceData,
  ): void {
    if (!summary) return

    // Peligro critical: interrumpir speech actual
    if (guidanceData?.has_danger && guidanceData.priority === 'critical') {
      this.lastSummary = summary
      this.lastSpeakTime = Date.now()
      ttsManager.speak(summary, TtsPriority.INTERRUPT)
      return
    }

    // Peligro high: prioridad alta con cooldown reducido (1.5s)
    if (guidanceData?.has_danger && guidanceData.priority === 'high') {
      const now = Date.now()
      if (now - this.lastSpeakTime < 1500) return
      if (ttsManager.isSpeakingNow()) return

      this.lastSummary = summary
      this.lastSpeakTime = Date.now()
      ttsManager.speak(summary, TtsPriority.HIGH)
      return
    }

    // Navegación normal: cooldown estándar
    this.speakNavigationSummary(summary, changes)
  }

  private speakNavigationSummary(
    summary: string,
    changes?: RealtimeChanges,
  ): void {
    if (!changes?.has_significant_change && summary === this.lastSummary) return

    const now = Date.now()
    if (now - this.lastSpeakTime < this.minIntervalMs) return
    if (ttsManager.isSpeakingNow()) return

    this.lastSummary = summary
    this.lastSpeakTime = Date.now()
    ttsManager.speak(summary, TtsPriority.LOW)
  }

  stop(): void {
    ttsManager.stop()
  }

  reset(): void {
    ttsManager.stop()
    this.lastSpeakTime = 0
    this.lastSummary = ''
  }
}
