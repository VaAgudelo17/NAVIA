/**
 * TTS inteligente para modos tiempo real (Web)
 *
 * Delega toda la síntesis de voz al ttsManager central.
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

import { type NaviaMode } from './api'
import { ttsManager, TtsPriority } from './ttsManager'

interface RealtimeChanges {
  appeared: string[]
  disappeared: string[]
  zone_changes: Array<{ name: string; from_zone: string; to_zone: string }>
  has_significant_change: boolean
}

interface RiskData {
  has_danger: boolean
  priority: string
  alert_text: string
}

export class RealtimeTtsManager {
  private lastSpeakTime = 0
  private minIntervalMs = 3000
  private mode: NaviaMode = 'navegacion'
  private lastSummary = ''

  setMode(mode: NaviaMode): void {
    this.mode = mode
  }

  speakResult(
    summary: string,
    changes?: RealtimeChanges,
    riskData?: RiskData,
  ): void {
    if (this.mode === 'riesgo') {
      this.speakRiskAlert(riskData)
      return
    }

    this.speakNavigationSummary(summary, changes)
  }

  private speakNavigationSummary(
    summary: string,
    changes?: RealtimeChanges,
  ): void {
    if (!changes?.has_significant_change && summary === this.lastSummary) return
    if (!summary) return

    const now = Date.now()
    if (now - this.lastSpeakTime < this.minIntervalMs) return
    if (ttsManager.isSpeakingNow()) return

    this.lastSummary = summary
    this.lastSpeakTime = Date.now()
    // speak() usa Piper TTS (voz natural) por defecto con fallback a Web Speech
    ttsManager.speak(summary, TtsPriority.LOW)
  }

  private speakRiskAlert(riskData?: RiskData): void {
    if (!riskData?.has_danger || !riskData.alert_text) return

    const now = Date.now()

    if (riskData.priority === 'critical') {
      this.lastSpeakTime = Date.now()
      ttsManager.speak(riskData.alert_text, TtsPriority.INTERRUPT)
    } else {
      if (now - this.lastSpeakTime < this.minIntervalMs) return
      if (ttsManager.isSpeakingNow()) return

      this.lastSpeakTime = Date.now()
      ttsManager.speak(riskData.alert_text, TtsPriority.HIGH)
    }
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
