/**
 * TTS inteligente para modo tiempo real (Web)
 *
 * Solo narra cambios significativos usando Web Speech API:
 * - Objetos que aparecen/desaparecen
 * - Alertas de zona: cuando un objeto entra en "muy cerca" (peligro)
 *
 * Mínimo 3 segundos entre frases.
 */

interface ZoneChange {
  name: string
  from_zone: string
  to_zone: string
}

interface RealtimeChanges {
  appeared: string[]
  disappeared: string[]
  zone_changes: ZoneChange[]
  has_significant_change: boolean
}

export class RealtimeTtsManager {
  private lastSpeakTime = 0
  private minIntervalMs = 3000

  speakChanges(changes: RealtimeChanges): void {
    if (!changes.has_significant_change) return

    const now = Date.now()
    if (now - this.lastSpeakTime < this.minIntervalMs) return
    if (window.speechSynthesis.speaking) return

    let text = ''

    // Prioridad 1: Alertas de zona "muy cerca" (peligro)
    const dangerAlerts = (changes.zone_changes || []).filter(
      (zc) => zc.to_zone === 'muy_cerca'
    )
    if (dangerAlerts.length > 0) {
      const names = dangerAlerts.map((z) => z.name).join(', ')
      text = `Precaución, ${names} muy cerca`
    }
    // Prioridad 2: Objetos que aparecen
    else if (changes.appeared.length > 0 && changes.disappeared.length === 0) {
      const items = changes.appeared.join(', ')
      text = changes.appeared.length === 1
        ? `Nuevo: ${items}`
        : `Nuevos: ${items}`
    }
    // Prioridad 3: Objetos que desaparecen
    else if (changes.disappeared.length > 0 && changes.appeared.length === 0) {
      text = `${changes.disappeared.join(', ')} ya no visible`
    }
    // Prioridad 4: Ambos
    else if (changes.appeared.length > 0 && changes.disappeared.length > 0) {
      text = `Ahora: ${changes.appeared.join(', ')}`
    }

    if (!text) return

    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'es-ES'
    utterance.rate = 1.1
    window.speechSynthesis.speak(utterance)
    this.lastSpeakTime = now
  }

  stop(): void {
    window.speechSynthesis.cancel()
  }

  reset(): void {
    this.stop()
    this.lastSpeakTime = 0
  }
}
