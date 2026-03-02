/**
 * ============================================================================
 * NAVIA Frontend Web - TTS Queue Manager
 * ============================================================================
 * Singleton que gestiona TODA la síntesis de voz en la app web.
 * Previene cancelaciones con una cola de prioridades.
 *
 * SIEMPRE usa Piper TTS (voz natural VITS) para TODO el audio,
 * incluyendo INTERRUPT. Cae a Web Speech API solo si el backend
 * no responde (fallback automático).
 *
 * Prioridades:
 * - INTERRUPT (0): Alertas de peligro, errores críticos
 * - HIGH (1): Resultados de análisis, descripciones
 * - LOW (2): Feedback de UI ("Modo Navegación")
 * ============================================================================
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const TTS_ENDPOINT = `${API_BASE_URL}/api/v1/tts`

export enum TtsPriority {
  INTERRUPT = 0,
  HIGH = 1,
  LOW = 2,
}

interface QueueItem {
  text: string
  priority: TtsPriority
  useBackend: boolean
}

class TtsManager {
  private queue: QueueItem[] = []
  private isPlaying = false
  private enabled = true
  private currentAudio: HTMLAudioElement | null = null

  /**
   * Habla texto usando Piper TTS del backend (voz natural) para TODAS
   * las prioridades, incluyendo INTERRUPT.
   * Si el backend falla, cae a Web Speech API automáticamente.
   */
  speak(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    if (!this.enabled || !text) return

    if (priority === TtsPriority.INTERRUPT) {
      this.clearQueueAndStop()
      this.enqueue({ text, priority, useBackend: true })
    } else if (priority === TtsPriority.LOW) {
      // LOW se descarta si algo está sonando o hay cola
      if (this.isPlaying || this.queue.length > 0) return
      this.enqueue({ text, priority, useBackend: true })
    } else {
      this.enqueue({ text, priority, useBackend: true })
    }
  }

  /**
   * Habla texto usando Piper TTS del backend (voz natural, para textos largos).
   * Si el backend falla, cae a Web Speech API.
   */
  speakFromBackend(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    if (!this.enabled || !text) return

    if (priority === TtsPriority.INTERRUPT) {
      this.clearQueueAndStop()
      this.enqueue({ text, priority, useBackend: true })
    } else if (priority === TtsPriority.LOW) {
      if (this.isPlaying || this.queue.length > 0) return
      this.enqueue({ text, priority, useBackend: true })
    } else {
      this.enqueue({ text, priority, useBackend: true })
    }
  }

  /** Detiene todo el audio y limpia la cola. */
  stop(): void {
    this.clearQueueAndStop()
  }

  /** Retorna si hay audio reproduciéndose. */
  isSpeakingNow(): boolean {
    return this.isPlaying
  }

  /** Habilita o deshabilita TTS. */
  setEnabled(value: boolean): void {
    this.enabled = value
    if (!value) this.stop()
  }

  /** Retorna si TTS está habilitado. */
  isEnabled(): boolean {
    return this.enabled
  }

  // --- Internals ---

  private enqueue(item: QueueItem): void {
    this.queue.push(item)
    // Ordenar por prioridad (menor número = mayor prioridad)
    this.queue.sort((a, b) => a.priority - b.priority)
    this.processNext()
  }

  private processNext(): void {
    if (this.isPlaying || this.queue.length === 0) return

    const item = this.queue.shift()!
    this.isPlaying = true

    if (item.useBackend) {
      this.playFromBackend(item.text).catch(() => {
        // Asegurar que el estado esté limpio antes del fallback
        this.isPlaying = false
        this.currentAudio = null
        // Fallback a Web Speech API
        this.playWebSpeech(item.text)
      })
    } else {
      this.playWebSpeech(item.text)
    }
  }

  private playWebSpeech(text: string): void {
    if (typeof window === 'undefined' || !window.speechSynthesis) {
      this.isPlaying = false
      this.processNext()
      return
    }

    // Detener cualquier audio previo (backend o Web Speech)
    if (this.currentAudio) {
      this.currentAudio.pause()
      this.currentAudio = null
    }
    window.speechSynthesis.cancel()

    this.isPlaying = true
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'es-ES'
    utterance.rate = 0.9
    utterance.pitch = 1

    utterance.onend = () => {
      this.isPlaying = false
      this.processNext()
    }
    utterance.onerror = () => {
      this.isPlaying = false
      this.processNext()
    }

    window.speechSynthesis.speak(utterance)
  }

  private async playFromBackend(text: string): Promise<void> {
    // Detener cualquier audio previo antes de iniciar nuevo
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel()
    }
    if (this.currentAudio) {
      this.currentAudio.pause()
      this.currentAudio = null
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 8000)

    try {
      const response = await fetch(TTS_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      })

      clearTimeout(timeout)

      if (!response.ok) throw new Error(`TTS backend error: ${response.status}`)

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)

      await new Promise<void>((resolve, reject) => {
        const audio = new Audio(url)
        this.currentAudio = audio

        audio.onended = () => {
          URL.revokeObjectURL(url)
          this.currentAudio = null
          this.isPlaying = false
          resolve()
          this.processNext()
        }
        audio.onerror = () => {
          URL.revokeObjectURL(url)
          this.currentAudio = null
          reject(new Error('Audio playback error'))
        }

        audio.play().catch(reject)
      })
    } catch (err) {
      clearTimeout(timeout)
      // Re-throw para que el caller haga fallback a Web Speech
      throw err
    }
  }

  private clearQueueAndStop(): void {
    this.queue = []

    // Detener Web Speech
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel()
    }

    // Detener audio HTML5
    if (this.currentAudio) {
      this.currentAudio.pause()
      this.currentAudio = null
    }

    this.isPlaying = false
  }
}

// Singleton
export const ttsManager = new TtsManager()
