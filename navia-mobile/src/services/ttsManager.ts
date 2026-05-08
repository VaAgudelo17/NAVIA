/**
 * TTS Manager — NAVIA
 *
 * Usa expo-speech (síntesis nativa del dispositivo) para evitar
 * los problemas de expo-av con la Nueva Arquitectura de React Native.
 *
 * La interfaz pública es idéntica a la versión anterior para que el
 * resto del código no necesite cambios.
 */

import * as Speech from 'expo-speech';

export enum TtsPriority {
  INTERRUPT = 0,
  HIGH = 1,
  LOW = 2,
}

/**
 * Normaliza texto antes de enviarlo al TTS para mejorar pronunciación.
 */
function normalizeForTts(text: string): string {
  return text
    .replace(/\s*\/\s*/g, ' y ')
    .replace(/\s*·\s*/g, ', ')
    .replace(/\s*[-–]\s*/g, ', ')
    .replace(/\bampliado\b/gi, 'aumentado')
    .replace(/\bampliada\b/gi, 'aumentada')
    .replace(/\babriendo\b/gi, 'iniciando')
    .replace(/\bcargando\b/gi, 'iniciando')
    .replace(/\briesgo\b/gi, 'peligro')
    .replace(/\bgalería\b/gi, 'fotos guardadas')
    .replace(/\bgaleria\b/gi, 'fotos guardadas')
    .replace(/\bIA\b/g, 'inteligencia artificial')
    .replace(/\bOCR\b/g, 'reconocimiento de texto')
    .replace(/\bene\.?(?=\s|,|$)/gi, 'enero')
    .replace(/\bfeb\.?(?=\s|,|$)/gi, 'febrero')
    .replace(/\bmar\.?(?=\s|,|$)/gi, 'marzo')
    .replace(/\babr\.?(?=\s|,|$)/gi, 'abril')
    .replace(/\bmay\.?(?=\s|,|$)/gi, 'mayo')
    .replace(/\bjun\.?(?=\s|,|$)/gi, 'junio')
    .replace(/\bjul\.?(?=\s|,|$)/gi, 'julio')
    .replace(/\bago\.?(?=\s|,|$)/gi, 'agosto')
    .replace(/\bsept?\.?(?=\s|,|$)/gi, 'septiembre')
    .replace(/\boct\.?(?=\s|,|$)/gi, 'octubre')
    .replace(/\bnov\.?(?=\s|,|$)/gi, 'noviembre')
    .replace(/\bdic\.?(?=\s|,|$)/gi, 'diciembre')
    .replace(/(\d+(?:[.,]\d+)?)\s*ms\b/gi, '$1 milisegundos')
    .replace(/(\d+(?:[.,]\d+)?)\s*min\b/gi, '$1 minutos')
    .replace(/(\d+(?:[.,]\d+)?)\s*seg\b/gi, '$1 segundos')
    .replace(/(\d+(?:[.,]\d+)?)\s*hr?s?\b/gi, '$1 horas')
    .replace(/(\d+(?:[.,]\d+)?)\s*%/g, '$1 por ciento')
    .replace(/(\d+)[.,]0+(?=\D|$)/g, '$1')
    .replace(/(\d+)[.,](\d+)/g, '$1 con $2')
    .replace(/\$\s*(\d+)/g, '$1 pesos')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

class TtsManager {
  private static instance: TtsManager;

  private queue: Array<{ text: string; priority: TtsPriority; reading: boolean }> = [];
  private speaking = false;
  private enabled = true;
  private speakingListeners: Set<(speaking: boolean) => void> = new Set();

  private setSpeakingState(v: boolean): void {
    if (this.speaking === v) return;
    this.speaking = v;
    this.speakingListeners.forEach((cb) => { try { cb(v); } catch { /**/ } });
  }

  onSpeakingChange(callback: (speaking: boolean) => void): () => void {
    this.speakingListeners.add(callback);
    return () => { this.speakingListeners.delete(callback); };
  }

  private constructor() {
    // expo-speech no necesita inicialización asíncrona
  }

  static getInstance(): TtsManager {
    if (!TtsManager.instance) TtsManager.instance = new TtsManager();
    return TtsManager.instance;
  }

  speak(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    if (!this.enabled || !text?.trim()) return;
    this.enqueue(normalizeForTts(text.trim()), priority, false);
  }

  speakFromBackend(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    if (!this.enabled || !text?.trim()) return;
    this.enqueue(normalizeForTts(text.trim()), priority, false);
  }

  speakReading(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    if (!this.enabled || !text?.trim()) return;
    this.enqueue(normalizeForTts(text.trim()), priority, true);
  }

  stop(): void {
    this.queue = [];
    Speech.stop();
    this.setSpeakingState(false);
  }

  isSpeaking(): boolean { return this.speaking; }
  setEnabled(v: boolean): void { this.enabled = v; if (!v) this.stop(); }
  isEnabled(): boolean { return this.enabled; }

  private enqueue(text: string, priority: TtsPriority, reading: boolean): void {
    if (priority === TtsPriority.INTERRUPT) {
      this.queue = [];
      Speech.stop();
      this.setSpeakingState(true);
      this.speakNow(text, reading);
      return;
    }
    if (priority === TtsPriority.HIGH && this.speaking) return;
    if (priority === TtsPriority.LOW && (this.speaking || this.queue.length > 0)) return;

    this.queue.push({ text, priority, reading });
    this.processQueue();
  }

  private processQueue(): void {
    if (this.speaking || this.queue.length === 0) return;
    this.queue.sort((a, b) => a.priority - b.priority);
    const item = this.queue.shift()!;
    this.setSpeakingState(true);
    this.speakNow(item.text, item.reading);
  }

  private speakNow(text: string, reading: boolean): void {
    // Velocidad adaptada: lectura más lenta para documentos largos
    const rate = reading ? 0.82 : 0.90;

    Speech.speak(text, {
      language: 'es-ES',
      rate,
      pitch: 1.0,
      onDone: () => {
        this.setSpeakingState(false);
        this.processQueue();
      },
      onError: (err) => {
        console.warn('[TTS] Error de voz:', err);
        this.setSpeakingState(false);
        this.processQueue();
      },
      onStopped: () => {
        this.setSpeakingState(false);
      },
    });
  }

  /**
   * Stub de compatibilidad — ya no se necesita prewarm porque expo-speech
   * usa el motor nativo del dispositivo (siempre listo, sin latencia).
   */
  prewarm?(_phrases: string[]): void { /* no-op */ }
}

export const ttsManager = TtsManager.getInstance();
