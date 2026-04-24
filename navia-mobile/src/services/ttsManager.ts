/**
 * TTS Manager — NAVIA
 *
 * Una sola voz (Piper backend) en toda la app.
 *
 * Estrategia:
 *   - Cache hit  → reproduce Piper al instante (sin latencia).
 *   - Cache miss → espera al backend Piper y reproduce.
 *
 * Las frases de navegación se repiten mucho, por lo que después de los primeros
 * segundos todas salen del caché y se escucha Piper sin latencia perceptible.
 */

import { Audio } from 'expo-av';
import { ttsCache } from './ttsCache';

export enum TtsPriority {
  INTERRUPT = 0,
  HIGH = 1,
  LOW = 2,
}

/**
 * Normaliza texto antes de enviarlo a Piper para mejorar pronunciación.
 * - Reemplaza caracteres que Piper no maneja bien (/, ·, etc.)
 * - Corrige palabras que el modelo español pronuncia mal
 */
function normalizeForTts(text: string): string {
  return text
    // Separadores visuales → pausa natural
    .replace(/\s*\/\s*/g, ' y ')
    .replace(/\s*·\s*/g, ', ')
    .replace(/\s*[-–]\s*/g, ', ')
    // Palabras con pronunciación problemática en el modelo es_MX
    .replace(/\bampliado\b/gi, 'aumentado')
    .replace(/\bampliada\b/gi, 'aumentada')
    .replace(/\babriendo\b/gi, 'iniciando')
    .replace(/\bcargando\b/gi, 'iniciando')
    .replace(/\briesgo\b/gi, 'peligro')
    // Abreviaturas comunes
    .replace(/\bIA\b/g, 'inteligencia artificial')
    .replace(/\bOCR\b/g, 'reconocimiento de texto')
    // Limpiar espacios múltiples
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

// Frases de navegación que se pre-generan al arrancar la app
const PREWARM_PHRASES = [
  'Camino libre.',
  'Atención.',
  'Precaución.',
  'Bienvenido a NAVIA, tu asistente visual. Selecciona un modo y toca Iniciar Cámara.',
];

class TtsManager {
  private static instance: TtsManager;

  private queue: Array<{ text: string; priority: TtsPriority; reading: boolean }> = [];
  private speaking = false;
  private enabled = true;
  private currentSound: Audio.Sound | null = null;
  private currentFetchController: AbortController | null = null;

  private constructor() {
    this.initAudio();
    // Inicializar caché y pre-generar frases comunes
    ttsCache.init().then(() => ttsCache.prewarm(PREWARM_PHRASES));
  }

  private async initAudio(): Promise<void> {
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
      });
    } catch { /**/ }
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

  /** Para contenido de lectura: preprocesa el texto para pausas naturales. */
  speakReading(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    if (!this.enabled || !text?.trim()) return;
    this.enqueue(this.preprocessForReading(normalizeForTts(text.trim())), priority, true);
  }

  stop(): void {
    this.queue = [];
    this.currentFetchController?.abort();
    this.currentFetchController = null;
    if (this.currentSound) {
      this.currentSound.stopAsync().catch(() => {});
      this.currentSound.unloadAsync().catch(() => {});
      this.currentSound = null;
    }
    this.speaking = false;
  }

  isSpeaking(): boolean { return this.speaking; }
  setEnabled(v: boolean): void { this.enabled = v; if (!v) this.stop(); }
  isEnabled(): boolean { return this.enabled; }

  private enqueue(text: string, priority: TtsPriority, reading: boolean): void {
    if (priority === TtsPriority.INTERRUPT) {
      this.stop();
      this.speaking = true;
      this.speakNow(text, reading, priority).finally(() => {
        this.speaking = false;
        this.processQueue();
      });
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
    this.speaking = true;
    this.speakNow(item.text, item.reading, item.priority).finally(() => {
      this.speaking = false;
      this.processQueue();
    });
  }

  private async speakNow(text: string, _reading: boolean, _priority: TtsPriority): Promise<void> {
    // 1. Buscar en caché — reproduce Piper al instante si está guardado
    try {
      const cached = await ttsCache.get(text);
      if (cached) {
        await this.playAudioFile(cached);
        return;
      }
    } catch { /**/ }

    // 2. Cache miss: esperar a Piper y reproducir (queda cacheado para futuras llamadas)
    try {
      const controller = new AbortController();
      this.currentFetchController = controller;
      const uri = await ttsCache.fetchAndStore(text, controller.signal);
      this.currentFetchController = null;
      await this.playAudioFile(uri);
    } catch (err: any) {
      this.currentFetchController = null;
      // AbortError: el usuario detuvo el TTS, no es un error
    }
  }

  private async playAudioFile(fileUri: string): Promise<void> {
    const { sound } = await Audio.Sound.createAsync(
      { uri: fileUri },
      { shouldPlay: true, volume: 1.0 }
    );
    this.currentSound = sound;

    await new Promise<void>((resolve) => {
      sound.setOnPlaybackStatusUpdate((status) => {
        if (!status.isLoaded) return;
        if (status.didJustFinish) {
          sound.unloadAsync().catch(() => {});
          this.currentSound = null;
          resolve();
        }
      });
    });
  }

  private preprocessForReading(text: string): string {
    return text
      .replace(/:\s+/g, ': ')
      .replace(/\.\s+/g, '. ')
      .replace(/\s*[-–]\s+/g, ', ')
      .replace(/[#*_~`|]/g, ' ')
      .replace(/[ \t]{2,}/g, ' ')
      .trim();
  }
}

export const ttsManager = TtsManager.getInstance();
