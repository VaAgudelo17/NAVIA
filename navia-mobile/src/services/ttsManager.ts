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
    // "galería" se pronuncia como "gaderia" en el modelo Piper es_MX
    .replace(/\bgalería\b/gi, 'fotos guardadas')
    .replace(/\bgaleria\b/gi, 'fotos guardadas')
    // Abreviaturas técnicas
    .replace(/\bIA\b/g, 'inteligencia artificial')
    .replace(/\bOCR\b/g, 'reconocimiento óptico de texto')
    // Meses abreviados en español (formato corto de toLocaleDateString)
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
    // Unidades de tiempo / cantidad
    .replace(/(\d+(?:[.,]\d+)?)\s*ms\b/gi, '$1 milisegundos')
    .replace(/(\d+(?:[.,]\d+)?)\s*min\b/gi, '$1 minutos')
    .replace(/(\d+(?:[.,]\d+)?)\s*seg\b/gi, '$1 segundos')
    .replace(/(\d+(?:[.,]\d+)?)\s*hr?s?\b/gi, '$1 horas')
    // Porcentajes: "85%" → "85 por ciento"
    .replace(/(\d+(?:[.,]\d+)?)\s*%/g, '$1 por ciento')
    // Precios: descartar decimales que sean .00 o .0 (suena raro: "veinte mil punto cero cero")
    // 20000.00 → 20000  ;  5.0 → 5  ;  100.5 → se mantiene
    .replace(/(\d+)[.,]0+(?=\D|$)/g, '$1')
    // Decimales que SÍ tienen valor: "19.99" → "19 con 99"
    .replace(/(\d+)[.,](\d+)/g, '$1 con $2')
    // Símbolo de pesos pegado: "$5000" → "5000 pesos"
    .replace(/\$\s*(\d+)/g, '$1 pesos')
    // Limpiar espacios múltiples
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

// Frases que se pre-generan al arrancar la app para que se reproduzcan
// instantáneamente sin esperar al backend.
const PREWARM_PHRASES = [
  'Camino libre.',
  'Atención.',
  'Precaución.',
  'Bienvenido a NAVIA, tu asistente visual. Selecciona un modo y toca Iniciar Cámara.',
  'Modo Navegación. Navegación asistida con peligro.',
  'Modo Exploración. Describe el entorno.',
  'Modo Lectura. Lee textos.',
  'Cámara lista. Toca la pantalla para capturar.',
  'Cerrando cámara.',
  'iniciando fotos guardadas.',
  'Cerrando fotos guardadas.',
  'Detenido.',
  'Volviendo al inicio.',
  'Volviendo al historial.',
  'Procesando imagen, por favor espera.',
  'iniciando configuración visual.',
  'Configuración guardada. De vuelta al inicio.',
  'iniciando historial.',
  'Voz activada.',
  'Voz desactivada.',
];

class TtsManager {
  private static instance: TtsManager;

  private queue: Array<{ text: string; priority: TtsPriority; reading: boolean }> = [];
  private speaking = false;
  private enabled = true;
  private currentSound: Audio.Sound | null = null;
  private currentFetchController: AbortController | null = null;
  private stopPromise: Promise<void> | null = null;
  private playEpoch = 0;
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
    this.initAudio();
    ttsCache.init().then(() => ttsCache.prewarm(PREWARM_PHRASES));
  }

  private async initAudio(): Promise<void> {
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
      });
    } catch (e) {
      console.warn('[TTS] initAudio error:', e);
    }
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
    this.enqueue(this.preprocessForReading(normalizeForTts(text.trim())), priority, true);
  }

  /** Pre-guarda audio base64 del backend en caché con la misma clave que usaría speakReading. */
  async preCacheReadingAudio(text: string, audioBase64: string, ext: 'mp3' | 'wav'): Promise<void> {
    const key = this.preprocessForReading(normalizeForTts(text.trim()));
    await ttsCache.storeBase64(key, audioBase64, ext);
  }

  stop(): void {
    this.playEpoch++;
    this.queue = [];
    this.currentFetchController?.abort();
    this.currentFetchController = null;
    const sound = this.currentSound;
    this.currentSound = null;
    this.setSpeakingState(false);
    if (sound) {
      this.stopPromise = (async () => {
        try { await sound.stopAsync(); } catch { /**/ }
        try { await sound.unloadAsync(); } catch { /**/ }
      })();
    }
  }

  isSpeaking(): boolean { return this.speaking; }
  setEnabled(v: boolean): void { this.enabled = v; if (!v) this.stop(); }
  isEnabled(): boolean { return this.enabled; }

  private enqueue(text: string, priority: TtsPriority, reading: boolean): void {
    if (priority === TtsPriority.INTERRUPT) {
      this.stop();
      this.setSpeakingState(true);
      this.speakNow(text, reading, priority).finally(() => {
        this.setSpeakingState(false);
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
    this.setSpeakingState(true);
    this.speakNow(item.text, item.reading, item.priority).finally(() => {
      this.setSpeakingState(false);
      this.processQueue();
    });
  }

  private async speakNow(text: string, _reading: boolean, _priority: TtsPriority): Promise<void> {
    if (this.stopPromise) {
      const p = this.stopPromise;
      this.stopPromise = null;
      try { await p; } catch { /**/ }
    }

    // 1. Cache hit → reproducir al instante
    try {
      const cached = await ttsCache.get(text);
      if (cached) {
        await this.playAudioFile(cached);
        return;
      }
    } catch (err: any) {
      console.warn('[TTS] Error en caché:', err?.message ?? err);
    }

    // 2. Cache miss → descargar del backend y reproducir
    try {
      const controller = new AbortController();
      this.currentFetchController = controller;
      const uri = await ttsCache.fetchAndStore(text, controller.signal);
      this.currentFetchController = null;
      await this.playAudioFile(uri);
    } catch (err: any) {
      this.currentFetchController = null;
      if (err?.name !== 'AbortError') {
        console.warn('[TTS] Error reproduciendo audio:', err?.message ?? err);
      }
    }
  }

  private async playAudioFile(fileUri: string): Promise<void> {
    const myEpoch = this.playEpoch;

    const { sound } = await Audio.Sound.createAsync(
      { uri: fileUri },
      { shouldPlay: false, volume: 1.0 }
    );

    if (myEpoch !== this.playEpoch) {
      sound.unloadAsync().catch(() => {});
      return;
    }

    this.currentSound = sound;

    let resolved = false;
    const playbackEnded = new Promise<void>((resolve) => {
      sound.setOnPlaybackStatusUpdate((status) => {
        if (resolved) return;
        if (!status.isLoaded) {
          resolved = true;
          resolve();
          return;
        }
        if (status.didJustFinish) {
          resolved = true;
          sound.unloadAsync().catch(() => {});
          if (this.currentSound === sound) this.currentSound = null;
          resolve();
        }
      });
    });

    try {
      await sound.playAsync();
    } catch { /**/ }

    await playbackEnded;
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
