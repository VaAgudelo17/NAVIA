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
// instantáneamente sin esperar al backend (importante cuando el TTS está
// en un servidor lento como HF Spaces).
const PREWARM_PHRASES = [
  // Navegación
  'Camino libre.',
  'Atención.',
  'Precaución.',
  // Bienvenida
  'Bienvenido a NAVIA, tu asistente visual. Selecciona un modo y toca Iniciar Cámara.',
  // Confirmaciones de modo (ya normalizadas por normalizeForTts: galería→fotos guardadas, etc.)
  'Modo Navegación. Navegación asistida con peligro.',
  'Modo Exploración. Describe el entorno.',
  'Modo Lectura. Lee textos.',
  // Acciones de UI
  'Cámara lista. Toca la pantalla para capturar.',
  'Cerrando cámara.',
  'iniciando fotos guardadas.', // "Abriendo galería" después de normalizeForTts
  'Cerrando fotos guardadas.',  // "Cerrando galería" después de normalizeForTts
  'Detenido.',
  'Volviendo al inicio.',
  'Volviendo al historial.',
  'Procesando imagen, por favor espera.',
  'iniciando configuración visual.', // "Abriendo configuración..."
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
  // Promesa que resuelve cuando el sonido anterior termina de detenerse/descargarse.
  // speakNow la espera antes de reproducir uno nuevo para evitar doble voz.
  private stopPromise: Promise<void> | null = null;
  // Contador de "época": se incrementa en cada stop(). Permite que un
  // playAudioFile en vuelo detecte si hubo un stop mientras estaba cargando
  // el sonido, y descartarlo en lugar de reproducirlo (evita voces mezcladas
  // cuando el usuario interrumpe durante una carga lenta del backend).
  private playEpoch = 0;
  // Listeners para que la UI pueda reaccionar a cambios del estado de habla
  // (ej. animación de onda de voz que solo se muestra mientras suena el TTS).
  private speakingListeners: Set<(speaking: boolean) => void> = new Set();

  /** Setter que actualiza el estado y notifica listeners. */
  private setSpeakingState(v: boolean): void {
    if (this.speaking === v) return;
    this.speaking = v;
    this.speakingListeners.forEach((cb) => { try { cb(v); } catch { /**/ } });
  }

  /** Suscribirse a cambios del estado de habla. Devuelve una función de cleanup. */
  onSpeakingChange(callback: (speaking: boolean) => void): () => void {
    this.speakingListeners.add(callback);
    return () => { this.speakingListeners.delete(callback); };
  }

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
    // Invalidar cualquier playAudioFile en vuelo para que descarte su sonido
    // si termina de cargar después de este stop.
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
    // Esperar a que cualquier sonido anterior termine de detenerse antes de
    // empezar uno nuevo. Sin esto, dos voces se solapan brevemente.
    if (this.stopPromise) {
      const p = this.stopPromise;
      this.stopPromise = null;
      try { await p; } catch { /**/ }
    }

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
    // Capturar la época ANTES de crear el sonido. Si stop() se llama mientras
    // createAsync está cargando (puede tomar segundos), playEpoch cambia.
    // Después comparamos: si la época cambió, descartamos el sonido sin
    // reproducirlo. Cero riesgo de "voces mezcladas".
    const myEpoch = this.playEpoch;

    const { sound } = await Audio.Sound.createAsync(
      { uri: fileUri },
      { shouldPlay: false, volume: 1.0 }
    );

    // ¿Stop fue llamado durante la carga? Descartar este sonido.
    if (myEpoch !== this.playEpoch) {
      sound.unloadAsync().catch(() => {});
      return;
    }

    this.currentSound = sound;

    // Registrar listener antes de reproducir para no perder eventos
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

    // Iniciar reproducción ahora que currentSound ya está registrado
    try {
      await sound.playAsync();
    } catch { /* sound puede haber sido detenido en una stop() concurrente */ }

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
