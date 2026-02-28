/**
 * TTS Queue Manager para NAVIA
 *
 * Singleton que controla TODA la síntesis de voz en la app.
 * Resuelve los problemas de voces que se cancelan entre sí
 * usando una cola con prioridades.
 *
 * Por defecto usa Piper TTS (voz natural VITS) para TODO el audio.
 * Cae a expo-speech si el backend no responde.
 * Excepción: INTERRUPT siempre usa expo-speech (latencia cero).
 */

import * as Speech from 'expo-speech';
import { Audio } from 'expo-av';
import { File as ExpoFile, Paths } from 'expo-file-system';
import { TTS_CONFIG, API_BASE_URL, API_ENDPOINTS } from '../constants/config';

// ============================================================================
// PRIORIDADES
// ============================================================================

export enum TtsPriority {
  /** Alertas de peligro, errores. Interrumpe todo y habla ya. */
  INTERRUPT = 0,
  /** Resultados de análisis, descripciones. Se encola tras el speech actual. */
  HIGH = 1,
  /** Feedback de UI ("Modo Navegación"). Se descarta si algo está sonando. */
  LOW = 2,
}

// ============================================================================
// TIPOS INTERNOS
// ============================================================================

interface QueueItem {
  text: string;
  priority: TtsPriority;
  useBackend: boolean;
}

// ============================================================================
// TTS MANAGER
// ============================================================================

class TtsManager {
  private static instance: TtsManager;

  private queue: QueueItem[] = [];
  private playing = false;
  private currentSound: Audio.Sound | null = null;
  private enabled = true;

  private constructor() {}

  static getInstance(): TtsManager {
    if (!TtsManager.instance) {
      TtsManager.instance = new TtsManager();
    }
    return TtsManager.instance;
  }

  // ==========================================================================
  // API PÚBLICA
  // ==========================================================================

  /**
   * Habla usando Piper TTS del backend por defecto (voz natural).
   * Cae a expo-speech automáticamente si el backend no responde.
   * INTERRUPT siempre usa expo-speech para respuesta instantánea.
   */
  speak(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    if (!this.enabled || !text?.trim()) return;
    // INTERRUPT se fuerza a local en enqueue(), aquí marcamos backend para el resto
    this.enqueue(text.trim(), priority, true);
  }

  /**
   * Habla con Piper TTS del backend (voz natural). Para descripciones largas.
   * Si el backend no responde, cae a expo-speech automáticamente.
   */
  speakFromBackend(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    if (!this.enabled || !text?.trim()) return;
    this.enqueue(text.trim(), priority, true);
  }

  /**
   * Detiene todo: speech actual + limpia la cola.
   */
  stop(): void {
    this.queue = [];
    this.stopCurrent();
  }

  /**
   * ¿Está hablando o hay algo en cola?
   */
  isSpeaking(): boolean {
    return this.playing;
  }

  /**
   * Activa o desactiva el TTS globalmente.
   */
  setEnabled(value: boolean): void {
    this.enabled = value;
    if (!value) {
      this.stop();
    }
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  // ==========================================================================
  // COLA DE PRIORIDADES
  // ==========================================================================

  private enqueue(text: string, priority: TtsPriority, useBackend: boolean): void {
    // INTERRUPT: limpia todo y ejecuta inmediatamente
    if (priority === TtsPriority.INTERRUPT) {
      this.queue = [];
      this.stopCurrent();
      // Pequeño delay para asegurar que Speech.stop() terminó
      setTimeout(() => {
        this.queue.push({ text, priority, useBackend: false }); // INTERRUPT siempre usa expo-speech (velocidad)
        this.processQueue();
      }, 50);
      return;
    }

    // LOW: descartar si algo está sonando o hay cola
    if (priority === TtsPriority.LOW) {
      if (this.playing || this.queue.length > 0) {
        return;
      }
    }

    this.queue.push({ text, priority, useBackend });
    this.processQueue();
  }

  private async processQueue(): Promise<void> {
    if (this.playing || this.queue.length === 0) return;

    // Ordenar: INTERRUPT (0) primero, LOW (2) último
    this.queue.sort((a, b) => a.priority - b.priority);

    const item = this.queue.shift()!;
    this.playing = true;

    try {
      if (item.useBackend) {
        try {
          await this.playBackendTts(item.text);
        } catch {
          // Fallback a expo-speech
          console.log('[TTS] Backend falló, usando expo-speech como fallback');
          await this.playExpoSpeech(item.text);
        }
      } else {
        await this.playExpoSpeech(item.text);
      }
    } catch (err) {
      console.warn('[TTS] Error reproduciendo:', err);
    } finally {
      this.playing = false;
      // Procesar siguiente en cola
      if (this.queue.length > 0) {
        this.processQueue();
      }
    }
  }

  // ==========================================================================
  // TIER 1: EXPO-SPEECH (instantáneo, device-side)
  // ==========================================================================

  private playExpoSpeech(text: string): Promise<void> {
    return new Promise<void>((resolve) => {
      // Safety timeout: si los callbacks no se disparan en 30s, resolver de todas formas
      const safetyTimeout = setTimeout(() => {
        console.warn('[TTS] Safety timeout en expo-speech, forzando resolve');
        resolve();
      }, 30000);

      // Limpiar cualquier speech anterior
      Speech.stop();

      // Pequeño delay después de stop() para que el sistema limpie
      setTimeout(() => {
        Speech.speak(text, {
          language: TTS_CONFIG.language,
          pitch: TTS_CONFIG.pitch,
          rate: TTS_CONFIG.rate,
          onDone: () => {
            clearTimeout(safetyTimeout);
            resolve();
          },
          onError: (err) => {
            clearTimeout(safetyTimeout);
            console.warn('[TTS] expo-speech error:', err);
            resolve();
          },
          onStopped: () => {
            clearTimeout(safetyTimeout);
            resolve();
          },
        });
      }, 100);
    });
  }

  // ==========================================================================
  // TIER 2: PIPER TTS VIA BACKEND (voz VITS natural)
  // ==========================================================================

  private async playBackendTts(text: string): Promise<void> {
    const fileName = `navia_tts_${Date.now()}.wav`;
    let tempFile: InstanceType<typeof ExpoFile> | null = null;

    try {
      // 1. Fetch del audio WAV desde el backend (POST con JSON body)
      const controller = new AbortController();
      const fetchTimeout = setTimeout(() => controller.abort(), 8000);

      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.TTS}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });

      clearTimeout(fetchTimeout);

      if (!response.ok) {
        throw new Error(`TTS backend error: ${response.status}`);
      }

      // 2. Leer respuesta como ArrayBuffer y escribir a disco con la nueva File API
      const arrayBuffer = await response.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);

      if (bytes.length < 100) {
        throw new Error('TTS response too small, likely empty');
      }

      tempFile = new ExpoFile(Paths.cache, fileName);
      tempFile.create({ overwrite: true });
      tempFile.write(bytes);

      // 3. Reproducir con expo-av
      const { sound } = await Audio.Sound.createAsync({ uri: tempFile.uri });
      this.currentSound = sound;

      await new Promise<void>((resolve, reject) => {
        const safetyTimeout = setTimeout(() => {
          console.warn('[TTS] Safety timeout en audio backend');
          resolve();
        }, 60000);

        sound.setOnPlaybackStatusUpdate((status) => {
          if (status.isLoaded && status.didJustFinish) {
            clearTimeout(safetyTimeout);
            resolve();
          }
        });
        sound.playAsync().catch((err) => {
          clearTimeout(safetyTimeout);
          reject(err);
        });
      });

      // 4. Limpiar
      await sound.unloadAsync().catch(() => {});
      this.currentSound = null;

      try { tempFile.delete(); } catch {}
    } catch (error) {
      // Limpiar archivo temporal si existe
      try { if (tempFile?.exists) tempFile.delete(); } catch {}
      throw error; // El caller hará fallback a expo-speech
    }
  }

  // ==========================================================================
  // CONTROL
  // ==========================================================================

  private stopCurrent(): void {
    // Parar expo-speech
    Speech.stop();

    // Parar expo-av sound
    if (this.currentSound) {
      try {
        this.currentSound.stopAsync().catch(() => {});
        this.currentSound.unloadAsync().catch(() => {});
      } catch {
        // Ya estaba detenido/descargado
      }
      this.currentSound = null;
    }

    this.playing = false;
  }
}

// ============================================================================
// EXPORT SINGLETON
// ============================================================================

export const ttsManager = TtsManager.getInstance();
