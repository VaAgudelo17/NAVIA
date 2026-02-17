/**
 * TTS Queue Manager para NAVIA
 *
 * Singleton que controla TODA la síntesis de voz en la app.
 * Resuelve los problemas de voces que se cancelan entre sí
 * usando una cola con prioridades.
 *
 * Tier 1: expo-speech (instantáneo) → feedback de UI
 * Tier 2: Piper TTS en backend (voz VITS) → descripciones largas
 */

import * as Speech from 'expo-speech';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';
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
   * Habla con expo-speech (instantáneo). Para feedback de UI y textos cortos.
   */
  speak(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    if (!this.enabled || !text?.trim()) return;
    this.enqueue(text.trim(), priority, false);
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
    // Timeout de 8 segundos para el fetch
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);

    try {
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.TTS}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        throw new Error(`TTS backend error: ${response.status}`);
      }

      // Leer el audio como base64
      const blob = await response.blob();
      const reader = new FileReader();

      const base64Data = await new Promise<string>((resolve, reject) => {
        reader.onloadend = () => {
          const result = reader.result as string;
          // Extraer solo la parte base64 del data URI
          const base64 = result.split(',')[1];
          resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });

      // Guardar a archivo temporal
      const tempPath = `${FileSystem.cacheDirectory}navia_tts_${Date.now()}.wav`;
      await FileSystem.writeAsStringAsync(tempPath, base64Data, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Reproducir con expo-av
      const { sound } = await Audio.Sound.createAsync({ uri: tempPath });
      this.currentSound = sound;

      await new Promise<void>((resolve, reject) => {
        // Safety timeout para audio largo (max 60s)
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

      // Limpiar
      await sound.unloadAsync().catch(() => {});
      this.currentSound = null;

      // Borrar archivo temporal
      FileSystem.deleteAsync(tempPath, { idempotent: true }).catch(() => {});
    } catch (error) {
      clearTimeout(timeout);
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
