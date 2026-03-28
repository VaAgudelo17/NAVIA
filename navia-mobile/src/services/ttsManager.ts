/**
 * TTS Manager para NAVIA
 * Usa expo-speech (voz del celular)
 */

import * as Speech from 'expo-speech';
import { TTS_CONFIG } from '../constants/config';

export enum TtsPriority {
  INTERRUPT = 0,
  HIGH = 1,
  LOW = 2,
}

class TtsManager {
  private static instance: TtsManager;

  private queue: Array<{ text: string; priority: TtsPriority }> = [];
  private speaking = false;
  private enabled = true;

  private constructor() {}

  static getInstance(): TtsManager {
    if (!TtsManager.instance) {
      TtsManager.instance = new TtsManager();
    }
    return TtsManager.instance;
  }

  speak(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    console.log('[TTS] speak llamado:', text?.substring(0, 30), 'enabled:', this.enabled);
    if (!this.enabled || !text?.trim()) {
      console.log('[TTS] speak ignorado');
      return;
    }
    this.enqueue(text.trim(), priority);
  }

  speakFromBackend(text: string, priority: TtsPriority = TtsPriority.HIGH): void {
    console.log('[TTS] speakFromBackend llamado:', text?.substring(0, 30));
    if (!this.enabled || !text?.trim()) return;
    this.enqueue(text.trim(), priority);
  }

  stop(): void {
    this.queue = [];
    Speech.stop();
    this.speaking = false;
  }

  isSpeaking(): boolean {
    return this.speaking;
  }

  setEnabled(value: boolean): void {
    this.enabled = value;
    if (!value) this.stop();
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  private enqueue(text: string, priority: TtsPriority): void {
    console.log('[TTS] enqueue:', priority, 'speaking:', this.speaking);
    
    // INTERRUPT: limpia todo y habla ahora
    if (priority === TtsPriority.INTERRUPT) {
      this.queue = [];
      Speech.stop();
      setTimeout(() => this.speakNow(text), 100);
      return;
    }

    // HIGH: si está hablando, descartar
    if (priority === TtsPriority.HIGH) {
      if (this.speaking) {
        console.log('[TTS] HIGH descartado porque ya está hablando');
        return;
      }
    }

    // LOW: solo si no hay nada
    if (priority === TtsPriority.LOW) {
      if (this.speaking || this.queue.length > 0) return;
    }

    this.queue.push({ text, priority });
    this.processQueue();
  }

  private processQueue(): void {
    if (this.speaking || this.queue.length === 0) return;

    this.queue.sort((a, b) => a.priority - b.priority);
    const item = this.queue.shift()!;
    this.speakNow(item.text);
  }

  private speakNow(text: string): void {
    this.speaking = true;
    console.log('[TTS] speakNow llamado:', text.substring(0, 50));
    
    Speech.speak(text, {
      language: TTS_CONFIG.language,
      pitch: TTS_CONFIG.pitch,
      rate: TTS_CONFIG.rate,
      onDone: () => {
        this.speaking = false;
        if (this.queue.length > 0) {
          this.processQueue();
        }
      },
      onError: (err) => {
        console.warn('[TTS] error:', err);
        this.speaking = false;
        if (this.queue.length > 0) {
          this.processQueue();
        }
      },
      onStopped: () => {
        this.speaking = false;
      },
    });
  }
}

export const ttsManager = TtsManager.getInstance();
