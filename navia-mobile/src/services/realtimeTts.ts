/**
 * TTS inteligente para modo tiempo real
 *
 * Solo narra cambios significativos:
 * - Objetos que aparecen/desaparecen
 * - Alertas de zona: cuando un objeto entra en "muy cerca" (peligro)
 *
 * Mínimo 3 segundos entre frases para no saturar al usuario.
 */

import * as Speech from 'expo-speech';
import { TTS_CONFIG, REALTIME_CONFIG } from '../constants/config';

interface ZoneChange {
  name: string;
  from_zone: string;
  to_zone: string;
}

interface RealtimeChanges {
  appeared: string[];
  disappeared: string[];
  zone_changes: ZoneChange[];
  has_significant_change: boolean;
}

export class RealtimeTtsManager {
  private lastSpeakTime = 0;
  private isSpeaking = false;

  async speakChanges(changes: RealtimeChanges): Promise<void> {
    if (!changes.has_significant_change) return;

    const now = Date.now();
    if (now - this.lastSpeakTime < REALTIME_CONFIG.ttsMinInterval) return;
    if (this.isSpeaking) return;

    let text = '';

    // Prioridad 1: Alertas de zona "muy cerca" (peligro)
    const dangerAlerts = (changes.zone_changes || []).filter(
      (zc) => zc.to_zone === 'muy_cerca'
    );
    if (dangerAlerts.length > 0) {
      const names = dangerAlerts.map((z) => z.name).join(', ');
      text = `Precaución, ${names} muy cerca`;
    }
    // Prioridad 2: Objetos que aparecen
    else if (changes.appeared.length > 0 && changes.disappeared.length === 0) {
      const items = changes.appeared.join(', ');
      text = changes.appeared.length === 1
        ? `Nuevo: ${items}`
        : `Nuevos: ${items}`;
    }
    // Prioridad 3: Objetos que desaparecen
    else if (changes.disappeared.length > 0 && changes.appeared.length === 0) {
      text = `${changes.disappeared.join(', ')} ya no visible`;
    }
    // Prioridad 4: Ambos
    else if (changes.appeared.length > 0 && changes.disappeared.length > 0) {
      text = `Ahora: ${changes.appeared.join(', ')}`;
    }

    if (!text) return;

    this.isSpeaking = true;
    this.lastSpeakTime = now;

    try {
      await Speech.stop();
      await new Promise<void>((resolve) => {
        Speech.speak(text, {
          language: TTS_CONFIG.language,
          pitch: TTS_CONFIG.pitch,
          rate: 1.1,
          onDone: () => resolve(),
          onError: () => resolve(),
        });
      });
    } catch {
      // Ignorar errores TTS en modo tiempo real
    } finally {
      this.isSpeaking = false;
    }
  }

  stop(): void {
    Speech.stop();
    this.isSpeaking = false;
  }

  reset(): void {
    this.stop();
    this.lastSpeakTime = 0;
  }
}
