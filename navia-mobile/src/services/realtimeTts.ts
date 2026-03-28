/**
 * TTS inteligente para navegación en tiempo real (Mobile)
 *
 * Delega al ttsManager singleton para evitar conflictos de audio.
 *
 * Pipeline unificado (navegación + riesgo):
 * - Usa el summary del backend (instrucciones priorizadas de navegación)
 * - Alertas critical interrumpen el speech actual (INTERRUPT)
 * - Alertas high tienen prioridad alta con cooldown reducido
 * - Instrucciones normales tienen cooldown de 3s
 */

import { ttsManager, TtsPriority } from './ttsManager';
import { REALTIME_CONFIG } from '../constants/config';
import { NaviaMode } from '../types/api';

interface RealtimeChanges {
  appeared: string[];
  disappeared: string[];
  zone_changes: Array<{ name: string; from_zone: string; to_zone: string }>;
  has_significant_change: boolean;
}

interface GuidanceData {
  has_danger: boolean;
  priority: string;
  path_clear: boolean;
}

// Tiempo mínimo sin peligro antes de decir "camino libre" (ms)
const SAFE_CONFIRMATION_DELAY = 3000;
// Tiempo mínimo entre dos INTERRUPTs seguidos (evita cortar audio que acaba de empezar)
const MIN_INTERRUPT_INTERVAL = 1500;
// Cooldown por obstáculo único: no repetir advertencia del mismo obstáculo/posición
const OBSTACLE_REPEAT_COOLDOWN = 6000;

export class RealtimeTtsManager {
  private lastSpeakTime = 0;
  private mode: NaviaMode = 'navegacion';
  private lastSummary = '';
  private lastGuidanceKey = '';
  private lastPathClear = true;
  private lastDangerTime = 0;      // última vez que hubo peligro real
  private lastInterruptTime = 0;   // última vez que se lanzó INTERRUPT
  private lastHighDangerSpeak = 0; // cooldown global para HIGH: evita que cambios de
                                   // posición (frente↔lateral) re-disparen el TTS
  private obstacleLastSpokenAt: Record<string, number> = {};  // guidance_key → timestamp

  setMode(mode: NaviaMode): void {
    this.mode = mode;
  }

  /**
   * Habla el resultado de navegación unificada.
   *
   * Prioridad de speech:
   * 1. Peligro critical → interrumpe inmediatamente
   * 2. Camino recién bloqueado → interrumpe inmediatamente
   * 3. Peligro high → prioridad alta, cooldown 800ms
   * 4. "Camino libre" → solo si han pasado 5s desde el último peligro
   * 5. Navegación normal → cooldown 2s
   */
  speakResult(
    summary: string,
    changes?: RealtimeChanges,
    guidanceData?: GuidanceData,
    guidanceKey?: string,
  ): void {
    if (!summary) return;

    const now = Date.now();

    // Registrar cuándo hubo peligro real
    if (guidanceData?.has_danger) {
      this.lastDangerTime = now;
    }

    const pathJustBlocked = this.lastPathClear && guidanceData?.path_clear === false;

    // Actualizar estado de camino.
    // IMPORTANTE: solo volver a "libre" si han pasado los 5s de seguridad.
    // Si el depth fluctúa y path_clear va false→true→false, sin este control
    // se dispararía INTERRUPT en cada transición, cortando el audio constantemente.
    if (guidanceData !== undefined) {
      if (guidanceData.path_clear === false) {
        this.lastPathClear = false;
      } else if ((now - this.lastDangerTime) >= SAFE_CONFIRMATION_DELAY) {
        this.lastPathClear = true;
      }
      // Dentro del periodo de peligro: mantener lastPathClear=false
      // para que pathJustBlocked no vuelva a dispararse por fluctuación
    }

    // "Camino libre" solo se dice si pasaron 5s desde el último peligro.
    // Evita que salte "camino libre" inmediatamente después de una advertencia
    // solo porque el depth map fluctuó un frame.
    const isCaminoLibre = guidanceData?.path_clear === true && !guidanceData?.has_danger;
    if (isCaminoLibre && (now - this.lastDangerTime) < SAFE_CONFIRMATION_DELAY) {
      return;
    }

    // Cooldown por guidance_key — aplica a TODAS las instrucciones (peligrosas o no).
    // Evita que fluctuaciones de zona (cerca↔muy_cerca) o cambios menores en la frase
    // combinada ("camino libre + lateral") re-disparen el TTS del mismo obstáculo.
    // No aplica a CRITICAL (siempre interrumpe).
    const isObstacleCooldownActive = (
      guidanceKey &&
      guidanceData?.priority !== 'critical' &&
      (now - (this.obstacleLastSpokenAt[guidanceKey] ?? 0)) < OBSTACLE_REPEAT_COOLDOWN
    );
    if (isObstacleCooldownActive) return;

    // Peligro critical: interrumpir speech actual (siempre, sin cooldown)
    if (guidanceData?.has_danger && guidanceData.priority === 'critical') {
      if ((now - this.lastInterruptTime) >= MIN_INTERRUPT_INTERVAL) {
        this.lastSummary = summary;
        this.lastSpeakTime = now;
        this.lastInterruptTime = now;
        if (guidanceKey) this.obstacleLastSpokenAt[guidanceKey] = now;
        ttsManager.speak(summary, TtsPriority.INTERRUPT);
      }
      return;
    }

    // Camino recién bloqueado: interrumpir una sola vez (con cooldown de 3s)
    if (pathJustBlocked && (now - this.lastInterruptTime) >= MIN_INTERRUPT_INTERVAL) {
      this.lastSummary = summary;
      this.lastSpeakTime = now;
      this.lastInterruptTime = now;
      if (guidanceKey) this.obstacleLastSpokenAt[guidanceKey] = now;
      ttsManager.speak(summary, TtsPriority.INTERRUPT);
      return;
    }

    // Peligro high: prioridad alta
    if (guidanceData?.has_danger && guidanceData.priority === 'high') {
      if (now - this.lastHighDangerSpeak < REALTIME_CONFIG.ttsMinInterval) return;
      if (now - this.lastSpeakTime < REALTIME_CONFIG.ttsMinInterval) return;
      if (ttsManager.isSpeaking()) return;
      if (summary === this.lastSummary) return;

      this.lastSummary = summary;
      this.lastSpeakTime = now;
      this.lastHighDangerSpeak = now;
      if (guidanceKey) this.obstacleLastSpokenAt[guidanceKey] = now;
      ttsManager.speak(summary, TtsPriority.HIGH);
      return;
    }

    // Navegación normal (medium, none, camino libre confirmado): cooldown estándar
    this.speakNavigationSummary(summary, changes, guidanceKey);
  }

  private speakNavigationSummary(
    summary: string,
    changes?: RealtimeChanges,
    guidanceKey?: string,
  ): void {
    // Suprimir si el mismo obstáculo/contexto ya fue hablado recientemente.
    // Usar guidance_key cuando esté disponible (más estable que comparar texto exacto,
    // que cambia por fluctuaciones de proximidad en frases combinadas).
    const sameContext = guidanceKey
      ? guidanceKey === this.lastGuidanceKey
      : summary === this.lastSummary;
    if (!changes?.has_significant_change && sameContext) return;
    if (!summary) return;

    const now = Date.now();
    if (now - this.lastSpeakTime < REALTIME_CONFIG.ttsMinInterval) return;
    if (ttsManager.isSpeaking()) return;

    this.lastSummary = summary;
    this.lastGuidanceKey = guidanceKey ?? '';
    this.lastSpeakTime = now;
    if (guidanceKey) this.obstacleLastSpokenAt[guidanceKey] = now;
    ttsManager.speak(summary, TtsPriority.LOW);
  }

  stop(): void {
    ttsManager.stop();
  }

  reset(): void {
    ttsManager.stop();
    this.lastSpeakTime = 0;
    this.lastSummary = '';
    this.lastGuidanceKey = '';
    this.lastPathClear = true;
    this.lastDangerTime = 0;
    this.lastInterruptTime = 0;
    this.lastHighDangerSpeak = 0;
    this.obstacleLastSpokenAt = {};
  }
}
