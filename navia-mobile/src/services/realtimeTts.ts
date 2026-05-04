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
  /** El backend marcó la escena como estable (mismo objeto en misma posición ≥5 frames). */
  scene_stable?: boolean;
}

// Tiempo mínimo sin peligro antes de decir "camino libre" (ms)
const SAFE_CONFIRMATION_DELAY = 3000;
// Tiempo mínimo entre dos INTERRUPTs seguidos (evita cortar audio que acaba de empezar)
const MIN_INTERRUPT_INTERVAL = 1500;
// Cooldown por obstáculo único: no repetir advertencia del mismo obstáculo.
// 8s da suficiente espacio para que se reproduzca el audio + procesamiento
// del usuario, sin que la misma alerta de "escritorio al frente" se repita.
const OBSTACLE_REPEAT_COOLDOWN = 8000;
// Cooldown extendido cuando la escena está estable (usuario parado): 15s
const STABLE_SCENE_COOLDOWN = 15000;
// Cada cuánto reasegurar al usuario "camino libre" si todo está OK (ms)
const REASSURANCE_INTERVAL = 12000;
// Mínimo absoluto entre cualquier dos TTS no-críticos. CRITICAL y HIGH
// danger lo saltan (peligro inmediato debe ser inmediato).
const MIN_GLOBAL_TTS_INTERVAL = 2500;

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
  private lastReassurance = 0;     // última vez que se reaseguró "camino libre"

  // Cola de alerta pendiente — modelo Waze. Si llega una alerta mientras el
  // ttsManager está hablando, NO se reproduce. Se guarda como pendiente, y
  // cuando el audio actual termina, se reproduce la última pendiente (si
  // sigue siendo relevante: <2s vieja, scene no cambió drásticamente).
  private pendingAlert: {
    summary: string;
    guidanceKey: string;
    priority: TtsPriority;
    queuedAt: number;
  } | null = null;
  private speakingListenerCleanup: (() => void) | null = null;

  constructor() {
    // Suscribirse al evento de fin de habla del ttsManager.
    // Cuando se vuelve false (terminó audio), se reproduce la pendiente.
    this.speakingListenerCleanup = ttsManager.onSpeakingChange((speaking) => {
      if (!speaking) this.flushPendingAlert();
    });
  }

  /** Reproduce la alerta pendiente si todavía es relevante. */
  private flushPendingAlert(): void {
    if (!this.pendingAlert) return;
    const alert = this.pendingAlert;
    this.pendingAlert = null;

    // Si la alerta es vieja (>2s), descartarla — la escena ya cambió
    const age = Date.now() - alert.queuedAt;
    if (age > 2000) return;

    // Reproducir la alerta diferida
    this.lastSummary = alert.summary;
    this.lastGuidanceKey = alert.guidanceKey;
    this.lastSpeakTime = Date.now();
    if (alert.guidanceKey) {
      this.obstacleLastSpokenAt[alert.guidanceKey] = Date.now();
    }
    ttsManager.speak(alert.summary, alert.priority);
  }

  /** Decide: hablar ya, encolar, o descartar. Modelo Waze. */
  private speakOrQueue(
    summary: string,
    guidanceKey: string | undefined,
    priority: TtsPriority,
  ): void {
    const now = Date.now();

    // Si hay audio sonando, encolar (reemplaza pendiente anterior)
    if (ttsManager.isSpeaking()) {
      this.pendingAlert = {
        summary,
        guidanceKey: guidanceKey ?? '',
        priority,
        queuedAt: now,
      };
      return;
    }

    // No hay audio sonando: hablar directamente
    this.lastSummary = summary;
    this.lastGuidanceKey = guidanceKey ?? '';
    this.lastSpeakTime = now;
    if (guidanceKey) this.obstacleLastSpokenAt[guidanceKey] = now;
    ttsManager.speak(summary, priority);
  }

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

    // Reassurance periódica: cada 12s sin nada relevante, decir "Camino libre"
    // para que el usuario sepa que la app sigue activa. Sin esto, si camina por
    // un pasillo abierto durante 30s sin obstáculos, no oye nada y duda si el
    // sistema funciona.
    if (
      isCaminoLibre &&
      !ttsManager.isSpeaking() &&
      (now - this.lastSpeakTime) >= REASSURANCE_INTERVAL &&
      (now - this.lastReassurance) >= REASSURANCE_INTERVAL
    ) {
      this.lastReassurance = now;
      this.lastSpeakTime = now;
      ttsManager.speak('Camino libre.', TtsPriority.LOW);
      return;
    }

    // Cooldown por guidance_key — aplica a TODAS las instrucciones (peligrosas o no).
    // Evita que fluctuaciones de zona (cerca↔muy_cerca) o cambios menores en la frase
    // combinada ("camino libre + lateral") re-disparen el TTS del mismo obstáculo.
    // No aplica a CRITICAL (siempre interrumpe).
    //
    // Si el backend marca scene_stable=true (usuario parado, mismo obstáculo
    // varios frames), extender el cooldown a 18s para no spamearle "cama al
    // frente" cada 6 segundos. Si el usuario se mueve o el objeto cambia de
    // zona, scene_stable vuelve a false y el cooldown estándar aplica.
    const cooldownMs = guidanceData?.scene_stable
      ? STABLE_SCENE_COOLDOWN
      : OBSTACLE_REPEAT_COOLDOWN;
    const isObstacleCooldownActive = (
      guidanceKey &&
      guidanceData?.priority !== 'critical' &&
      (now - (this.obstacleLastSpokenAt[guidanceKey] ?? 0)) < cooldownMs
    );
    if (isObstacleCooldownActive) return;

    // Cooldown GLOBAL: solo aplica a alertas no-peligrosas (informativas).
    // CRITICAL y HIGH danger lo saltan — un peligro real no espera 2.5s.
    const isCritical = guidanceData?.has_danger && guidanceData.priority === 'critical';
    const isHighDanger = guidanceData?.has_danger && guidanceData.priority === 'high';
    const isUrgent = isCritical || isHighDanger;
    if (!isUrgent && (now - this.lastSpeakTime) < MIN_GLOBAL_TTS_INTERVAL) {
      return;
    }

    // ESCENA ESTABLE (modelo Waze): el backend marca scene_stable=true cuando
    // el conjunto de obstáculos lleva varios frames idéntico (usuario parado
    // o caminando muy lento sin que cambie nada). En ese caso suprimimos
    // completamente el TTS — el usuario YA escuchó la advertencia y repetirla
    // solo confunde. La voz vuelve cuando aparezca un nuevo obstáculo o
    // alguno cambie de proximidad/zona, porque el fingerprint cambia y
    // scene_stable vuelve a false.
    if (!isCritical && guidanceData?.scene_stable) {
      return;
    }

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

    // Peligro HIGH: modelo Waze. Si hay audio sonando, NO interrumpe — se
    // queda como alerta pendiente. Cuando el audio actual termina, la
    // pendiente se reproduce (si sigue siendo relevante, <2s).
    // Esto evita "escritorio escritorio escritorio" cortándose entre frames.
    if (guidanceData?.has_danger && guidanceData.priority === 'high') {
      if (now - this.lastHighDangerSpeak < REALTIME_CONFIG.ttsMinInterval) return;
      if (summary === this.lastSummary) return;

      this.lastHighDangerSpeak = now;
      this.speakOrQueue(summary, guidanceKey, TtsPriority.HIGH);
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

    // Modelo Waze: si está hablando, encolar; si no, hablar
    this.speakOrQueue(summary, guidanceKey, TtsPriority.LOW);
  }

  /** Limpieza al desmontar (importante para no fugar listeners). */
  cleanup(): void {
    if (this.speakingListenerCleanup) {
      this.speakingListenerCleanup();
      this.speakingListenerCleanup = null;
    }
    this.pendingAlert = null;
  }

  stop(): void {
    ttsManager.stop();
  }

  /**
   * Reinicia SOLO el estado interno (cooldowns, fingerprints…). NO detiene
   * el ttsManager global: el caller decide si quiere callar la voz.
   * Esto evita matar audio que el caller acaba de iniciar (ej. "Detenido.").
   */
  reset(): void {
    this.lastSpeakTime = 0;
    this.lastSummary = '';
    this.lastGuidanceKey = '';
    this.lastPathClear = true;
    this.lastDangerTime = 0;
    this.lastInterruptTime = 0;
    this.lastHighDangerSpeak = 0;
    this.lastReassurance = 0;
    this.obstacleLastSpokenAt = {};
    this.pendingAlert = null;
  }
}
