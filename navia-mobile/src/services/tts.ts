/**
 * Servicio de Text-to-Speech para NAVIA
 *
 * Wrapper de compatibilidad sobre ttsManager.
 * Todas las llamadas delegan al singleton ttsManager
 * que maneja la cola de prioridades y evita cancelaciones.
 */

import { ttsManager, TtsPriority } from './ttsManager';

/** Habla el texto proporcionado (prioridad HIGH) */
export function speak(text: string): void {
  ttsManager.speak(text, TtsPriority.HIGH);
}

/** Detiene la reproducción actual y limpia la cola */
export function stop(): void {
  ttsManager.stop();
}

/** Verifica si está hablando actualmente */
export function isSpeaking(): boolean {
  return ttsManager.isSpeaking();
}

/** Habla un mensaje de bienvenida */
export function speakWelcome(): void {
  ttsManager.speak(
    'Bienvenido a NAVIA, tu asistente visual. Selecciona un modo y toca Iniciar Cámara.',
    TtsPriority.HIGH,
  );
}

/** Habla un mensaje de error (prioridad INTERRUPT - interrumpe todo) */
export function speakError(message: string): void {
  ttsManager.speak(`Error: ${message}`, TtsPriority.INTERRUPT);
}

/** Habla un mensaje de procesamiento */
export function speakProcessing(): void {
  ttsManager.speak('Procesando imagen, por favor espera.', TtsPriority.HIGH);
}
