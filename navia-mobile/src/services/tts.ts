/**
 * Servicio de Text-to-Speech para NAVIA
 * Usa expo-speech para convertir texto a voz
 */

import * as Speech from 'expo-speech';
import { TTS_CONFIG } from '../constants/config';

/**
 * Habla el texto proporcionado
 */
export async function speak(text: string): Promise<void> {
  if (!text || text.trim() === '') {
    return;
  }

  // Detener cualquier reproducción anterior
  await stop();

  return new Promise((resolve, reject) => {
    Speech.speak(text, {
      language: TTS_CONFIG.language,
      pitch: TTS_CONFIG.pitch,
      rate: TTS_CONFIG.rate,
      onDone: () => resolve(),
      onError: (error) => reject(error),
    });
  });
}

/**
 * Detiene la reproducción actual
 */
export async function stop(): Promise<void> {
  await Speech.stop();
}

/**
 * Verifica si está hablando actualmente
 */
export async function isSpeaking(): Promise<boolean> {
  return await Speech.isSpeakingAsync();
}

/**
 * Obtiene las voces disponibles
 */
export async function getVoices(): Promise<Speech.Voice[]> {
  return await Speech.getAvailableVoicesAsync();
}

/**
 * Habla un mensaje de bienvenida
 */
export async function speakWelcome(): Promise<void> {
  await speak('Bienvenido a NAVIA, tu asistente visual. Toca el botón central para capturar una imagen.');
}

/**
 * Habla un mensaje de error
 */
export async function speakError(message: string): Promise<void> {
  await speak(`Error: ${message}`);
}

/**
 * Habla un mensaje de procesamiento
 */
export async function speakProcessing(): Promise<void> {
  await speak('Procesando imagen, por favor espera.');
}
