/**
 * Caché persistente de frases TTS (Piper)
 *
 * Cada texto se hashea → nombre de archivo WAV en disco.
 * Segunda vez que se pide la misma frase: se reproduce al instante.
 * Sobrevive entre sesiones (usa el directorio de caché del SO).
 */

import * as FileSystem from 'expo-file-system/legacy';
import { API_BASE_URL, API_ENDPOINTS } from '../constants/config';

const CACHE_DIR = `${FileSystem.cacheDirectory}navia_tts_v2/`;
const CHUNK = 4096;

function simpleHash(text: string): string {
  let h = 0;
  for (let i = 0; i < text.length; i++) {
    h = (Math.imul(31, h) + text.charCodeAt(i)) | 0;
  }
  return Math.abs(h).toString(36);
}

function extFromContentType(contentType: string | null): 'mp3' | 'wav' {
  if (!contentType) return 'wav';
  if (contentType.includes('mpeg') || contentType.includes('mp3')) return 'mp3';
  return 'wav';
}

class TtsPhraseCache {
  private mem = new Map<string, string>(); // text → fileUri
  private ready = false;

  async init(): Promise<void> {
    if (this.ready) return;
    try {
      const info = await FileSystem.getInfoAsync(CACHE_DIR);
      if (!info.exists) {
        await FileSystem.makeDirectoryAsync(CACHE_DIR, { intermediates: true });
      }
      this.ready = true;
    } catch { /* seguir sin caché persistente si falla */ }
  }

  private uriFor(text: string, ext: 'mp3' | 'wav' = 'wav'): string {
    return `${CACHE_DIR}${simpleHash(text)}.${ext}`;
  }

  /** Devuelve la URI del archivo si está en caché (disco o memoria). */
  async get(text: string): Promise<string | null> {
    // Memoria primero
    const mem = this.mem.get(text);
    if (mem) {
      try {
        const info = await FileSystem.getInfoAsync(mem);
        if (info.exists) return mem;
      } catch { /**/ }
      this.mem.delete(text);
    }

    if (!this.ready) return null;

    // Comprobar ambas extensiones (.mp3 para gTTS, .wav para Piper)
    for (const ext of ['mp3', 'wav'] as const) {
      const uri = this.uriFor(text, ext);
      try {
        const info = await FileSystem.getInfoAsync(uri);
        if (info.exists) {
          this.mem.set(text, uri);
          return uri;
        }
      } catch { /**/ }
    }

    return null;
  }

  /**
   * Descarga el audio desde el backend, lo guarda en disco y retorna la URI.
   * Opcionalmente acepta una AbortSignal para cancelar si el usuario para la app.
   */
  async fetchAndStore(text: string, signal?: AbortSignal): Promise<string> {
    const res = await fetch(`${API_BASE_URL}${API_ENDPOINTS.TTS}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal,
    });
    if (!res.ok) throw new Error(`TTS HTTP ${res.status}`);

    // Usar extensión correcta según lo que devuelve el backend (MP3 para gTTS, WAV para Piper)
    const ext = extFromContentType(res.headers.get('Content-Type'));
    const uri = this.uriFor(text, ext);

    const buf = await res.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i += CHUNK) {
      binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
    }

    if (this.ready) {
      await FileSystem.writeAsStringAsync(uri, btoa(binary), {
        encoding: FileSystem.EncodingType.Base64,
      });
      this.mem.set(text, uri);
      return uri;
    }

    // Sin caché en disco: guardar en temp con extensión correcta
    const tmp = `${FileSystem.cacheDirectory}navia_tts_${Date.now()}.${ext}`;
    await FileSystem.writeAsStringAsync(tmp, btoa(binary), {
      encoding: FileSystem.EncodingType.Base64,
    });
    return tmp;
  }

  /** Pre-genera frases comunes en segundo plano al arrancar la app. */
  prewarm(phrases: string[]): void {
    this.init().then(() => {
      for (const p of phrases) {
        this.get(p).then((cached) => {
          if (!cached) this.fetchAndStore(p).catch(() => {});
        });
      }
    });
  }
}

export const ttsCache = new TtsPhraseCache();
