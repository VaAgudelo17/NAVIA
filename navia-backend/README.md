---
title: NAVIA Backend
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# NAVIA — Backend de Asistencia Visual

Backend FastAPI del proyecto NAVIA: asistencia visual para personas con discapacidad visual mediante IA.

## Modos

- **Navegación** (WebSocket): YOLO + Depth-Anything-V2 + ByteTrack para detección de obstáculos en tiempo real.
- **Exploración** (HTTP): YOLO-World 417 clases + posición espacial + color dominante.
- **Lectura** (HTTP): Gemini 2.5 Flash con fallback a Tesseract.

## Variables de entorno

Configurar en **Settings → Secrets** del Space:

- `GEMINI_API_KEY` (obligatoria) — clave de Google AI Studio para el modo lectura.

Otras variables (con defaults razonables, no obligatorio cambiar):

- `TTS_BACKEND=piper`
- `TTS_MODEL_NAME=es_MX-claude-high`
- `DEBUG_MODE=False`

## Endpoints

- `GET /api/v1/health` — health check
- `POST /api/v1/analyze/exploracion` — descripción del entorno
- `POST /api/v1/analyze/lectura` — OCR inteligente
- `POST /api/v1/analyze/navegacion` — navegación de un solo frame
- `WS  /api/v1/ws/realtime` — navegación en tiempo real
- `POST /api/v1/tts` — síntesis de voz Piper

Documentación interactiva en `/docs`.
