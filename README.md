# NAVIA - Navegacion Visual Asistida con Inteligencia Artificial

**Aplicacion de asistencia visual para personas con discapacidad visual**, desarrollada como proyecto de tesis en la **Universidad San Buenaventura Cali**.

NAVIA procesa imagenes en tiempo real y bajo demanda para detectar obstaculos, estimar distancias, leer textos y describir entornos, convirtiendo todo en instrucciones de audio claras y priorizadas en espanol.

---

## Tabla de Contenidos

- [Arquitectura](#arquitectura)
- [Modos de Uso](#modos-de-uso)
- [Stack Tecnologico](#stack-tecnologico)
- [Modelos de IA](#modelos-de-ia)
- [Requisitos Previos](#requisitos-previos)
- [Instalacion](#instalacion)
- [Ejecucion](#ejecucion)
- [API Endpoints](#api-endpoints)
- [Base de Datos](#base-de-datos)
- [Estructura del Proyecto](#estructura-del-proyecto)

---

## Arquitectura

NAVIA se compone de dos modulos que se comunican via HTTP y WebSocket:

```
┌─────────────────┐          ┌─────────────────┐
│  navia-mobile   │          │  navia-backend  │
│  (React Native  │          │  (FastAPI)      │
│   + Expo)       │          │  Puerto: 8000   │
│                 │          │                 │
│  Camara movil   │─────────>│  YOLO-World v2  │
│  TTS nativo     │<─────────│  Depth Anything │
│  AsyncStorage   │          │  Florence-2     │
│                 │          │  Piper TTS      │
└─────────────────┘          │  Tesseract OCR  │
                             │  Gemini 2.0     │
                             │  ByteTrack      │
                             │  SQLite         │
                             └─────────────────┘
```

**Flujo de datos en tiempo real (WebSocket):**

1. El cliente captura frames de la camara (~2 FPS)
2. Los envia como base64 JPEG al backend via WebSocket
3. El backend ejecuta: deteccion → profundidad → tracking → navegacion
4. Responde con objetos detectados, alertas de peligro e instrucciones de voz
5. El cliente reproduce las instrucciones por TTS con sistema de prioridades

---

## Modos de Uso

### 1. Navegacion

Modo principal para **movilidad peatonal segura**. Funciona en tiempo real via WebSocket.

**Pipeline completo:**
- Deteccion de objetos con YOLO-World v2 (~298 clases en espanol)
- Estimacion de profundidad con Depth Anything V2
- Filtrado a clases relevantes para caminata (vehiculos, obstaculos, personas, escaleras, bordillos, etc.)
- Clasificacion de altura del obstaculo (suelo / cuerpo / cabeza)
- Analisis de movimiento entre frames (acercandose / alejandose / estatico)
- Scoring de riesgo combinado: `peso_base x proximidad x altura x movimiento`
- Generacion de instrucciones priorizadas (maximo 3 por frame)

**Zonas de distancia:**
| Zona | Significado |
|------|-------------|
| `muy_cerca` | Peligro inminente, accion requerida |
| `cerca` | Precaucion, obstaculo cercano |
| `lejos` | Informativo, sin accion necesaria |

**Ejemplo de salida:** *"Cuidado: bicicleta se acerca por la derecha. Bordillo muy cerca frente a ti."*

---

### 2. Exploracion

Descripcion estructurada del entorno mediante captura de foto.

- Deteccion con umbral de confianza elevado (0.40)
- Filtrado semantico (ignora objetos irrelevantes: cielo, piso, paredes)
- Priorizacion inteligente (confianza x tamano x cercania al centro)
- Estimacion de distancia basada en tamano del bounding box
- Extraccion de color dominante del objeto principal
- Captioning opcional con Florence-2
- Maximo 3 objetos + 1 bloque de texto (evita sobrecarga cognitiva)

**Ejemplo de salida:** *"Un gato esta frente a ti, muy cerca, a nivel del suelo. Es de color naranja."*

---

### 3. Lectura

Lectura inteligente de documentos con clasificacion automatica.

**Pipeline dual de OCR:**
- **Primario:** Gemini 2.0 Flash (OCR + clasificacion + narrativa en una sola llamada)
- **Fallback:** Tesseract OCR local con clasificacion por reglas

**Tipos de documento soportados (27+):**
- Facturas, recibos, cartas, contratos, formularios
- Identificaciones, boletos, recetas medicas, resultados de laboratorio
- Interfaces digitales (menus, login, configuracion, redes sociales)
- Chats, notificaciones, correos
- Tablas nutricionales, horarios, instrucciones

**Funcionalidades:**
- Clasificacion automatica del tipo de documento
- Extraccion de campos estructurados (fechas, montos, emails, telefonos)
- Analisis de calidad de imagen (borrosa, oscura, baja resolucion)
- Optimizacion de prosodia para Piper TTS
- Soporte PDF (primera pagina convertida a imagen via PyMuPDF)

---

## Stack Tecnologico

### Backend (Python)

| Componente | Tecnologia |
|------------|-----------|
| Framework | FastAPI >= 0.115.0 |
| Servidor | Uvicorn >= 0.32.0 (ASGI) |
| ORM | SQLAlchemy >= 2.0.0 (async) |
| Base de datos | SQLite (aiosqlite) / PostgreSQL (asyncpg) |
| Validacion | Pydantic >= 2.9.0 |
| Vision | OpenCV >= 4.10.0, Pillow >= 11.0.0 |
| ML | ultralytics >= 8.3.0, transformers >= 4.41.0 |
| OCR | pytesseract >= 0.3.13, google-genai >= 1.0.0 |
| TTS | piper-tts >= 1.2.0 |
| PDF | pymupdf >= 1.23.0 |

### App Movil (TypeScript)

| Componente | Tecnologia |
|------------|-----------|
| Framework | Expo ~54.0.33 |
| UI | React Native 0.81.5 |
| Camara | expo-camera ~17.0.10 |
| Audio | expo-av ~16.0.8, expo-speech ~14.0.8 |
| Storage | @react-native-async-storage 2.2.0 |
| Hapticos | expo-haptics ~15.0.8 |

---

## Modelos de IA

| Modelo | Funcion | Tamano | Ejecucion |
|--------|---------|--------|-----------|
| **YOLO-World v2 Small** | Deteccion de objetos (vocabulario abierto, ~298 clases) | ~30 MB | CPU/GPU local |
| **Depth Anything V2 Small** | Estimacion monocular de profundidad | ~98 MB | CPU local |
| **Florence-2 Base** | Captioning de escenas | ~1 GB | CPU local |
| **Piper TTS (VITS)** | Sintesis de voz en espanol (modelo `es_ES-davefx-medium`) | ~100 MB | CPU local |
| **Tesseract OCR** | Reconocimiento optico de caracteres (espanol + ingles) | Sistema | CPU local |
| **Gemini 2.0 Flash** | OCR avanzado + clasificacion de documentos | Cloud | API de Google |
| **ByteTrack** | Tracking de objetos entre frames (IoU + Kalman filter) | Incluido en ultralytics | CPU local |

### Vocabulario de Deteccion (298 clases)

Las clases estan organizadas en categorias y traducidas al espanol con articulos correctos:

- **Personas:** persona, nino, bebe, usuario de silla de ruedas
- **Vehiculos y calle:** coche, autobus, bicicleta, motocicleta, semaforo, senal de alto, paso de peatones...
- **Hogar y navegacion interior:** puerta, escaleras, ventana, lampara, interruptor, ventilador...
- **Muebles:** silla, mesa, sofa, cama, estanteria...
- **Cocina:** plato, taza, olla, cubiertos, electrodomesticos...
- **Electronica:** television, telefono, laptop, audifonos...
- **Animales:** perro, gato, pajaros, animales de granja...
- **Y mas:** ropa, joyeria, juguetes, oficina, seguridad y salud, bano...

---

## Requisitos Previos

- **Python** 3.11+
- **Node.js** 18+ y npm
- **Tesseract OCR** instalado en el sistema
- **Expo CLI** (para la app movil)
- (Opcional) **GEMINI_API_KEY** para el modo lectura avanzado

### Instalacion de Tesseract (macOS)

```bash
brew install tesseract tesseract-lang
```

---

## Instalacion

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd NAVIA
```

### 2. Backend

```bash
cd navia-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configurar variables de entorno:

```bash
cp .env.example .env
# Editar .env con tus valores (GEMINI_API_KEY, TESSERACT_CMD, etc.)
```

### 3. App Movil

```bash
cd navia-mobile
npm install
```

Configurar la IP del backend en `src/constants/config.ts`:

```typescript
export const API_BASE_URL = 'http://<tu-ip-local>:8000';
```

---

## Ejecucion

### Backend + App Movil

```bash
./start-mobile.sh
```

### Manual

```bash
# Terminal 1: Backend
cd navia-backend && source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Mobile
cd navia-mobile
npx expo start
```

### URLs de acceso

| Servicio | URL |
|----------|-----|
| Backend API | http://localhost:8000 |
| Documentacion Swagger | http://localhost:8000/docs |
| Documentacion ReDoc | http://localhost:8000/redoc |
| Expo (mobile) | Escanear QR con Expo Go |

---

## API Endpoints

Todos los endpoints estan bajo el prefijo `/api/v1`.

### Salud

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/health` | Estado basico del servidor |
| GET | `/health/detailed` | Estado detallado (modelos cargados, servicios disponibles) |

### Analisis de Imagenes

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/analyze/navegacion` | Navegacion asistida con deteccion de riesgo |
| POST | `/analyze/exploracion` | Descripcion estructurada del entorno |
| POST | `/analyze/lectura` | Lectura inteligente de documentos |
| POST | `/analyze/ocr` | Extraccion de texto (OCR) |
| POST | `/analyze/objects` | Deteccion de objetos |
| POST | `/analyze/scene` | Analisis completo de escena |

### WebSocket (Tiempo Real)

| Protocolo | Endpoint | Descripcion |
|-----------|----------|-------------|
| WS | `/ws/realtime` | Deteccion en tiempo real via frames de camara |

### Text-to-Speech

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/tts` | Sintetizar texto a audio WAV (Piper TTS) |

### Historial

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/history` | Listar historial (paginado, filtrable por modo) |
| GET | `/history/{id}` | Detalle de un analisis |
| DELETE | `/history/{id}` | Eliminar un registro |
| DELETE | `/history` | Limpiar todo el historial |

### Preferencias

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/preferences` | Obtener preferencias del usuario |
| PUT | `/preferences` | Actualizar preferencias |

---

## Base de Datos

SQLite en desarrollo (archivo `navia.db`), compatible con PostgreSQL para produccion.

### Tabla `analysis_history`

| Columna | Tipo | Descripcion |
|---------|------|-------------|
| `id` | String(36), PK | UUID del registro |
| `mode` | String(20) | Modo: navegacion, exploracion, lectura |
| `reading_mode` | String(20) | Sub-modo de lectura (si aplica) |
| `result_summary` | Text | Resumen corto del resultado |
| `result_data` | JSON | Resultado completo serializado |
| `processing_time_ms` | Float | Tiempo de procesamiento |
| `object_count` | Integer | Objetos detectados |
| `has_text` | Boolean | Si se detecto texto |
| `has_danger` | Boolean | Si se detecto peligro |
| `created_at` | DateTime | Fecha del analisis |

### Tabla `user_preferences`

| Columna | Tipo | Descripcion |
|---------|------|-------------|
| `id` | Integer, PK | Auto-incremental |
| `key` | String(100), unique | Nombre de la preferencia |
| `value` | String(500) | Valor |
| `updated_at` | DateTime | Ultima actualizacion |

---

## Estructura del Proyecto

```
NAVIA/
├── navia-backend/
│   ├── main.py                          # Punto de entrada FastAPI
│   ├── requirements.txt                 # Dependencias Python
│   ├── .env                             # Variables de entorno
│   ├── navia.db                         # Base de datos SQLite
│   ├── yolov8s-worldv2.pt              # Modelo YOLO-World
│   ├── models/
│   │   └── es_ES-davefx-medium.onnx    # Modelo de voz Piper TTS
│   └── app/
│       ├── core/
│       │   └── config.py                # Configuracion central
│       ├── api/
│       │   ├── router.py                # Router principal
│       │   └── endpoints/
│       │       ├── health.py            # Health check
│       │       ├── image_analysis.py    # Analisis de imagenes
│       │       ├── realtime_ws.py       # WebSocket tiempo real
│       │       ├── tts.py               # Text-to-Speech
│       │       ├── history.py           # Historial
│       │       └── preferences.py       # Preferencias
│       ├── db/
│       │   ├── database.py              # Configuracion SQLAlchemy
│       │   └── models.py               # Modelos de BD
│       ├── models/
│       │   └── schemas.py               # Schemas Pydantic
│       ├── services/
│       │   ├── navigation_guidance_service.py  # Pipeline unificado navegacion + riesgo
│       │   ├── object_detection_service.py     # Deteccion YOLO-World
│       │   ├── depth_estimation_service.py     # Estimacion de profundidad
│       │   ├── exploration_service.py          # Modo exploracion
│       │   ├── smart_reading_service.py        # Modo lectura inteligente
│       │   ├── tracking_service.py             # ByteTrack tracking
│       │   ├── realtime_detection_service.py   # Estado de sesion WebSocket
│       │   ├── captioning_service.py           # Florence-2 captioning
│       │   ├── ocr_service.py                  # Tesseract OCR
│       │   ├── gemini_service.py               # Gemini Vision OCR
│       │   ├── tts_service.py                  # Piper TTS
│       │   ├── document_classifier.py          # Clasificador de documentos
│       │   ├── reading_optimizer.py            # Optimizador de lectura
│       │   ├── scene_description_service.py    # Descripcion de escenas
│       │   ├── semantic_priority_service.py    # Priorizacion semantica
│       │   └── history_service.py              # Guardado de historial
│       └── utils/
│           ├── image_utils.py           # Procesamiento de imagenes
│           └── pdf_utils.py             # Conversion de PDF
│
├── navia-mobile/
│   ├── package.json
│   ├── app.json                         # Configuracion Expo
│   ├── App.tsx                          # Componente raiz
│   └── src/
│       ├── screens/
│       │   └── HomeScreen.tsx           # Pantalla principal
│       ├── components/
│       │   ├── AnimatedEye.tsx          # Ojo animado
│       │   └── Button.tsx               # Boton personalizado
│       ├── services/
│       │   ├── api.ts                   # Cliente de API
│       │   ├── websocket.ts             # Cliente WebSocket
│       │   ├── ttsManager.ts            # Gestor de TTS
│       │   ├── realtimeTts.ts           # TTS tiempo real
│       │   └── storage.ts              # AsyncStorage (historial local)
│       ├── hooks/
│       │   └── useRealtimeDetection.ts  # Hook de deteccion
│       ├── context/
│       │   └── PreferencesContext.tsx    # Contexto de preferencias
│       ├── constants/
│       │   └── config.ts               # Configuracion (URLs, colores, modos)
│       └── types/
│           └── api.ts                   # Tipos TypeScript
│
├── start-mobile.sh                      # Iniciar backend + mobile
└── README.md
```

---

## Variables de Entorno

### Backend (`.env`)

| Variable | Requerida | Descripcion |
|----------|-----------|-------------|
| `API_HOST` | No | Host del servidor (default: `0.0.0.0`) |
| `API_PORT` | No | Puerto del servidor (default: `8000`) |
| `DEBUG_MODE` | No | Modo debug, precarga modelos (default: `True`) |
| `TESSERACT_CMD` | Si | Ruta al binario de Tesseract |
| `YOLO_MODEL` | No | Archivo del modelo YOLO (default: `yolov8m-worldv2.pt`) |
| `YOLO_CONFIDENCE_THRESHOLD` | No | Umbral de confianza (default: `0.35`) |
| `DATABASE_URL` | No | URL de la base de datos (default: SQLite local) |
| `GEMINI_API_KEY` | Opcional | API key de Google Gemini (para modo lectura avanzado) |
| `GEMINI_ENABLED` | No | Habilitar Gemini OCR (default: `True`) |

---

## Sistema de TTS (Text-to-Speech)

NAVIA usa un sistema de **cola con prioridades** para evitar que las voces se cancelen entre si:

| Prioridad | Uso | Comportamiento |
|-----------|-----|----------------|
| **INTERRUPT** | Alertas de peligro, errores criticos | Cancela todo y habla inmediatamente |
| **HIGH** | Resultados de analisis, feedback de botones | Se encola, nunca se descarta |
| **LOW** | Instrucciones automaticas de navegacion en tiempo real | Se descarta si hay algo sonando |

**Cadena de fallback:**
1. Piper TTS del backend (voz VITS natural en espanol)
2. Si falla → expo-speech (mobile)

---

## Proyecto de Tesis

**Universidad San Buenaventura Cali**

NAVIA es una aplicacion de asistencia visual que busca mejorar la autonomia y seguridad en la movilidad de personas con discapacidad visual, utilizando tecnicas de vision por computadora y procesamiento de lenguaje natural para proporcionar informacion del entorno en tiempo real a traves de audio.
