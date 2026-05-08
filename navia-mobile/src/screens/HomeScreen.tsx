/**
 * Pantalla principal de NAVIA
 *
 * 3 modos:
 * - Navegación: tiempo real, instrucciones de navegación + detección de riesgo
 * - Exploración: foto, descripción estructurada del entorno
 * - Lectura: foto, OCR + lectura inteligente
 *
 * TTS: Usa ttsManager con cola de prioridades para evitar
 * que las voces se cancelen entre sí. Cada interacción tiene
 * feedback de audio para usuarios ciegos.
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  View,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  Alert,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { Text } from '../components/AppText';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';

import { Button } from '../components/Button';
import { AnimatedEye } from '../components/AnimatedEye';
import { VoiceWave } from '../components/VoiceWave';
import { ANALYSIS_MODES, REALTIME_MODES, AnalysisMode } from '../constants/config';
import { ThemeColors } from '../constants/themes';
import {
  analyzeNavigation,
  analyzeExploration,
  analyzeReading,
  checkHealth,
} from '../services/api';
import { ttsManager, TtsPriority } from '../services/ttsManager';
import {
  NavigationResponse,
  ExplorationResponse,
  SmartReadingResponse,
  ReadingMode,
} from '../types/api';
import { useRealtimeDetection, RealtimeSessionSummary } from '../hooks/useRealtimeDetection';
import { usePreferences } from '../context/PreferencesContext';
import { saveToHistory, getHistory, clearHistory, HistoryEntry } from '../services/storage';
import { SettingsScreen } from './SettingsScreen';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

type AppState = 'home' | 'camera' | 'processing' | 'results' | 'error' | 'realtime' | 'history';

// ============================================================================
// NARRATIVAS DEL DETALLE DEL HISTORIAL
// ----------------------------------------------------------------------------
// El detalle se divide en secciones que el usuario puede tocar para escuchar
// cada parte por separado. Esto evita que la voz junte todo y confunda.
// Cada función devuelve una frase corta enfocada en SU sección.
// ============================================================================

const HISTORY_MODE_LABELS: Record<string, string> = {
  navegacion: 'Navegación',
  exploracion: 'Exploración',
  lectura: 'Lectura',
};

/** Resumen corto (1-2 frases) leído al abrir y al tocar la card "Resumen". */
function buildShortSummary(item: HistoryEntry): string {
  const data = (item.resultData || {}) as any;
  if (item.mode === 'navegacion') {
    const obstacleCounts = (data.topObstacles ?? data.obstacle_counts) as Record<string, number> | undefined;
    const dangers = data.dangerCount ?? data.danger_count;
    const unique = obstacleCounts ? Object.keys(obstacleCounts).length : 0;
    if (unique > 0 && dangers !== undefined) {
      return `Sesión de navegación con ${unique} tipos de obstáculos detectados y ${dangers} alertas de peligro.`;
    }
    return 'Sesión de navegación.';
  }
  if (item.mode === 'exploracion') {
    const objCount = data.object_count;
    const objects = data.objects as any[] | undefined;
    if (objCount !== undefined && objects && objects.length) {
      const avg = objects.reduce((s, o) => s + (o.confidence || 0), 0) / objects.length;
      return `Análisis del entorno con ${objCount} objetos detectados a ${(avg * 100).toFixed(0)} por ciento de confianza promedio.`;
    }
    return 'Análisis del entorno.';
  }
  if (item.mode === 'lectura') {
    const docType = data.document_type;
    const confidence = data.confidence;
    const c = typeof confidence === 'number' ? confidence : Number(confidence);
    const pct = c <= 1 ? c * 100 : c;
    if (docType && !isNaN(pct)) {
      return `Lectura de un documento tipo ${docType} con ${pct.toFixed(0)} por ciento de exactitud.`;
    }
    if (docType) return `Lectura de un documento tipo ${docType}.`;
    return 'Lectura de un documento.';
  }
  return item.resultSummary || 'Sin resumen.';
}

/** Narrativa de la card SESIÓN (fecha, hora, duración). */
function buildSessionSpeech(item: HistoryEntry): string {
  const date = new Date(item.createdAt);
  const dateLong = date.toLocaleDateString('es-ES', {
    day: 'numeric', month: 'long', year: 'numeric',
  });
  const hour = date.getHours();
  const minute = date.getMinutes();
  const horaTxt = `${hour} y ${minute < 10 ? '0' + minute : minute} minutos`;

  const parts: string[] = [];
  parts.push(`Sesión del ${dateLong} a las ${horaTxt}.`);

  if (item.mode === 'navegacion') {
    const data = (item.resultData || {}) as any;
    const durationSec = data.durationSeconds ?? data.session_duration_seconds;
    if (durationSec !== undefined) {
      const min = Number(durationSec) / 60;
      if (min < 1) {
        parts.push(`Duración: ${Math.round(Number(durationSec))} segundos.`);
      } else {
        parts.push(`Duración: ${min.toFixed(1)} minutos.`);
      }
    }
  }
  return parts.join(' ');
}

/** Narrativa de la card ESTADÍSTICAS (datos específicos del modo). */
function buildStatsSpeech(item: HistoryEntry): string {
  const data = (item.resultData || {}) as any;
  const parts: string[] = [];

  if (item.mode === 'navegacion') {
    const frames = data.framesProcessed ?? data.frames_processed;
    const dangers = data.dangerCount ?? data.danger_count;
    const obstacleCounts = (data.topObstacles ?? data.obstacle_counts) as Record<string, number> | undefined;
    if (frames !== undefined) parts.push(`${frames} frames procesados.`);
    if (obstacleCounts) {
      const total = Object.values(obstacleCounts).reduce((s, n) => s + n, 0);
      parts.push(`${total} detecciones totales de ${Object.keys(obstacleCounts).length} objetos únicos.`);
    }
    if (dangers !== undefined) parts.push(`${dangers} alertas de peligro.`);
  } else if (item.mode === 'exploracion') {
    if (data.object_count !== undefined) parts.push(`${data.object_count} objetos detectados.`);
    parts.push(data.has_text ? 'Se detectó texto en la imagen.' : 'No se detectó texto.');
  } else if (item.mode === 'lectura') {
    if (data.document_type) parts.push(`Tipo de documento: ${data.document_type}.`);
    if (data.word_count !== undefined) parts.push(`${data.word_count} palabras detectadas.`);
    const source = data.ocr_source || data.source;
    if (source) parts.push(`Motor usado: ${source}.`);
  }
  return parts.join(' ') || 'Sin estadísticas.';
}

/** Narrativa de la card OBSTÁCULOS MÁS FRECUENTES (solo navegación). */
function buildObstaclesSpeech(item: HistoryEntry): string {
  const data = (item.resultData || {}) as any;
  const obstacleCounts = (data.topObstacles ?? data.obstacle_counts) as Record<string, number> | undefined;
  if (!obstacleCounts) return 'Sin obstáculos registrados.';
  const top = Object.entries(obstacleCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([n, c]) => `${n} ${c} veces`)
    .join(', ');
  return `Obstáculos más frecuentes: ${top}.`;
}

/** Narrativa de la card OBJETOS DETECTADOS (solo exploración). */
function buildObjectsSpeech(item: HistoryEntry): string {
  const data = (item.resultData || {}) as any;
  const objects = data.objects as any[] | undefined;
  if (!objects || objects.length === 0) return 'No se detectaron objetos.';
  const top = objects
    .slice(0, 5)
    .map((o) => {
      const conf = typeof o.confidence === 'number' ? o.confidence : 0;
      return `${o.name_es || o.name}, ${(conf * 100).toFixed(0)} por ciento`;
    })
    .join('; ');
  return `Objetos detectados: ${top}.`;
}

/** Narrativa breve para la card EXACTITUD. */
function buildAccuracySpeech(pct: number, mode: string): string {
  const what =
    mode === 'lectura' ? 'OCR del documento'
    : mode === 'exploracion' ? 'confianza promedio'
    : 'detección promedio';
  return `Exactitud: ${pct.toFixed(0)} por ciento. ${what}.`;
}

/** Narrativa CORTA que se reproduce al ABRIR el detalle.
 *  Solo da un saludo + resumen breve + invita a tocar cards.
 *  Las cards se tocan para escuchar cada sección por separado.
 *  Esto evita esperas largas en backends lentos como HF Spaces. */
function buildHistoryDetailNarrative(item: HistoryEntry): string {
  const mode = HISTORY_MODE_LABELS[item.mode] || item.mode;
  const summary = buildShortSummary(item);
  return `Abriendo detalle de ${mode}. ${summary} Toca cada sección para escuchar más.`;
}

// Configuración de cada modo
const MODE_CONFIG: Record<AnalysisMode, { label: string; icon: string; description: string }> = {
  navegacion: { label: 'Navegación', icon: 'compass', description: 'Navegación asistida con detección de riesgo' },
  exploracion: { label: 'Exploración', icon: 'search', description: 'Describe el entorno' },
  lectura: { label: 'Lectura', icon: 'document-text', description: 'Lee textos' },
};

export function HomeScreen() {
  // Preferencias persistentes (desde Context + AsyncStorage)
  const {
    analysisMode,
    readingMode,
    ttsEnabled,
    theme,
    fontScale,
    setAnalysisMode,
    setReadingMode,
    setTtsEnabled,
  } = usePreferences();

  const [settingsVisible, setSettingsVisible] = useState(false);
  const C = theme;
  const fs = (size: number) => Math.round(size * fontScale);
  // Regenera el StyleSheet cuando cambia el tema. Sin esto, los estilos
  // quedan fijos al tema "normal" porque StyleSheet.create() es estático.
  const styles = useMemo(() => createStyles(C), [C]);

  // Estados principales
  const [appState, setAppState] = useState<AppState>('home');
  const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Estados de imagen y resultados
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [navResult, setNavResult] = useState<NavigationResponse | null>(null);
  const [explorationResult, setExplorationResult] = useState<ExplorationResponse | null>(null);
  const [smartResult, setSmartResult] = useState<SmartReadingResponse | null>(null);

  // Estados de TTS
  const [isSpeakingState, setIsSpeakingState] = useState(false);

  // Historial local
  const [historyItems, setHistoryItems] = useState<HistoryEntry[]>([]);
  // Item del historial seleccionado para ver detalle (null = lista)
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<HistoryEntry | null>(null);
  // Detección de doble tap por item: guarda último id + timestamp
  const lastTapRef = useRef<{ id: string; time: number }>({ id: '', time: 0 });

  // Cámara
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  // Detección en tiempo real (Navegación con riesgo integrado)
  const [realtimeActive, setRealtimeActive] = useState(false);
  const handleRealtimeSessionEnd = useCallback((session: RealtimeSessionSummary) => {
    if (session.framesProcessed > 0) {
      saveToHistory({
        mode: 'navegacion',
        resultSummary: session.summary,
        resultData: {
          type: 'realtime_session',
          durationSeconds: session.durationSeconds,
          framesProcessed: session.framesProcessed,
          dangerCount: session.dangerCount,
          topObstacles: session.topObstacles,
        },
      });
    }
  }, []);
  const { wsStatus, latestResult } = useRealtimeDetection({
    cameraRef,
    enabled: realtimeActive,
    ttsEnabled,
    mode: analysisMode,
    onSessionEnd: handleRealtimeSessionEnd,
  });

  const isRealtimeMode = REALTIME_MODES.includes(analysisMode);

  // Ref para evitar que el useEffect deshabilite el TTS
  // mientras está anunciando "Voz desactivada / activada"
  const skipTtsSync = useRef(false);

  // Sincronizar ttsEnabled con ttsManager
  useEffect(() => {
    if (skipTtsSync.current) return;
    ttsManager.setEnabled(ttsEnabled);
  }, [ttsEnabled]);

  // Verificar conexión con el backend + bienvenida al iniciar
  useEffect(() => {
    checkBackendConnection();
    // Delay para que Audio.setAudioModeAsync termine antes de reproducir
    const t = setTimeout(() => {
      ttsManager.speak(
        'Bienvenido a NAVIA, tu asistente visual. Selecciona un modo y toca Iniciar Cámara.',
        TtsPriority.INTERRUPT,
      );
    }, 600);
    return () => clearTimeout(t);
  }, []);

  // Al abrir el detalle de un item del historial, leer el resumen completo.
  // El ref evita que Strict Mode (que ejecuta useEffect dos veces en dev)
  // llame speak() dos veces — eso causaba flicker en el estado speaking
  // y la animación de la onda no funcionaba bien.
  const lastSpokenItemRef = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedHistoryItem) {
      lastSpokenItemRef.current = null;
      return;
    }
    if (lastSpokenItemRef.current === selectedHistoryItem.id) return;
    lastSpokenItemRef.current = selectedHistoryItem.id;
    const narrative = buildHistoryDetailNarrative(selectedHistoryItem);
    if (narrative) ttsManager.speak(narrative, TtsPriority.INTERRUPT);
  }, [selectedHistoryItem]);

  const checkBackendConnection = async () => {
    const maxRetries = 5;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        await checkHealth();
        setIsBackendConnected(true);
        console.log(`[NAVIA] Backend conectado (intento ${attempt})`);
        return;
      } catch {
        console.warn(`[NAVIA] Health check intento ${attempt}/${maxRetries} falló`);
        if (attempt < maxRetries) {
          await new Promise(r => setTimeout(r, attempt * 1000));
        }
      }
    }
    setIsBackendConnected(false);
    ttsManager.speak('Sin conexión al servidor. Verifica tu conexión.', TtsPriority.HIGH);
  };

  // Abrir cámara
  const handleOpenCamera = async () => {
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        ttsManager.speak(
          'Permiso de cámara denegado. NAVIA necesita la cámara para funcionar.',
          TtsPriority.INTERRUPT,
        );
        Alert.alert('Permiso denegado', 'Se necesita acceso a la cámara.');
        return;
      }
    }

    if (isRealtimeMode) {
      setAppState('realtime');
      setRealtimeActive(true);
      const modeLabel = MODE_CONFIG[analysisMode].label;
      ttsManager.stop();
      ttsManager.speak(`Modo ${modeLabel} activado. Apunta la cámara.`, TtsPriority.INTERRUPT);
    } else {
      setAppState('camera');
      ttsManager.stop();
      ttsManager.speak('Cámara lista. Toca la pantalla para capturar.', TtsPriority.INTERRUPT);
    }
  };

  // Capturar foto
  const handleCapture = async () => {
    if (!cameraRef.current) return;

    try {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
      });

      if (photo?.uri) {
        setCapturedImage(photo.uri);
        setAppState('processing');
        await processImage(photo.uri, 'Foto capturada. Procesando imagen, por favor espera.');
      }
    } catch (err) {
      console.error('Error capturing photo:', err);
      handleError('No se pudo capturar la foto');
    }
  };

  // Seleccionar imagen de la galería
  const handlePickImage = async () => {
    ttsManager.speak('Abriendo galería.', TtsPriority.HIGH);
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setCapturedImage(result.assets[0].uri);
      setAppState('processing');
      await processImage(result.assets[0].uri);
    } else {
      ttsManager.speak('Cerrando galería.', TtsPriority.HIGH);
    }
  };

  // Procesar imagen según el modo
  const processImage = async (imageUri: string, processingMessage = 'Procesando imagen, por favor espera.') => {
    // Detener cualquier audio anterior antes de empezar un nuevo análisis
    ttsManager.stop();
    ttsManager.speak(processingMessage, TtsPriority.HIGH);

    try {
      switch (analysisMode) {
        case 'navegacion': {
          const result = await analyzeNavigation(imageUri);
          setNavResult(result);
          if (result.priority && result.priority !== 'none') {
            triggerNavigationHaptic(result.priority);
          }
          // Detener "Procesando..." antes de hablar el resultado
          ttsManager.stop();
          ttsManager.speakFromBackend(result.instruction, TtsPriority.HIGH);
          saveToHistory({
            mode: 'navegacion',
            resultSummary: result.instruction,
            resultData: result as unknown as Record<string, unknown>,
          });
          break;
        }
        case 'exploracion': {
          const result = await analyzeExploration(imageUri);
          setExplorationResult(result);
          ttsManager.stop();
          ttsManager.speakFromBackend(result.description, TtsPriority.HIGH);
          saveToHistory({
            mode: 'exploracion',
            resultSummary: result.description?.substring(0, 200) || '',
            resultData: result as unknown as Record<string, unknown>,
          });
          break;
        }
        case 'lectura': {
          const result = await analyzeReading(imageUri);
          setSmartResult(result);
          ttsManager.stop();

          // Feedback de calidad
          if (result.image_quality?.feedback_text && !result.image_quality.is_acceptable) {
            ttsManager.speak(result.image_quality.feedback_text, TtsPriority.INTERRUPT);
          }

          // Narrative o mensaje por defecto
          const message = result.narrative || 'No se detectó texto claro en la imagen.';

          // Si el backend incluyó el audio pre-generado, cachearlo antes de speak
          // para que se reproduzca al instante sin segundo round-trip
          if (result.audio_base64 && result.audio_format) {
            await ttsManager.preCacheReadingAudio(
              message,
              result.audio_base64,
              result.audio_format as 'mp3' | 'wav',
            );
          }

          ttsManager.speakReading(message, TtsPriority.HIGH);
          
          saveToHistory({
            mode: 'lectura',
            resultSummary: result.narrative?.substring(0, 200) || '',
            resultData: result as unknown as Record<string, unknown>,
          });
          break;
        }
        // "riesgo" fue unificado con "navegacion" en el pipeline
      }
      setAppState('results');
    } catch (err: any) {
      handleError(err.message || 'Error procesando la imagen');
    }
  };

  // Vibración según nivel de peligro en navegación
  const triggerNavigationHaptic = (priority: string) => {
    switch (priority) {
      case 'critical':
        // Doble golpe fuerte: peligro inmediato
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
        setTimeout(() => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error), 300);
        break;
      case 'high':
        // Un golpe fuerte: obstáculo importante
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
        break;
      case 'medium':
        // Golpe suave: obstáculo menor
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        break;
      // 'none': sin vibración
    }
  };

  const handleError = (message: string) => {
    setError(message);
    setAppState('error');
    ttsManager.speak(`Error: ${message}`, TtsPriority.INTERRUPT);
  };

  const handleReset = () => {
    ttsManager.stop();
    setRealtimeActive(false);
    setAppState('home');
    setCapturedImage(null);
    setNavResult(null);
    setExplorationResult(null);
    setSmartResult(null);
    setError(null);
    setSelectedHistoryItem(null);
  };

  // Repetir resultado por TTS
  const handleRepeat = async () => {
    if (isSpeakingState) {
      ttsManager.stop();
      setIsSpeakingState(false);
      return;
    }

    setIsSpeakingState(true);
    try {
      if (navResult) ttsManager.speakFromBackend(navResult.instruction, TtsPriority.HIGH);
      else if (explorationResult) ttsManager.speakFromBackend(explorationResult.description, TtsPriority.HIGH);
      else if (smartResult) ttsManager.speakReading(smartResult.narrative, TtsPriority.HIGH);
    } finally {
      setIsSpeakingState(false);
    }
  };

  // ============================================================================
  // RENDERIZADO
  // ============================================================================

  const renderContent = () => {
    // Si hay item de historial seleccionado, mostrar detalle (sobre cualquier estado).
    if (selectedHistoryItem) return renderHistoryDetail(selectedHistoryItem);
    switch (appState) {
      case 'camera': return renderCamera();
      case 'realtime': return renderRealtime();
      case 'processing': return renderProcessing();
      case 'results': return renderResults();
      case 'error': return renderError();
      case 'history': return renderHistory();
      default: return renderHome();
    }
  };

  // Pantalla de inicio
  const renderHome = () => (
    <View style={[styles.homeContainer, { backgroundColor: C.background }]}>
      {/* Botón de ajustes — arriba a la derecha */}
      <TouchableOpacity
        style={[styles.settingsBtn, { backgroundColor: C.surface }]}
        onPress={() => {
          ttsManager.speak('Abriendo configuración visual.', TtsPriority.HIGH);
          setSettingsVisible(true);
        }}
        accessibilityLabel="Configuración visual: temas y tamaño de letra"
        accessibilityRole="button"
      >
        <Ionicons name="settings-outline" size={22} color={C.primary} />
      </TouchableOpacity>

      <View style={styles.header}>
        <AnimatedEye size={64} color={C.primary} />
        <Text style={[styles.title, { color: C.text, fontSize: fs(42) }]}>NAVIA</Text>
        <Text style={[styles.subtitle, { color: C.textSecondary, fontSize: fs(18) }]}>Asistente Visual con IA</Text>
      </View>

      {/* Indicador de conexión */}
      <View style={styles.connectionStatus}>
        <View
          style={[
            styles.connectionDot,
            {
              backgroundColor:
                isBackendConnected === null ? C.warning
                  : isBackendConnected ? C.success
                  : C.error,
            },
          ]}
        />
        <Text style={[styles.connectionText, { color: C.textSecondary, fontSize: fs(14) }]}>
          {isBackendConnected === null ? 'Conectando...'
            : isBackendConnected ? 'Servidor conectado'
            : 'Sin conexión al servidor'}
        </Text>
      </View>

      {/* Onda de voz: visible mientras el TTS habla */}
      <VoiceWave color={C.primary} />

      {/* Selector de modo - grid 2x2 */}
      <View style={styles.modeSelector}>
        <Text style={[styles.modeLabel, { color: C.textSecondary, fontSize: fs(14) }]}>Modo de análisis:</Text>
        <View style={styles.modeButtons}>
          {Object.entries(ANALYSIS_MODES).map(([key, value]) => {
            const config = MODE_CONFIG[value];
            const isActive = analysisMode === value;
            return (
              <TouchableOpacity
                key={key}
                style={[
                  styles.modeButton,
                  {
                    borderColor: C.primary,
                    backgroundColor: isActive ? C.primary : 'transparent',
                  },
                ]}
                onPress={() => {
                  setAnalysisMode(value);
                  ttsManager.stop();
                  ttsManager.speak(`Modo ${config.label}. ${config.description}.`, TtsPriority.INTERRUPT);
                }}
                accessibilityRole="radio"
                accessibilityState={{ selected: isActive }}
                accessibilityLabel={`Modo ${config.label}. ${config.description}`}
              >
                <Ionicons
                  name={config.icon as any}
                  size={20}
                  color={isActive ? C.background : C.primary}
                />
                <Text
                  style={[
                    styles.modeButtonText,
                    { color: isActive ? C.background : C.primary, fontSize: fs(14) },
                  ]}
                >
                  {config.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {/* Botones principales — estilo pill */}
      <View style={styles.mainButtons}>
        <Button
          title={isRealtimeMode ? 'Iniciar Cámara' : 'Abrir Cámara'}
          onPress={handleOpenCamera}
          size="xl"
          disabled={isBackendConnected === false}
          icon={<Ionicons name="camera" size={28} color={C.background} />}
          style={styles.mainButton}
        />

        {!isRealtimeMode && (
          <Button
            title="Subir Imagen"
            onPress={handlePickImage}
            variant="outline"
            size="large"
            disabled={isBackendConnected === false}
            icon={<Ionicons name="image" size={24} color={C.primary} />}
            style={styles.secondaryButton}
          />
        )}
      </View>

      {/* Toggle TTS */}
      <TouchableOpacity
        style={styles.ttsToggle}
        onPress={() => {
          ttsManager.stop();
          ttsManager.setEnabled(true);
          if (ttsEnabled) {
            skipTtsSync.current = true;
            ttsManager.speak('Voz desactivada.', TtsPriority.INTERRUPT);
            setTimeout(() => {
              setTtsEnabled(false);
              skipTtsSync.current = false;
            }, 4000);
          } else {
            setTtsEnabled(true);
            ttsManager.speak('Voz activada.', TtsPriority.INTERRUPT);
          }
        }}
        accessibilityLabel={ttsEnabled ? 'Desactivar voz' : 'Activar voz'}
        accessibilityRole="switch"
        accessibilityState={{ checked: ttsEnabled }}
      >
        <Ionicons
          name={ttsEnabled ? 'volume-high' : 'volume-mute'}
          size={24}
          color={ttsEnabled ? C.primary : C.textSecondary}
        />
        <Text style={[styles.ttsToggleText, { color: ttsEnabled ? C.primary : C.textSecondary, fontSize: fs(14) }]}>
          {ttsEnabled ? 'Voz activada' : 'Voz desactivada'}
        </Text>
      </TouchableOpacity>

      {/* Botón Historial */}
      <TouchableOpacity
        style={styles.historyButton}
        onPress={handleOpenHistory}
        accessibilityLabel="Ver historial de análisis"
        accessibilityRole="button"
      >
        <Ionicons name="time" size={20} color={C.primary} />
        <Text style={[styles.historyButtonText, { color: C.primary, fontSize: fs(14) }]}>Ver Historial</Text>
      </TouchableOpacity>
    </View>
  );

  // Abrir historial
  const handleOpenHistory = async () => {
    ttsManager.speak('Abriendo historial.', TtsPriority.HIGH);
    const items = await getHistory(20);
    setHistoryItems(items);
    setAppState('history');
  };

  // Limpiar historial
  const handleClearHistory = async () => {
    await clearHistory();
    setHistoryItems([]);
    ttsManager.speak('Historial borrado.', TtsPriority.HIGH);
  };

  // Pantalla de Historial
  // Vista de detalle de un item del historial (doble tap)
  const renderHistoryDetail = (item: HistoryEntry) => {
    const modeLabels: Record<string, string> = {
      navegacion: 'Navegación',
      exploracion: 'Exploración',
      lectura: 'Lectura',
    };
    const modeIcons: Record<string, string> = {
      navegacion: 'compass',
      exploracion: 'search',
      lectura: 'document-text',
    };

    const date = new Date(item.createdAt);
    const dateStr = date.toLocaleDateString('es-ES', {
      day: 'numeric', month: 'long', year: 'numeric',
    });
    const timeStr = date.toLocaleTimeString('es-ES', {
      hour: '2-digit', minute: '2-digit',
    });

    const data = (item.resultData || {}) as any;
    const mode = item.mode;
    const objects = data.objects as any[] | undefined;

    // ── Calcular exactitud (porcentaje destacado) ─────────────────────────
    let accuracyPct: number | null = null;
    if (mode === 'lectura' && data.confidence !== undefined) {
      const c = typeof data.confidence === 'number' ? data.confidence : Number(data.confidence);
      accuracyPct = c <= 1 ? c * 100 : c;
    } else if (mode === 'exploracion' && objects && objects.length > 0) {
      const avg = objects.reduce((s, o) => s + (o.confidence || 0), 0) / objects.length;
      accuracyPct = avg * 100;
    } else if (mode === 'navegacion' && data.avg_confidence !== undefined) {
      const c = Number(data.avg_confidence);
      accuracyPct = c <= 1 ? c * 100 : c;
    }

    // ── Construir secciones de stats por modo ─────────────────────────────
    interface Stat { icon: string; label: string; value: string }
    const sessionStats: Stat[] = [
      { icon: 'calendar-outline', label: 'Fecha', value: dateStr },
      { icon: 'time-outline', label: 'Hora', value: timeStr },
    ];
    const detailStats: Stat[] = [];

    if (mode === 'navegacion') {
      // claves camelCase (móvil guarda así, ver handleRealtimeSessionEnd)
      const frames = data.framesProcessed ?? data.frames_processed;
      const dangers = data.dangerCount ?? data.danger_count;
      const obstacleCounts =
        (data.topObstacles ?? data.obstacle_counts) as Record<string, number> | undefined;
      const durationSec = data.durationSeconds ?? data.session_duration_seconds;

      if (durationSec !== undefined) {
        const min = Number(durationSec) / 60;
        sessionStats.push({
          icon: 'hourglass-outline',
          label: 'Duración',
          value: min < 1
            ? `${Math.round(Number(durationSec))} segundos`
            : `${min.toFixed(1)} minutos`,
        });
      }
      if (frames !== undefined) {
        detailStats.push({
          icon: 'film-outline',
          label: 'Frames procesados',
          value: String(frames),
        });
        if (durationSec && Number(durationSec) > 0) {
          const fps = Number(frames) / Number(durationSec);
          detailStats.push({
            icon: 'speedometer-outline',
            label: 'Tasa de análisis',
            value: `${fps.toFixed(1)} frames / segundo`,
          });
        }
      }
      if (obstacleCounts) {
        const total = Object.values(obstacleCounts).reduce((s, n) => s + n, 0);
        const uniqueObjects = Object.keys(obstacleCounts).length;
        detailStats.push({
          icon: 'apps-outline',
          label: 'Detecciones totales',
          value: String(total),
        });
        detailStats.push({
          icon: 'cube-outline',
          label: 'Objetos únicos',
          value: String(uniqueObjects),
        });
      }
      if (dangers !== undefined) {
        detailStats.push({
          icon: 'alert-circle-outline',
          label: 'Alertas de peligro',
          value: String(dangers),
        });
      }
    } else if (mode === 'exploracion') {
      if (data.object_count !== undefined) {
        detailStats.push({
          icon: 'cube-outline',
          label: 'Objetos detectados',
          value: String(data.object_count),
        });
      }
      detailStats.push({
        icon: 'reader-outline',
        label: 'Texto en imagen',
        value: data.has_text ? 'Sí' : 'No',
      });
    } else if (mode === 'lectura') {
      if (data.document_type) {
        detailStats.push({
          icon: 'document-outline',
          label: 'Tipo de documento',
          value: String(data.document_type),
        });
      }
      if (data.word_count !== undefined) {
        detailStats.push({
          icon: 'text-outline',
          label: 'Palabras detectadas',
          value: String(data.word_count),
        });
      }
      const source = data.ocr_source || data.source;
      if (source) {
        detailStats.push({
          icon: 'cog-outline',
          label: 'Motor usado',
          value: String(source),
        });
      }
    }

    // ── Top obstáculos (solo navegación) ──────────────────────────────────
    const obstacleCounts =
      (data.topObstacles ?? data.obstacle_counts) as Record<string, number> | undefined;
    const topObstacles = obstacleCounts
      ? Object.entries(obstacleCounts).sort((a, b) => b[1] - a[1]).slice(0, 6)
      : [];

    // ── Helper: hacer una card tocable que reproduce su sección ──────────
    // stop() antes de cada speak garantiza que NUNCA se solapen audios.
    const speakSection = (narrative: string) => {
      ttsManager.stop();
      ttsManager.speak(narrative, TtsPriority.INTERRUPT);
    };

    // ── Render de fila de stat ────────────────────────────────────────────
    const renderStatRow = (s: Stat, idx: number, last: boolean) => (
      <View
        key={idx}
        style={{
          flexDirection: 'row',
          alignItems: 'center',
          paddingVertical: 10,
          borderBottomWidth: last ? 0 : 1,
          borderBottomColor: C.border,
          gap: 12,
        }}
      >
        <Ionicons name={s.icon as any} size={20} color={C.primary} />
        <Text style={{ flex: 1, color: C.textSecondary, fontSize: fs(14) }}>{s.label}</Text>
        <Text style={{ color: C.text, fontSize: fs(14), fontWeight: '600' }}>{s.value}</Text>
      </View>
    );

    return (
      <View style={styles.resultsContainer}>
        <ScrollView contentContainerStyle={styles.resultsContent}>
          {/* Header */}
          <View style={styles.historyHeader}>
            <View style={[styles.statusBadge, { backgroundColor: C.primary + '20' }]}>
              <Ionicons
                name={(modeIcons[mode] || 'help-circle') as any}
                size={16}
                color={C.primary}
              />
              <Text style={[styles.statusBadgeText, { color: C.primary }]}>
                {modeLabels[mode] || mode}
              </Text>
            </View>
            <Text style={[styles.resultTitle, { color: C.text, fontSize: fs(22), marginTop: 8 }]}>
              Detalle de la sesión
            </Text>
          </View>

          {/* Onda de voz — visible mientras el TTS habla */}
          <VoiceWave color={C.primary} />

          {/* Card EXACTITUD destacada (con número grande) — tocable */}
          {accuracyPct !== null && (
            <TouchableOpacity
              activeOpacity={0.7}
              onPress={() => speakSection(buildAccuracySpeech(accuracyPct!, mode))}
              accessibilityLabel="Exactitud"
              accessibilityHint="Toca para escuchar"
              accessibilityRole="button"
              style={[
                styles.resultCard,
                {
                  backgroundColor: C.primary + '15',
                  alignItems: 'center',
                  paddingVertical: 24,
                },
              ]}
            >
              <Text style={{ color: C.textSecondary, fontSize: fs(13), letterSpacing: 1 }}>
                EXACTITUD
              </Text>
              <Text style={{ color: C.primary, fontSize: fs(56), fontWeight: 'bold', marginTop: 4 }}>
                {accuracyPct.toFixed(0)}%
              </Text>
              <Text style={{ color: C.textSecondary, fontSize: fs(13), marginTop: 4 }}>
                {mode === 'lectura' ? 'OCR del documento'
                  : mode === 'exploracion' ? 'confianza promedio'
                  : 'detección promedio'}
              </Text>
            </TouchableOpacity>
          )}

          {/* Card RESUMEN (corto, generado localmente, tocable) */}
          <TouchableOpacity
            activeOpacity={0.7}
            onPress={() => speakSection(buildShortSummary(item))}
            accessibilityLabel="Resumen"
            accessibilityHint="Toca para escuchar el resumen"
            accessibilityRole="button"
            style={[styles.resultCard, { backgroundColor: C.surface }]}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <Ionicons name="document-text-outline" size={18} color={C.primary} />
              <Text style={{ color: C.textSecondary, fontSize: fs(13), fontWeight: '700', letterSpacing: 1 }}>
                RESUMEN
              </Text>
              <View style={{ flex: 1 }} />
              <Ionicons name="volume-medium-outline" size={16} color={C.textSecondary} />
            </View>
            <Text style={[styles.resultDescription, { color: C.text, fontSize: fs(15) }]}>
              {buildShortSummary(item)}
            </Text>
          </TouchableOpacity>

          {/* Card SESIÓN — tocable */}
          <TouchableOpacity
            activeOpacity={0.7}
            onPress={() => speakSection(buildSessionSpeech(item))}
            accessibilityLabel="Sesión"
            accessibilityHint="Toca para escuchar fecha, hora y duración"
            accessibilityRole="button"
            style={[styles.resultCard, { backgroundColor: C.surface }]}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <Ionicons name="information-circle-outline" size={18} color={C.primary} />
              <Text style={{ color: C.textSecondary, fontSize: fs(13), fontWeight: '700', letterSpacing: 1 }}>
                SESIÓN
              </Text>
              <View style={{ flex: 1 }} />
              <Ionicons name="volume-medium-outline" size={16} color={C.textSecondary} />
            </View>
            {sessionStats.map((s, i) => renderStatRow(s, i, i === sessionStats.length - 1))}
          </TouchableOpacity>

          {/* Card ESTADÍSTICAS específicas del modo — tocable */}
          {detailStats.length > 0 && (
            <TouchableOpacity
              activeOpacity={0.7}
              onPress={() => speakSection(buildStatsSpeech(item))}
              accessibilityLabel="Estadísticas"
              accessibilityHint="Toca para escuchar las estadísticas"
              accessibilityRole="button"
              style={[styles.resultCard, { backgroundColor: C.surface }]}
            >
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <Ionicons name="stats-chart-outline" size={18} color={C.primary} />
                <Text style={{ color: C.textSecondary, fontSize: fs(13), fontWeight: '700', letterSpacing: 1 }}>
                  ESTADÍSTICAS
                </Text>
                <View style={{ flex: 1 }} />
                <Ionicons name="volume-medium-outline" size={16} color={C.textSecondary} />
              </View>
              {detailStats.map((s, i) => renderStatRow(s, i, i === detailStats.length - 1))}
            </TouchableOpacity>
          )}

          {/* Card OBSTÁCULOS MÁS FRECUENTES (solo navegación) — tocable */}
          {topObstacles.length > 0 && (
            <TouchableOpacity
              activeOpacity={0.7}
              onPress={() => speakSection(buildObstaclesSpeech(item))}
              accessibilityLabel="Obstáculos más frecuentes"
              accessibilityHint="Toca para escuchar"
              accessibilityRole="button"
              style={[styles.resultCard, { backgroundColor: C.surface }]}
            >
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <Ionicons name="warning-outline" size={18} color={C.primary} />
                <Text style={{ color: C.textSecondary, fontSize: fs(13), fontWeight: '700', letterSpacing: 1 }}>
                  OBSTÁCULOS MÁS FRECUENTES
                </Text>
                <View style={{ flex: 1 }} />
                <Ionicons name="volume-medium-outline" size={16} color={C.textSecondary} />
              </View>
              {topObstacles.map(([name, count], i) => (
                <View
                  key={i}
                  style={{
                    flexDirection: 'row',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    paddingVertical: 8,
                    borderBottomWidth: i < topObstacles.length - 1 ? 1 : 0,
                    borderBottomColor: C.border,
                  }}
                >
                  <Text style={{ color: C.text, fontSize: fs(14), textTransform: 'capitalize' }}>
                    {name}
                  </Text>
                  <Text style={{ color: C.primary, fontSize: fs(14), fontWeight: '700' }}>
                    {count}×
                  </Text>
                </View>
              ))}
            </TouchableOpacity>
          )}

          {/* Card OBJETOS DETECTADOS (exploración) — tocable */}
          {objects && objects.length > 0 && (
            <TouchableOpacity
              activeOpacity={0.7}
              onPress={() => speakSection(buildObjectsSpeech(item))}
              accessibilityLabel="Objetos detectados"
              accessibilityHint="Toca para escuchar"
              accessibilityRole="button"
              style={[styles.resultCard, { backgroundColor: C.surface }]}
            >
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <Ionicons name="grid-outline" size={18} color={C.primary} />
                <Text style={{ color: C.textSecondary, fontSize: fs(13), fontWeight: '700', letterSpacing: 1 }}>
                  OBJETOS DETECTADOS ({objects.length})
                </Text>
                <View style={{ flex: 1 }} />
                <Ionicons name="volume-medium-outline" size={16} color={C.textSecondary} />
              </View>
              {objects.slice(0, 15).map((o: any, idx: number) => {
                const conf = typeof o.confidence === 'number' ? o.confidence : 0;
                const dist = o.distance_zone || o.distance_estimate || '';
                const last = idx === Math.min(15, objects.length) - 1;
                return (
                  <View
                    key={idx}
                    style={{
                      flexDirection: 'row',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      paddingVertical: 8,
                      borderBottomWidth: last ? 0 : 1,
                      borderBottomColor: C.border,
                    }}
                  >
                    <View style={{ flex: 1 }}>
                      <Text style={{ color: C.text, fontSize: fs(14), textTransform: 'capitalize' }}>
                        {o.name_es || o.name || 'desconocido'}
                      </Text>
                      {dist ? (
                        <Text style={{ color: C.textSecondary, fontSize: fs(12), marginTop: 2 }}>
                          {String(dist).replace('_', ' ')}
                        </Text>
                      ) : null}
                    </View>
                    <View
                      style={{
                        backgroundColor: C.primary + '20',
                        paddingHorizontal: 10,
                        paddingVertical: 4,
                        borderRadius: 12,
                      }}
                    >
                      <Text style={{ color: C.primary, fontSize: fs(13), fontWeight: '700' }}>
                        {(conf * 100).toFixed(0)}%
                      </Text>
                    </View>
                  </View>
                );
              })}
              {objects.length > 15 && (
                <Text style={{ color: C.textSecondary, fontSize: fs(12), marginTop: 8, fontStyle: 'italic' }}>
                  …y {objects.length - 15} objetos más.
                </Text>
              )}
            </TouchableOpacity>
          )}
        </ScrollView>

        <View style={styles.historyFixedActions}>
          <Button
            title="Volver al historial"
            onPress={() => {
              ttsManager.stop();
              setSelectedHistoryItem(null);
              ttsManager.speak('Volviendo al historial.', TtsPriority.HIGH);
            }}
            size="large"
            icon={<Ionicons name="arrow-back" size={20} color={C.background} />}
            style={styles.resultActionButton}
          />
        </View>
      </View>
    );
  };

  const renderHistory = () => {
    const modeLabels: Record<string, string> = {
      navegacion: 'Navegación',
      exploracion: 'Exploración',
      lectura: 'Lectura',
    };

    const modeIcons: Record<string, string> = {
      navegacion: 'compass',
      exploracion: 'search',
      lectura: 'document-text',
    };

    return (
      <View style={styles.resultsContainer}>
        {/* Solo el historial hace scroll */}
        <ScrollView contentContainerStyle={styles.resultsContent}>
          {/* Header */}
          <View style={styles.historyHeader}>
            <Text style={styles.resultTitle}>Historial de Análisis</Text>
            <Text style={styles.historySubtitle}>
              {historyItems.length} registro{historyItems.length !== 1 ? 's' : ''}
            </Text>
          </View>

          {/* Onda de voz */}
          <VoiceWave color={C.primary} />

          {/* Lista vacía */}
          {historyItems.length === 0 && (
            <View style={styles.historyEmpty}>
              <Ionicons name="time-outline" size={48} color={C.textSecondary} />
              <Text style={styles.historyEmptyText}>
                Aún no hay análisis en tu historial.
              </Text>
              <Text style={styles.historyEmptyHint}>
                Usa cualquier modo para empezar.
              </Text>
            </View>
          )}

          {/* Lista de items */}
          {historyItems.map((item) => {
            const date = new Date(item.createdAt);
            // Visual: corto. Para hablar: completo y natural.
            const timeStr = date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });
            const dateStr = date.toLocaleDateString('es-ES', { day: 'numeric', month: 'short' });
            const dateLong = date.toLocaleDateString('es-ES', { day: 'numeric', month: 'long' });
            // Convertir hora 24h a 12h con momento del día (mañana/tarde/noche)
            const h24 = date.getHours();
            const minute = date.getMinutes();
            const period =
              h24 < 6 ? 'de la madrugada'
              : h24 < 12 ? 'de la mañana'
              : h24 < 19 ? 'de la tarde'
              : 'de la noche';
            const h12 = h24 === 0 ? 12 : h24 > 12 ? h24 - 12 : h24;
            const hourSpoken = minute === 0
              ? `${h12} en punto ${period}`
              : `${h12} y ${minute < 10 ? '0' + minute : minute} ${period}`;
            const modeLabel = modeLabels[item.mode] || item.mode;

            return (
              <TouchableOpacity
                key={item.id}
                style={styles.historyItem}
                onPress={() => {
                  // Doble tap (≤400 ms entre dos toques en el MISMO item) abre el detalle.
                  // Single tap anuncia: el modo (especificando que es un registro),
                  // fecha completa y hora natural ("3 de mayo a las 8 y 30 de la noche").
                  const now = Date.now();
                  const isDoubleTap =
                    lastTapRef.current.id === item.id &&
                    (now - lastTapRef.current.time) < 400;
                  lastTapRef.current = { id: item.id, time: now };

                  if (isDoubleTap) {
                    ttsManager.stop();
                    setSelectedHistoryItem(item);
                  } else {
                    ttsManager.stop();
                    ttsManager.speak(
                      `Este registro corresponde al modo ${modeLabel}. Sesión del ${dateLong} a las ${hourSpoken}. Toca dos veces para ver los detalles.`,
                      TtsPriority.INTERRUPT,
                    );
                  }
                }}
                accessibilityLabel={`Modo ${modeLabel}, ${dateLong} a las ${hourSpoken}`}
                accessibilityHint="Toca dos veces para ver los detalles"
              >
                <View style={styles.historyItemIcon}>
                  <Ionicons
                    name={(modeIcons[item.mode] || 'help-circle') as any}
                    size={20}
                    color={C.primary}
                  />
                </View>
                <View style={styles.historyItemContent}>
                  <View style={styles.historyItemHeader}>
                    <Text style={styles.historyItemMode}>
                      {modeLabels[item.mode] || item.mode}
                      {item.readingMode ? ` · ${item.readingMode}` : ''}
                    </Text>
                    <Text style={styles.historyItemDate}>{dateStr} {timeStr}</Text>
                  </View>
                  <Text style={styles.historyItemSummary} numberOfLines={2}>
                    {item.resultSummary || 'Sin resumen disponible'}
                  </Text>
                </View>
              </TouchableOpacity>
            );
          })}
        </ScrollView>

        {/* Botones fijos — siempre visibles sin necesidad de bajar */}
        <View style={styles.historyFixedActions}>
          {historyItems.length > 0 && (
            <Button
              title="Borrar Historial"
              onPress={handleClearHistory}
              variant="outline"
              size="large"
              icon={<Ionicons name="trash" size={20} color={C.error} />}
              style={styles.resultActionButton}
            />
          )}
          <Button
            title="Volver"
            onPress={() => {
              handleReset();
              ttsManager.speak('Volviendo al inicio.', TtsPriority.HIGH);
            }}
            size="large"
            icon={<Ionicons name="arrow-back" size={20} color={C.background} />}
            style={styles.resultActionButton}
          />
        </View>
      </View>
    );
  };

  // Vista de tiempo real (Navegación con riesgo integrado)
  const renderRealtime = () => {

    return (
      <View style={styles.cameraContainer}>
        <CameraView ref={cameraRef} style={styles.camera} facing="back">
          <View style={styles.realtimeOverlay}>
            {/* Barra de estado superior */}
            <View style={styles.realtimeStatusBar}>
              <View style={styles.realtimeStatusLeft}>
                <View
                  style={[
                    styles.connectionDot,
                    { backgroundColor: wsStatus === 'connected' ? C.success : C.warning },
                  ]}
                />
                <Text style={styles.realtimeStatusText}>
                  {MODE_CONFIG[analysisMode].label}
                  {wsStatus === 'connected' ? '' : ' - Conectando...'}
                </Text>
              </View>
              {latestResult && (
                <Text style={styles.realtimeStatusText}>
                  {latestResult.processing_time_ms}ms
                  {latestResult.tracked_count != null ? ` | ${latestResult.tracked_count} obj` : ''}
                </Text>
              )}
            </View>

            {/* Indicador de estado:
                - PELIGRO (rojo)    = objeto intrínsecamente peligroso (vehículo, escalera, balcón...)
                - ATENCIÓN (naranja) = obstáculo cercano o acercándose que puede impedir el paso
                - CAMINO LIBRE (verde) = sin obstáculos relevantes al frente */}
            {latestResult && (
              <View style={styles.riskIndicatorContainer}>
                {(() => {
                  const isCritical =
                    latestResult.alert_type === 'peligro' ||
                    (latestResult.has_danger && latestResult.priority === 'critical');
                  const isWarning =
                    latestResult.alert_type === 'atencion' ||
                    latestResult.has_danger ||
                    latestResult.path_clear === false;
                  const bg = isCritical ? '#EF4444' : isWarning ? '#F97316' : '#22C55E';
                  const icon = isCritical ? 'warning' : isWarning ? 'alert-circle' : 'checkmark-circle';
                  const label = isCritical ? '¡PELIGRO!' : isWarning ? 'ATENCIÓN' : 'CAMINO LIBRE';
                  return (
                    <View style={[styles.riskIndicator, { backgroundColor: bg }]}>
                      <Ionicons name={icon} size={48} color="white" />
                      <Text style={styles.riskIndicatorText}>{label}</Text>
                    </View>
                  );
                })()}
              </View>
            )}

            {/* Resumen de detección en la parte inferior */}
            <View style={styles.realtimeSummaryContainer}>
              <View style={styles.realtimeSummary}>
                <Text style={styles.realtimeSummaryText}>
                  {latestResult?.summary || 'Analizando entorno...'}
                </Text>
                {/* Lista de objetos detectados */}
                {latestResult && latestResult.objects.length > 0 && (
                  <View style={styles.realtimeObjectList}>
                    {[...latestResult.objects]
                      .sort((a, b) => {
                        const order = { high: 0, medium: 1, low: 2 };
                        return (order[a.priority ?? 'low'] ?? 2) - (order[b.priority ?? 'low'] ?? 2);
                      })
                      .slice(0, 5)
                      .map((obj, idx) => {
                      const zoneColor =
                        obj.distance_zone === 'muy_cerca' ? '#EF4444' :
                        obj.distance_zone === 'cerca' ? '#F59E0B' : '#22C55E';
                      const priorityIcon =
                        obj.priority === 'high' ? 'alert-circle' :
                        obj.priority === 'medium' ? 'remove-circle' : null;
                      return (
                        <View key={idx} style={styles.realtimeObjectItem}>
                          <View style={styles.realtimeObjLeft}>
                            <View style={[styles.zoneDot, { backgroundColor: zoneColor }]} />
                            {priorityIcon && (
                              <Ionicons
                                name={priorityIcon as any}
                                size={12}
                                color={obj.priority === 'high' ? '#EF4444' : '#F59E0B'}
                              />
                            )}
                            <Text style={[
                              styles.realtimeObjectName,
                              obj.priority === 'high' && { fontWeight: '700' },
                            ]}>{obj.name_es}</Text>
                          </View>
                          <Text style={[styles.realtimeObjectDistance, { color: zoneColor }]}>
                            {obj.distance_estimate || `${(obj.confidence * 100).toFixed(0)}%`}
                          </Text>
                        </View>
                      );
                    })}
                  </View>
                )}
              </View>
            </View>
          </View>
        </CameraView>

        {/* Controles */}
        <View style={styles.realtimeControls}>
          <TouchableOpacity
            style={styles.stopRealtimeButton}
            onPress={() => {
              handleReset();
              ttsManager.speak('Detenido.', TtsPriority.INTERRUPT);
            }}
            accessibilityLabel="Detener"
            accessibilityRole="button"
          >
            <Ionicons name="stop" size={28} color={C.text} />
            <Text style={styles.stopRealtimeText}>Detener</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // Vista de cámara (Exploración y Lectura)
  // Toca cualquier parte de la cámara para capturar — más accesible para personas ciegas
  const renderCamera = () => (
    <View style={styles.cameraContainer}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back" />

      {/* Capa táctil sobre la cámara (posición absoluta, sin ser hijo de CameraView) */}
      <TouchableOpacity
        style={styles.cameraOverlay}
        onPress={handleCapture}
        activeOpacity={0.9}
        accessibilityLabel="Toca la pantalla para capturar foto"
        accessibilityRole="button"
      >
        <View style={styles.cameraFrame} />
        <Text style={styles.cameraTapHint}>Toca para capturar</Text>
      </TouchableOpacity>

      <View style={styles.cameraControls}>
        <TouchableOpacity
          style={styles.cameraButton}
          onPress={() => {
            handleReset();
            ttsManager.speak('Cerrando cámara.', TtsPriority.HIGH);
          }}
          accessibilityLabel="Cerrar cámara"
          accessibilityRole="button"
        >
          <Ionicons name="close" size={32} color={C.text} />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.cameraButton}
          onPress={handlePickImage}
          accessibilityLabel="Subir imagen desde galería"
          accessibilityRole="button"
        >
          <Ionicons name="images" size={32} color={C.text} />
        </TouchableOpacity>
      </View>
    </View>
  );

  // Pantalla de procesamiento
  const renderProcessing = () => (
    <View style={styles.processingContainer}>
      {capturedImage && (
        <Image source={{ uri: capturedImage }} style={styles.previewImage} />
      )}
      <ActivityIndicator size="large" color={C.primary} style={styles.loader} />
      <Text style={styles.processingText}>Analizando imagen...</Text>
      <VoiceWave color={C.primary} />
    </View>
  );

  // Pantalla de resultados
  const renderResults = () => (
    <ScrollView style={styles.resultsContainer} contentContainerStyle={styles.resultsContent}>
      {capturedImage && (
        <Image source={{ uri: capturedImage }} style={styles.resultImage} />
      )}

      {/* Onda de voz */}
      <VoiceWave color={C.primary} />

      <View style={styles.resultCard}>
        <Text style={styles.resultTitle}>
          {MODE_CONFIG[analysisMode].label}
        </Text>

        {/* Resultado de Navegación */}
        {navResult && (
          <View>
            <Text style={styles.resultDescription}>{navResult.instruction}</Text>
            {navResult.path_clear && (
              <View style={[styles.statusBadge, { backgroundColor: C.success + '20' }]}>
                <Text style={[styles.statusBadgeText, { color: C.success }]}>
                  Camino libre
                </Text>
              </View>
            )}
            {navResult.obstacles.length > 0 && (
              <View style={styles.obstacleList}>
                {navResult.obstacles.map((obj, idx) => {
                  const zoneColor =
                    obj.distance_zone === 'muy_cerca' ? '#EF4444' :
                    obj.distance_zone === 'cerca' ? '#F59E0B' : '#22C55E';
                  return (
                    <View key={idx} style={styles.objectItem}>
                      <View style={styles.objectInfo}>
                        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                          <View style={[styles.zoneDot, { backgroundColor: zoneColor }]} />
                          <Text style={styles.objectName}>{obj.name_es}</Text>
                        </View>
                        {obj.distance_estimate && (
                          <Text style={[styles.objectDistance, { color: zoneColor }]}>
                            {obj.distance_estimate}
                          </Text>
                        )}
                      </View>
                    </View>
                  );
                })}
              </View>
            )}
          </View>
        )}

        {/* Resultado de Exploración */}
        {explorationResult && (
          <View>
            <Text style={styles.resultDescription}>{explorationResult.description}</Text>
            {explorationResult.has_text && (
              <View style={styles.textBox}>
                <Text style={styles.textBoxLabel}>Texto detectado:</Text>
                <Text style={styles.textBoxContent}>{explorationResult.detected_text}</Text>
              </View>
            )}
            {explorationResult.object_count > 0 && (
              <Text style={styles.objectCount}>
                {explorationResult.object_count} objeto(s) detectado(s)
              </Text>
            )}
          </View>
        )}

        {/* Resultado de Lectura Inteligente */}
        {smartResult && (
          <View>
            {/* Badge de tipo de documento */}
            <View style={[styles.statusBadge, { backgroundColor: C.primary + '20' }]}>
              <Ionicons name="document-text" size={16} color={C.primary} />
              <Text style={[styles.statusBadgeText, { color: C.primary }]}>
                {smartResult.document_type_label}
              </Text>
            </View>

            {/* Narrativa principal */}
            <Text style={styles.resultDescription}>{smartResult.narrative}</Text>

            {/* Totales destacados (si hay) */}
            {(smartResult.extracted_fields?.totals?.length ?? 0) > 0 && (
              <View style={styles.totalsHighlight}>
                <Ionicons name="cash" size={16} color={C.warning} />
                <Text style={styles.totalsText}>
                  {smartResult.extracted_fields!.totals.join(' | ')}
                </Text>
              </View>
            )}

            {/* Caption visual (para imágenes con poco texto) */}
            {smartResult.visual_caption && (
              <View style={styles.textBox}>
                <Text style={styles.textBoxLabel}>Descripción visual:</Text>
                <Text style={styles.textBoxContent}>{smartResult.visual_caption}</Text>
              </View>
            )}

            {/* Stats */}
            <Text style={styles.confidence}>
              {smartResult.word_count} palabras
              {smartResult.ocr_confidence ? ` • ${smartResult.ocr_confidence.toFixed(0)}% confianza` : ''}
              {` • ${smartResult.reading_mode}`}
            </Text>

            {/* Indicador de calidad de imagen — solo si es ilegible */}
            {smartResult.image_quality && !smartResult.image_quality.is_acceptable && (
              <View style={[
                styles.totalsHighlight,
                {
                  backgroundColor: C.error + '20',
                  marginTop: 8,
                },
              ]}>
                <Ionicons
                  name="warning"
                  size={16}
                  color={C.error}
                />
                <Text style={[
                  styles.totalsText,
                  { color: C.error },
                ]}>
                  {smartResult.image_quality.feedback_text || 'No pude leer. Intenta tomar otra foto.'}
                </Text>
              </View>
            )}
          </View>
        )}

        {/* Detalles de obstáculos (parte de navegación unificada) */}
        {navResult && navResult.obstacle_details && navResult.obstacle_details.length > 0 && (
          <View>
            <View style={[
              styles.statusBadge,
              {
                backgroundColor: navResult.has_danger
                  ? (navResult.priority === 'critical' ? '#EF444420' : '#F59E0B20')
                  : '#22C55E20',
              },
            ]}>
              <Ionicons
                name={navResult.has_danger ? 'warning' : 'checkmark-circle'}
                size={20}
                color={navResult.has_danger
                  ? (navResult.priority === 'critical' ? '#EF4444' : '#F59E0B')
                  : '#22C55E'}
              />
              <Text style={[
                styles.statusBadgeText,
                {
                  color: navResult.has_danger
                    ? (navResult.priority === 'critical' ? '#EF4444' : '#F59E0B')
                    : '#22C55E',
                },
              ]}>
                {navResult.has_danger
                  ? (navResult.priority === 'critical' ? 'PELIGRO' : 'PRECAUCIÓN')
                  : 'SEGURO'}
              </Text>
            </View>
            <View style={styles.obstacleList}>
              {navResult.obstacle_details.map((detail, idx) => (
                <View key={idx} style={styles.objectItem}>
                  <View style={styles.objectInfo}>
                    <Text style={styles.objectName}>{detail.name}</Text>
                    <Text style={[styles.objectDistance, {
                      color: detail.risk_score >= 0.75 ? '#EF4444' :
                             detail.risk_score >= 0.5 ? '#F59E0B' : '#F59E0B',
                    }]}>
                      {detail.proximity === 'muy_cerca' ? 'Muy cerca' : 'Cerca'} - {detail.position}
                      {detail.movement === 'acercandose' ? ' - Se acerca' : ''}
                    </Text>
                  </View>
                </View>
              ))}
            </View>
          </View>
        )}
      </View>

      <View style={styles.resultActions}>
        <Button
          title="Nueva Imagen"
          onPress={() => {
            handleReset();
            ttsManager.speak('Nueva captura.', TtsPriority.HIGH);
          }}
          variant="outline"
          size="large"
          icon={<Ionicons name="refresh" size={20} color={C.primary} />}
          style={styles.resultActionButton}
        />
        <Button
          title={isSpeakingState ? 'Detener' : 'Repetir'}
          onPress={handleRepeat}
          size="large"
          icon={
            <Ionicons
              name={isSpeakingState ? 'stop' : 'volume-high'}
              size={20}
              color={C.background}
            />
          }
          style={styles.resultActionButton}
        />
      </View>
    </ScrollView>
  );

  // Pantalla de error
  const renderError = () => (
    <View style={styles.errorContainer}>
      <Ionicons name="alert-circle" size={64} color={C.error} />
      <Text style={styles.errorTitle}>Error</Text>
      <Text style={styles.errorMessage}>{error}</Text>
      <VoiceWave color={C.error} />
      <Button
        title="Intentar de nuevo"
        onPress={() => {
          handleReset();
          ttsManager.speak('Volviendo al inicio.', TtsPriority.HIGH);
        }}
        size="large"
      />
    </View>
  );

  return (
    <View style={[styles.container, { backgroundColor: C.background }]}>
      <StatusBar style={C.background === '#ffffff' ? 'dark' : 'light'} />
      {renderContent()}
      <SettingsScreen
        visible={settingsVisible}
        onClose={() => setSettingsVisible(false)}
      />
    </View>
  );
}

const createStyles = (C: ThemeColors) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.background,
  },
  // Home
  settingsBtn: {
    position: 'absolute',
    top: 52,
    right: 20,
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10,
  },
  homeContainer: {
    flex: 1,
    padding: 24,
    justifyContent: 'center',
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 42,
    fontWeight: 'bold',
    color: C.text,
    marginTop: 12,
  },
  subtitle: {
    fontSize: 18,
    color: C.textSecondary,
    marginTop: 4,
  },
  connectionStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 32,
  },
  connectionDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  connectionText: {
    color: C.textSecondary,
    fontSize: 14,
  },
  modeSelector: {
    marginBottom: 32,
  },
  modeLabel: {
    color: C.textSecondary,
    fontSize: 14,
    marginBottom: 12,
    textAlign: 'center',
  },
  modeButtons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 8,
  },
  modeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    width: '47%',
    minHeight: 48,
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: C.primary,
    gap: 6,
  },
  modeButtonActive: {
    backgroundColor: C.primary,
  },
  modeButtonText: {
    color: C.primary,
    fontSize: 14,
    fontWeight: '500',
  },
  modeButtonTextActive: {
    color: C.background,
  },
  readingModeSelector: {
    marginBottom: 16,
  },
  readingModeLabel: {
    color: C.textSecondary,
    fontSize: 12,
    marginBottom: 8,
    textAlign: 'center',
  },
  readingModeButtons: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 8,
  },
  readingModeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: C.primary,
    gap: 4,
  },
  readingModeButtonActive: {
    backgroundColor: C.primary,
  },
  readingModeButtonText: {
    color: C.primary,
    fontSize: 13,
    fontWeight: '500',
  },
  readingModeButtonTextActive: {
    color: C.background,
  },
  mainButtons: {
    gap: 14,
  },
  // Pill: bordes muy redondeados — el único cambio que se mantiene del rediseño
  mainButton: {
    width: '100%',
    borderRadius: 32,
    minHeight: 60,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.18,
    shadowRadius: 8,
    elevation: 5,
  },
  secondaryButton: {
    width: '100%',
    borderRadius: 32,
    minHeight: 56,
  },
  ttsToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 32,
    minHeight: 48,
    gap: 8,
  },
  ttsToggleText: {
    color: C.textSecondary,
    fontSize: 14,
  },
  // Camera
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 20,
  },
  cameraTapHint: {
    color: C.text,
    fontSize: 15,
    opacity: 0.7,
    backgroundColor: 'rgba(0,0,0,0.4)',
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 20,
  },
  cameraFrame: {
    width: SCREEN_WIDTH - 48,
    height: SCREEN_WIDTH - 48,
    borderWidth: 2,
    borderColor: C.primary,
    borderRadius: 16,
    opacity: 0.5,
  },
  cameraControls: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  cameraButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: C.surface,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: C.primary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: C.text,
  },
  // Realtime
  realtimeOverlay: {
    flex: 1,
    justifyContent: 'space-between',
  },
  realtimeStatusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 60,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(0,0,0,0.4)',
    paddingBottom: 12,
  },
  realtimeStatusLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  realtimeStatusText: {
    color: C.text,
    fontSize: 14,
    fontWeight: '600',
  },
  riskIndicatorContainer: {
    alignItems: 'center',
    marginTop: 20,
  },
  riskIndicator: {
    alignItems: 'center',
    justifyContent: 'center',
    width: 120,
    height: 120,
    borderRadius: 60,
    opacity: 0.9,
  },
  riskIndicatorText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
    marginTop: 4,
  },
  realtimeSummaryContainer: {
    padding: 20,
    paddingBottom: 100,
  },
  realtimeSummary: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 16,
    padding: 16,
  },
  realtimeSummaryText: {
    color: C.text,
    fontSize: 18,
    fontWeight: '500',
    lineHeight: 26,
  },
  realtimeObjectList: {
    marginTop: 8,
    gap: 4,
  },
  realtimeObjectItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  realtimeObjLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  zoneDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  realtimeObjectName: {
    color: C.text,
    fontSize: 14,
    fontWeight: '500',
  },
  realtimeObjectDistance: {
    color: C.primary,
    fontSize: 12,
    fontWeight: '600',
  },
  realtimeControls: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  stopRealtimeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.error,
    paddingHorizontal: 40,
    paddingVertical: 16,
    borderRadius: 30,
    minHeight: 52,
    gap: 8,
  },
  stopRealtimeText: {
    color: C.text,
    fontSize: 16,
    fontWeight: '600',
  },
  // Processing
  processingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  previewImage: {
    width: SCREEN_WIDTH - 48,
    height: SCREEN_WIDTH - 48,
    borderRadius: 16,
    marginBottom: 24,
  },
  loader: {
    marginBottom: 16,
  },
  processingText: {
    color: C.text,
    fontSize: 18,
  },
  // Results
  resultsContainer: {
    flex: 1,
  },
  resultsContent: {
    padding: 24,
  },
  resultImage: {
    width: '100%',
    height: 250,
    borderRadius: 16,
    marginBottom: 16,
  },
  resultCard: {
    backgroundColor: C.surface,
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: C.text,
    marginBottom: 12,
  },
  resultDescription: {
    fontSize: 16,
    color: C.text,
    lineHeight: 24,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    gap: 6,
    marginBottom: 12,
  },
  statusBadgeText: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  obstacleList: {
    marginTop: 12,
  },
  textBox: {
    backgroundColor: C.secondary,
    borderRadius: 12,
    padding: 12,
    marginTop: 12,
  },
  textBoxLabel: {
    fontSize: 12,
    color: C.textSecondary,
    marginBottom: 4,
  },
  textBoxContent: {
    fontSize: 14,
    color: C.text,
  },
  objectCount: {
    fontSize: 14,
    color: C.textSecondary,
    marginTop: 12,
  },
  totalsHighlight: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.warning + '15',
    borderRadius: 12,
    padding: 12,
    marginTop: 12,
    gap: 8,
  },
  totalsText: {
    flex: 1,
    fontSize: 15,
    fontWeight: '600',
    color: C.warning,
  },
  confidence: {
    fontSize: 14,
    color: C.textSecondary,
    marginTop: 8,
  },
  noResult: {
    fontSize: 16,
    color: C.textSecondary,
    fontStyle: 'italic',
  },
  objectItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: C.secondary,
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
  },
  objectInfo: {
    flex: 1,
    marginRight: 8,
  },
  objectName: {
    fontSize: 14,
    color: C.text,
  },
  objectDistance: {
    fontSize: 12,
    color: C.primary,
    marginTop: 2,
  },
  resultActions: {
    flexDirection: 'row',
    gap: 12,
  },
  historyFixedActions: {
    flexDirection: 'row',
    gap: 12,
    paddingHorizontal: 24,
    paddingVertical: 16,
    paddingBottom: 32,
    backgroundColor: C.background,
    borderTopWidth: 1,
    borderTopColor: C.border,
  },
  resultActionButton: {
    flex: 1,
    minHeight: 52,
  },
  // Historial
  historyButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 12,
    marginTop: 8,
  },
  historyButtonText: {
    fontSize: 15,
    color: C.primary,
  },
  historyHeader: {
    marginBottom: 16,
  },
  historySubtitle: {
    fontSize: 14,
    color: C.textSecondary,
    marginTop: 4,
  },
  historyEmpty: {
    alignItems: 'center',
    paddingVertical: 48,
    gap: 12,
  },
  historyEmptyText: {
    fontSize: 16,
    color: C.textSecondary,
    textAlign: 'center',
  },
  historyEmptyHint: {
    fontSize: 14,
    color: C.textSecondary,
    textAlign: 'center',
    fontStyle: 'italic',
  },
  historyItem: {
    flexDirection: 'row',
    backgroundColor: C.surface,
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    gap: 12,
  },
  historyItemIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: C.primary + '15',
    alignItems: 'center',
    justifyContent: 'center',
  },
  historyItemContent: {
    flex: 1,
  },
  historyItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  historyItemMode: {
    fontSize: 14,
    fontWeight: '600',
    color: C.primary,
  },
  historyItemDate: {
    fontSize: 12,
    color: C.textSecondary,
  },
  historyItemSummary: {
    fontSize: 14,
    color: C.text,
    lineHeight: 20,
  },
  // Error
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: C.error,
    marginTop: 16,
    marginBottom: 8,
  },
  errorMessage: {
    fontSize: 16,
    color: C.textSecondary,
    textAlign: 'center',
    marginBottom: 24,
  },
});
