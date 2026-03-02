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

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  Alert,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';

import { Button } from '../components/Button';
import { AnimatedEye } from '../components/AnimatedEye';
import { COLORS, ANALYSIS_MODES, REALTIME_MODES, AnalysisMode } from '../constants/config';
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

const { width: SCREEN_WIDTH } = Dimensions.get('window');

type AppState = 'home' | 'camera' | 'processing' | 'results' | 'error' | 'realtime' | 'history';

// Configuración de cada modo
const MODE_CONFIG: Record<AnalysisMode, { label: string; icon: string; description: string }> = {
  navegacion: { label: 'Navegación', icon: 'compass', description: 'Navegación asistida con detección de riesgo' },
  exploracion: { label: 'Exploración', icon: 'eye', description: 'Describe el entorno' },
  lectura: { label: 'Lectura', icon: 'document-text', description: 'Lee textos' },
};

export function HomeScreen() {
  // Preferencias persistentes (desde Context + AsyncStorage)
  const {
    analysisMode,
    readingMode,
    ttsEnabled,
    setAnalysisMode,
    setReadingMode,
    setTtsEnabled,
  } = usePreferences();

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

  // Sincronizar ttsEnabled con ttsManager
  useEffect(() => {
    ttsManager.setEnabled(ttsEnabled);
  }, [ttsEnabled]);

  // Verificar conexión con el backend + bienvenida al iniciar
  useEffect(() => {
    checkBackendConnection();
    // Bienvenida: el usuario ciego necesita saber que la app abrió
    ttsManager.speak(
      'Bienvenido a NAVIA, tu asistente visual. Selecciona un modo y toca Iniciar Cámara.',
      TtsPriority.HIGH,
    );
  }, []);

  const checkBackendConnection = async () => {
    try {
      await checkHealth();
      setIsBackendConnected(true);
    } catch {
      setIsBackendConnected(false);
      // Informar al usuario ciego que no hay conexión
      ttsManager.speak('Sin conexión al servidor. Verifica tu conexión.', TtsPriority.HIGH);
    }
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
      ttsManager.speak(`Modo ${modeLabel} activado. Apunta la cámara.`, TtsPriority.HIGH);
    } else {
      setAppState('camera');
      ttsManager.speak('Cámara activada. Toca el botón central para capturar.', TtsPriority.HIGH);
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
        ttsManager.speak('Foto capturada.', TtsPriority.HIGH);
        await processImage(photo.uri);
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
    }
  };

  // Procesar imagen según el modo
  const processImage = async (imageUri: string) => {
    ttsManager.speak('Procesando imagen, por favor espera.', TtsPriority.HIGH);

    try {
      switch (analysisMode) {
        case 'navegacion': {
          const result = await analyzeNavigation(imageUri);
          setNavResult(result);
          ttsManager.speakFromBackend(result.instruction, TtsPriority.HIGH);
          // Guardar en historial local
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
          // Si hay problemas de calidad, hablar feedback primero
          if (result.image_quality?.feedback_text) {
            const qualityPriority = result.image_quality.is_acceptable
              ? TtsPriority.HIGH
              : TtsPriority.INTERRUPT;
            ttsManager.speak(result.image_quality.feedback_text, qualityPriority);
          }
          ttsManager.speakFromBackend(result.narrative, TtsPriority.HIGH);
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
      else if (smartResult) ttsManager.speakFromBackend(smartResult.narrative, TtsPriority.HIGH);
    } finally {
      setIsSpeakingState(false);
    }
  };

  // ============================================================================
  // RENDERIZADO
  // ============================================================================

  const renderContent = () => {
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
    <View style={styles.homeContainer}>
      <View style={styles.header}>
        <AnimatedEye size={48} color={COLORS.primary} />
        <Text style={styles.title}>NAVIA</Text>
        <Text style={styles.subtitle}>Asistente Visual con IA</Text>
      </View>

      {/* Indicador de conexión */}
      <View style={styles.connectionStatus}>
        <View
          style={[
            styles.connectionDot,
            {
              backgroundColor:
                isBackendConnected === null ? COLORS.warning
                  : isBackendConnected ? COLORS.success
                  : COLORS.error,
            },
          ]}
        />
        <Text style={styles.connectionText}>
          {isBackendConnected === null ? 'Conectando...'
            : isBackendConnected ? 'Servidor conectado'
            : 'Sin conexión al servidor'}
        </Text>
      </View>

      {/* Selector de modo - grid 2x2 */}
      <View style={styles.modeSelector}>
        <Text style={styles.modeLabel}>Modo de análisis:</Text>
        <View style={styles.modeButtons}>
          {Object.entries(ANALYSIS_MODES).map(([key, value]) => {
            const config = MODE_CONFIG[value];
            return (
              <TouchableOpacity
                key={key}
                style={[
                  styles.modeButton,
                  analysisMode === value && styles.modeButtonActive,
                ]}
                onPress={() => {
                  setAnalysisMode(value);
                  ttsManager.speak(`Modo ${config.label}. ${config.description}.`, TtsPriority.HIGH);
                }}
                accessibilityRole="radio"
                accessibilityState={{ selected: analysisMode === value }}
                accessibilityLabel={`Modo ${config.label}. ${config.description}`}
              >
                <Ionicons
                  name={config.icon as any}
                  size={20}
                  color={analysisMode === value ? COLORS.background : COLORS.primary}
                />
                <Text
                  style={[
                    styles.modeButtonText,
                    analysisMode === value && styles.modeButtonTextActive,
                  ]}
                >
                  {config.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {/* Botones principales */}
      <View style={styles.mainButtons}>
        <Button
          title={isRealtimeMode ? 'Iniciar Cámara' : 'Abrir Cámara'}
          onPress={handleOpenCamera}
          size="xl"
          disabled={isBackendConnected === false}
          icon={<Ionicons name="camera" size={28} color={COLORS.background} />}
          style={styles.mainButton}
        />

        {!isRealtimeMode && (
          <Button
            title="Subir Imagen"
            onPress={handlePickImage}
            variant="outline"
            size="large"
            disabled={isBackendConnected === false}
            icon={<Ionicons name="image" size={24} color={COLORS.primary} />}
            style={styles.secondaryButton}
          />
        )}
      </View>

      {/* Toggle TTS */}
      <TouchableOpacity
        style={styles.ttsToggle}
        onPress={() => {
          const newValue = !ttsEnabled;
          setTtsEnabled(newValue);
          // Siempre habilitar temporalmente para que se escuche la confirmación
          ttsManager.setEnabled(true);
          if (!newValue) {
            ttsManager.speak('Voz desactivada.', TtsPriority.INTERRUPT);
            // Desactivar después de que termine de hablar
            setTimeout(() => ttsManager.setEnabled(false), 3000);
          } else {
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
          color={ttsEnabled ? COLORS.primary : COLORS.textSecondary}
        />
        <Text style={styles.ttsToggleText}>
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
        <Ionicons name="time" size={20} color={COLORS.primary} />
        <Text style={styles.historyButtonText}>Ver Historial</Text>
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
  const renderHistory = () => {
    const modeLabels: Record<string, string> = {
      navegacion: 'Navegación',
      exploracion: 'Exploración',
      lectura: 'Lectura',
    };

    const modeIcons: Record<string, string> = {
      navegacion: 'compass',
      exploracion: 'eye',
      lectura: 'document-text',
    };

    return (
      <View style={styles.resultsContainer}>
        <ScrollView contentContainerStyle={styles.resultsContent}>
          {/* Header */}
          <View style={styles.historyHeader}>
            <Text style={styles.resultTitle}>Historial de Análisis</Text>
            <Text style={styles.historySubtitle}>
              {historyItems.length} registro{historyItems.length !== 1 ? 's' : ''}
            </Text>
          </View>

          {/* Lista vacía */}
          {historyItems.length === 0 && (
            <View style={styles.historyEmpty}>
              <Ionicons name="time-outline" size={48} color={COLORS.textSecondary} />
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
            const timeStr = date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });
            const dateStr = date.toLocaleDateString('es-ES', { day: 'numeric', month: 'short' });

            return (
              <TouchableOpacity
                key={item.id}
                style={styles.historyItem}
                onPress={() => {
                  ttsManager.speakFromBackend(item.resultSummary, TtsPriority.HIGH);
                }}
                accessibilityLabel={`${modeLabels[item.mode] || item.mode}. ${item.resultSummary}`}
              >
                <View style={styles.historyItemIcon}>
                  <Ionicons
                    name={(modeIcons[item.mode] || 'help-circle') as any}
                    size={20}
                    color={COLORS.primary}
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

          {/* Botones */}
          <View style={styles.resultActions}>
            {historyItems.length > 0 && (
              <Button
                title="Borrar Historial"
                onPress={handleClearHistory}
                variant="outline"
                size="large"
                icon={<Ionicons name="trash" size={20} color={COLORS.error} />}
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
              icon={<Ionicons name="arrow-back" size={20} color={COLORS.background} />}
              style={styles.resultActionButton}
            />
          </View>
        </ScrollView>
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
                    { backgroundColor: wsStatus === 'connected' ? COLORS.success : COLORS.warning },
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

            {/* Indicador de peligro (navegación con riesgo integrado) */}
            {latestResult?.has_danger && (
              <View style={styles.riskIndicatorContainer}>
                <View
                  style={[
                    styles.riskIndicator,
                    {
                      backgroundColor: latestResult.priority === 'critical' ? '#EF4444' : '#F59E0B',
                    },
                  ]}
                >
                  <Ionicons
                    name="warning"
                    size={48}
                    color="white"
                  />
                  <Text style={styles.riskIndicatorText}>
                    {latestResult.priority === 'critical' ? 'PELIGRO' : 'PRECAUCIÓN'}
                  </Text>
                </View>
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
              ttsManager.speak('Detenido.', TtsPriority.HIGH);
            }}
            accessibilityLabel="Detener"
            accessibilityRole="button"
          >
            <Ionicons name="stop" size={28} color={COLORS.text} />
            <Text style={styles.stopRealtimeText}>Detener</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // Vista de cámara (Exploración y Lectura)
  const renderCamera = () => (
    <View style={styles.cameraContainer}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back">
        <View style={styles.cameraOverlay}>
          <View style={styles.cameraFrame} />
        </View>
      </CameraView>

      <View style={styles.cameraControls}>
        <TouchableOpacity
          style={styles.cameraButton}
          onPress={() => {
            handleReset();
            ttsManager.speak('Cancelado.', TtsPriority.HIGH);
          }}
          accessibilityLabel="Cancelar y volver"
          accessibilityRole="button"
        >
          <Ionicons name="close" size={32} color={COLORS.text} />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.captureButton}
          onPress={handleCapture}
          accessibilityLabel="Capturar foto"
          accessibilityRole="button"
        >
          <View style={styles.captureButtonInner} />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.cameraButton}
          onPress={handlePickImage}
          accessibilityLabel="Subir imagen desde galería"
          accessibilityRole="button"
        >
          <Ionicons name="images" size={32} color={COLORS.text} />
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
      <ActivityIndicator size="large" color={COLORS.primary} style={styles.loader} />
      <Text style={styles.processingText}>Analizando imagen...</Text>
    </View>
  );

  // Pantalla de resultados
  const renderResults = () => (
    <ScrollView style={styles.resultsContainer} contentContainerStyle={styles.resultsContent}>
      {capturedImage && (
        <Image source={{ uri: capturedImage }} style={styles.resultImage} />
      )}

      <View style={styles.resultCard}>
        <Text style={styles.resultTitle}>
          {MODE_CONFIG[analysisMode].label}
        </Text>

        {/* Resultado de Navegación */}
        {navResult && (
          <View>
            <Text style={styles.resultDescription}>{navResult.instruction}</Text>
            {navResult.path_clear && (
              <View style={[styles.statusBadge, { backgroundColor: COLORS.success + '20' }]}>
                <Text style={[styles.statusBadgeText, { color: COLORS.success }]}>
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
            <View style={[styles.statusBadge, { backgroundColor: COLORS.primary + '20' }]}>
              <Ionicons name="document-text" size={16} color={COLORS.primary} />
              <Text style={[styles.statusBadgeText, { color: COLORS.primary }]}>
                {smartResult.document_type_label}
              </Text>
            </View>

            {/* Narrativa principal */}
            <Text style={styles.resultDescription}>{smartResult.narrative}</Text>

            {/* Totales destacados (si hay) */}
            {smartResult.extracted_fields.totals.length > 0 && (
              <View style={styles.totalsHighlight}>
                <Ionicons name="cash" size={16} color={COLORS.warning} />
                <Text style={styles.totalsText}>
                  {smartResult.extracted_fields.totals.join(' | ')}
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

            {/* Indicador de calidad de imagen */}
            {smartResult.image_quality && smartResult.image_quality.issues.length > 0 && (
              <View style={[
                styles.totalsHighlight,
                {
                  backgroundColor: smartResult.image_quality.is_acceptable
                    ? COLORS.warning + '20'
                    : COLORS.error + '20',
                  marginTop: 8,
                },
              ]}>
                <Ionicons
                  name={smartResult.image_quality.is_acceptable ? 'alert-circle' : 'warning'}
                  size={16}
                  color={smartResult.image_quality.is_acceptable ? COLORS.warning : COLORS.error}
                />
                <Text style={[
                  styles.totalsText,
                  {
                    color: smartResult.image_quality.is_acceptable ? COLORS.warning : COLORS.error,
                  },
                ]}>
                  {smartResult.image_quality.issues.map(i => i.message).join(' ')}
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
          icon={<Ionicons name="refresh" size={20} color={COLORS.primary} />}
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
              color={COLORS.background}
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
      <Ionicons name="alert-circle" size={64} color={COLORS.error} />
      <Text style={styles.errorTitle}>Error</Text>
      <Text style={styles.errorMessage}>{error}</Text>
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
    <View style={styles.container}>
      <StatusBar style="light" />
      {renderContent()}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  // Home
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
    color: COLORS.text,
    marginTop: 12,
  },
  subtitle: {
    fontSize: 18,
    color: COLORS.textSecondary,
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
    color: COLORS.textSecondary,
    fontSize: 14,
  },
  modeSelector: {
    marginBottom: 32,
  },
  modeLabel: {
    color: COLORS.textSecondary,
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
    borderColor: COLORS.primary,
    gap: 6,
  },
  modeButtonActive: {
    backgroundColor: COLORS.primary,
  },
  modeButtonText: {
    color: COLORS.primary,
    fontSize: 14,
    fontWeight: '500',
  },
  modeButtonTextActive: {
    color: COLORS.background,
  },
  readingModeSelector: {
    marginBottom: 16,
  },
  readingModeLabel: {
    color: COLORS.textSecondary,
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
    borderColor: COLORS.primary,
    gap: 4,
  },
  readingModeButtonActive: {
    backgroundColor: COLORS.primary,
  },
  readingModeButtonText: {
    color: COLORS.primary,
    fontSize: 13,
    fontWeight: '500',
  },
  readingModeButtonTextActive: {
    color: COLORS.background,
  },
  mainButtons: {
    gap: 16,
  },
  mainButton: {
    width: '100%',
  },
  secondaryButton: {
    width: '100%',
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
    color: COLORS.textSecondary,
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
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraFrame: {
    width: SCREEN_WIDTH - 48,
    height: SCREEN_WIDTH - 48,
    borderWidth: 2,
    borderColor: COLORS.primary,
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
    backgroundColor: COLORS.surface,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: COLORS.primary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: COLORS.text,
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
    color: COLORS.text,
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
    color: COLORS.text,
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
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '500',
  },
  realtimeObjectDistance: {
    color: COLORS.primary,
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
    backgroundColor: COLORS.error,
    paddingHorizontal: 40,
    paddingVertical: 16,
    borderRadius: 30,
    minHeight: 52,
    gap: 8,
  },
  stopRealtimeText: {
    color: COLORS.text,
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
    color: COLORS.text,
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
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 12,
  },
  resultDescription: {
    fontSize: 16,
    color: COLORS.text,
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
    backgroundColor: COLORS.secondary,
    borderRadius: 12,
    padding: 12,
    marginTop: 12,
  },
  textBoxLabel: {
    fontSize: 12,
    color: COLORS.textSecondary,
    marginBottom: 4,
  },
  textBoxContent: {
    fontSize: 14,
    color: COLORS.text,
  },
  objectCount: {
    fontSize: 14,
    color: COLORS.textSecondary,
    marginTop: 12,
  },
  totalsHighlight: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.warning + '15',
    borderRadius: 12,
    padding: 12,
    marginTop: 12,
    gap: 8,
  },
  totalsText: {
    flex: 1,
    fontSize: 15,
    fontWeight: '600',
    color: COLORS.warning,
  },
  confidence: {
    fontSize: 14,
    color: COLORS.textSecondary,
    marginTop: 8,
  },
  noResult: {
    fontSize: 16,
    color: COLORS.textSecondary,
    fontStyle: 'italic',
  },
  objectItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: COLORS.secondary,
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
    color: COLORS.text,
  },
  objectDistance: {
    fontSize: 12,
    color: COLORS.primary,
    marginTop: 2,
  },
  resultActions: {
    flexDirection: 'row',
    gap: 12,
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
    color: COLORS.primary,
  },
  historyHeader: {
    marginBottom: 16,
  },
  historySubtitle: {
    fontSize: 14,
    color: COLORS.textSecondary,
    marginTop: 4,
  },
  historyEmpty: {
    alignItems: 'center',
    paddingVertical: 48,
    gap: 12,
  },
  historyEmptyText: {
    fontSize: 16,
    color: COLORS.textSecondary,
    textAlign: 'center',
  },
  historyEmptyHint: {
    fontSize: 14,
    color: COLORS.textSecondary,
    textAlign: 'center',
    fontStyle: 'italic',
  },
  historyItem: {
    flexDirection: 'row',
    backgroundColor: COLORS.surface,
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    gap: 12,
  },
  historyItemIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.primary + '15',
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
    color: COLORS.primary,
  },
  historyItemDate: {
    fontSize: 12,
    color: COLORS.textSecondary,
  },
  historyItemSummary: {
    fontSize: 14,
    color: COLORS.text,
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
    color: COLORS.error,
    marginTop: 16,
    marginBottom: 8,
  },
  errorMessage: {
    fontSize: 16,
    color: COLORS.textSecondary,
    textAlign: 'center',
    marginBottom: 24,
  },
});
