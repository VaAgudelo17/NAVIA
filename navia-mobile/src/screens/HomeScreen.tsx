/**
 * Pantalla principal de NAVIA
 *
 * 4 modos:
 * - Navegación: tiempo real, instrucciones cortas de navegación
 * - Exploración: foto, descripción estructurada del entorno
 * - Lectura: foto, OCR puro
 * - Riesgo: tiempo real, alertas de peligro
 */

import React, { useState, useEffect, useRef } from 'react';
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
  analyzeRisk,
  checkHealth,
} from '../services/api';
import { speak, stop, speakProcessing, speakError } from '../services/tts';
import {
  NavigationResponse,
  ExplorationResponse,
  OCRResponse,
  RiskResponse,
} from '../types/api';
import { useRealtimeDetection } from '../hooks/useRealtimeDetection';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

type AppState = 'home' | 'camera' | 'processing' | 'results' | 'error' | 'realtime';

// Configuración de cada modo
const MODE_CONFIG: Record<AnalysisMode, { label: string; icon: string; description: string }> = {
  navegacion: { label: 'Navegación', icon: 'compass', description: 'Detecta obstáculos' },
  exploracion: { label: 'Exploración', icon: 'eye', description: 'Describe el entorno' },
  lectura: { label: 'Lectura', icon: 'document-text', description: 'Lee textos' },
  riesgo: { label: 'Riesgo', icon: 'warning', description: 'Alerta de peligros' },
};

export function HomeScreen() {
  // Estados principales
  const [appState, setAppState] = useState<AppState>('home');
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>(ANALYSIS_MODES.NAVEGACION);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Estados de imagen y resultados
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [navResult, setNavResult] = useState<NavigationResponse | null>(null);
  const [explorationResult, setExplorationResult] = useState<ExplorationResponse | null>(null);
  const [ocrResult, setOcrResult] = useState<OCRResponse | null>(null);
  const [riskResult, setRiskResult] = useState<RiskResponse | null>(null);

  // Estados de TTS
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);

  // Cámara
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  // Detección en tiempo real (para Navegación y Riesgo)
  const [realtimeActive, setRealtimeActive] = useState(false);
  const { wsStatus, latestResult } = useRealtimeDetection({
    cameraRef,
    enabled: realtimeActive,
    ttsEnabled,
    mode: analysisMode,
  });

  const isRealtimeMode = REALTIME_MODES.includes(analysisMode);

  // Verificar conexión con el backend al iniciar
  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      await checkHealth();
      setIsBackendConnected(true);
    } catch {
      setIsBackendConnected(false);
    }
  };

  // Abrir cámara
  const handleOpenCamera = async () => {
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        Alert.alert('Permiso denegado', 'Se necesita acceso a la cámara.');
        return;
      }
    }

    if (isRealtimeMode) {
      // Navegación y Riesgo: entran directo a modo realtime
      setAppState('realtime');
      setRealtimeActive(true);
      const modeLabel = MODE_CONFIG[analysisMode].label;
      if (ttsEnabled) {
        speak(`Modo ${modeLabel} activado. Apunta la cámara.`);
      }
    } else {
      // Exploración y Lectura: capturan foto
      setAppState('camera');
      if (ttsEnabled) {
        speak('Cámara activada. Toca el botón central para capturar.');
      }
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
        await processImage(photo.uri);
      }
    } catch (error) {
      console.error('Error capturing photo:', error);
      handleError('No se pudo capturar la foto');
    }
  };

  // Seleccionar imagen de la galería
  const handlePickImage = async () => {
    if (ttsEnabled) speak('Abriendo galería');
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
    if (ttsEnabled) await speakProcessing();

    try {
      switch (analysisMode) {
        case 'navegacion': {
          const result = await analyzeNavigation(imageUri);
          setNavResult(result);
          if (ttsEnabled) await speak(result.instruction);
          break;
        }
        case 'exploracion': {
          const result = await analyzeExploration(imageUri);
          setExplorationResult(result);
          if (ttsEnabled) await speak(result.description);
          break;
        }
        case 'lectura': {
          const result = await analyzeReading(imageUri);
          setOcrResult(result);
          if (ttsEnabled) {
            await speak(result.has_text ? `Texto: ${result.text}` : 'No se detectó texto.');
          }
          break;
        }
        case 'riesgo': {
          const result = await analyzeRisk(imageUri);
          setRiskResult(result);
          if (ttsEnabled) {
            await speak(result.has_danger ? result.alert_text : 'Sin peligros detectados.');
          }
          break;
        }
      }
      setAppState('results');
    } catch (error: any) {
      handleError(error.message || 'Error procesando la imagen');
    }
  };

  const handleError = (message: string) => {
    setError(message);
    setAppState('error');
    if (ttsEnabled) speakError(message);
  };

  const handleReset = () => {
    stop();
    setRealtimeActive(false);
    setAppState('home');
    setCapturedImage(null);
    setNavResult(null);
    setExplorationResult(null);
    setOcrResult(null);
    setRiskResult(null);
    setError(null);
  };

  // Repetir resultado por TTS
  const handleRepeat = async () => {
    if (isSpeaking) {
      await stop();
      setIsSpeaking(false);
      return;
    }

    setIsSpeaking(true);
    try {
      if (navResult) await speak(navResult.instruction);
      else if (explorationResult) await speak(explorationResult.description);
      else if (ocrResult?.has_text) await speak(ocrResult.text);
      else if (riskResult) await speak(riskResult.has_danger ? riskResult.alert_text : 'Sin peligros.');
    } finally {
      setIsSpeaking(false);
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
                  if (ttsEnabled) speak(`Modo ${config.label}`);
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
          speak(newValue ? 'Voz activada' : 'Voz desactivada');
          setTtsEnabled(newValue);
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
    </View>
  );

  // Vista de tiempo real (Navegación y Riesgo)
  const renderRealtime = () => {
    const isRiskMode = analysisMode === 'riesgo';

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
                </Text>
              )}
            </View>

            {/* Indicador de peligro para modo Riesgo */}
            {isRiskMode && (
              <View style={styles.riskIndicatorContainer}>
                <View
                  style={[
                    styles.riskIndicator,
                    {
                      backgroundColor: latestResult?.has_danger
                        ? (latestResult.priority === 'critical' ? '#EF4444' : '#F59E0B')
                        : '#22C55E',
                    },
                  ]}
                >
                  <Ionicons
                    name={latestResult?.has_danger ? 'warning' : 'checkmark-circle'}
                    size={48}
                    color="white"
                  />
                  <Text style={styles.riskIndicatorText}>
                    {latestResult?.has_danger
                      ? (latestResult.priority === 'critical' ? 'PELIGRO' : 'PRECAUCIÓN')
                      : 'SEGURO'}
                  </Text>
                </View>
              </View>
            )}

            {/* Resumen de detección en la parte inferior */}
            <View style={styles.realtimeSummaryContainer}>
              <View style={styles.realtimeSummary}>
                <Text style={styles.realtimeSummaryText}>
                  {latestResult?.summary || (isRiskMode ? 'Monitoreando peligros...' : 'Analizando entorno...')}
                </Text>
                {/* Lista de objetos solo en modo Navegación */}
                {!isRiskMode && latestResult && latestResult.objects.length > 0 && (
                  <View style={styles.realtimeObjectList}>
                    {latestResult.objects.slice(0, 5).map((obj, idx) => {
                      const zoneColor =
                        obj.distance_zone === 'muy_cerca' ? '#EF4444' :
                        obj.distance_zone === 'cerca' ? '#F59E0B' : '#22C55E';
                      return (
                        <View key={idx} style={styles.realtimeObjectItem}>
                          <View style={styles.realtimeObjLeft}>
                            <View style={[styles.zoneDot, { backgroundColor: zoneColor }]} />
                            <Text style={styles.realtimeObjectName}>{obj.name_es}</Text>
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
              if (ttsEnabled) speak('Detenido');
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
            if (ttsEnabled) speak('Cancelado');
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

        {/* Resultado de Lectura */}
        {ocrResult && (
          <View>
            {ocrResult.has_text ? (
              <>
                <Text style={styles.resultDescription}>{ocrResult.text}</Text>
                <Text style={styles.confidence}>
                  {ocrResult.word_count} palabras
                  {ocrResult.confidence ? ` • ${ocrResult.confidence.toFixed(0)}% confianza` : ''}
                </Text>
              </>
            ) : (
              <Text style={styles.noResult}>No se detectó texto en la imagen.</Text>
            )}
          </View>
        )}

        {/* Resultado de Riesgo */}
        {riskResult && (
          <View>
            <View style={[
              styles.statusBadge,
              {
                backgroundColor: riskResult.has_danger
                  ? (riskResult.priority === 'critical' ? '#EF444420' : '#F59E0B20')
                  : '#22C55E20',
              },
            ]}>
              <Ionicons
                name={riskResult.has_danger ? 'warning' : 'checkmark-circle'}
                size={20}
                color={riskResult.has_danger
                  ? (riskResult.priority === 'critical' ? '#EF4444' : '#F59E0B')
                  : '#22C55E'}
              />
              <Text style={[
                styles.statusBadgeText,
                {
                  color: riskResult.has_danger
                    ? (riskResult.priority === 'critical' ? '#EF4444' : '#F59E0B')
                    : '#22C55E',
                },
              ]}>
                {riskResult.has_danger
                  ? (riskResult.priority === 'critical' ? 'PELIGRO' : 'PRECAUCIÓN')
                  : 'SEGURO'}
              </Text>
            </View>
            <Text style={styles.resultDescription}>
              {riskResult.has_danger ? riskResult.alert_text : 'No se detectaron peligros.'}
            </Text>
            {riskResult.dangers.length > 0 && (
              <View style={styles.obstacleList}>
                {riskResult.dangers.map((d, idx) => (
                  <View key={idx} style={styles.objectItem}>
                    <View style={styles.objectInfo}>
                      <Text style={styles.objectName}>{d.object_name}</Text>
                      <Text style={[styles.objectDistance, {
                        color: d.danger_level === 'critical' ? '#EF4444' : '#F59E0B',
                      }]}>
                        {d.distance_zone === 'muy_cerca' ? 'Muy cerca' : 'Cerca'} - {d.position}
                      </Text>
                    </View>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}
      </View>

      <View style={styles.resultActions}>
        <Button
          title="Nueva Imagen"
          onPress={() => {
            handleReset();
            if (ttsEnabled) speak('Nueva captura');
          }}
          variant="outline"
          size="large"
          icon={<Ionicons name="refresh" size={20} color={COLORS.primary} />}
          style={styles.resultActionButton}
        />
        <Button
          title={isSpeaking ? 'Detener' : 'Repetir'}
          onPress={handleRepeat}
          size="large"
          icon={
            <Ionicons
              name={isSpeaking ? 'stop' : 'volume-high'}
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
          if (ttsEnabled) speak('Volviendo al inicio');
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
