/**
 * Pantalla principal de NAVIA
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
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';

import { Button } from '../components/Button';
import { AnimatedEye } from '../components/AnimatedEye';
import { COLORS, ANALYSIS_MODES, AnalysisMode } from '../constants/config';
import { analyzeScene, extractText, detectObjects, checkHealth } from '../services/api';
import { speak, stop, speakProcessing, speakError } from '../services/tts';
import { SceneDescriptionResponse, OCRResponse, ObjectDetectionResponse } from '../types/api';
import { useRealtimeDetection } from '../hooks/useRealtimeDetection';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

type AppState = 'home' | 'camera' | 'processing' | 'results' | 'error' | 'realtime';

export function HomeScreen() {
  // Estados principales
  const [appState, setAppState] = useState<AppState>('home');
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>(ANALYSIS_MODES.SCENE);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Estados de imagen y resultados
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [sceneResult, setSceneResult] = useState<SceneDescriptionResponse | null>(null);
  const [ocrResult, setOcrResult] = useState<OCRResponse | null>(null);
  const [objectsResult, setObjectsResult] = useState<ObjectDetectionResponse | null>(null);

  // Estados de TTS
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);

  // Cámara
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  // Detección en tiempo real
  const [realtimeActive, setRealtimeActive] = useState(false);
  const { wsStatus, latestResult } = useRealtimeDetection({
    cameraRef,
    enabled: realtimeActive,
    ttsEnabled,
  });

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

  // Solicitar permisos de cámara
  const handleOpenCamera = async () => {
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        Alert.alert('Permiso denegado', 'Se necesita acceso a la cámara para usar esta función.');
        return;
      }
    }

    if (analysisMode === ANALYSIS_MODES.REALTIME) {
      setAppState('realtime');
      setRealtimeActive(true);
      if (ttsEnabled) {
        speak('Modo tiempo real activado. Apunta la cámara para detectar objetos.');
      }
    } else {
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

  // Procesar imagen con el backend
  const processImage = async (imageUri: string) => {
    if (ttsEnabled) {
      await speakProcessing();
    }

    try {
      let result;

      switch (analysisMode) {
        case ANALYSIS_MODES.TEXT:
          result = await extractText(imageUri);
          setOcrResult(result);
          if (ttsEnabled) {
            if (result.has_text) {
              await speak(`Texto encontrado: ${result.text}`);
            } else {
              await speak('No se detectó texto en la imagen.');
            }
          }
          break;

        case ANALYSIS_MODES.OBJECTS:
          result = await detectObjects(imageUri);
          setObjectsResult(result);
          if (ttsEnabled) {
            await speak(result.summary);
          }
          break;

        case ANALYSIS_MODES.SCENE:
        default:
          result = await analyzeScene(imageUri);
          setSceneResult(result);
          if (ttsEnabled) {
            await speak(result.description);
          }
          break;
      }

      setAppState('results');
    } catch (error: any) {
      handleError(error.message || 'Error procesando la imagen');
    }
  };

  // Manejar errores
  const handleError = (message: string) => {
    setError(message);
    setAppState('error');
    if (ttsEnabled) {
      speakError(message);
    }
  };

  // Reiniciar app
  const handleReset = () => {
    stop();
    setRealtimeActive(false);
    setAppState('home');
    setCapturedImage(null);
    setSceneResult(null);
    setOcrResult(null);
    setObjectsResult(null);
    setError(null);
  };

  // Repetir resultado
  const handleRepeat = async () => {
    if (isSpeaking) {
      await stop();
      setIsSpeaking(false);
      return;
    }

    setIsSpeaking(true);
    try {
      if (sceneResult) {
        await speak(sceneResult.description);
      } else if (ocrResult?.has_text) {
        await speak(ocrResult.text);
      } else if (objectsResult) {
        await speak(objectsResult.summary);
      }
    } finally {
      setIsSpeaking(false);
    }
  };

  // Renderizar según el estado
  const renderContent = () => {
    switch (appState) {
      case 'camera':
        return renderCamera();
      case 'realtime':
        return renderRealtime();
      case 'processing':
        return renderProcessing();
      case 'results':
        return renderResults();
      case 'error':
        return renderError();
      default:
        return renderHome();
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
                isBackendConnected === null
                  ? COLORS.warning
                  : isBackendConnected
                  ? COLORS.success
                  : COLORS.error,
            },
          ]}
        />
        <Text style={styles.connectionText}>
          {isBackendConnected === null
            ? 'Conectando...'
            : isBackendConnected
            ? 'Servidor conectado'
            : 'Sin conexión al servidor'}
        </Text>
      </View>

      {/* Selector de modo - grid 2x2 */}
      <View style={styles.modeSelector}>
        <Text style={styles.modeLabel}>Modo de análisis:</Text>
        <View style={styles.modeButtons}>
          {Object.entries(ANALYSIS_MODES).map(([key, value]) => {
            const label = value === 'scene'
              ? 'Escena'
              : value === 'text'
              ? 'Texto'
              : value === 'realtime'
              ? 'En Vivo'
              : 'Objetos';
            return (
              <TouchableOpacity
                key={key}
                style={[
                  styles.modeButton,
                  analysisMode === value && styles.modeButtonActive,
                ]}
                onPress={() => setAnalysisMode(value)}
                accessibilityRole="radio"
                accessibilityState={{ selected: analysisMode === value }}
                accessibilityLabel={`Modo ${label}`}
              >
                <Ionicons
                  name={
                    value === 'scene'
                      ? 'eye'
                      : value === 'text'
                      ? 'document-text'
                      : value === 'realtime'
                      ? 'radio'
                      : 'cube'
                  }
                  size={20}
                  color={analysisMode === value ? COLORS.background : COLORS.primary}
                />
                <Text
                  style={[
                    styles.modeButtonText,
                    analysisMode === value && styles.modeButtonTextActive,
                  ]}
                >
                  {label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {/* Botones principales */}
      <View style={styles.mainButtons}>
        <Button
          title="Abrir Cámara"
          onPress={handleOpenCamera}
          size="xl"
          disabled={isBackendConnected === false}
          icon={<Ionicons name="camera" size={28} color={COLORS.background} />}
          style={styles.mainButton}
        />

        <Button
          title="Subir Imagen"
          onPress={handlePickImage}
          variant="outline"
          size="large"
          disabled={isBackendConnected === false}
          icon={<Ionicons name="image" size={24} color={COLORS.primary} />}
          style={styles.secondaryButton}
        />
      </View>

      {/* Toggle TTS */}
      <TouchableOpacity
        style={styles.ttsToggle}
        onPress={() => setTtsEnabled(!ttsEnabled)}
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

  // Vista de tiempo real
  const renderRealtime = () => (
    <View style={styles.cameraContainer}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
      >
        {/* Overlay con resultados de detección */}
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
                {wsStatus === 'connected' ? 'En vivo' : 'Conectando...'}
              </Text>
            </View>
            {latestResult && (
              <Text style={styles.realtimeStatusText}>
                {latestResult.processing_time_ms}ms
              </Text>
            )}
          </View>

          {/* Resumen de detección en la parte inferior */}
          <View style={styles.realtimeSummaryContainer}>
            <View style={styles.realtimeSummary}>
              <Text style={styles.realtimeSummaryText}>
                {latestResult?.summary || 'Apunta la cámara para detectar...'}
              </Text>
              {latestResult && latestResult.objects.length > 0 && (
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
          onPress={handleReset}
          accessibilityLabel="Detener detección en tiempo real"
          accessibilityRole="button"
        >
          <Ionicons name="stop" size={28} color={COLORS.text} />
          <Text style={styles.stopRealtimeText}>Detener</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  // Vista de cámara
  const renderCamera = () => (
    <View style={styles.cameraContainer}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
      >
        <View style={styles.cameraOverlay}>
          <View style={styles.cameraFrame} />
        </View>
      </CameraView>

      <View style={styles.cameraControls}>
        <TouchableOpacity
          style={styles.cameraButton}
          onPress={handleReset}
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
        <Text style={styles.resultTitle}>Resultado</Text>

        {sceneResult && (
          <View>
            <Text style={styles.resultDescription}>{sceneResult.description}</Text>
            {sceneResult.has_text && (
              <View style={styles.textBox}>
                <Text style={styles.textBoxLabel}>Texto detectado:</Text>
                <Text style={styles.textBoxContent}>{sceneResult.detected_text}</Text>
              </View>
            )}
            {sceneResult.object_count > 0 && (
              <Text style={styles.objectCount}>
                {sceneResult.object_count} objeto(s) detectado(s)
              </Text>
            )}
          </View>
        )}

        {ocrResult && (
          <View>
            {ocrResult.has_text ? (
              <>
                <Text style={styles.resultDescription}>{ocrResult.text}</Text>
                <Text style={styles.confidence}>
                  {ocrResult.word_count} palabras • {ocrResult.confidence?.toFixed(0)}% confianza
                </Text>
              </>
            ) : (
              <Text style={styles.noResult}>No se detectó texto en la imagen.</Text>
            )}
          </View>
        )}

        {objectsResult && (
          <View>
            <Text style={styles.resultDescription}>{objectsResult.summary}</Text>
            {objectsResult.objects.map((obj, idx) => {
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
                  <Text style={styles.objectConfidence}>
                    {(obj.confidence * 100).toFixed(0)}%
                  </Text>
                </View>
              );
            })}
          </View>
        )}
      </View>

      <View style={styles.resultActions}>
        <Button
          title="Nueva Imagen"
          onPress={handleReset}
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
      <Button title="Intentar de nuevo" onPress={handleReset} size="large" />
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
  objectConfidence: {
    fontSize: 14,
    color: COLORS.textSecondary,
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
