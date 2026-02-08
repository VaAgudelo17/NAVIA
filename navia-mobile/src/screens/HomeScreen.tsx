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
import { COLORS, ANALYSIS_MODES, AnalysisMode } from '../constants/config';
import { analyzeScene, extractText, detectObjects, checkHealth } from '../services/api';
import { speak, stop, speakProcessing, speakError } from '../services/tts';
import { SceneDescriptionResponse, OCRResponse, ObjectDetectionResponse } from '../types/api';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

type AppState = 'home' | 'camera' | 'processing' | 'results' | 'error';

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
    setAppState('camera');
    if (ttsEnabled) {
      speak('Cámara activada. Toca el botón central para capturar.');
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
        <Ionicons name="eye" size={48} color={COLORS.primary} />
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

      {/* Selector de modo */}
      <View style={styles.modeSelector}>
        <Text style={styles.modeLabel}>Modo de análisis:</Text>
        <View style={styles.modeButtons}>
          {Object.entries(ANALYSIS_MODES).map(([key, value]) => (
            <TouchableOpacity
              key={key}
              style={[
                styles.modeButton,
                analysisMode === value && styles.modeButtonActive,
              ]}
              onPress={() => setAnalysisMode(value)}
            >
              <Ionicons
                name={
                  value === 'scene'
                    ? 'eye'
                    : value === 'text'
                    ? 'document-text'
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
                {value === 'scene' ? 'Escena' : value === 'text' ? 'Texto' : 'Objetos'}
              </Text>
            </TouchableOpacity>
          ))}
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
        <TouchableOpacity style={styles.cameraButton} onPress={handleReset}>
          <Ionicons name="close" size={32} color={COLORS.text} />
        </TouchableOpacity>

        <TouchableOpacity style={styles.captureButton} onPress={handleCapture}>
          <View style={styles.captureButtonInner} />
        </TouchableOpacity>

        <TouchableOpacity style={styles.cameraButton} onPress={handlePickImage}>
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
            {objectsResult.objects.map((obj, idx) => (
              <View key={idx} style={styles.objectItem}>
                <Text style={styles.objectName}>{obj.name_es}</Text>
                <Text style={styles.objectConfidence}>
                  {(obj.confidence * 100).toFixed(0)}%
                </Text>
              </View>
            ))}
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
    justifyContent: 'center',
    gap: 8,
  },
  modeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
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
  objectName: {
    fontSize: 14,
    color: COLORS.text,
  },
  objectConfidence: {
    fontSize: 14,
    color: COLORS.textSecondary,
  },
  resultActions: {
    flexDirection: 'row',
    gap: 12,
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
