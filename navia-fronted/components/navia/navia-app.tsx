"use client"

import React, { useState, useRef, useCallback, useEffect } from "react"
import { Camera, Upload, Volume2, VolumeX, Loader2, Eye, FileText, Compass, AlertTriangle, RefreshCw, X, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import { AnimatedEye } from "@/components/navia/AnimatedEye"
import {
  analyzeNavigation,
  analyzeExploration,
  analyzeReading,
  analyzeRisk,
  checkHealth,
  type NaviaMode,
  type NavigationResponse,
  type ExplorationResponse,
  type OCRResponse,
  type RiskResponse,
} from "@/lib/api"
import { useRealtimeDetection } from "@/hooks/useRealtimeDetection"
import { ttsManager, TtsPriority } from "@/lib/ttsManager"

// Estados posibles de la aplicación
type AppState = "idle" | "camera" | "capturing" | "processing" | "results" | "error" | "realtime"

// Configuración de modos
const MODE_CONFIG: Record<NaviaMode, { label: string; Icon: typeof Compass; description: string }> = {
  navegacion: { label: "Navegación", Icon: Compass, description: "Instrucciones de navegación en tiempo real" },
  exploracion: { label: "Exploración", Icon: Eye, description: "Descripción estructurada del entorno" },
  lectura: { label: "Lectura", Icon: FileText, description: "Lectura de texto y documentos" },
  riesgo: { label: "Riesgo", Icon: AlertTriangle, description: "Alertas de peligro en tiempo real" },
}

// Modos que usan WebSocket (tiempo real con cámara)
const REALTIME_MODES: NaviaMode[] = ['navegacion', 'riesgo']

export function NaviaApp() {
  // Estados principales
  const [appState, setAppState] = useState<AppState>("idle")
  const [analysisMode, setAnalysisMode] = useState<NaviaMode>("navegacion")
  const [error, setError] = useState<string | null>(null)
  const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null)

  // Estados de resultados
  const [navResult, setNavResult] = useState<NavigationResponse | null>(null)
  const [explorationResult, setExplorationResult] = useState<ExplorationResponse | null>(null)
  const [ocrResult, setOcrResult] = useState<OCRResponse | null>(null)
  const [riskResult, setRiskResult] = useState<RiskResponse | null>(null)

  // Estados de TTS (Text-to-Speech)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [ttsEnabled, setTtsEnabled] = useState(true)

  // Estados de imagen
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [processingProgress, setProcessingProgress] = useState(0)

  // Referencias
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // Detección en tiempo real
  const [realtimeActive, setRealtimeActive] = useState(false)
  const { wsStatus, latestResult } = useRealtimeDetection({
    videoRef,
    canvasRef,
    enabled: realtimeActive,
    ttsEnabled,
    mode: analysisMode,
  })

  const isRealtimeMode = REALTIME_MODES.includes(analysisMode)

  // Sincronizar estado de isSpeaking con ttsManager
  useEffect(() => {
    const interval = setInterval(() => {
      setIsSpeaking(ttsManager.isSpeakingNow())
    }, 200)
    return () => clearInterval(interval)
  }, [])

  // Sincronizar ttsEnabled con ttsManager
  useEffect(() => {
    ttsManager.setEnabled(ttsEnabled)
  }, [ttsEnabled])

  // Verificar conexión con el backend al iniciar + bienvenida
  useEffect(() => {
    const checkBackend = async () => {
      try {
        await checkHealth()
        setIsBackendConnected(true)
        ttsManager.speak("Bienvenido a NAVIA. Selecciona un modo y captura una imagen.", TtsPriority.HIGH)
      } catch {
        setIsBackendConnected(false)
        ttsManager.speak("Sin conexión al servidor. Verifica que el backend esté ejecutándose.", TtsPriority.HIGH)
      }
    }
    checkBackend()
  }, [])

  // Detener TTS
  const stopSpeaking = useCallback(() => {
    ttsManager.stop()
  }, [])

  // Iniciar cámara
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 1280, height: 720 }
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream

        if (isRealtimeMode) {
          setAppState("realtime")
          setRealtimeActive(true)
          const modeLabel = MODE_CONFIG[analysisMode].label
          ttsManager.speak(`Modo ${modeLabel} activado.`, TtsPriority.LOW)
        } else {
          setAppState("camera")
          ttsManager.speak("Cámara activada. Toca el botón central para capturar.", TtsPriority.LOW)
        }
      }
    } catch {
      setError("No se pudo acceder a la cámara. Por favor, permite el acceso.")
      setAppState("error")
      ttsManager.speak("Permiso de cámara denegado.", TtsPriority.INTERRUPT)
    }
  }

  // Detener cámara
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }, [])

  // Capturar imagen de la cámara
  const captureImage = async () => {
    if (!videoRef.current || !canvasRef.current) return

    setAppState("capturing")
    ttsManager.speak("Foto capturada.", TtsPriority.LOW)

    const video = videoRef.current
    const canvas = canvasRef.current

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.drawImage(video, 0, 0)

    const imageDataUrl = canvas.toDataURL("image/jpeg", 0.8)
    setCapturedImage(imageDataUrl)

    stopCamera()

    await processImage(imageDataUrl)
  }

  // Manejar subida de archivo
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = async (e) => {
      const imageDataUrl = e.target?.result as string
      setCapturedImage(imageDataUrl)
      await processImage(imageDataUrl)
    }
    reader.readAsDataURL(file)
  }

  // Procesar imagen con el backend
  const processImage = async (imageDataUrl: string) => {
    setAppState("processing")
    setProcessingProgress(10)
    ttsManager.speak("Procesando imagen, por favor espera.", TtsPriority.HIGH)

    try {
      // Convertir data URL a File
      const response = await fetch(imageDataUrl)
      const blob = await response.blob()
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" })

      setProcessingProgress(30)

      switch (analysisMode) {
        case "navegacion": {
          const result = await analyzeNavigation(file)
          setNavResult(result)
          setProcessingProgress(100)
          setAppState("results")
          ttsManager.speakFromBackend(result.instruction, TtsPriority.HIGH)
          break
        }

        case "exploracion": {
          setProcessingProgress(50)
          const result = await analyzeExploration(file)
          setExplorationResult(result)
          setProcessingProgress(100)
          setAppState("results")
          ttsManager.speakFromBackend(result.description, TtsPriority.HIGH)
          break
        }

        case "lectura": {
          const result = await analyzeReading(file)
          setOcrResult(result)
          setProcessingProgress(100)
          setAppState("results")
          if (result.has_text) {
            ttsManager.speakFromBackend(`Texto encontrado: ${result.text}`, TtsPriority.HIGH)
          } else {
            ttsManager.speak("No se detectó texto en la imagen.", TtsPriority.HIGH)
          }
          break
        }

        case "riesgo": {
          const result = await analyzeRisk(file)
          setRiskResult(result)
          setProcessingProgress(100)
          setAppState("results")
          ttsManager.speakFromBackend(result.alert_text, TtsPriority.HIGH)
          break
        }
      }

    } catch (err) {
      console.error("Error processing image:", err)
      setError(err instanceof Error ? err.message : "Error procesando la imagen")
      setAppState("error")
      ttsManager.speak("Ocurrió un error al procesar la imagen.", TtsPriority.INTERRUPT)
    }
  }

  // Reiniciar aplicación
  const reset = () => {
    setRealtimeActive(false)
    stopCamera()
    stopSpeaking()
    setAppState("idle")
    setCapturedImage(null)
    setNavResult(null)
    setExplorationResult(null)
    setOcrResult(null)
    setRiskResult(null)
    setError(null)
    setProcessingProgress(0)
  }

  // Repetir último resultado
  const repeatResult = () => {
    if (navResult) {
      ttsManager.speakFromBackend(navResult.instruction, TtsPriority.HIGH)
    } else if (explorationResult) {
      ttsManager.speakFromBackend(explorationResult.description, TtsPriority.HIGH)
    } else if (ocrResult && ocrResult.has_text) {
      ttsManager.speakFromBackend(ocrResult.text, TtsPriority.HIGH)
    } else if (riskResult) {
      ttsManager.speakFromBackend(riskResult.alert_text, TtsPriority.HIGH)
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <header className="p-4 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-3">
          <AnimatedEye className="h-8 w-8 text-primary" />
          <h1 className="text-2xl font-bold">NAVIA</h1>
        </div>
        <div className="flex items-center gap-2">
          {/* Indicador de conexión */}
          <div className={cn(
            "w-3 h-3 rounded-full",
            isBackendConnected === null && "bg-yellow-500 animate-pulse",
            isBackendConnected === true && "bg-green-500",
            isBackendConnected === false && "bg-red-500"
          )} />

          {/* Botón TTS */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => {
              const newValue = !ttsEnabled
              // Temporalmente habilitamos para decir el estado
              if (!newValue) {
                ttsManager.speak("Voz desactivada", TtsPriority.INTERRUPT)
                // Desactivar después de que termine de hablar
                setTimeout(() => setTtsEnabled(false), 1500)
              } else {
                setTtsEnabled(true)
                // Pequeño delay para que se active antes de hablar
                setTimeout(() => ttsManager.speak("Voz activada", TtsPriority.INTERRUPT), 100)
              }
            }}
            aria-label={ttsEnabled ? "Desactivar voz" : "Activar voz"}
          >
            {ttsEnabled ? <Volume2 className="h-5 w-5" /> : <VolumeX className="h-5 w-5" />}
          </Button>
        </div>
      </header>

      {/* Contenido principal */}
      <main className="flex-1 p-4 flex flex-col items-center justify-center">

        {/* Estado: Idle - Pantalla inicial */}
        {appState === "idle" && (
          <div className="w-full max-w-md space-y-6">
            <Card>
              <CardHeader className="text-center">
                <CardTitle>Asistente Visual</CardTitle>
                <CardDescription>
                  Selecciona un modo y captura una imagen para recibir asistencia
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                {/* Selector de modo - grid 2x2 */}
                <div
                  className="grid grid-cols-2 gap-2"
                  role="radiogroup"
                  aria-label="Modo de análisis"
                >
                  {(Object.entries(MODE_CONFIG) as [NaviaMode, typeof MODE_CONFIG[NaviaMode]][]).map(([mode, config]) => (
                    <Button
                      key={mode}
                      variant={analysisMode === mode ? "default" : "outline"}
                      onClick={() => {
                        setAnalysisMode(mode)
                        ttsManager.speak(`Modo ${config.label}`, TtsPriority.LOW)
                      }}
                      role="radio"
                      aria-checked={analysisMode === mode}
                      className="h-12 text-sm px-3 w-full"
                    >
                      <config.Icon className="h-4 w-4 shrink-0 mr-1.5" />
                      {config.label}
                    </Button>
                  ))}
                </div>

                {/* Botones principales */}
                <div className="flex flex-col gap-3">
                  <Button
                    size="xl"
                    onClick={startCamera}
                    disabled={isBackendConnected === false}
                    className="w-full min-h-[56px]"
                    aria-label="Abrir cámara"
                  >
                    <Camera className="h-6 w-6 mr-2 shrink-0" />
                    Abrir Cámara
                  </Button>

                  {/* Subir imagen solo para modos no-realtime */}
                  {!isRealtimeMode && (
                    <Button
                      variant="outline"
                      size="lg"
                      onClick={() => {
                        ttsManager.speak("Abriendo galería", TtsPriority.LOW)
                        fileInputRef.current?.click()
                      }}
                      disabled={isBackendConnected === false}
                      className="w-full min-h-[48px]"
                      aria-label="Subir imagen desde galería"
                    >
                      <Upload className="h-5 w-5 mr-2 shrink-0" />
                      Subir Imagen
                    </Button>
                  )}

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </div>

                {/* Advertencia si no hay conexión */}
                {isBackendConnected === false && (
                  <p className="text-destructive text-sm text-center">
                    No se puede conectar con el servidor. Verifica que el backend esté ejecutándose.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {/* Estado: Camera - Vista de cámara */}
        {appState === "camera" && (
          <div className="w-full max-w-md space-y-4">
            <div className="relative rounded-lg overflow-hidden bg-black aspect-[4/3]">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              {/* Overlay con guías */}
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute inset-4 border-2 border-primary/50 rounded-lg" />
              </div>
            </div>

            <div className="flex gap-3 justify-center items-center">
              <Button
                variant="outline"
                size="icon-lg"
                onClick={() => {
                  stopCamera()
                  setAppState("idle")
                  ttsManager.speak("Cancelado", TtsPriority.LOW)
                }}
                aria-label="Cancelar y volver"
              >
                <X className="h-6 w-6" />
              </Button>

              <Button
                size="icon-xl"
                onClick={captureImage}
                className="animate-pulse-glow"
                aria-label="Capturar foto"
              >
                <Camera className="h-8 w-8" />
              </Button>

              <Button
                variant="outline"
                size="icon-lg"
                onClick={() => {
                  ttsManager.speak("Abriendo galería", TtsPriority.LOW)
                  fileInputRef.current?.click()
                }}
                aria-label="Subir imagen desde galería"
              >
                <Upload className="h-6 w-6" />
              </Button>
            </div>
          </div>
        )}

        {/* Estado: Processing - Procesando imagen */}
        {(appState === "capturing" || appState === "processing") && (
          <div className="w-full max-w-md space-y-6 text-center">
            {capturedImage && (
              <div className="rounded-lg overflow-hidden">
                <img src={capturedImage} alt="Imagen capturada" className="w-full" />
              </div>
            )}

            <div className="space-y-3">
              <Loader2 className="h-12 w-12 animate-spin mx-auto text-primary" />
              <p className="text-lg">Analizando imagen...</p>
              <Progress value={processingProgress} className="w-full" />
            </div>
          </div>
        )}

        {/* Estado: Results - Mostrando resultados */}
        {appState === "results" && (
          <div className="w-full max-w-md space-y-4">
            {capturedImage && (
              <div className="rounded-lg overflow-hidden">
                <img src={capturedImage} alt="Imagen analizada" className="w-full" />
              </div>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {React.createElement(MODE_CONFIG[analysisMode].Icon, { className: "h-5 w-5" })}
                  {MODE_CONFIG[analysisMode].label}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Resultado de Navegación */}
                {navResult && (
                  <div className="space-y-3">
                    <p className="text-lg leading-relaxed font-medium">{navResult.instruction}</p>

                    <div className="flex items-center gap-2 text-sm">
                      <div className={cn(
                        "w-3 h-3 rounded-full",
                        navResult.path_clear ? "bg-green-500" : "bg-amber-500"
                      )} />
                      <span className="text-muted-foreground">
                        {navResult.path_clear ? "Camino libre" : "Obstáculos detectados"}
                      </span>
                    </div>

                    {navResult.obstacles.length > 0 && (
                      <ul className="space-y-2">
                        {navResult.obstacles.map((obj, idx) => {
                          const zoneColor =
                            obj.distance_zone === 'muy_cerca' ? 'bg-red-500' :
                            obj.distance_zone === 'cerca' ? 'bg-amber-500' : 'bg-green-500'
                          const textColor =
                            obj.distance_zone === 'muy_cerca' ? 'text-red-500' :
                            obj.distance_zone === 'cerca' ? 'text-amber-500' : 'text-green-500'
                          return (
                            <li key={idx} className="flex justify-between items-center p-2 bg-secondary rounded">
                              <div className="flex items-start gap-2">
                                <div className={cn("w-2 h-2 rounded-full mt-1.5 shrink-0", zoneColor)} />
                                <div>
                                  <span>{obj.name_es}</span>
                                  {obj.distance_estimate && (
                                    <p className={cn("text-xs mt-0.5", textColor)}>{obj.distance_estimate}</p>
                                  )}
                                </div>
                              </div>
                              <span className="text-sm text-muted-foreground">
                                {(obj.confidence * 100).toFixed(0)}%
                              </span>
                            </li>
                          )
                        })}
                      </ul>
                    )}
                  </div>
                )}

                {/* Resultado de Exploración */}
                {explorationResult && (
                  <div className="space-y-3">
                    <p className="text-lg leading-relaxed">{explorationResult.description}</p>

                    {explorationResult.has_text && (
                      <div className="p-3 bg-secondary rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Texto detectado:</p>
                        <p>{explorationResult.detected_text}</p>
                      </div>
                    )}

                    {explorationResult.object_count > 0 && (
                      <p className="text-sm text-muted-foreground">
                        {explorationResult.object_count} objeto(s) detectado(s)
                      </p>
                    )}
                  </div>
                )}

                {/* Resultado de Lectura (OCR) */}
                {ocrResult && (
                  <div className="space-y-3">
                    {ocrResult.has_text ? (
                      <>
                        <p className="text-lg leading-relaxed">{ocrResult.text}</p>
                        <p className="text-sm text-muted-foreground">
                          {ocrResult.word_count} palabras • {ocrResult.confidence?.toFixed(0)}% confianza
                        </p>
                      </>
                    ) : (
                      <p className="text-muted-foreground">No se detectó texto en la imagen.</p>
                    )}
                  </div>
                )}

                {/* Resultado de Riesgo */}
                {riskResult && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        "w-4 h-4 rounded-full",
                        riskResult.has_danger
                          ? riskResult.priority === 'critical' ? "bg-red-500" : "bg-amber-500"
                          : "bg-green-500"
                      )} />
                      <p className="text-lg font-medium">{riskResult.alert_text}</p>
                    </div>

                    {riskResult.dangers.length > 0 && (
                      <ul className="space-y-2">
                        {riskResult.dangers.map((danger, idx) => (
                          <li key={idx} className="flex justify-between items-center p-2 bg-secondary rounded">
                            <div className="flex items-start gap-2">
                              <div className={cn(
                                "w-2 h-2 rounded-full mt-1.5 shrink-0",
                                danger.danger_level === 'critical' ? 'bg-red-500' : 'bg-amber-500'
                              )} />
                              <div>
                                <span>{danger.object_name}</span>
                                <p className="text-xs text-muted-foreground mt-0.5">
                                  {danger.distance_zone} • {danger.position}
                                </p>
                              </div>
                            </div>
                            <span className={cn(
                              "text-xs font-medium px-2 py-0.5 rounded",
                              danger.danger_level === 'critical' ? 'bg-red-500/20 text-red-500' : 'bg-amber-500/20 text-amber-500'
                            )}>
                              {danger.danger_level}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Botones de acción */}
            <div className="grid grid-cols-2 gap-3">
              <Button
                variant="outline"
                size="lg"
                onClick={() => {
                  reset()
                  ttsManager.speak("Nueva captura", TtsPriority.LOW)
                }}
                className="min-h-[48px]"
                aria-label="Tomar nueva imagen"
              >
                <RefreshCw className="h-5 w-5 mr-2 shrink-0" />
                Nueva Imagen
              </Button>

              <Button
                size="lg"
                onClick={repeatResult}
                disabled={isSpeaking}
                className="min-h-[48px]"
                aria-label={isSpeaking ? "Hablando resultado" : "Repetir resultado en voz"}
              >
                <Volume2 className="h-5 w-5 mr-2 shrink-0" />
                {isSpeaking ? "Hablando..." : "Repetir"}
              </Button>
            </div>
          </div>
        )}

        {/* Estado: Realtime - Detección en tiempo real */}
        {appState === "realtime" && (
          <div className="w-full max-w-md space-y-4">
            <div className="relative rounded-lg overflow-hidden bg-black aspect-[4/3]">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              {/* Overlay con indicador de estado */}
              <div className="absolute top-0 left-0 right-0 flex items-center justify-between p-3 bg-black/50">
                <div className="flex items-center gap-2">
                  <div className={cn(
                    "w-3 h-3 rounded-full",
                    wsStatus === "connected" ? "bg-green-500 animate-pulse" : "bg-yellow-500"
                  )} />
                  <span className="text-white text-sm font-medium">
                    {MODE_CONFIG[analysisMode].label} - {wsStatus === "connected" ? "En vivo" : "Conectando..."}
                  </span>
                </div>
                {latestResult && (
                  <span className="text-white/70 text-xs">
                    {latestResult.processing_time_ms}ms
                    {latestResult.tracked_count != null && ` | ${latestResult.tracked_count} obj`}
                  </span>
                )}
              </div>

              {/* Indicador de riesgo para modo riesgo */}
              {analysisMode === "riesgo" && (
                <div className="absolute top-14 right-3">
                  <div className={cn(
                    "w-8 h-8 rounded-full border-2 border-white",
                    latestResult?.has_danger
                      ? latestResult?.priority === 'critical' ? "bg-red-500 animate-pulse" : "bg-amber-500"
                      : "bg-green-500"
                  )} />
                </div>
              )}

              {/* Resumen de detección */}
              <div className="absolute bottom-0 left-0 right-0 p-3 bg-black/70">
                <p className="text-white text-lg font-medium">
                  {latestResult?.summary || (analysisMode === "riesgo" ? "Monitoreando peligros..." : "Apunta la cámara para navegar...")}
                </p>
                {analysisMode === "navegacion" && latestResult && latestResult.object_count > 0 && (
                  <p className="text-primary text-sm mt-1">
                    {latestResult.object_count} objeto(s) detectado(s)
                  </p>
                )}
              </div>
            </div>

            {/* Lista de objetos detectados (solo navegación) */}
            {analysisMode === "navegacion" && latestResult && latestResult.objects.length > 0 && (
              <Card>
                <CardContent className="py-3">
                  <ul className="space-y-1">
                    {[...latestResult.objects]
                      .sort((a, b) => {
                        const order: Record<string, number> = { high: 0, medium: 1, low: 2 }
                        return (order[a.priority ?? 'low'] ?? 2) - (order[b.priority ?? 'low'] ?? 2)
                      })
                      .slice(0, 5).map((obj, idx) => {
                      const zoneColor =
                        obj.distance_zone === 'muy_cerca' ? 'bg-red-500' :
                        obj.distance_zone === 'cerca' ? 'bg-amber-500' : 'bg-green-500'
                      const textColor =
                        obj.distance_zone === 'muy_cerca' ? 'text-red-500' :
                        obj.distance_zone === 'cerca' ? 'text-amber-500' : 'text-green-500'
                      return (
                        <li key={idx} className="flex justify-between items-center p-2 bg-secondary rounded text-sm">
                          <div className="flex items-start gap-2">
                            <div className={cn("w-2 h-2 rounded-full mt-1 shrink-0", zoneColor)} />
                            <div>
                              <span className={obj.priority === 'high' ? 'font-bold' : ''}>{obj.name_es}</span>
                              {obj.distance_estimate && (
                                <p className={cn("text-xs mt-0.5", textColor)}>{obj.distance_estimate}</p>
                              )}
                            </div>
                          </div>
                          <span className="text-muted-foreground">
                            {(obj.confidence * 100).toFixed(0)}%
                          </span>
                        </li>
                      )
                    })}
                  </ul>
                </CardContent>
              </Card>
            )}

            {/* Botón para detener */}
            <div className="flex justify-center">
              <Button
                variant="destructive"
                size="lg"
                onClick={() => {
                  reset()
                  ttsManager.speak("Detenido", TtsPriority.LOW)
                }}
                className="w-full min-h-[48px]"
                aria-label="Detener detección en tiempo real"
              >
                <Square className="h-5 w-5 mr-2 shrink-0" />
                Detener
              </Button>
            </div>
          </div>
        )}

        {/* Estado: Error */}
        {appState === "error" && (
          <div className="w-full max-w-md">
            <Card className="border-destructive">
              <CardHeader>
                <CardTitle className="text-destructive">Error</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p>{error}</p>
                <Button onClick={() => { reset(); ttsManager.speak("Volviendo al inicio", TtsPriority.LOW) }} className="w-full min-h-[48px]" aria-label="Intentar de nuevo">
                  Intentar de nuevo
                </Button>
              </CardContent>
            </Card>
          </div>
        )}
      </main>

      {/* Canvas oculto para captura */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Footer */}
      <footer className="p-4 border-t border-border text-center text-sm text-muted-foreground">
        <p>NAVIA - Proyecto de Tesis USB</p>
      </footer>
    </div>
  )
}
