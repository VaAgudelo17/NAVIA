"use client"

import React, { useState, useRef, useCallback, useEffect } from "react"
import { Camera, Upload, Volume2, VolumeX, Loader2, Eye, FileText, Box, RefreshCw, X, Radio, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import {
  analyzeScene,
  extractText,
  detectObjects,
  checkHealth,
  type SceneDescriptionResponse,
  type OCRResponse,
  type ObjectDetectionResponse
} from "@/lib/api"
import { useRealtimeDetection } from "@/hooks/useRealtimeDetection"

// Estados posibles de la aplicación
type AppState = "idle" | "camera" | "capturing" | "processing" | "results" | "error" | "realtime"

// Modos de análisis
type AnalysisMode = "scene" | "text" | "objects" | "realtime"

export function NaviaApp() {
  // Estados principales
  const [appState, setAppState] = useState<AppState>("idle")
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("scene")
  const [error, setError] = useState<string | null>(null)
  const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null)

  // Estados de resultados
  const [sceneResult, setSceneResult] = useState<SceneDescriptionResponse | null>(null)
  const [ocrResult, setOcrResult] = useState<OCRResponse | null>(null)
  const [objectsResult, setObjectsResult] = useState<ObjectDetectionResponse | null>(null)

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
  })

  // Verificar conexión con el backend al iniciar
  useEffect(() => {
    const checkBackend = async () => {
      try {
        await checkHealth()
        setIsBackendConnected(true)
      } catch {
        setIsBackendConnected(false)
      }
    }
    checkBackend()
  }, [])

  // Función para hablar texto usando Web Speech API
  const speak = useCallback((text: string) => {
    if (!ttsEnabled || !text) return

    // Cancelar cualquier síntesis en curso
    window.speechSynthesis.cancel()

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = "es-ES"
    utterance.rate = 0.9
    utterance.pitch = 1

    utterance.onstart = () => setIsSpeaking(true)
    utterance.onend = () => setIsSpeaking(false)
    utterance.onerror = () => setIsSpeaking(false)

    window.speechSynthesis.speak(utterance)
  }, [ttsEnabled])

  // Detener TTS
  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel()
    setIsSpeaking(false)
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

        if (analysisMode === "realtime") {
          setAppState("realtime")
          setRealtimeActive(true)
          speak("Modo tiempo real activado. Apunta la cámara para detectar objetos.")
        } else {
          setAppState("camera")
          speak("Cámara activada. Toca el botón central para capturar.")
        }
      }
    } catch (err) {
      setError("No se pudo acceder a la cámara. Por favor, permite el acceso.")
      setAppState("error")
      speak("No se pudo acceder a la cámara.")
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

    // Procesar imagen
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
    speak("Procesando imagen, por favor espera.")

    try {
      // Convertir data URL a File
      const response = await fetch(imageDataUrl)
      const blob = await response.blob()
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" })

      setProcessingProgress(30)

      let result
      switch (analysisMode) {
        case "text":
          result = await extractText(file)
          setOcrResult(result)
          setProcessingProgress(100)
          setAppState("results")

          if (result.has_text) {
            speak(`Texto encontrado: ${result.text}`)
          } else {
            speak("No se detectó texto en la imagen.")
          }
          break

        case "objects":
          result = await detectObjects(file)
          setObjectsResult(result)
          setProcessingProgress(100)
          setAppState("results")
          speak(result.summary)
          break

        case "scene":
        default:
          setProcessingProgress(50)
          result = await analyzeScene(file)
          setSceneResult(result)
          setProcessingProgress(100)
          setAppState("results")
          speak(result.description)
          break
      }

    } catch (err) {
      console.error("Error processing image:", err)
      setError(err instanceof Error ? err.message : "Error procesando la imagen")
      setAppState("error")
      speak("Ocurrió un error al procesar la imagen.")
    }
  }

  // Reiniciar aplicación
  const reset = () => {
    setRealtimeActive(false)
    stopCamera()
    stopSpeaking()
    setAppState("idle")
    setCapturedImage(null)
    setSceneResult(null)
    setOcrResult(null)
    setObjectsResult(null)
    setError(null)
    setProcessingProgress(0)
  }

  // Repetir último resultado
  const repeatResult = () => {
    if (sceneResult) {
      speak(sceneResult.description)
    } else if (ocrResult && ocrResult.has_text) {
      speak(ocrResult.text)
    } else if (objectsResult) {
      speak(objectsResult.summary)
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <header className="p-4 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Eye className="h-8 w-8 text-primary" />
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
              setTtsEnabled(!ttsEnabled)
              if (ttsEnabled) stopSpeaking()
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
                  Captura una imagen para obtener una descripción de lo que hay en ella
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                {/* Selector de modo - grid 2x2 en móvil, 4 en desktop */}
                <div
                  className="grid grid-cols-2 gap-2"
                  role="radiogroup"
                  aria-label="Modo de análisis"
                >
                  {([
                    { mode: "scene" as AnalysisMode, label: "Escena", Icon: Eye },
                    { mode: "text" as AnalysisMode, label: "Texto", Icon: FileText },
                    { mode: "objects" as AnalysisMode, label: "Objetos", Icon: Box },
                    { mode: "realtime" as AnalysisMode, label: "En Vivo", Icon: Radio },
                  ]).map(({ mode, label, Icon }) => (
                    <Button
                      key={mode}
                      variant={analysisMode === mode ? "default" : "outline"}
                      onClick={() => setAnalysisMode(mode)}
                      role="radio"
                      aria-checked={analysisMode === mode}
                      className="h-12 text-sm px-3 w-full"
                    >
                      <Icon className="h-4 w-4 shrink-0 mr-1.5" />
                      {label}
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
                    aria-label="Abrir cámara para capturar imagen"
                  >
                    <Camera className="h-6 w-6 mr-2 shrink-0" />
                    Abrir Cámara
                  </Button>

                  <Button
                    variant="outline"
                    size="lg"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isBackendConnected === false}
                    className="w-full min-h-[48px]"
                    aria-label="Subir imagen desde galería"
                  >
                    <Upload className="h-5 w-5 mr-2 shrink-0" />
                    Subir Imagen
                  </Button>

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
                onClick={() => fileInputRef.current?.click()}
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
                  {analysisMode === "scene" && <Eye className="h-5 w-5" />}
                  {analysisMode === "text" && <FileText className="h-5 w-5" />}
                  {analysisMode === "objects" && <Box className="h-5 w-5" />}
                  Resultado
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Descripción de escena */}
                {sceneResult && (
                  <div className="space-y-3">
                    <p className="text-lg leading-relaxed">{sceneResult.description}</p>

                    {sceneResult.has_text && (
                      <div className="p-3 bg-secondary rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Texto detectado:</p>
                        <p>{sceneResult.detected_text}</p>
                      </div>
                    )}

                    {sceneResult.object_count > 0 && (
                      <p className="text-sm text-muted-foreground">
                        {sceneResult.object_count} objeto(s) detectado(s)
                      </p>
                    )}
                  </div>
                )}

                {/* Resultado de OCR */}
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

                {/* Resultado de detección de objetos */}
                {objectsResult && (
                  <div className="space-y-3">
                    <p className="text-lg">{objectsResult.summary}</p>

                    {objectsResult.objects.length > 0 && (
                      <ul className="space-y-2">
                        {objectsResult.objects.map((obj, idx) => {
                          const zoneColor =
                            obj.distance_zone === 'muy_cerca' ? 'bg-red-500' :
                            obj.distance_zone === 'cerca' ? 'bg-amber-500' : 'bg-green-500';
                          const textColor =
                            obj.distance_zone === 'muy_cerca' ? 'text-red-500' :
                            obj.distance_zone === 'cerca' ? 'text-amber-500' : 'text-green-500';
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
                          );
                        })}
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
                onClick={reset}
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
                    {wsStatus === "connected" ? "En vivo" : "Conectando..."}
                  </span>
                </div>
                {latestResult && (
                  <span className="text-white/70 text-xs">
                    {latestResult.processing_time_ms}ms
                  </span>
                )}
              </div>

              {/* Resumen de detección */}
              <div className="absolute bottom-0 left-0 right-0 p-3 bg-black/70">
                <p className="text-white text-lg font-medium">
                  {latestResult?.summary || "Apunta la cámara para detectar..."}
                </p>
                {latestResult && latestResult.object_count > 0 && (
                  <p className="text-primary text-sm mt-1">
                    {latestResult.object_count} objeto(s) detectado(s)
                  </p>
                )}
              </div>
            </div>

            {/* Lista de objetos detectados */}
            {latestResult && latestResult.objects.length > 0 && (
              <Card>
                <CardContent className="py-3">
                  <ul className="space-y-1">
                    {latestResult.objects.slice(0, 5).map((obj, idx) => {
                      const zoneColor =
                        obj.distance_zone === 'muy_cerca' ? 'bg-red-500' :
                        obj.distance_zone === 'cerca' ? 'bg-amber-500' : 'bg-green-500';
                      const textColor =
                        obj.distance_zone === 'muy_cerca' ? 'text-red-500' :
                        obj.distance_zone === 'cerca' ? 'text-amber-500' : 'text-green-500';
                      return (
                        <li key={idx} className="flex justify-between items-center p-2 bg-secondary rounded text-sm">
                          <div className="flex items-start gap-2">
                            <div className={cn("w-2 h-2 rounded-full mt-1 shrink-0", zoneColor)} />
                            <div>
                              <span>{obj.name_es}</span>
                              {obj.distance_estimate && (
                                <p className={cn("text-xs mt-0.5", textColor)}>{obj.distance_estimate}</p>
                              )}
                            </div>
                          </div>
                          <span className="text-muted-foreground">
                            {(obj.confidence * 100).toFixed(0)}%
                          </span>
                        </li>
                      );
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
                onClick={reset}
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
                <Button onClick={reset} className="w-full min-h-[48px]" aria-label="Intentar de nuevo">
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
