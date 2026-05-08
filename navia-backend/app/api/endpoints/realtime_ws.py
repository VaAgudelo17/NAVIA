"""
============================================================================
NAVIA Backend - Endpoint WebSocket para Detección en Tiempo Real
============================================================================
Recibe frames de cámara via WebSocket, procesa con YOLO-World + Depth Anything,
y devuelve resultados con navegación unificada (navegación + riesgo).

Pipeline por frame:
1. Decodificar base64 → imagen OpenCV
2. YOLO-World v2 → detección de objetos (vocabulario abierto)
3. Depth Anything V2 → mapa de profundidad
4. ByteTrack → tracking espacial (IoU + Kalman filter)
5. Exponential smoothing + zone persistence
6. NavigationGuidanceService → pipeline unificado:
   - Filtrado de clases relevantes para caminata
   - Clasificación de altura (suelo/cuerpo/cabeza)
   - Análisis de movimiento (se acerca/aleja/estático)
   - Scoring de riesgo combinado
   - Instrucciones priorizadas para TTS

Protocolo:
- Cliente envía: {"type": "frame", "data": "<base64 JPEG>", "frame_id": N}
- Servidor responde: {"type": "detection", "objects": [...], "guidance": {...}}
============================================================================
"""

import asyncio
import base64
import json
import time
import logging
from collections import Counter

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.utils.image_utils import bytes_to_cv2_image, resize_image_if_needed
from app.services.object_detection_service import get_object_detection_service
from app.services.realtime_detection_service import RealtimeSessionState
from app.services.depth_estimation_service import ZONE_LABELS, DepthEstimationService
from app.services.navigation_guidance_service import get_navigation_guidance_service, IGNORE_CLASSES, PEDESTRIAN_RELEVANT_CLASSES
from app.services.history_service import save_to_history

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket Tiempo Real"])


@router.websocket("/realtime")
async def realtime_detection(websocket: WebSocket):
    """
    WebSocket para detección de objetos en tiempo real con profundidad
    y navegación unificada (navegación + riesgo en un solo pipeline).
    """
    await websocket.accept()
    logger.info("Nueva conexión WebSocket de tiempo real")

    await websocket.send_json({
        "type": "status",
        "state": "connected",
        "message": "Conexión establecida. Listo para recibir frames."
    })

    session_state = RealtimeSessionState()
    detector = get_object_detection_service()
    detector.configure_for_navigation()

    latest_frame = {"data": None, "frame_id": None}
    frame_event = asyncio.Event()
    connection_alive = True
    config = {
        "confidence_threshold": settings.WS_REALTIME_CONFIDENCE_THRESHOLD,
        "mode": "navegacion",
    }
    guidance_service = get_navigation_guidance_service()

    # Flag para disparar la clasificación de entorno solo en el primer frame
    scene_intro_done = [False]

    # Umbral de profundidad para detectar superficie plana grande al frente (pared/puerta).
    # 0.42: suficientemente alto para ignorar fluctuaciones normales del depth map,
    # pero detectable cuando hay una puerta/pared a ~1-1.5m.
    _WALL_DEPTH_THRESHOLD = 0.52           # subido: ignora fluctuaciones menores del depth
    _last_wall_warning_frame = [-15]       # cooldown de 15 frames (~5s a 3fps)
    _phantom_consecutive = [0]             # frames consecutivos con phantom detectado
    _PHANTOM_MIN_FRAMES = 3                # confirmar phantom solo si persiste ≥3 frames

    # --- Estadísticas de sesión para historial ---
    session_stats = {
        "start_time": time.time(),
        "frames_processed": 0,
        "danger_count": 0,
        "obstacle_names": Counter(),   # {"coche": 5, "persona": 3}
        "max_risk_score": 0.0,
        "total_processing_ms": 0,
    }

    async def _classify_scene_task(image_copy) -> None:
        """
        Fire-and-forget: clasifica el entorno del primer frame con Gemini
        y envía un mensaje 'scene_intro' al cliente. No bloquea la navegación.
        """
        try:
            from app.services.gemini_service import get_gemini_service
            gemini = get_gemini_service()
            if not gemini.is_available:
                return

            _loop = asyncio.get_event_loop()
            description = await asyncio.wait_for(
                _loop.run_in_executor(None, gemini.classify_scene, image_copy),
                timeout=6.0,
            )

            if description and connection_alive:
                try:
                    await websocket.send_json({
                        "type": "scene_intro",
                        "description": description,
                    })
                except Exception:
                    pass  # Cliente desconectado antes de que Gemini respondiera

        except asyncio.TimeoutError:
            logger.warning("[SceneIntro] Timeout (>6s), sin clasificación de entorno")
        except Exception as e:
            logger.warning(f"[SceneIntro] Error: {e}")

    async def receive_loop():
        """Recibe frames y configuración del cliente."""
        nonlocal connection_alive
        try:
            while connection_alive:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                except asyncio.TimeoutError:
                    # Sin frames en 10s: el cliente probablemente se desconectó sin
                    # cerrar el socket correctamente (red móvil caída, app en fondo).
                    logger.warning("WebSocket sin actividad 10s, cerrando conexión")
                    connection_alive = False
                    frame_event.set()
                    break
                msg = json.loads(raw)

                if msg.get("type") == "frame":
                    data = msg.get("data")
                    if data and len(data) <= settings.WS_MAX_FRAME_SIZE:
                        latest_frame["data"] = data
                        latest_frame["frame_id"] = msg.get("frame_id")
                        frame_event.set()
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Frame inválido o muy grande",
                            "code": "INVALID_FRAME"
                        })

                elif msg.get("type") == "config":
                    if "confidence_threshold" in msg:
                        config["confidence_threshold"] = msg["confidence_threshold"]
                    # "navegacion" ahora incluye riesgo unificado
                    # Se acepta "navegacion" o "riesgo" por compatibilidad
                    if "mode" in msg:
                        config["mode"] = "navegacion"
                        logger.info("Modo WebSocket: navegacion (unificado)")

                elif msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            connection_alive = False
            frame_event.set()
        except Exception as e:
            logger.error(f"Error en receive_loop: {e}")
            connection_alive = False
            frame_event.set()

    async def process_loop():
        """Procesa frames con YOLO + Depth Anything + smoothing + navegación unificada."""
        nonlocal connection_alive
        loop = asyncio.get_event_loop()

        try:
            while connection_alive:
                await frame_event.wait()
                frame_event.clear()

                if not connection_alive:
                    break

                frame_data = latest_frame["data"]
                frame_id = latest_frame["frame_id"]

                if frame_data is None:
                    continue

                latest_frame["data"] = None

                try:
                    start_time = time.time()

                    # Decodificar base64 → cv2
                    image_bytes = base64.b64decode(frame_data)
                    cv2_image = bytes_to_cv2_image(image_bytes)
                    cv2_image = resize_image_if_needed(
                        cv2_image,
                        max_dimension=settings.WS_REALTIME_MAX_DIMENSION
                    )

                    # Primer frame: clasificar entorno en paralelo (fire-and-forget)
                    if not scene_intro_done[0]:
                        scene_intro_done[0] = True
                        asyncio.create_task(_classify_scene_task(cv2_image.copy()))

                    # Detección + profundidad en thread pool (CPU-bound)
                    result = await loop.run_in_executor(
                        None,
                        detector.detect_objects,
                        cv2_image,
                        config.get("confidence_threshold"),
                    )

                    processing_time = int((time.time() - start_time) * 1000)

                    # Calcular cambios con smoothing de profundidad + tracking
                    raw_depths = result.get("raw_depths", {})
                    h, w = cv2_image.shape[:2]
                    changes = session_state.compute_changes(
                        result["objects"],
                        raw_depths=raw_depths,
                        img_shape=(h, w),
                    )

                    # Aplicar zonas suavizadas a los objetos
                    smoothed_zones = changes.get("smoothed_zones", {})

                    # Actualizar distance_zone con zonas suavizadas
                    for obj in result["objects"]:
                        if obj.name_es in smoothed_zones:
                            obj.distance_zone = smoothed_zones[obj.name_es]

                    # ── FILTRO DE COHERENCIA YOLO ↔ DEPTH ──────────────────
                    # Descarta detecciones cuyo depth dentro del bbox no
                    # corresponde con la zona declarada (reduce ~50% de los
                    # falsos positivos tipo gato/perro/caja).
                    depth_map_for_filter = result.get("depth_map")
                    if depth_map_for_filter is not None:
                        before = len(result["objects"])
                        result["objects"] = DepthEstimationService.filter_depth_coherent(
                            result["objects"], depth_map_for_filter
                        )
                        dropped = before - len(result["objects"])
                        if dropped > 0:
                            logger.debug(
                                f"[Coherence] Descartados {dropped} objetos por incoherencia depth-yolo"
                            )

                    objects_data = []
                    for obj in result["objects"]:
                        # Excluir del display objetos en lista negra o irrelevantes
                        # para navegación (evita mostrar cortinas, espejos, etc.)
                        if obj.name_es in IGNORE_CLASSES:
                            continue
                        if obj.name_es not in PEDESTRIAN_RELEVANT_CLASSES:
                            continue

                        zone = smoothed_zones.get(
                            obj.name_es, obj.distance_zone or "lejos"
                        )
                        label = ZONE_LABELS.get(zone, zone)

                        obj_data = {
                            "name": obj.name,
                            "name_es": obj.name_es,
                            "confidence": obj.confidence,
                            "bounding_box": {
                                "x_min": obj.bounding_box.x_min,
                                "y_min": obj.bounding_box.y_min,
                                "x_max": obj.bounding_box.x_max,
                                "y_max": obj.bounding_box.y_max,
                            },
                            "distance_zone": zone,
                            "distance_estimate": label,
                        }
                        if obj.priority:
                            obj_data["priority"] = obj.priority
                        objects_data.append(obj_data)

                    # ── DETECCIÓN DE OBSTÁCULO FANTASMA POR DEPTH ─────────
                    # Si YOLO no clasificó nada relevante (lista vacía después
                    # del filtro de coherencia y de descartar irrelevantes),
                    # buscamos en el depth map regiones muy cercanas que
                    # podrían ser paredes, árboles o cualquier obstáculo
                    # desconocido. Inyectamos el fantasma como un DetectedObject
                    # con name_es="obstáculo" para que entre al pipeline normal.
                    depth_map_for_phantom = result.get("depth_map")
                    relevant_objs = [
                        o for o in result["objects"]
                        if o.name_es in PEDESTRIAN_RELEVANT_CLASSES
                        and o.name_es not in IGNORE_CLASSES
                    ]
                    if depth_map_for_phantom is not None and not relevant_objs:
                        phantom = DepthEstimationService.detect_phantom_obstacle(
                            depth_map_for_phantom,
                            result["objects"],
                            threshold=settings.DEPTH_ZONE_MUY_CERCA,
                            min_area_ratio=0.10,
                        )
                        if phantom:
                            _phantom_consecutive[0] += 1
                        else:
                            _phantom_consecutive[0] = 0

                        # Solo inyectar el phantom si persiste ≥3 frames consecutivos.
                        # Un spike de un solo frame es casi siempre ruido del sensor
                        # (reflejo, cambio de luz, superficie plana lejana).
                        if phantom and _phantom_consecutive[0] >= _PHANTOM_MIN_FRAMES:
                            from app.models.schemas import DetectedObject, BoundingBox
                            x1, y1, x2, y2 = phantom["bbox"]
                            phantom_obj = DetectedObject(
                                name="unknown_obstacle",
                                name_es="obstáculo",
                                confidence=0.50,
                                bounding_box=BoundingBox(
                                    x_min=x1, y_min=y1, x_max=x2, y_max=y2,
                                ),
                                distance_zone=DepthEstimationService.depth_to_zone(
                                    phantom["depth"]
                                ),
                                distance_estimate="muy cerca",
                            )
                            result["objects"].append(phantom_obj)
                            logger.info(
                                f"[Phantom] Obstáculo confirmado tras {_phantom_consecutive[0]} frames "
                                f"({phantom['area_ratio']*100:.0f}% del frame, "
                                f"depth={phantom['depth']:.2f}, {phantom['position']})"
                            )
                    else:
                        _phantom_consecutive[0] = 0

                    # Si hay frames de phantom acumulando (aún sin confirmar),
                    # marcar como sospechoso — el depth ya detectó algo.
                    depth_suspicious = _phantom_consecutive[0] > 0

                    # Pipeline unificado: navegación + riesgo
                    guidance = guidance_service.generate_guidance(
                        result["objects"], w, h,
                        track_movement=True,
                    )

                    # Suprimir "camino libre" cuando el depth map acumula frames
                    # de objeto cercano sin confirmar aún (phantom en progreso).
                    # Evita que el usuario oiga "camino libre" mientras hay algo
                    # bloqueando el camino que YOLO no ha clasificado todavía.
                    if depth_suspicious and guidance.get("path_clear") and not guidance.get("has_danger"):
                        guidance["path_clear"] = False
                        guidance["instruction"] = ""

                    # --- Detección de obstáculo grande / pared por profundidad ---
                    # YOLO no detecta bien superficies planas (paredes, puertas grandes).
                    # Si el mapa de profundidad muestra el centro MUY cercano y guidance
                    # dice "camino libre", anulamos ese resultado.
                    depth_map = result.get("depth_map")
                    if (
                        depth_map is not None
                        and guidance["path_clear"]
                        and (session_stats["frames_processed"] - _last_wall_warning_frame[0]) >= 15
                    ):
                        directional = DepthEstimationService.compute_directional_depth(depth_map)
                        center_depth = directional["centro"]
                        if center_depth > _WALL_DEPTH_THRESHOLD:
                            guidance["path_clear"] = False
                            guidance["has_danger"] = True
                            guidance["alert_type"] = "atencion"

                            # Si YOLO detectó un objeto específico al frente, usarlo.
                            # Solo decir "algo" / "obstáculo" si no hay detección específica.
                            front_obj = next(
                                (o for o in result["objects"]
                                 if o.bounding_box.x_min < w * 0.65
                                 and o.bounding_box.x_max > w * 0.35),
                                None,
                            )
                            obj_name = front_obj.name_es.capitalize() if front_obj else None

                            if center_depth > 0.83:
                                guidance["priority"] = "critical"
                                if obj_name:
                                    guidance["instruction"] = f"¡Cuidado! {obj_name} muy cerca al frente, detente."
                                else:
                                    guidance["instruction"] = "¡Cuidado! Superficie muy cerca al frente, detente."
                            elif center_depth > 0.65:
                                guidance["priority"] = "high"
                                if obj_name:
                                    guidance["instruction"] = f"Atención: {obj_name} bloqueando el camino al frente."
                                else:
                                    guidance["instruction"] = "Atención: algo bloqueando el camino al frente."
                            else:
                                guidance["priority"] = "high"
                                if obj_name:
                                    guidance["instruction"] = f"Atención: {obj_name} al frente, reduce la velocidad."
                                else:
                                    guidance["instruction"] = "Atención: obstáculo al frente, reduce la velocidad."
                            _last_wall_warning_frame[0] = session_stats["frames_processed"]
                            # Asignar guidance_key fija para detecciones de pared/superficie
                            # para que el cooldown de 12s del TTS aplique correctamente.
                            # Usar el MISMO formato que el guidance_key normal
                            # (solo nombre del objeto en minúscula). Esto permite
                            # que el cooldown por obstáculo en el móvil agrupe
                            # alertas del wall detect con alertas normales del
                            # mismo objeto, evitando repeticiones cortantes.
                            guidance["_wall_key"] = (obj_name or "superficie").lower()
                            logger.info(
                                f"[WallDetect] depth={center_depth:.2f} → "
                                f"{guidance['instruction']}"
                            )

                    # Clave estable: solo nombre del objeto top, sin posición ni
                    # proximidad. Si entre frames el ranking cambia (silla→mesa
                    # →escritorio en una habitación llena), antes la guidance_key
                    # cambiaba y el cooldown se reseteaba, generando spam de TTS.
                    # Ahora el cooldown aplica por "objeto principal de la escena".
                    top_obs = (guidance.get("obstacle_details") or [{}])[0]
                    if guidance.get("_wall_key"):
                        guidance_key = guidance["_wall_key"]
                    elif top_obs.get("name"):
                        guidance_key = top_obs["name"]
                    else:
                        guidance_key = ""

                    response = {
                        "type": "detection",
                        "mode": "navegacion",
                        "frame_id": frame_id,
                        "objects": objects_data,
                        "object_count": result["object_count"],
                        "processing_time_ms": processing_time,
                        "timestamp": int(time.time() * 1000),
                        "changes": changes,
                        "tracked_count": changes.get("tracked_count"),
                        # --- Navegación unificada ---
                        "summary": guidance["instruction"],
                        "has_danger": guidance["has_danger"],
                        "priority": guidance["priority"],
                        "path_clear": guidance["path_clear"],
                        "obstacle_details": guidance["obstacle_details"],
                        "guidance_key": guidance_key,
                        "alert_type": guidance.get("alert_type"),
                        "scene_stable": guidance.get("scene_stable", False),
                    }

                    await websocket.send_json(response)

                    # --- Log cada 10 frames para seguimiento en terminal ---
                    if session_stats["frames_processed"] % 10 == 0:
                        obj_summary = ", ".join(
                            f"{o['name_es']}({o['distance_zone']})"
                            for o in objects_data[:5]
                        ) or "ninguno"
                        logger.info(
                            f"[Nav f#{session_stats['frames_processed']}] "
                            f"objetos=[{obj_summary}] | "
                            f"alert={guidance.get('alert_type') or 'none'} "
                            f"priority={guidance['priority']} | "
                            f"\"{guidance['instruction'][:60]}\""
                        )

                    # --- Acumular estadísticas de sesión ---
                    session_stats["frames_processed"] += 1
                    session_stats["total_processing_ms"] += processing_time
                    if guidance["has_danger"]:
                        session_stats["danger_count"] += 1
                    for obs in guidance.get("obstacle_details", []):
                        session_stats["obstacle_names"][obs["name"]] += 1
                        if obs.get("risk_score", 0) > session_stats["max_risk_score"]:
                            session_stats["max_risk_score"] = obs["risk_score"]

                except Exception as e:
                    logger.error(f"Error procesando frame: {e}")
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Error procesando frame: {str(e)}",
                            "code": "PROCESSING_ERROR"
                        })
                    except Exception:
                        connection_alive = False
                        break

        except Exception as e:
            logger.error(f"Error en process_loop: {e}")
            connection_alive = False

    try:
        await asyncio.gather(receive_loop(), process_loop())
    except WebSocketDisconnect:
        logger.info("Cliente WebSocket desconectado")
    except Exception as e:
        logger.error(f"Error en sesión WebSocket: {e}")
    finally:
        detector.configure_for_full()
        guidance_service.reset_movement_state()

        # --- Guardar resumen de sesión en historial ---
        frames = session_stats["frames_processed"]
        if frames > 0:
            duration_s = time.time() - session_stats["start_time"]
            duration_min = duration_s / 60
            top_obstacles = session_stats["obstacle_names"].most_common(5)
            danger_count = session_stats["danger_count"]

            # Resumen corto legible
            if top_obstacles:
                obs_text = ", ".join(
                    f"{name} ({count}x)" for name, count in top_obstacles
                )
                summary = (
                    f"Sesión de {duration_min:.1f} min, "
                    f"{frames} frames. "
                    f"Obstáculos: {obs_text}. "
                    f"Alertas de peligro: {danger_count}."
                )
            else:
                summary = (
                    f"Sesión de {duration_min:.1f} min, "
                    f"{frames} frames. Camino libre."
                )

            avg_ms = session_stats["total_processing_ms"] / frames

            try:
                await save_to_history(
                    mode="navegacion",
                    result_data={
                        "type": "realtime_session",
                        "duration_seconds": round(duration_s, 1),
                        "frames_processed": frames,
                        "danger_count": danger_count,
                        "max_risk_score": round(session_stats["max_risk_score"], 2),
                        "top_obstacles": dict(top_obstacles),
                        "avg_processing_ms": round(avg_ms, 1),
                    },
                    result_summary=summary,
                    processing_time_ms=avg_ms,
                    object_count=sum(session_stats["obstacle_names"].values()),
                    has_danger=danger_count > 0,
                )
                logger.info(f"Historial de sesión realtime guardado: {summary[:80]}...")
            except Exception as e:
                logger.error(f"Error guardando historial de sesión realtime: {e}")

        logger.info(
            f"Sesión WebSocket finalizada. "
            f"Frames procesados: {session_state.frame_count}"
        )
