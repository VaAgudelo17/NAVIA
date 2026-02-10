"""
============================================================================
NAVIA Backend - Endpoint WebSocket para Detección en Tiempo Real
============================================================================
Recibe frames de cámara via WebSocket, procesa con YOLOv8 + Depth Anything,
y devuelve resultados con zonas de distancia suavizadas.

Pipeline por frame:
1. Decodificar base64 → imagen OpenCV
2. YOLO v8s → detección de objetos
3. Depth Anything V2 → mapa de profundidad
4. Clasificar objetos en zonas (muy_cerca / cerca / lejos)
5. Exponential smoothing + zone persistence
6. Solo hablar si zona persiste varios frames

Protocolo:
- Cliente envía: {"type": "frame", "data": "<base64 JPEG>", "frame_id": N}
- Servidor responde: {"type": "detection", "objects": [...], "changes": {...}}
============================================================================
"""

import asyncio
import base64
import json
import time
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.utils.image_utils import bytes_to_cv2_image, resize_image_if_needed
from app.services.object_detection_service import get_object_detection_service
from app.services.realtime_detection_service import RealtimeSessionState
from app.services.depth_estimation_service import ZONE_LABELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket Tiempo Real"])


@router.websocket("/realtime")
async def realtime_detection(websocket: WebSocket):
    """
    WebSocket para detección de objetos en tiempo real con profundidad.
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

    latest_frame = {"data": None, "frame_id": None}
    frame_event = asyncio.Event()
    connection_alive = True
    config = {
        "confidence_threshold": settings.WS_REALTIME_CONFIDENCE_THRESHOLD
    }

    async def receive_loop():
        """Recibe frames y configuración del cliente."""
        nonlocal connection_alive
        try:
            while connection_alive:
                raw = await websocket.receive_text()
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
        """Procesa frames con YOLO + Depth Anything + smoothing."""
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

                    # Detección + profundidad en thread pool (CPU-bound)
                    result = await loop.run_in_executor(
                        None,
                        detector.detect_objects,
                        cv2_image,
                        config.get("confidence_threshold"),
                    )

                    processing_time = int((time.time() - start_time) * 1000)

                    # Calcular cambios con smoothing de profundidad
                    raw_depths = result.get("raw_depths", {})
                    changes = session_state.compute_changes(
                        result["objects"],
                        raw_depths=raw_depths,
                    )

                    # Aplicar zonas suavizadas a los objetos
                    smoothed_zones = changes.get("smoothed_zones", {})

                    objects_data = []
                    for obj in result["objects"]:
                        # Usar zona suavizada si está disponible
                        zone = smoothed_zones.get(
                            obj.name_es, obj.distance_zone or "lejos"
                        )
                        label = ZONE_LABELS.get(zone, zone)

                        objects_data.append({
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
                        })

                    response = {
                        "type": "detection",
                        "frame_id": frame_id,
                        "objects": objects_data,
                        "object_count": result["object_count"],
                        "summary": result["summary"],
                        "processing_time_ms": processing_time,
                        "timestamp": int(time.time() * 1000),
                        "changes": changes,
                    }
                    await websocket.send_json(response)

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
        logger.info(
            f"Sesión WebSocket finalizada. "
            f"Frames procesados: {session_state.frame_count}"
        )
