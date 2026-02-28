"""
============================================================================
NAVIA Backend - Servicio de Historial
============================================================================
Servicio para guardar automaticamente los resultados de analisis
en la base de datos. Se invoca desde los endpoints de analisis
de forma asincrona (fire-and-forget) para no bloquear la respuesta.
============================================================================
"""

import logging
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def save_to_history(
    mode: str,
    result_data: dict,
    result_summary: str = "",
    reading_mode: str = None,
    image_filename: str = None,
    processing_time_ms: float = None,
    object_count: int = 0,
    has_text: bool = False,
    has_danger: bool = False,
) -> str | None:
    """
    Guarda un resultado de analisis en el historial.

    Se ejecuta de forma independiente usando su propia sesion de BD
    para no interferir con la sesion del request principal.

    Importa database y models dentro de la funcion para evitar
    capturar la referencia a async_session_factory antes de que
    init_db() la inicialice durante el lifespan.

    Args:
        mode: Modo de analisis (navegacion, exploracion, lectura, riesgo)
        result_data: Diccionario con el resultado completo
        result_summary: Resumen corto para mostrar en lista
        reading_mode: Sub-modo de lectura (solo para modo lectura)
        image_filename: Nombre del archivo de imagen
        processing_time_ms: Tiempo de procesamiento
        object_count: Numero de objetos detectados
        has_text: Si se detecto texto
        has_danger: Si se detecto peligro

    Returns:
        ID del registro creado, o None si hubo error
    """
    # Importar aqui para obtener la referencia actualizada despues de init_db()
    from app.db import database
    from app.db.models import AnalysisHistory

    if database.async_session_factory is None:
        logger.warning("Base de datos no inicializada, no se guarda historial")
        return None

    try:
        async with database.async_session_factory() as session:
            # Serializar result_data para asegurar que es JSON-compatible
            safe_result = _make_json_safe(result_data)

            record = AnalysisHistory(
                mode=mode,
                reading_mode=reading_mode,
                result_summary=result_summary[:500] if result_summary else "",
                result_data=safe_result,
                image_filename=image_filename,
                processing_time_ms=processing_time_ms,
                object_count=object_count,
                has_text=has_text,
                has_danger=has_danger,
            )
            session.add(record)
            await session.commit()

            logger.info(f"Historial guardado: mode={mode}, id={record.id[:8]}...")
            return record.id

    except Exception as e:
        logger.error(f"Error guardando historial: {e}")
        return None


def _make_json_safe(data) -> dict:
    """
    Convierte un objeto a un diccionario JSON-serializable.
    Maneja objetos Pydantic, datetimes, y otros tipos no serializables.
    """
    try:
        if hasattr(data, "model_dump"):
            # Pydantic v2
            data = data.model_dump()
        elif hasattr(data, "dict"):
            # Pydantic v1
            data = data.dict()

        # Pasar por JSON para limpiar tipos no serializables
        return json.loads(json.dumps(data, default=str))
    except Exception:
        return {"raw": str(data)}
