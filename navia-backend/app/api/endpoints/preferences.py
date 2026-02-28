"""
============================================================================
NAVIA Backend - Endpoints de Preferencias del Usuario
============================================================================
Permite guardar y consultar las preferencias del usuario:
- Modo de analisis preferido
- Sub-modo de lectura
- Estado del TTS (activado/desactivado)

Endpoints:
- GET  /preferences          - Obtener todas las preferencias
- PUT  /preferences          - Actualizar preferencias (batch)
- GET  /preferences/{key}    - Obtener una preferencia especifica
============================================================================
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import UserPreferences
from app.models.schemas import (
    BaseResponse,
    PreferencesResponse,
    PreferencesUpdateRequest,
    PreferenceItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/preferences",
    tags=["Preferencias"],
)

# Preferencias validas con sus valores por defecto
DEFAULT_PREFERENCES = {
    "analysis_mode": "navegacion",
    "reading_mode": "detallado",
    "tts_enabled": "true",
}


@router.get(
    "",
    response_model=PreferencesResponse,
    summary="Obtener todas las preferencias",
    description="""
    Retorna todas las preferencias del usuario.
    Si una preferencia no ha sido configurada, se retorna su valor por defecto.

    **Preferencias disponibles:**
    - `analysis_mode`: navegacion, exploracion, lectura, riesgo (default: navegacion)
    - `reading_mode`: resumen, detallado, financiero (default: detallado)
    - `tts_enabled`: true, false (default: true)
    """,
)
async def get_preferences(
    db: AsyncSession = Depends(get_db),
) -> PreferencesResponse:
    """Obtiene todas las preferencias del usuario."""
    try:
        # Empezar con valores por defecto
        preferences = dict(DEFAULT_PREFERENCES)

        # Sobrescribir con valores guardados en BD
        result = await db.execute(select(UserPreferences))
        records = result.scalars().all()

        for record in records:
            preferences[record.key] = record.value

        return PreferencesResponse(
            success=True,
            message="Preferencias cargadas",
            preferences=preferences,
        )

    except Exception as e:
        logger.error(f"Error obteniendo preferencias: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "",
    response_model=PreferencesResponse,
    summary="Actualizar preferencias",
    description="""
    Actualiza una o mas preferencias del usuario.
    Enviar un diccionario con las preferencias a actualizar.

    **Ejemplo:**
    ```json
    {
        "preferences": {
            "analysis_mode": "exploracion",
            "tts_enabled": "false"
        }
    }
    ```
    """,
)
async def update_preferences(
    request: PreferencesUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> PreferencesResponse:
    """Actualiza preferencias del usuario."""
    try:
        for key, value in request.preferences.items():
            # Buscar si ya existe
            result = await db.execute(
                select(UserPreferences).where(UserPreferences.key == key)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Actualizar existente
                existing.value = str(value)
                existing.updated_at = datetime.now(timezone.utc)
            else:
                # Crear nuevo
                new_pref = UserPreferences(
                    key=key,
                    value=str(value),
                )
                db.add(new_pref)

        await db.flush()

        # Retornar todas las preferencias actualizadas
        preferences = dict(DEFAULT_PREFERENCES)
        result = await db.execute(select(UserPreferences))
        records = result.scalars().all()
        for record in records:
            preferences[record.key] = record.value

        logger.info(f"Preferencias actualizadas: {list(request.preferences.keys())}")

        return PreferencesResponse(
            success=True,
            message="Preferencias actualizadas correctamente",
            preferences=preferences,
        )

    except Exception as e:
        logger.error(f"Error actualizando preferencias: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{key}",
    response_model=PreferenceItem,
    summary="Obtener una preferencia especifica",
)
async def get_preference(
    key: str,
    db: AsyncSession = Depends(get_db),
) -> PreferenceItem:
    """Obtiene el valor de una preferencia especifica."""
    try:
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.key == key)
        )
        record = result.scalar_one_or_none()

        if record:
            return PreferenceItem(key=record.key, value=record.value)

        # Si no existe, retornar default
        if key in DEFAULT_PREFERENCES:
            return PreferenceItem(key=key, value=DEFAULT_PREFERENCES[key])

        raise HTTPException(
            status_code=404,
            detail=f"Preferencia no encontrada: {key}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo preferencia: {e}")
        raise HTTPException(status_code=500, detail=str(e))
