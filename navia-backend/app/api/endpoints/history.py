"""
============================================================================
NAVIA Backend - Endpoints de Historial de Analisis
============================================================================
Permite consultar, ver detalles y eliminar registros del historial
de analisis realizados por NAVIA.

Endpoints:
- GET    /history          - Listar historial (paginado, filtrable)
- GET    /history/{id}     - Detalle de un analisis
- DELETE /history/{id}     - Eliminar un registro
- DELETE /history          - Limpiar todo el historial
============================================================================
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, desc

from app.db.database import get_db
from app.db.models import AnalysisHistory
from app.models.schemas import (
    BaseResponse,
    HistoryItemResponse,
    HistoryListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/history",
    tags=["Historial"],
)


@router.get(
    "",
    response_model=HistoryListResponse,
    summary="Listar historial de analisis",
    description="""
    Retorna el historial de analisis paginado y ordenado por fecha descendente.

    **Filtros opcionales:**
    - `mode`: Filtrar por modo (navegacion, exploracion, lectura, riesgo)
    - `page`: Numero de pagina (default: 1)
    - `page_size`: Registros por pagina (default: 20, max: 100)
    """,
)
async def list_history(
    mode: str = Query(
        default=None,
        description="Filtrar por modo: navegacion, exploracion, lectura, riesgo"
    ),
    page: int = Query(default=1, ge=1, description="Numero de pagina"),
    page_size: int = Query(default=20, ge=1, le=100, description="Registros por pagina"),
    db: AsyncSession = Depends(get_db),
) -> HistoryListResponse:
    """Lista el historial de analisis con paginacion."""
    try:
        # Construir query base
        query = select(AnalysisHistory)
        count_query = select(func.count(AnalysisHistory.id))

        # Filtrar por modo si se especifica
        if mode:
            valid_modes = ("navegacion", "exploracion", "lectura", "riesgo")
            if mode not in valid_modes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Modo invalido. Opciones: {', '.join(valid_modes)}"
                )
            query = query.where(AnalysisHistory.mode == mode)
            count_query = count_query.where(AnalysisHistory.mode == mode)

        # Obtener total
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Paginacion y ordenamiento
        offset = (page - 1) * page_size
        query = query.order_by(desc(AnalysisHistory.created_at)).offset(offset).limit(page_size)

        # Ejecutar query
        result = await db.execute(query)
        records = result.scalars().all()

        # Convertir a response
        items = [
            HistoryItemResponse(
                id=record.id,
                mode=record.mode,
                reading_mode=record.reading_mode,
                result_summary=record.result_summary or "",
                result_data=record.result_data or {},
                image_filename=record.image_filename,
                processing_time_ms=record.processing_time_ms,
                object_count=record.object_count,
                has_text=record.has_text,
                has_danger=record.has_danger,
                created_at=record.created_at,
            )
            for record in records
        ]

        return HistoryListResponse(
            success=True,
            message=f"Se encontraron {total} registro(s)",
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listando historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{history_id}",
    response_model=HistoryItemResponse,
    summary="Detalle de un analisis",
    description="Retorna el detalle completo de un registro del historial.",
)
async def get_history_detail(
    history_id: str,
    db: AsyncSession = Depends(get_db),
) -> HistoryItemResponse:
    """Obtiene el detalle de un registro del historial."""
    try:
        result = await db.execute(
            select(AnalysisHistory).where(AnalysisHistory.id == history_id)
        )
        record = result.scalar_one_or_none()

        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"Registro no encontrado: {history_id}"
            )

        return HistoryItemResponse(
            id=record.id,
            mode=record.mode,
            reading_mode=record.reading_mode,
            result_summary=record.result_summary or "",
            result_data=record.result_data or {},
            image_filename=record.image_filename,
            processing_time_ms=record.processing_time_ms,
            object_count=record.object_count,
            has_text=record.has_text,
            has_danger=record.has_danger,
            created_at=record.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo detalle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/{history_id}",
    response_model=BaseResponse,
    summary="Eliminar un registro del historial",
)
async def delete_history_item(
    history_id: str,
    db: AsyncSession = Depends(get_db),
) -> BaseResponse:
    """Elimina un registro individual del historial."""
    try:
        result = await db.execute(
            select(AnalysisHistory).where(AnalysisHistory.id == history_id)
        )
        record = result.scalar_one_or_none()

        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"Registro no encontrado: {history_id}"
            )

        await db.delete(record)
        await db.flush()

        return BaseResponse(
            success=True,
            message=f"Registro {history_id} eliminado correctamente",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando registro: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "",
    response_model=BaseResponse,
    summary="Limpiar todo el historial",
    description="Elimina todos los registros del historial. Esta accion no se puede deshacer.",
)
async def clear_history(
    db: AsyncSession = Depends(get_db),
) -> BaseResponse:
    """Elimina todos los registros del historial."""
    try:
        result = await db.execute(delete(AnalysisHistory))
        count = result.rowcount

        return BaseResponse(
            success=True,
            message=f"Se eliminaron {count} registro(s) del historial",
        )

    except Exception as e:
        logger.error(f"Error limpiando historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))
