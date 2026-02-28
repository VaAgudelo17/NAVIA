"""
============================================================================
NAVIA Backend - Configuración de Base de Datos (SQLAlchemy Async)
============================================================================
Este módulo configura el motor de base de datos y la fábrica de sesiones
usando SQLAlchemy 2.0 con soporte async.

Uso:
    - En endpoints FastAPI: usar `get_db()` como dependencia
    - En startup/shutdown: usar `init_db()` y `close_db()`
============================================================================
"""

import logging
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# BASE DECLARATIVA
# ============================================================================
# Todos los modelos SQLAlchemy heredan de esta clase base.
# Esto permite que SQLAlchemy conozca todas las tablas del proyecto.

class Base(DeclarativeBase):
    """Clase base para todos los modelos SQLAlchemy."""
    pass


# ============================================================================
# MOTOR Y SESIÓN (se inicializan en startup)
# ============================================================================

engine = None
async_session_factory = None


async def init_db():
    """
    Inicializa el motor de base de datos y crea las tablas.

    Se ejecuta durante el startup de la aplicación FastAPI.
    Crea todas las tablas definidas en los modelos si no existen.
    """
    global engine, async_session_factory

    logger.info(f"Conectando a base de datos: {settings.DATABASE_URL}")

    # Crear motor async
    # Para SQLite necesitamos check_same_thread=False
    connect_args = {}
    if settings.DATABASE_URL.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DB_ECHO,
        connect_args=connect_args,
    )

    # Crear fábrica de sesiones
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Crear tablas (importar modelos antes para que Base los conozca)
    from app.db import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Base de datos inicializada correctamente")


async def close_db():
    """
    Cierra la conexión a la base de datos.

    Se ejecuta durante el shutdown de la aplicación FastAPI.
    """
    global engine
    if engine:
        await engine.dispose()
        logger.info("Conexión a base de datos cerrada")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependencia FastAPI para obtener una sesión de base de datos.

    Uso en endpoints:
        @router.get("/ejemplo")
        async def ejemplo(db: AsyncSession = Depends(get_db)):
            ...

    La sesión se cierra automáticamente al finalizar el request.
    """
    if async_session_factory is None:
        raise RuntimeError("Base de datos no inicializada. Llame a init_db() primero.")

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
