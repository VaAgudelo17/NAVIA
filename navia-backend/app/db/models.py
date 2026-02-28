"""
============================================================================
NAVIA Backend - Modelos de Base de Datos (SQLAlchemy ORM)
============================================================================
Define las tablas de la base de datos para persistir:
- Historial de analisis realizados
- Preferencias del usuario

Cada modelo hereda de Base (DeclarativeBase) y se mapea
automaticamente a una tabla en la base de datos.
============================================================================
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    Float,
    Integer,
    DateTime,
    JSON,
)
from app.db.database import Base


class AnalysisHistory(Base):
    """
    Registro de cada analisis realizado por NAVIA.

    Guarda un resumen y los datos completos del resultado
    para que el usuario pueda consultar su historial.

    Atributos:
        id: Identificador unico (UUID)
        mode: Modo de analisis (navegacion, exploracion, lectura, riesgo)
        reading_mode: Sub-modo de lectura (resumen, detallado, financiero) - solo para modo lectura
        result_summary: Resumen corto del resultado (para listar en historial)
        result_data: Resultado completo serializado como JSON
        image_filename: Nombre del archivo de imagen procesado (opcional)
        processing_time_ms: Tiempo de procesamiento en milisegundos
        object_count: Numero de objetos detectados
        has_text: Si se detecto texto en la imagen
        has_danger: Si se detecto peligro (solo modo riesgo)
        created_at: Fecha y hora del analisis
    """
    __tablename__ = "analysis_history"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="ID unico del registro"
    )
    mode = Column(
        String(20),
        nullable=False,
        index=True,
        comment="Modo: navegacion, exploracion, lectura, riesgo"
    )
    reading_mode = Column(
        String(20),
        nullable=True,
        comment="Sub-modo de lectura: resumen, detallado, financiero"
    )
    result_summary = Column(
        Text,
        nullable=False,
        default="",
        comment="Resumen corto del resultado para mostrar en lista"
    )
    result_data = Column(
        JSON,
        nullable=False,
        default=dict,
        comment="Resultado completo serializado como JSON"
    )
    image_filename = Column(
        String(255),
        nullable=True,
        comment="Nombre del archivo de imagen procesado"
    )
    processing_time_ms = Column(
        Float,
        nullable=True,
        comment="Tiempo de procesamiento en milisegundos"
    )
    object_count = Column(
        Integer,
        nullable=True,
        default=0,
        comment="Numero de objetos detectados"
    )
    has_text = Column(
        Boolean,
        nullable=True,
        default=False,
        comment="Si se detecto texto en la imagen"
    )
    has_danger = Column(
        Boolean,
        nullable=True,
        default=False,
        comment="Si se detecto peligro"
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment="Fecha y hora del analisis"
    )

    def __repr__(self):
        return f"<AnalysisHistory(id={self.id[:8]}..., mode={self.mode}, created_at={self.created_at})>"


class UserPreferences(Base):
    """
    Preferencias del usuario almacenadas como pares clave-valor.

    Permite guardar configuraciones como:
    - analysis_mode: modo de analisis preferido
    - reading_mode: sub-modo de lectura preferido
    - tts_enabled: si el TTS esta activado

    Atributos:
        id: Identificador auto-incremental
        key: Nombre de la preferencia (unico)
        value: Valor de la preferencia (serializado como string)
        updated_at: Ultima vez que se actualizo
    """
    __tablename__ = "user_preferences"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="ID auto-incremental"
    )
    key = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Nombre de la preferencia"
    )
    value = Column(
        String(500),
        nullable=False,
        default="",
        comment="Valor de la preferencia"
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Ultima actualizacion"
    )

    def __repr__(self):
        return f"<UserPreferences(key={self.key}, value={self.value})>"
