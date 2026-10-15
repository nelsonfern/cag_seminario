"""Persistencia de consultas, metricas y ejecuciones de evaluacion.

SQLite se usa porque el acceso es relacional y frecuente: cada consulta
del chat, cada medicion y cada ejecucion del experimento tienen que poder
relacionarse despues para reconstruir el detalle.

Cada corrida del experimento se guarda con su identificador, su marca de
tiempo y el modelo usado, de modo que los resultados sean reproducibles y
auditables. Nada se sobrescribe: el enunciado pide conservar el detalle
necesario para poder volver a ejecutar el experimento.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger("evaluation_repository")

ESQUEMA = """
CREATE TABLE IF NOT EXISTS consultas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    consulta_id     TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL,
    pregunta        TEXT NOT NULL,
    estrategia      TEXT NOT NULL,
    respuesta       TEXT NOT NULL,
    grounded        INTEGER NOT NULL,
    es_abstencion   INTEGER NOT NULL,
    latencia_ms     REAL NOT NULL,
    tokens_entrada  INTEGER,
    tokens_salida   INTEGER,
    tokens_cache    INTEGER,
    cache_reportada INTEGER NOT NULL DEFAULT 0,
    modelo          TEXT NOT NULL,
    proveedor       TEXT NOT NULL,
    fuentes         TEXT NOT NULL DEFAULT '[]',
    advertencias    TEXT NOT NULL DEFAULT '[]',
    detalle         TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ejecuciones (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    finished_at     TEXT,
    modelo          TEXT NOT NULL,
    proveedor       TEXT NOT NULL,
    estado          TEXT NOT NULL,
    configuracion   TEXT NOT NULL DEFAULT '{}',
    resumen         TEXT NOT NULL DEFAULT '{}',
    csv_path        TEXT
);

CREATE TABLE IF NOT EXISTS resultados (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    pregunta_id     TEXT NOT NULL,
    estrategia      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    respuesta       TEXT NOT NULL,
    groundtruth     TEXT NOT NULL DEFAULT '',
    grounded        INTEGER NOT NULL DEFAULT 0,
    es_abstencion   INTEGER NOT NULL DEFAULT 0,
    latencia_ms     REAL NOT NULL,
    tokens_entrada  INTEGER,
    tokens_salida   INTEGER,
    tokens_cache    INTEGER,
    cache_reportada INTEGER NOT NULL DEFAULT 0,
    metricas        TEXT NOT NULL DEFAULT '{}',
    detalle         TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES ejecuciones(run_id)
);

CREATE INDEX IF NOT EXISTS idx_resultados_run ON resultados(run_id);
CREATE INDEX IF NOT EXISTS idx_consultas_created ON consultas(created_at);
CREATE INDEX IF NOT EXISTS idx_consultas_estrategia ON consultas(estrategia);
"""


def _ahora() -> str:
    """Marca de tiempo UTC en ISO 8601, con precision de microsegundos."""
    return datetime.now(UTC).isoformat()


class EvaluationRepository:
    """Acceso a SQLite para consultas, ejecuciones y resultados."""

    def __init__(self, ruta: Path) -> None:
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._inicializar()

    @contextmanager
    def _conexion(self) -> Iterator[sqlite3.Connection]:
        """Abre una conexion con claves foraneas activas.

        row_factory devuelve filas como diccionarios: el codigo no depende
        del orden de las columnas, asi que agregar una no rompe nada.
        """
        conexion = sqlite3.connect(self.ruta, isolation_level=None)
        conexion.row_factory = sqlite3.Row
        conexion.execute("PRAGMA foreign_keys = ON")
        try:
            yield conexion
        finally:
            conexion.close()

    def _inicializar(self) -> None:
        with self._conexion() as conexion:
            conexion.executescript(ESQUEMA)

    # -----------------------------------------------------------------
    # Consultas del chat
    # -----------------------------------------------------------------
    def registrar_consulta(self, **campos: Any) -> str:
        """Guarda una consulta del chat y devuelve su identificador."""
        consulta_id = str(uuid.uuid4())
        fila = {
            "consulta_id": consulta_id,
            "created_at": _ahora(),
            "pregunta": campos.get("pregunta", ""),
            "estrategia": campos.get("estrategia", ""),
            "respuesta": campos.get("respuesta", ""),
            "grounded": int(campos.get("grounded", False)),
            "es_abstencion": int(campos.get("es_abstencion", False)),
            "latencia_ms": float(campos.get("latencia_ms", 0.0)),
            "tokens_entrada": campos.get("tokens_entrada"),
            "tokens_salida": campos.get("tokens_salida"),
            "tokens_cache": campos.get("tokens_cache"),
            # Distingue "el proveedor no lo informo" de "informo cero".
            "cache_reportada": int(campos.get("cache_reportada", False)),
            "modelo": campos.get("modelo", ""),
            "proveedor": campos.get("proveedor", ""),
            "fuentes": json.dumps(campos.get("fuentes", []), ensure_ascii=False),
            "advertencias": json.dumps(campos.get("advertencias", []), ensure_ascii=False),
            "detalle": json.dumps(campos.get("detalle", {}), ensure_ascii=False),
        }

        columnas = ", ".join(fila)
        marcadores = ", ".join(f":{c}" for c in fila)
        with self._conexion() as conexion:
            conexion.execute(f"INSERT INTO consultas ({columnas}) VALUES ({marcadores})", fila)
        return consulta_id

    def contar_consultas(self) -> int:
        with self._conexion() as conexion:
            return conexion.execute("SELECT COUNT(*) AS n FROM consultas").fetchone()["n"]

    def listar_consultas(self, limite: int = 50) -> list[dict[str, Any]]:
        with self._conexion() as conexion:
            filas = conexion.execute(
                "SELECT * FROM consultas ORDER BY id DESC LIMIT ?", (limite,)
            ).fetchall()
        return [dict(f) for f in filas]

    # -----------------------------------------------------------------
    # Ejecuciones del experimento
    # -----------------------------------------------------------------
    def iniciar_ejecucion(
        self,
        modelo: str,
        proveedor: str,
        configuracion: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        """Registra el inicio de una corrida del experimento.

        El `run_id` acepta un valor explicito para permitir reanudar una
        corrida interrumpida sin duplicar resultados.
        """
        identificador = run_id or f"run-{uuid.uuid4().hex[:12]}"
        with self._conexion() as conexion:
            conexion.execute(
                """
                INSERT OR REPLACE INTO ejecuciones
                    (run_id, created_at, modelo, proveedor, estado, configuracion)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    identificador,
                    _ahora(),
                    modelo,
                    proveedor,
                    "en_curso",
                    json.dumps(configuracion or {}, ensure_ascii=False),
                ),
            )
        logger.info("Ejecucion iniciada: %s (modelo=%s)", identificador, modelo)
        return identificador

    def finalizar_ejecucion(
        self,
        run_id: str,
        estado: str = "completada",
        resumen: dict[str, Any] | None = None,
        csv_path: str | None = None,
    ) -> None:
        with self._conexion() as conexion:
            conexion.execute(
                """
                UPDATE ejecuciones
                   SET finished_at = ?, estado = ?, resumen = ?, csv_path = ?
                 WHERE run_id = ?
                """,
                (
                    _ahora(),
                    estado,
                    json.dumps(resumen or {}, ensure_ascii=False),
                    csv_path,
                    run_id,
                ),
            )

    def registrar_resultado(self, run_id: str, **campos: Any) -> None:
        """Guarda el resultado de una pregunta dentro de una corrida."""
        fila = {
            "run_id": run_id,
            "pregunta_id": campos.get("pregunta_id", ""),
            "estrategia": campos.get("estrategia", ""),
            "created_at": _ahora(),
            "respuesta": campos.get("respuesta", ""),
            "groundtruth": campos.get("groundtruth", ""),
            "grounded": int(campos.get("grounded", False)),
            "es_abstencion": int(campos.get("es_abstencion", False)),
            "latencia_ms": float(campos.get("latencia_ms", 0.0)),
            "tokens_entrada": campos.get("tokens_entrada"),
            "tokens_salida": campos.get("tokens_salida"),
            "tokens_cache": campos.get("tokens_cache"),
            "cache_reportada": int(campos.get("cache_reportada", False)),
            "metricas": json.dumps(campos.get("metricas", {}), ensure_ascii=False),
            "detalle": json.dumps(campos.get("detalle", {}), ensure_ascii=False),
        }
        columnas = ", ".join(fila)
        marcadores = ", ".join(f":{c}" for c in fila)
        with self._conexion() as conexion:
            conexion.execute(f"INSERT INTO resultados ({columnas}) VALUES ({marcadores})", fila)

    def obtener_ejecucion(self, run_id: str) -> dict[str, Any] | None:
        with self._conexion() as conexion:
            fila = conexion.execute(
                "SELECT * FROM ejecuciones WHERE run_id = ?", (run_id,)
            ).fetchone()
        if fila is None:
            return None

        ejecucion = dict(fila)
        ejecucion["configuracion"] = json.loads(ejecucion["configuracion"])
        ejecucion["resumen"] = json.loads(ejecucion["resumen"])

        with self._conexion() as conexion:
            resultados = conexion.execute(
                "SELECT * FROM resultados WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()

        # `sqlite3.Row` no admite asignacion por indice: se convierte a
        # diccionario antes de deserializar los campos JSON.
        filas = []
        for fila in resultados:
            registro = dict(fila)
            registro["metricas"] = json.loads(registro["metricas"])
            registro["detalle"] = json.loads(registro["detalle"])
            filas.append(registro)

        ejecucion["resultados"] = filas
        ejecucion["total_resultados"] = len(ejecucion["resultados"])
        return ejecucion

    def listar_ejecuciones(self, limite: int = 20) -> list[dict[str, Any]]:
        with self._conexion() as conexion:
            filas = conexion.execute(
                """
                SELECT e.run_id, e.created_at, e.finished_at, e.modelo, e.proveedor,
                       e.estado, e.resumen,
                       (SELECT COUNT(*) FROM resultados r WHERE r.run_id = e.run_id) AS total
                  FROM ejecuciones e
                 ORDER BY e.created_at DESC
                 LIMIT ?
                """,
                (limite,),
            ).fetchall()

        ejecuciones = []
        for fila in filas:
            ejecucion = dict(fila)
            ejecucion["resumen"] = json.loads(ejecucion["resumen"])
            ejecuciones.append(ejecucion)
        return ejecuciones
