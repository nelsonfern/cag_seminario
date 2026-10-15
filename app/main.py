"""Punto de entrada de TramitaIA.

La superficie HTTP completa vive en `app.api.routes`: estado, fuentes,
chat y resultados del experimento. Este modulo solo arma la aplicacion,
configura CORS, calienta los servicios y monta el frontend.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging

logger = logging.getLogger("tramitaia.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepara y libera los recursos del ciclo de vida de la aplicacion.

    **Calienta CAG y RAG antes de aceptar la primera consulta.** No es
    una optimizacion: sin esto, la primera consulta paga la carga del
    modelo de embeddings (unos 30 segundos) y el usuario la lee como
    "RAG es mas lento". El panel de comparacion mostraria un numero de
    arranque, no de recuperacion.

    Los proveedores se inicializan en un hilo aparte para que el arranque
    no bloquee el event loop.
    """
    configure_logging()
    settings = get_settings()
    settings.ensure_directories()
    logger.info(
        "Iniciando %s | proveedor=%s modelo=%s",
        settings.app_name,
        settings.llm_provider,
        settings.model_name,
    )

    if settings.warmup_on_startup:
        await asyncio.to_thread(calentar_servicios)

    yield
    logger.info("Deteniendo %s", get_settings().app_name)


def calentar_servicios() -> None:
    """Construye el contexto de CAG y el indice de RAG.

    Cada falla se registra y se sigue: que RAG no pueda arrancar no debe
    impedir que CAG responda. La consulta lo va a avisar.
    """
    from app.services.cag import obtener_servicio_cag
    from app.services.rag import obtener_servicio_rag

    inicio = time.perf_counter()

    try:
        preparado = obtener_servicio_cag().preparar(force=True)
        logger.info(
            "CAG listo: %d documentos, ~%d tokens, %.0f ms",
            preparado.documentos_incluidos,
            preparado.tokens_estimados,
            (time.perf_counter() - inicio) * 1000,
        )
    except Exception:
        logger.warning("No se pudo calentar CAG", exc_info=True)

    try:
        busqueda = obtener_servicio_rag().recuperar("tramite de la provincia")
        logger.info(
            "RAG listo: %d fragmentos, %.0f ms",
            busqueda.obtenidos,
            (time.perf_counter() - inicio) * 1000,
        )
    except Exception:
        logger.warning("No se pudo calentar RAG", exc_info=True)


DIRECTORIO_FRONTEND = Path(__file__).resolve().parents[1] / "frontend"

# Solo se sirven estos archivos. Una lista blanca explicita evita que un
# error de ruta termine sirviendo archivos del proyecto, entre ellos el
# `.env` con la clave de API.
ARCHIVOS_FRONTEND = {"index.html", "styles.css", "app.js"}

TIPOS_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


def _servir_archivo(nombre: str) -> Response:
    """Devuelve un archivo del frontend, o 404 si no esta permitido."""
    if nombre not in ARCHIVOS_FRONTEND:
        raise HTTPException(status_code=404, detail="Archivo no permitido")

    ruta = DIRECTORIO_FRONTEND / nombre
    if not ruta.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Falta {nombre}. El frontend deberia estar en frontend/.",
        )

    return Response(
        ruta.read_bytes(),
        media_type=TIPOS_MIME.get(ruta.suffix, "application/octet-stream"),
    )


def montar_frontend(app: FastAPI) -> None:
    """Sirve la interfaz estatica desde el mismo servidor que la API.

    Va montada en la propia app para que en la exposicion baste con
    `uvicorn app.main:app`, sin levantar un segundo servidor.
    """

    @app.get("/", include_in_schema=False)
    async def raiz() -> Response:
        return _servir_archivo("index.html")

    @app.get("/app.js", include_in_schema=False)
    async def script() -> Response:
        return _servir_archivo("app.js")

    @app.get("/styles.css", include_in_schema=False)
    async def estilos() -> Response:
        return _servir_archivo("styles.css")


def create_app() -> FastAPI:
    """Construye la instancia de FastAPI."""

    # La configuracion se resuelve por peticion y no al importar el modulo.
    # Ligarla al import hace que la app no pueda reconfigurarse ni probarse
    # con distintos proveedores.
    def config() -> Settings:
        return get_settings()

    app = FastAPI(
        title=config().app_name,
        description=(
            "Agente conversacional que explica tramites publicos de La Rioja, "
            "Argentina, usando exclusivamente fuentes oficiales. "
            "Compara CAG (Cache-Augmented Generation) contra RAG "
            "(Retrieval-Augmented Generation)."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config().cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # Los endpoints de chat, fuentes y evaluacion viven en
    # `app.api.routes`. Se incluyen aca para que toda la superficie HTTP
    # quede registrada en un unico lugar.
    app.include_router(router)

    montar_frontend(app)

    return app


app = create_app()
