"""Punto de entrada de TramitaIA.

En la Fase 1 solo expone `GET /health`. Los endpoints de chat, fuentes y
evaluacion se agregan en la Fase 6.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging

logger = logging.getLogger("tramitaia.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepara y libera los recursos del ciclo de vida de la aplicacion."""
    configure_logging()
    settings = get_settings()
    settings.ensure_directories()
    logger.info(
        "Iniciando %s | proveedor=%s modelo=%s",
        settings.app_name,
        settings.llm_provider,
        settings.model_name,
    )
    yield
    logger.info("Deteniendo %s", get_settings().app_name)


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

    @app.get("/health", tags=["sistema"], summary="Estado de la aplicacion")
    async def health() -> dict[str, object]:
        """Reporta el estado de la aplicacion.

        No expone rutas absolutas del sistema ni ninguna configuracion
        sensible: solo lo necesario para diagnosticar.
        """
        settings = config()
        configurado = settings.llm_configurado
        return {
            "status": "ok" if configurado else "degradado",
            "app": settings.app_name,
            "version": app.version,
            "llm_provider": settings.llm_provider,
            # Nombre del modelo: no es una credencial y ayuda a reproducir
            # una ejecucion concreta del experimento.
            "model": settings.model_name if configurado else None,
            "llm_configurado": configurado,
        }

    return app


app = create_app()
