"""Fixtures compartidas de la suite de pruebas.

Las pruebas nunca deben:
  - tocar la base de datos real del proyecto,
  - escribir en `data/` ni en `results/`,
  - llamar a un proveedor de LLM real.

Cada prueba trabaja sobre directorios temporales.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

# -----------------------------------------------------------------------------
# Entorno de pruebas
# -----------------------------------------------------------------------------
# Estas variables se fijan a NIVEL DE MODULO, no en un fixture.
#
# pytest importa los modulos de prueba durante la recoleccion, antes de
# ejecutar ningún fixture. Como `app.core.config` instancia `settings` al
# importarse, un fixture llegaria tarde y la configuracion quedaria
# cacheada con los valores reales del entorno del desarrollador.
_FORCED_ENV = {
    "LLM_PROVIDER": "fake",
    "LOG_LEVEL": "WARNING",
    # Cada prueba construye su propia app. Calentar CAG y RAG en cada
    # una tardaria 30 segundos y, peor, dejaría el mismo servicio
    # compartido entre pruebas con resultados que dependen del orden.
    "WARMUP_ON_STARTUP": "false",
}
os.environ.update(_FORCED_ENV)

import pytest  # noqa: E402  (debe ir despues de fijar el entorno)

from app.core.config import Settings, get_settings  # noqa: E402


@pytest.fixture
def settings_de_prueba(tmp_path: Path) -> Settings:
    """Configuracion aislada, con todo lo escribible en un directorio temporal."""
    return Settings(
        sqlite_path=tmp_path / "test.db",
        raw_path=tmp_path / "raw",
        processed_path=tmp_path / "processed",
        results_path=tmp_path / "results",
        cag_cache_dir=tmp_path / "cag_cache",
        chroma_dir=tmp_path / "chroma",
        llm_provider="fake",
    )


@pytest.fixture
def cliente() -> Iterator:
    """Cliente HTTP de prueba contra la aplicacion FastAPI."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    # Se construye una app nueva: el lifespan corre en el contexto del
    # cliente, lo que inicializa logging y directorios.
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def limpiar_cache_config() -> Iterator[None]:
    """Limpia la cache de configuracion antes y despues de cada prueba."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def limpiar_servicios() -> Iterator[None]:
    """Reinicia los servicios con memoria antes y despues de cada prueba.

    `obtener_servicio_cag` y `obtener_servicio_rag` devuelven un unico
    objeto por proceso. Sin este reinicio, el contexto de CAG y el indice
    de RAG que creo una prueba se filtran a la siguiente, y el resultado
    depende del orden de ejecucion.

    No es un detalle teorico: antes de agregar este fixture, dos pruebas
    de RAG fallaban de forma intermitente segun si habia corrido antes la
    suite de la API.
    """
    _reiniciar()
    yield
    _reiniciar()


def _reiniciar() -> None:
    from app.services.cag import reiniciar_servicio_cag
    from app.services.rag import reiniciar_servicio_rag

    # Se importan adentro para no arrastrar los servicios al collection.
    reiniciar_servicio_cag()
    reiniciar_servicio_rag()
