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
