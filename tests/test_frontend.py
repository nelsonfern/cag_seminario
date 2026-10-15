"""Pruebas del frontend servido por la API.

Se verifica que los tres archivos se sirvan y, sobre todo, que **no se
sirva nada mas**. Un backend estatico mal configurado puede terminar
expadiendo el `.env` con la clave de API.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api


class TestFrontend:
    def test_sirve_la_pagina(self, cliente: TestClient) -> None:
        r = cliente.get("/")

        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_sirve_los_estilos(self, cliente: TestClient) -> None:
        r = cliente.get("/styles.css")

        assert r.status_code == 200
        assert "text/css" in r.headers["content-type"]
        assert "--primario" in r.text

    def test_sirve_el_script(self, cliente: TestClient) -> None:
        r = cliente.get("/app.js")

        assert r.status_code == 200
        assert "javascript" in r.headers["content-type"]
        assert "/chat" in r.text


class TestFrontendSeguro:
    """La superficie estatica no debe exponer el proyecto."""

    @pytest.mark.parametrize(
        "ruta",
        [
            "/.env",
            "/../.env",
            "/data/processed/corpus.json",
            "/requirements.txt",
            "/app/main.py",
            "/pyproject.toml",
        ],
    )
    def test_no_sirve_archivos_del_proyecto(self, cliente: TestClient, ruta: str) -> None:
        r = cliente.get(ruta)

        assert r.status_code in (404, 307, 308), f"{ruta} devolvio {r.status_code}"

    def test_no_filtra_la_clave_en_la_pagina(self, cliente: TestClient) -> None:
        """La pagina no puede traer nada del `.env`."""
        cuerpo = cliente.get("/").text.lower()

        assert "api_key" not in cuerpo
        assert "sk-" not in cuerpo


class TestFrontendContenido:
    """El enunciado pide pantallas concretas; se comprueba que existen."""

    @pytest.mark.parametrize(
        "selector",
        [
            'name="estrategia"',  # selector CAG/RAG
            'id="conversacion"',  # historial
            'id="pregunta"',  # campo de entrada
            'id="ejemplos"',  # preguntas de ejemplo
            'id="metricas-cuerpo"',  # panel de metricas
            'id="estado"',  # estado del servicio
        ],
    )
    def test_la_pagina_tiene_los_elementos(self, cliente: TestClient, selector: str) -> None:
        assert selector in cliente.get("/").text

    def test_el_script_consume_la_api(self, cliente: TestClient) -> None:
        """El frontend tiene que pegarle a los endpoints correctos."""
        script = cliente.get("/app.js").text

        assert "/chat" in script
        assert "/health" in script

    def test_el_script_maneja_errores(self, cliente: TestClient) -> None:
        """Un error del proveedor no puede romper la interfaz."""
        script = cliente.get("/app.js").text

        assert "catch" in script
        assert "No se pudo contactar al servidor" in script
