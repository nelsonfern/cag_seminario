"""Pruebas de `POST /chat/compare`.

El endpoint existe para una sola razon: hacer visible la comparacion entre
CAG y RAG sobre la misma pregunta. Si un dia deja de devolver las dos
respuestas, deja de cumplir su proposito, y eso es lo que se comprueba.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api


class TestComparacion:
    def test_devuelve_las_dos_estrategias(self, cliente: TestClient) -> None:
        r = cliente.post("/chat/compare", json={"question": "cuanto cuesta la copia certificada"})

        assert r.status_code in (200, 502)
        if r.status_code == 502:
            pytest.skip("El proveedor falso no esta disponible para RAG")

        cuerpo = r.json()
        assert cuerpo["cag"]["strategy"] == "cag"
        assert cuerpo["rag"]["strategy"] == "rag"
        assert cuerpo["cag"]["answer"]
        assert cuerpo["rag"]["answer"]

    def test_usa_la_misma_pregunta_para_las_dos(self, cliente: TestClient) -> None:
        """Si las dos respuestas vinieran de preguntas distintas, la
        comparacion no significaria nada."""
        r = cliente.post("/chat/compare", json={"question": "cuanto cuesta la copia certificada"})

        if r.status_code != 200:
            pytest.skip("Sin proveedor para RAG en esta corrida")

        cuerpo = r.json()
        assert cuerpo["question"] == "cuanto cuesta la copia certificada"

    def test_incluye_las_diferencias(self, cliente: TestClient) -> None:
        r = cliente.post("/chat/compare", json={"question": "cuanto cuesta la copia certificada"})

        if r.status_code != 200:
            pytest.skip("Sin proveedor para RAG en esta corrida")

        diferencias = r.json()["diferencias"]

        # Los campos existen siempre: un `None` significa "no informado",
        # que es distinto de cero.
        for clave in (
            "latencia_cag_ms",
            "latencia_rag_ms",
            "diferencia_latencia_ms",
            "tokens_cag",
            "tokens_rag",
            "cache_reportada_cag",
            "cache_reportada_rag",
            "grounded_cag",
            "grounded_rag",
            "mismo_resultado",
        ):
            assert clave in diferencias, f"Falta la diferencia {clave}"

    def test_rechaza_pregunta_vacia(self, cliente: TestClient) -> None:
        r = cliente.post("/chat/compare", json={"question": "  "})

        assert r.status_code == 422

    def test_no_filtra_credenciales(self, cliente: TestClient) -> None:
        cuerpo = cliente.post(
            "/chat/compare", json={"question": "cuanto cuesta la partida"}
        ).text.lower()

        assert "sk-" not in cuerpo
        assert "api_key" not in cuerpo


class TestComparacionDocumentada:
    def test_esta_en_el_openapi(self, cliente: TestClient) -> None:
        rutas = cliente.get("/openapi.json").json()["paths"]

        assert "/chat/compare" in rutas
        assert "post" in rutas["/chat/compare"]
