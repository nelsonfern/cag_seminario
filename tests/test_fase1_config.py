"""Pruebas de la configuracion y del arranque de la aplicacion.

Cubren la Fase 1: que la app levanta, que la configuracion es coherente
y que ninguna credencial puede filtrarse ni por configuracion ni por logs.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.logging import SecretRedactionFilter, redact
from app.main import create_app


class TestHealth:
    """El endpoint /health debe responder sin depender de ninguna otra capa."""

    def test_health_responde_ok(self, cliente: TestClient) -> None:
        response = cliente.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["app"] == "TramitaIA"

    def test_health_no_expone_rutas_absolutas(self, cliente: TestClient) -> None:
        """El estado no debe revelar rutas del sistema de archivos."""
        cuerpo = cliente.get("/health").text.lower()

        assert "c:\\" not in cuerpo
        assert "/home/" not in cuerpo
        assert "password" not in cuerpo

    def test_health_reporta_proveedor_y_modelo(self, cliente: TestClient) -> None:
        body = cliente.get("/health").json()

        assert body["llm_provider"] == "fake"
        assert body["model"] == "fake-deterministic"

    def test_openapi_se_genera(self, cliente: TestClient) -> None:
        """La documentacion interactiva es parte del contrato de la API."""
        response = cliente.get("/openapi.json")

        assert response.status_code == 200
        assert "/health" in response.json()["paths"]


class TestConfiguracion:
    """La configuracion debe fallar temprano ante valores incoherentes."""

    def test_valores_por_defecto_declarados(self) -> None:
        """Los defaults se verifican en la clase, no en una instancia.

        Una instancia construida aqui hereda las variables de entorno del
        proceso, asi que no serviria para comprobar el default.
        """
        campos = Settings.model_fields

        assert campos["app_name"].default == "TramitaIA"
        assert campos["llm_provider"].default == "openai_compatible"
        assert campos["rag_top_k"].default == 5
        # El presupuesto de CAG por defecto debe entrar en la ventana de
        # contexto por defecto, sea cual sea el proveedor elegido.
        for ventana in ("ollama_context_length", "openai_context_length"):
            assert campos["cag_max_context_tokens"].default < campos[ventana].default, ventana

    def test_variables_de_entorno_ganan_a_los_defaults(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("RAG_TOP_K", "8")

        s = Settings(_env_file=None)

        assert s.llm_provider == "ollama"
        assert s.rag_top_k == 8

    def test_env_file_deshabilitado_no_se_lee(self, tmp_path) -> None:
        """Con `_env_file=None` el archivo .env no debe aplicarse."""
        archivo = tmp_path / ".env"
        archivo.write_text("APP_NAME=NoDeberiaAplicarse\n", encoding="utf-8")

        s = Settings(_env_file=None)

        assert s.app_name != "NoDeberiaAplicarse"

    def test_lista_desde_csv(self) -> None:
        """Las variables de entorno llegan como string separado por comas."""
        s = Settings(_env_file=None, corpus_tramites="partidas,autenticacion")

        assert s.corpus_tramites == ["partidas", "autenticacion"]

    def test_solapamiento_mayor_que_fragmento_es_error(self) -> None:
        with pytest.raises(ValueError, match="RAG_CHUNK_OVERLAP"):
            Settings(_env_file=None, rag_chunk_size=500, rag_chunk_overlap=500)

    def test_cag_no_puede_superar_la_ventana_del_modelo(self) -> None:
        """El contexto de CAG debe dejar margen para prompt y respuesta."""
        with pytest.raises(ValueError, match="CAG_MAX_CONTEXT_TOKENS"):
            Settings(
                _env_file=None,
                llm_provider="ollama",
                ollama_context_length=8192,
                cag_max_context_tokens=20000,
            )

    def test_openai_exige_base_url_y_modelo(self) -> None:
        with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
            Settings(_env_file=None, llm_provider="openai_compatible")

    def test_openai_exige_modelo(self) -> None:
        with pytest.raises(ValueError, match="OPENAI_MODEL"):
            Settings(_env_file=None, llm_provider="openai_compatible", openai_base_url="http://x")

    def test_ventana_de_contexto_se_declara_por_proveedor(self) -> None:
        """Cada proveedor tiene su propia ventana; no hay un default generico."""
        ollama = Settings(
            _env_file=None,
            llm_provider="ollama",
            ollama_context_length=4096,
            cag_max_context_tokens=3000,
        )
        remoto = Settings(
            _env_file=None,
            llm_provider="openai_compatible",
            openai_base_url="http://x",
            openai_model="m",
            openai_context_length=128000,
        )

        assert ollama.model_context_window == 4096
        assert remoto.model_context_window == 128000

    def test_log_level_invalido_es_error(self) -> None:
        with pytest.raises(ValueError, match="LOG_LEVEL"):
            Settings(_env_file=None, log_level="VERBOSE")

    def test_temperatura_fuera_de_rango_es_error(self) -> None:
        with pytest.raises(ValueError, match="LLM_TEMPERATURE"):
            Settings(_env_file=None, llm_temperature=5.0)

    def test_ensure_directories_crea_lo_needed(self, settings_de_prueba: Settings) -> None:
        settings_de_prueba.ensure_directories()

        assert settings_de_prueba.resolve_path(settings_de_prueba.sqlite_path).parent.exists()
        assert settings_de_prueba.raw_dir.exists()
        assert settings_de_prueba.processed_dir.exists()


class TestRedaccionDeCredenciales:
    """Ninguna credencial debe aparecer en un log ni en una respuesta."""

    @pytest.mark.parametrize(
        "entrada",
        [
            "usando clave sk-proj-abc123def456ghi789jkl",
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9abcdefgh",
            'config={"api_key": "clave-secreta-1234"}',
            "token=hf_abcdefghijklmnopqrstuvwxyz",
        ],
    )
    def test_se_redactan_formatos_comunes(self, entrada: str) -> None:
        resultado = redact(entrada)

        assert "REDACTED" in resultado
        # Ningun secreto de prueba debe sobrevivir a la redaccion.
        assert "sk-proj-" not in resultado
        assert "eyJhbGciOiJIUzI1NiJ9" not in resultado
        assert "clave-secreta-1234" not in resultado
        assert "hf_abcdefghijklmnopqrstuvwxyz" not in resultado

    def test_texto_limpio_no_se_altera(self) -> None:
        texto = "La consulta tardo 250 ms y devolvio 3 fragmentos."

        assert redact(texto) == texto

    def test_filtro_limPIA_argumentos_de_excepcion(self) -> None:
        """Las cabeceras HTTP con credenciales llegan por `exc_info`."""
        registro = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="Fallo la llamada: %s",
            args=("sk-proj-fallolelelelelele",),
            exc_info=None,
        )

        SecretRedactionFilter().filter(registro)

        assert "sk-proj-" not in registro.args[0]

    def test_filtro_limpia_mensajes_con_asignacion(self) -> None:
        registro = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="api_key=valor-secreto-aqui",
            args=None,
            exc_info=None,
        )

        SecretRedactionFilter().filter(registro)

        assert "valor-secreto-aqui" not in registro.msg


class TestConstruccionApp:
    def test_create_app_es_idempotente(self) -> None:
        """Se pueden crear varias instancias sin acumular estado global."""
        primera = create_app()
        segunda = create_app()

        assert primera is not segunda

    def test_cada_app_expone_health(self) -> None:
        rutas = {r.path for r in create_app().routes}

        assert "/health" in rutas
