"""Configuracion central de TramitaIA.

Toda la configuracion se lee de variables de entorno o del archivo `.env`.
No hay ninguna credencial escrita en el codigo: este modulo solo lee
nombres de variables, nunca valores secretos.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Ruta raiz del proyecto (dos niveles arriba de app/core) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuracion de la aplicacion, cargada desde el entorno."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_` esta reservado por Pydantic; campos como `model_name` lo rompen.
        protected_namespaces=(),
    )

    # -------------------------------------------------------------------------
    # Aplicacion
    # -------------------------------------------------------------------------
    app_name: str = "TramitaIA"
    log_level: str = "INFO"
    sqlite_path: Path = Path("data/tramitaia.db")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"]
    )

    # -------------------------------------------------------------------------
    # Proveedor de LLM
    # -------------------------------------------------------------------------
    llm_provider: Literal["fake", "ollama", "openai_compatible"] = "openai_compatible"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_context_length: int = 32768

    # --- Endpoint estilo OpenAI ---
    # OpenRouter es el proveedor recomendado: una sola clave, muchos modelos
    # y exposicion de las metricas de prompt caching que pide el enunciado.
    # Se deja un valor por defecto para que importar el paquete no dependa
    # de que el `.env` exista. Sin modelo, la API responde con un 401 claro.
    openai_base_url: str = "https://api.groq.com/openai/v1"
    openai_api_key: str = ""
    # Sin default a proposito: el modelo debe elegirse de forma explicita.
    # Fijarlo por codigo impide reproducir una ejecucion concreta.
    openai_model: str = ""
    # Ventana de contexto declarada del modelo remoto.
    # Se declara a mano porque cada modelo tiene la suya, y el presupuesto de
    # CAG se valida contra este valor. Un default inflado dejaria pasar
    # contextos que el proveedor rechaza en produccion.
    openai_context_length: int = 32768

    # --- Generacion ---
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_max_output_tokens: int = 1024
    llm_request_timeout_s: float = 120.0
    llm_max_retries: int = 2

    # --- Prompt caching del proveedor ---
    # Solo se activa si el proveedor lo expone de forma documentada.
    # Si el proveedor no reporta metricas de cache, se registra como no disponible.
    # Nunca se inventan valores de cache.
    llm_prompt_caching_enabled: bool = False

    # -------------------------------------------------------------------------
    # CAG
    # -------------------------------------------------------------------------
    cag_max_context_tokens: int = 6000
    cag_cache_dir: Path = Path("data/processed/cag_cache")
    cag_allow_partitioning: bool = True

    # -------------------------------------------------------------------------
    # RAG
    # -------------------------------------------------------------------------
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_chunk_size: int = 900
    rag_chunk_overlap: int = 150
    rag_top_k: int = 5
    vector_store: Literal["chroma", "faiss"] = "chroma"
    chroma_dir: Path = Path("data/processed/chroma")
    faiss_index_path: Path = Path("data/processed/faiss/index.faiss")

    # -------------------------------------------------------------------------
    # Ingesta documental
    # -------------------------------------------------------------------------
    corpus_tramites: list[str] = Field(default_factory=lambda: ["partidas"])
    fetch_respect_robots: bool = True
    fetch_user_agent: str = "TramitaIA/0.1 (proyecto academico)"
    fetch_delay_s: float = 1.0

    # -------------------------------------------------------------------------
    # Validaciones
    # -------------------------------------------------------------------------
    @field_validator("cors_origins", "corpus_tramites", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Acepta tanto una lista como un string separado por comas.

        Las variables de entorno siempre llegan como string.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in allowed:
            raise ValueError(
                f"LOG_LEVEL invalido: {value!r}. Valores permitidos: {sorted(allowed)}"
            )
        return normalized

    @field_validator("llm_temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"LLM_TEMPERATURE debe estar entre 0.0 y 2.0, recibio {value}")
        return value

    @field_validator("rag_chunk_overlap")
    @classmethod
    def _validate_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"RAG_CHUNK_OVERLAP no puede ser negativo, recibio {value}")
        return value

    @model_validator(mode="after")
    def _validate_dependencies(self) -> Settings:
        """Verifica coherencia entre campos relacionados."""
        # El solapamiento debe ser menor que el tamano del fragmento,
        # si no el fragmentado no avanza.
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({self.rag_chunk_overlap}) debe ser menor que "
                f"RAG_CHUNK_SIZE ({self.rag_chunk_size})"
            )

        if self.rag_top_k < 1:
            raise ValueError(f"RAG_TOP_K debe ser al menos 1, recibio {self.rag_top_k}")

        # El presupuesto de contexto de CAG debe dejar lugar para el prompt
        # del sistema y para la respuesta generada.
        # Solo se valida si hay un proveedor usable: sin configuracion no
        # existe una ventana real con la cual comparar.
        if self.llm_configurado and self.cag_max_context_tokens >= self.model_context_window:
            raise ValueError(
                f"CAG_MAX_CONTEXT_TOKENS ({self.cag_max_context_tokens}) no puede superar "
                f"la ventana de contexto del modelo ({self.model_context_window}). "
                "Dejese margen para el prompt y la respuesta."
            )

        return self

    # -------------------------------------------------------------------------
    # Propiedades derivadas
    # -------------------------------------------------------------------------
    @property
    def llm_configurado(self) -> bool:
        """Indica si el proveedor elegido tiene lo minimo para funcionar.

        Importar el paquete no debe fallar por falta de configuracion:
        la app tiene que poder arrancar y reportar el problema en `/health`,
        no romperse al importar.
        """
        if self.llm_provider == "openai_compatible":
            return bool(self.openai_base_url and self.openai_api_key and self.openai_model)
        if self.llm_provider == "ollama":
            return bool(self.ollama_base_url and self.ollama_model)
        return True

    @property
    def model_context_window(self) -> int:
        """Ventana de contexto efectiva del modelo configurado."""
        if self.llm_provider == "ollama":
            return self.ollama_context_length
        if self.llm_provider == "openai_compatible":
            return self.openai_context_length
        # El proveedor `fake` no tiene ventana real: se declara una comoda.
        return 8192

    @property
    def base_url(self) -> str:
        """URL base del proveedor de LLM seleccionado."""
        if self.llm_provider == "ollama":
            return self.ollama_base_url.rstrip("/")
        return self.openai_base_url.rstrip("/")

    @property
    def model_name(self) -> str:
        """Identificador del modelo configurado."""
        if self.llm_provider == "ollama":
            return self.ollama_model
        if self.llm_provider == "openai_compatible":
            return self.openai_model
        return "fake-deterministic"

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed"

    @property
    def results_dir(self) -> Path:
        return PROJECT_ROOT / "results"

    @property
    def corpus_manifest_path(self) -> Path:
        return PROJECT_ROOT / "data" / "manifest.json"

    def resolve_path(self, path: Path) -> Path:
        """Convierte una ruta relativa a absoluta respecto del proyecto."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    def ensure_directories(self) -> None:
        """Crea los directorios que la aplicacion necesita en runtime."""
        for directory in (
            self.resolve_path(self.sqlite_path).parent,
            self.raw_dir,
            self.processed_dir,
            self.resolve_path(self.cag_cache_dir),
            self.resolve_path(self.chroma_dir),
            self.results_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la configuracion cacheada.

    Se cachea para releer `.env` una sola vez por proceso.
    En las pruebas se usa `get_settings.cache_clear()`.
    """
    return Settings()


settings = get_settings()
