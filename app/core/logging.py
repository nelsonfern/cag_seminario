"""Configuracion de logging con filtrado de credenciales.

El proyecto registra llamadas al proveedor de LLM, latencias y consumo de
tokens. Eso implica que un log poderia terminar volcando una clave de API.
Este modulo instala un filtro que reemplaza cualquier valor sensible
antes de que llegue al archivo o a la consola.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable
from typing import Any

from app.core.config import settings

# Prefijos de variables de entorno que contienen credenciales.
_SECRET_ENV_PREFIXES = ("OPENAI_API_KEY", "OLLAMA_API_KEY", "ANTHROPIC_API_KEY")

_REDACTED = "***REDACTED***"

# Cada regla es (patron, plantilla de reemplazo).
# La plantilla permite conservar el nombre de la clave, que ayuda a
# diagnosticar, sustituyendo solo el valor: `api_key=***REDACTED***`.
_SECRET_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Cabecera Authorization: Bearer <token>
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{12,}"), r"\1 " + _REDACTED),
    # Formatos tipo OpenAI: sk-proj-..., sk-...
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}"), _REDACTED),
    # Token de Hugging Face: hf_...
    (re.compile(r"\bhf_[A-Za-z0-9]{12,}"), _REDACTED),
    # Asignaciones tipo api_key=..., "api_key": "...", access_token: ...
    # Se admiten comillas alrededor del nombre de la clave porque el caso
    # habitual es un JSON o un cabecero con diccionario.
    (
        re.compile(
            r"(?i)(['\"]?\b(?:api[_\-]?key|access[_\-]?token|secret|authorization)\b"
            r"['\"]?\s*[=:]\s*['\"]?)([A-Za-z0-9._\-]{8,})(['\"]?)"
        ),
        r"\1" + _REDACTED + r"\3",
    ),
)


def _build_secret_values() -> list[str]:
    """Recopila los valores de credenciales presentes en el entorno.

    Se lee el entorno y no `settings`, porque el valor puede haber sido
    inyectado directamente en la variable sin pasar por el archivo .env.
    """
    import os

    values: list[str] = []
    for name in _SECRET_ENV_PREFIXES:
        value = os.environ.get(name, "").strip()
        # Solo los valores suficientemente largos para no filtrar ruido.
        if len(value) >= 8:
            values.append(value)
    return values


def redact(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Reemplaza credenciales dentro de un texto."""
    result = text
    for secret in [*_build_secret_values(), *extra_secrets]:
        if secret:
            result = result.replace(secret, _REDACTED)
    for pattern, replacement in _SECRET_RULES:
        result = pattern.sub(replacement, result)
    return result


class SecretRedactionFilter(logging.Filter):
    """Filtro de logging que elimina credenciales de los registros.

    Actua sobre el mensaje ya formateado, de modo que cubre tanto el mensaje
    del log como los argumentos de excepcion, que suelen ser la via por la
    que una clave se filtra sin querer.
    """

    def __init__(self, extra_secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._extra_secrets = list(extra_secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg, self._extra_secrets)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: redact(value, self._extra_secrets) if isinstance(value, str) else value
                    for key, value in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact(value, self._extra_secrets) if isinstance(value, str) else value
                    for value in record.args
                )

        # `exc_info` ya renderizado puede contener cabeceras con credenciales.
        if record.exc_text:
            record.exc_text = redact(record.exc_text, self._extra_secrets)

        return True


def configure_logging(level: str | None = None) -> None:
    """Configura el logging de la aplicacion.

    Idempotente: se puede llamar varias veces sin duplicar handlers.
    """
    resolved_level = (level or settings.log_level).upper()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    # El filtro se aplica en el handler y no en el logger raiz para cubrir
    # tambien los logs de librerias de terceros (httpx, chromadb, ...).
    handler.addFilter(SecretRedactionFilter())

    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Reemplaza handlers previos para no duplicar lineas al recargar.
    for existing in list(root.handlers):
        root.removeHandler(existing)

    root.addHandler(handler)

    # Estas librerias son ruidosas a nivel DEBUG.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger con el prefijo del proyecto."""
    return logging.getLogger(f"tramitaia.{name}")


def log_extra(**fields: Any) -> dict[str, Any]:
    """Construye el diccionario de contexto para un registro.

    Filtra los valores None para no ensuciar la salida.
    """
    return {key: value for key, value in fields.items() if value is not None}
