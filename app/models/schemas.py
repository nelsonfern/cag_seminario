"""Esquemas compartidos del proyecto.

Este modulo concentra los modelos de dominio que usan varios servicios.
Los modelos especificos de la API (peticiones y respuestas HTTP) viven en
`app/api/routes.py` para no mezclar capas.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TramitaIABaseModel(BaseModel):
    """Base comun de los modelos del proyecto.

    `extra="forbid"` hace que un campo no declarado sea un error explicito
    en lugar de ignorarse en silencio. Importa para detectar que un documento
    trae metadatos que el pipeline no esta contemplating.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# -----------------------------------------------------------------------------
# Vocabulario documental
# -----------------------------------------------------------------------------
class TipoDocumento(StrEnum):
    """Formato de origen del documento."""

    PDF = "pdf"
    HTML = "html"
    TXT = "txt"
    DOCX = "docx"
    DESCONOCIDO = "desconocido"


class EstadoVigencia(StrEnum):
    """Vigencia del documento.

    Existe `DESCONOCIDA` a proposito: si la fuente no declara cuando entra en
    vigor o deja de estar vigente, se registra como desconocida. No se infiere
    que un documento sigue vigente solo por estar publicado.
    """

    VIGENTE = "vigente"
    NO_VIGENTE = "no_vigente"
    PARCIAL = "parcial"
    DESCONOCIDA = "desconocida"


class Estrategia(StrEnum):
    """Estrategia de generacion aumentada."""

    CAG = "cag"
    RAG = "rag"


class EstadoFuente(StrEnum):
    """Estado de verificacion de una fuente en el manifiesto."""

    PENDIENTE = "pendiente"
    VERIFICADA = "verificada"
    NO_ACCESIBLE = "no_accesible"
    INCORPORADA_MANUAL = "incorporada_manual"
    OBSERVADA = "obsoleta"


# -----------------------------------------------------------------------------
# Metadatos documentales
# -----------------------------------------------------------------------------
class MetadatosDocumento(TramitaIABaseModel):
    """Metadatos de un documento del corpus.

    Los campos de fecha son opcionales a proposito: una fuente puede no
    declararlos. Lo que no se conoce se deja en `None` y se refleja como
    "desconocido" en las respuestas del agente.
    """

    # --- Identificacion ---
    # Identificador estable, derivado del contenido. No cambia entre ejecuciones.
    document_id: str = Field(description="Identificador estable del documento")
    titulo: str = Field(description="Titulo del documento")
    organismo: str = Field(description="Organismo responsable de la informacion")
    tramite: str = Field(description="Tramite al que pertenece el documento")
    tipo_documento: TipoDocumento = Field(
        default=TipoDocumento.DESCONOCIDO, description="Formato de origen"
    )

    # --- Procedencia ---
    url_origen: str | None = Field(
        default=None, description="URL de origen, si el documento se descargo de la web"
    )
    archivo_original: str = Field(description="Nombre original del archivo")
    fecha_consulta: date = Field(description="Fecha en que se consulto la fuente")
    fecha_publicacion: date | None = Field(
        default=None, description="Fecha de publicacion declarada por la fuente"
    )
    fecha_actualizacion: date | None = Field(
        default=None, description="Fecha de actualizacion declarada por la fuente"
    )
    vigencia: EstadoVigencia = Field(
        default=EstadoVigencia.DESCONOCIDA,
        description="Vigencia declarada por la fuente; desconocida si no se declara",
    )

    # --- Trazabilidad tecnica ---
    hash_contenido: str = Field(description="SHA-256 del contenido normalizado")
    estado_verificacion: EstadoFuente = Field(
        default=EstadoFuente.VERIFICADA, description="Estado de verificacion de la fuente"
    )
    observaciones: str | None = Field(
        default=None, description="Observaciones sobre la fuente o su obtencion"
    )


class DocumentoProcesado(TramitaIABaseModel):
    """Documento con su contenido ya normalizado."""

    metadatos: MetadatosDocumento
    contenido: str = Field(description="Texto normalizado completo")
    secciones: list[str] = Field(
        default_factory=list, description="Encabezados detectados en el documento"
    )


class Fragmento(TramitaIABaseModel):
    """Fragmento del documento, base del indice de RAG."""

    fragmento_id: str = Field(description="Identificador estable del fragmento")
    document_id: str = Field(description="Documento al que pertenece")
    tramite: str
    organismo: str
    contenido: str
    indice: int = Field(description="Posicion del fragmento dentro del documento")
    seccion: str | None = Field(
        default=None, description="Seccion o encabezado donde comienza el fragmento"
    )
    url_origen: str | None = None
    titulo: str = ""


class Cita(TramitaIABaseModel):
    """Referencia a un documento efectivamente utilizado en una respuesta."""

    document_id: str
    titulo: str
    organismo: str
    url_origen: str | None = None
    tramite: str
    # Fragmento o seccion concreta que respalda la afirmacion, si se conoce.
    fragmento_id: str | None = None
    seccion: str | None = None


# -----------------------------------------------------------------------------
# Respuesta del LLM
# -----------------------------------------------------------------------------
class UsoTokens(TramitaIABaseModel):
    """Consumo de tokens.

    Todos los campos son opcionales: cada proveedor reporta una combinacion
    distinta. Lo que un proveedor no informa se queda en None en lugar de
    estimarse o inventarse.
    """

    tokens_entrada: int | None = None
    tokens_salida: int | None = None
    tokens_totales: int | None = None

    # Metricas de prompt caching informadas por el proveedor.
    # None significa "el proveedor no lo reporta", no cero.
    tokens_cache_lectura: int | None = Field(
        default=None, description="Tokens servidos desde la cache del proveedor"
    )
    tokens_cache_escritura: int | None = Field(
        default=None, description="Tokens escritos en la cache del proveedor"
    )
    cache_disponible: bool = Field(
        default=False, description="True solo si el proveedor reporto metricas de cache"
    )

    @property
    def costo_estimado_usd(self) -> float | None:
        """Costo en USD si se conocen los precios unitarios del proveedor.

        Deliberadamente devuelve None. El precio por token cambia y depende
        del modelo; calcularlo con valores hardcodeados seria inventar una
        metrica. Se implementara en la fase de evaluacion, leyendo precios
        de una configuracion explicita.
        """
        return None


class RespuestaLLM(TramitaIABaseModel):
    """Respuesta cruda del proveedor de modelo de lenguaje."""

    texto: str
    modelo: str
    proveedor: str
    uso: UsoTokens = Field(default_factory=UsoTokens)
    latencia_ms: float = Field(description="Latencia total de la llamada")
    # Metadatos del proveedor para trazar la ejecucion del experimento.
    metadatos: dict[str, Any] = Field(default_factory=dict)


class RespuestaAgente(TramitaIABaseModel):
    """Respuesta que devuelve el agente, ya enrichecida con fuentes."""

    answer: str
    strategy: Estrategia
    sources: list[Cita] = Field(default_factory=list)
    # True cuando la respuesta se apoya en evidencia documental.
    grounded: bool = Field(description="False si el agente se abstuvo por falta de evidencia")
    latency_ms: float
    token_usage: UsoTokens | None = None
    warnings: list[str] = Field(default_factory=list)
    # Respuesta estructurada, util para el dataset de evaluacion.
    datos_estructurados: dict[str, Any] | None = None
