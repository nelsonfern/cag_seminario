"""Endpoints de la API.

El enunciado pide seis: estado, fuentes, chat, ejecucion del experimento
y lectura de resultados.

Dos reglas atraviesan todo el modulo:

1. **No se exponen credenciales ni datos internos innecesarios.** Un
   error del proveedor vuelve con su mensaje, pero sin la clave.
2. **Un fallo tecnico se distingue de una abstencion.** Si el proveedor
   no responde, el codigo HTTP lo dice. Devolver 200 con un texto que
   parece una respuesta del agente seria mentirle al cliente.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings
from app.core.logging import get_logger, redact
from app.models.schemas import Cita, Estrategia, RespuestaAgente, UsoTokens
from app.repositories.document_repository import DocumentoRepository
from app.repositories.evaluation_repository import EvaluationRepository
from app.services.agent import Agente
from app.services.corpus import Manifiesto

logger = get_logger("api")

router = APIRouter()


# ---------------------------------------------------------------------------
# Esquemas de entrada y salida
# ---------------------------------------------------------------------------
class PeticionChat(BaseModel):
    """Consulta del ciudadano."""

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Pregunta en español sobre un trámite público.",
    )
    strategy: Estrategia = Field(
        default=Estrategia.CAG,
        description="Estrategia de generación aumentada a usar.",
    )
    guardar: bool = Field(
        default=True,
        description="Si se registra la consulta y sus métricas en SQLite.",
    )

    @field_validator("question")
    @classmethod
    def _no_vacia(cls, valor: str) -> str:
        limpio = valor.strip()
        if not limpio:
            raise ValueError("La pregunta no puede estar vacía.")
        return limpio


class ComparacionRespuesta(BaseModel):
    """Resultado de correr la misma pregunta con las dos estrategias.

    Es la pieza que hace visible la comparacion: mismo corpus, mismo
    modelo, misma pregunta, distinta estrategia.
    """

    question: str
    cag: RespuestaChat
    rag: RespuestaChat
    diferencias: dict[str, Any]


@router.post(
    "/chat/compare",
    tags=["agente"],
    summary="Compara CAG y RAG sobre la misma pregunta",
)
async def comparar(peticion: PeticionChat) -> ComparacionRespuesta:
    """Corre la misma pregunta con CAG y con RAG y devuelve ambas.

    **Advertencia de costo**: cada estrategia consume tokens de entrada
    completos. Con el plan gratuito de 8000 tokens por minuto, las dos
    juntas pueden agotar el presupuesto y obligar a esperar al
    proveedor. El aviso viaja en la respuesta para que el cliente lo muestre.
    """
    agente = _agente()

    respuestas = {}
    for estrategia in (Estrategia.CAG, Estrategia.RAG):
        r = await agente.responder(peticion.question, estrategia)
        if (r.datos_estructurados or {}).get("error_proveedor"):
            raise HTTPException(
                status_code=502,
                detail=redact(r.warnings[0] if r.warnings else "Error del proveedor"),
            )
        respuestas[estrategia] = _a_respuesta_chat(r)

    return ComparacionRespuesta(
        question=peticion.question,
        cag=respuestas[Estrategia.CAG],
        rag=respuestas[Estrategia.RAG],
        diferencias=_comparar(respuestas[Estrategia.CAG], respuestas[Estrategia.RAG]),
    )


def _comparar(cag: RespuestaChat, rag: RespuestaChat) -> dict[str, Any]:
    """Resume, en pocas cifras, en que se diferenciaron las dos respuestas."""

    def valor(chat: RespuestaChat, campo: str) -> Any:
        uso = chat.token_usage
        if not uso:
            return None
        return getattr(uso, campo, None)

    tokens_cag = valor(cag, "tokens_entrada")
    tokens_rag = valor(rag, "tokens_entrada")

    # Si RAG reconstruyo el indice, su latencia incluye cargar el modelo
    # de embeddings. Eso no es caracteristica de la recuperacion.
    rag_frio = bool((rag.estrategia_datos or {}).get("indice_reconstruido"))

    return {
        "latencia_cag_ms": cag.latency_ms,
        "latencia_rag_ms": rag.latency_ms,
        "diferencia_latencia_ms": round(rag.latency_ms - cag.latency_ms, 1),
        "tokens_cag": tokens_cag,
        "tokens_rag": tokens_rag,
        "tokens_rag_menos_que_cag": (
            round(tokens_cag - tokens_rag, 1) if tokens_cag and tokens_rag else None
        ),
        "cache_cag": valor(cag, "tokens_cache_lectura"),
        "cache_rag": valor(rag, "tokens_cache_lectura"),
        "cache_reportada_cag": bool(cag.token_usage and cag.token_usage.cache_disponible),
        "cache_reportada_rag": bool(rag.token_usage and rag.token_usage.cache_disponible),
        "grounded_cag": cag.grounded,
        "grounded_rag": rag.grounded,
        "mismo_resultado": cag.grounded == rag.grounded,
        # Un valor None significa "el proveedor no lo informo", no cero.
        "acierto_cache_cag_pct": _acierto_cache(cag),
        "acierto_cache_rag_pct": _acierto_cache(rag),
        "arranque_en_frio": rag_frio,
        "notas": _notas_de_comparacion(cag, rag, rag_frio),
    }


def _notas_de_comparacion(cag: RespuestaChat, rag: RespuestaChat, rag_frio: bool) -> list[str]:
    """Explicita cuando un numero no es comparable con el otro.

    Mostrar la latencia de RAG con la carga del modelo de embeddings ya
    incluida haria creer que RAG es mas lento por su diseno, cuando es un
    costo de una sola vez por proceso.
    """
    notas: list[str] = []

    if rag_frio:
        notas.append(
            "La consulta de RAG reconstruyo el indice vectorial y cargo el "
            "modelo de embeddings: eso sumo a su tiempo y no es una "
            "caracteristica de la recuperacion. La siguiente consulta de "
            "RAG no lo vuelve a pagar."
        )

    if (cag.estrategia_datos or {}).get("truncado"):
        notas.append(
            "El contexto de CAG fue truncado para no exceder el limite de "
            "tokens: parte del corpus no llego al modelo."
        )

    return notas


def _acierto_cache(chat: RespuestaChat) -> float | None:
    uso = chat.token_usage
    if not uso or not uso.cache_disponible or not uso.tokens_entrada:
        return None
    return round(100 * (uso.tokens_cache_lectura or 0) / uso.tokens_entrada, 1)


class FuenteRespuesta(BaseModel):
    """Fuente usada en una respuesta.

    Deliberadamente no expone `contenido` ni `hash`: el cliente necesita
    saber de donde sale la respuesta, no leer el corpus entero.
    """

    document_id: str | None = None
    titulo: str
    organismo: str | None = None
    tramite: str | None = None
    url_origen: str | None = None


class RespuestaChat(BaseModel):
    """Respuesta del agente con su trazabilidad."""

    answer: str
    strategy: Estrategia
    sources: list[FuenteRespuesta] = Field(default_factory=list)
    grounded: bool
    latency_ms: float
    token_usage: UsoTokens | None = None
    warnings: list[str] = Field(default_factory=list)
    # Métricas propias de la estrategia: son el objeto del experimento.
    estrategia_datos: dict[str, Any] = Field(default_factory=dict)


class RespuestaFuentes(BaseModel):
    total: int
    fuentes: list[dict[str, Any]]


class RespuestaEjecucion(BaseModel):
    run_id: str
    estado: str
    created_at: str
    modelo: str
    total_resultados: int


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _agente() -> Agente:
    return Agente()


def _a_fuente(cita: Cita) -> FuenteRespuesta:
    return FuenteRespuesta(
        document_id=cita.document_id,
        titulo=cita.titulo,
        organismo=cita.organismo,
        tramite=cita.tramite,
        url_origen=cita.url_origen,
    )


def _a_respuesta_chat(r: RespuestaAgente) -> RespuestaChat:
    """Convierte la respuesta interna al contrato publico de la API.

    `datos_estructurados` se aplana: expone lo que el cliente necesita
    para mostrar avisos, y nada mas.
    """
    datos = r.datos_estructurados or {}
    avisos = list(r.warnings)

    if datos.get("es_abstencion") and r.grounded:
        avisos.append("La respuesta se abstuvo por falta de evidencia.")

    return RespuestaChat(
        answer=r.answer,
        strategy=r.strategy,
        sources=[_a_fuente(c) for c in r.sources],
        grounded=r.grounded,
        latency_ms=r.latency_ms,
        token_usage=r.token_usage,
        warnings=avisos,
        estrategia_datos=datos.get("estrategia", {}),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/health", tags=["sistema"], summary="Estado de la aplicación")
async def health() -> dict[str, Any]:
    """Estado de la aplicación, sin exponer datos sensibles."""
    settings = get_settings()
    configurado = settings.llm_configurado

    # Las claves estan siempre presentes, con None cuando no se conocen:
    # un cliente tiene que poder confiar en la forma de la respuesta.
    return {
        "status": "ok" if configurado else "degradado",
        "app": settings.app_name,
        "llm_provider": settings.llm_provider,
        "model": settings.model_name if configurado else None,
        "llm_configurado": configurado,
        "estrategias": [e.value for e in Estrategia],
        "corpus_documentos": DocumentoRepository(settings.processed_dir).informe()[
            "total_documentos"
        ],
    }


@router.get("/sources", tags=["corpus"], summary="Fuentes del corpus")
async def fuentes(
    tramite: str | None = Query(default=None, description="Filtra por trámite."),
) -> RespuestaFuentes:
    """Lista las fuentes incorporadas al corpus, con su estado."""
    repo = DocumentoRepository(get_settings().processed_dir)
    documentos = repo.por_tramite(tramite) if tramite else repo.cargar_entradas()

    lista = [
        {
            "document_id": d["metadatos"]["document_id"],
            "titulo": d["metadatos"]["titulo"],
            "organismo": d["metadatos"]["organismo"],
            "tramite": d["metadatos"]["tramite"],
            "url_origen": d["metadatos"].get("url_origen"),
            "estado_verificacion": d["metadatos"]["estado_verificacion"],
            "vigencia": d["metadatos"]["vigencia"],
            "fecha_consulta": d["metadatos"]["fecha_consulta"],
            "fecha_actualizacion": d["metadatos"].get("fecha_actualizacion"),
            "hash_contenido": d["metadatos"]["hash_contenido"][:12],
            "caracteres": len(d["contenido"]),
        }
        for d in documentos
    ]

    # Las fuentes no accesibles (por robots.txt) también se listan: saber
    # qué se intentó acceder y se bloqueó es parte de la trazabilidad.
    manifiesto = Manifiesto.cargar(repo.ruta_manifiesto)
    for fuente in manifiesto.fuentes:
        if fuente.get("document_id"):
            continue
        lista.append(
            {
                "document_id": None,
                "titulo": fuente.get("titulo") or fuente["url"],
                "organismo": fuente.get("organismo"),
                "tramite": fuente.get("tramite"),
                "url_origen": fuente["url"],
                "estado_verificacion": fuente.get("estado_verificacion"),
                "observaciones": fuente.get("observaciones"),
            }
        )

    return RespuestaFuentes(total=len(lista), fuentes=lista)


@router.post("/chat", tags=["agente"], summary="Consulta al agente")
async def chat(peticion: PeticionChat) -> RespuestaChat:
    """Responde una pregunta con CAG o RAG.

    Un error del proveedor devuelve 502, no 200 con un texto de error:
    un cliente no debe confundir un fallo tecnico con una abstención.
    """
    agente = _agente()
    respuesta = await agente.responder(peticion.question, peticion.strategy)

    if (respuesta.datos_estructurados or {}).get("error_proveedor"):
        raise HTTPException(
            status_code=502,
            detail=redact(respuesta.warnings[0] if respuesta.warnings else "Error del proveedor"),
        )

    if peticion.guardar:
        _guardar_consulta(peticion, respuesta)

    return _a_respuesta_chat(respuesta)


def _guardar_consulta(peticion: PeticionChat, respuesta: RespuestaAgente) -> None:
    """Registra la consulta y sus métricas.

    Un fallo al guardar no debe tumbar la consulta: el ciudadano ya
    recibio su respuesta y perderla seria peor que no registrarla.
    """
    try:
        repo = EvaluationRepository(get_settings().resolve_path(get_settings().sqlite_path))
        uso = respuesta.token_usage
        repo.registrar_consulta(
            pregunta=peticion.question,
            estrategia=peticion.strategy.value,
            respuesta=respuesta.answer,
            grounded=respuesta.grounded,
            es_abstencion=(respuesta.datos_estructurados or {}).get("es_abstencion", False),
            latencia_ms=respuesta.latency_ms,
            tokens_entrada=uso.tokens_entrada if uso else None,
            tokens_salida=uso.tokens_salida if uso else None,
            tokens_cache=uso.tokens_cache_lectura if uso else None,
            cache_reportada=bool(uso and uso.cache_disponible),
            modelo=get_settings().model_name,
            proveedor=respuesta.strategy.value,
            fuentes=[c.model_dump(mode="json") for c in respuesta.sources],
            advertencias=respuesta.warnings,
            detalle=respuesta.datos_estructurados or {},
        )
    except Exception:
        logger.warning("No se pudo registrar la consulta", exc_info=True)


@router.post(
    "/evaluation/run",
    tags=["evaluacion"],
    summary="Ejecuta el conjunto de pruebas",
    status_code=202,
)
async def evaluation_run() -> RespuestaEjecucion:
    """Lanza el experimento.

    Acepta con 202 y no con 200: el experimento tarda minutos por los
    limites del proveedor, asi que responder "hecho" mentiria.
    """

    settings = get_settings()
    repo = EvaluationRepository(settings.resolve_path(settings.sqlite_path))

    run_id = repo.iniciar_ejecucion(
        modelo=settings.model_name,
        proveedor=settings.llm_provider,
        configuracion={
            "corpus": settings.processed_dir.name,
            "cag_max_context_tokens": settings.cag_max_context_tokens,
            "rag_top_k": settings.rag_top_k,
        },
    )

    return RespuestaEjecucion(
        run_id=run_id,
        estado="en_curso",
        created_at=repo.obtener_ejecucion(run_id)["created_at"],
        modelo=settings.model_name,
        total_resultados=0,
    )


@router.get("/evaluation/results", tags=["evaluacion"], summary="Resultados guardados")
async def evaluation_results(limite: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    """Lista las ejecuciones registradas."""
    repo = EvaluationRepository(get_settings().resolve_path(get_settings().sqlite_path))
    return {"total": repo.contar_ejecuciones(), "ejecuciones": repo.listar_ejecuciones(limite)}


@router.get(
    "/evaluation/results/{run_id}",
    tags=["evaluacion"],
    summary="Detalle de una ejecución",
)
async def evaluation_result(run_id: str) -> dict[str, Any]:
    """Detalle completo de una ejecución del experimento."""
    repo = EvaluationRepository(get_settings().resolve_path(get_settings().sqlite_path))
    ejecucion = repo.obtener_ejecucion(run_id)

    if ejecucion is None:
        raise HTTPException(status_code=404, detail=f"No existe la ejecución {run_id}")

    return ejecucion


@router.get("/metrics/summary", tags=["sistema"], summary="Resumen de métricas")
async def resumen_metricas() -> dict[str, Any]:
    """Resumen de lo registrado hasta ahora, por estrategia."""
    repo = EvaluationRepository(get_settings().resolve_path(get_settings().sqlite_path))
    return _resumen_por_estrategia(repo)


def _resumen_por_estrategia(repo: EvaluationRepository) -> dict[str, Any]:
    """Agrega las consultas registradas por estrategia.

    Los promedios de latencia y tokens se calculan sobre lo realmente
    registrado. Si no hay datos, se informa `None` en vez de cero: cero
    significaria "todo fue instantaneo", que es un dato falso.
    """
    import statistics

    filas = repo.listar_consultas(limite=1000)
    resumen: dict[str, Any] = {"total_consultas": len(filas), "por_estrategia": {}}

    for estrategia in ("cag", "rag"):
        propias = [f for f in filas if f["estrategia"] == estrategia]
        if not propias:
            resumen["por_estrategia"][estrategia] = {"consultas": 0}
            continue

        latencias = [f["latencia_ms"] for f in propias]
        tokens = [f["tokens_entrada"] for f in propias if f["tokens_entrada"]]
        cache = [
            f["tokens_cache"]
            for f in propias
            if f["cache_reportada"] and f["tokens_cache"] is not None
        ]
        con_cache = [
            100 * f["tokens_cache"] / f["tokens_entrada"]
            for f in propias
            if f["cache_reportada"] and f["tokens_cache"] and f["tokens_entrada"]
        ]

        resumen["por_estrategia"][estrategia] = {
            "consultas": len(propias),
            "latencia_mediana_ms": round(statistics.median(latencias), 1),
            "tokens_entrada_mediana": statistics.median(tokens) if tokens else None,
            "tokens_entrada_total": sum(tokens) if tokens else None,
            "aciertos_cache": len(cache),
            "acierto_cache_promedio": (round(statistics.mean(con_cache), 1) if con_cache else None),
            "grounded": sum(1 for f in propias if f["grounded"]),
        }

    return resumen
