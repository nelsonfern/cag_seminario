"""Runner del experimento.

Ejecuta el mismo conjunto de preguntas con CAG y con RAG, en el mismo
proceso y con el mismo modelo, y guarda el detalle de cada respuesta.

Reglas que gobiernan el diseño:

- **Ningun resultado se inventa.** Si una consulta falla, se registra el
  fallo con su motivo y no se cuenta como respuesta.
- **Las ejecuciones frias y calientes se separan.** El cache del proveedor
  se construye a lo largo de la primera consulta, asi que promediar todo
  junto daria una tasa de acierto que no corresponde a ninguna consulta.
- **Se respeta el limite del proveedor.** Con 8.000 tokens por minuto, las
  consultas tienen que pausarse o la corrida termina en errores 429.
"""

from __future__ import annotations

import asyncio
import csv
import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger, redact
from app.models.schemas import Estrategia
from app.repositories.evaluation_repository import EvaluationRepository
from app.services.agent import Agente

logger = get_logger("evaluation")


@dataclass
class Metrica:
    """Resultado de una pregunta con una estrategia."""

    pregunta_id: str
    estrategia: str
    respuesta: str
    grounded: bool
    es_abstencion: bool
    latencia_ms: float
    tokens_entrada: int | None
    tokens_salida: int | None
    tokens_cache: int | None
    cache_reportada: bool
    error: str | None = None

    @property
    def acierto_cache(self) -> float | None:
        """Porcentaje de acierto de cache, o None si no se reporto."""
        if not self.cache_reportada or not self.tokens_entrada or not self.tokens_cache:
            return None
        return 100 * self.tokens_cache / self.tokens_entrada


def _percentiles(valores: list[float]) -> dict[str, float]:
    """Percentiles de latencia.

    La mediana sola no alcanza: con pocos datos puede esconder una cola
    lenta, y en un experimento la cola es la que se va a notar.
    """
    if not valores:
        return {}
    ordenados = sorted(valores)
    n = len(ordenados)

    def pct(p: float) -> float:
        if n == 1:
            return ordenados[0]
        posicion = p / 100 * (n - 1)
        bajo = int(posicion)
        alto = min(bajo + 1, n - 1)
        fraccion = posicion - bajo
        return ordenados[bajo] * (1 - fraccion) + ordenados[alto] * fraccion

    return {
        "p50": round(statistics.median(ordenados), 1),
        "p90": round(pct(90), 1),
        "p95": round(pct(95), 1),
        "min": round(ordenados[0], 1),
        "max": round(ordenados[-1], 1),
    }


def resumir_ejecucion(metricas: list[Metrica]) -> dict[str, Any]:
    """Agrega las metricas de una corrida, separando frias de calientes."""
    resumen: dict[str, Any] = {}

    for estrategia in ("cag", "rag"):
        propias = [m for m in metricas if m.estrategia == estrategia]
        if not propias:
            resumen[estrategia] = {"total": 0}
            continue

        latencias = [m.latencia_ms for m in propias]
        entradas = [m.tokens_entrada for m in propias if m.tokens_entrada]
        salidas = [m.tokens_salida for m in propias if m.tokens_salida]
        aciertos = [m.acierto_cache for m in propias if m.acierto_cache is not None]

        # La primera consulta es la frie: el cache aun no existe.
        frias = propias[:1]
        calientes = propias[1:]

        resumen[estrategia] = {
            "total": len(propias),
            "con_error": sum(1 for m in propias if m.error),
            "latencia": _percentiles(latencias),
            "tokens_entrada_total": sum(entradas) if entradas else None,
            "tokens_entrada_mediana": (statistics.median(entradas) if entradas else None),
            "tokens_salida_total": sum(salidas) if salidas else None,
            "grounded": sum(1 for m in propias if m.grounded),
            "abstenciones": sum(1 for m in propias if m.es_abstencion),
            "acierto_cache_promedio": round(statistics.mean(aciertos), 1) if aciertos else None,
            "consultas_con_cache": len(aciertos),
            # Separadas porque no son comparables entre si.
            "fria": {
                "latencia_ms": round(frias[0].latencia_ms, 1) if frias else None,
                "tokens_entrada": frias[0].tokens_entrada if frias else None,
                "acierto_cache": frias[0].acierto_cache if frias else None,
            },
            "calientes": {
                "total": len(calconientes := calientes),
                "latencia": _percentiles([m.latencia_ms for m in calientes]),
                "acierto_cache_promedio": (
                    round(
                        statistics.mean(
                            [a for a in (m.acierto_cache for m in calientes) if a is not None]
                        ),
                        1,
                    )
                    if any(m.acierto_cache is not None for m in calientes)
                    else None
                ),
            },
        }
        del calconientes

    return resumen


def exportar_csv(metricas: list[Metrica], ruta: Path, ejecutable: str) -> Path:
    """Escribe el detalle de la corrida en CSV, una fila por respuesta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.DictWriter(
            f,
            fieldnames=[
                "pregunta_id",
                "estrategia",
                "ejecutable",
                "respuesta",
                "grounded",
                "abstencion",
                "latencia_ms",
                "tokens_entrada",
                "tokens_salida",
                "tokens_cache",
                "acierto_cache_pct",
                "error",
            ],
        )
        escritor.writeheader()
        for m in metricas:
            escritor.writerow(
                {
                    "pregunta_id": m.pregunta_id,
                    "estrategia": m.estrategia,
                    "ejecutable": ejecutable,
                    "respuesta": m.resposta(),
                    "grounded": m.grounded,
                    "abstencion": m.es_abstencion,
                    "latencia_ms": round(m.latencia_ms, 1),
                    "tokens_entrada": m.tokens_entrada,
                    "tokens_salida": m.tokens_salida,
                    "tokens_cache": m.tokens_cache,
                    "acierto_cache_pct": (
                        round(m.acierto_cache, 1) if m.acierto_cache is not None else None
                    ),
                    "error": m.error,
                }
            )
    return ruta


def _resposta(self: Metrica) -> str:
    """Texto de la respuesta en una sola linea, para el CSV."""
    return " ".join(self.respuesta.split())


Metrica.resposta = _resposta


async def ejecutar_experimento(
    preguntas: list[dict[str, Any]] | None = None,
    limite: int | None = None,
    pausa: float = 16.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Ejecuta el experimento completo y devuelve el resumen."""
    from app.evaluation.dataset import cargar_dataset

    settings = get_settings()
    repo = EvaluationRepository(settings.resolve_path(settings.sqlite_path))

    if preguntas is None:
        preguntas = cargar_dataset()
    if limite:
        preguntas = preguntas[:limite]

    if not preguntas:
        raise ValueError("No hay preguntas para evaluar.")

    identificador = run_id or repo.iniciar_ejecucion(
        modelo=settings.model_name,
        proveedor=settings.llm_provider,
        configuracion={
            "preguntas": len(preguntas),
            "cag_max_context_tokens": settings.cag_max_context_tokens,
            "rag_top_k": settings.rag_top_k,
            "pausa_s": pausa,
            "ejecutado": datetime.now(UTC).isoformat(),
        },
    )

    agente = Agente()
    metricas: list[Metrica] = []

    # Se prepara CAG antes de empezar: su cache de aplicacion no debe
    # contarse como trabajo de la primera consulta.
    from app.services.cag import obtener_servicio_cag

    obtener_servicio_cag().preparar(force=True)

    logger.info("Experimento %s: %d preguntas x 2 estrategias", identificador, len(preguntas))

    for indice, pregunta in enumerate(preguntas, start=1):
        for estrategia in (Estrategia.CAG, Estrategia.RAG):
            metrica = await _ejecutar_una(agente, pregunta, estrategia)
            metricas.append(metrica)

            repo.registrar_resultado(
                identificador,
                pregunta_id=pregunta["id"],
                estrategia=estrategia.value,
                respuesta=metrica.respuesta,
                groundtruth=pregunta.get("referencia", ""),
                grounded=metrica.grounded,
                es_abstencion=metrica.es_abstencion,
                latencia_ms=metrica.latencia_ms,
                tokens_entrada=metrica.tokens_entrada,
                tokens_salida=metrica.tokens_salida,
                tokens_cache=metrica.tokens_cache,
                cache_reportada=metrica.cache_reportada,
                metricas={"acierto_cache_pct": metrica.acierto_cache},
                detalle={"error": metrica.error} if metrica.error else {},
            )

            print(
                f"  [{indice}/{len(preguntas)}] {pregunta['id']:<6} "
                f"{estrategia.value.upper():<4} "
                f"grounded={metrica.grounded!s:<5} {metrica.latencia_ms:>7.0f}ms "
                f"in={metrica.tokens_entrada} "
                f"{'ERROR: ' + redact(metrica.error) if metrica.error else ''}"
            )

            # El limite de tokens por minuto del proveedor obliga a pausar.
            # Sin esto, la mitad de la corrida falla con 429.
            await asyncio.sleep(pausa)

    resumen = resumir_ejecucion(metricas)
    csv_path = exportar_csv(
        metricas,
        settings.results_dir / f"resultados-{identificador}.csv",
        settings.model_name,
    )
    (settings.results_dir / f"resumen-{identificador}.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    repo.finalizar_ejecucion(
        identificador,
        "completada",
        resumen,
        str(csv_path.relative_to(settings.results_dir.parent)),
    )
    return {"run_id": identificador, "resumen": resumen, "csv": str(csv_path)}


async def _ejecutar_una(
    agente: Agente,
    pregunta: dict[str, Any],
    estrategia: Estrategia,
) -> Metrica:
    """Ejecuta una pregunta con una estrategia y mide el resultado."""
    from app.services.llm import LLMError

    try:
        respuesta = await agente.responder(pregunta["texto"], estrategia)
    except LLMError as exc:
        # Un fallo se registra como fallo. No se inventa una respuesta.
        return Metrica(
            pregunta_id=pregunta["id"],
            estrategia=estrategia.value,
            respuesta="",
            grounded=False,
            es_abstencion=False,
            latencia_ms=0.0,
            tokens_entrada=None,
            tokens_salida=None,
            tokens_cache=None,
            cache_reportada=False,
            error=redact(str(exc)),
        )

    uso = respuesta.token_usage
    datos = respuesta.datos_estructurados or {}

    return Metrica(
        pregunta_id=pregunta["id"],
        estrategia=estrategia.value,
        respuesta=respuesta.answer,
        grounded=respuesta.grounded,
        es_abstencion=bool(datos.get("es_abstencion")),
        latencia_ms=respuesta.latency_ms,
        tokens_entrada=uso.tokens_entrada if uso else None,
        tokens_salida=uso.tokens_salida if uso else None,
        tokens_cache=uso.tokens_cache_lectura if uso else None,
        cache_reportada=bool(uso and uso.cache_disponible),
    )


def escribir_csv_directo(metricas: list[Metrica], ruta: Path) -> Path:
    """Atajo para exportar sin ejecutar nada."""
    return exportar_csv(metricas, ruta, get_settings().model_name)


def _t() -> float:
    return time.perf_counter()
