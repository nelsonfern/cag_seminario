"""Corre el experimento completo y escribe el informe.

Fase 8 del seminario. Ejecuta las 50 preguntas con CAG y con RAG, en el
mismo proceso, con el mismo modelo y sobre el mismo corpus, y despues
juzga cada respuesta con el evaluador de `scorer.py`.

Uso:

    python scripts\\correr_experimento.py                 # las 50
    python scripts\\correr_experimento.py --limite 5      # prueba rapida
    python scripts\\correr_experimento.py --solo-cag      # sin RAG

LIMITE DEL PROVEEDOR
====================
El plan gratuito de Groq permite 8.000 tokens por minuto. Cada consulta
de CAG manda 3.952 tokens, asi que entran unas dos por minuto y la
tercera tiene que esperar. Con 50 preguntas x 2 estrategias son 100
consultas, y el ritmo lo marca el proveedor, no este script.

Los fallos no se rellenan: si una consulta falla, queda registrada como
fallo con su motivo y cuenta como fallo en el resumen.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.evaluation.dataset import PREGUNTAS_BASE, cargar_dataset
from app.evaluation.scorer import (
    comparar_estrategias,
    juzgar,
    resumir,
)
from app.models.schemas import Estrategia
from app.services.agent import Agente

logger = get_logger("experimento")

# Pausa entre consultas. Con 3.952 tokens por llamada de CAG y un limite
# de 8.000 por minuto, hay que dejar respirar al proveedor.
PAUSA = 18.0


async def ejecutar(
    preguntas: list[dict[str, Any]],
    estrategias: list[Estrategia],
    pausa: float,
) -> dict[str, Any]:
    agente = Agente()

    # CAG se prepara antes de empezar: el costo de construir el contexto
    # no tiene que contaminar la primera medicion.
    from app.services.cag import obtener_servicio_cag
    from app.services.rag import obtener_servicio_rag

    obtener_servicio_cag().preparar(force=True)
    obtener_servicio_rag().recuperar("calentamiento")

    resultados: list[dict[str, Any]] = []
    veredictos: dict[str, list[dict[str, Any]]] = {e.value: [] for e in estrategias}

    for indice, pregunta in enumerate(preguntas, start=1):
        for estrategia in estrategias:
            fila = await _una(agente, pregunta, estrategia)
            resultados.append(fila)

            if fila["error"]:
                print(
                    f"  [{indice}/{len(preguntas)}] {pregunta['id']:<4} "
                    f"{estrategia.value.upper():<4} ERROR: {fila['error'][:60]}",
                    flush=True,
                )
            else:
                v = _juzgar(pregunta, fila)
                veredictos[estrategia.value].append(v)
                marca = "ok " if v["acierto"] else "MAL"
                print(
                    f"  [{indice}/{len(preguntas)}] {pregunta['id']:<4} "
                    f"{estrategia.value.upper():<4} {marca} "
                    f"{fila['latencia_ms']:>7.0f}ms "
                    f"in={fila['tokens_entrada']} "
                    f"{v['fallo'][:44]}",
                    flush=True,
                )

            # Se guarda despues de cada consulta, no al final.
            #
            # El proveedor limita a 8.000 tokens por minuto y avisa que
            # espere cientos de segundos. Una corrida completa puede
            # tardar horas y, si se corta, sin esto se pierde todo lo
            # hecho. Un informe parcial es un informe real: dice cuantas
            # consultas se hicieron y cuales fueron.
            _guardar_parcial(resultados, veredictos, estrategias, len(preguntas))

            await asyncio.sleep(pausa)

    informe: dict[str, Any] = {
        "generado": datetime.now(UTC).isoformat(),
        "modelo": get_settings().model_name,
        "proveedor": get_settings().llm_provider,
        "total_preguntas": len(preguntas),
        "estrategias": [e.value for e in estrategias],
        "por_estrategia": {e.value: resumir(veredictos[e.value]) for e in estrategias},
        "comparacion": {},
        "detalle": resultados,
        "veredictos": veredictos,
    }

    if len(estrategias) == 2:
        informe["comparacion"] = comparar_estrategias(veredictos["cag"], veredictos["rag"])

    return informe


async def _una(agente: Agente, pregunta: dict[str, Any], estrategia: Estrategia) -> dict[str, Any]:
    """Ejecuta una pregunta y guarda todo lo medido."""
    inicio = asyncio.get_event_loop().time()
    try:
        r = await agente.responder(pregunta["texto"], estrategia)
    except Exception as exc:
        logger.error("Fallo en %s/%s: %s", pregunta["id"], estrategia.value, exc)
        return {
            "id": pregunta["id"],
            "categoria": pregunta["categoria"],
            "estrategia": estrategia.value,
            "error": str(exc)[:200],
            "respuesta": "",
            "grounded": False,
            "latencia_ms": 0.0,
            "tokens_entrada": None,
            "tokens_cache": None,
            "tokens_salida": None,
        }

    latencia = (asyncio.get_event_loop().time() - inicio) * 1000
    uso = r.token_usage
    return {
        "id": pregunta["id"],
        "categoria": pregunta["categoria"],
        "estrategia": estrategia.value,
        "error": None,
        "respuesta": r.answer,
        "grounded": r.grounded,
        "latencia_ms": round(latencia, 1),
        "tokens_entrada": uso.tokens_entrada if uso else None,
        "tokens_cache": uso.tokens_cache_lectura if uso else None,
        "tokens_salida": uso.tokens_salida if uso else None,
        "cache_reportada": bool(uso and uso.cache_disponible),
        "fuentes": len(r.sources),
    }


def _juzgar(pregunta: dict[str, Any], fila: dict[str, Any]) -> dict[str, Any]:
    return juzgar(fila["respuesta"], fila["grounded"], pregunta)


def _guardar_parcial(
    resultados: list[dict[str, Any]],
    veredictos: dict[str, list[dict[str, Any]]],
    estrategias: list[Estrategia],
    total_preguntas: int,
) -> Path:
    """Escribe el informe a medias, despues de cada consulta.

    Se escribe siempre en el mismo archivo, `informe-parcial.json`, que
    se sobrescribe. Guardar por consulta sin diferencia entre versiones
    llenaria el disco de copias casi iguales.
    """
    informe: dict[str, Any] = {
        "generado": datetime.now(UTC).isoformat(),
        "modelo": get_settings().model_name,
        "proveedor": get_settings().llm_provider,
        "estado": "en_curso",
        "total_preguntas": total_preguntas,
        "consultas_hechas": len(resultados),
        "por_estrategia": {e.value: resumir(veredictos[e.value]) for e in estrategias},
        "detalle": resultados,
    }
    if {e.value for e in estrategias} == {"cag", "rag"}:
        informe["comparacion"] = comparar_estrategias(veredictos["cag"], veredictos["rag"])

    destino = get_settings().results_dir / "informe-parcial.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def imprimir_informe(informe: dict[str, Any]) -> None:
    print()
    print("=" * 74)
    print("  INFORME DEL EXPERIMENTO")
    print("=" * 74)
    print(f"  modelo   : {informe['modelo']}")
    print(f"  preguntas: {informe['total_preguntas']}")
    print(f"  cuando   : {informe['generado'][:19]}")
    print()

    for estrategia, resumen in informe["por_estrategia"].items():
        if not resumen["total"]:
            continue
        print(f"--- {estrategia.upper()} " + "-" * 66)
        print(
            f"  acierto                : {resumen['aciertos']}/{resumen['total']}"
            f"  ({resumen['tasa_acierto']}%)"
        )
        print(f"  abstenciones correctas: {resumen['abstenciones_correctas']}")
        print(f"  abstenciones indebidas : {resumen['abstenciones_indebidas']}")
        print(f"  respondio sin evidencia: {resumen['respuesta_indebida']}")
        if resumen["fallos"]:
            tipos = ", ".join(f"{k}={v}" for k, v in resumen["fallos"].items())
            print(f"  fallos                 : {tipos}")
        print()
        for categoria, datos in resumen["por_categoria"].items():
            extra = ""
            if "debe_abstenerse" in datos:
                extra = f"  (debía abstenerse en {datos['debe_abstenerse']})"
            print(
                f"    {categoria:<18} {datos['aciertos']:>2}/{datos['total']:<2}"
                f" {datos['tasa']:>5}%{extra}"
            )
        print()

    comparacion = informe.get("comparacion") or {}
    if comparacion.get("preguntas_comparadas"):
        print("--- CAG CONTRA RAG " + "-" * 57)
        print(
            f"  aciertos CAG : {comparacion['aciertos_cag']}/"
            f"{comparacion['preguntas_comparadas']}  ({comparacion['tasa_cag']}%)"
        )
        print(
            f"  aciertos RAG : {comparacion['aciertos_rag']}/"
            f"{comparacion['preguntas_comparadas']}  ({comparacion['tasa_rag']}%)"
        )
        print(f"  solo CAG acierta: {comparacion['solo_cag_acierta'] or 'ninguna'}")
        print(f"  solo RAG acierta: {comparacion['solo_rag_acierta'] or 'ninguna'}")
        print(f"  ninguna acierta : {comparacion['ninguna_acierta'] or 'ninguna'}")
        print()

    for estrategia in informe["estrategias"]:
        filas = [f for f in informe["detalle"] if f["estrategia"] == estrategia]
        if not filas:
            continue
        lat = sorted(f["latencia_ms"] for f in filas)
        tok = [f["tokens_entrada"] for f in filas if f["tokens_entrada"]]
        cache = [
            100 * f["tokens_cache"] / f["tokens_entrada"]
            for f in filas
            if f.get("cache_reportada") and f["tokens_cache"] and f["tokens_entrada"]
        ]
        n = len(lat)
        print(
            f"  {estrategia.upper():<5} latencia mediana {lat[n // 2]:>7.0f} ms"
            f" | tokens mediana {sorted(tok)[len(tok) // 2] if tok else 0:>5}"
            f" | acerto de cache"
            f" {round(sum(cache) / len(cache), 1) if cache else 'n/d'}%"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento CAG contra RAG")
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--pausa", type=float, default=PAUSA)
    parser.add_argument("--solo-cag", action="store_true")
    args = parser.parse_args()

    configure_logging()
    preguntas = cargar_dataset() or PREGUNTAS_BASE
    if args.limite:
        preguntas = preguntas[: args.limite]

    estrategias = [Estrategia.CAG]
    if not args.solo_cag:
        estrategias.append(Estrategia.RAG)

    print()
    print(f"Experimento: {len(preguntas)} preguntas x {len(estrategias)} estrategias")
    print(f"Pausa entre consultas: {args.pausa} s")
    print("El limite del proveedor es de 8.000 tokens por minuto; esto va a tardar.")
    print()

    informe = asyncio.run(ejecutar(preguntas, estrategias, args.pausa))
    imprimir_informe(informe)

    destino = (
        get_settings().results_dir / f"informe-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    # Los veredictos no van al JSON: ya estan Installados en el resumen.
    informe.pop("veredictos", None)
    destino.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"informe guardado en {destino}")
    print()


if __name__ == "__main__":
    main()
