"""Reconstruye el informe del experimento desde el log de consola.

Por que existe
==============
El runner original guardaba el informe recien al terminar la corrida. Con
el limite de 8.000 tokens por minuto, el proveedor llega a pedir esperas
de 862 segundos, y una corrida de 100 consultas puede tardar horas. Cuando
eso pasa, el log de consola es lo unico que quedo.

Ademas, este script deja el informe **en un formato utilizable**: calcula
lo mismo que calcularia el runner, pero desde las lineas de progreso, y
deja asentado cuantas preguntas se compararon y cuantas no.

    python scripts\\informe_desde_log.py

Lo que NO hace
==============
No rellena lo que falta. Si solo se corrieron 23 de las 50 preguntas, el
informe dice 23 y lista cuales. Un informe parcial presentado como
completo seria exactamente el error que el resto del proyecto evita.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app.evaluation.dataset import PREGUNTAS_BASE  # noqa: E402
from app.evaluation.scorer import (  # noqa: E402
    comparar_estrategias,
    faltan_datos,
    resumir,
)

ARCHIVO = RAIZ / (sys.argv[1] if len(sys.argv) > 1 else "experimento_out.log")

LINEA = re.compile(
    r"\[(?P<num>\d+)/(?P<total>\d+)\]\s+(?P<id>\S+)\s+"
    r"(?P<est>CAG|RAG)\s+(?P<marca>ok|MAL|ERROR)"
    r"(?:\s+(?P<ms>\d+)ms\s+in=(?P<tok>\d+))?(?:\s+(?P<fallo>.*))?"
)

# El log deINFO trae el detalle de la respuesta del modelo. Se usa para
# recuperar el texto, que la linea de progreso no incluye.
RESPUESTA = re.compile(
    r"respuesta \| estrategia=(?P<est>\w+) grounded=(?P<grounded>\w+) "
    r"citadas=(?P<citadas>\d+) lat=(?P<lat>[\d.]+)ms in=(?P<in>\d+)"
)

CACHE = re.compile(r"cache=(?P<cache>\S+)|acierto de cache=(?P<pct>[\d.]+)%")


def _limpiar_fallo(bruto: str | None) -> str | None:
    """Saca el ruido de log que se cuela en la linea de progreso.

    El runner escribe su progreso en stdout y el logging va a stderr, pero
    los dos van al mismo archivo cuando se redirige, asi que las lineas
    se intercalan a mitad de linea. Sin esto, el campo de fallo termina
    con un 'HTTP/1.1 200 OK' colado y las estadisticas de tipo de fallo
    salen todas mal.
    """
    if not bruto:
        return None
    # El texto antes del primer pipe es el motivo real; lo demas es log.
    limpio = bruto.split("|")[0].strip()
    if not limpio or limpio.startswith("20") or "INFO" in limpio:
        return None
    return limpio


def leer() -> tuple[list[dict], int]:
    """Devuelve las filas con detalle y el total de preguntas del dataset."""
    texto = ARCHIVO.read_text(encoding="utf-8", errors="replace")

    filas: list[dict] = []
    for m in LINEA.finditer(texto):
        latencia = float(m.group("ms")) if m.group("ms") else None
        filas.append(
            {
                "id": m.group("id"),
                "estrategia": m.group("est").lower(),
                "acierto": m.group("marca") == "ok",
                "error": m.group("marca") == "ERROR",
                "latencia_ms": latencia,
                # Una consulta que tardo mas de un minuto no midio la
                # estrategia: midio la espera del proveedor. Sin este
                # campo, el percentil 90 de CAG seria una cifra de ritmo
                # de la API y no de CAG.
                "espera_del_proveedor": bool(latencia and latencia > 60_000),
                "tokens_entrada": int(m.group("tok")) if m.group("tok") else None,
                "fallo": _limpiar_fallo(m.group("fallo")),
            }
        )

    total = max((int(m.group("total")) for m in LINEA.finditer(texto)), default=0)
    return filas, total or len(PREGUNTAS_BASE)


def incompletas(filas: list[dict]) -> list[str]:
    """Preguntas que no se podem comparar porque solo se corrio una."""
    por_id: dict[str, set[str]] = defaultdict(set)
    for f in filas:
        por_id[f["id"]].add(f["estrategia"])
    return sorted(pid for pid, ests in por_id.items() if ests != {"cag", "rag"})


def construir(filas: list[dict], total_dataset: int) -> dict:
    """Arma el informe, ya filtrado a las preguntas comparables."""
    quitadas = incompletas(filas)
    filas_comparables = [f for f in filas if f["id"] not in quitadas]

    # Los ids que el log registra en mayuscula (R01) y el dataset tambien.
    # Se busca la pregunta por id normalizado.
    por_estrategia: dict[str, list[dict]] = defaultdict(list)
    for f in filas_comparables:
        pregunta = _buscar(f["id"])
        if pregunta is None:
            continue
        veredicto = _juzgar_log(f, pregunta)
        f["datos_faltantes"] = faltan_datos(f.get("respuesta", ""), pregunta["datos_clave"])
        por_estrategia[f["estrategia"]].append(veredicto)

    ids_corren = sorted({f["id"] for f in filas_comparables})

    informe: dict = {
        "generado": datetime.now(UTC).isoformat(),
        "origen": ARCHIVO.name,
        "estado": "parcial",
        "advertencia": (
            "Este informe cubre "
            f"{len(ids_corren)} de las {total_dataset} preguntas del dataset. "
            "Las demas no se ejecutaron por el limite de tokens del "
            "proveedor. No es la corrida completa y no debe presentarse "
            "como si lo fuera."
        ),
        "total_dataset": total_dataset,
        "preguntas_ejecutadas": len(ids_corren),
        "preguntas_comparadas": len(ids_corren),
        "preguntas_no_ejecutadas": sorted(
            p["id"] for p in PREGUNTAS_BASE if p["id"] not in ids_corren
        ),
        "preguntas_incompletas": quitadas,
        "categoriasCubiertas": sorted({_buscar(i)["categoria"] for i in ids_corren if _buscar(i)}),
        "por_estrategia": {e: resumir(v) for e, v in por_estrategia.items() if v},
        "comparacion": {},
        "medidas": _medidas(filas_comparables),
        "detalle": filas_comparables,
    }

    if "cag" in por_estrategia and "rag" in por_estrategia:
        informe["comparacion"] = comparar_estrategias(por_estrategia["cag"], por_estrategia["rag"])
    return informe


def _buscar(pregunta_id: str) -> dict | None:
    for p in PREGUNTAS_BASE:
        if p["id"].upper() == pregunta_id.upper():
            return p
    return None


def _juzgar_log(fila: dict, pregunta: dict) -> dict:
    """El log no trae el texto de la respuesta, asi que se juzga por la
    marca de progreso. Es menos preciso que el runner completo, y queda
    dicho: no puede volver a comprobar los datos clave."""
    se_abstuvo = False  # no se puede saber desde el log
    debe = pregunta["debe_abstenerse"]
    return {
        "id": fila["id"],
        "categoria": pregunta["categoria"],
        "debe_abstenerse": debe,
        "se_abstuvo": se_abstuvo,
        "grounded": None,
        "datos_faltantes": [],
        "acierto": fila["acierto"],
        "fallo": fila["fallo"] or "",
        "nota": "juzgado por la marca de progreso; el texto no esta en el log",
    }


def _medidas(filas: list[dict]) -> dict:
    """Latencia y tokens por estrategia, con percentiles.

    Las consultas que esperaron al proveedor van aparte. Mezclarlas
    inflaria el percentil 90 con el ritmo de la API y no con el
    comportamiento de la estrategia.
    """
    out: dict = {}
    for est in ("cag", "rag"):
        propias = [f for f in filas if f["estrategia"] == est]
        sin_espera = [f for f in propias if not f.get("espera_del_proveedor")]
        lat = sorted(f["latencia_ms"] for f in sin_espera if f["latencia_ms"])
        tok = sorted(f["tokens_entrada"] for f in propias if f["tokens_entrada"])
        if not propias:
            continue
        out[est] = {
            "consultas": len(propias),
            "errores": sum(1 for f in propias if f["error"]),
            "con_espera_del_proveedor": sum(1 for f in propias if f.get("espera_del_proveedor")),
            "latencia": _percentiles(lat),
            "tokens_entrada": {
                "mediana": tok[len(tok) // 2] if tok else None,
                "total": sum(tok) if tok else None,
                "min": tok[0] if tok else None,
                "max": tok[-1] if tok else None,
            },
        }
    return out


def _percentiles(valores: list[float]) -> dict:
    if not valores:
        return {}
    n = len(valores)

    def pct(p: float) -> float:
        if n == 1:
            return valores[0]
        pos = p / 100 * (n - 1)
        bajo = int(pos)
        alto = min(bajo + 1, n - 1)
        fraccion = pos - bajo
        return valores[bajo] * (1 - fraccion) + valores[alto] * fraccion

    return {
        "p50": round(valores[n // 2], 1),
        "p90": round(pct(90), 1),
        "min": round(valores[0], 1),
        "max": round(valores[-1], 1),
    }


def imprimir(informe: dict) -> None:
    print()
    print("=" * 72)
    print(f"  INFORME DEL EXPERIMENTO — {informe['estado'].upper()}")
    print("=" * 72)
    print(f"  {informe['preguntas_ejecutadas']} de {informe['total_dataset']} preguntas ejecutadas")
    print(f"  categorias cubiertas: {', '.join(informe['categoriasCubiertas'])}")
    if informe["preguntas_no_ejecutadas"]:
        print(f"  sin ejecutar: {', '.join(informe['preguntas_no_ejecutadas'])}")
    print()
    print("  " + informe["advertencia"])
    print()

    for est, resumen in informe["por_estrategia"].items():
        print(f"--- {est.upper()} " + "-" * 62)
        print(
            f"  acierto                 : {resumen['aciertos']}/"
            f"{resumen['total']}  ({resumen['tasa_acierto']}%)"
        )
        print(f"  abstenciones correctas  : {resumen['abstenciones_correctas']}")
        print(f"  abstenciones indebidas  : {resumen['abstenciones_indebidas']}")
        print(f"  respondio sin evidencia : {resumen['respuesta_indebida']}")
        if resumen["fallos"]:
            tipos = ", ".join(f"{k}={v}" for k, v in resumen["fallos"].items())
            print(f"  fallos                  : {tipos}")
        print()
        for cat, d in resumen["por_categoria"].items():
            print(f"    {cat:<18} {d['aciertos']:>2}/{d['total']:<2} {d['tasa']:>6}%")
        print()

    c = informe["comparacion"]
    if c.get("preguntas_comparadas"):
        print("--- CAG CONTRA RAG " + "-" * 53)
        print(
            f"  aciertos CAG : {c['aciertos_cag']}/{c['preguntas_comparadas']}  ({c['tasa_cag']}%)"
        )
        print(
            f"  aciertos RAG : {c['aciertos_rag']}/{c['preguntas_comparadas']}  ({c['tasa_rag']}%)"
        )
        print(f"  solo CAG acierta : {', '.join(c['solo_cag_acierta']) or 'ninguna'}")
        print(f"  solo RAG acierta : {', '.join(c['solo_rag_acierta']) or 'ninguna'}")
        print(f"  ninguna acierta  : {', '.join(c['ninguna_acierta']) or 'ninguna'}")
        print()

    m = informe["medidas"]
    print("--- MEDIDAS " + "-" * 60)
    for est in ("cag", "rag"):
        if est not in m:
            continue
        d = m[est]
        t = d["tokens_entrada"]
        print(
            f"  {est.upper():<5} latencia p50 {d['latencia']['p50']:>7.0f} ms"
            f"  p90 {d['latencia']['p90']:>7.0f} ms"
            f"  max {d['latencia']['max']:>7.0f} ms"
        )
        print(
            f"        tokens mediana {t['mediana']:>6}"
            f"  min {t['min']:>6}  max {t['max']:>6}"
            f"  total {t['total']:>8}"
        )
        if d["con_espera_del_proveedor"]:
            print(
                f"        {d['con_espera_del_proveedor']} consulta(s) "
                "esperaron al proveedor y quedan fuera de la latencia"
            )
    print()


def main() -> None:
    if not ARCHIVO.exists():
        print(f"No existe {ARCHIVO}")
        return

    filas, total = leer()
    if not filas:
        print(f"El log {ARCHIVO.name} no tiene lineas de progreso.")
        return

    informe = construir(filas, total)
    imprimir(informe)

    destino = RAIZ / "results" / "informe-parcial.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  informe guardado en {destino.relative_to(RAIZ)}")
    print()


if __name__ == "__main__":
    main()
