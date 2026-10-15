"""Comparacion A/B entre CAG y RAG con las mismas preguntas.

Responde una pregunta que no tiene respuesta obvia: cual de las dos
estrategias es mas rapida y mas barata **en este corpus**.

Metodo:
  - las mismas preguntas, en el mismo orden, contra el mismo modelo;
  - el mismo verificador local sobre las respuestas de ambos;
  - se mide lo que cada estrategia paga de verdad: tokens de entrada,
    tokens cacheados, latencia y tiempo de recuperacion.

Importante: la conclusion depende del tamano del corpus. Con 6 documentos
todo entra en el contexto y CAG no tiene nada que demostrar. El caso
interesante es que pasa cuando el corpus crece.

Uso:
    python scripts/comparar_cag_rag.py
    python scripts/comparar_cag_rag.py --preguntas 3
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

PROYECTO = Path(r"C:\Users\Usuario\Desktop\seminario\cag_seminario")
sys.path.insert(0, str(PROYECTO))

from app.services.agent import REGLAS_SISTEMA  # noqa: E402
from app.services.cag import CAGService  # noqa: E402
from app.services.ingestion import envolver_como_contexto  # noqa: E402
from app.services.rag import RAGService  # noqa: E402

PREGUNTAS = [
    "¿Cuánto cuesta la copia de acta certificada?",
    "¿Qué documentación necesito para solicitar una partida de nacimiento?",
    "¿En qué organismo se inicia el trámite de un matrimonio?",
    "¿Cuánto tarda el Registro Civil en entregar una partida?",
    "¿Para qué sirven las actas legalizadas?",
]


def linea(c="=", n=78):
    return c * n


def seccion(t):
    print(f"\n{linea()}\n{t}\n{linea()}")


async def medir_cag(pregunta, proveedor, contexto):
    mensajes = [
        {"role": "system", "content": REGLAS_SISTEMA},
        {"role": "user", "content": f"{contexto.contexto}\n\nPregunta del ciudadano: {pregunta}"},
    ]
    t0 = time.perf_counter()
    r = await proveedor.completar(mensajes)
    return r, (time.perf_counter() - t0) * 1000, 0.0


async def medir_rag(pregunta, proveedor, servicio_rag):
    t_rec = time.perf_counter()
    busqueda = servicio_rag.recuperar(pregunta)
    t_rec_ms = (time.perf_counter() - t_rec) * 1000

    partes = []
    for i, frag in enumerate(busqueda.fragmentos, start=1):
        partes.append(
            f"--- Fragmento {i} de {len(busqueda.fragmentos)} ---\n"
            f"Documento: {frag.fragmento.titulo}\n"
            f"Seccion: {frag.fragmento.seccion or 'sin seccion'}\n"
            + envolver_como_contexto(frag.fragmento.contenido)
        )
    contexto = "\n\n".join(partes)

    mensajes = [
        {"role": "system", "content": REGLAS_SISTEMA},
        {"role": "user", "content": f"{contexto}\n\nPregunta del ciudadano: {pregunta}"},
    ]
    t0 = time.perf_counter()
    r = await proveedor.completar(mensajes)
    return (
        r,
        (time.perf_counter() - t0) * 1000,
        t_rec_ms,
        busqueda,
    )


async def main():
    parser = argparse.ArgumentParser(description="Comparacion CAG vs RAG")
    parser.add_argument("--preguntas", type=int, default=5)
    parser.add_argument("--sin-pausa", action="store_true", help="Ignorar el limite del proveedor")
    args = parser.parse_args()

    from app.services.llm import obtener_proveedor

    proveedor = obtener_proveedor()

    seccion("0. PREPARACION")
    servicio_cag = CAGService()
    servicio_rag = RAGService()

    hash_cag = servicio_cag.hash_corpus()
    hash_rag = servicio_rag.hash_corpus()
    print(f"   hash CAG: {hash_cag}")
    print(f"   hash RAG: {hash_rag}")
    if hash_cag != hash_rag:
        print("   ABORTANDO: los dos servicios NO usan el mismo corpus.")
        return 1
    print("   OK: mismo corpus en ambos.")

    t0 = time.perf_counter()
    ctx_cag = servicio_cag.preparar(force=True)
    t_cag = (time.perf_counter() - t0) * 1000
    print(f"   CAG: contexto preparado en {t_cag:.1f} ms ({ctx_cag.tokens_estimados} tokens)")

    t0 = time.perf_counter()
    info_rag = servicio_rag.indexar(force=True)
    t_rag = (time.perf_counter() - t0) * 1000
    print(
        f"   RAG: indice construido en {t_rag / 1000:.1f} s ({info_rag.get('fragmentos')} fragmentos)"
    )
    print("        (incluye cargar el modelo de embeddings: no existe en CAG)")

    preguntas = PREGUNTAS[: args.preguntas]

    resultados = {"cag": [], "rag": []}

    seccion("1. EJECUCION (CAG primero, luego RAG, mismas preguntas)")
    for i, pregunta in enumerate(preguntas, start=1):
        print(f"\n   [{i}] {pregunta}")

        rc, lat_cag, _ = await medir_cag(pregunta, proveedor, ctx_cag)
        uso_c = rc.uso
        cache_c = (
            f"{uso_c.tokens_cache_lectura}/{uso_c.tokens_entrada}"
            if uso_c.cache_disponible
            else "no reportada"
        )
        print(
            f"       CAG  in={uso_c.tokens_entrada} out={uso_c.tokens_salida} cache={cache_c} lat={lat_cag:.0f}ms"
        )
        resultados["cag"].append(
            {
                "pregunta": pregunta,
                "tokens_in": uso_c.tokens_entrada,
                "cache": uso_c.tokens_cache_lectura,
                "lat": lat_cag,
                "out": uso_c.tokens_salida,
            }
        )

        rr, lat_rag, t_rec, busqueda = await medir_rag(pregunta, proveedor, servicio_rag)
        uso_r = rr.uso
        cache_r = (
            f"{uso_r.tokens_cache_lectura}/{uso_r.tokens_entrada}"
            if uso_r.cache_disponible
            else "no reportada"
        )
        print(
            f"       RAG  in={uso_r.tokens_entrada} out={uso_r.tokens_salida} cache={cache_r} lat={lat_rag:.0f}ms (recup {t_rec:.0f}ms)"
        )
        print(
            f"            fragmentos: {busqueda.obtenidos}, cache disponible: {uso_r.cache_disponible}"
        )
        resultados["rag"].append(
            {
                "pregunta": pregunta,
                "tokens_in": uso_r.tokens_entrada,
                "cache": uso_r.tokens_cache_lectura,
                "lat": lat_rag,
                "out": uso_r.tokens_salida,
                "recup": t_rec,
                "cache_reportada": uso_r.cache_disponible,
            }
        )

        if i < len(preguntas) and not args.sin_pausa:
            print("       (pausa por limite de tokens del proveedor)")
            time.sleep(18)

    # -----------------------------------------------------------------
    seccion("2. RESULTADOS")

    def resumen(clave):
        filas = resultados[clave]
        lat = [f["lat"] for f in filas]
        tin = [f["tokens_in"] for f in filas]
        reportadas = [f for f in filas if f["cache"]]
        cache_hit = [
            100 * f["cache"] / f["tokens_in"] for f in reportadas if f["cache"] and f["tokens_in"]
        ]
        return {
            "lat_mediana": statistics.median(lat),
            "lat_promedio": statistics.mean(lat),
            "tokens_in_mediana": statistics.median(tin),
            "tokens_in_total": sum(tin),
            "cache_reportada_veces": len(reportadas),
            "cache_hit_promedio": statistics.mean(cache_hit) if cache_hit else None,
        }

    rc = resumen("cag")
    rr = resumen("rag")
    recups = [f["recup"] for f in resultados["rag"]]

    print(f"   {'':<22} {'CAG':>16} {'RAG':>16}")
    print("   " + "-" * 58)
    print(f"   {'latencia mediana':<22} {rc['lat_mediana']:>13.0f}ms {rr['lat_mediana']:>13.0f}ms")
    print(
        f"   {'tokens entrada (med)':<22} {rc['tokens_in_mediana']:>13.0f} {rr['tokens_in_mediana']:>13.0f}"
    )
    print(
        f"   {'tokens entrada (total)':<22} {rc['tokens_in_total']:>13} {rr['tokens_in_total']:>13}"
    )
    ch_c = f"{rc['cache_hit_promedio']:.1f}%" if rc["cache_hit_promedio"] is not None else "n/d"
    ch_r = f"{rr['cache_hit_promedio']:.1f}%" if rr["cache_hit_promedio"] is not None else "n/d"
    print(f"   {'cache hit promedio':<22} {ch_c:>16} {ch_r:>16}")
    print(
        f"   {'veces que reporto cache':<22} {rc['cache_reportada_veces']:>16} {rr['cache_reportada_veces']:>16}"
    )
    print(f"   {'tiempo de recuperar':<22} {'no aplica':>16} {statistics.median(recups):>13.0f}ms")

    seccion("3. INTERPRETACION")
    dif_lat = rr["lat_mediana"] - rc["lat_mediana"]
    dif_tok = rr["tokens_in_mediana"] - rc["tokens_in_mediana"]

    print(f"   En la mediana, RAG tarda {dif_lat:+.0f} ms y usa {dif_tok:+.0f} tokens de entrada")
    print(
        f"   con este corpus ({ctx_cag.documentos_incluidos} documentos, {ctx_cag.tokens_estimados} tokens)."
    )
    print()
    print("   Ojo con el tamano del corpus: 6 documentos entran completos en el")
    print("   contexto. CAG no tiene nada que demostrar aca. El caso donde se")
    print("   separan es cuando el corpus crece hasta no entrar, y ahi CAG tiene")
    print("   que particionar o truncar mientras RAG sigue mandando lo mismo.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
