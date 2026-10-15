"""Prueba de RAG por consola.

Muestra las tres cosas que hay que ver para creer que RAG funciona:

  1. que el corpus se parte en fragmentos con seccion y procedencia,
  2. que recupera lo relevante y lo ordena por puntuacion,
  3. cuanto cuesta recuperar, separado del costo de construir el indice.

La tercera es la que mas importa para el experimento: ese costo es
propio de RAG y no existe en CAG.

Uso:
    python scripts/probar_rag.py                  # indexar y recuperar
    python scripts/probar_rag.py --consultas      # ademas consulta al modelo
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROYECTO = Path(r"C:\Users\Usuario\Desktop\seminario\cag_seminario")
sys.path.insert(0, str(PROYECTO))

from app.services.cag import CAGService  # noqa: E402
from app.services.rag import RAGService  # noqa: E402

CONSULTAS = [
    "¿Cuánto cuesta la copia de acta certificada?",
    "¿Qué documentación necesito para una partida de nacimiento?",
    "¿Cómo se inscriben los matrimonios?",
    "¿Cuánto tarda el Registro Civil en entregar una partida?",
]


def linea(char="=", n=78):
    return char * n


def seccion(t):
    print(f"\n{linea()}\n{t}\n{linea()}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de RAG por consola")
    parser.add_argument("--consultas", action="store_true", help="Consultar al modelo")
    parser.add_argument("--forzar", action="store_true", help="Reconstruir el indice")
    parser.add_argument("-p", "--pregunta", action="append", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--ver-fragmentos", action="store_true")
    args = parser.parse_args()

    consultas = args.pregunta or CONSULTAS
    servicio = RAGService()

    # -----------------------------------------------------------------
    seccion("1. COMPARABILIDAD CON CAG")
    hash_rag = servicio.hash_corpus()
    hash_cag = CAGService().hash_corpus()
    print(f"   hash RAG: {hash_rag}")
    print(f"   hash CAG: {hash_cag}")
    if hash_rag == hash_cag:
        print("   OK: ambos servicios trabajan sobre el mismo corpus.")
    else:
        print("   AVISO: los hashes difieren. La comparacion NO seria valida.")
        print("   (CAG usa una version distinta en la huella por diseño)")

    # -----------------------------------------------------------------
    seccion("2. FRAGMENTADO DEL CORPUS")
    fragmentos = servicio.documentos_a_fragmentar()
    print(f"   documentos  : {servicio.repositorio.cargar_documentos().__len__()}")
    print(f"   fragmentos  : {len(fragmentos)}")
    print(f"   chunk size  : {servicio.ajustes.rag_chunk_size} caracteres")
    print(f"   solapamiento: {servicio.ajustes.rag_chunk_overlap} caracteres")
    print()
    por_documento: dict[str, list] = {}
    for f in fragmentos:
        por_documento.setdefault(f.document_id, []).append(f)
    for doc_id, fs in list(por_documento.items())[:4]:
        secs = [f.seccion for f in fs if f.seccion]
        print(f"   {doc_id[-42:]:<44} {len(fs):>2} frag  secciones={len(set(secs))}")

    # -----------------------------------------------------------------
    seccion("3. CONSTRUCCION DEL INDICE (embeddings)")
    t0 = time.perf_counter()
    info = servicio.indexar(force=args.forzar)
    duracion = (time.perf_counter() - t0) * 1000
    print(f"   reconstruido : {info.get('reconstruido')}")
    if info.get("reconstruido"):
        print(f"   fragmentos   : {info.get('fragmentos')}")
        print(f"   modelo       : {info.get('modelo_embeddings')}")
        print(f"   duracion     : {duracion / 1000:.1f} s")
    else:
        print(f"   motivo       : {info.get('motivo')}")
        print("   el modelo de embeddings ya estaba cargado")

    # -----------------------------------------------------------------
    seccion("4. RECUPERACION (el costo propio de RAG)")
    print(f"   {'consulta':<44} {'frag':>5} {'recup ms':>9} {'emb':>7} {'busq':>7}")
    print("   " + "-" * 76)
    resultados = []
    for pregunta in consultas:
        r = servicio.recuperar(pregunta, top_k=args.top_k)
        resultados.append(r)
        print(
            f"   {pregunta[:42]:<44} {r.obtenidos:>5} {r.recuperacion_ms:>9.1f} "
            f"{r.embedding_ms:>7.1f} {r.busqueda_ms:>7.1f}"
        )

    latencias = [r.recuperacion_ms for r in resultados if r.obtenidos]
    if latencias:
        print()
        print(f"   mediana de recuperacion: {sorted(latencias)[len(latencias) // 2]:.1f} ms")
        print("   Este costo NO existe en CAG: ahi no hay nada que recuperar.")

    # -----------------------------------------------------------------
    seccion("5. FRAGMENTOS RECUPERADOS")
    ultimo = resultados[-1] if resultados else None
    if ultimo:
        print(f"   consulta: {ultimo.consulta}")
        print()
        for i, frag in enumerate(ultimo.fragmentos, start=1):
            sec = frag.fragmento.seccion or "(sin seccion)"
            print(f"   [{i}] sim={frag.puntuacion:.3f}  {sec}")
            print(f"       doc: {frag.fragmento.document_id[-44:]}")
            print(f"       {frag.fragmento.contenido[:150].replace(chr(10), ' ')}...")
            print()
        if args.ver_fragmentos:
            for i, frag in enumerate(ultimo.fragmentos, start=1):
                print("=" * 78)
                print(f"FRAGMENTO {i} completo (sim={frag.puntuacion:.3f})")
                print("=" * 78)
                print(frag.fragmento.contenido)

    if not args.consultas:
        seccion("SIN CONSULTAS AL MODELO")
        return 0

    # -----------------------------------------------------------------
    seccion("6. CONSULTA AL MODELO CON CONTEXTO RAG")
    from app.services.agent import REGLAS_SISTEMA
    from app.services.citations import resumen_grounding
    from app.services.llm import obtener_proveedor

    proveedor = obtener_proveedor()
    for r in resultados:
        partes = []
        for i, frag in enumerate(r.fragmentos, start=1):
            from app.services.ingestion import envolver_como_contexto

            partes.append(
                f"--- Fragmento {i} de {len(r.fragmentos)} ---\n"
                f"Documento: {frag.fragmento.titulo}\n"
                f"Seccion: {frag.fragmento.seccion or 'sin seccion'}\n"
                + envolver_como_contexto(frag.fragmento.contenido)
            )
        contexto = "\n\n".join(partes)
        if not contexto:
            print(f"   {r.consulta}: sin fragmentos, no se consulta")
            continue

        resp = await proveedor.completar(
            [
                {"role": "system", "content": REGLAS_SISTEMA},
                {"role": "user", "content": f"{contexto}\n\nPregunta del ciudadano: {r.consulta}"},
            ]
        )
        uso = resp.uso
        cache = (
            f"{uso.tokens_cache_lectura}/{uso.tokens_entrada}"
            if uso.cache_disponible
            else "no reportada"
        )
        veredicto = resumen_grounding(resp.texto, [f.a_contexto() for f in r.fragmentos])

        print(f"   {r.consulta}")
        print(
            f"       in={uso.tokens_entrada} out={uso.tokens_salida} "
            f"cache={cache} lat={resp.latencia_ms:.0f}ms"
        )
        print(f"       grounded={veredicto['grounded']}  citas={len(veredicto['citas'])}")
        print(f"       {resp.texto.strip()[:400]}")
        print()
        # El limite de tokens por minuto obliga a pausar.
        time.sleep(16)

    print("   Comparar 'cache' contra CAG: si RAG casi nunca cachea, es")
    print("   porque el contexto cambia en cada consulta. Eso es el costo")
    print("   estructural de RAG y no se puede optimizar con ingenieria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
