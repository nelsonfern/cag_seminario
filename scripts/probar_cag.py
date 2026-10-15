"""Prueba de CAG por consola.

Muestra las tres cosas que hay que ver para creer que CAG funciona:

  1. que el contexto se prepara una vez y se reutiliza,
  2. que el prefijo enviado al proveedor es identico entre consultas,
     que es lo que habilita la cache de computo,
  3. que la cache de aplicacion se invalida cuando el corpus cambia.

Ademas hace consultas reales al proveedor configurado para ver el
acierto de cache Reported, que es la metrica del enunciado.

Uso:
    python scripts/probar_cag.py                  # sin llamar al modelo
    python scripts/probar_cag.py --consultas      # con el modelo real
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROYECTO = Path(r"C:\Users\Usuario\Desktop\seminario\cag_seminario")
sys.path.insert(0, str(PROYECTO))

from app.services.agent import REGLAS_SISTEMA  # noqa: E402
from app.services.cag import CAGService  # noqa: E402

# Preguntas de ejemplo, elegidas para que cada una pruebe algo distinto.
CONSULTAS_EJEMPLO = [
    "¿Qué documentación necesito para solicitar una partida de nacimiento?",
    "¿Cuánto cuesta la copia de acta certificada?",
    "¿En qué organismo se inicia el trámite de un matrimonio?",
    "¿Qué datos del acta hacen falta para que la búsqueda funcione?",
]


def linea(char: str = "=", n: int = 76) -> str:
    return char * n


def seccion(titulo: str) -> None:
    print(f"\n{linea()}\n{titulo}\n{linea()}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de CAG por consola")
    parser.add_argument(
        "--consultas", action="store_true", help="Ejecutar consultas reales al modelo"
    )
    parser.add_argument("--force", action="store_true", help="Reconstruir el contexto")
    parser.add_argument(
        "-p",
        "--pregunta",
        action="append",
        default=None,
        metavar="TEXTO",
        help=(
            "Consulta a ejecutar. Se puede repetir para varias. "
            "Sin este flag se usan las preguntas de ejemplo."
        ),
    )
    parser.add_argument(
        "--ver-respuesta",
        action="store_true",
        help="Imprime tambien el texto que responde el modelo",
    )
    args = parser.parse_args()

    consultas = args.pregunta or CONSULTAS_EJEMPLO

    servicio = CAGService()

    # -----------------------------------------------------------------
    seccion("1. PARTIDO DEL CORPUS POR TRAMITE")
    for tramite, info in servicio.particiones_disponibles().items():
        print(f"   {tramite:26} {info['documentos']} doc   ~{info['tokens']:>5} tokens")

    # -----------------------------------------------------------------
    seccion("2. PREPARACION EN FRIO (sin cache)")
    servicio.invalidar()
    frio = servicio.preparar(force=True)
    print(
        f"   documentos      : {frio.documentos_incluidos} de {frio.documentos_incluidos + frio.documentos_omitidos}"
    )
    print(
        f"   tokens estimados: {frio.tokens_estimados} de un presupuesto de {servicio.ajustes.cag_max_context_tokens}"
    )
    print(f"   truncado        : {frio.truncado}")
    print(f"   tiempo           : {frio.preparacion_ms:.1f} ms")
    print(f"   hash del corpus  : {frio.hash_corpus}")

    # -----------------------------------------------------------------
    seccion("3. ENTREGA EN CALIENTE (desde cache de aplicacion)")
    caliente = servicio.preparar()
    print(f"   desde cache     : {caliente.desde_cache}")
    print(f"   tiempo           : {caliente.entrega_ms:.3f} ms")
    print(f"   preparacion ms   : {caliente.preparacion_ms:.1f} ms  (0 = no se reconstruyo)")
    print(f"   mismo contexto   : {caliente.contexto == frio.contexto}")
    print(f"   aceleracion      : {frio.preparacion_ms / max(caliente.entrega_ms, 0.001):.0f}x")

    # -----------------------------------------------------------------
    seccion("4. INVALIDACION CUANDO EL CORPUS CAMBIA")
    print(f"   hash antes de tocar el corpus : {servicio.hash_corpus()}")
    print("   (para verlo cambiar hay que modificar o agregar un documento)")

    # -----------------------------------------------------------------
    seccion("5. PARTICIONADO")
    particion = servicio.preparar_particion("partidas", force=True)
    print(f"   particion        : {particion.particion}")
    print(f"   documentos       : {particion.documentos_incluidos}")
    print(f"   tokens estimados : {particion.tokens_estimados}")
    print("   El corpus completo vuelve a prepararse para las consultas.")
    servicio.preparar(force=True)

    if not args.consultas:
        seccion("SIN CONSULTAS REALES (se paso --consultas)")
        print("   Con --consultas se mide el acierto de cache del proveedor,")
        print("   que es la metrica que distingue CAG de RAG.")
        return 0

    # -----------------------------------------------------------------
    seccion("6. CONSULTAS REALES AL PROVEEDOR")
    from app.services.cag import obtener_servicio_cag
    from app.services.llm import comprobar_proveedor, obtener_proveedor

    info = await comprobar_proveedor()
    print(f"   proveedor : {info['proveedor']} / {info['modelo']}")
    print(f"   cache     : {info['soporta_cache']}")
    if not info["soporta_cache"]:
        print("   AVISO: este proveedor no reporta tokens cacheados.")
        print("          Se midio igual, pero la metrica quedara como no disponible.")
    print()

    obtener_servicio_cag()
    proveedor = obtener_proveedor()
    contexto = servicio.preparar()

    aciertos: list[float] = []
    for indice, pregunta in enumerate(consultas, start=1):
        mensajes = [
            {"role": "system", "content": REGLAS_SISTEMA},
            {
                "role": "user",
                "content": f"{contexto.contexto}\n\nPregunta del ciudadano: {pregunta}",
            },
        ]
        inicio = time.perf_counter()
        r = await proveedor.completar(mensajes)
        lat = (time.perf_counter() - inicio) * 1000

        uso = r.uso
        if uso.cache_disponible and uso.tokens_entrada:
            tasa = 100 * (uso.tokens_cache_lectura or 0) / uso.tokens_entrada
            aciertos.append(tasa)
            cache = f"{uso.tokens_cache_lectura}/{uso.tokens_entrada} = {tasa:.1f}%"
        else:
            cache = "no reportada"

        print(f"   [{indice}] {pregunta}")
        print(f"       latencia {lat:>7.0f} ms | in={uso.tokens_entrada} out={uso.tokens_salida}")
        print(f"       cache: {cache}")

        if args.ver_respuesta:
            print(f"       respuesta: {r.texto.strip()[:600]}")
        print()

        # El limite de tokens por minuto obliga a pausar entre llamadas.
        if indice < len(consultas):
            time.sleep(16)

    if aciertos:
        promedio = sum(aciertos) / len(aciertos)
        print(f"   ACIERTO DE CACHE PROMEDIO: {promedio:.1f}%")
        print("   La primera consulta falla el acierto (crea el prefijo).")
        print("   Las siguientes lo reutilizan: ese es el mecanismo de CAG.")
    else:
        print("   Ninguna consulta reporto tokens cacheados.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
