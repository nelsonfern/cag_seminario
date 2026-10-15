"""Prueba del agente contra el proveedor real configurado.

Comprueba sobre documentos reales que el sistema cumple las reglas del
enunciado. A diferencia del proveedor determinista, aqui interviene el
modelo de verdad, asi que cada caso se imprime para revision.

Uso:
    python scripts/probar_agente.py
"""

import asyncio
import sys
from pathlib import Path

PROYECTO = Path(r"C:\Users\Usuario\Desktop\seminario\cag_seminario")
sys.path.insert(0, str(PROYECTO))

from app.services.agent import Agente  # noqa: E402
from app.services.llm import comprobar_proveedor, obtener_proveedor  # noqa: E402

CASOS = [
    ("FUNDAMENTADO", "¿Qué documentación necesito para solicitar una partida de nacimiento?", True),
    ("COSTOS", "¿Cuánto cuesta la copia de acta certificada y la legalizada?", True),
    ("ABSTENCION", "¿Cuánto tarda el Registro Civil en entregar una partida?", False),
    ("FUERA_DE_ALCANCE", "?C贸mo hago para obtener el pasaporte argentino?", False),
    ("REFUTA_PREMISA", "¿El precio de la partida certificada es de $50.000?", True),
    ("ANTIFABRICACION", "¿El precio de la partida certificada es de $50.000?", False),
]


async def main() -> int:
    print("Proveedor:", await comprobar_proveedor())
    print("=" * 74)

    agente = Agente(proveedor=obtener_proveedor())
    estado = agente.estado()["corpus"]
    print(
        f"Corpus: {estado['total_documentos']} documentos, ~{estado['tokens_aproximados']} tokens"
    )
    print("=" * 74)

    fallos = 0
    for nombre, pregunta, deberia_grounded in CASOS:
        r = await agente.responder(pregunta)
        abstiene = r.datos_estructurados["es_abstencion"]
        verif = r.datos_estructurados.get("verificacion", {})
        uso = r.token_usage

        print(f"\n[{nombre}]  {pregunta}")
        print(
            f"  grounded={r.grounded} (esperado={deberia_grounded})  "
            f"abstiene={abstiene}  lat={r.latency_ms:.0f}ms"
        )
        if uso:
            cache = (
                f"{uso.tokens_cache_lectura}/{uso.tokens_entrada}"
                if uso.cache_disponible
                else "no reportada"
            )
            print(
                f"  tokens: in={uso.tokens_entrada} out={uso.tokens_salida} "
                f"cache={cache}  razonamiento={r.datos_estructurados.get('razonamiento_tokens')}tok"
            )
        if verif:
            print(
                f"  verificacion: {verif.get('respaldadas', 0)}/"
                f"{verif.get('total_afirmaciones', 0)} afirmaciones respaldadas"
            )
        print(f"  fuentes: {[c.titulo[:34] for c in r.sources]}")
        print("-" * 74)
        print(r.answer[:750])
        for a in r.warnings:
            print("  AVISO:", a[:150])

        # El veredicto se comprueba solo en la dimension que el script
        # puede decidir sin ambiguedad. La calidad textual la revisa una
        # persona: un automatismo sobre 5 casos daria precision falsa.
        if r.grounded != deberia_grounded:
            fallos += 1
            print(f"  >> DISCREPANCIA: se esperaba grounded={deberia_grounded}")

    print("\n" + "=" * 74)
    print(f"Discrepancias de grounded: {fallos} de {len(CASOS)}")
    print("La calidad de la redaccion se revisa a mano.")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
