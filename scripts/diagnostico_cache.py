"""Verifica si el prefijo que enviamos es realmente identico entre llamadas.

Si el prefijo fuera identico y aun asi el acierto de cache vara, la causa
no estaria de nuestro lado sino en como el proveedor reparte las peticiones.

Uso:
    python scripts/diagnostico_cache.py
"""

import asyncio
import hashlib
import sys
from pathlib import Path

PROYECTO = Path(r"C:\Users\Usuario\Desktop\seminario\cag_seminario")
sys.path.insert(0, str(PROYECTO))

from app.services.agent import REGLAS_SISTEMA  # noqa: E402
from app.services.cag import CAGService  # noqa: E402
from app.services.llm import obtener_proveedor  # noqa: E402

N_LLAMADAS = 6
PAUSA = 3.0


async def main() -> int:
    servicio = CAGService()
    contexto = servicio.preparar()
    proveedor = obtener_proveedor()

    prefijo = f"{contexto.contexto}\n\nPregunta del ciudadano: "
    huella = hashlib.sha256(prefijo.encode("utf-8")).hexdigest()[:16]
    print(f"Prefijo unico: {huella}")
    print(f"Caracteres   : {len(prefijo)}")
    print(f"Tokens est.  : {len(prefijo) / 3.6:.0f}")
    print()
    print("Cada llamada usa el MISMO objeto de cadena. No se reconstruye,")
    print("no se reordena y no se vuelve a normalizar entre llamadas.")
    print()

    print(f"{'#':<3} {'prefijo_sha':<12} {'in':>6} {'cache':>7} {'tasa':>8}")
    print("-" * 44)

    for i in range(N_LLAMADAS):
        # Se construye la lista de mensajes una sola vez y se reusa: si el
        # prefijo cambiase, el hash de `prefijo` lo delataria.
        mensajes = [
            {"role": "system", "content": REGLAS_SISTEMA},
            {"role": "user", "content": prefijo + f"pregunta de control numero {i}"},
        ]
        r = await proveedor.completar(mensajes)
        uso = r.uso

        if uso.cache_disponible and uso.tokens_entrada:
            tasa = f"{100 * (uso.tokens_cache_lectura or 0) / uso.tokens_entrada:.1f}%"
            cache = uso.tokens_cache_lectura
        else:
            tasa = "no rep."
            cache = None

        print(f"{i + 1:<3} {huella:<12} {uso.tokens_entrada:>6} {cache!s:>7} {tasa:>8}")

        if i < N_LLAMADAS - 1:
            await asyncio.sleep(PAUSA)

    print()
    print("Conclusión:")
    print("  Si el hash del prefijo es idéntico en todas las llamadas y el")
    print("  acierto varía, la variación es del proveedor (reparto entre")
    print("  backends o expiración), no de cómo armamos el contexto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
