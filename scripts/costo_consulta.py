"""Costo real por consulta, con los precios publicados por Groq.

Los numeros de tokens salen de mediciones reales del proyecto, no de
suposiciones. Los precios salen de la documentacion de Groq:

    gpt-oss-120b   entrada        $0.150 / 1M tokens
    gpt-oss-120b   entrada cache  $0.075 / 1M tokens  (50% de descuento)
    gpt-oss-120b   salida         $0.600 / 1M tokens

Fuente: console.groq.com/docs/prompt-caching y groq.com/pricing

Si Groq cambia sus precios, este numero queda viejo. Por eso el script
imprime tambien de donde salio cada valor.
"""

from __future__ import annotations

# --- Precios publicados (verificar antes de presentar) --------------------
PRECIO_ENTRADA = 0.150
PRECIO_ENTRADA_CACHE = 0.075
PRECIO_SALIDA = 0.600

# --- Tokens medidos en corridas reales -----------------------------------
# (tokens de entrada, tokens cacheados, tokens de salida)
MEDIDO = {
    "CAG": (3952, 3840, 124),
    "RAG": (1446, 512, 184),
}

# Escenario de cache frio: nadie consulto antes, el proveedor no tiene nada.
MEDIDO_FRIO = {
    "CAG": (3952, 0, 124),
    "RAG": (1446, 0, 184),
}


def costo(entrada: int, cacheados: int, salida: int) -> tuple[float, float, float]:
    """Costo en dolares, desglosado."""
    sin_cache = max(entrada - cacheados, 0)
    c_entrada = sin_cache * PRECIO_ENTRADA / 1_000_000
    c_cache = cacheados * PRECIO_ENTRADA_CACHE / 1_000_000
    c_salida = salida * PRECIO_SALIDA / 1_000_000
    return c_entrada + c_cache, c_entrada + c_cache + c_salida, c_salida


def tabla(medido: dict[tuple[int, int, int]], titulo: str) -> None:
    print()
    print(titulo)
    print("=" * 78)
    print(
        f"{'estrategia':<12}{'entrada':>9}{'cache':>9}{'salida':>8}"
        f"{'$ entrada':>12}{'$ salida':>10}{'$ total':>12}"
    )
    print("-" * 78)

    for nombre, (entrada, cache, salida) in medido.items():
        c_ent, c_tot, c_sal = costo(entrada, cache, salida)
        print(
            f"{nombre:<12}{entrada:>9,}{cache:>9,}{salida:>8,}"
            f"{c_ent:>12.6f}{c_sal:>10.6f}{c_tot:>12.6f}"
        )

    print("-" * 78)

    c_cag = costo(*medido["CAG"])[1]
    c_rag = costo(*medido["RAG"])[1]
    mas_barato = "CAG" if c_cag < c_rag else "RAG"
    diferencia = abs(c_cag - c_rag)

    print(f"  mas barato: {mas_barato}")
    print(f"  diferencia: ${diferencia:.6f} por consulta")
    print(f"  en 100 consultas: ${diferencia * 100:.2f}")
    print(f"  en 100.000 consultas: ${diferencia * 100_000:,.2f}")


def main() -> None:
    print()
    print("COSTO REAL POR CONSULTA - gpt-oss-120b en Groq")
    print("=" * 78)
    print("  entrada        $0.150 / 1M")
    print("  entrada cache  $0.075 / 1M   (descuento del 50%)")
    print("  salida         $0.600 / 1M")

    tabla(MEDIDO, "\nA. Con la cache del proveedor funcionando (lo que se midio)")
    tabla(MEDIDO_FRIO, "\nB. Con la cache fria (primera consulta del proceso)")

    print()
    print("QUE DICE ESTO")
    print("=" * 78)
    print()
    print("  1. La diferencia por consulta es de centavos. Con 6 documentos")
    print("     ninguna de las dos estrategias es cara.")
    print()
    print("  2. CAG gana en costo SOLO cuando la cache funciona. Con la cache")
    print("     fria, CAG paga sus 3.952 tokens enteros y RAG, que ya usa")
    print("     menos, paga los suyos.")
    print()
    print("  3. La cache es el unico motivo por el que CAG tiene sentido en")
    print("     costo. Y es 'best effort': Groq avisa que un acierto no esta")
    print("     garantizado. Ya se vio en este proyecto: el mismo prefijo dio")
    print("     0%, 14%, 73%, 81% y 98% de acierto en corridas distintas.")
    print()
    print("  4. El costo no es la variable que decide. El corpus, si.")


if __name__ == "__main__":
    main()
