"""¿El fallo de RAG es del modelo o de la recuperacion?

EXPERIMENTO
===========
RAG se abstuvo de una pregunta cuya respuesta esta en el corpus. Hay dos
explicaciones posibles:

    A) El modelo es incapaz de responder aunque tenga la evidencia.
    B) La evidencia nunca llego al prompt, asi que no hay nada que pueda
       responder y se abstiene con honestidad.

Para separar las dos se arma el MISMO prompt que recibio RAG, pero se le
agrega el fragmento que faltaba. Unica variable que cambia: la evidencia.

    - Si responde con esa diferencia, la causa fue la recuperacion (B).
    - Si sigue absteniendose, la causa es el modelo (A).

Este test NO pasa por el recuperador ni por CAG: es el prompt de RAG con
un fragmento pegado a mano.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import configure_logging
from app.services.agent import REGLAS_SISTEMA, Agente
from app.services.ingestion import envolver_como_contexto
from app.services.llm import obtener_proveedor

PREGUNTA = "¿Qué documentación necesito para una partida de nacimiento?"

# El fragmento que RAG no entrego. Sale del indice, verificado a mano.
FRAGMENTO_PERDIDO = """--- Fragmento que RAG no entrego ---
Seccion: Documentación requerida
""" + envolver_como_contexto(
    """*Certificado médico de nacimiento original con sello del profesional
firmante o certificado de nacido vivo expedido por el hospital o la clínica.
*Documento de identidad de ambos progenitores.
*Si los progenitores no pueden asistir, deben llevar la partida ya expedida.
El trámite es presencial."""
)


async def main() -> None:
    configure_logging()
    agente = Agente(proveedor=obtener_proveedor())

    # 1. El prompt que RAG construyo, tal cual.
    contexto_rag, documentos_rag, metricas_rag = agente._contexto_rag(PREGUNTA)
    print("=" * 74)
    print("1. LO QUE RAG ENTREGO AL MODELO")
    print("=" * 74)
    print(f"  fragmentos: {metricas_rag['fragmentos_recuperados']}")
    print(f"  secciones : {[d.metadatos.titulo.split('-')[0].strip() for d in documentos_rag]}")
    print(
        f"  contiene 'Documentacion requerida'? {'SI' if 'Documentaci' in contexto_rag else 'NO'}"
    )
    print()

    # 2. El mismo prompt MAS el fragmento que faltaba.
    contexto_arreglado = contexto_rag + "\n\n" + FRAGMENTO_PERDIDO
    print("=" * 74)
    print("2. EL MISMO PROMPT, MAS EL FRAGMENTO QUE FALTA")
    print("=" * 74)
    print(f"  largo original : {len(contexto_rag):,} caracteres")
    print(f"  largo con el fragmento: {len(contexto_arreglado):,} caracteres")
    print()

    for etiqueta, contexto in (
        ("SIN el fragmento (lo que vio RAG)", contexto_rag),
        ("CON el fragmento (variable unica)", contexto_arreglado),
    ):
        prompt = agente._armar_prompt(contexto, PREGUNTA, documentos_rag)
        crudo = await agente.proveedor.completar(
            [
                {"role": "system", "content": REGLAS_SISTEMA},
                {"role": "user", "content": prompt},
            ]
        )
        print("-" * 74)
        print(f"{etiqueta}")
        print("-" * 74)
        print(f"  {crudo.texto.strip()[:400]}")
        print()

        # Pausa para no pegarle al limite del proveedor.
        await asyncio.sleep(18)

    print("=" * 74)
    print("COMO SE LEE ESTO")
    print("=" * 74)
    print()
    print("  Si la primera se abstiene y la segunda contesta, el modelo")
    print("  puede responder. Lo que fallo fue la recuperacion, no el")
    print("  modelo. Y CAG no gana por ser mas inteligente: gana porque")
    print("  no recupera, y por lo tanto no puede dejar pasar nada.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
