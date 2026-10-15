"""Demo del fallo de recuperacion de RAG, sin gastar un solo token.

QUE MUESTRA
===========
Dos consultas que se diferencian en UNA palabra. El fragmento que
contiene la respuesta existe, esta bien cortado y esta bien indexado,
pero en una de las dos consultas cae fuera del corte y el agente se
abstiene.

    "¿Qué documentación necesito para una PARTIDA de nacimiento?"
        -> el fragmento "Documentación requerida" queda en el puesto 8
        -> cae fuera del top-5
        -> el agente se abstiene

    "¿Qué documentación necesito para INSCRIBIR un nacimiento?"
        -> el mismo fragmento queda en el puesto 5
        -> entra al top-5
        -> el agente contesta

POR QUE IMPORTA PARA EL SEMINARIO
=================================
Esta es la demostracion central del proyecto, y funciona al reves de lo
que suele suponerse:

- CAG gana en CALIDAD, no en velocidad. Ve el corpus entero, asi que
  no puede perder un dato por no haberlo recuperado.
- RAG es mas barato en tokens, pero puede perder informacion en
  silencio: el dato existe, no llego al modelo, y el agente se abstiene
  con toda honestidad.

El abstain no es un error del modelo: es un sintoma de que el
recuperador no entrego la evidencia.

COMO SE USA
===========
    python scripts\\demo_recuperacion.py

No llama al proveedor de lenguaje: la recuperacion es local. Solo carga
el modelo de embeddings, que tarda unos 30 segundos la primera vez.
En una demo en vivo eso es tiempo muerto unacceptable, asi que conviene
lanzarlo antes de empezar a exponer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import configure_logging
from app.services.rag import obtener_servicio_rag

# Las dos consultas. Lo unico que cambia es "partida" por "inscribir un".
CONSULTAS = [
    (
        "¿Qué documentación necesito para una partida de nacimiento?",
        "con la palabra 'partida'",
    ),
    (
        "¿Qué documentación necesito para inscribir un nacimiento?",
        "con la palabra 'inscribir'",
    ),
]

# La seccion que contiene la respuesta que el ciudadano quiere.
SECCION_BUSCADA = "documentación requerida"


def main() -> None:
    configure_logging()
    servicio = obtener_servicio_rag()

    print()
    print("Preparando el indice vectorial (la primera vez tarda ~30 s)...")
    construccion = servicio.indexar()
    print(f"  listo: {construccion.get('fragmentos', construccion.get('total', '?'))} fragmentos")

    print()
    print("=" * 74)
    print("  RAG: cuando el recuperador deja pasar la respuesta")
    print("=" * 74)
    print()

    resultados = []
    for consulta, nota in CONSULTAS:
        busqueda = servicio.recuperar(consulta, top_k=12)

        puesto = None
        for i, recuperado in enumerate(busqueda.fragmentos, start=1):
            seccion = (recuperado.fragmento.seccion or "").lower()
            if seccion.startswith(SECCION_BUSCADA[:16]):
                puesto = i
                break

        resultados.append((consulta, nota, puesto, busqueda.obtenidos))

        print(f"PREGUNTA: {consulta}")
        print(f"  {nota}")
        print()

        for i, recuperado in enumerate(busqueda.fragmentos[:6], start=1):
            seccion = recuperado.fragmento.seccion or "sin seccion"
            dentro = "dentro del corte" if i <= 5 else "FUERA del corte"
            marca = ""
            if seccion.lower().startswith(SECCION_BUSCADA[:16]):
                marca = "   <<< ESTA ES LA RESPUESTA"
            print(f"  {i:>2}. {recuperado.puntuacion:.3f}  {dentro:<17}  {seccion[:38]:<40}{marca}")

        if puesto is None:
            print(f"  >> '{SECCION_BUSCADA}' NO APARECIO en los primeros 12")
        elif puesto <= 5:
            print(f"  >> '{SECCION_BUSCADA}' salio en el puesto {puesto}: LLEGA")
        else:
            print(f"  >> '{SECCION_BUSCADA}' salio en el puesto {puesto}: NO LLEGA al corte")
        print()

    print("=" * 74)
    print("  LECTURA")
    print("=" * 74)

    _, _, puesto_a, _ = resultados[0]
    _, _, puesto_b, _ = resultados[1]

    if puesto_a and puesto_b:
        print()
        print(
            f"  El mismo fragmento quedo en el puesto {puesto_b} con una "
            f"palabra y en el puesto {puesto_a} con la otra."
        )
        print()
        if puesto_a > 5 >= puesto_b:
            print("  Una sola palabra decide si el agente puede responder o tiene que abstenerse.")
            print("  La informacion NO se perdio: estaba en el corpus y en el indice.")
            print("  Lo que se perdio fue la posibilidad de recuperarla.")
        else:
            print("  En esta corrida el corte no cambio el resultado.")
    else:
        print()
        print("  El fragmento buscado no se encontro. Habria que revisar")
        print("  el indice antes de usar esta demo.")

    print()
    print("  Que hace CAG con las mismas dos consultas: las dos se responden,")
    print("  porque CAG no recupera: entrega el corpus entero siempre.")
    print()
    print("  Conclusion para el seminario: CAG no es mas rapido con este")
    print("  corpus. Es mas confiable, y esta demo muestra exactamente por que.")
    print()


if __name__ == "__main__":
    main()
