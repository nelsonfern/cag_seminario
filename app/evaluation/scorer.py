"""Juzga las respuestas del experimento contra el dataset.

Searon tres formas de fallar y conviene no mezclarlas, porque dicen cosas
distintas sobre el sistema:

1. **Confabulacion.** Se afirmo un dato que no esta en la referencia.
   Es el peor fallo: el ciudadano recibe informacion falsa con formato
   de oficial.

2. **Respuesta incompleta.** No confabula, pero le falta un dato clave.
   Es un fallo de cobertura, no de honestidad.

3. **Abstencion indebida.** No respondio y la respuesta existia. Un
   fallo de cobertura disfrazado de prudencia.

Y hay un cuarto, que solo aplica a las preguntas fuera de corpus:

4. **Fallo de abstencion.** Respondio cuando no debia. Acertar en las
   diez de fuera de corpus es obligatoire abstain en las diez.

El evaluador NO intenta calificar la calidad de la prosa. Solo comprueba
hechos. Un evaluador automatico de redaccion seria otro proyecto.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.evaluation.dataset import PREGUNTAS_BASE

# Palabras que delatan una abstencion, en cualquiera de sus formas.
# Un modelo que se abstiene lo dice de maneras distintas, asi que se
# buscan varias en vez de una sola frase exacta.
MARCAS_ABSTENCION = (
    "no tengo informacion",
    "no hay informacion",
    "no consta",
    "no figura",
    "no aparece",
    "no puedo",
    "no se pudo",
    "no estoy en condiciones",
    "no encuentro",
    "sin informacion suficiente",
    "sin detalles",
    "no obra en la fuente",
    "no surge de la fuente",
)

# La abstencion se expresa de demasiadas formas para una lista cerrada.
# Este patron cubre la familia entera: "no" seguido del verbo, admita o
# no el "se" y las formas que el modelo use.
VERBOS_SIN_DECLARAR = (
    r"declara\w*|especifica\w*|indica\w*|menciona\w*|publica\w*|"
    r"figura\w*|consta\w*|informa\w*|detalla\w*|aclara\w*|"
    r"dice\w*|senala\w*|documenta\w*|obsta\w*|surge\w*|"
    r"reporta\w*|refiere\w*|encuentra\w*|distingue\w*|dispone\w*"
)

# El verbo "estar" aparece en dos tiempos y en varias personas: "no esta
# declarado", "no está publicado", "no puede ser especificado". Fijar la
# forma literal "estar" dejaba pasar todas las demas.
FORMAS_ESTAR = r"(?:estar|esta|estan|estaba|estuvo)"

MODALES = r"(?:puedo|puede|podria|podrian)"

# Tres formas de decir lo mismo, todas "no" + verbo de ausencia:
#   "no declara"          -> sin modal, sin estar
#   "no se especifica"    -> con "se" reflexivo
#   "no puede estar dado" -> con modal y verbo estar
PATRON_SIN_DECLARAR = re.compile(
    rf"\bno\s+(?:se\s+|te\s+)?"
    rf"(?:(?:{MODALES})\s+)?"
    rf"(?:{FORMAS_ESTAR}\s+)?"
    rf"(?:{VERBOS_SIN_DECLARAR})\b"
)


def normalizar(texto: str) -> str:
    """Minusculas sin acentos, para que las comparaciones no dependan de
    como el modelo escribio una tilde."""
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def _partes(texto: str) -> list[str]:
    """Trocea el texto en fragmentos con sentido propio.

    Se parte por oraciones y punto y coma, y ademas por cada coma. Hace
    falta la coma porque una abstencion y una afirmacion pueden venir en
    la misma linea: "No consta ningun horario, la atencion es de 8 a 14".
    Trocear solo por oraciones dejaria pasar la afirmacion.

    No se exige mayuscula despues de la coma porque en español es
    corriente que siga en minuscula, y ponerla hacia que la particion
    nunca ocurriera en el caso que importa.
    """
    primero = re.split(r"(?<=[.!?;])\s+|\n+", texto)
    partes: list[str] = []
    for trozo in primero:
        partes.extend(re.split(r",\s+", trozo))
    return [p for p in partes if p.strip()]


def _quitar_marcas(texto: str) -> str:
    """Deja solo las partes que NO son una abstencion declarada."""
    return " ".join(p for p in _partes(texto) if not _es_marca_de_ausencia(p))


def _es_marca_de_ausencia(parte: str) -> bool:
    n = normalizar(parte)
    if any(normalizar(m) in n for m in MARCAS_ABSTENCION):
        return True
    return bool(PATRON_SIN_DECLARAR.search(n))


def _tiene_contenido(texto: str) -> bool:
    """Hay una afirmacion concreta detras de la abstencion.

    Dos criterios, en este orden:

    1. Un numero. Cualquier dato util de un tramite —un costo, un plazo,
       una cantidad de testigos, un telefono inventado— trae digitos. Es
       la señal mas fuerte y la que atrapa la confabulacion.
    2. Un bloque largo de palabras. Cubre respuestas como "el tramite es
       gratuito", que son correctas y no traen ningun digito.

    El umbral es alto a proposito: explicar por que no se sabe no es lo
    mismo que dar una respuesta, y un par de frases de contexto no
    deberian convertir una abstencion en respuesta.
    """
    limpio = normalizar(texto).strip()
    if not limpio:
        return False
    if re.search(r"\d", limpio):
        return True
    return len(limpio.split()) >= 25


def es_abstencion(texto: str) -> bool:
    """Detecta una abstencion aunque el modelo la exprese con sus palabras.

    Importa la parte de abajo del todo. Una respuesta que dice "no tengo
    informacion" y despues inventa un numero de telefono **no** es una
    abstencion: es el peor fallo posible, disfrazado de prudencia. Por eso
    se quitan los fragmentos de abstencion y se mira lo que queda.

    Limitacion conocida: no distingue "no se declara" seguido de una
    explicacion larga de una respuesta sin numeros. Ante la duda se cuenta
    como abstencion, que es el error mas conservador.
    """
    limpio = normalizar(texto)
    tiene_marca = any(normalizar(m) in limpio for m in MARCAS_ABSTENCION)
    tiene_patron = bool(PATRON_SIN_DECLARAR.search(limpio))
    if not (tiene_marca or tiene_patron):
        return False
    return not _tiene_contenido(_quitar_marcas(texto))


def faltan_datos(texto: str, datos_clave: list[str]) -> list[str]:
    """Datos clave que no aparecen en la respuesta.

    La comparacion es laxa a proposito: se busca el numero sin exigir que
    este escrito como "1100", "$1.100" o "mil cien".
    """
    limpio = normalizar(texto)
    faltan = []

    for dato in datos_clave:
        objetivo = normalizar(dato)
        if objetivo in limpio:
            continue

        # Los numeros se comparan tambien sin separadores: "1.100" y
        # "1100" son el mismo dato.
        if re.fullmatch(r"[\d.,$ ]+", dato):
            solo_digitos = re.sub(r"[^\d]", "", objetivo)
            if solo_digitos and solo_digitos in re.sub(r"[^\d]", "", limpio):
                continue

        faltan.append(dato)

    return faltan


def juzgar(respuesta: str, grounded: bool, pregunta: dict[str, Any]) -> dict[str, Any]:
    """Juzga una respuesta contra su pregunta.

    `grounded` es el veredicto del verificador local, no el del modelo:
    el sistema decide si lo que dijo el modelo se sostiene en las fuentes.
    """
    se_abstuvo = es_abstencion(respuesta)
    debe = pregunta["debe_abstenerse"]
    faltan = faltan_datos(respuesta, pregunta["datos_clave"])

    veredicto: dict[str, Any] = {
        "id": pregunta["id"],
        "categoria": pregunta["categoria"],
        "debe_abstenerse": debe,
        "se_abstuvo": se_abstuvo,
        "grounded": grounded,
        "datos_faltantes": faltan,
        # Acierto es "hizo lo que correspondia", no "sueno bien".
        "acierto": False,
        "fallo": "",
    }

    if debe:
        # Fuera de corpus: lo correcto es no inventar.
        if se_abstuvo:
            veredicto["acierto"] = True
            veredicto["fallo"] = ""
        else:
            veredicto["fallo"] = "respondio sin evidencia en el corpus"
    else:
        # Con respuesta en el corpus: responder bien y completo.
        if se_abstuvo:
            veredicto["fallo"] = "abstencion indebida: la respuesta existia"
        elif faltan:
            veredicto["fallo"] = f"respuesta incompleta: falta {', '.join(faltan)}"
        elif not grounded:
            # Dice los datos correctos pero el verificador no pudo
            # sostenerlos. Se cuenta como acierto parcial, no como error.
            veredicto["acierto"] = True
            veredicto["fallo"] = "respuesta correcta sin respaldo verificable"
        else:
            veredicto["acierto"] = True

    return veredicto


def resumir(veredictos: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega los veredictos por categoría y por tipo de fallo."""
    total = len(veredictos)
    aciertos = sum(1 for v in veredictos if v["acierto"])
    abstenciones = sum(1 for v in veredictos if v["se_abstuvo"])
    obligatorias = sum(1 for v in veredictos if v["debe_abstenerse"])

    por_categoria: dict[str, dict[str, Any]] = {}
    for categoria in sorted({v["categoria"] for v in veredictos}):
        propias = [v for v in veredictos if v["categoria"] == categoria]
        ca = sum(1 for v in propias if v["categoria"] == "fuera_de_corpus")
        acertadas = sum(1 for v in propias if v["acierto"])
        por_categoria[categoria] = {
            "total": len(propias),
            "aciertos": acertadas,
            "tasa": round(100 * acertadas / len(propias), 1),
            "abstenciones": sum(1 for v in propias if v["se_abstuvo"]),
        }
        if ca:
            por_categoria[categoria]["debe_abstenerse"] = ca
            por_categoria[categoria]["fallo_de_abstencion"] = ca - acertadas

    fallos: dict[str, int] = {}
    for v in veredictos:
        if v["fallo"]:
            tipo = (
                "respondio sin evidencia"
                if v["fallo"] == "respondio sin evidencia en el corpus"
                else "abstencion indebida"
                if v["fallo"].startswith("abstencion indebida")
                else "respuesta incompleta"
                if v["fallo"].startswith("respuesta incompleta")
                else "sin respaldo verificable"
            )
            fallos[tipo] = fallos.get(tipo, 0) + 1

    return {
        "total": total,
        "aciertos": aciertos,
        "tasa_acierto": round(100 * aciertos / total, 1) if total else None,
        "abstenciones": abstenciones,
        "abstenciones_correctas": sum(
            1 for v in veredictos if v["se_abstuvo"] and v["debe_abstenerse"]
        ),
        "abstenciones_indebidas": sum(
            1 for v in veredictos if v["se_abstuvo"] and not v["debe_abstenerse"]
        ),
        "respuesta_indebida": obligatorias
        - sum(1 for v in veredictos if v["se_abstuvo"] and v["debe_abstenerse"]),
        "por_categoria": por_categoria,
        "fallos": fallos,
    }


def comparar_estrategias(
    veredictos_cag: list[dict[str, Any]],
    veredictos_rag: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compara CAG y RAG sobre las mismas preguntas.

    Solo se comparan las preguntas que ambas respondieron, que en una
    corrida completa son todas.
    """
    por_id_cag = {v["id"]: v for v in veredictos_cag}
    por_id_rag = {v["id"]: v for v in veredictos_rag}
    comunes = sorted(set(por_id_cag) & set(por_id_rag))

    aciertos_cag = sum(1 for i in comunes if por_id_cag[i]["acierto"])
    aciertos_rag = sum(1 for i in comunes if por_id_rag[i]["acierto"])

    solo_cag = [i for i in comunes if por_id_cag[i]["acierto"] and not por_id_rag[i]["acierto"]]
    solo_rag = [i for i in comunes if por_id_rag[i]["acierto"] and not por_id_cag[i]["acierto"]]

    return {
        "preguntas_comparadas": len(comunes),
        "aciertos_cag": aciertos_cag,
        "aciertos_rag": aciertos_rag,
        "tasa_cag": round(100 * aciertos_cag / len(comunes), 1) if comunes else None,
        "tasa_rag": round(100 * aciertos_rag / len(comunes), 1) if comunes else None,
        "solo_cag_acierta": solo_cag,
        "solo_rag_acierta": solo_rag,
        "ambas_aciertan": sum(
            1 for i in comunes if por_id_cag[i]["acierto"] and por_id_rag[i]["acierto"]
        ),
        "ninguna_acierta": [
            i for i in comunes if not por_id_cag[i]["acierto"] and not por_id_rag[i]["acierto"]
        ],
    }


def buscar_pregunta(pregunta_id: str) -> dict[str, Any]:
    for pregunta in PREGUNTAS_BASE:
        if pregunta["id"] == pregunta_id:
            return pregunta
    raise KeyError(f"No existe la pregunta {pregunta_id}")
