"""Trazabilidad y verificacion de citas.

El enunciado es claro en dos puntos que este modulo resuelve:

1. "No generar citas a documentos que realmente no se hayan utilizado."
2. "No inventar requisitos, costos, plazos, enlaces ni datos personales."

La solucion NO es pedirle al modelo que cite. Es lo contrario: el modelo
responde, y el sistema decide por separado que documentos respaldan esa
respuesta, comparando la respuesta contra el corpus local.

Si se le pidiera al modelo que cite, un documento no utilizado apareceria
como citado con total naturalidad, y el mismo criterio que genera la cita
la validaria. El control tiene que ser externo al modelo.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import Cita, MetadatosDocumento

# Palabras vacias: aparecen en toda respuesta y no aportan evidencia.
# Sin filtrarlas, cualquier documento comparte palabras con cualquier
# respuesta y el criterio de citacion dejaria de discriminar.
_VACIAS = frozenset(
    [
        "para",
        "porque",
        "como",
        "cuando",
        "donde",
        "este",
        "esta",
        "esto",
        "estos",
        "estas",
        "aquel",
        "aquella",
        "puede",
        "pueden",
        "tiene",
        "tienen",
        "hacer",
        "hace",
        "segun",
        "sobre",
        "entre",
        "hasta",
        "desde",
        "tramite",
        "tramites",
        "informacion",
        "acerca",
    ]
)

# Un monto, una ley o un plazo son afirmaciones verificables y de alto
# riesgo: si aparecen, tienen que estar respaldadas por el texto fuente.
# El monto se escribe como $1.100 en español, pero los modelos lo emiten
# con espacio fino ($1 100), que es notacion francesa y es real.
# Errores que esto evita:
#   - `[\d.,]+` arrastraba el punto final de la oracion ("$1.100.") y
#     marcaba como inventado un precio bien informado;
#   - cortar en el espacio separaba "$50 000" en "$50", generando un falso
#     positivo de Montana sin respaldo.
# Por eso se admiten punto, coma y espacio como separadores de miles.
_RE_MONTOS = re.compile(r"\$\s?\d{1,3}(?:[.,\u00a0\u202f ]\d{3})+(?:,\d+)?|\$\s?\d+(?:,\d+)?")
_RE_LEYES = re.compile(r"(?:ley|decreto|resoluci[oó]n)\s*(?:n[°ºo]\s*)?[\d.,]+", re.IGNORECASE)
_RE_PLAZOS = re.compile(
    r"\b\d+\s*(?:d[ií]as?|horas?|mes(?:es)?|a[nñ]os?|semanas?)\b", re.IGNORECASE
)
_RE_URLS = re.compile(r"https?://\S+|www\.\S+")

# Formas en que el agente declara que no tiene evidencia. Se usan para
# decidir si la respuesta es una abstincion, tanto para las metricas
# como para no mostrar fuentes que no respaldan nada.
_SEÑALES_ABSTENCION = (
    "no tengo informacion suficiente",
    "no hay informacion suficiente",
    "no encuentro informacion",
    "no dispongo de informacion",
    "no puedo determinar",
    "la informacion disponible no alcanza",
    "no consta en las fuentes",
    "no figura en las fuentes",
    "no se encuentra en el corpus",
)


@dataclass
class DocumentoEnContexto:
    """Documento que se envio al modelo, con su texto searchable."""

    metadatos: MetadatosDocumento
    contenido: str
    #: Fragmentos concretos que se enviaron, si la estrategia fue RAG.
    fragmentos: list[str] = field(default_factory=list)

    @property
    def texto_buscable(self) -> str:
        """Texto contra el que se verifican las afirmaciones.

        En RAG solo se buscan las afirmaciones dentro de los fragmentos
        enviados: si un dato esta en el documento pero no se recupero, el
        agente no podia usarlo y no debe aparentar que si.
        """
        return "\n".join(self.fragmentos) if self.fragmentos else self.contenido

    @property
    def normalizado(self) -> str:
        """Minusculas y sin tildes, para comparar sin ruido ortografico."""
        return _normalizar(self.texto_buscable)


def _normalizar(texto: str) -> str:
    """Minusculas, sin tildes y con espacios colapsados.

    Comparar con y sin tildes evita que "acta" y "acta" se traten como
    palabras distintas por una diferencia ortografica.
    """
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", sin_acentos)


def _significado(texto: str) -> set[str]:
    """Palabras con valor informativo de un texto."""
    palabras = re.findall(r"[a-z0-9$%]{4,}", _normalizar(texto))
    return {p for p in palabras if p not in _VACIAS}


def es_abstencion(respuesta: str) -> bool:
    """Detecta si la respuesta es una abstincion declarada."""
    normalizada = _normalizar(respuesta)
    return any(señal in normalizada for señal in _SEÑALES_ABSTENCION)


def afirmaciones_verificables(respuesta: str) -> dict[str, list[str]]:
    """Extrae las afirmaciones que deben estar respaldadas por una fuente.

    Montos, referencias a normativa y plazos son los datos que un agente
    tendria mas tendance a inventar, y los mas daninos si los inventa.
    """
    return {
        "montos": [m.group(0).strip() for m in _RE_MONTOS.finditer(respuesta)],
        "normativa": [m.group(0).strip() for m in _RE_LEYES.finditer(respuesta)],
        "plazos": [m.group(0).strip() for m in _RE_PLAZOS.finditer(respuesta)],
    }


def _canonico_monto(texto: str) -> str | None:
    """Reduce un monto a su cifra, sin depender del separador.

    La fuente escribe `$1.100` y el modelo escribe `$1 100` o `$1,100`.
    Son el mismo numero con notaciones distintas. Compararlos como texto
    produciria un falso positivo de "monto sin respaldo", que es el peor
    error posible en esta verificacion: marcar como inventado un dato que
    el ciudadano leeria como correcto.
    """
    limpio = re.sub(r"[^\d.,\u00a0\u202f ]", "", texto)
    if not limpio:
        return None

    separadores = ".,\u00a0\u202f "
    entero: list[str] = []
    decimal = ""
    i = 0
    while i < len(limpio):
        caracter = limpio[i]
        if caracter in separadores:
            siguientes = limpio[i + 1 : i + 4]
            if re.fullmatch(r"\d{3}", siguientes):
                # Tres digitos tras el separador: es separador de miles.
                entero.append(siguientes)
                i += 4
                continue
            coincide = re.match(r"^\d+", siguientes)
            if coincide:
                decimal = "." + coincide.group(0)
                i += coincide.end()
                continue
            i += 1
            continue
        entero.append(caracter)
        i += 1

    cifra = "".join(entero).lstrip("0") or "0"
    return cifra + decimal


def verificar_afirmaciones(
    respuesta: str,
    documentos: list[DocumentoEnContexto],
) -> dict[str, Any]:
    """Comprueba si cada afirmacion verificable aparece en alguna fuente.

    Los montos se comparan por cifra y no por texto, porque la fuente y el
    modelo usan notaciones distintas para el mismo numero. El resto se
    busca de forma literal pero insensible a tildes y mayusculas.
    """
    afirmaciones = afirmaciones_verificables(respuesta)
    corpus_normalizado = [d.normalizado for d in documentos]

    # Montos presentes en el corpus, reducidos a cifra.
    montos_corpus: set[str] = set()
    for texto in corpus_normalizado:
        for monto in _RE_MONTOS.findall(texto):
            canonico = _canonico_monto(monto)
            if canonico:
                montos_corpus.add(canonico)

    resultado: dict[str, Any] = {
        "total_afirmaciones": 0,
        "respaldadas": 0,
        "sin_respaldo": [],
        "hay_afirmacion_sin_respaldo": False,
    }

    for categoria, valores in afirmaciones.items():
        for valor in valores:
            resultado["total_afirmaciones"] += 1

            if categoria == "montos":
                canonico = _canonico_monto(valor)
                if canonico is not None and canonico in montos_corpus:
                    resultado["respaldadas"] += 1
                else:
                    resultado["sin_respaldo"].append({"tipo": categoria, "valor": valor})
                continue

            if any(_normalizar(valor) in c for c in corpus_normalizado):
                resultado["respaldadas"] += 1
            else:
                resultado["sin_respaldo"].append({"tipo": categoria, "valor": valor})

    resultado["hay_afirmacion_sin_respaldo"] = bool(resultado["sin_respaldo"])
    return resultado


def seleccionar_citas(
    respuesta: str,
    documentos: list[DocumentoEnContexto],
    maximo: int = 4,
) -> list[Cita]:
    """Determina que documentos respaldan la respuesta.

    Criterio: un documento se cita si aporta al menos un termino con valor
    informativo que aparece en la respuesta. Se excluyen los que solo
    aportan palabras vacias.

    Las citas NO las decide el modelo: salen de comparar su respuesta con
    lo que se le dio. Es la unica forma de garantizar que no se cite un
    documento que no se uso.
    """
    if es_abstencion(respuesta) or not respuesta.strip():
        # Una abstincion no se apoya en ninguna fuente. Listar fuentes
        # sugeriria que si hay evidencia, que es lo contrario de la verdad.
        return []

    arquitectura = _significado(respuesta)
    if not arquitectura:
        return []

    puntuaciones: list[tuple[int, str, DocumentoEnContexto]] = []
    for documento in documentos:
        comunes = arquitectura & _significado(documento.normalizado)
        if comunes:
            puntuaciones.append((len(comunes), documento.metadatos.document_id, documento))

    if not puntuaciones:
        return []

    # Orden estable: mas solapamiento primero y, a igualdad, por id, para
    # que dos ejecuciones produzcan la misma lista de fuentes.
    puntuaciones.sort(key=lambda x: (-x[0], x[1]))

    citas: list[Cita] = []
    for _, _, documento in puntuaciones[:maximo]:
        meta = documento.metadatos
        citas.append(
            Cita(
                document_id=meta.document_id,
                titulo=meta.titulo,
                organismo=meta.organismo,
                url_origen=meta.url_origen,
                tramite=meta.tramite,
            )
        )
    return citas


def detectar_urls_no_verificadas(respuesta: str) -> list[str]:
    """Urls que escribio el modelo y que no provienen del corpus.

    El enunciado prohibe inventar enlaces. Estas se reportan aparte para
    marcarlas como no verificadas en vez de mostrarlas como fuente.
    """
    urls = [u.rstrip(".,);") for u in _RE_URLS.findall(respuesta)]
    return [u for u in urls if not re.search(r"(larioja|argentina\.gob)\.(gob|ar)", u)]


def resumen_grounding(
    respuesta: str,
    documentos: list[DocumentoEnContexto],
) -> dict[str, Any]:
    """Consolida el veredicto de fundamentacion de una respuesta."""
    citas = seleccionar_citas(respuesta, documentos)
    verificacion = verificar_afirmaciones(respuesta, documentos)
    urls_no_verificadas = detectar_urls_no_verificadas(respuesta)
    abstincion = es_abstencion(respuesta)

    # Hay grounding cuando la respuesta se apoya en al menos una fuente real.
    # Una abstincion nunca lo esta, por definicion.
    grounded = bool(citas) and not abstincion

    advertencias: list[str] = []
    if abstincion:
        advertencias.append("La respuesta no se apoya en evidencia documental suficiente.")
    if verificacion["hay_afirmacion_sin_respaldo"]:
        detalles = ", ".join(
            f"{a['valor']} ({a['tipo']})" for a in verificacion["sin_respaldo"][:3]
        )
        advertencias.append(
            "La respuesta menciona datos que no aparecen textualmente en las "
            f"fuentes consultadas: {detalles}. Conviene verificarlos."
        )
    if urls_no_verificadas:
        advertencias.append(
            "La respuesta incluye enlaces que no provienen del corpus: "
            f"{', '.join(urls_no_verificadas[:3])}."
        )

    return {
        "grounded": grounded,
        "es_abstencion": abstincion,
        "citas": citas,
        "verificacion": verificacion,
        "urls_no_verificadas": urls_no_verificadas,
        "advertencias": advertencias,
    }
