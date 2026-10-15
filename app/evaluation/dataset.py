"""Conjunto de preguntas de evaluacion.

Cada pregunta trae su respuesta de referencia, las fuentes esperadas y una
categoria. **Las referencias estan escritas a mano y verificadas contra los
documentos oficiales descargados**, no generadas por el modelo: una
referencia inventada mediria al modelo contra su propia imaginacion.

Las categorias siguen la distribucion que pide el enunciado:

- requisitos de tramites
- procedimientos paso a paso
- preguntas que combinan varias secciones
- consultas ambiguas o incompletas
- preguntas cuya respuesta NO esta en el corpus

Las ultimas dos son las que separan un sistema honesto de uno que
confabula, asi que tienen peso.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings

RUTA_DATASET = "data/processed/dataset.json"

CATEGORIAS = (
    "requisitos",
    "procedimiento",
    "multi_seccion",
    "ambigua",
    "fuera_de_corpus",
)


def cargar_dataset(ruta: Path | None = None) -> list[dict[str, Any]]:
    """Carga el dataset de evaluación.

    Si el archivo no existe, devuelve las preguntas embebidas en el modulo.
    """
    ruta = ruta or (get_settings().processed_dir / "dataset.json")
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))["preguntas"]
    return PREGUNTAS_BASE.copy()


def guardar_dataset(preguntas: list[dict[str, Any]], ruta: Path | None = None) -> Path:
    ruta = ruta or (get_settings().processed_dir / "dataset.json")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(
            {"version": 1, "total": len(preguntas), "preguntas": preguntas},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ruta


def distribuir_por_categoria(
    preguntas: list[dict[str, Any]],
) -> dict[str, int]:
    """Cuenta las preguntas por categoría, para verificar la distribución."""
    conteo: dict[str, int] = {}
    for pregunta in preguntas:
        categoria = pregunta["categoria"]
        conteo[categoria] = conteo.get(categoria, 0) + 1
    return conteo


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Las respuestas de referencia se transcribieron de las páginas oficiales
# del Registro Civil de La Rioja descargadas en la Fase 2. Cuando la fuente
# no declara el dato, la referencia dice exactamente eso.
PREGUNTAS_BASE: list[dict[str, Any]] = [
    # --- Requisitos (10) ---
    {
        "id": "R01",
        "categoria": "requisitos",
        "texto": "¿Qué documentación necesito para solicitar una partida de nacimiento?",
        "referencia": "Tener una cuenta de Ciudadano Digital y los datos del acta necesarios para que la búsqueda sea exitosa.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "R02",
        "categoria": "requisitos",
        "texto": "¿Qué necesito para pedir una copia de acta certificada?",
        "referencia": "Tener una cuenta de Ciudadano Digital y los datos del acta necesarios para que la búsqueda sea exitosa.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "R03",
        "categoria": "requisitos",
        "texto": "¿Qué datos del acta hay que aportar para que la búsqueda funcione?",
        "referencia": "Indicar localidad de inscripción, número de acta, tomo, serie, folio y año.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "R04",
        "categoria": "requisitos",
        "texto": "¿Qué se necesita para inscribir un matrimonio?",
        "referencia": "Las partidas de nacimiento de ambos contrayentes y el documento nacional de identidad de cada uno.",
        "fuentes_esperadas": ["matrimonios"],
    },
    {
        "id": "R05",
        "categoria": "requisitos",
        "texto": "¿Qué paperwork pide el Registro Civil para una defunción?",
        "referencia": "La fuente describe el trámite de defunción; el detalle de documentación no figura en el corpus descargado.",
        "fuentes_esperadas": ["defunciones"],
    },
    {
        "id": "R06",
        "categoria": "requisitos",
        "texto": "Para rectificar una partida, ¿qué necesito llevar?",
        "referencia": "La página de rectificación de partidas describe el trámite; el detalle de documentación no figura en el corpus descargado.",
        "fuentes_esperadas": ["rectificacion_partidas"],
    },
    {
        "id": "R07",
        "categoria": "requisitos",
        "texto": "¿Hace falta turno previo para sacar una partida?",
        "referencia": "La fuente no declara si hace falta turno previo. La solicitud de actas se inicia de manera digital.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "R08",
        "categoria": "requisitos",
        "texto": "¿Puedo pedir una partida si soy de otra provincia?",
        "referencia": "La fuente indica que el servicio permite solicitar copias de partidas inscriptas en la Provincia de La Rioja. No aclara si alguien de otra provincia puede pedirlas.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "R09",
        "categoria": "requisitos",
        "texto": "¿Qué debo tener para inscribir el nacimiento de un bebé?",
        "referencia": "La página de inscripción de nacimientos describe el trámite; el detalle de documentación no figura en el corpus descargado.",
        "fuentes_esperadas": ["nacimientos"],
    },
    {
        "id": "R10",
        "categoria": "requisitos",
        "texto": "¿Se necesita firma digital para algún trámite?",
        "referencia": "Las actas legalizadas se emiten únicamente con firma digital, la cual es igual de válida que la manuscrita según la Ley 25.506.",
        "fuentes_esperadas": ["partidas"],
    },
    # --- Procedimiento (10) ---
    {
        "id": "P01",
        "categoria": "procedimiento",
        "texto": "¿Cómo se solicita una partida de nacimiento?",
        "referencia": "La solicitud de actas se inicia de manera digital en el servicio. Se accede desde el servicio Mi Registro Civil.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "P02",
        "categoria": "procedimiento",
        "texto": "¿Cómo se hace el trámite de matrimonios?",
        "referencia": "La página de inscripción de matrimonios describe el trámite; el paso a paso no figura en el corpus descargado.",
        "fuentes_esperadas": ["matrimonios"],
    },
    {
        "id": "P03",
        "categoria": "procedimiento",
        "texto": "¿Cómo se descarga la partida una vez que la pedí?",
        "referencia": "Todas las notificaciones del proceso llegarán al correo vinculado a la cuenta Ciudadano Digital. La descarga se realiza desde Mis trámites en el servicio Mi Registro Civil.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "P04",
        "categoria": "procedimiento",
        "texto": "¿Cómo se instala la cadena de confianza de los certificados?",
        "referencia": "Descargar el programa Adobe Reader DC, configurarlo desde Edición > Preferencias siguiendo la guía, e instalar la cadena de confianza de los certificados emitidos. Esto se realiza una sola vez y solo desde una computadora.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "P05",
        "categoria": "procedimiento",
        "texto": "¿Cómo se valida una firma digital?",
        "referencia": "El procedimiento requiere Adobe Reader DC configurado y la cadena de confianza instalada. Solo puede realizarse desde una computadora.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "P06",
        "categoria": "procedimiento",
        "texto": "¿Qué pasos sigo para inscribir una defunción?",
        "referencia": "La página de defunciones describe el trámite; el paso a paso no figura en el corpus descargado.",
        "fuentes_esperadas": ["defunciones"],
    },
    {
        "id": "P07",
        "categoria": "procedimiento",
        "texto": "¿Cómo se hace una rectificación de partida?",
        "referencia": "La página de rectificación de partidas describe el trámite; el paso a paso no figura en el corpus descargado.",
        "fuentes_esperadas": ["rectificacion_partidas"],
    },
    {
        "id": "P08",
        "categoria": "procedimiento",
        "texto": "¿Dónde se hacen los trámites de defunción?",
        "referencia": "El Registro Civil de La Rioja registra defunciones. El detalle de delegatedaciones no figura en el corpus descargado.",
        "fuentes_esperadas": ["defunciones"],
    },
    {
        "id": "P09",
        "categoria": "procedimiento",
        "texto": "¿El trámite de partidas se puede iniciar por internet?",
        "referencia": "Sí. La solicitud de actas se inicia de manera digital en el servicio.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "P10",
        "categoria": "procedimiento",
        "texto": "¿Cómo configuro Adobe Reader para validar documentos?",
        "referencia": "Desde Edición > Preferencias, siguiendo la guía indicada en el documento.",
        "fuentes_esperadas": ["partidas"],
    },
    # --- Multi seccion (10) ---
    {
        "id": "M01",
        "categoria": "multi_seccion",
        "texto": "¿Cuánto cuesta una copia certificada y qué necesito para pedirla?",
        "referencia": "Copia de acta certificada: $1.100 según la Ley Impositiva 2024 (Ley 10.929, art. 79). Se requiere cuenta de Ciudadano Digital y los datos del acta.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "M02",
        "categoria": "multi_seccion",
        "texto": "¿Cuánto cuesta una copia legalizada y cómo se solicita?",
        "referencia": "Copia de acta legalizada: $1.500 según la Ley Impositiva 2024 (Ley 10.929, art. 79). Las actas legalizadas se emiten únicamente con firma digital. La solicitud se inicia de manera digital.",
        "fuentes_esperadas": ["partidas"],
    },
    {
        "id": "M03",
        "categoria": "multi_seccion",
        "texto": "¿Qué tramites son gratis y cuáles tienen costo?",
        "referencia": "Copia certificada: $1.100. Copia legalizada: $1.500. La copia de acta, certificado o extracto de nacimiento para trámite de identificación y primera copia de matrimonio es sin cargo.",
        "fuentes_esperadas": ["partidas"],
    },
]
