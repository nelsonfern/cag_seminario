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
# Esquema de cada pregunta
# ---------------------------------------------------------------------------
#   id              identificador estable, para citarlo en el informe
#   categoria       de que tipo de prueba es
#   texto           la consulta, tal como la haria un ciudadano
#   referencia      la respuesta correcta segun las fuentes descargadas
#   datos_clave     hechos que TIENEN que aparecer en una respuesta correcta
#   debe_abstenerse  si la conducta correcta es no responder
#   fuentes         tramites del corpus que deberian aparecer como fuente
#
# `datos_clave` existe para no tener que leer cien respuestas a ojo. Son
# cadenas cortas que se buscan en la respuesta; si el dato no esta, la
# respuesta no sirve aunque suene correcta.
#
# `debe_abstenerse` es la metrica mas importante del experimento. En las
# preguntas fuera de corpus, responder es un error: el agente esta
# inventando. Un sistema que contesta todo tiene 100% de acierto y 0%
# de honestidad.


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Las respuestas de referencia se transcribieron de las paginas oficiales
# del Registro Civil de La Rioja descargadas en la Fase 2. Cuando la
# fuente no declara el dato, la referencia dice exactamente eso.
#
# Hay tres casos donde la fuente esta incompleta y NO se relleno:
#
#   - El costo del matrimonio aparece como "$??" en la pagina.
#   - El costo de la rectificacion tambien aparece como "$??".
#   - El costo de la partida sale de la Ley Impositiva 2024 (Ley 10.929
#     art. 79), pero la propia fuente advierte que la Municipalidad puede
#     fijar su propia tasa.
#
# Esos tres casos son los mas valioso del experimento: el sistema tiene
# que decir "no lo dice la fuente" y no inventar un numero.

PREGUNTAS_BASE: list[dict[str, Any]] = [
    # ==================================================================
    # REQUISITOS (10)
    # ==================================================================
    {
        "id": "R01",
        "categoria": "requisitos",
        "texto": "¿Qué documentación necesito para solicitar una partida de nacimiento?",
        "referencia": "Tener una cuenta de Ciudadano Digital y los datos del acta necesarios para que la búsqueda sea exitosa.",
        "datos_clave": ["Ciudadano Digital"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "R02",
        "categoria": "requisitos",
        "texto": "¿Las actas de nacimiento para trámites de identificación tienen costo?",
        "referencia": "No. La copia de acta, certificado o extracto de nacimiento para trámite de identificación es sin cargo, así como la primera copia de matrimonio.",
        "datos_clave": ["Sin cargo"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "R03",
        "categoria": "requisitos",
        "texto": "¿Qué datos del acta hay que aportar para que la búsqueda funcione?",
        "referencia": "Indicar localidad de inscripción, número de acta, tomo, serie, folio y año.",
        "datos_clave": ["tomo", "folio"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "R04",
        "categoria": "requisitos",
        "texto": "¿Qué se necesita para inscribir un matrimonio?",
        "referencia": "DNI original y vigente de ambos contrayentes, actas de nacimiento actualizadas, DNI de los testigos, y segun el estado civil documentos adicionales: soltero/a, acta de matrimonio con marginal de divorcio, o acta de defunción y acta de matrimonio anterior. Si nació fuera del país, apostilla o visa consular y traducción oficial. Testigos: mínimo 2 y máximo 4 mayores de edad.",
        "datos_clave": ["DNI", "testigos"],
        "debe_abstenerse": False,
        "fuentes": ["matrimonios"],
    },
    {
        "id": "R05",
        "categoria": "requisitos",
        "texto": "¿Qué documentación pide el Registro Civil para inscribir una defunción?",
        "referencia": "Certificado médico de defunción emitido por un profesional habilitado, DNI del fallecido original, DNI de quien solicita la inscripción original y copia, y que la inscripción se realice dentro de las 48 horas hábiles luego de ocurrido el fallecimiento.",
        "datos_clave": ["48", "certificado médico de defunción"],
        "debe_abstenerse": False,
        "fuentes": ["defunciones"],
    },
    {
        "id": "R06",
        "categoria": "requisitos",
        "texto": "Para rectificar una partida, ¿qué necesito llevar?",
        "referencia": "Nota explicativa del error a corregir y documentación que lo respalde, DNI original del solicitante, datos del acta a rectificar para su localización, pruebas documentales que acrediten el dato correcto, y si es familiar directo, representante legal o apoderado, documentación que acredite el vínculo o autorización.",
        "datos_clave": ["nota explicativa", "DNI"],
        "debe_abstenerse": False,
        "fuentes": ["rectificacion_partidas"],
    },
    {
        "id": "R07",
        "categoria": "requisitos",
        "texto": "¿Hace falta turno previo para sacar una partida?",
        "referencia": "La fuente no declara si hace falta turno previo. La solicitud de actas se inicia de manera digital en el servicio Mi Registro Civil.",
        "datos_clave": [],
        "respuesta_es_ausencia": True,
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "R08",
        "categoria": "requisitos",
        "texto": "¿Cuántos testigos necesito para casarme?",
        "referencia": "Mínimo 2 y máximo 4 testigos mayores de edad, con DNI original y vigente.",
        "datos_clave": ["2", "4", "testigos"],
        "debe_abstenerse": False,
        "fuentes": ["matrimonios"],
    },
    {
        "id": "R09",
        "categoria": "requisitos",
        "texto": "¿Qué necesito para inscribir el nacimiento de un bebé?",
        "referencia": "Certificado médico de nacimiento original con sello del profesional firmante o certificado de nacido vivo expedido por el hospital o la clínica, y documento de identidad de ambos progenitores. Si están casados, libreta de familia o datos de inscripción del acta de matrimonio. Si son extranjeros, DNI argentino para extranjeros, pasaporte o cédula sin vencer. Si alguno de los progenitores no posee documento, 2 testigos mayores de 18 años con DNI vigente.",
        "datos_clave": ["certificado médico", "progenitores"],
        "debe_abstenerse": False,
        "fuentes": ["nacimientos"],
    },
    {
        "id": "R10",
        "categoria": "requisitos",
        "texto": "¿Hace falta firma digital para algún trámite?",
        "referencia": "Las actas legalizadas se emiten únicamente con firma digital, la cual es igual de válida que la manuscrita según la Ley 25.506.",
        "datos_clave": ["firma digital", "25.506"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    # ==================================================================
    # PROCEDIMIENTO (10)
    # ==================================================================
    {
        "id": "P01",
        "categoria": "procedimiento",
        "texto": "¿Cómo se solicita una partida de nacimiento?",
        "referencia": "La solicitud de actas se inicia de manera digital en el servicio Mi Registro Civil. Todas las notificaciones del proceso llegarán al correo vinculado a la cuenta de Ciudadano Digital, y la descarga se realiza desde Mis trámites.",
        "datos_clave": ["digital", "Ciudadano Digital"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "P02",
        "categoria": "procedimiento",
        "texto": "¿Cómo es el trámite para casarse?",
        "referencia": "Se solicita turno con dos meses de anticipación en la Oficina de Matrimonio Casa Central. Al menos uno de los contrayentes debe tener domicilio en La Rioja Capital. Se concurre con DNI original y vigente de ambos, actas de nacimiento actualizadas y DNI de los testigos.",
        "datos_clave": ["turno", "domicilio"],
        "debe_abstenerse": False,
        "fuentes": ["matrimonios"],
    },
    {
        "id": "P03",
        "categoria": "procedimiento",
        "texto": "¿Cómo se descarga la partida una vez que la pedí?",
        "referencia": "Todas las notificaciones del proceso llegarán al correo vinculado a la cuenta de Ciudadano Digital. La descarga se realiza desde Mis trámites en el servicio Mi Registro Civil.",
        "datos_clave": ["Mis trámites", "correo"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "P04",
        "categoria": "procedimiento",
        "texto": "¿Cómo se instala la cadena de confianza de los certificados?",
        "referencia": "Descargar el programa Adobe Reader DC, configurarlo desde Edición > Preferencias siguiendo la guía, e instalar la cadena de confianza de los certificados emitidos. Esto se realiza una sola vez y solo desde una computadora.",
        "datos_clave": ["Adobe Reader", "cadena de confianza"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "P05",
        "categoria": "procedimiento",
        "texto": "¿Cómo se valida una firma digital?",
        "referencia": "El procedimiento requiere Adobe Reader DC configurado y la cadena de confianza instalada. Solo puede realizarse desde una computadora.",
        "datos_clave": ["Adobe Reader", "computadora"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "P06",
        "categoria": "procedimiento",
        "texto": "¿Qué pasos sigo para inscribir una defunción?",
        "referencia": "Concurrir a la funeraria municipal o privada y solicitar el servicio funerario. Reunir toda la documentación necesaria. Concurrir al registro civil de la localidad de fallecimiento, consultando previamente la modalidad de atención con o sin turno. Si se contrató un servicio funerario privado, la empresa puede presentar la documentación. En el momento de la inscripción se obtiene el acta de defunción y, en caso de corresponder, la autorización de libre tránsito.",
        "datos_clave": ["funeraria", "acta de defunción"],
        "debe_abstenerse": False,
        "fuentes": ["defunciones"],
    },
    {
        "id": "P07",
        "categoria": "procedimiento",
        "texto": "¿Cómo se hace una rectificación de partida?",
        "referencia": "Sacar turno en el registro civil más cercano. Presentarse con la documentación requerida. El registro remite el trámite a la Dirección General, donde se genera un expediente y se evalúa. Si se observa el trámite, se notifica por correo electrónico; si se aprueba, se emite la resolución. La resolución se envía al registro civil correspondiente para marginar el acta. Luego se solicita una copia actualizada a través de Ciudadano Digital.",
        "datos_clave": ["turno", "Ciudadano Digital"],
        "debe_abstenerse": False,
        "fuentes": ["rectificacion_partidas"],
    },
    {
        "id": "P08",
        "categoria": "procedimiento",
        "texto": "¿Cómo se inscribe el nacimiento de un bebé?",
        "referencia": "El trámite es presencial en el registro civil correspondiente al lugar de nacimiento. Pasos: sacar turno en el registro civil correspondiente, asistir con la documentación el día del turno, y hacer la inscripción y entrega, donde se realiza el acta y se puede iniciar el DNI.",
        "datos_clave": ["presencial", "turno"],
        "debe_abstenerse": False,
        "fuentes": ["nacimientos"],
    },
    {
        "id": "P09",
        "categoria": "procedimiento",
        "texto": "¿El trámite de partidas se puede iniciar por internet?",
        "referencia": "Sí. La solicitud de actas se inicia de manera digital en el servicio Mi Registro Civil.",
        "datos_clave": ["digital"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "P10",
        "categoria": "procedimiento",
        "texto": "¿Cómo configuro Adobe Reader para validar documentos?",
        "referencia": "Desde Edición > Preferencias, siguiendo la guía indicada en el documento.",
        "datos_clave": ["Preferencias"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    # ==================================================================
    # MULTI SECCION (10)
    # Combina secciones distintas del mismo documento o de varios.
    # ==================================================================
    {
        "id": "M01",
        "categoria": "multi_seccion",
        "texto": "¿Cuánto cuesta una copia certificada y qué necesito para pedirla?",
        "referencia": "La copia de acta certificada cuesta $1.100 según la Ley Impositiva 2024 (Ley 10.929, artículo 79). Se requiere cuenta de Ciudadano Digital y los datos del acta para que la búsqueda sea exitosa.",
        "datos_clave": ["1.100", "Ciudadano Digital"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "M02",
        "categoria": "multi_seccion",
        "texto": "¿Cuánto cuesta una copia legalizada y cómo se solicita?",
        "referencia": "La copia de acta legalizada cuesta $1.500 según la Ley Impositiva 2024 (Ley 10.929, artículo 79). Las actas legalizadas se emiten únicamente con firma digital. La solicitud se inicia de manera digital.",
        "datos_clave": ["1.500", "firma digital"],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "M03",
        "categoria": "multi_seccion",
        "texto": "¿Qué trámites son gratis y cuáles tienen costo?",
        "referencia": "La inscripción de nacimientos y la inscripción de defunciones son gratuitas. La copia de acta certificada cuesta $1.100 y la legalizada $1.500. Para matrimonios y rectificaciones la fuente no declara el monto.",
        "datos_clave": ["gratuito", "1.100"],
        "debe_abstenerse": False,
        "fuentes": ["nacimientos", "defunciones", "partidas"],
    },
    {
        "id": "M04",
        "categoria": "multi_seccion",
        "texto": "Mi bebé nació hace más de un mes, ¿qué tengo que hacer y qué documentación llevo?",
        "referencia": "El plazo legal es de 40 días corridos desde el nacimiento. Entre el día 41 y el 60, la inscripción la realiza de oficio el registro civil de la localidad de nacimiento, también debe ser iniciada por los progenitores y con los mismos requisitos, más el certificado negativo de inscripción emitido por ese registro civil. La documentación es certificado médico de nacimiento o certificado de nacido vivo, y documento de identidad de ambos progenitores.",
        "datos_clave": ["40", "41", "certificado negativo"],
        "debe_abstenerse": False,
        "fuentes": ["nacimientos"],
    },
    {
        "id": "M05",
        "categoria": "multi_seccion",
        "texto": "Quiero casarme: ¿con cuánta anticipación saco el turno y qué llevo?",
        "referencia": "Los turnos se solicitan con dos meses de anticipación en la Oficina de Matrimonio Casa Central, y al menos uno de los contrayentes debe tener domicilio en La Rioja Capital. Llevar DNI original y vigente de ambos contrayentes, actas de nacimiento actualizadas, y DNI de los testigos, que son mínimo 2 y máximo 4 mayores de edad.",
        "datos_clave": ["dos meses", "2", "4", "La Rioja"],
        "debe_abstenerse": False,
        "fuentes": ["matrimonios"],
    },
    {
        "id": "M06",
        "categoria": "multi_seccion",
        "texto": "¿Cuánto cuesta casarse?",
        "referencia": "La fuente menciona que el costo surge de la Ley Impositiva N° 10.929, artículo 79, año 2024, pero los montos aparecen literalmente como '$??' en la página oficial. La fuente no declara el valor. Agrega que los costos pueden variar según la modalidad y los servicios adicionales solicitados.",
        "datos_clave": ["10.929"],
        "debe_abstenerse": False,
        "fuentes": ["matrimonios"],
    },
    {
        "id": "M07",
        "categoria": "multi_seccion",
        "texto": "¿Cuánto cuesta rectificar una partida?",
        "referencia": "La fuente declara el costo por la registración de notas de referencia, pero el monto aparece literalmente como '$??'. La fuente no declara el valor.",
        "datos_clave": ["notas de referencia"],
        "debe_abstenerse": False,
        "fuentes": ["rectificacion_partidas"],
    },
    {
        "id": "M08",
        "categoria": "multi_seccion",
        "texto": "Falleció una persona sin DNI, ¿cómo se hace la inscripción?",
        "referencia": "Si la persona nacida con vida, se debe realizar la inscripción del nacimiento, la inscripción de la defunción y presentar el DNI de los progenitores. En caso de muerte fetal se necesitan el DNI de los progenitores y el Certificado de Defunción fetal. En general, la inscripción debe realizarse dentro de las 48 horas hábiles.",
        "datos_clave": ["48", "nacimiento"],
        "debe_abstenerse": False,
        "fuentes": ["defunciones"],
    },
    {
        "id": "M09",
        "categoria": "multi_seccion",
        "texto": "¿Qué pasa si estoy soltero pero ya estoy unión por el registro civil?",
        "referencia": "Para inscribir el nacimiento, si los progenitores están casados deben presentar la libreta de familia o los datos de inscripción del acta de matrimonio para poder localizar el acta y adjuntarla al trámite. Si uno de los progenitores no posee documento, debe acreditar su identidad con 2 testigos mayores de 18 años con DNI vigente.",
        "datos_clave": ["libreta de familia", "testigos"],
        "debe_abstenerse": False,
        "fuentes": ["nacimientos", "matrimonios"],
    },
    {
        "id": "M10",
        "categoria": "multi_seccion",
        "texto": "Me casé en La Rioja y perdí el acta, ¿qué hago?",
        "referencia": "Para inscribir un matrimonio hay que presentar el acta de matrimonio con marginal de divorcio si el contrayente es divorciado. Si el acta se perdió, no hay un trámite de reposición específico en el corpus. Para weddings con estado civil de viudo/a se presentan el acta de defunción y el acta de matrimonio del cónyuge anterior.",
        "datos_clave": ["marginal"],
        "debe_abstenerse": False,
        "fuentes": ["matrimonios"],
    },
    # ==================================================================
    # AMBIGUA (10)
    # La pregunta esta incompleta o mezcla dos tramites. La conducta
    # correcta es responder cubriendo la ambiguedad o pedir
    # precisiones, no elegir un tramite por su cuenta.
    # ==================================================================
    {
        "id": "A01",
        "categoria": "ambigua",
        "texto": "Necesito un papel del Registro Civil. ¿Qué me dan?",
        "referencia": "El Registro Civil emite partidas de nacimiento, matrimonio y defunción, y sus copias certificadas, legalizadas y simples. La pregunta no dice cuál de ellos, así que la respuesta debe=listar las opciones o pedir precisiones.",
        "datos_clave": [],
        "debe_abstenerse": False,
        "fuentes": ["guia_tramites"],
    },
    {
        "id": "A02",
        "categoria": "ambigua",
        "texto": "¿Cuánto sale un trámite?",
        "referencia": "Depende del trámite: la inscripción de nacimientos y la de defunciones son gratuitas; la copia certificada cuesta $1.100 y la legalizada $1.500. Para matrimonios y rectificaciones la fuente no declara el monto. La respuesta debe detallar en lugar de dar un único precio.",
        "datos_clave": ["gratuito", "1.100"],
        "debe_abstenerse": False,
        "fuentes": ["nacimientos", "defunciones", "partidas", "matrimonios"],
    },
    {
        "id": "A03",
        "categoria": "ambigua",
        "texto": "Tengo que llevar a mi bebé al registro civil, ¿qué necesito?",
        "referencia": "No está claro si se refiere a inscribir el nacimiento del bebé o a sacar su partida. Si es la inscripción: certificado médico de nacimiento o certificado de nacido vivo, DNI de ambos progenitores, y el trámite es presencial con turno previo. Si es la partida: cuenta de Ciudadano Digital y los datos del acta.",
        "datos_clave": [],
        "debe_abstenerse": False,
        "fuentes": ["nacimientos", "partidas"],
    },
    {
        "id": "A04",
        "categoria": "ambigua",
        "texto": "¿Cuánto tarda?",
        "referencia": "La pregunta no dice qué trámite. El corpus solo declara plazos de vencimiento, no de entrega: 40 días corridos para inscribir un nacimiento, 48 horas hábiles para inscribir una defunción y dos meses de anticipación para el turno de matrimonio. El tiempo de entrega de una partida no está en la fuente.",
        "datos_clave": [],
        "debe_abstenerse": False,
        "fuentes": [],
    },
    {
        "id": "A05",
        "categoria": "ambigua",
        "texto": "¿Puedo pedir una partida si soy de otra provincia?",
        "referencia": "La fuente indica que el servicio permite solicitar copias de partidas inscriptas en la Provincia de La Rioja, y no aclara el caso de quien no es de la provincia. La respuesta debe decir que la fuente no lo aclara.",
        "datos_clave": [],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    {
        "id": "A06",
        "categoria": "ambigua",
        "texto": "¿Dónde queda el registro civil?",
        "referencia": "La fuente menciona la Oficina de Matrimonio Casa Central, el registro civil correspondiente a la localidad de nacimiento, el registro civil de la localidad de fallecimiento y el registro civil más cercano al domicilio. No da direcciones ni horarios. La respuesta debe decirlo así.",
        "datos_clave": [],
        "debe_abstenerse": False,
        "fuentes": ["matrimonios", "nacimientos", "defunciones"],
    },
    {
        "id": "A07",
        "categoria": "ambigua",
        "texto": "Quiero cambiar un dato de mi partida. ¿Me lo pueden cambiar?",
        "referencia": "Sí, mediante rectificación de partidas, que corrige errores u omisiones materiales en actas inscriptas en La Rioja. Es presencial, con turno en el registro civil más cercano al domicilio. La respuesta debe describir el trámite y no prometer un resultado.",
        "datos_clave": ["rectificación", "turno"],
        "debe_abstenerse": False,
        "fuentes": ["rectificacion_partidas"],
    },
    {
        "id": "A08",
        "categoria": "ambigua",
        "texto": "¿Me pueden pasar con un libro de familia?",
        "referencia": "La fuente menciona la libreta de familia solo en un caso concreto: cuando los progenitores están casados y van a inscribir un nacimiento, deben presentar la libreta de familia o los datos de inscripción del acta de matrimonio. No hay un trámite general de entrega de libretas de familia en el corpus.",
        "datos_clave": ["libreta de familia", "nacimiento"],
        "debe_abstenerse": False,
        "fuentes": ["nacimientos"],
    },
    {
        "id": "A09",
        "categoria": "ambigua",
        "texto": "Estoy vencido y no sé qué trámite es.",
        "referencia": "La pregunta es ambigua: hay plazos para inscribir un nacimiento, para inscribir una defunción y para pedir turnos de matrimonio. La fuente no permite determinar de qué trámite se trata. La respuesta debe pedir precisiones.",
        "datos_clave": [],
        "debe_abstenerse": False,
        "fuentes": [],
    },
    {
        "id": "A10",
        "categoria": "ambigua",
        "texto": "¿Puedo pedir el acta de millasación si me casé en otra provincia?",
        "referencia": "La fuente trata de las partidas inscriptas en la Provincia de La Rioja y no aclara el caso de un matrimonio celebrado en otra provincia. La respuesta debe decir que la fuente no lo aclara.",
        "datos_clave": [],
        "debe_abstenerse": False,
        "fuentes": ["partidas"],
    },
    # ==================================================================
    # FUERA DE CORPUS (10)
    # La respuesta NO esta en las fuentes. Acertar es abstenerse.
    # Un sistema que contesta estas diez tiene un 0% de honestidad.
    # ==================================================================
    {
        "id": "F01",
        "categoria": "fuera_de_corpus",
        "texto": "¿A qué hora atención el Registro Civil?",
        "referencia": "La fuente no declara horarios de atención.",
        "datos_clave": [],
        "debe_abstenerse": True,
        "fuentes": [],
    },
    {
        "id": "F02",
        "categoria": "fuera_de_corpus",
        "texto": "¿Cuál es el teléfono del Registro Civil de La Rioja?",
        "referencia": "La fuente no declara teléfonos de contacto.",
        "datos_clave": [],
        "debe_abstenerse": True,
        "fuentes": [],
    },
    {
        "id": "F03",
        "categoria": "fuera_de_corpus",
        "texto": "¿Cuánto tarda en estar lista una partida que pedí?",
        "referencia": "La fuente no declara plazos de entrega de las partidas.",
        "datos_clave": [],
        "debe_abstenerse": True,
        "fuentes": [],
    },
    {
        "id": "F04",
        "categoria": "fuera_de_corpus",
        "texto": "¿Cómo hago el pasaporte argentino?",
        "referencia": "La guía de trámites menciona el pasaporte en una línea, pero no describe ningún requisito, costo ni procedimiento.",
        "datos_clave": [],
        "debe_abstenerse": True,
        "fuentes": ["guia_tramites"],
    },
    {
        "id": "F05",
        "categoria": "fuera_de_corpus",
        "texto": "¿Cómo saco el DNI por primera vez a mi recién nacido?",
        "referencia": "La fuente menciona que al inscribir el nacimiento se puede iniciar el DNI del bebé junto con el acta, pero no describe el trámite del DNI en sí.",
        "datos_clave": [],
        "debe_abstenerse": True,
        "fuentes": ["nacimientos", "guia_tramites"],
    },
    {
        "id": "F06",
        "categoria": "fuera_de_corpus",
        "texto": "¿Cómo se hace la unión convivencial?",
        "referencia": "La guía de trámites menciona la inscripción de unión convivencial en una línea, pero no describe requisitos, costo ni procedimiento.",
        "datos_clave": [],
        "debe_abstenerse": True,
        "fuentes": ["guia_tramites"],
    },
    {
        "id": "F07",
        "categoria": "fuera_de_corpus",
        "texto": "¿Qué hago para sumar un apellido a mi nombre?",
        "referencia": "La guía de trámites menciona la adición de apellido en una línea, pero no describe el procedimiento.",
        "datos_clave": [],
        "debe_abstenerse": True,
        "fuentes": ["guia_tramites"],
    },
    {
        "id": "F08",
        "categoria": "fuera_de_corpus",
        "texto": "Quiero reconocer a mi hijo. ¿Cómo se hace?",
        "referencia": "La guía de trámites menciona el reconocimiento de paternidad o maternidad en una sola línea, pero no describe ningún requisito, costo ni procedimiento.",
        "datos_clave": [],
        "debe_abstenerse": True,
        "fuentes": ["guia_tramites"],
    },
    {
        "id": "F09",
        "categoria": "fuera_de_corpus",
        "texto": "¿Cuánto cuesta el turno para casarse en el Registro Civil?",
        "referencia": "La fuente no declara el costo del turno de matrimonio. Solo declara los montos de la Ley Impositiva, que aparecen sin valor.",
        "datos_clave": [],
        "debe_abstenerse": True,
        "fuentes": ["matrimonios"],
    },
    {
        "id": "F10",
        "categoria": "fuera_de_corpus",
        "texto": "¿Se puede hacer el trámite de partidas un fin de semana?",
        "referencia": "La fuente no declara días y horarios de atención, ni si hay algún régimen especial.",
        "datos_clave": [],
        "debe_abstenerse": True,
        "fuentes": [],
    },
]
