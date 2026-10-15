"""Construccion del corpus documental a partir de archivos y fuentes.

Separa dos conceptos que el enunciado pide no confundir:

- **Identificador estable**: derivado de la procedencia (URL o nombre de
  archivo). No cambia aunque el contenido cambie, para que las citas de
  respuestas anteriores sigan apuntando al mismo documento.

- **Hash de contenido**: cambia si el documento cambio. Es lo que permite
  detectar que una fuente se actualizo y reconstruir los indices.

Si el identificador se derivara del contenido, cada actualizacion de la
fuente invalidaria todas las citas ya emitidas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from app.models.schemas import (
    EstadoFuente,
    EstadoVigencia,
    MetadatosDocumento,
    TipoDocumento,
)
from app.services.ingestion import (
    IngestionError,
    evaluar_vigencia,
    extraer_documento,
)


@dataclass
class FuenteDeclarada:
    """Como el desarrollador declara una fuente antes de incorporarla.

    Los metadatos que la pagina declara se complementan con lo que el
    desarrollador sabe y la pagina no dice (el organismo responsable, a que
    tramite pertenece). Lo que ninguno de los dos aclare queda en `None`.
    """

    url: str
    tramite: str
    organismo: str = "Registro Civil de La Rioja"
    titulo: str | None = None
    # Allowlist de rutas: evita que un link dinamicamente generado termine
    # descargando un endpoint interno del sitio.
    permitido: bool = True
    observaciones: str | None = None
    fecha_consulta: date = field(default_factory=date.today)


def _slug(texto: str) -> str:
    """Convierte un texto en un fragmento seguro para nombres de archivo."""
    limpio = re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")
    return limpio or "documento"


def identificador_estable(url: str | None, nombre_archivo: str) -> str:
    """Genera el identificador estable de un documento.

    Se deriva de la procedencia, nunca del contenido: es lo que permite
    que una actualizacion de la fuente se detecte como *cambio* en vez de
    aparecer como un documento nuevo sin relacion con el anterior.
    """
    if url:
        base = re.sub(r"^https?://", "", url).rstrip("/")
        return f"doc-{_slug(base)[:64]}"
    return f"doc-{_slug(Path(nombre_archivo).stem)}"


def nombre_archivo_desde_url(url: str) -> str:
    """Deriva un nombre de archivo legible a partir de la URL."""
    sin_esquema = re.sub(r"^https?://", "", url)
    sin_query = sin_esquema.split("?")[0].split("#")[0]
    ultimo = sin_query.rstrip("/").split("/")[-1]
    if ultimo and "." in ultimo:
        return ultimo
    # Paginas sin extension (por ejemplo "/tramites") reciben un nombre estable.
    return f"{_slug(ultimo or sin_query) or 'pagina'}.html"


def construir_documento(
    ruta: Path,
    fuente: FuenteDeclarada,
    fecha_consulta: date | None = None,
) -> MetadatosDocumento:
    """Procesa un archivo y devuelve sus metadatos consolidados.

    Lanza `IngestionError` si el archivo no se puede procesar. No se
    degrada a un documento vacio en silencio: entraria al corpus y el agente
    se abstendria por una causa que nadie registro.
    """
    contenido, extraidos = extraer_documento(ruta)

    if not contenido.strip():
        raise IngestionError(f"El documento quedo vacio tras normalizar: {ruta.name}")

    nombre_original = ruta.name
    fecha_consulta = fecha_consulta or fuente.fecha_consulta

    fecha_publicacion = _coerzar_fecha(extraidos.get("fecha_publicacion"))
    fecha_actualizacion = _coerzar_fecha(
        extraidos.get("fecha_actualizacion")
        or extraidos.get("fecha_modificacion_pdf")
        or extraidos.get("fecha_modificacion_docx")
    )

    titulo = fuente.titulo or extraidos.get("titulo") or nombre_original
    estado = EstadoFuente.VERIFICADA if fuente.permitido else EstadoFuente.NO_ACCESIBLE

    return MetadatosDocumento(
        document_id=identificador_estable(fuente.url, nombre_original),
        titulo=str(titulo).strip(),
        organismo=fuente.organismo,
        tramite=fuente.tramite,
        tipo_documento=extraidos.get("tipo_documento", TipoDocumento.DESCONOCIDO),
        url_origen=fuente.url,
        archivo_original=nombre_original,
        fecha_consulta=fecha_consulta,
        fecha_publicacion=fecha_publicacion,
        fecha_actualizacion=fecha_actualizacion,
        # La vigencia se deriva solo de lo que la fuente declara.
        # No se infiere por estar publicada.
        vigencia=evaluar_vigencia(fecha_actualizacion, fecha_consulta),
        hash_contenido=str(extraidos["hash_contenido"]),
        estado_verificacion=estado,
        observaciones=fuente.observaciones,
    )


def _coerzar_fecha(valor: Any) -> date | None:
    """Acepta date, datetime o cadena ISO y devuelve date."""
    if valor is None:
        return None
    if isinstance(valor, date):
        return valor
    try:
        return date.fromisoformat(str(valor)[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Manifiesto de fuentes
# ---------------------------------------------------------------------------
class Manifiesto:
    """Registro de estado de cada fuente del corpus.

    El manifiesto responde a tres preguntas que el corpus por si solo no
    puede: de donde viene cada documento, si la fuente sigue siendo
    accesible y cuando se reviso por ultima vez.
    """

    VERSION = 1

    def __init__(self, datos: dict[str, Any] | None = None) -> None:
        self.datos: dict[str, Any] = datos or {
            "version": self.VERSION,
            "descripcion": (
                "Manifiesto de fuentes de TramitaIA. Cada entrada registra la "
                "procedencia y el estado de verificacion de un documento."
            ),
            "fuentes": [],
        }

    @property
    def fuentes(self) -> list[dict[str, Any]]:
        return self.datos.setdefault("fuentes", [])

    def registrar(self, entrada: dict[str, Any]) -> None:
        """Agrega o actualiza la entrada de una fuente, por URL."""
        url = entrada["url"]
        for existente in self.fuentes:
            if existente.get("url") == url:
                # Se conserva la primera fecha de incorporacion: el
                # historial de cuando aparecio la fuente importa para auditar.
                entrada.setdefault("fecha_incorporacion", existente.get("fecha_incorporacion"))
                existente.update(entrada)
                return
        self.fuentes.append(entrada)

    def marcar_estado(self, url: str, estado: EstadoFuente, nota: str | None = None) -> None:
        from datetime import date as _date

        for fuente in self.fuentes:
            if fuente.get("url") == url:
                fuente["estado_verificacion"] = estado.value
                fuente["fecha_ultima_revision"] = _date.today().isoformat()
                if nota:
                    fuente["observaciones"] = nota
                return

    def buscar_por_url(self, url: str) -> dict[str, Any] | None:
        for fuente in self.fuentes:
            if fuente.get("url") == url:
                return fuente
        return None

    def guardar(self, ruta: Path) -> None:
        import json

        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(
            json.dumps(self.datos, indent=2, ensure_ascii=False, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def cargar(cls, ruta: Path) -> Manifiesto:
        import json

        if not ruta.exists():
            return cls()
        try:
            return cls(json.loads(ruta.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            # Un manifiesto corrupto no debe borrar el historial: se conserva
            # con el nombre .corrupto y se empieza de cero, avisando.
            ruta.rename(ruta.with_suffix(".json.corrupto"))
            return cls()


def detectar_duplicados(documentos: list[MetadatosDocumento]) -> list[tuple[str, ...]]:
    """Agrupa documentos que comparten el mismo hash de contenido.

    Dos fuentes distintas con el mismo hash son el mismo documento. Se
    detectan en vez de descartarse en silencio, para que la decision de
    cual conservar quede registrada.
    """
    por_hash: dict[str, list[str]] = {}
    for documento in documentos:
        por_hash.setdefault(documento.hash_contenido, []).append(documento.document_id)

    grupos: list[tuple[str, ...]] = []
    for ids in por_hash.values():
        if len(ids) > 1:
            grupos.append(tuple(sorted(ids)))
    return grupos


def resumir_vigencia(documentos: list[MetadatosDocumento]) -> dict[str, int]:
    """Cuenta documentos por estado de vigencia, para el informe de ingesta."""
    conteo = dict.fromkeys(EstadoVigencia, 0)
    for documento in documentos:
        conteo[documento.vigencia] += 1
    return {estado.value: total for estado, total in conteo.items() if total}
