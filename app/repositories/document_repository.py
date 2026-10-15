"""Persistencia del corpus documental.

El corpus vive en `data/processed/` como JSON, que es versionable y
legible sin herramientas. SQLite queda para lo que es consulta
frecuente y relacional: consultas, metricas y evaluaciones.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from app.models.schemas import EstadoFuente, MetadatosDocumento
from app.services.corpus import (
    FuenteDeclarada,
    Manifiesto,
    construir_documento,
    detectar_duplicados,
    resumir_vigencia,
)


class DocumentoRepository:
    """Guarda y carga el corpus normalizado."""

    def __init__(self, directorio: Path) -> None:
        self.directorio = directorio
        self.directorio.mkdir(parents=True, exist_ok=True)

    @property
    def ruta_corpus(self) -> Path:
        return self.directorio / "corpus.json"

    @property
    def ruta_manifiesto(self) -> Path:
        return self.directorio.parent / "manifest.json"

    # -----------------------------------------------------------------
    # Escritura
    # -----------------------------------------------------------------
    def guardar_documento(
        self,
        documento: MetadatosDocumento,
        contenido: str,
        secciones: list[str] | None = None,
        fecha_consulta: date | None = None,
    ) -> None:
        """Agrega o reemplaza un documento en el corpus.

        Reemplazar en vez de duplicar es correcto porque el identificador
        es estable: si el mismo documento cambio, se actualiza el mismo
        registro y el hash deja registro del cambio.
        """
        corpus = self._cargar_crudo()
        entradas = corpus["documentos"]

        entrada = {
            # Los metadatos se guardan limpios: el modelo usa
            # `extra="forbid"`, asi que cualquier campo extra aqui haria
            # fallar la relectura del corpus.
            "metadatos": documento.model_dump(mode="json"),
            "contenido": contenido,
            "secciones": secciones or [],
            "historial_ingesta": [],
        }

        for indice, existente in enumerate(entradas):
            if existente["metadatos"]["document_id"] == documento.document_id:
                # Se conserva el historial de ingestas: que un documento haya
                # cambiado de contenido es informacion relevante para auditar.
                entrada["historial_ingesta"] = existente.get("historial_ingesta", [])
                cambio = existente["metadatos"].get("hash_contenido")
                if cambio and cambio != documento.hash_contenido:
                    entrada["historial_ingesta"].append(
                        {
                            "hash_anterior": cambio,
                            "fecha": fecha_consulta.isoformat() if fecha_consulta else None,
                            "cambio": "contenido_distinto",
                        }
                    )
                entradas[indice] = entrada
                break
        else:
            entradas.append(entrada)

        corpus["documentos"] = entradas
        self._guardar_crudo(corpus)

    def procesar_archivo(
        self,
        ruta: Path,
        fuente: FuenteDeclarada,
        fecha_consulta: date | None = None,
    ) -> MetadatosDocumento:
        """Ingesta un archivo, lo guarda y actualiza el manifiesto."""
        from app.services.ingestion import extraer_documento

        # Se extrae una sola vez y se reutiliza: `construir_documento` vuelve a
        # leer el archivo para calcular el hash, pero parsear dos veces un
        # PDF pesado es tiempo desperdiciado.
        contenido, extraidos = extraer_documento(ruta)
        metadatos = construir_documento(ruta, fuente, fecha_consulta)

        self.guardar_documento(
            metadatos,
            contenido,
            extraidos.get("secciones"),
            fecha_consulta or metadatos.fecha_consulta,
        )

        manifiesto = Manifiesto.cargar(self.ruta_manifiesto)
        manifiesto.registrar(
            {
                "url": fuente.url or f"archivo://{metadatos.archivo_original}",
                "titulo": metadatos.titulo,
                "organismo": metadatos.organismo,
                "tramite": metadatos.tramite,
                "document_id": metadatos.document_id,
                "tipo_documento": metadatos.tipo_documento.value,
                "archivo_original": metadatos.archivo_original,
                "estado_verificacion": metadatos.estado_verificacion.value,
                "hash_contenido": metadatos.hash_contenido,
                "fecha_consulta": metadatos.fecha_consulta.isoformat(),
                "fecha_publicacion": (
                    metadatos.fecha_publicacion.isoformat() if metadatos.fecha_publicacion else None
                ),
                "fecha_actualizacion": (
                    metadatos.fecha_actualizacion.isoformat()
                    if metadatos.fecha_actualizacion
                    else None
                ),
                "vigencia": metadatos.vigencia.value,
                "fecha_ultima_revision": date.today().isoformat(),
                "fecha_incorporacion": date.today().isoformat(),
                "observaciones": metadatos.observaciones,
            }
        )
        manifiesto.guardar(self.ruta_manifiesto)
        return metadatos

    def limpiar(self) -> int:
        """Borra el corpus y devuelve cuantos documentos habia.

        Sin esto, quitar un archivo de `data/raw/` no lo saca del corpus:
        `guardar_documento` agrega y reemplaza, pero nunca borra. Con el
        tiempo el corpus deriva de las fuentes reales y deja de ser
        reproducible desde `data/raw/`.
        """
        if not self.ruta_corpus.exists():
            return 0
        anterior = len(self._cargar_crudo()["documentos"])
        self.ruta_corpus.unlink()
        return anterior

    # -----------------------------------------------------------------
    # Lectura
    # -----------------------------------------------------------------
    def cargar_documentos(self) -> list[MetadatosDocumento]:
        corpus = self._cargar_crudo()
        return [MetadatosDocumento(**d["metadatos"]) for d in corpus["documentos"]]

    def cargar_contenidos(self) -> dict[str, str]:
        """Devuelve `document_id -> contenido normalizado`."""
        corpus = self._cargar_crudo()
        return {d["metadatos"]["document_id"]: d["contenido"] for d in corpus["documentos"]}

    def cargar_entradas(self) -> list[dict[str, Any]]:
        """Devuelve el corpus completo con metadatos, contenido y secciones."""
        return list(self._cargar_crudo()["documentos"])

    def por_tramite(self, tramite: str | None = None) -> list[dict[str, Any]]:
        """Filtra el corpus por tramite. Sin tramite, devuelve todo."""
        entradas = self.cargar_entradas()
        if not tramite:
            return entradas
        return [e for e in entradas if e["metadatos"]["tramite"] == tramite]

    def documentos_validos(self) -> list[MetadatosDocumento]:
        """Documentos utilizables: los que no estan marcados como inaccesibles."""
        return [
            d
            for d in self.cargar_documentos()
            if d.estado_verificacion != EstadoFuente.NO_ACCESIBLE
        ]

    # -----------------------------------------------------------------
    # Diagnostico
    # -----------------------------------------------------------------
    def informe(self) -> dict[str, Any]:
        """Resumen del estado del corpus, para el manifiesto y la consola."""
        documentos = self.cargar_documentos()
        duplicados = detectar_duplicados(documentos)

        por_tramite: dict[str, int] = {}
        for documento in documentos:
            por_tramite[documento.tramite] = por_tramite.get(documento.tramite, 0) + 1

        caracteres = sum(len(c) for c in self.cargar_contenidos().values())

        return {
            "total_documentos": len(documentos),
            "por_tramite": por_tramite,
            "por_vigencia": resumir_vigencia(documentos),
            "duplicados_por_contenido": [list(g) for g in duplicados],
            "caracteres_totales": caracteres,
            "tokens_aproximados": caracteres // 4,
            "sin_url_de_origen": [d.document_id for d in documentos if not d.url_origen],
        }

    # -----------------------------------------------------------------
    # Interno
    # -----------------------------------------------------------------
    def _cargar_crudo(self) -> dict[str, Any]:
        if not self.ruta_corpus.exists():
            return {"documentos": []}
        try:
            return json.loads(self.ruta_corpus.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise

    def _guardar_crudo(self, corpus: dict[str, Any]) -> None:
        self.ruta_corpus.write_text(
            json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def cargar_manifiesto(directorio: Path) -> Manifiesto:
    """Atajo para leer el manifiesto desde cualquier punto del proyecto."""
    return Manifiesto.cargar(directorio.parent / "manifest.json")


def documentos_desde(entradas: Iterable[dict[str, Any]]) -> list[MetadatosDocumento]:
    """Convierte entradas crudas del corpus en modelos validados."""
    return [MetadatosDocumento(**e["metadatos"]) for e in entradas]
