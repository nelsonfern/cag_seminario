"""Servicio CAG: contexto documental preparado y reutilizable.

CAG (Cache-Augmented Generation) envia al modelo un contexto documental
preparado una sola vez y lo reutiliza en cada consulta. El objetivo es que
el proveedor pueda cachear ese prefijo, de modo que las consultas
siguientes solo paguen lo nuevo.

Distingue dos cachés que el enunciado pide no confundir:

1. **La caché de aplicación**: el archivo en disco donde queda el contexto
   ya preparado. Es nuestra y la controlamos enteramente.
2. **La caché de cómputo del proveedor**: la que reporta el servicio
   (Groq: `usage.prompt_tokens_details.cached_tokens`). No la controlamos
   y puede no existir.

Se miden y se guardan por separado, porque miden cosas distintas: la
primera es trabajo nuestro que ahorramos, la segunda es un ahorro real
facturado por el proveedor.

Una consecuencia no obvia: **el orden del contexto determina los aciertos
de caché**. Si el prefijo cambia entre consultas, el proveedor no puede
reutilizar nada. Por eso los documentos se ordenan de forma determinista y
el presupuesto se aplica siempre igual.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.schemas import MetadatosDocumento
from app.repositories.document_repository import DocumentoRepository
from app.services.citations import DocumentoEnContexto
from app.services.ingestion import envolver_como_contexto
from app.services.llm import estimar_tokens

logger = get_logger("cag")

VERSION_CONTEXTO = 1


@dataclass
class ContextoPreparado:
    """Contexto listo para enviarse al modelo."""

    contexto: str
    documentos: list[DocumentoEnContexto] = field(default_factory=list)

    # --- Trazabilidad ---
    #: Hash del corpus en el momento de preparar. Si cambia, la cache de
    #: aplicacion queda invalidada.
    hash_corpus: str = ""
    #: Version del formato de contexto. Cambiarla invalida la cache aunque
    #: el corpus no haya cambiado, porque el texto es otro.
    version_formato: int = VERSION_CONTEXTO
    #: Tramites incluidos, por si se prepared por particion.
    tramites: list[str] = field(default_factory=list)
    #: Nombre de la particion, si el corpus se dividio.
    particion: str | None = None

    # --- Metricas ---
    tokens_estimados: int = 0
    documentos_incluidos: int = 0
    documentos_omitidos: int = 0
    #: True si hubo que recortar para respetar el presupuesto de tokens.
    truncado: bool = False
    #: Tiempo de construccion del contexto, en milisegundos.
    #: Es 0.0 cuando el contexto se sirvio desde la cache de aplicacion:
    #: reportar aqui el tiempo de una preparacion anterior falsearia la
    #: comparacion entre una ejecucion fria y una caliente.
    preparacion_ms: float = 0.0
    #: Tiempo que tardo en devolver el contexto, ya sea construyolo o
    #: leyendolo de la cache.
    entrega_ms: float = 0.0
    #: True si el contexto vino de la cache de aplicacion.
    desde_cache: bool = False

    def resumen(self) -> dict[str, Any]:
        """Metricas del contexto, para logs y para el informe del experimento."""
        datos = asdict(self)
        datos.pop("contexto", None)
        datos.pop("documentos", None)
        datos["caracteres"] = len(self.contexto)
        return datos


class CAGService:
    """Prepara, cachea y entrega el contexto documental de CAG."""

    def __init__(
        self,
        repositorio: DocumentoRepository | None = None,
        ajustes: Settings | None = None,
    ) -> None:
        self.ajustes = ajustes or get_settings()
        self._repositorio = repositorio
        self._en_memoria: ContextoPreparado | None = None

    @property
    def repositorio(self) -> DocumentoRepository:
        if self._repositorio is None:
            self._repositorio = DocumentoRepository(self.ajustes.processed_dir)
        return self._repositorio

    @property
    def ruta_cache(self) -> Path:
        return self.ajustes.resolve_path(self.ajustes.cag_cache_dir) / "contexto.json"

    # -----------------------------------------------------------------
    # Huella del corpus
    # -----------------------------------------------------------------
    def hash_corpus(self) -> str:
        """Huella del contenido del corpus, independiente del orden.

        Se ordena por `document_id` para que reordenar los archivos en disco
        no invalide la cache sin motivo. Se usa el hash de contenido de
        cada documento, no su texto completo: cambiar una palabra en un
        documento de 20 KB tiene que notarse, y no tiene por qué depender
        del resto del archivo.
        """
        documentos = sorted(self.repositorio.cargar_documentos(), key=lambda d: d.document_id)
        if not documentos:
            return "corpus-vacio"

        partes = [f"v{VERSION_CONTEXTO}"]
        for documento in documentos:
            if documento.estado_verificacion.value == "no_accesible":
                continue
            partes.append(f"{documento.document_id}:{documento.hash_contenido}")
        return hashlib.sha256("|".join(partes).encode("utf-8")).hexdigest()[:16]

    # -----------------------------------------------------------------
    # Preparacion
    # -----------------------------------------------------------------
    def preparar(
        self,
        force: bool = False,
        tramites: Iterable[str] | None = None,
    ) -> ContextoPreparado:
        """Prepara el contexto reutilizable.

        Con `force` se reconstruye ignorando la cache. Sin `force`, se
        devuelve la cache solo si su huella coincide con la del corpus
        actual.
        """
        if not force and self._en_memoria is not None:
            if self._en_memoria.hash_corpus == self.hash_corpus():
                inicio_entrega = time.perf_counter()
                # Se devuelve una COPIA y el original queda intacto en
                # `_en_memoria`. Devolver el mismo objeto permitiria que
                # quien lo lee lo modificara y contaminara la cache para
                # las consultas siguientes.
                return replace(
                    self._en_memoria,
                    desde_cache=True,
                    # El tiempo de preparacion original no se arrastra: esta
                    # entrega fue una lectura de cache, y medir eso es lo que
                    # distingue una ejecucion caliente de una fria.
                    preparacion_ms=0.0,
                    entrega_ms=round((time.perf_counter() - inicio_entrega) * 1000, 3),
                )

        if not force:
            desde_disco = self._leer_cache()
            if desde_disco is not None and desde_disco.hash_corpus == self.hash_corpus():
                logger.info("Contexto CAG servido desde la cache de aplicacion")
                desde_disco = replace(
                    desde_disco,
                    desde_cache=True,
                    preparacion_ms=0.0,
                    entrega_ms=0.0,
                )
                self._en_memoria = desde_disco
                return desde_disco

        inicio = time.perf_counter()
        entradas = self.repositorio.cargar_entradas()

        documentos = [
            DocumentoEnContexto(
                metadatos=MetadatosDocumento(**e["metadatos"]),
                contenido=e["contenido"],
            )
            for e in entradas
            if e["metadatos"]["estado_verificacion"] != "no_accesible"
        ]

        if not documentos:
            contexto = ContextoPreparado(
                contexto="",
                documentos=[],
                hash_corpus=self.hash_corpus(),
                preparacion_ms=(time.perf_counter() - inicio) * 1000,
            )
            self._en_memoria = contexto
            logger.warning("El corpus no tiene documentos utilizables")
            return contexto

        # Orden determinista: es lo que permite que el proveedor cachee el
        # prefijo entre consultas. Ordenar por fecha o por orden de carga
        # haria que el prefijo cambiase y la cache no serviria de nada.
        documentos.sort(key=lambda d: d.metadatos.document_id)

        if tramites:
            seleccion = {t.strip() for t in tramites}
            documentos = [d for d in documentos if d.metadatos.tramite in seleccion]
        else:
            documentos = self._seleccionar_por_presupuesto(documentos)

        cuerpo, truncado = self._armar_contexto(documentos)
        omitidos = len(
            [e for e in entradas if e["metadatos"]["estado_verificacion"] != "no_accesible"]
        ) - len(documentos)

        preparado = ContextoPreparado(
            contexto=cuerpo,
            documentos=documentos,
            hash_corpus=self.hash_corpus(),
            tramites=sorted({d.metadatos.tramite for d in documentos}),
            tokens_estimados=estimar_tokens(cuerpo),
            documentos_incluidos=len(documentos),
            documentos_omitidos=max(0, omitidos),
            truncado=truncado,
            preparacion_ms=round((time.perf_counter() - inicio) * 1000, 2),
            entrega_ms=round((time.perf_counter() - inicio) * 1000, 2),
            desde_cache=False,
        )

        self._en_memoria = replace(preparado, desde_cache=False)
        self._escribir_cache(preparado)
        logger.info(
            "Contexto CAG preparado: %d documentos, ~%d tokens%s en %.1f ms",
            preparado.documentos_incluidos,
            preparado.tokens_estimados,
            " (truncado)" if truncado else "",
            preparado.preparacion_ms,
        )
        return preparado

    def _seleccionar_por_presupuesto(
        self, documentos: list[DocumentoEnContexto]
    ) -> list[DocumentoEnContexto]:
        """Elige que documentos entran segun el presupuesto de tokens.

        No se parte el presupuesto a mitades entre tramites: eso produciria
        contextos distintos segun la consulta y el proveedor no podria
        reaprovechar el prefijo. Si no entran todos, entran los que
        caben **en orden estable**, y el sistema lo dice explicitamente.
        """
        presupuesto = self.ajustes.cag_max_context_tokens
        elegidos: list[DocumentoEnContexto] = []
        gastado = 0
        for documento in documentos:
            costo = estimar_tokens(documento.contenido)
            if gastado + costo > presupuesto:
                if not self.ajustes.cag_allow_partitioning:
                    break
                continue
            elegidos.append(documento)
            gastado += costo
        return elegidos

    def _armar_contexto(self, documentos: list[DocumentoEnContexto]) -> tuple[str, bool]:
        """Construye el texto del contexto y lo recorta si hace falta.

        Si hay que recortar, se recorta por documento entero y no a mitad de
        uno: una frase cortada produce citas sin sentido y affirmative
        entidades que el modelo podria completar por su cuenta.
        """
        partes: list[str] = []
        tokens = 0
        presupuesto = self.ajustes.cag_max_context_tokens
        truncado = False

        for indice, documento in enumerate(documentos, start=1):
            meta = documento.metadatos
            cabecera = (
                f"--- Documento {indice} de {len(documentos)} ---\n"
                f"Titulo: {meta.titulo}\n"
                f"Organismo: {meta.organismo}\n"
                f"Tramite: {meta.tramite}\n"
                f"Fecha de actualizacion declarada: "
                f"{meta.fecha_actualizacion.isoformat() if meta.fecha_actualizacion else 'no declarada'}\n"
                f"Vigencia: {meta.vigencia.value}\n"
            )
            bloque = cabecera + envolver_como_contexto(documento.contenido)
            costo = estimar_tokens(bloque)

            if tokens + costo > presupuesto:
                truncado = True
                logger.warning(
                    "Contexto CAG truncado en el documento %s (%s)",
                    meta.document_id,
                    meta.titulo,
                )
                break

            partes.append(bloque)
            tokens += costo

        cuerpo = "\n\n".join(partes)
        return cuerpo, truncado

    # -----------------------------------------------------------------
    # Cache de aplicacion
    # -----------------------------------------------------------------
    def invalidar(self) -> None:
        """Descarta la cache en memoria y en disco."""
        self._en_memoria = None
        if self.ruta_cache.exists():
            self.ruta_cache.unlink()
            logger.info("Cache de aplicacion CAG invalidada")

    def _escribir_cache(self, contexto: ContextoPreparado) -> None:
        """Guarda el contexto en disco.

        Se guarda tambien una copia de los documentos que lo componen, no
        solo el texto: el agente los necesita para verificar la respuesta,
        y reconstruirlos por separado podria dar una version distinta del
        corpus que se envio al modelo.
        """
        self.ruta_cache.parent.mkdir(parents=True, exist_ok=True)
        self.ruta_cache.write_text(
            json.dumps(
                {
                    "version": VERSION_CONTEXTO,
                    "hash_corpus": contexto.hash_corpus,
                    "preparado_en": datetime.now(UTC).isoformat(),
                    "contexto": contexto.contexto,
                    "resumen": contexto.resumen(),
                    "documentos": [
                        {
                            "metadatos": d.metadatos.model_dump(mode="json"),
                            "contenido": d.contenido,
                        }
                        for d in contexto.documentos
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _leer_cache(self) -> ContextoPreparado | None:
        if not self.ruta_cache.exists():
            return None
        try:
            from app.models.schemas import MetadatosDocumento

            datos = json.loads(self.ruta_cache.read_text(encoding="utf-8"))
            documentos = [
                DocumentoEnContexto(
                    metadatos=MetadatosDocumento(**d["metadatos"]),
                    contenido=d["contenido"],
                )
                for d in datos["documentos"]
            ]
            resumen = dict(datos["resumen"])
            resumen.pop("caracteres", None)
            return ContextoPreparado(**resumen, contexto=datos["contexto"], documentos=documentos)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Una cache ilegible no debe impedir trabajar: se descarta y
            # se vuelve a preparar. Perder un archivo de cache es barato;
            # no poder arrancar, no.
            logger.warning("Cache CAG ilegible; se regenerara")
            self.ruta_cache.unlink(missing_ok=True)
            return None

    # -----------------------------------------------------------------
    # Particionado
    # -----------------------------------------------------------------
    def particiones_disponibles(self) -> dict[str, dict[str, Any]]:
        """Tamano de cada tramite, para decidir como partir el corpus."""
        totales: dict[str, dict[str, Any]] = {}
        for documento in self.repositorio.cargar_documentos():
            if documento.estado_verificacion.value == "no_accesible":
                continue
            entrada = totales.setdefault(documento.tramite, {"documentos": 0, "tokens": 0})
            entrada["documentos"] += 1
            contenido = self.repositorio.cargar_contenidos().get(documento.document_id, "")
            entrada["tokens"] += estimar_tokens(contenido)
        return dict(sorted(totales.items()))

    def preparar_particion(self, tramite: str, force: bool = False) -> ContextoPreparado:
        """Prepara el contexto de un solo tramite.

        Es la via para mostrar el trade-off real de CAG: si el corpus crece
        hasta no entrar, hay que elegir que tramites entran en el
        contexto, y que informacion se queda afuera.
        """
        contexto = self.preparar(force=force, tramites=[tramite])
        contexto.particion = tramite
        return contexto

    # -----------------------------------------------------------------
    # Diagnostico
    # -----------------------------------------------------------------
    def estado(self) -> dict[str, Any]:
        """Estado de la cache de aplicacion, sin preparar nada."""
        existe = self.ruta_cache.exists()
        coincidencia: bool | None = None
        if existe:
            try:
                datos = json.loads(self.ruta_cache.read_text(encoding="utf-8"))
                coincidencia = datos.get("hash_corpus") == self.hash_corpus()
            except (json.JSONDecodeError, OSError):
                coincidencia = None

        return {
            "ruta_cache": str(self.ruta_cache),
            "existe": existe,
            # None = no se puede determinar. No se informa "vigente" por
            # defecto cuando no hay forma de comprobarlo.
            "cache_valida": coincidencia if existe else False,
            "hash_corpus_actual": self.hash_corpus(),
            "presupuesto_tokens": self.ajustes.cag_max_context_tokens,
            "particiones": self.particiones_disponibles(),
        }


_SERVICIO: CAGService | None = None


def obtener_servicio_cag() -> CAGService:
    """Servicio compartido, para no reconstruir el contexto en cada consulta."""
    global _SERVICIO
    if _SERVICIO is None:
        _SERVICIO = CAGService()
    return _SERVICIO


def reiniciar_servicio_cag() -> None:
    """Descarta el servicio cacheado. Lo usan las pruebas."""
    global _SERVICIO
    _SERVICIO = None
