"""Servicio RAG: recuperacion de fragmentos relevantes por consulta.

RAG (Retrieval-Augmented Generation) parte el corpus en fragmentos,
los indexa con embeddings y recupera solo los relevantes para cada
consulta. El contraste con CAG es el objeto del proyecto:

- **CAG** envia siempre el mismo contexto, que el proveedor puede cachear.
  Paga el prefijo completo una vez y reutiliza el resto.
- **RAG** envia un contexto distinto en cada consulta, porque los
  fragmentos recuperados cambian. No hay prefijo estable, asi que la
  cache del proveedor no puedeoscope. A cambio manda menos tokens.

Que la comparacion sea valida depende de que ambos usen **el mismo corpus,
el mismo modelo de lenguaje y el mismo sistema de verificacion**. Este
servicio carga los documentos del mismo repositorio que usa CAG.

Dos costos propio de RAG se miden aca y no existen en CAG:

1. El tiempo de recuperar (embeddings de la consulta + busqueda vectorial).
2. El tiempo de cargar el modelo de embeddings, que fue de **67 segundos
   en frio** en la maquina de desarrollo. Ocurre una sola vez por proceso
   y es la razon por la que el experimento tiene que separar ejecuciones
   frias de calientes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.schemas import EstadoFuente, EstadoVigencia, MetadatosDocumento, TipoDocumento
from app.repositories.document_repository import DocumentoRepository
from app.services.citations import DocumentoEnContexto
from app.services.ingestion import normalizar_texto

logger = get_logger("rag")

VERSION_INDICE = 1

#: Cabeceras de seccion dentro del texto, para atribuir cada fragmento a la
#: parte del documento de la que salio.
_RE_ENCABEZADO = "• "


@dataclass
class Fragmento:
    """Fragmento indexado, con su procedencia."""

    fragmento_id: str
    document_id: str
    contenido: str
    indice: int
    tramite: str
    organismo: str
    titulo: str
    seccion: str | None = None
    url_origen: str | None = None

    def a_metadatos(self) -> dict[str, Any]:
        """Metadatos para el indice vectorial.

        Chroma exige escalares: por eso no se guarda el contenido aqui sino
        aparte, en el JSON de fragmentos.
        """
        return {
            "document_id": self.document_id,
            "indice": self.indice,
            "tramite": self.tramite,
            "organismo": self.organismo,
            "titulo": self.titulo,
            "seccion": self.seccion or "",
            "url_origen": self.url_origen or "",
        }


@dataclass
class FragmentoRecuperado:
    """Fragmento devuelto por una consulta, con su puntuacion."""

    fragmento: Fragmento
    puntuacion: float
    #: Distancia del indice vectorial. Se conserva aunque no se exponga,
    #: porque es lo que permite revisar por que se recupero algo.
    distancia: float | None = None

    # Metadatos del documento de origen, necesarios para que el agente
    # pueda verificar la respuesta. Sin ellos no habria forma de saber de
    # que documento salio el fragmento.
    fecha_consulta: date = field(default_factory=date.today)
    vigencia: EstadoVigencia = EstadoVigencia.DESCONOCIDA
    tipo_documento: Any = TipoDocumento.HTML

    @property
    def document_id(self) -> str:
        return self.fragmento.document_id

    def a_contexto(self) -> DocumentoEnContexto:
        """Convierte el fragmento en algo verificable por el agente.

        Solo se expone el fragmento recuperado, no el documento entero. Si
        un dato esta en el documento pero no se recupero, el agente no
        podia usarlo y no debe aparentar que si.
        """
        return DocumentoEnContexto(
            metadatos=MetadatosDocumento(
                document_id=self.fragmento.document_id,
                titulo=self.fragmento.titulo,
                organismo=self.fragmento.organismo,
                tramite=self.fragmento.tramite,
                tipo_documento=self.tipo_documento,
                url_origen=self.fragmento.url_origen,
                archivo_original=f"{self.fragmento.document_id}.html",
                fecha_consulta=self.fecha_consulta,
                vigencia=self.vigencia,
                hash_contenido="",
            ),
            contenido=self.fragmento.contenido,
            fragmentos=[self.fragmento.contenido],
        )


@dataclass
class ResultadoBusqueda:
    """Resultado de recuperar fragmentos para una consulta."""

    consulta: str
    fragmentos: list[FragmentoRecuperado] = field(default_factory=list)

    # --- Metricas. Son el objeto del experimento. ---
    #: Tiempo total de recuperar, en milisegundos.
    recuperacion_ms: float = 0.0
    #: Tiempo de generar el embedding de la consulta.
    embedding_ms: float = 0.0
    #: Tiempo de buscar en el indice vectorial.
    busqueda_ms: float = 0.0
    #: Cantidad de fragmentos solicitados y obtenidos.
    solicitados: int = 0
    obtenidos: int = 0
    #: True si el indice se construyo en esta llamada.
    indice_reconstruido: bool = False
    #: Huella del corpus con el que se construyo el indice.
    hash_corpus: str = ""

    def resumen(self) -> dict[str, Any]:
        datos = {k: v for k, v in self.__dict__.items() if k not in ("fragmentos",)}
        datos["documentos_unicos"] = len({f.document_id for f in self.fragmentos})
        return datos


# ---------------------------------------------------------------------------
# Fragmentado
# ---------------------------------------------------------------------------
def fragmentar(
    contenido: str,
    tamano: int,
    solapamiento: int,
    secciones: Sequence[str] | None = None,
) -> list[tuple[str, str | None]]:
    """Parte un documento en fragmentos, respetando sus secciones.

    Intenta no partir por la mitad una frase. Un corte en medio de una
    oracion produce fragmentos que no significan nada por si solos y que
    el modelo puede completar por su cuenta.

    Devuelve pares `(texto, seccion)`.
    """
    pares = (
        _dividir_por_titulos(contenido, secciones)
        if secciones
        else _dividir_por_secciones(contenido)
    )

    fragmentos: list[tuple[str, str | None]] = []
    for titulo, cuerpo in pares:
        for trozo in _cortar(cuerpo, tamano, solapamiento):
            limpio = trozo.strip()
            if len(limpio) < 40:
                # Un fragmento de una linea no aporta senal en el indice y
                # solo suma ruido a la recuperacion.
                continue
            fragmentos.append((limpio, titulo))

    if not fragmentos and contenido.strip() and len(contenido.strip()) >= 40:
        # Documento sin secciones ni cortes posibles: se entrega entero.
        # El minimo de 40 caracteres tambien aplica aqui: un documento de
        # dos lineas no aporta nada al indice y solo lo llenaria de ruido.
        fragmentos.append((contenido.strip(), None))

    return fragmentos


def _dividir_por_titulos(
    contenido: str,
    secciones: Sequence[str],
) -> list[tuple[str | None, str]]:
    """Parte el texto usando los encabezados que la ingesta ya detecto.

    La Fase 2 los extrajo del HTML (h1, h2, h3) y los guardo. Usarlos es
    mas fiable que deducirlos del texto: una heuristica sobre lineas cortas
    confundiria un encabezado con un dato.

    Un titulo que no aparece en el texto se ignora. Es preferible perder
    una division a inventar una posicion equivocada.
    """
    lineas = contenido.split("\n")
    conocidas = set(secciones)

    posiciones = [
        (indice, linea.strip()) for indice, linea in enumerate(lineas) if linea.strip() in conocidas
    ]

    if not posiciones:
        return _dividir_por_secciones(contenido)

    resultado: list[tuple[str | None, str]] = []

    # Lo anterior al primer encabezado es el preambulo.
    if posiciones[0][0] > 0:
        preambulo = "\n".join(lineas[: posiciones[0][0]]).strip()
        if preambulo:
            resultado.append((posiciones[0][1], preambulo))

    for indice, (linea_inicio, titulo) in enumerate(posiciones):
        fin = posiciones[indice + 1][0] if indice + 1 < len(posiciones) else len(lineas)
        cuerpo = "\n".join(lineas[linea_inicio + 1 : fin]).strip()
        if cuerpo:
            resultado.append((titulo, cuerpo))

    return resultado


def _dividir_por_secciones(contenido: str) -> list[tuple[str | None, str]]:
    """Separa el documento por sus encabezados.

    El corpus viene de paginas con `h1`, `h2` y `h3`, y la extraccion los
    conserva como lineas propias. Se usan como cortes naturales: partir
    entre secciones produce fragmentos mucho mas utiles que partir por
    cantidad de caracteres.
    """
    if _RE_ENCABEZADO not in contenido:
        return [(None, contenido)]

    lineas = contenido.split("\n")
    secciones: list[tuple[str | None, list[str]]] = []
    titulo_actual: str | None = None
    cuerpo: list[str] = []

    for linea in lineas:
        if linea.startswith(_RE_ENCABEZADO):
            if cuerpo:
                secciones.append((titulo_actual, cuerpo))
            titulo_actual = linea.removeprefix(_RE_ENCABEZADO).strip()
            cuerpo = []
        else:
            cuerpo.append(linea)

    if cuerpo:
        secciones.append((titulo_actual, cuerpo))

    return [(t, "\n".join(c)) for t, c in secciones if "\n".join(c).strip()]


def _cortar(texto: str, tamano: int, solapamiento: int) -> list[str]:
    """Corta un texto en trozos de `tamano` caracteres."""
    if len(texto) <= tamano:
        return [texto]

    trozos: list[str] = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        if fin < len(texto):
            # Se busca un corte en un salto de linea o un punto, para no
            # partir palabras ni frases.
            ventana = texto.rfind("\n", inicio + tamano // 2, fin)
            if ventana == -1:
                ventana = texto.rfind(". ", inicio + tamano // 2, fin)
                if ventana != -1:
                    ventana += 1
            if ventana > inicio:
                fin = ventana
        trozos.append(texto[inicio:fin])
        # El solapamiento avanza en pasos de (tamano - solapamiento).
        paso = max(1, tamano - solapamiento)
        inicio = inicio + paso
    return [t for t in trozos if t.strip()]


# ---------------------------------------------------------------------------
# Servicio
# ---------------------------------------------------------------------------
class RAGService:
    """Indexa el corpus y recupera fragmentos por consulta."""

    def __init__(
        self,
        repositorio: DocumentoRepository | None = None,
        ajustes: Settings | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.ajustes = ajustes or get_settings()
        self._repositorio = repositorio
        self._embeddings = embeddings
        self._coleccion: Any = None
        self._hash_built: str | None = None

    @property
    def repositorio(self) -> DocumentoRepository:
        if self._repositorio is None:
            self._repositorio = DocumentoRepository(self.ajustes.processed_dir)
        return self._repositorio

    @property
    def embeddings(self) -> Embeddings:
        """Modelo de embeddings, cargado una sola vez por proceso.

        Cargar el modelo cuesta ~67 s en frio. Es el mayor costo unico de
        RAG y no existe en CAG; por eso se mide aparte.
        """
        if self._embeddings is None:
            self._embeddings = Embeddings(self.ajustes.embedding_model)
        return self._embeddings

    @property
    def ruta_indice(self) -> Path:
        return self.ajustes.resolve_path(self.ajustes.chroma_dir)

    @property
    def ruta_fragmentos(self) -> Path:
        return self.ajustes.processed_dir / "fragmentos.json"

    # -----------------------------------------------------------------
    # Huella del corpus
    # -----------------------------------------------------------------
    def hash_corpus(self) -> str:
        """Huella del corpus, comparable con la de CAG.

        Usa los mismos hashes de contenido que CAG. Si coinciden, ambos
        servicios estan trabajando sobre exactamente la misma fuente, que
        es lo que hace valida la comparacion.
        """
        documentos = sorted(self.repositorio.cargar_documentos(), key=lambda d: d.document_id)
        if not documentos:
            return "corpus-vacio"

        partes = []
        for documento in documentos:
            if documento.estado_verificacion == EstadoFuente.NO_ACCESIBLE:
                continue
            partes.append(f"{documento.document_id}:{documento.hash_contenido}")
        return hashlib.sha256("|".join(partes).encode("utf-8")).hexdigest()[:16]

    # -----------------------------------------------------------------
    # Construccion del indice
    # -----------------------------------------------------------------
    def indexar(self, force: bool = False) -> dict[str, Any]:
        """Construye el indice vectorial si el corpus cambio.

        Devuelve metricas de la construccion, que son distintas de las de
        CAG: aqui el costo es generar embeddings, no solo concatenar texto.
        """
        inicio = time.perf_counter()
        huella = self.hash_corpus()

        if not force and self._hash_built == huella and self._coleccion is not None:
            return {"reconstruido": False, "motivo": "sin cambios", "hash": huella}

        documentos = self.documentos_a_fragmentar()

        if not documentos:
            logger.warning("No hay documentos para indexar")
            self._hash_built = huella
            return {"reconstruido": False, "motivo": "corpus vacio", "hash": huella}

        textos = [f.contenido for f in documentos]
        embeddings = self.embeddings.codificar(textos)

        self._abrir_coleccion(recrear=True)
        self._coleccion.add(
            ids=[f.fragmento_id for f in documentos],
            embeddings=embeddings.tolist(),
            documents=textos,
            metadatas=[f.a_metadatos() for f in documentos],
        )

        self._hash_built = huella
        self._guardar_fragmentos(documentos)

        duracion = (time.perf_counter() - inicio) * 1000
        logger.info(
            "Indice RAG construido: %d fragmentos en %.1f ms",
            len(documentos),
            duracion,
        )
        return {
            "reconstruido": True,
            "hash": huella,
            "fragmentos": len(documentos),
            "duracion_ms": round(duracion, 1),
            "modelo_embeddings": self.embeddings.modelo_id,
        }

    def documentos_a_fragmentar(self) -> list[Fragmento]:
        """Parte el corpus en fragmentos, con identificador estable."""
        entradas = self.repositorio.cargar_entradas()
        fragmentos: list[Fragmento] = []

        for entrada in entradas:
            meta = entrada["metadatos"]
            if meta["estado_verificacion"] == EstadoFuente.NO_ACCESIBLE.value:
                continue

            contenido = normalizar_texto(entrada["contenido"])
            piezas = fragmentar(
                contenido,
                tamano=self.ajustes.rag_chunk_size,
                solapamiento=self.ajustes.rag_chunk_overlap,
                # Los encabezados los detecto la ingesta al procesar el HTML.
                # Preferirlos a adivinarlos desde el texto.
                secciones=entrada.get("secciones") or [],
            )

            for indice, (texto, seccion) in enumerate(piezas):
                # El id depende del documento y de la posicion, no del texto:
                # si el documento cambia, el indice se reconstruye entero y
                # los ids siguen siendo trazables.
                fragmentos.append(
                    Fragmento(
                        fragmento_id=f"{meta['document_id']}#{indice}",
                        document_id=meta["document_id"],
                        contenido=texto,
                        indice=indice,
                        tramite=meta["tramite"],
                        organismo=meta["organismo"],
                        titulo=meta["titulo"],
                        seccion=seccion,
                        url_origen=meta.get("url_origen"),
                    )
                )

        return fragmentos

    # -----------------------------------------------------------------
    # Recuperacion
    # -----------------------------------------------------------------
    def recuperar(self, consulta: str, top_k: int | None = None) -> ResultadoBusqueda:
        """Recupera los fragmentos mas relevantes para una consulta."""
        consulta = consulta.strip()
        if not consulta:
            raise ValueError("La consulta no puede estar vacia")

        cantidad = top_k or self.ajustes.rag_top_k
        inicio_total = time.perf_counter()

        info_indice = self.indexar()
        if self._coleccion is None:
            return ResultadoBusqueda(
                consulta=consulta,
                solicitados=cantidad,
                hash_corpus=self.hash_corpus(),
            )

        inicio_emb = time.perf_counter()
        vector = self.embeddings.codificar_consulta(consulta)
        embedding_ms = (time.perf_counter() - inicio_emb) * 1000

        inicio_bus = time.perf_counter()
        bruto = self._coleccion.query(
            query_embeddings=[vector.tolist()],
            n_results=cantidad,
            include=["documents", "metadatas", "distances"],
        )
        busqueda_ms = (time.perf_counter() - inicio_bus) * 1000

        fragmentos = self._reconstruir_fragmentos(bruto)
        total_ms = (time.perf_counter() - inicio_total) * 1000

        resultado = ResultadoBusqueda(
            consulta=consulta,
            fragmentos=fragmentos,
            recuperacion_ms=round(total_ms, 1),
            embedding_ms=round(embedding_ms, 1),
            busqueda_ms=round(busqueda_ms, 1),
            solicitados=cantidad,
            obtenidos=len(fragmentos),
            indice_reconstruido=bool(info_indice.get("reconstruido")),
            hash_corpus=self.hash_corpus(),
        )

        logger.info(
            "recuperacion | fragmentos=%d/%d | %.1f ms (emb %.1f + busq %.1f)",
            len(fragmentos),
            cantidad,
            total_ms,
            embedding_ms,
            busqueda_ms,
        )
        return resultado

    def _reconstruir_fragmentos(self, bruto: dict[str, Any]) -> list[FragmentoRecuperado]:
        """Vuelve a armar los fragmentos desde la respuesta del indice."""
        if not bruto or not bruto.get("ids"):
            return []

        ids = bruto["ids"][0]
        documentos = (bruto.get("documents") or [[]])[0]
        metadatos = (bruto.get("metadatas") or [[]])[0]
        distancias = (bruto.get("distances") or [[]])[0]

        recuperados: list[FragmentoRecuperado] = []
        for indice, fragmento_id in enumerate(ids):
            meta = metadatos[indice] if indice < len(metadatos) else {}
            distancia = distancias[indice] if indice < len(distancias) else None
            # Similaridad = 1 - distancia coseno, en escala 0 a 1.
            puntuacion = 1.0 - float(distancia) if distancia is not None else 0.0

            recuperados.append(
                FragmentoRecuperado(
                    fragmento=Fragmento(
                        fragmento_id=fragmento_id,
                        document_id=meta.get("document_id", ""),
                        contenido=documentos[indice] if indice < len(documentos) else "",
                        indice=int(meta.get("indice", 0) or 0),
                        tramite=meta.get("tramite", ""),
                        organismo=meta.get("organismo", ""),
                        titulo=meta.get("titulo", ""),
                        seccion=meta.get("seccion") or None,
                        url_origen=meta.get("url_origen") or None,
                    ),
                    puntuacion=round(puntuacion, 4),
                    distancia=round(float(distancia), 4) if distancia is not None else None,
                )
            )
        return recuperados

    # -----------------------------------------------------------------
    # Persistencia
    # -----------------------------------------------------------------
    def _abrir_coleccion(self, recrear: bool = False) -> None:
        """Abre la coleccion vectorial, creandola si hace falta."""
        if self._coleccion is not None and not recrear:
            return

        import chromadb

        self.ruta_indice.mkdir(parents=True, exist_ok=True)
        cliente = chromadb.PersistentClient(path=str(self.ruta_indice))
        nombre = "tramites"

        if recrear:
            # La coleccion puede no existir todavia: se ignora a proposito.
            with contextlib.suppress(Exception):
                cliente.delete_collection(nombre)

        self._coleccion = cliente.get_or_create_collection(
            nombre,
            metadata={"hnsw:space": "cosine", "descripcion": VERSION_INDICE},
        )

    def _guardar_fragmentos(self, fragmentos: Sequence[Fragmento]) -> None:
        """Guarda el texto de los fragmentos aparte del indice.

        Chroma guarda documentos, pero se recuperan por consulta y no en
        bloque. Tener el JSON permite auditar que se indexeo sin depender
        del proveedor vectorial.
        """
        self.ruta_fragmentos.write_text(
            json.dumps(
                {
                    "version": VERSION_INDICE,
                    "hash_corpus": self._hash_built,
                    "creado": datetime.now(UTC).isoformat(),
                    "modelo_embeddings": self.ajustes.embedding_model,
                    "fragmentos": [
                        {
                            "fragmento_id": f.fragmento_id,
                            "document_id": f.document_id,
                            "contenido": f.contenido,
                            "indice": f.indice,
                            "tramite": f.tramite,
                            "seccion": f.seccion,
                        }
                        for f in fragmentos
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def invalidar(self) -> None:
        """Cierra el indice para que se reconstruya en la proxima consulta."""
        self._coleccion = None
        self._hash_built = None

    # -----------------------------------------------------------------
    # Diagnostico
    # -----------------------------------------------------------------
    def estado(self) -> dict[str, Any]:
        return {
            "hash_corpus": self.hash_corpus(),
            "indice_hash": self._hash_built,
            "indice_al_day": self._hash_built == self.hash_corpus(),
            "ruta_indice": str(self.ruta_indice),
            "top_k": self.ajustes.rag_top_k,
            "chunk_size": self.ajustes.rag_chunk_size,
            "chunk_overlap": self.ajustes.rag_chunk_overlap,
            "modelo_embeddings": self.ajustes.embedding_model,
            "embeddings_cargados": self._embeddings is not None,
        }


class Embeddings:
    """Envoltura del modelo de embeddings.

    Se separa del servicio para poder sustituirlo en las pruebas sin
    cargar un modelo de 400 MB.
    """

    def __init__(self, modelo_id: str) -> None:
        self.modelo_id = modelo_id
        self._modelo: Any = None
        self.tiempo_carga_ms: float = 0.0

    @property
    def modelo(self) -> Any:
        if self._modelo is None:
            inicio = time.perf_counter()
            from sentence_transformers import SentenceTransformer

            self._modelo = SentenceTransformer(self.modelo_id)
            self.tiempo_carga_ms = (time.perf_counter() - inicio) * 1000
            logger.info(
                "Modelo de embeddings cargado en %.0f ms: %s",
                self.tiempo_carga_ms,
                self.modelo_id,
            )
        return self._modelo

    @property
    def dimensiones(self) -> int:
        return int(self.modelo.get_sentence_embedding_dimension())

    def codificar(self, textos: Sequence[str]) -> Any:
        """Vectoriza una lista de textos."""
        import numpy as np

        vectores = self.modelo.encode(
            list(textos),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectores, dtype="float32")

    def codificar_consulta(self, consulta: str) -> Any:
        return self.codificar([consulta])[0]

    def similitud(self, a: Sequence[str], b: Sequence[str]) -> list[float]:
        """Similitud coseno entre dos conjuntos de textos."""
        va = self.codificar(a)
        vb = self.codificar(b)
        return (va @ vb.T).diagonal().tolist()


class EmbeddingsFalsos(Embeddings):
    """Embeddings deterministas para pruebas, sin cargar ningun modelo.

    Usa coincidencia de terminos, que alcanza para comprobar que el
    pipeline recupera lo que debe. No es un modelo semantico y no se usa
    fuera de las pruebas: usarlo invalidaria cualquier medicion.
    """

    _DIMENSIONES = 64

    def __init__(self, modelo_id: str = "falsos") -> None:
        super().__init__(modelo_id)

    @property
    def dimensiones(self) -> int:
        return self._DIMENSIONES

    def codificar(self, textos: Sequence[str]) -> Any:
        import numpy as np

        vectores = []
        for texto in textos:
            vector = np.zeros(self._DIMENSIONES, dtype="float32")
            for palabra in set(texto.lower().split()):
                vector[hash(palabra) % self._DIMENSIONES] += 1.0
            norma = float((vector**2).sum() ** 0.5) or 1.0
            vectores.append(vector / norma)
        return np.asarray(vectores, dtype="float32")


_SERVICIO: RAGService | None = None


def obtener_servicio_rag() -> RAGService:
    global _SERVICIO
    if _SERVICIO is None:
        _SERVICIO = RAGService()
    return _SERVICIO


def reiniciar_servicio_rag() -> None:
    global _SERVICIO
    _SERVICIO = None
