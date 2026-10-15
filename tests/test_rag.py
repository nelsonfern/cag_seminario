"""Pruebas del servicio RAG.

Se usa `EmbeddingsFalsos` en lugar del modelo real: la suite tiene que
ser rapida y no cargar 400 MB de pesos. Lo que se verifica aqui es el
comportamiento del pipeline (fragmentado, indice, invalidacion,
metricas), no la calidad semantica de las embeddings, que se mide aparte
contra el modelo real.

La calidad de recuperacion con el modelo de verdad se comprueba en
`scripts/probar_rag.py`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from app.core.config import Settings
from app.models.schemas import EstadoFuente, EstadoVigencia, TipoDocumento
from app.repositories.document_repository import DocumentoRepository
from app.services.cag import CAGService
from app.services.rag import (
    EmbeddingsFalsos,
    RAGService,
    fragmentar,
)

pytestmark = pytest.mark.rag

CONTENIDO_PARTIDAS = """\
Solicitud de Partidas
Ultima actualizacion: 20/10/2025
Este tramite permite solicitar copias de partidas de nacimiento y matrimonio.
Iniciar tramite digital

Requisitos del Tramite
Tener una cuenta de Ciudadano Digital.
Datos del acta necesarios para que la busqueda sea exitosa en el sistema.

Costo
Segun la Ley Impositiva 2024 (Ley 10.929, art. 79):
Copia de acta certificada: $1.100
Copia de acta legalizada: $1.500

Informacion adicional
Indicar localidad de inscripcion, numero de acta, tomo, serie y folio.
"""

SECCIONES = [
    "Solicitud de Partidas",
    "Requisitos del Tramite",
    "Costo",
    "Informacion adicional",
]

CONTENIDO_MATRIMONIO = """\
Inscripcion de matrimonios

Para inscribir un matrimonio se necesitan las partidas de nacimiento de
ambos contrayentes y el documento nacional de identidad de cada uno.
"""


def _documento(document_id: str, tramite: str, contenido: str, secciones=None):
    from app.models.schemas import MetadatosDocumento

    return MetadatosDocumento(
        document_id=document_id,
        titulo=f"Titulo {document_id}",
        organismo="Registro Civil de La Rioja",
        tramite=tramite,
        tipo_documento=TipoDocumento.HTML,
        url_origen=f"https://registrocivil.larioja.gob.ar/{document_id}.html",
        archivo_original=f"{document_id}.html",
        fecha_consulta=date(2026, 10, 15),
        vigencia=EstadoVigencia.VIGENTE,
        hash_contenido=hashlib.sha256(contenido.encode("utf-8")).hexdigest(),
    )


@pytest.fixture
def repo(tmp_path: Path) -> DocumentoRepository:
    repositorio = DocumentoRepository(tmp_path / "processed")
    repositorio.guardar_documento(
        _documento("doc-partidas", "partidas", CONTENIDO_PARTIDAS, SECCIONES),
        CONTENIDO_PARTIDAS,
        SECCIONES,
    )
    repositorio.guardar_documento(
        _documento("doc-matrimonio", "matrimonios", CONTENIDO_MATRIMONIO, []),
        CONTENIDO_MATRIMONIO,
        [],
    )
    return repositorio


@pytest.fixture
def ajustes(tmp_path: Path, repo: DocumentoRepository) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="fake",
        processed_path=repo.directorio,
        sqlite_path=tmp_path / "t.db",
        chroma_dir=tmp_path / "chroma",
        rag_chunk_size=900,
        rag_chunk_overlap=150,
        rag_top_k=3,
    )


@pytest.fixture
def servicio(ajustes: Settings, repo: DocumentoRepository) -> RAGService:
    return RAGService(repositorio=repo, ajustes=ajustes, embeddings=EmbeddingsFalsos())


# ---------------------------------------------------------------------------
# Fragmentado
# ---------------------------------------------------------------------------
class TestFragmentado:
    def test_usa_las_secciones_de_la_ingesta(self) -> None:
        """Las secciones las detecto la Fase 2 al procesar el HTML.

        Adivinarlas desde el texto confundiria un encabezado con un dato.
        """
        piezas = fragmentar(CONTENIDO_PARTIDAS, 900, 150, SECCIONES)

        titulos = {t for _, t in piezas}

        assert "Costo" in titulos
        assert "Requisitos del Tramite" in titulos

    def test_separa_por_seccion(self) -> None:
        piezas = fragmentar(CONTENIDO_PARTIDAS, 400, 50, SECCIONES)

        for _, titulo in piezas:
            assert titulo in SECCIONES

    def test_respeta_el_tamano(self) -> None:
        piezas = fragmentar(CONTENIDO_PARTIDAS, 200, 50, SECCIONES)

        assert all(len(texto) <= 400 for texto, _ in piezas)

    def test_descarta_fragmentos_muy_cortos(self) -> None:
        """Una linea suelta no aporta senal y solo suma ruido."""
        piezas = fragmentar("Hola.\n", 900, 150, [])

        assert piezas == []

    def test_documento_sin_secciones(self) -> None:
        piezas = fragmentar(CONTENIDO_MATRIMONIO, 900, 150, [])

        assert piezas
        assert all(len(texto) >= 40 for texto, _ in piezas)

    def test_titulo_inexistente_no_rompe(self) -> None:
        """Un encabezado que no aparece se ignora, no se inventa su posicion."""
        piezas = fragmentar(CONTENIDO_PARTIDAS, 900, 150, ["Titulo que no existe en el texto"])

        assert piezas

    def test_preambulo_atribuido(self) -> None:
        piezas = fragmentar(CONTENIDO_PARTIDAS, 900, 150, SECCIONES)

        assert any(t == "Solicitud de Partidas" for _, t in piezas)

    def test_los_trozos_avanzan(self) -> None:
        """Los trozos deben avanzar, no quedarse siempre en el mismo punto."""
        texto = "\n".join(f"Linea numero {i} con contenido suficiente." for i in range(60))
        piezas = fragmentar(texto, 300, 0, [])

        assert len(piezas) >= 2
        assert piezas[1][0][:20] != piezas[0][0][:20]

    def test_solapamiento_comparte_contenido(self) -> None:
        """Con solapamiento, dos trozos consecutivos comparten texto.

        El solapamiento existe para que un dato que cae en el borde entre
        dos trozos no se pierda. Por eso hay MAS trozos, no menos: cada uno
        cubre `tamano` pero avanza `tamano - solapamiento`.
        """
        texto = "\n".join(f"Linea numero {i} con contenido suficiente." for i in range(60))
        con = fragmentar(texto, 300, 100, [])
        sin = fragmentar(texto, 300, 0, [])

        assert len(con) > len(sin)
        assert con[0][0][-50:] in con[1][0]


# ---------------------------------------------------------------------------
# Fragmentos del corpus
# ---------------------------------------------------------------------------
class TestFragmentosDelCorpus:
    def test_genera_fragmentos(self, servicio: RAGService) -> None:
        fragmentos = servicio.documentos_a_fragmentar()

        assert fragmentos
        assert all(f.document_id for f in fragmentos)

    def test_identificador_estable(self, servicio: RAGService) -> None:
        """El id depende del documento y la posicion, no del texto."""
        primero = servicio.documentos_a_fragmentar()
        segundo = servicio.documentos_a_fragmentar()

        assert [f.fragmento_id for f in primero] == [f.fragmento_id for f in segundo]

    def test_conserva_seccion_y_url(self, servicio: RAGService) -> None:
        fragmentos = servicio.documentos_a_fragmentar()
        con_seccion = [f for f in fragmentos if f.seccion]

        assert con_seccion
        assert all(f.url_origen for f in fragmentos)

    def test_documento_bloqueado_no_se_indexa(self, tmp_path: Path) -> None:
        """Una fuente no accesible no debe llegar al indice."""
        repo = DocumentoRepository(tmp_path / "solo_bloqueado")
        meta = _documento("doc-bloq", "partidas", CONTENIDO_PARTIDAS, SECCIONES)
        meta.estado_verificacion = EstadoFuente.NO_ACCESIBLE
        repo.guardar_documento(meta, CONTENIDO_PARTIDAS, SECCIONES)

        config = Settings(
            _env_file=None,
            llm_provider="fake",
            processed_path=repo.directorio,
            sqlite_path=tmp_path / "t.db",
            chroma_dir=tmp_path / "chroma",
        )
        servicio = RAGService(repositorio=repo, ajustes=config, embeddings=EmbeddingsFalsos())

        assert servicio.documentos_a_fragmentar() == []


# ---------------------------------------------------------------------------
# Indice
# ---------------------------------------------------------------------------
class TestIndice:
    def test_construye_el_indice(self, servicio: RAGService) -> None:
        info = servicio.indexar()

        assert info["reconstruido"] is True
        assert info["fragmentos"] > 0

    def test_no_reconstruye_sin_cambios(self, servicio: RAGService) -> None:
        servicio.indexar()

        segunda = servicio.indexar()

        assert segunda["reconstruido"] is False
        assert segunda["motivo"] == "sin cambios"

    def test_reconstruye_si_cambio_el_corpus(self, servicio: RAGService) -> None:
        servicio.indexar()

        servicio.repositorio.guardar_documento(
            _documento(
                "doc-matrimonio",
                "matrimonios",
                "CONTENIDO COMPLETAMENTE NUEVO " * 15,
                [],
            ),
            "CONTENIDO COMPLETAMENTE NUEVO " * 15,
            [],
        )

        assert servicio.indexar()["reconstruido"] is True

    def test_hash_cambia_con_el_corpus(self, servicio: RAGService) -> None:
        antes = servicio.hash_corpus()

        servicio.repositorio.guardar_documento(
            _documento("doc-nuevo", "nuevos", "Un tramite nuevo. " * 15),
            "Un tramite nuevo. " * 15,
            [],
        )

        assert servicio.hash_corpus() != antes

    def test_indice_al_dia(self, servicio: RAGService) -> None:
        servicio.indexar()

        estado = servicio.estado()

        assert estado["indice_al_day"] is True

    def test_guarda_los_fragmentos_para_auditar(self, servicio: RAGService) -> None:
        """Chroma no devuelve todos los fragmentos de una vez."""
        servicio.indexar()

        datos = json.loads(servicio.ruta_fragmentos.read_text(encoding="utf-8"))

        assert datos["fragmentos"]
        assert datos["modelo_embeddings"]
        assert all("contenido" in f for f in datos["fragmentos"])


# ---------------------------------------------------------------------------
# Recuperacion
# ---------------------------------------------------------------------------
class TestRecuperacion:
    def test_devuelve_fragmentos(self, servicio: RAGService) -> None:
        resultado = servicio.recuperar("cuanto cuesta la copia certificada")

        assert resultado.obtenidos > 0
        assert all(f.document_id for f in resultado.fragmentos)

    def test_respeta_top_k(self, servicio: RAGService) -> None:
        resultado = servicio.recuperar("costo de la copia", top_k=2)

        assert resultado.obtenidos <= 2
        assert resultado.solicitados == 2

    def test_ordena_por_puntuacion(self, servicio: RAGService) -> None:
        """Recuperar algo poco relevante al principio no sirve de nada."""
        resultado = servicio.recuperar("cuanto cuesta la copia certificada")

        puntuaciones = [f.puntuacion for f in resultado.fragmentos]

        assert puntuaciones == sorted(puntuaciones, reverse=True)

    def test_relevancia_por_parecido_terminal(self, servicio: RAGService) -> None:
        """La pregunta sobre costos debe traer el fragmento de costos."""
        resultado = servicio.recuperar("cuanto cuesta la copia certificada")

        assert any("certificada" in f.fragmento.contenido for f in resultado.fragmentos)

    def test_registra_metricas(self, servicio: RAGService) -> None:
        """La recuperacion es un costo propio de RAG y hay que medirlo."""
        resultado = servicio.recuperar("costo de la copia")

        assert resultado.recuperacion_ms > 0
        assert resultado.embedding_ms > 0
        assert resultado.busqueda_ms >= 0
        assert resultado.solicitados > 0

    def test_consulta_vacia(self, servicio: RAGService) -> None:
        with pytest.raises(ValueError, match="no puede estar vacia"):
            servicio.recuperar("   ")

    def test_corpus_vacio_no_rompe(self, tmp_path: Path) -> None:
        vacio = DocumentoRepository(tmp_path / "vacio")
        config = Settings(
            _env_file=None,
            llm_provider="fake",
            processed_path=vacio.directorio,
            chroma_dir=tmp_path / "c",
            sqlite_path=tmp_path / "t.db",
        )
        servicio = RAGService(repositorio=vacio, ajustes=config, embeddings=EmbeddingsFalsos())

        resultado = servicio.recuperar("cualquier cosa")

        assert resultado.obtenidos == 0
        assert resultado.fragmentos == []

    def test_a_contexto_expone_solo_el_fragmento(self, servicio: RAGService) -> None:
        """El agente solo puede verificar contra lo que recibio.

        Si un dato esta en el documento pero no se recupero, no debe
        aparecer como respaldado.
        """
        resultado = servicio.recuperar("cuanto cuesta la copia certificada")
        contexto = resultado.fragmentos[0].a_contexto()

        assert contexto.fragmentos == [resultado.fragmentos[0].fragmento.contenido]
        assert contexto.metadatos.document_id == resultado.fragmentos[0].document_id

    def test_metadatos_traen_vigencia(self, servicio: RAGService) -> None:
        resultado = servicio.recuperar("costo de la copia")
        contexto = resultado.fragmentos[0].a_contexto()

        assert contexto.metadatos.vigencia is not None
        assert contexto.metadatos.url_origen


# ---------------------------------------------------------------------------
# Comparabilidad con CAG
# ---------------------------------------------------------------------------
class TestComparabilidadConCAG:
    def test_mismo_hash_de_corpus(self, servicio: RAGService, ajustes: Settings) -> None:
        """Es la garantia de que la comparacion usa el mismo material.

        Si los hashes difieren, CAG y RAG estaria respondiendo sobre
        corpus distintos y la comparacion no valdria nada.
        """
        hash_rag = servicio.hash_corpus()
        hash_cag = CAGService(repositorio=servicio.repositorio, ajustes=ajustes).hash_corpus()

        assert hash_rag == hash_cag

    def test_hash_no_depende_del_formato(self, tmp_path: Path, repo) -> None:
        """La version del formato va aparte: el hash describe el corpus."""
        a = Settings(
            _env_file=None,
            llm_provider="fake",
            processed_path=repo.directorio,
            sqlite_path=tmp_path / "a.db",
            chroma_dir=tmp_path / "a",
        )
        b = Settings(
            _env_file=None,
            llm_provider="fake",
            processed_path=repo.directorio,
            sqlite_path=tmp_path / "b.db",
            chroma_dir=tmp_path / "b",
            rag_chunk_size=400,
        )

        assert (
            RAGService(repositorio=repo, ajustes=a).hash_corpus()
            == RAGService(repositorio=repo, ajustes=b).hash_corpus()
        )


# ---------------------------------------------------------------------------
# Embeddings falsos
# ---------------------------------------------------------------------------
class TestEmbeddingsFalsos:
    def test_es_determinista(self) -> None:
        e = EmbeddingsFalsos()

        primero = e.codificar(["texto uno"])
        segundo = e.codificar(["texto uno"])

        assert primero.tolist() == segundo.tolist()

    def test_normaliza(self) -> None:

        e = EmbeddingsFalsos()
        vector = e.codificar(["un texto con palabras varias"])[0]

        assert pytest.approx(float((vector**2).sum()), abs=1e-5) == 1.0

    def test_textos_distintos_dan_vectores_distintos(self) -> None:
        e = EmbeddingsFalsos()

        a = e.codificar(["costo de la copia certificada"])
        b = e.codificar(["horario de atencion al publico"])[0]

        assert a[0].tolist() != b.tolist()

    def test_texto_mas_largo_mas_parecido(self) -> None:
        """Permite comprobar la relevancia sin cargar el modelo real."""
        e = EmbeddingsFalsos()
        consulta = "cuanto cuesta la copia de acta certificada"

        a = e.similitud([consulta], ["costo de la copia de acta certificada"])[0]
        b = e.similitud([consulta], ["horarios de atencion al publico"])[0]

        assert a > b
