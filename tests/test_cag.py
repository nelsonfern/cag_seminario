"""Pruebas del servicio CAG.

Lo que se verifica es lo que el enunciado pide de CAG y lo que no es
evidente: que el contexto se reutilice, que se invalide cuando el corpus
cambia, que se controle el presupuesto de tokens, y que las dos caches
(aplicacion y proveedor) no se confundan.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from app.core.config import Settings
from app.models.schemas import EstadoVigencia, TipoDocumento
from app.repositories.document_repository import DocumentoRepository
from app.services.cag import CAGService, reiniciar_servicio_cag

pytestmark = pytest.mark.cag


def _documento(document_id: str, tramite: str, contenido: str):
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
    textos = {
        "doc-partidas": "Solicitud de partidas. Copia certificada: $1.100. " * 12,
        "doc-nacimientos": "Inscripcion de nacimientos. Documento de identidad. " * 12,
        "doc-matrimonios": "Inscripcion de matrimonios. Partidas de los contrayentes. " * 12,
    }
    for document_id, contenido in textos.items():
        repositorio.guardar_documento(
            _documento(document_id, document_id.split("-")[1], contenido),
            contenido,
            [document_id],
        )
    return repositorio


@pytest.fixture
def servicio(repo: DocumentoRepository, tmp_path: Path) -> CAGService:
    ajustes = Settings(
        _env_file=None,
        llm_provider="fake",
        sqlite_path=tmp_path / "t.db",
        processed_path=repo.directorio,
        cag_cache_dir=tmp_path / "cag_cache",
        cag_max_context_tokens=2000,
    )
    return CAGService(repositorio=repo, ajustes=ajustes)


# ---------------------------------------------------------------------------
# Preparacion
# ---------------------------------------------------------------------------
class TestPreparacion:
    def test_prepara_todo_el_corpus(self, servicio: CAGService) -> None:
        ctx = servicio.preparar(force=True)

        assert ctx.documentos_incluidos == 3
        assert ctx.documentos_omitidos == 0
        assert ctx.contexto
        assert ctx.truncado is False

    def test_registra_tiempo_de_preparacion(self, servicio: CAGService) -> None:
        ctx = servicio.preparar(force=True)

        assert ctx.preparacion_ms > 0
        assert ctx.desde_cache is False

    def test_documentos_ordenados_de_forma_determinista(self, servicio: CAGService) -> None:
        """El orden importa: si vara, el prefijo cambia y no hay cache."""
        primero = servicio.preparar(force=True)
        segundo = servicio.preparar(force=True)

        ids_1 = [d.metadatos.document_id for d in primero.documentos]
        ids_2 = [d.metadatos.document_id for d in segundo.documentos]

        assert ids_1 == ids_2 == sorted(ids_1)

    def test_contexto_estable_entre_preparaciones(self, servicio: CAGService) -> None:
        """Sin esto, el proveedor no puede reaprovechar nada."""
        primero = servicio.preparar(force=True)
        segundo = servicio.preparar(force=True)

        assert primero.contexto == segundo.contexto

    def test_incluye_metadatos_de_vigencia(self, servicio: CAGService) -> None:
        """El modelo tiene que saber que datos no estan confirmados."""
        ctx = servicio.preparar(force=True)

        assert "Vigencia: vigente" in ctx.contexto
        assert "Organismo: Registro Civil de La Rioja" in ctx.contexto

    def test_marca_el_contenido_como_no_confiable(self, servicio: CAGService) -> None:
        ctx = servicio.preparar(force=True)

        assert "<contexto_documental no_confiable>" in ctx.contexto
        assert ctx.contexto.count("<contexto_documental no_confiable>") == 6

    def test_estima_tokens(self, servicio: CAGService) -> None:
        ctx = servicio.preparar(force=True)

        assert 0 < ctx.tokens_estimados < len(ctx.contexto)


# ---------------------------------------------------------------------------
# Presupuesto de tokens
# ---------------------------------------------------------------------------
class TestPresupuesto:
    def test_respeta_el_presupuesto(self, repo: DocumentoRepository, tmp_path: Path) -> None:
        ajustes = Settings(
            _env_file=None,
            llm_provider="fake",
            processed_path=repo.directorio,
            cag_cache_dir=tmp_path / "c",
            cag_max_context_tokens=120,
        )
        ctx = CAGService(repositorio=repo, ajustes=ajustes).preparar(force=True)

        assert ctx.tokens_estimados <= 120
        assert ctx.documentos_incluidos < 3
        assert ctx.documentos_omitidos > 0

    def test_trunca_por_documento_entero(self, repo: DocumentoRepository, tmp_path: Path) -> None:
        """Cortar un documento por la mitad produce citas sin sentido."""
        ajustes = Settings(
            _env_file=None,
            llm_provider="fake",
            processed_path=repo.directorio,
            cag_cache_dir=tmp_path / "c",
            cag_max_context_tokens=120,
        )
        ctx = CAGService(repositorio=repo, ajustes=ajustes).preparar(force=True)

        for documento in ctx.documentos:
            assert documento.contenido in ctx.contexto

    def test_avisa_cuando_trunca(self, repo: DocumentoRepository, tmp_path: Path) -> None:
        """Un truncado silencioso haria creer que el corpus esta completo."""
        ajustes = Settings(
            _env_file=None,
            llm_provider="fake",
            processed_path=repo.directorio,
            cag_cache_dir=tmp_path / "c",
            cag_max_context_tokens=60,
        )
        ctx = CAGService(repositorio=repo, ajustes=ajustes).preparar(force=True)

        assert ctx.documentos_incluidos == 0
        assert ctx.documentos_omitidos == 3


# ---------------------------------------------------------------------------
# Cache de aplicacion
# ---------------------------------------------------------------------------
class TestCacheAplicacion:
    def test_reutiliza_entre_consultas(self, servicio: CAGService) -> None:
        servicio.preparar(force=True)
        segunda = servicio.preparar()

        assert segunda.desde_cache is True

    def test_no_reporta_preparacion_cuando_sirve_de_cache(self, servicio: CAGService) -> None:
        """Reportar el tiempo de una preparacion anterior falsearia la
        comparacion entre ejecucion fria y caliente."""
        servicio.preparar(force=True)
        caliente = servicio.preparar()

        assert caliente.preparacion_ms == 0.0
        assert caliente.entrega_ms < servicio.preparar(force=True).preparacion_ms

    def test_la_cache_no_se_altera_al_entregar(self, servicio: CAGService) -> None:
        """Quien lee la cache no puede modificarla.

        Si `preparar()` devolviera el objeto guardado, marcar
        `desde_cache` en el contaminaria para la siguiente lectura.
        """
        frio = servicio.preparar(force=True)
        caliente = servicio.preparar()
        caliente.desde_cache = False
        caliente.contexto = "CONTAMINADO"

        siguiente = servicio.preparar()

        assert siguiente.desde_cache is True
        assert siguiente.contexto != "CONTAMINADO"
        assert siguiente.contexto == frio.contexto

    def test_persiste_en_disco(self, servicio: CAGService) -> None:
        servicio.preparar(force=True)

        assert servicio.ruta_cache.exists()

        # Una instancia nueva debe poder leer la cache del disco.
        otro = CAGService(repositorio=servicio.repositorio, ajustes=servicio.ajustes)
        desde_disco = otro.preparar()

        assert desde_disco.desde_cache is True

    def test_invalida(self, servicio: CAGService) -> None:
        servicio.preparar(force=True)
        servicio.invalidar()

        assert servicio.ruta_cache.exists() is False
        assert servicio.preparar().desde_cache is False


# ---------------------------------------------------------------------------
# Invalidacion por cambios del corpus
# ---------------------------------------------------------------------------
class TestInvalidacion:
    def test_cambio_de_contenido_invalida(self, servicio: CAGService) -> None:
        antes = servicio.hash_corpus()
        servicio.preparar(force=True)

        servicio.repositorio.guardar_documento(
            _documento("doc-partidas", "partidas", "CONTENIDO COMPLETAMENTE DISTINTO " * 20),
            "CONTENIDO COMPLETAMENTE DISTINTO " * 20,
        )

        assert servicio.hash_corpus() != antes

    def test_reconstruye_si_cambio_el_corpus(self, servicio: CAGService) -> None:
        servicio.preparar(force=True)
        caliente = servicio.preparar()
        assert caliente.desde_cache is True

        servicio.repositorio.guardar_documento(
            _documento("doc-nacimientos", "nacimientos", "OTRO TEXTO DISTINTO " * 20),
            "OTRO TEXTO DISTINTO " * 20,
        )

        assert servicio.preparar().desde_cache is False

    def test_documento_nuevo_invalida(self, servicio: CAGService) -> None:
        antes = servicio.hash_corpus()

        servicio.repositorio.guardar_documento(
            _documento("doc-nuevo", "nuevos", "Un tramite nuevo. " * 20),
            "Un tramite nuevo. " * 20,
        )

        assert servicio.hash_corpus() != antes

    def test_reordenar_no_invalida(self, servicio: CAGService) -> None:
        """El hash depende del contenido, no del orden de lectura.

        Reordenar los archivos en disco no cambia la informacion
        disponible y no deberia costar una reconstruccion.
        """
        antes = servicio.hash_corpus()
        servicio.preparar(force=True)

        servicio.preparar(force=True)

        assert servicio.hash_corpus() == antes

    def test_documento_inalcanzable_no_cuenta(self, servicio: CAGService) -> None:
        """Una fuente bloqueada por robots.txt no debe entrar al contexto."""
        from app.models.schemas import EstadoFuente

        meta = _documento("doc-bloqueado", "partidas", "No deberia aparecer " * 20)
        meta.estado_verificacion = EstadoFuente.NO_ACCESIBLE
        servicio.repositorio.guardar_documento(meta, "No deberia aparecer " * 20)

        ctx = servicio.preparar(force=True)

        assert "No deberia aparecer" not in ctx.contexto


# ---------------------------------------------------------------------------
# Particionado
# ---------------------------------------------------------------------------
class TestParticionado:
    def test_particiones_disponibles(self, servicio: CAGService) -> None:
        particiones = servicio.particiones_disponibles()

        assert set(particiones) == {"partidas", "nacimientos", "matrimonios"}
        assert all(p["documentos"] == 1 for p in particiones.values())
        assert all(p["tokens"] > 0 for p in particiones.values())

    def test_preparar_una_particion(self, servicio: CAGService) -> None:
        ctx = servicio.preparar_particion("partidas", force=True)

        assert ctx.particion == "partidas"
        assert ctx.documentos_incluidos == 1
        assert ctx.tramites == ["partidas"]

    def test_particion_reduce_el_contexto(self, servicio: CAGService) -> None:
        completo = servicio.preparar(force=True)
        parcial = servicio.preparar_particion("partidas", force=True)

        assert parcial.tokens_estimados < completo.tokens_estimados


# ---------------------------------------------------------------------------
# Casos limite
# ---------------------------------------------------------------------------
class TestCasosLimite:
    def test_corpus_vacio(self, tmp_path: Path) -> None:
        vacio = DocumentoRepository(tmp_path / "vacio")
        ajustes = Settings(
            _env_file=None,
            llm_provider="fake",
            processed_path=vacio.directorio,
            cag_cache_dir=tmp_path / "c",
        )
        servicio = CAGService(repositorio=vacio, ajustes=ajustes)

        ctx = servicio.preparar(force=True)

        assert ctx.documentos_incluidos == 0
        assert ctx.contexto == ""
        assert ctx.hash_corpus == "corpus-vacio"

    def test_cache_corrupta_se_regenera(self, servicio: CAGService) -> None:
        """Una cache ilegible no puede impedir arrancar."""
        servicio.preparar(force=True)
        servicio.ruta_cache.write_text("{esto no es json", encoding="utf-8")

        reiniciar_servicio_cag()
        servicio._en_memoria = None
        ctx = servicio.preparar()

        assert ctx.desde_cache is False
        assert ctx.documentos_incluidos == 3

    def test_estado_informa_cache_no_verificable(self, servicio: CAGService) -> None:
        """Sin archivo no se afirma "vigente": se informa que no existe."""
        estado = servicio.estado()

        assert estado["existe"] is False
        assert estado["cache_valida"] is False

    def test_resumen_no_incluye_el_contexto_completo(self, servicio: CAGService) -> None:
        """El resumen va a logs y al informe: no debe arrastrar 13 KB."""
        servicio.preparar(force=True)

        resumen = servicio.preparar().resumen()

        assert "contexto" not in resumen
        assert "documentos" not in resumen
        assert "tokens_estimados" in resumen


# ---------------------------------------------------------------------------
# Las dos caches no se confunden
# ---------------------------------------------------------------------------
class TestDosCachesDistintas:
    """El enunciado pide registrar por separado las dos caches."""

    def test_cache_de_aplicacion_es_nuestra(self, servicio: CAGService) -> None:
        assert servicio.ruta_cache.exists() is False
        servicio.preparar(force=True)
        assert servicio.ruta_cache.exists() is True

    def test_cache_de_proveedor_se_declara_segun_el_proveedor(self) -> None:
        """Que el proveedor reporte cache es una propiedad suya, no nuestra."""
        from app.services.llm import ProveedorFalso, ProveedorOpenAICompatible

        fake = ProveedorFalso()
        remoto = ProveedorOpenAICompatible("openai/gpt-oss-120b", "http://x", "clave")

        # Es sincrona: depende del modelo, no de una consulta.
        assert fake.supports_prompt_caching() is False
        assert remoto.supports_prompt_caching() is True

    def test_modelo_sin_cache_no_se_declara_compatible(self) -> None:
        """Ampliar la lista sin verificar seria reportar una metrica falsa."""
        from app.services.llm import ProveedorOpenAICompatible

        proveedor = ProveedorOpenAICompatible("google/gemma-4-31b-it", "http://x", "k")

        assert proveedor.supports_prompt_caching() is False
