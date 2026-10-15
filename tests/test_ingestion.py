"""Pruebas de la ingesta documental.

Se prueba con archivos reales generados en el momento, no con mocks: el
riesgo de esta capa esta en si PyMuPDF, BeautifulSoup o la normalizacion
se comportan como se espera con contenido de verdad.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.models.schemas import EstadoFuente, EstadoVigencia, TipoDocumento
from app.repositories.document_repository import DocumentoRepository
from app.services.corpus import (
    FuenteDeclarada,
    Manifiesto,
    construir_documento,
    identificador_estable,
    nombre_archivo_desde_url,
)
from app.services.ingestion import (
    IngestionError,
    calcular_hash,
    detectar_tipo,
    envolver_como_contexto,
    evaluar_vigencia,
    extraer_documento,
    normalizar_texto,
)

pytestmark = pytest.mark.ingestion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
HTML_MUESTRA = """<!DOCTYPE html>
<html lang="es">
<head>
  <title>Solicitud de Partidas - Registro Civil La Rioja</title>
  <meta name="description" content="Solicitud de partidas en La Rioja.">
  <link rel="canonical" href="https://ejemplo.gob.ar/tramite-partidas.html">
  <script type="application/ld+json">
  {"@context": "https://schema.org", "@type": "Article",
   "datePublished": "2025-01-20", "dateModified": "2025-10-20"}
  </script>
  <script>console.log("esto no debe aparecer en el texto");</script>
  <style>.oculto { display: none; }</style>
</head>
<body>
  <nav>Inicio | Contacto | Login</nav>
  <header>Bandeja superior</header>
  <main>
    <h1>Solicitud de Partidas</h1>
    <p class="tramite-updated">Última actualización: 20/10/2025</p>
    <p>Este trámite permite solicitar copias de partidas.</p>
    <h2>Requisitos del Trámite</h2>
    <ul>
      <li>Tener una cuenta de Ciudadano Digital.</li>
      <li>Datos del acta necesarios para la búsqueda.</li>
    </ul>
    <h2>Costo</h2>
    <p>Según la Ley Impositiva 2024 (Ley 10.929, art. 79):</p>
    <ul>
      <li>Copia certificada: <strong>$1.100</strong></li>
      <li>Copia legalizada: <strong>$1.500</strong></li>
    </ul>
  </main>
  <footer>Pie de página</footer>
</body>
</html>
"""


@pytest.fixture
def html_valido(tmp_path: Path) -> Path:
    ruta = tmp_path / "tramite-partidas.html"
    ruta.write_text(HTML_MUESTRA, encoding="utf-8")
    return ruta


@pytest.fixture
def txt_valido(tmp_path: Path) -> Path:
    ruta = tmp_path / "nota.txt"
    ruta.write_text("Partida de nacimiento.\n\nArancel: sin cargo.", encoding="utf-8")
    return ruta


@pytest.fixture
def pdf_valido(tmp_path: Path) -> Path:
    pymupdf = pytest.importorskip("pymupdf")
    ruta = tmp_path / "ley.pdf"
    documento = pymupdf.open()
    pagina = documento.new_page()
    pagina.insert_text((72, 100), "Ley 10.929 - Ley Impositiva")
    documento.save(ruta)
    documento.close()
    return ruta


@pytest.fixture
def pdf_escaneado(tmp_path: Path) -> Path:
    """PDF sin capa de texto, como el que sale de un escaneo."""
    pymupdf = pytest.importorskip("pymupdf")
    ruta = tmp_path / "escaneado.pdf"
    documento = pymupdf.open()
    documento.new_page()
    documento.save(ruta)
    documento.close()
    return ruta


@pytest.fixture
def fuente_partidas() -> FuenteDeclarada:
    return FuenteDeclarada(
        url="https://registrocivil.larioja.gob.ar/tramite-solicitud-partidas.html",
        tramite="partidas",
        organismo="Registro Civil de La Rioja",
        fecha_consulta=date(2026, 10, 15),
    )


# ---------------------------------------------------------------------------
# Normalizacion
# ---------------------------------------------------------------------------
class TestNormalizacion:
    def test_colapsa_espacios_y_saltos(self) -> None:
        sucio = "Hola    mundo \t con   espacios\r\n\r\n\r\n\r\ny  saltos."
        limpio = normalizar_texto(sucio)

        assert limpio == "Hola mundo con espacios\n\ny saltos."

    def test_conserva_tildes_y_ene(self) -> None:
        """El espanol de las fuentes debe sobrevivir intacto."""
        texto = "Trámite, búsqueda, años, DNI"
        limpio = normalizar_texto(texto)

        for palabra in ("Trámite", "búsqueda", "años", "DNI"):
            assert palabra in limpio

    def test_unifica_comillas_tipograficas(self) -> None:
        """Las comillas curvas complican el conteo de tokens sin aportar nada."""
        limpio = normalizar_texto("dijo “hola” y ‘adiós’")

        assert limpio == "dijo \"hola\" y 'adiós'"

    def test_elimina_caracteres_invisibles(self) -> None:
        limpio = normalizar_texto("texto﻿ con espacio")

        assert limpio == "texto con espacio"

    def test_texto_vacio(self) -> None:
        assert normalizar_texto("") == ""
        assert normalizar_texto("   \n\n  ") == ""


class TestHash:
    def test_es_determinista(self) -> None:
        """El mismo contenido da siempre el mismo hash: es lo que permite
        detectar que un documento no cambio entre ejecuciones."""
        assert calcular_hash("texto") == calcular_hash("texto")

    def test_cambia_con_el_contenido(self) -> None:
        assert calcular_hash("texto") != calcular_hash("otro texto")

    def test_ignora_diferencias_de_espaciado(self) -> None:
        """Un reformateo no debe contar como documento modificado."""
        assert calcular_hash(normalizar_texto("a  b")) == calcular_hash(normalizar_texto("a b"))


# ---------------------------------------------------------------------------
# Deteccion de formato
# ---------------------------------------------------------------------------
class TestDeteccionFormato:
    def test_detecta_html(self, html_valido: Path) -> None:
        assert detectar_tipo(html_valido) is TipoDocumento.HTML

    def test_detecta_txt(self, txt_valido: Path) -> None:
        assert detectar_tipo(txt_valido) is TipoDocumento.TXT

    def test_detecta_pdf(self, pdf_valido: Path) -> None:
        assert detectar_tipo(pdf_valido) is TipoDocumento.PDF

    def test_extension_engañosa(self, pdf_valido: Path, tmp_path: Path) -> None:
        """Un PDF guardado con extension .html sigue siendo un PDF.

        Decidirlo solo por la extension daria texto vacio sin aviso.
        """
        mentiroso = tmp_path / "ley.html"
        mentiroso.write_bytes(pdf_valido.read_bytes())

        assert detectar_tipo(mentiroso) is TipoDocumento.PDF
        texto, _ = extraer_documento(mentiroso)
        assert "Ley 10.929" in texto

    def test_formato_no_soportado(self, tmp_path: Path) -> None:
        raro = tmp_path / "datos.bin"
        raro.write_bytes(b"\x00\x01\x02\x03")

        assert detectar_tipo(raro) is TipoDocumento.DESCONOCIDO
        with pytest.raises(IngestionError, match="no soportado"):
            extraer_documento(raro)

    def test_archivo_inexistente(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="no existe"):
            extraer_documento(tmp_path / "no-existe.pdf")


# ---------------------------------------------------------------------------
# Extraccion
# ---------------------------------------------------------------------------
class TestExtraccionHTML:
    def test_extrae_contenido_util(self, html_valido: Path) -> None:
        texto, _ = extraer_documento(html_valido)

        assert "Este trámite permite solicitar copias de partidas." in texto
        # Los costos deben llegar intactos: son el dato mas sensible del corpus.
        assert "$1.100" in texto and "$1.500" in texto

    def test_ignora_navegacion_scripts_y_estilos(self, html_valido: Path) -> None:
        """Navegacion, scripts y CSS no son contenido y contaminarian el indice."""
        texto, _ = extraer_documento(html_valido)

        for ruido in ("Inicio | Contacto", "console.log", "display: none", "Pie de página"):
            assert ruido not in texto

    def test_lee_metadatos_de_json_ld(self, html_valido: Path) -> None:
        """Las fechas del JSON-LD deben leerse antes de borrar los scripts."""
        _, meta = extraer_documento(html_valido)

        assert meta["fecha_publicacion"] == date(2025, 1, 20)
        assert meta["fecha_actualizacion"] == date(2025, 10, 20)

    def test_lee_fecha_del_cuerpo(self, tmp_path: Path) -> None:
        """Paginas sin JSON-LD declaran la fecha en el texto."""
        ruta = tmp_path / "p.html"
        ruta.write_text(
            "<html><body><main><h1>T</h1>"
            "<p>Última actualización: 05/03/2026</p><p>Contenido.</p>"
            "</main></body></html>",
            encoding="utf-8",
        )

        _, meta = extraer_documento(ruta)

        assert meta["fecha_actualizacion"] == date(2026, 3, 5)

    def test_detecta_secciones(self, html_valido: Path) -> None:
        """Las secciones permiten fragmentar por parte del documento."""
        _, meta = extraer_documento(html_valido)

        assert "Costo" in meta["secciones"]
        assert "Requisitos del Trámite" in meta["secciones"]

    def test_html_sin_contenido_falla_explicitamente(self, tmp_path: Path) -> None:
        """Fallar en silencio meteria un documento vacio al corpus y el agente
        se abstendria por una causa que nadie registro."""
        ruta = tmp_path / "vacio.html"
        ruta.write_text("<html><body></body></html>", encoding="utf-8")

        with pytest.raises(IngestionError, match="no contiene texto"):
            extraer_documento(ruta)


class TestExtraccionOtrosFormatos:
    def test_txt(self, txt_valido: Path) -> None:
        texto, meta = extraer_documento(txt_valido)

        assert "Partida de nacimiento." in texto
        assert meta["codificacion"] == "utf-8"

    def test_pdf_registra_paginas(self, pdf_valido: Path) -> None:
        """El numero de pagina permite citar el origen con precision."""
        texto, meta = extraer_documento(pdf_valido)

        assert "[pagina 1]" in texto
        assert "Ley 10.929" in texto
        assert meta["n_paginas"] == 1

    def test_pdf_escaneado_falla_con_mensaje_util(self, pdf_escaneado: Path) -> None:
        """Un escaneo sin OCR no debe entrar al corpus fingiendo estar vacio."""
        with pytest.raises(IngestionError, match="OCR"):
            extraer_documento(pdf_escaneado)


# ---------------------------------------------------------------------------
# Identidad de documentos
# ---------------------------------------------------------------------------
class TestIdentidad:
    def test_id_estable_no_depende_del_contenido(self) -> None:
        """Si el id dependiera del contenido, cada actualizacion de la fuente
        invalidaria las citas ya emitidas."""
        url = "https://ejemplo.gob.ar/tramite.html"

        assert identificador_estable(url, "a.html") == identificador_estable(url, "b.html")

    def test_id_difiere_entre_fuentes(self) -> None:
        assert identificador_estable("https://a.gob.ar/x.html", "x") != identificador_estable(
            "https://b.gob.ar/y.html", "y"
        )

    def test_id_para_archivo_local(self) -> None:
        assert identificador_estable(None, "ley-10929.pdf") == "doc-ley-10929"

    def test_nombre_desde_url(self) -> None:
        assert (
            nombre_archivo_desde_url("https://registrocivil.larioja.gob.ar/tramite-partidas.html")
            == "tramite-partidas.html"
        )

    def test_nombre_de_url_sin_extension(self) -> None:
        assert nombre_archivo_desde_url("https://ejemplo.gob.ar/tramites") == "tramites.html"

    def test_nombre_de_url_con_query(self) -> None:
        assert nombre_archivo_desde_url("https://ejemplo.gob.ar/p.html?id=7#seccion") == "p.html"


class TestMetadatos:
    def test_conserva_datos_declarados(self, html_valido: Path, fuente_partidas) -> None:
        meta = construir_documento(html_valido, fuente_partidas)

        assert meta.organismo == "Registro Civil de La Rioja"
        assert meta.tramite == "partidas"
        assert meta.url_origen == fuente_partidas.url
        assert meta.archivo_original == "tramite-partidas.html"
        assert meta.fecha_actualizacion == date(2025, 10, 20)

    def test_lo_no_declarado_queda_vacio(self, html_valido: Path, fuente_partidas) -> None:
        """Lo que la fuente no dice no se completa con suposiciones."""
        meta = construir_documento(html_valido, fuente_partidas)

        assert meta.observaciones is None

    def test_hash_estable_entre_ejecuciones(self, html_valido: Path, fuente_partidas) -> None:
        primero = construir_documento(html_valido, fuente_partidas)
        segundo = construir_documento(html_valido, fuente_partidas)

        assert primero.hash_contenido == segundo.hash_contenido


# ---------------------------------------------------------------------------
# Vigencia
# ---------------------------------------------------------------------------
class TestVigencia:
    def test_sin_fecha_es_desconocida(self) -> None:
        """No se infiere vigencia por estar publicado."""
        assert evaluar_vigencia(None) is EstadoVigencia.DESCONOCIDA

    def test_reciente_es_vigente(self) -> None:
        assert evaluar_vigencia(date(2026, 9, 1), date(2026, 10, 15)) is EstadoVigencia.VIGENTE

    def test_antigua_es_parcial(self) -> None:
        """Con mas de un ano no se puede afirmar vigente sin mirar la norma."""
        assert evaluar_vigencia(date(2025, 1, 20), date(2026, 10, 15)) is EstadoVigencia.PARCIAL


# ---------------------------------------------------------------------------
# Repositorio y manifiesto
# ---------------------------------------------------------------------------
class TestRepositorio:
    @pytest.fixture
    def repo(self, tmp_path: Path) -> DocumentoRepository:
        return DocumentoRepository(tmp_path / "processed")

    def test_guarda_y_relee(
        self, repo: DocumentoRepository, html_valido: Path, fuente_partidas
    ) -> None:
        repo.procesar_archivo(html_valido, fuente_partidas)

        documentos = repo.cargar_documentos()

        assert len(documentos) == 1
        # El id sale de la URL, no del contenido.
        assert documentos[0].document_id == identificador_estable(
            fuente_partidas.url, "tramite-partidas.html"
        )
        assert repo.cargar_contenidos()[documentos[0].document_id]

    def test_reingesta_actualiza_en_lugar_de_duplicar(
        self, repo: DocumentoRepository, html_valido: Path, fuente_partidas
    ) -> None:
        repo.procesar_archivo(html_valido, fuente_partidas)
        repo.procesar_archivo(html_valido, fuente_partidas)

        assert len(repo.cargar_documentos()) == 1

    def test_cambio_de_contenido_queda_en_historial(
        self, repo: DocumentoRepository, html_valido: Path, fuente_partidas
    ) -> None:
        """Que un documento haya cambiado es informacion que hay que auditar."""
        repo.procesar_archivo(html_valido, fuente_partidas)
        html_valido.write_text(HTML_MUESTRA.replace("$1.100", "$1.300"), encoding="utf-8")
        repo.procesar_archivo(html_valido, fuente_partidas)

        entradas = repo.cargar_entradas()
        historial = entradas[0]["historial_ingesta"]

        assert len(entradas) == 1
        assert historial and historial[0]["cambio"] == "contenido_distinto"

    def test_corpus_se_relee_sin_extras(
        self, repo: DocumentoRepository, html_valido: Path, fuente_partidas
    ) -> None:
        """Los metadatos usan extra=forbid: un campo extra romperia la lectura."""
        repo.procesar_archivo(html_valido, fuente_partidas)

        assert repo.cargar_documentos()

    def test_filtra_por_tramite(
        self, repo: DocumentoRepository, html_valido: Path, fuente_partidas
    ) -> None:
        repo.procesar_archivo(html_valido, fuente_partidas)
        repo.procesar_archivo(
            html_valido,
            FuenteDeclarada(url="https://x.gob.ar/otros.html", tramite="defunciones"),
        )

        assert len(repo.por_tramite("partidas")) == 1
        assert len(repo.por_tramite("defunciones")) == 1
        assert len(repo.por_tramite()) == 2
        assert repo.por_tramite("inexistente") == []

    def test_informe_detecta_duplicados(self, repo: DocumentoRepository, html_valido: Path) -> None:
        """Dos URLs con el mismo contenido son el mismo documento."""
        repo.procesar_archivo(
            html_valido,
            FuenteDeclarada(url="https://a.gob.ar/x.html", tramite="partidas"),
        )
        copia = html_valido.parent / "copia.html"
        copia.write_text(HTML_MUESTRA, encoding="utf-8")
        repo.procesar_archivo(
            copia, FuenteDeclarada(url="https://b.gob.ar/x.html", tramite="partidas")
        )

        informe = repo.informe()

        assert informe["total_documentos"] == 2
        assert len(informe["duplicados_por_contenido"]) == 1

    def test_documento_inaccesible_no_cuenta_como_valido(
        self, repo: DocumentoRepository, html_valido: Path
    ) -> None:
        repo.procesar_archivo(
            html_valido,
            FuenteDeclarada(url="https://x.gob.ar/x.html", tramite="p", permitido=False),
        )

        assert repo.cargar_documentos()[0].estado_verificacion is EstadoFuente.NO_ACCESIBLE
        assert repo.documentos_validos() == []


class TestManifiesto:
    def test_registra_y_busca(self, tmp_path: Path) -> None:
        manifiesto = Manifiesto()
        manifiesto.registrar({"url": "https://x.gob.ar/a.html", "titulo": "A"})

        assert manifiesto.buscar_por_url("https://x.gob.ar/a.html") is not None
        assert manifiesto.buscar_por_url("https://x.gob.ar/b.html") is None

    def test_reregistrar_actualiza_sin_duplicar(self, tmp_path: Path) -> None:
        manifiesto = Manifiesto()
        manifiesto.registrar({"url": "u", "titulo": "A", "fecha_incorporacion": "2026-01-01"})
        manifiesto.registrar({"url": "u", "titulo": "B"})

        assert len(manifiesto.fuentes) == 1
        assert manifiesto.fuentes[0]["titulo"] == "B"
        # La fecha de incorporacion original se conserva: importa para auditar.
        assert manifiesto.fuentes[0]["fecha_incorporacion"] == "2026-01-01"

    def test_persistencia(self, tmp_path: Path) -> None:
        ruta = tmp_path / "manifest.json"
        Manifiesto().registrar({"url": "u", "titulo": "A"})
        Manifiesto.cargar(ruta)

        m = Manifiesto()
        m.registrar({"url": "u", "titulo": "A"})
        m.guardar(ruta)

        assert Manifiesto.cargar(ruta).buscar_por_url("u") is not None

    def test_manifiesto_corrupto_no_borra_historico(self, tmp_path: Path) -> None:
        """Un JSON roto se aparta en vez de sobrescribir el archivo original."""
        ruta = tmp_path / "manifest.json"
        ruta.write_text("{roto", encoding="utf-8")

        m = Manifiesto.cargar(ruta)

        assert m.fuentes == []
        assert ruta.with_suffix(".json.corrupto").exists()


# ---------------------------------------------------------------------------
# Defensa frente a inyeccion
# ---------------------------------------------------------------------------
class TestContextoNoConfiable:
    def test_marca_el_contenido(self) -> None:
        envuelto = envolver_como_contexto("El arancel es $500.")

        assert envuelto.startswith("<contexto_documental no_confiable>")
        assert "NO instrucciones" in envuelto
        assert "El arancel es $500." in envuelto

    def test_cierra_el_bloque(self) -> None:
        """Sin cierre, un documento podria 'escapar' hacia el resto del prompt."""
        envuelto = envolver_como_contexto("texto")

        assert envuelto.count("<contexto_documental no_confiable>") == 2
        assert envuelto.count("</contexto_documental>") == 2


# ---------------------------------------------------------------------------
# Corpus real
# ---------------------------------------------------------------------------
class TestCorpusReal:
    """Pruebas sobre los documentos efectivamente descargados.

    Se saltan si el corpus aun no se generó: la suite debe pasar en una
    instalacion limpia, sin red.
    """

    @pytest.fixture
    def repo_real(self) -> DocumentoRepository:
        from app.core.config import get_settings

        repo = DocumentoRepository(get_settings().processed_dir)
        if not repo.ruta_corpus.exists():
            pytest.skip("El corpus no se ha descargado todavia.")
        return repo

    def test_hay_documentos_ingeridos(self, repo_real: DocumentoRepository) -> None:
        documentos = repo_real.cargar_documentos()

        assert len(documentos) >= 5
        assert all(d.hash_contenido for d in documentos)
        assert all(d.fecha_consulta for d in documentos)

    def test_los_costos_no_se_pierden(self, repo_real: DocumentoRepository) -> None:
        """El corpus debe conservar los montos citados en la fuente."""
        contenidos = repo_real.cargar_contenidos()
        unido = "\n".join(contenidos.values())

        assert "$1.100" in unido
        assert "Ley 10.929" in unido

    def test_todo_documento_tiene_url_de_origen(self, repo_real: DocumentoRepository) -> None:
        """Sin URL no hay trazabilidad, que es el punto del sistema."""
        assert repo_real.informe()["sin_url_de_origen"] == []

    def test_manifesto_refleja_el_corpus(self, repo_real: DocumentoRepository) -> None:
        manifiesto = Manifiesto.cargar(repo_real.ruta_manifiesto)
        # Una fuente bloqueada nunca se ingesto y por eso no tiene
        # document_id: se usa get() en lugar de [].
        en_manifesto = {f["document_id"] for f in manifiesto.fuentes if f.get("document_id")}
        en_corpus = {d.document_id for d in repo_real.cargar_documentos()}

        assert en_corpus <= en_manifesto

    def test_fuente_bloqueada_queda_registrada(self, repo_real: DocumentoRepository) -> None:
        """Una fuente no accesible debe quedar en el manifiesto, no desaparecer."""
        manifiesto = Manifiesto.cargar(repo_real.ruta_manifiesto)
        bloqueadas = [
            f
            for f in manifiesto.fuentes
            if f.get("estado_verificacion") == EstadoFuente.NO_ACCESIBLE.value
        ]

        assert bloqueadas, "Se esperaba al menos una fuente bloqueada por robots.txt"
        assert "argentina.gob.ar" in json.dumps(bloqueadas)
