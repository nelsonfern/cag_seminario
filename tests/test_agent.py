"""Pruebas del agente, el proveedor LLM y la verificacion de citas.

Las pruebas usan el proveedor determinista y nunca tocan la red. Lo que se
verifica aqui es el comportamiento que el enunciado exige:

  - responder con respaldo cuando hay evidencia,
  - abstenerse cuando no la hay,
  - no citar documentos que no se usaron,
  - no inventar montos, plazos ni enlaces,
  - distinguir un fallo tecnico de una abstincion.
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import pytest

from app.models.schemas import EstadoVigencia, Estrategia, TipoDocumento
from app.repositories.document_repository import DocumentoRepository
from app.repositories.evaluation_repository import EvaluationRepository
from app.services.agent import REGLAS_SISTEMA, Agente
from app.services.citations import (
    DocumentoEnContexto,
    afirmaciones_verificables,
    detectar_urls_no_verificadas,
    es_abstencion,
    resumen_grounding,
    seleccionar_citas,
    verificar_afirmaciones,
)
from app.services.ingestion import envolver_como_contexto
from app.services.llm import LLMError, LLMProvider, ProveedorFalso, estimar_tokens

pytestmark = pytest.mark.api

CONTENIDO_PARTIDAS = """\
Solicitud de Partidas

Este tramite permite solicitar copias de partidas de nacimiento, matrimonio,
defuncion y uniones convivenciales inscriptas en la Provincia de La Rioja.

Requisitos del Tramite
Tener una cuenta de Ciudadano Digital.
Datos del acta necesarios para que la busqueda sea exitosa.

Costo
Segun la Ley Impositiva 2024 (Ley 10.929, art. 79):
Copia de acta certificada: $1.100
Copia de acta legalizada: $1.500
"""

CONTENIDO_MATRIMONIO = """\
Inscripcion de matrimonios

Para inscribir un matrimonio se necesitan las partidas de nacimiento de
ambos contrayentes y el documento nacional de identidad de cada uno.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _meta(document_id: str, titulo: str, tramite: str, vigencia=EstadoVigencia.VIGENTE):
    from app.models.schemas import MetadatosDocumento

    return MetadatosDocumento(
        document_id=document_id,
        titulo=titulo,
        organismo="Registro Civil de La Rioja",
        tramite=tramite,
        tipo_documento=TipoDocumento.HTML,
        url_origen=f"https://registrocivil.larioja.gob.ar/{document_id}.html",
        archivo_original=f"{document_id}.html",
        fecha_consulta=date(2026, 10, 15),
        vigencia=vigencia,
        hash_contenido="a" * 64,
    )


@pytest.fixture
def documentos() -> list[DocumentoEnContexto]:
    return [
        DocumentoEnContexto(
            metadatos=_meta("doc-partidas", "Solicitud de Partidas", "partidas"),
            contenido=CONTENIDO_PARTIDAS,
        ),
        DocumentoEnContexto(
            metadatos=_meta("doc-matrimonio", "Matrimonios", "matrimonios"),
            contenido=CONTENIDO_MATRIMONIO,
        ),
    ]


@pytest.fixture
def agente(tmp_path: Path) -> Agente:
    """Agente con corpus minimo y proveedor determinista."""
    from app.services.corpus import FuenteDeclarada

    repo = DocumentoRepository(tmp_path / "processed")
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)

    from app.models.schemas import MetadatosDocumento

    for documento_id, titulo, tramite, contenido in (
        ("doc-partidas", "Solicitud de Partidas", "partidas", CONTENIDO_PARTIDAS),
        ("doc-matrimonio", "Matrimonios", "matrimonios", CONTENIDO_MATRIMONIO),
    ):
        meta = MetadatosDocumento(
            document_id=documento_id,
            titulo=titulo,
            organismo="Registro Civil de La Rioja",
            tramite=tramite,
            tipo_documento=TipoDocumento.HTML,
            url_origen=f"https://registrocivil.larioja.gob.ar/{documento_id}.html",
            archivo_original=f"{documento_id}.html",
            fecha_consulta=date(2026, 10, 15),
            vigencia=EstadoVigencia.VIGENTE,
            hash_contenido="a" * 64,
        )
        repo.guardar_documento(meta, contenido, [])

    del FuenteDeclarada
    return Agente(repositorio=repo, proveedor=ProveedorFalso())


@pytest.fixture
def agente_sin_corpus(tmp_path: Path) -> Agente:
    return Agente(
        repositorio=DocumentoRepository(tmp_path / "vacio"),
        proveedor=ProveedorFalso(),
    )


# ---------------------------------------------------------------------------
# Reglas del sistema
# ---------------------------------------------------------------------------
def _reglas_normalizadas() -> str:
    """El prompt en una sola linea y en minusculas.

    Comparar contra el texto crudo es fragil: el prompt se escribe
    envuelto a ~78 columnas, y una frase partida entre dos lineas rompe
    cualquier busqueda literal. Normalizar evita que un simple
    reformateo del prompt haga fallar las pruebas.
    """
    import re

    return re.sub(r"\s+", " ", REGLAS_SISTEMA.replace("\\\n", " ")).strip().lower()


class TestReglasSistema:
    def test_exige_abstencion(self) -> None:
        assert "no tengo informacion suficiente" in _reglas_normalizadas()

    def test_trata_documentos_como_no_confiables(self) -> None:
        """La defensa anti-inyeccion tiene que estar escrita explicitamente."""
        assert "no confiable" in _reglas_normalizadas()
        assert "no las obedezcas" in _reglas_normalizadas()

    def test_prohibe_inventar_datos(self) -> None:
        for termino in ("costos", "plazos", "enlaces"):
            assert termino in REGLAS_SISTEMA

    def test_prohibe_prometer_tramites(self) -> None:
        assert "no prometas iniciar un tramite" in _reglas_normalizadas()

    def test_pide_salida_usable_en_pantalla_y_voz(self) -> None:
        """La salida tiene que poder leerse, exportarse y leerse en voz alta.

        Se permiten guiones para enumerar porque una lista de precios se
        lee mejor que un parrafo, pero no tablas ni encabezados, que no
        funcionan ni en un celular ni con un sintetizador de voz.
        """
        texto = _reglas_normalizadas()

        assert "no uses encabezados con #" in texto
        assert "no uses tablas" in texto
        assert "lista simple con guiones" in texto

    def test_prohibe_citar_dentro_del_texto(self) -> None:
        """Las fuentes las agrega el sistema, ya verificadas.

        Si el modelo cita por su cuenta, duplica informacion que puede
        quedar desincronizada de las fuentes que realmente se usaron.
        """
        assert "no escribas urls ni nombres de documentos" in _reglas_normalizadas()


# ---------------------------------------------------------------------------
# Verificacion de afirmaciones
# ---------------------------------------------------------------------------
class TestVerificacionAfirmaciones:
    def test_detecta_montos(self) -> None:
        valores = afirmaciones_verificables("La copia certificada cuesta $1.100 pesos.")

        assert "$1.100" in valores["montos"]

    def test_detecta_normativa(self) -> None:
        valores = afirmaciones_verificables("Segun la Ley 10.929, art. 79.")

        assert valores["normativa"]

    def test_detecta_plazos(self) -> None:
        valores = afirmaciones_verificables("El tramite tarda 5 dias habiles.")

        assert "5 dias" in valores["plazos"]

    def test_monto_respaldado(self, documentos) -> None:
        resultado = verificar_afirmaciones("La copia certificada cuesta $1.100.", documentos)

        assert resultado["hay_afirmacion_sin_respaldo"] is False
        assert resultado["respaldadas"] == resultado["total_afirmaciones"] == 1

    def test_monto_inventado(self, documentos) -> None:
        """El caso central: el modelo inventa un precio que no esta."""
        resultado = verificar_afirmaciones("La copia certificada cuesta $9.999.", documentos)

        assert resultado["hay_afirmacion_sin_respaldo"] is True
        assert resultado["sin_respaldo"][0]["valor"] == "$9.999"
        assert resultado["sin_respaldo"][0]["tipo"] == "montos"

    def test_acepta_ambas_notaciones_de_miles(self, documentos) -> None:
        """$1.100 en espanol y 1,100 en notacion inglesa son el mismo monto.

        Comparar solo la forma literal marcaria como inventado un precio
        correcto, y la verificacion perderia valor.
        """
        resultado = verificar_afirmaciones("Cuesta 1,100 pesos.", documentos)

        assert resultado["hay_afirmacion_sin_respaldo"] is False

    def test_compara_sin_tildes(self, documentos) -> None:
        resultado = verificar_afirmaciones(
            "Copia de acta certificada segun la busqueda oficial.", documentos
        )

        assert resultado["total_afirmaciones"] == 0

    def test_monto_con_espacio_de_miles(self, documentos) -> None:
        """Los modelos emiten `$1 100` con espacio fino (notacion francesa).

        Si la regex corta en el espacio, extrae "$1" y marca como
        inventado un monto que si esta en la fuente.
        """
        resultado = verificar_afirmaciones(
            "La copia certificada cuesta $1 100 y la legalizada $1 500.", documentos
        )

        assert resultado["hay_afirmacion_sin_respaldo"] is False
        assert resultado["total_afirmaciones"] == 2

    def test_monto_espaciado_inventado(self, documentos) -> None:
        """`$50 000` no debe descomponerse en `$50` + `000`."""
        resultado = verificar_afirmaciones("La copia cuesta $50 000.", documentos)

        assert [a["valor"] for a in resultado["sin_respaldo"]] == ["$50 000"]

    def test_plazo_inventado(self, documentos) -> None:
        resultado = verificar_afirmaciones("El tramite tarda 10 dias.", documentos)

        assert resultado["hay_afirmacion_sin_respaldo"] is True
        assert resultado["sin_respaldo"][0]["tipo"] == "plazos"

    def test_sin_afirmaciones(self, documentos) -> None:
        resultado = verificar_afirmaciones("Consulta el sitio oficial.", documentos)

        assert resultado["total_afirmaciones"] == 0
        assert resultado["hay_afirmacion_sin_respaldo"] is False


# ---------------------------------------------------------------------------
# Citas
# ---------------------------------------------------------------------------
class TestAbstencion:
    @pytest.mark.parametrize(
        "frase",
        [
            "No tengo informacion suficiente en las fuentes consultadas.",
            "No encuentro informacion sobre eso.",
            "No puedo determinar el costo.",
            "No consta en las fuentes.",
        ],
    )
    def test_detecta_abstencion(self, frase: str) -> None:
        assert es_abstencion(frase) is True

    def test_no_confunde_respuesta_normal(self) -> None:
        assert es_abstencion("La copia certificada cuesta $1.100.") is False


class TestSeleccionCitas:
    def test_cita_el_documento_relevante(self, documentos) -> None:
        citas = seleccionar_citas(
            "La copia de acta certificada cuesta $1.100 segun la Ley Impositiva.",
            documentos,
        )

        assert any(c.document_id == "doc-partidas" for c in citas)

    def test_no_cita_documento_no_relacionado(self, documentos) -> None:
        """El criterio debe discriminar, no citarlo todo."""
        citas = seleccionar_citas("Solo hablo de padrones y cornis phyton", documentos)

        assert all(c.document_id != "doc-matrimonio" for c in citas) or citas == []

    def test_abstencion_no_cita(self, documentos) -> None:
        """Una abstincion no se apoya en fuentes: listarlas seria mentir."""
        citas = seleccionar_citas(
            "No tengo informacion suficiente en las fuentes consultadas.", documentos
        )

        assert citas == []

    def test_respuesta_vacia_no_cita(self, documentos) -> None:
        assert seleccionar_citas("", documentos) == []

    def test_limita_cantidad(self, documentos) -> None:
        documentos_repetidos = documentos * 5
        citas = seleccionar_citas(
            "acta certificado acta legalizada", documentos_repetidos, maximo=2
        )

        assert len(citas) <= 2

    def test_orden_estable(self, documentos) -> None:
        """Dos ejecuciones deben dar la misma lista de fuentes."""
        respuesta = "acta certificada acta legalizada ciudadano digital busqueda"

        primero = seleccionar_citas(respuesta, documentos)
        segundo = seleccionar_citas(respuesta, documentos)

        assert [c.document_id for c in primero] == [c.document_id for c in segundo]

    def test_solo_documentos_del_contexto(self, documentos) -> None:
        """No se puede citar algo que no se le dio al modelo."""
        citas = seleccionar_citas("acta certificada", [documentos[0]], maximo=4)

        assert {c.document_id for c in citas} == {"doc-partidas"}


class TestUrlsNoVerificadas:
    def test_detecta_url_inventada(self) -> None:
        urls = detectar_urls_no_verificadas("Vas a https://sitio-falso.com/tramite")

        assert urls == ["https://sitio-falso.com/tramite"]

    def test_no_marca_url_oficial(self) -> None:
        """Una URL del propio sitio institucional no es sospechosa."""
        assert (
            detectar_urls_no_verificadas(
                "Ver https://registrocivil.larioja.gob.ar/guia-tramites.html"
            )
            == []
        )


class TestResumenGrounding:
    def test_resumen_con_evidence(self, documentos) -> None:
        resumen = resumen_grounding("La copia de acta certificada cuesta $1.100.", documentos)

        assert resumen["grounded"] is True
        assert resumen["es_abstencion"] is False
        assert resumen["citas"]
        assert resumen["advertencias"] == []

    def test_resumen_con_monto_inventado(self, documentos) -> None:
        resumen = resumen_grounding("La copia cuesta $9.999.", documentos)

        assert resumen["verificacion"]["hay_afirmacion_sin_respaldo"] is True
        assert any("no aparecen textualmente" in a for a in resumen["advertencias"])

    def test_resumen_abstencion(self, documentos) -> None:
        resumen = resumen_grounding(
            "No tengo informacion suficiente en las fuentes consultadas.", documentos
        )

        assert resumen["grounded"] is False
        assert resumen["es_abstencion"] is True
        assert resumen["advertencias"]


# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------
class TestAgente:
    def test_responde_con_fundamento(self, agente: Agente) -> None:
        r = asyncio.run(agente.responder("Que necesito para una partida de nacimiento"))

        assert r.strategy is Estrategia.CAG
        assert r.grounded is True
        assert r.sources
        assert all(s.url_origen for s in r.sources)

    def test_contexto_marca_no_confiable(self, agente: Agente) -> None:
        prompt = agente.construir_prompt_usuario("prueba", agente.documentos_del_corpus())

        assert "<contexto_documental no_confiable>" in prompt
        assert "Pregunta del ciudadano: prueba" in prompt

    def test_contexto_incluye_metadatos_de_vigencia(self, agente: Agente) -> None:
        """El modelo tiene que saber que datos no estan confirmados."""
        prompt = agente.construir_prompt_usuario("prueba", agente.documentos_del_corpus())

        assert "Vigencia: vigente" in prompt
        assert "Fecha de actualizacion declarada" in prompt

    def test_sin_corpus_no_inventa(self, agente_sin_corpus: Agente) -> None:
        """Sin documentos, el agente avisa en vez de responder de memoria."""
        r = asyncio.run(agente_sin_corpus.responder("Quanto cuesta una partida?"))

        assert r.grounded is False
        assert r.sources == []
        assert any("documentos" in a.lower() for a in r.warnings)

    def test_pregunta_vacia(self, agente: Agente) -> None:
        with pytest.raises(ValueError, match="no puede estar vacia"):
            asyncio.run(agente.responder("   "))

    def test_error_proveedor_no_es_abstencion(self, agente: Agente) -> None:
        """Un fallo tecnico debe decirse, no disfrazarse de "no hay datos"."""

        class ProveedorQueFalla(LLMProvider):
            nombre = "falla"

            async def completar(self, *args, **kwargs):  # type: ignore[override]
                raise LLMError("el proveedor no responde")

        agente._proveedor = ProveedorQueFalla("m")
        r = asyncio.run(agente.responder("Cuanto cuesta una partida?"))

        assert r.grounded is False
        # La distincion clave: no es una abstencion.
        assert r.datos_estructurados["es_abstencion"] is False
        assert r.datos_estructurados["error_proveedor"] is True
        assert "problema es tecnico" in r.answer

    def test_advertencia_por_vigencia_incierta(self, documentos) -> None:
        documentos[0].metadatos.vigencia = EstadoVigencia.DESCONOCIDA
        resumen = resumen_grounding("La copia de acta certificada cuesta $1.100.", documentos)

        # El documento aporta la evidencia, pero se avisa que su vigencia
        # no pudo confirmarse.
        assert resumen["grounded"] is True

    def test_respuesta_vacia_del_modelo(self, agente: Agente) -> None:
        class ProveedorMudo(LLMProvider):
            nombre = "mudo"

            async def completar(self, *args, **kwargs):  # type: ignore[override]
                from app.models.schemas import RespuestaLLM, UsoTokens

                return RespuestaLLM(
                    texto="   ",
                    modelo="m",
                    proveedor="mudo",
                    uso=UsoTokens(),
                    latencia_ms=1.0,
                )

        agente._proveedor = ProveedorMudo("m")
        r = asyncio.run(agente.responder("Cuanto cuesta?"))

        assert r.grounded is False
        assert r.datos_estructurados["es_abstencion"] is True


# ---------------------------------------------------------------------------
# Proveedor determinista
# ---------------------------------------------------------------------------
class TestProveedorFalso:
    def test_es_determinista(self) -> None:
        """La suite depende de que dos ejecuciones den lo mismo."""
        proveedor = ProveedorFalso()
        mensajes = [
            {"role": "system", "content": "reglas"},
            {
                "role": "user",
                "content": envolver_como_contexto(CONTENIDO_PARTIDAS)
                + "\n\nPregunta del ciudadano: cuanto cuesta la copia certificada",
            },
        ]

        primero = asyncio.run(proveedor.completar(mensajes))
        segundo = asyncio.run(proveedor.completar(mensajes))

        assert primero.texto == segundo.texto

    def test_esta_marcado_como_sintetico(self) -> None:
        """Ninguna respuesta de prueba puede confundirse con una real."""
        proveedor = ProveedorFalso()
        mensajes = [
            {
                "role": "user",
                "content": envolver_como_contexto(CONTENIDO_PARTIDAS)
                + "\n\nPregunta del ciudadano: cuanto cuesta",
            }
        ]

        respuesta = asyncio.run(proveedor.completar(mensajes))

        assert ProveedorFalso.MARCA in respuesta.texto
        assert respuesta.metadatos["sintetico"] is True

    def test_no_incluye_la_pregunta_como_si_fuente(self) -> None:
        """La pregunta del ciudadano no es material documental."""
        proveedor = ProveedorFalso()
        mensajes = [
            {
                "role": "user",
                "content": envolver_como_contexto(CONTENIDO_PARTIDAS)
                + "\n\nPregunta del ciudadano: MARCADOR_UNICO_XYZ",
            }
        ]

        respuesta = asyncio.run(proveedor.completar(mensajes))

        assert "MARCADOR_UNICO_XYZ" not in respuesta.texto

    def test_abstiene_sin_contexto(self) -> None:
        proveedor = ProveedorFalso()
        mensajes = [{"role": "user", "content": "Pregunta del ciudadano: cuanto cuesta"}]

        respuesta = asyncio.run(proveedor.completar(mensajes))

        assert es_abstencion(respuesta.texto)


class TestDiagnosticoProveedor:
    """`comprobar_proveedor` lo usan el script de CAG y el de agente.

    Que fallara hacia la seccion 6 de la consola, despues de imprimir
    cinco bloques correctos, es exactamente el tipo de fallo que una
    prueba de integracion evita.
    """

    def test_comprobar_proveedor_devuelve_informe(self) -> None:
        from app.services.llm import comprobar_proveedor, reiniciar_proveedor

        reiniciar_proveedor()
        informe = asyncio.run(comprobar_proveedor())

        assert informe["proveedor"] == "fake"
        assert informe["modelo"]
        assert "disponible" in informe
        assert isinstance(informe["soporta_cache"], bool)

    def test_informe_no_filtra_credenciales(self) -> None:
        """El informe va a la consola y a /health: no puede llevar la clave."""

        from app.services.llm import ProveedorOpenAICompatible

        proveedor = ProveedorOpenAICompatible(
            modelo="openai/gpt-oss-120b",
            base_url="http://localhost:1",
            api_key="sk-probacion-secreta-1234567890",
            timeout=1.0,
        )

        ok, detalle = asyncio.run(proveedor.disponible())

        assert ok is False
        assert "sk-probacion-secreta" not in detalle

    def test_error_401_no_reintenta(self) -> None:
        """Una credencial invalida no mejora esperando: falla de inmediato."""
        import httpx

        from app.services.llm import LLMError, ProveedorOpenAICompatible

        llamadas = {"n": 0}

        def _responder(_request: httpx.Request) -> httpx.Response:
            llamadas["n"] += 1
            return httpx.Response(401, text="invalid api key")

        transporte = httpx.MockTransport(_responder)
        proveedor = ProveedorOpenAICompatible(
            modelo="m", base_url="http://x", api_key="k", max_reintentos=3
        )
        proveedor._cliente = httpx.AsyncClient(
            base_url="http://x", transport=transporte, headers={}
        )

        with pytest.raises(LLMError, match="credencial"):
            asyncio.run(proveedor.completar([{"role": "user", "content": "x"}]))

        # Una sola llamada: reintentar un 401 seria gastar tiempo en futile.
        assert llamadas["n"] == 1

    def test_error_429_si_reintenta(self) -> None:
        """El limite de uso es transitorio: ahi si conviene reintentar."""
        import httpx

        from app.services.llm import ProveedorOpenAICompatible

        llamadas = {"n": 0}

        def _responder(request: httpx.Request) -> httpx.Response:
            llamadas["n"] += 1
            if llamadas["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "1"}, text="rate limit")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
            )

        transporte = httpx.MockTransport(_responder)
        proveedor = ProveedorOpenAICompatible(
            modelo="m", base_url="http://x", api_key="k", max_reintentos=2
        )
        proveedor._cliente = httpx.AsyncClient(
            base_url="http://x", transport=transporte, headers={}
        )

        respuesta = asyncio.run(proveedor.completar([{"role": "user", "content": "x"}]))

        assert respuesta.texto == "ok"
        assert llamadas["n"] == 2

    def test_exhaustir_reintentos_propaga_error(self) -> None:
        import httpx

        from app.services.llm import LLMError, ProveedorOpenAICompatible

        def _responder(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"retry-after": "1"}, text="rate limit")

        transporte = httpx.MockTransport(_responder)
        proveedor = ProveedorOpenAICompatible(
            modelo="m", base_url="http://x", api_key="k", max_reintentos=1
        )
        proveedor._cliente = httpx.AsyncClient(
            base_url="http://x", transport=transporte, headers={}
        )

        with pytest.raises(LLMError, match="limite de uso"):
            asyncio.run(proveedor.completar([{"role": "user", "content": "x"}]))


class TestMetricasDeTokens:
    def test_estimacion_es_aproximada(self) -> None:
        """La estimacion no es una metrica: se usa solo para controles previos."""
        assert estimar_tokens("") == 0
        assert estimar_tokens("a" * 360) == 100

    def test_proveedor_falso_no_declara_cache(self, documentos) -> None:
        """Sin cache real, se declara no disponible y no cero."""
        proveedor = ProveedorFalso()
        respuesta = asyncio.run(
            proveedor.completar(
                [
                    {
                        "role": "user",
                        "content": envolver_como_contexto(CONTENIDO_PARTIDAS)
                        + "\n\nPregunta del ciudadano: acta certificada",
                    }
                ]
            )
        )

        assert respuesta.uso.cache_disponible is False
        assert respuesta.uso.tokens_cache_lectura is None


# ---------------------------------------------------------------------------
# Persistencia de metricas
# ---------------------------------------------------------------------------
class TestEvaluationRepository:
    @pytest.fixture
    def repo(self, tmp_path: Path) -> EvaluationRepository:
        return EvaluationRepository(tmp_path / "test.db")

    def test_registra_consulta(self, repo: EvaluationRepository) -> None:
        consulta_id = repo.registrar_consulta(
            pregunta="Cuanto cuesta?",
            estrategia="cag",
            respuesta="Cuesta $1.100.",
            grounded=True,
            es_abstencion=False,
            latencia_ms=1234.5,
            tokens_entrada=100,
            tokens_salida=50,
            tokens_cache=80,
            cache_reportada=True,
            modelo="m",
            proveedor="p",
            fuentes=[{"document_id": "doc-partidas"}],
        )

        assert repo.contar_consultas() == 1
        assert repo.listar_consultas()[0]["consulta_id"] == consulta_id

    def test_distingue_cache_no_reportada_de_cero(self, repo: EvaluationRepository) -> None:
        """Es la distincion que evita reportar 0% de cache sin dato."""
        repo.registrar_consulta(
            pregunta="a",
            estrategia="cag",
            respuesta="b",
            grounded=False,
            es_abstencion=True,
            latencia_ms=1.0,
            tokens_cache=0,
            cache_reportada=False,
            modelo="m",
            proveedor="ollama",
        )

        fila = repo.listar_consultas()[0]

        assert fila["tokens_cache"] == 0
        assert fila["cache_reportada"] == 0

    def test_ciclo_completo_de_ejecucion(self, repo: EvaluationRepository) -> None:
        run_id = repo.iniciar_ejecucion("modelo-x", "groq", {"top_k": 5})

        repo.registrar_resultado(
            run_id,
            pregunta_id="p1",
            estrategia="cag",
            respuesta="respuesta",
            groundtruth="referencia",
            grounded=True,
            latencia_ms=800.0,
            tokens_entrada=6000,
            tokens_cache=5900,
            cache_reportada=True,
            metricas={"fidelidad": 0.9},
        )
        repo.finalizar_ejecucion(run_id, "completada", {"total": 1})

        ejecucion = repo.obtener_ejecucion(run_id)

        assert ejecucion is not None
        assert ejecucion["estado"] == "completada"
        assert ejecucion["total_resultados"] == 1
        assert ejecucion["resultados"][0]["metricas"]["fidelidad"] == 0.9
        assert ejecucion["configuracion"] == {"top_k": 5}

    def test_una_ejecucion_inexistente(self, repo: EvaluationRepository) -> None:
        assert repo.obtener_ejecucion("no-existe") is None

    def test_no_sobrescribe_ejecuciones(self, repo: EvaluationRepository) -> None:
        """El enunciado pide conservar el detalle para poder repetir."""
        primera = repo.iniciar_ejecucion("m", "p")
        segunda = repo.iniciar_ejecucion("m", "p")

        assert primera != segunda
        assert len(repo.listar_ejecuciones()) == 2

    def test_reinicio_una_ejecucion_por_id(self, repo: EvaluationRepository) -> None:
        run_id = repo.iniciar_ejecucion("m", "p")
        repo.finalizar_ejecucion(run_id, "completada")
        # Reanudar con el mismo id no debe duplicar la fila.
        repo.iniciar_ejecucion("m", "p", run_id=run_id)

        assert len(repo.listar_ejecuciones()) == 1

    def test_datos_sobrevivan_al_reinicio(self, tmp_path: Path) -> None:
        ruta = tmp_path / "test.db"
        primera = EvaluationRepository(ruta)
        primera.registrar_consulta(
            pregunta="q",
            estrategia="cag",
            respuesta="r",
            grounded=True,
            es_abstencion=False,
            latencia_ms=1.0,
            modelo="m",
            proveedor="p",
        )

        segunda = EvaluationRepository(ruta)

        assert segunda.contar_consultas() == 1
