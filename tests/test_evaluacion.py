"""Pruebas del dataset y del evaluador.

El evaluador decide si el experimento vale algo. Si el evaluador esta
mal, los 100 resultados no dicen nada. Estas pruebas lo cubren con casos
que se ven a ojo.
"""

from __future__ import annotations

import pytest

from app.evaluation.dataset import PREGUNTAS_BASE, distribuir_por_categoria
from app.evaluation.scorer import (
    buscar_pregunta,
    comparar_estrategias,
    es_abstencion,
    faltan_datos,
    juzgar,
    resumir,
)


class TestDataset:
    def test_tiene_cincuenta_preguntas(self) -> None:
        assert len(PREGUNTAS_BASE) == 50

    def test_diez_por_categoria(self) -> None:
        conteo = distribuir_por_categoria(PREGUNTAS_BASE)
        assert conteo == {
            "requisitos": 10,
            "procedimiento": 10,
            "multi_seccion": 10,
            "ambigua": 10,
            "fuera_de_corpus": 10,
        }

    def test_los_identificadores_son_unicos(self) -> None:
        ids = [p["id"] for p in PREGUNTAS_BASE]
        assert len(set(ids)) == len(ids)

    def test_todas_tienen_los_campos_obligatorios(self) -> None:
        for p in PREGUNTAS_BASE:
            for campo in ("id", "categoria", "texto", "referencia", "debe_abstenerse"):
                assert campo in p, f"{p['id']} no tiene {campo}"

    def test_solo_fuera_de_corpus_exige_abstencion(self) -> None:
        """Una pregunta con respuesta en el corpus no debe exigir callarse."""
        for p in PREGUNTAS_BASE:
            if p["categoria"] == "fuera_de_corpus":
                assert p["debe_abstenerse"] is True, p["id"]
            else:
                assert p["debe_abstenerse"] is False, p["id"]

    def test_no_hay_placeholders(self) -> None:
        """Un texto de relleno habriafalseado el experimento entero."""
        for p in PREGUNTAS_BASE:
            texto = (p["texto"] + p["referencia"]).lower()
            for relleno in ("prueba unit", "lorem", "xxx", "todo", "nada", "..."):
                assert relleno not in texto, f"{p['id']} contiene '{relleno}'"

    def test_las_diez_de_fuera_son_imposibles_de_responder(self) -> None:
        """Las preguntas fuera de corpus tienen que ser de verdad insolubles."""
        for p in PREGUNTAS_BASE:
            if p["categoria"] != "fuera_de_corpus":
                continue
            # Si tiene datos clave, es respondible y esta mal clasificada.
            assert p["datos_clave"] == [], f"{p['id']} es respondible"
            referencia = p["referencia"].lower()
            assert (
                "no declara" in referencia
                or "no describe" in referencia
                or "no aclara" in referencia
            ), f"{p['id']}: {referencia[:70]}"

    def test_las_preguntas_con_respuesta_tienen_datos_clave(self) -> None:
        """Sin datos clave no hay forma de juzgar la respuesta.

        La unica excepcion es una pregunta cuya respuesta correcta es que
        la fuente no dice nada, y eso tiene que estar marcado de forma
        explicita para que no parezca un forgoton.
        """
        sin_datos = [
            p
            for p in PREGUNTAS_BASE
            if not p["datos_clave"] and p["categoria"] != "fuera_de_corpus"
        ]
        con_prefijo = [p["id"] for p in sin_datos if p["id"].startswith(("R", "P", "M"))]
        sin_marca = [i for i in con_prefijo if not _es_ausencia(i)]
        assert not sin_marca, f"Sin datos clave y sin marcar como ausencia: {sin_marca}"


def _es_ausencia(pregunta_id: str) -> bool:
    return buscar_pregunta(pregunta_id).get("respuesta_es_ausencia", False)


class TestDeteccionAbstencion:
    @pytest.mark.parametrize(
        "texto",
        [
            "No tengo informacion suficiente en las fuentes consultadas.",
            "La fuente no declara el monto.",
            "No consta en la documentacion disponible.",
            "No puedo confirmar ese dato.",
            "No figura ningun horario.",
        ],
    )
    def test_detecta_abstenciones(self, texto: str) -> None:
        assert es_abstencion(texto) is True

    @pytest.mark.parametrize(
        "texto",
        [
            "La copia certificada cuesta $1.100 segun la Ley Impositiva.",
            "Necesitas el certificado medico y el DNI de ambos progenitores.",
            "El plazo es de 40 dias corridos.",
        ],
    )
    def test_no_confunde_una_respuesta_con_abstencion(self, texto: str) -> None:
        assert es_abstencion(texto) is False

    def test_funciona_sin_acentos(self) -> None:
        """El modelo puede escribir con o sin tilde; el filtro no puede fallar."""
        con = "No tengo información suficiente."
        sin = "No tengo informacion suficiente."
        assert es_abstencion(con) is True
        assert es_abstencion(sin) is True

    def test_ignora_tildes_en_las_marcas(self) -> None:
        assert es_abstencion("No está declarado en la fuente.") is True


class TestDatosClave:
    def test_encuentra_numero_con_separadores(self) -> None:
        assert faltan_datos("Cuesta $1.100.", ["1.100"]) == []

    def test_encuentra_numero_sin_separadores(self) -> None:
        """Si el modelo escribe 1100, es el mismo dato."""
        assert faltan_datos("Cuesta $1100.", ["1.100"]) == []

    def test_encuentra_palabra_sin_tilde(self) -> None:
        assert faltan_datos("Certificado medico de nacimiento.", ["certificado médico"]) == []

    def test_detecta_lo_que_falta(self) -> None:
        assert faltan_datos("Cuesta mil cien pesos.", ["2", "4"]) == ["2", "4"]


class TestJuzgar:
    def test_acierto_completo(self) -> None:
        pregunta = buscar_pregunta("M01")
        veredicto = juzgar(
            "La copia certificada cuesta $1.100. Necesitás cuenta de Ciudadano Digital.",
            True,
            pregunta,
        )
        assert veredicto["acierto"] is True
        assert veredicto["fallo"] == ""

    def test_respuesta_incompleta(self) -> None:
        """Dice el precio pero no el requisito: cobertura, no honestidad."""
        pregunta = buscar_pregunta("M01")
        veredicto = juzgar("La copia certificada cuesta $1.100.", True, pregunta)
        assert veredicto["acierto"] is False
        assert "incompleta" in veredicto["fallo"]

    def test_abstencion_indebida(self) -> None:
        pregunta = buscar_pregunta("R08")
        veredicto = juzgar("No tengo informacion suficiente.", False, pregunta)
        assert veredicto["acierto"] is False
        assert "abstencion indebida" in veredicto["fallo"]

    def test_abstencion_correcta_fuera_de_corpus(self) -> None:
        pregunta = buscar_pregunta("F01")
        veredicto = juzgar("La fuente no declara horarios.", False, pregunta)
        assert veredicto["acierto"] is True
        assert veredicto["se_abstuvo"] is True

    def test_confabulacion_fuera_de_corpus(self) -> None:
        """El fallo mas grave: inventar un dato que no existe."""
        pregunta = buscar_pregunta("F02")
        veredicto = juzgar("El telefono es 0264-4300000 y atiende de 8 a 14.", True, pregunta)
        assert veredicto["acierto"] is False
        assert "respondio sin evidencia" in veredicto["fallo"]

    def test_no_penaliza_el_salto_de_fecha(self) -> None:
        """LaLey 10.929 sin el símbolo ° no es un dato faltante."""
        pregunta = buscar_pregunta("M06")
        veredicto = juzgar("El costo surge de la Ley 10.929, artículo 79.", True, pregunta)
        assert veredicto["datos_faltantes"] == []


class TestResumen:
    def test_cuenta_cada_tipo_de_fallo(self) -> None:
        veredictos = [
            juzgar("Cuesta $1.100 y necesitás Ciudadano Digital.", True, buscar_pregunta("M01")),
            juzgar("Cuesta $1.100.", True, buscar_pregunta("M01")),
            juzgar("No tengo informacion.", False, buscar_pregunta("R08")),
            juzgar("La fuente no declara el monto.", False, buscar_pregunta("F09")),
        ]
        resumen = resumir(veredictos)

        assert resumen["total"] == 4
        assert resumen["aciertos"] == 2
        assert resumen["abstenciones_indebidas"] == 1
        assert resumen["abstenciones_correctas"] == 1
        assert resumen["respuesta_indebida"] == 0

    def test_vacio_no_inventa_ceros(self) -> None:
        """Sin datos, la tasa es None. Cero seria un resultado falso."""
        resumen = resumir([])
        assert resumen["total"] == 0
        assert resumen["tasa_acierto"] is None

    def test_detecta_quien_respondio_sin_evidencia(self) -> None:
        veredictos = [
            juzgar("Atienden de 8 a 14.", True, buscar_pregunta("F01")),
            judgments := juzgar("No se declara.", False, buscar_pregunta("F02")),
        ]
        resumen = resumir(veredictos)
        assert resumen["respuesta_indebida"] == 1


class TestComparacion:
    def test_identifica_quien_acerto_y_quien_no(self) -> None:
        cag = [
            {"id": "A", "acierto": True},
            {"id": "B", "acierto": False},
            {"id": "C", "acierto": True},
        ]
        rag = [
            {"id": "A", "acierto": True},
            {"id": "B", "acierto": True},
            {"id": "C", "acierto": False},
        ]
        comparacion = comparar_estrategias(cag, rag)

        assert comparacion["preguntas_comparadas"] == 3
        assert comparacion["solo_cag_acierta"] == ["C"]
        assert comparacion["solo_rag_acierta"] == ["B"]
        assert comparacion["ambas_aciertan"] == 1
        assert comparacion["ninguna_acierta"] == []
