"""Verificacion del detector de abstencion, con casos que se ven a ojo.

Un evaluador que confunde "no tengo informacion" con una respuesta real
deja pasar el peor fallo posible: un sistema que dice que no sabe y
despues inventa un numero de telefono.
"""

from __future__ import annotations

import pytest

from app.evaluation.scorer import es_abstencion

ABSTENCIONES = [
    "No tengo informacion suficiente en las fuentes consultadas.",
    "La fuente no declara el monto.",
    "No consta en la documentacion disponible.",
    "No puedo confirmar ese dato.",
    "No figura ningun horario.",
    "No esta declarado en la fuente.",
    "No se especifica ningun costo.",
    "No se encuentra publicado el horario.",
    "No se indica ningun plazo de entrega.",
    "Sin informacion suficiente para responder.",
]

RESPUESTAS = [
    "La copia certificada cuesta $1.100 segun la Ley Impositiva.",
    "Necesitas el certificado medico y el DNI de ambos progenitores.",
    "El plazo es de 40 dias corridos y se necesitan 2 testigos.",
    "Los turnos se piden con dos meses de anticipacion en Casa Central.",
    "La inscripcion es gratuita y se puede iniciar el DNI del bebe.",
]

# El caso importante: dice que no sabe y despues afirma algo.
CONTRADICTORIAS = [
    "No tengo informacion. Pero el telefono es 0264-4300000.",
    "La fuente no declara el monto, aunque figura que cuesta $1.100.",
    "No consta ningun horario, la atencion es de 8 a 14 horas.",
]


class TestDetectorAbstencion:
    @pytest.mark.parametrize("texto", ABSTENCIONES)
    def test_reconoce_abstenciones(self, texto: str) -> None:
        assert es_abstencion(texto) is True, texto

    @pytest.mark.parametrize("texto", RESPUESTAS)
    def test_no_confunde_respuestas_con_abstenciones(self, texto: str) -> None:
        assert es_abstencion(texto) is False, texto

    @pytest.mark.parametrize("texto", CONTRADICTORIAS)
    def test_atrapa_la_abstencion_contradictoria(self, texto: str) -> None:
        """Decir 'no tengo informacion' y despues inventar no es abstenerse.

        Es el peor fallo posible disfrazado de prudencia, y tiene que
        quedar registrado como respuesta indebida.
        """
        assert es_abstencion(texto) is False, texto

    def test_funciona_con_y_sin_tildes(self) -> None:
        """El modelo puede escribir como quiera; el filtro no puede fallar."""
        for con, sin in (
            ("No tengo información suficiente.", "No tengo informacion suficiente."),
            ("No está declarado.", "No esta declarado."),
            ("La fuente no declara el monto.", "La fuente no declara el monto."),
        ):
            assert es_abstencion(con) is True, con
            assert es_abstencion(sin) is True, sin

    def test_una_frase_corta_sin_contenido_no_cuenta(self) -> None:
        """Quedarse en 'No se declara.' sin nada mas es abstenerse."""
        assert es_abstencion("No se declara.") is True

    def test_abstencion_con_explicacion_sigue_siendo_abstencion(self) -> None:
        """Explicar por que no se sabe no la convierte en respuesta."""
        texto = (
            "La fuente no declara el costo del turno de matrimonio. "
            "Solo menciona que surge de la Ley Impositiva, pero el monto "
            "aparece sin valor en la pagina oficial."
        )
        assert es_abstencion(texto) is True
