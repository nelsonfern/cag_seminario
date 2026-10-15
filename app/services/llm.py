"""Proveedores de modelo de lenguaje, intercambiables.

El resto del sistema nunca habla con un proveedor concreto: habla con la
interfaz `LLMProvider`. Cambiar de Ollama a una API es cambiar una
variable de entorno, no el codigo.

Reglas que este modulo respeta:

1. **Ninguna clave en el codigo.** Se leen de la configuracion.
2. **No se inventan metricas.** `UsoTokens` distingue explicitamente
   entre "el proveedor no lo reporta" (None) y "reporto cero" (0).
3. **Los fallos son explicitos.** Un error del proveedor se propaga como
   `LLMError`; nunca se degrada a una respuesta vacia, porque el agente
   interpretaria esa vacuidad como "no hay informacion" en lugar de
   "no pude preguntar".
"""

from __future__ import annotations

import abc
import asyncio
import json
import re
import time
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger, redact
from app.models.schemas import RespuestaLLM, UsoTokens

logger = get_logger("llm")


class LLMError(Exception):
    """Fallo del proveedor de modelo de lenguaje.

    Se distingue del hecho de "no hay informacion en el corpus": un error
    del proveedor debe llevar a una respuesta que lo diga, no a una
    abstincion que ocultaria un fallo tecnico.
    """


# ---------------------------------------------------------------------------
# Estimacion de tokens
# ---------------------------------------------------------------------------
# Se usa solo para decidir cuanto contexto entra antes de enviarlo, nunca
# para reportar consumo. El consumo reportado siempre viene del proveedor.
_CHARS_POR_TOKEN = 3.6  # aproximacion para espanol


def estimar_tokens(texto: str) -> int:
    """Estimacion aproximada de tokens.

    Deliberadamente burda y sin dependencias. Sirve para un control previo
    (no exceder la ventana) donde un error del 10% es irrelevante. Los
    numeros que van al informe de metricas salen del proveedor, no de aqui.
    """
    return int(len(texto) / _CHARS_POR_TOKEN)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
class LLMProvider(abc.ABC):
    """Contrato comun a todos los proveedores."""

    #: Nombre corto del proveedor, para logs y metricas.
    nombre: str = "abstracto"

    def __init__(self, modelo: str) -> None:
        self.modelo = modelo

    @abc.abstractmethod
    async def completar(
        self,
        mensajes: list[dict[str, str]],
        temperatura: float | None = None,
        tokens_maximos: int | None = None,
        esfuerzo_razonamiento: str | None = None,
    ) -> RespuestaLLM:
        """Genera una respuesta para el historial de mensajes."""

    async def disponible(self) -> tuple[bool, str]:
        """Comprueba si el proveedor responde. Devuelve (ok, detalle)."""
        return True, "no verificado"

    def supports_prompt_caching(self) -> bool:
        """Si el proveedor expone metricas de cache de prompt."""
        return False

    async def cerrar(self) -> None:
        """Libera recursos.

        No es abstracta a proposito: la mayoria de los proveedores no
        mantienen conexiones abiertas, y obligar a implementar un metodo
        vacio solo genera ruido. Quien si las mantiene lo sobrescribe.
        """
        return None


# ---------------------------------------------------------------------------
# Endpoint compatible con la API de OpenAI
# ---------------------------------------------------------------------------
class ProveedorOpenAICompatible(LLMProvider):
    """Sirve para OpenAI, Groq, OpenRouter, Together, LM Studio, vLLM...

    El formato `chat/completions` es el mas extendido entre proveedores
    abiertos, asi que una sola implementacion cubre la mayoria.
    """

    nombre = "openai_compatible"

    def __init__(
        self,
        modelo: str,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        max_reintentos: int = 2,
    ) -> None:
        super().__init__(modelo)
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self.max_reintentos = max_reintentos
        self._cliente: httpx.AsyncClient | None = None

    def _obtener_cliente(self) -> httpx.AsyncClient:
        if self._cliente is None:
            cabeceras = {"Content-Type": "application/json"}
            if self._api_key:
                cabeceras["Authorization"] = f"Bearer {self._api_key}"
            self._cliente = httpx.AsyncClient(
                base_url=self.base_url,
                headers=cabeceras,
                timeout=self._timeout,
            )
        return self._cliente

    async def supports_prompt_caching(self) -> bool:
        """Solo algunos proveedores reportan `cached_tokens`.

        No se asume: se comprueba contra la lista de modelos de Groq,
        que es donde el proyecto lo usa. Para el resto se devuelve False y
        el agente registra que la metrica no esta disponible, en vez de
        reportar cero.
        """
        return self.modelo in _MODELOS_CON_CACHE

    async def completar(
        self,
        mensajes: list[dict[str, str]],
        temperatura: float | None = None,
        tokens_maximos: int | None = None,
        esfuerzo_razonamiento: str | None = None,
    ) -> RespuestaLLM:
        ajustes = get_settings()
        payload: dict[str, Any] = {
            "model": self.modelo,
            "messages": mensajes,
            "temperature": (temperatura if temperatura is not None else ajustes.llm_temperature),
            "max_tokens": tokens_maximos or ajustes.llm_max_output_tokens,
            "stream": False,
        }
        if esfuerzo_razonamiento:
            payload["reasoning_effort"] = esfuerzo_razonamiento

        cliente = self._obtener_cliente()
        inicio = time.perf_counter()

        # Reintento respetando lo que el propio proveedor indica. Sin esto,
        # una consulta que cae en el limite se pierde, cuando el problema se
        # resuelve solo en unos segundos. Solo se reintenta lo transitorio:
        # un 401 por credencial invalida no mejora esperando.
        datos: dict[str, Any] | None = None
        ultimo_error: LLMError | None = None
        espera_sugerida = 0.0

        for intento in range(self.max_reintentos + 1):
            if intento > 0:
                # Se usa la espera que indica el proveedor en lugar de una
                # espera a ciegas: esperar de mas estira el experimento sin
                # agregar informacion.
                espera = espera_sugerida or 2.0**intento
                logger.warning(
                    "Reintento %d/%d; esperando %.1fs por limite del proveedor",
                    intento,
                    self.max_reintentos,
                    espera,
                )
                await asyncio.sleep(espera)

            try:
                respuesta = await cliente.post("/chat/completions", json=payload)
                respuesta.raise_for_status()
                datos = respuesta.json()
                break
            except httpx.HTTPStatusError as exc:
                codigo = exc.response.status_code
                detalle = redact(exc.response.text[:400])
                if codigo == 429 or codigo >= 500:
                    # 429 es limite de cuota y 5xx es fallo del proveedor:
                    # ambos se resuelven esperando.
                    espera_sugerida = _espera_de_cabeceras(exc.response.headers)
                    ultimo_error = LLMError(
                        f"El proveedor alcanzo su limite de uso ({codigo}). "
                        + (
                            f"Reintentar en {espera_sugerida:.0f}s."
                            if espera_sugerida
                            else "Reintentar en unos segundos."
                        )
                    )
                    continue
                if codigo in (401, 403):
                    raise LLMError(
                        f"El proveedor rechazo la credencial ({codigo}). "
                        "Revisar OPENAI_API_KEY y los permisos de la cuenta."
                    ) from exc
                raise LLMError(f"El proveedor respondio {codigo}: {detalle}") from exc
            except httpx.HTTPError as exc:
                ultimo_error = LLMError(f"No se pudo contactar al proveedor: {redact(str(exc))}")

        if datos is None:
            raise ultimo_error or LLMError("El proveedor no devolvio respuesta.")

        latencia_ms = (time.perf_counter() - inicio) * 1000

        if "choices" not in datos or not datos["choices"]:
            raise LLMError(f"Respuesta sin contenido: {redact(str(datos)[:300])}")

        mensaje = datos["choices"][0]["message"]
        texto = mensaje.get("content") or ""
        razonamiento = mensaje.get("reasoning")

        return RespuestaLLM(
            texto=texto,
            modelo=self.modelo,
            proveedor=self.nombre,
            uso=_uso_de_openai(datos.get("usage")),
            latencia_ms=round(latencia_ms, 1),
            metadatos={
                # El razonamiento se conserva aparte y no se muestra. Los
                # modelos gpt-oss lo emiten siempre y no tiene sentido
                # agregarlo a la respuesta ni a un sintetizador de voz.
                "razonamiento_tokens": len(razonamiento) if razonamiento else 0,
                "finish_reason": datos["choices"][0].get("finish_reason"),
            },
        )

    async def disponible(self) -> tuple[bool, str]:
        cliente = self._obtener_cliente()
        try:
            respuesta = await cliente.get("/models", timeout=15.0)
            respuesta.raise_for_status()
        except httpx.HTTPError as exc:
            return False, redact(str(exc))
        return True, "endpoint accesible"

    async def cerrar(self) -> None:
        if self._cliente is not None:
            await self._cliente.aclose()
            self._cliente = None


#: Unica lista de modelos que se sabe que reportan `cached_tokens`.
#: Se mantiene acotada a proposito: ampliarla sin verificar seria reportar
#: una metrica que el proveedor podria no enviar, que es exactamente lo que
#: el enunciado prohibe.
_MODELOS_CON_CACHE = frozenset(
    {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
    }
)


def _espera_de_cabeceras(cabeceras: httpx.Headers) -> float:
    """Cuantos segundos sugiere el proveedor esperar antes de reintentar.

    Se leen `retry-after` y `x-ratelimit-reset-tokens`. El limite que
    realmente aprieta en el plan gratuito de Groq no es el de peticiones
    (1000/dia) sino el de tokens (8.000/minuto), asi que esperar lo que el
    proveedor dice evita tanto fallar como demorarse de mas.
    """
    retry_after = cabeceras.get("retry-after")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass

    reset = cabeceras.get("x-ratelimit-reset-tokens")
    if reset:
        try:
            return max(1.0, float(reset))
        except ValueError:
            pass

    return 0.0


def _uso_de_openai(usage: dict[str, Any] | None) -> UsoTokens:
    """Traduce el bloque `usage` de la API de OpenAI.

    Los tokens cacheados vienen en `prompt_tokens_details.cached_tokens`.
    Si no viene ese campo, se deja en None y `cache_disponible` queda en
    False: se registra "no informado", nunca cero.
    """
    if not usage:
        return UsoTokens()

    entrada = usage.get("prompt_tokens")
    salida = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if total is None and (entrada is not None or salida is not None):
        total = (entrada or 0) + (salida or 0)

    detalles = usage.get("prompt_tokens_details") or {}
    cacheados = detalles.get("cached_tokens")

    return UsoTokens(
        tokens_entrada=entrada,
        tokens_salida=salida,
        tokens_totales=total,
        tokens_cache_lectura=cacheados,
        # Solo es True si el proveedor mando el campo. Un 0 real es
        # distinto de "no lo informo".
        cache_disponible=cacheados is not None,
    )


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
class ProveedorOllama(LLMProvider):
    """Proveedor local de Ollama.

    Queda disponible para desarrollo. En el hardware medido solo entra un
    modelo de 1.5B, que no sostiene las reglas anti-alucinacion del
    enunciado (ver docs/bitacora.md 1.2.3), por eso no es el predeterminado.
    """

    nombre = "ollama"

    def __init__(self, modelo: str, base_url: str, timeout: float = 300.0) -> None:
        super().__init__(modelo)
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cliente: httpx.AsyncClient | None = None

    def _obtener_cliente(self) -> httpx.AsyncClient:
        if self._cliente is None:
            self._cliente = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self._cliente

    async def completar(
        self,
        mensajes: list[dict[str, str]],
        temperatura: float | None = None,
        tokens_maximos: int | None = None,
        esfuerzo_razonamiento: str | None = None,
    ) -> RespuestaLLM:
        ajustes = get_settings()
        cliente = self._obtener_cliente()

        inicio = time.perf_counter()
        try:
            respuesta = await cliente.post(
                "/api/chat",
                json={
                    "model": self.modelo,
                    "messages": mensajes,
                    "stream": False,
                    "options": {
                        "temperature": (
                            temperatura if temperatura is not None else ajustes.llm_temperature
                        ),
                        "num_predict": tokens_maximos or ajustes.llm_max_output_tokens,
                        "num_ctx": ajustes.ollama_context_length,
                    },
                },
            )
            respuesta.raise_for_status()
            datos = respuesta.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Ollama respondio {exc.response.status_code}: {redact(exc.response.text[:400])}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"No se pudo contactar a Ollama en {self.base_url}: {redact(str(exc))}"
            ) from exc

        latencia_ms = (time.perf_counter() - inicio) * 1000
        mensaje = datos.get("message") or {}

        entrada = datos.get("prompt_eval_count")
        salida = datos.get("eval_count")

        return RespuestaLLM(
            texto=mensaje.get("content") or "",
            modelo=self.modelo,
            proveedor=self.nombre,
            uso=UsoTokens(
                tokens_entrada=entrada,
                tokens_salida=salida,
                tokens_totales=(entrada or 0) + (salida or 0),
                # Ollama no expone metricas de cache de prompt. Se deja
                # None: reportar 0 seria simular una metrica inexistente.
                cache_disponible=False,
            ),
            latencia_ms=round(latencia_ms, 1),
            metadatos={"finish_reason": datos.get("done_reason")},
        )

    async def disponible(self) -> tuple[bool, str]:
        cliente = self._obtener_cliente()
        try:
            respuesta = await cliente.get("/api/version", timeout=5.0)
            respuesta.raise_for_status()
            return True, f"ollama {respuesta.json().get('version', '?')}"
        except httpx.HTTPError:
            return False, (f"Ollama no responde en {self.base_url}. Iniciarlo con `ollama serve`.")

    async def cerrar(self) -> None:
        if self._cliente is not None:
            await self._cliente.aclose()
            self._cliente = None


# ---------------------------------------------------------------------------
# Proveedor determinista para pruebas
# ---------------------------------------------------------------------------
class ProveedorFalso(LLMProvider):
    """Proveedor sin red, determinista y explicito sobre su artificialidad.

    Extrae del contexto las frases que mas se parecen a la pregunta. No es
    un modelo de lenguaje: es un recuperador extractivo. Sirve para que la
    suite sea rapida y reproducible, y sus respuestas quedan marcadas para
    que ningun resultado de prueba pueda confundirse con una respuesta real.
    """

    nombre = "fake"

    MARCA = "[respuesta sintetica de prueba]"

    def __init__(self, modelo: str = "fake-extractivo") -> None:
        super().__init__(modelo)

    async def completar(
        self,
        mensajes: list[dict[str, str]],
        temperatura: float | None = None,
        tokens_maximos: int | None = None,
        esfuerzo_razonamiento: str | None = None,
    ) -> RespuestaLLM:
        inicio = time.perf_counter()

        contexto = ""
        pregunta = ""
        for mensaje in mensajes:
            if mensaje["role"] != "user":
                continue
            contenido = mensaje["content"]
            # El mensaje de usuario lleva el contexto y la pregunta juntos,
            # separados por un salto: hay que extraer ambos.
            if "<contexto_documental" in contenido:
                contexto = _quitar_marcadores(contenido)
                # La pregunta viene despues del ultimo cierre de contexto.
                partes = contenido.rsplit("</contexto_documental>", 1)
                if len(partes) == 2 and ":" in partes[1]:
                    pregunta = partes[1].split(":", 1)[1].strip()
            else:
                pregunta = contenido

        if not contexto.strip():
            texto = f"{self.MARCA}\nNo tengo informacion suficiente en las fuentes consultadas."
        else:
            frases = _frases_relevantes(contexto, pregunta, limite=3)
            if not frases:
                texto = f"{self.MARCA}\nNo tengo informacion suficiente en las fuentes consultadas."
            else:
                texto = f"{self.MARCA}\n" + "\n".join(f"- {f}" for f in frases)

        latencia_ms = (time.perf_counter() - inicio) * 1000
        # El conteo es real (lo produjo este codigo), no una estimacion del
        # proveedor: por eso se reporta como tal.
        return RespuestaLLM(
            texto=texto,
            modelo=self.modelo,
            proveedor=self.nombre,
            uso=UsoTokens(
                tokens_entrada=estimar_tokens(contexto + pregunta),
                tokens_salida=estimar_tokens(texto),
                tokens_totales=estimar_tokens(contexto + pregunta + texto),
                cache_disponible=False,
            ),
            latencia_ms=round(latencia_ms, 2),
            metadatos={"sintetico": True},
        )


_RE_MARCADOR = re.compile(r"</?contexto_documental[^>]*>")
_CUERPO_INSTRUCCIONES = re.compile(
    r"El siguiente texto son documentos oficiales.*?tu tarea\.", re.DOTALL
)


def _quitar_marcadores(texto: str) -> str:
    """Deja solo el contenido documental, sin marcadores ni preguntas.

    El mensaje del usuario termina con "Pregunta del ciudadano: ...". Si
    no se recorta, esa linea entra en el contexto y el proveedor la
    devuelve como si fuera una frase extraida de la fuente.
    """
    # Se corta en el ultimo cierre de contexto: lo que viene despues es
    # la pregunta, no material documental.
    if "</contexto_documental>" in texto:
        ultimo = texto.rfind("</contexto_documental>")
        texto = texto[: ultimo + len("</contexto_documental>")]

    limpio = _CUERPO_INSTRUCCIONES.sub("", texto)
    return _RE_MARCADOR.sub("", limpio).strip()


def _frases_relevantes(contexto: str, pregunta: str, limite: int = 3) -> list[str]:
    """Selecciona por solapamiento de palabras, de forma determinista."""
    terminos = {t for t in re.findall(r"[a-záéíóúñü]{4,}", pregunta.lower()) if t not in _VOCALES}
    if not terminos:
        return []

    puntuadas: list[tuple[int, int, str]] = []
    for indice, frase in enumerate(_frases(contexto)):
        palabras = set(re.findall(r"[a-záéíóúñü]{4,}", frase.lower()))
        comunes = terminos & palabras
        if comunes:
            # El indice desempata para que el resultado sea estable.
            puntuadas.append((len(comunes), -indice, frase.strip()))

    puntuadas.sort(reverse=True)
    return [f for _, _, f in puntuadas[:limite] if len(f) > 20]


def _frases(texto: str) -> list[str]:
    partes = re.split(r"(?<=[.!?])\s+|\n+", texto)
    return [p.strip() for p in partes if len(p.strip()) > 20]


_VOCALES = frozenset(
    {"este", "esta", "esto", "para", "como", "donde", "cuando", "cual", "tengo", "puedo"}
)


# ---------------------------------------------------------------------------
# Fabrica
# ---------------------------------------------------------------------------
def crear_proveedor(ajustes: Settings | None = None) -> LLMProvider:
    """Construye el proveedor indicado por la configuracion."""
    config = ajustes or get_settings()

    match config.llm_provider:
        case "fake":
            return ProveedorFalso()
        case "ollama":
            return ProveedorOllama(
                modelo=config.ollama_model,
                base_url=config.ollama_base_url,
                timeout=config.llm_request_timeout_s * 3,
            )
        case "openai_compatible":
            return ProveedorOpenAICompatible(
                modelo=config.openai_model,
                base_url=config.openai_base_url,
                api_key=config.openai_api_key,
                timeout=config.llm_request_timeout_s,
                max_reintentos=config.llm_max_retries,
            )
        case _:  # pragma: no cover - la validacion de config lo impide
            raise LLMError(f"Proveedor desconocido: {config.llm_provider}")


_PROVEEDOR: LLMProvider | None = None


def obtener_proveedor() -> LLMProvider:
    """Proveedor compartido, creado una sola vez por proceso.

    Se reutiliza para aprovechar las conexiones persistentes: abrir un
    cliente nuevo por peticion haria que la medicion de latencia midiera
    tambien el handshake TCP, que es ruido que no interesa comparar.
    """
    global _PROVEEDOR
    if _PROVEEDOR is None:
        _PROVEEDOR = crear_proveedor()
        logger.info("Proveedor LLM inicializado: %s / %s", _PROVEEDOR.nombre, _PROVEEDOR.modelo)
    return _PROVEEDOR


def reiniciar_proveedor() -> None:
    """Descarta el proveedor cacheado. Lo usan las pruebas."""
    global _PROVEEDOR
    _PROVEEDOR = None


async def comprobar_proveedor() -> dict[str, Any]:
    """Diagnostico del proveedor: util para /health y para la consola."""
    proveedor = obtener_proveedor()
    ok, detalle = await proveedor.disponible()
    return {
        "proveedor": proveedor.nombre,
        "modelo": proveedor.modelo,
        "disponible": ok,
        "detalle": redact(detalle),
        # Se consulta de verdad y no se supone: el agente lo necesita para
        # saber si puede registrar la metrica de cache.
        "soporta_cache": await proveedor.supports_prompt_caching(),
    }


def construir_json_de_ejemplo(esquema: dict[str, Any]) -> str:
    """Serializa un esquema paraFew-shot. Util para pedir JSON estructurado."""
    return json.dumps(esquema, ensure_ascii=False, indent=2)


async def esperar_jitter() -> None:
    """Espera aleatoria breve, para no golpear el proveedor en rafaga."""
    await asyncio.sleep(asyncio.random.uniform(0.0, 0.3))
