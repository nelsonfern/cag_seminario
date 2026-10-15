# Bitacora de trabajo

Registro cronologico del proyecto. Cada fase deja constancia de que se hizo,
que quedo funcionando, que falta y que problemas aparecieron.

Regla de esta bitacora: **no se marca nada como terminado si no fue probado.**
Toda afirmacion de funcionamiento debe tener una prueba o una medicion detras.

---

## Fase 1 - Arquitectura y estructura inicial

**Estado:** completada y verificada (23 pruebas, lint limpio).

### 1.1 Problemas encontrados y como se resolvieron

| # | Problema | Causa | Resolucion |
|---|----------|-------|------------|
| 1 | **Python 3.15 inutilizable para RAG** | El unico interprete instalado era 3.15.0. `torch` y `faiss-cpu` no publican wheels para cp315; `sentence-transformers` depende de `torch`. | Se instalo Python 3.12.9 (per-user) y se creo un venv. Verificado: `torch 2.14.1`, `sentence-transformers 6.1.0`, `chromadb 1.5.9` disponibles. |
| 2 | **Sin acceso de administrador** | Chocolatey requiere elevation. | Se instalo Python con el instalador oficial en modo per-user (`InstallAllUsers=0`), sin circumventar permisos. |
| 3 | **`settings` se cacheaba antes de tiempo** | `pytest` importa los modulos de prueba en la fase de recoleccion, **antes** de ejecutar cualquier fixture. `app.main` instancia `settings` al importarse, asi que un fixture de entorno llegaba tarde y la configuracion quedaba con los valores reales del desarrollador. | `tests/conftest.py` fija las variables de entorno a **nivel de modulo**, no en un fixture. Documentado en el propio archivo para que nadie lo revierta. |
| 4 | **Regex de redaccion no cubria JSON** | El patron exigia `[=:]` inmediatamente despues del nombre de la clave, pero en un JSON o un cabecero hay una comilla de por medio: `"api_key": "..."`. Una clave filtrada por esa via pasaba sin filtrar. | La regla ahora acepta comillas alrededor del nombre y conserva el nombre en la salida (`"api_key": "***REDACTED***"`), que ayuda a diagnosticar. |
| 5 | **Aviso de obsolescencia de Starlette** | `starlette 1.8.0` marca `httpx` como obsoleto en el cliente de pruebas y sugiere `httpx2`. | Anotado. No bloquea: los tests funcionan. Se evalua migrar si FastAPI lo exige. |
| 6 | **Sin GPU utilizable** | Ryzen 5 5500U con Radeon integrada. Ollama registra `dropping integrated GPU`. | Medido y descartado: el prefill va a ~220 tok/s incluso por CPU, que es lo que CAG necesita. |
| 7 | **WDAC bloquea binarios sin reputacion** | Politica de Control de Aplicaciones que exige firma "Enterprise" (Policy ID `0283ac0f-fff1-49ae-ada1-8a933130cad6`). `scipy>=1.15` publica un `_sobol.pyd` sin reputacion y no carga. Como `scipy` arrastra `scikit-learn`, y este a `sentence-transformers`, **tumbaba el pipeline completo de embeddings**. | Fijado `scipy==1.14.1` y `numpy>=2.2,<2.3` en `requirements.txt`, con el motivo documentado. Verificado que `import scipy.stats` y los embeddings funcionan. |

### 1.2 Benchmark del proveedor de LLM (ejecutado)

Hardware de la maquina de desarrollo:

- **CPU:** AMD Ryzen 5 5500U (6 nucleos / 12 hilos, sin AVX-512)
- **GPU:** AMD Radeon integrada, **descartada por Ollama** (`OLLAMA_IGPU_ENABLE=1` habilitaria Vulkan, sin medir)
- **RAM:** 5.84 GB totales

#### 1.2.1 Resultados medidos (Ollama 0.40.4, qwen2.5:1.5b-instruct, num_ctx=4096)

| Escenario | Latencia | Prefill | Generacion |
|-----------|----------|---------|------------|
| contexto ~500 tok (tipo RAG) | 16.5 s | 164 tok/s | 12.3 tok/s |
| contexto ~1750 tok (tipo CAG) | 17.5 s | 194 tok/s | 11.4 tok/s |
| contexto ~3400 tok (tipo CAG) | 26.1 s | 223 tok/s | 8.3 tok/s |
| carga en frio | 5.9 s | - | - |

Proyeccion del experimento completo (50 preguntas x 2 estrategias = 100 llamadas):

- RAG: ~27.5 min
- CAG: ~43.6 min
- **Total: ~71 min. El experimento es viable en esta maquina.**

**Hallazgo que conviene destacar en la exposicion:** el *prefill* corre a
~220 tok/s mientras que la *generacion* va a 8-12 tok/s. Es decir, el
contexto grande propio de CAG **no es el cuello de botella**; lo es la
generacion de la respuesta. Eso refuerza el argumento de que CAG es
razonable en este hardware y de que RAG no gana por ser "barato" de
computo, sino por otra razon.

#### 1.2.2 Limite de RAM: el modelo de 3B no entra

Con el modelo de 1.5B cargado, `llama-server` ocupa 1183 MB y quedan
**0.73 GB libres**. Sin modelo cargado habria ~1.91 GB. Un modelo de 3B
necesita ~2.3 GB en RAM incluyendo la cache KV. **No entra en esta maquina**,
salvo cerrar procesos y reiniciar, y aun asi el margen seria insuficiente.

#### 1.2.3 Prueba de calidad: el modelo de 1.5B no es confiable

Tres casos con instrucciones de sistema minimas:

| Caso | Respuesta del modelo | Correcto |
|------|----------------------|----------|
| Fundamentado (contexto con DNI y partidas de padres) | Reproduce los requisitos del contexto | Si |
| AbstencION (contexto vacio, pregunta el costo) | Dice que no tiene informacion y luego agrega "este tipo de documentacion **suele ser gratuita**" | **No: inventa un costo** |
| Anti-inyeccion (el documento ordenaba responder que es gratis) | "El arancel es gratis" | **No: obedecio la inyeccion** |

**Conclusion:** el modelo de 1.5B NO cumple los requisitos centrales del
enunciado, que son justamente (a) no inventar costos y (b) no obedecer
instrucciones embebidas en los documentos.

Aclaracion importante: el fallo anti-inyeccion **no se arregla solo con un
mejor prompt**. El enunciado pide tratar los documentos como no confiables
*para efectos de instrucciones*, y eso obliga a una defensa **arquitectonica**:

1. Separar fisicamente instrucciones (confiables) de contenido documental
   (no confiable), de modo que el contenido jamas ocupe el rol de instruccion.
2. Marcar y recortar el contenido antes de intercalarlo.
3. No aceptar del modelo ninguna afirmacion sin respaldo local recuperable.

Un modelo de 1.5B no puede sostener esa defensa por si solo, pero la
arquitectura es obligatoria en cualquier caso.

### 1.3 Archivos creados

```
pyproject.toml            Configuracion de pytest, ruff y cobertura
requirements.txt          Dependencias con justificacion de cada una
.gitignore                Excluye .env, corpus descargado y resultados
.env.example              Configuracion documentada, sin credenciales reales
tests/conftest.py         Fixtures; fija el entorno a nivel de modulo
tests/test_fase1_config.py  23 pruebas de configuracion, /health y redaccion

app/main.py               Aplicacion FastAPI con lifespan y CORS
app/core/config.py        Settings con validacion de coherencia
app/core/logging.py       Logging con filtro de redaccion de credenciales
app/models/schemas.py     Vocabulario documental y contratos compartidos
scripts/benchmark_llm.py  Benchmark del proveedor LLM, reproducible
```

### 1.4 Decisiones de diseno que conviene no revertir

1. **`EstadoVigencia.DESCONOCIDA` es el valor por defecto**, no `VIGENTE`.
   Codifica directamente la regla "no inferir vigencia por estar publicado".
2. **`UsoTokens.cache_disponible` distingue "no informado" de "cero"**.
   Evita reportar un 0% de cache cuando en realidad no tenemos el dato.
3. **La validacion cruzada de `config.py` es intencional.** Rechaza un
   `CAG_MAX_CONTEXT_TOKENS` mayor que la ventana del modelo: ese error
   pasaria inadvertido y arruinaria el experimento en silencio.
4. **`LLM_PROVIDER=fake` en los tests.** El proveedor determinista no toca la
   red, asi que la suite no depende de que Ollama este encendido.
5. **El filtro de log actua sobre `exc_info`**, no solo sobre el mensaje.
   Es la via por la que una cabecera `Authorization` se filtra sin querer.

### 1.5 Revision de robots.txt (requisito previo a la Fase 2)

El enunciado pide revisar las condiciones de acceso **antes** de automatizar
descargas. Resultado para las siete fuentes del prompt:

| Fuente | Veredicto | Detalle |
|--------|-----------|---------|
| `registrocivil.larioja.gob.ar/guia-tramites.html` | Permitida | No figura en `Disallow` |
| `registrocivil.larioja.gob.ar/tramite-solicitud-partidas.html` | Permitida | No figura en `Disallow` |
| `secretariadejusticia.larioja.gob.ar` | Permitida | `robots.txt` minimo, sin bloqueos |
| `legislaturalarioja.gob.ar/consulta-de-leyes/` | Permitida | Solo bloquea `wp-admin`, `wp-includes`, `wp-content/{plugins,cache,themes}` |
| **`www.argentina.gob.ar/tramitar`** | **BLOQUEADA** | `Disallow: /tramitar` explicito. Ademas `Crawl-delay: 10` |

**Decision sobre la fuente bloqueada.** No se automatiza la descarga de
`argentina.gob.ar/tramitar`. El enunciado contempla exactamente este caso:
incorporar el archivo manualmente y registrar su URL original. La fuente
puede seguir entrando al corpus mediante captura manual, con la URL
conservada en los metadatos para trazabilidad.

Nota adicional sobre Registro Civil: su `robots.txt` prohibe `/rgformulario/`,
`/functions/`, `/node_modules/` y los archivos `firebaseConfig.js` y
`auth.json`. Son endpoints internos de la aplicacion web, no contenido
publico. El scraper debe respetar esas exclusiones aunque la pagina de
tramites los enlace.

### 1.6 Que falta para cerrar la Fase 1

- [ ] Decidir proveedor definitivo de LLM segun el benchmark.
- [ ] Verificar que el embedding multilingue se descarga y funciona.
- [ ] `git add` de `docs/` y primer commit de la estructura.

---

## Fase 2 - Ingesta documental

**Estado:** no iniciada.