# TramitaIA — CAG vs. RAG para trámites públicos de La Rioja

Agente conversacional que responde consultas sobre trámites públicos de la
provincia de La Rioja (Argentina) usando **únicamente fuentes oficiales**.

El objetivo del proyecto no es el chatbot: es **comparar dos estrategias de
generación aumentada** con el mismo modelo, las mismas preguntas y las mismas
fuentes.

| | |
|---|---|
| **CAG** | Cache-Augmented Generation — contexto documental preparado una vez y reutilizado en cada consulta |
| **RAG** | Retrieval-Augmented Generation — recupera fragmentos relevantes para cada consulta |

---

## Estado del proyecto

**Fases 1 a 4 completadas y verificadas.** 178 pruebas pasan, lint limpio.

| Fase | Qué hace | Estado |
|------|----------|--------|
| 1 | Arquitectura, configuración validada, logging seguro | Completada |
| 2 | Ingesta documental, manifiesto de fuentes, corpus real | Completada |
| 3 | Proveedor de LLM intercambiable, agente con verificación local | Completada |
| 4 | CAG: contexto preparado, caché e invalidación por hash | Completada |
| 5 | RAG sobre el mismo corpus: troceado, indizado y verificado | Completada |
| 6 | API FastAPI, endpoints de comparacion y pruebas | Completada |
| 7 | Interfaz web con comparacion CAG/RAG lado a lado | Completada |
| 8 | Dataset de 50 preguntas y experimento | En curso |
| 9 | Documentacion final y guia de exposicion | Pendiente |

El detalle completo de cada fase, con los problemas encontrados y las
decisiones tomadas, está en **[`docs/bitacora.md`](docs/bitacora.md)**.

---

## Qué funciona hoy

Verificado con ejecuciones reales, no estimado:

- **Corpus real**: 6 documentos oficiales del Registro Civil de La Rioja,
  descargados respetando `robots.txt`, con metadatos trazables.
- **Abstención correcta**: consultas sobre información ausente del corpus
  (plazos, horarios) devuelven *"No tengo información suficiente en las
  fuentes consultadas"* en vez de inventar.
- **Citas verificadas localmente**: el sistema decide qué documentos
  respaldan la respuesta comparando el texto generado contra el corpus. No
  se le pide al modelo que cite.
- **Caché de CAG medida**: 14,2% de acierto en la primera consulta y 73,2%
  en la segunda, con prefijo byte a byte idéntico.
- **Detección de affirmationes inventadas**: los montos, plazos y
  referencias normativas se buscan en la fuente y se reportan como
  advertencia si no aparecen.

---

## Requisitos

- **Python 3.12** (obligatorio). El 3.13 y el 3.15 todavía no tienen
  wheels de `torch` ni de `faiss-cpu`, así que el proyecto no instala ahí.
- Windows, Linux o macOS.
- Una clave de un proveedor de LLM compatible con la API de OpenAI.
  El proyecto usa el plan gratuito de [Groq](https://console.groq.com/keys),
  que además reporta los tokens cacheados.

## Instalación

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuración

```bash
Copy-Item .env.example .env     # Windows
cp .env.example .env            # Linux / macOS
```

Editar `.env` y completar `OPENAI_API_KEY`. **Nunca versionar `.env`**: está
en `.gitignore` y este repositorio solo contiene `.env.example`.

```bash
# Descargar las fuentes oficiales y construir el corpus
python scripts/fetch_sources.py --listar   # ver qué se va a bajar
python scripts/fetch_sources.py            # descargar e ingerir
```

## Uso por consola

```bash
# Probar el agente contra el proveedor real
python scripts/proveedor_probe.py      # abstención, inyección, fundamentación
python scripts/probar_agente.py        # consultas completas

# Probar CAG
python scripts/probar_cag.py                       # sin llamar al modelo
python scripts/probar_cag.py --consultas --ver-respuesta -p "¿Cuánto cuesta la partida?"

# Medir si el prefijo enviado al proveedor es realmente idéntico
python scripts/diagnostico_cache.py
```

## Demos del seminario

Tres scripts que sostienen la conclusion. Los dos primeros **no gastan un
solo token**: la recuperacion es local.

```bash
# La palabra que rompe la recuperacion de RAG
python scripts/demo_recuperacion.py

# El modelo no era el problema: mismo prompt, +679 caracteres
python scripts/experimento_modelo.py

# Costo real por consulta con los precios publicados de Groq
python scripts/costo_consulta.py
```

## Experimento

```bash
# Las 50 preguntas con las dos estrategias
python scripts/correr_experimento.py

# Prueba rapida
python scripts/correr_experimento.py --limite 5
```

El plan gratuito del proveedor da 8.000 tokens por minuto y cada consulta
de CAG manda casi 4.000, asi que la corrida completa tarda alrededor de
media hora. El informe queda en `results/informe-*.json`.

## API

```bash
uvicorn app.main:app --reload
```

| Endpoint | Qué hace |
|----------|----------|
| `GET /health` | Estado de la aplicación |

Los endpoints de chat y de evaluación llegan en la fase 6.

---

## Estructura

```
app/
  core/         config.py (validación cruzada), logging.py (redacción de credenciales)
  models/       schemas.py — vocabulario documental y contratos
  services/
    ingestion.py   extracción PDF/HTML/TXT/DOCX, normalización, hash
    corpus.py      identidad estable de documentos, manifiesto, duplicados
    llm.py         interfaz LLMProvider + openai_compatible / ollama / fake
    citations.py   verificación de montos, normativa, plazos y citas
    agent.py       orquestación y reglas de seguridad
    cag.py         contexto preparado, caché e invalidación por hash
  repositories/
    document_repository.py   corpus y manifiesto
    evaluation_repository.py SQLite: consultas, ejecuciones, resultados

scripts/       utilidades reproducibles (fetch, probes, benchmarks)
tests/         178 pruebas
docs/          bitacora.md — el registro completo del trabajo
data/          corpus generado y manifiesto de fuentes
```

---

## Decisiones que conviene conocer

**Las citas las decide el sistema, no el modelo.** Pedirle al modelo que
cite garantiza que cite documentos que no usó, porque el mismo criterio
que genera la cita la validaría. Acá la verificación es local: se compara
la respuesta contra el corpus que se le entregó.

**Un fallo técnico no es una abstención.** Si el proveedor falla, la
respuesta lo dice. Colapsar ambos casos haría que un error de red pareciera
evidencia ausente.

**La vigencia tiene tres estados.** `DESCONOCIDA` cuando la fuente no la
declara, `PARCIAL` cuando tiene más de un año, `VIGENTE` solo cuando la
fuente declara fecha reciente. Nunca se infiere vigencia por estar publicado.

**El orden del contexto no es un detalle.** En CAG, si el prefijo varía
entre consultas, el proveedor no puede cachearlo y la estrategia pierde su
ventaja. Por eso los documentos se ordenan de forma determinista.

**Dos cachés, dos métricas.** La caché de aplicación es nuestra y se mide
en milisegundos. La caché de cómputo del proveedor se mide con los tokens
que él reporta. Cuando no las reporta, se deja en `None`, nunca en cero.

### Restricciones del entorno en Windows

`scipy==1.14.1` está fijado a propósito. Versiones posteriores publican un
`_sobol.pyd` que la política de Control de Aplicaciones de Windows bloquea,
y como `scipy` arrastra `scikit-learn` y `sentence-transformers`, el fallo
tumbaba el pipeline completo de embeddings. El motivo está documentado en
`requirements.txt` y en la bitácora.

---

## Fuentes

Todas de sitios oficiales del Gobierno de La Rioja. `argentina.gob.ar/tramitar`
está **bloqueada por su propio `robots.txt`** y se incorpora manualmente
cuando haga falta, conservando su URL de origen.

## Autor

Nelson Fernández ([@nelsonfern](https://github.com/nelsonfern))