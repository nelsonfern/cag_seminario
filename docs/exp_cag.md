Listed directory cag_seminario
Listed directory app
Viewed agent.py:1-100
Viewed agent.py:101-250
Searched for "caching"
Listed directory services
Viewed cag.py:1-120
Viewed cag.py:121-300
Viewed cag.py:301-448
Viewed agent.py:250-372
Searched for "CAGService"
Searched for "obtener_servicio_cag"
Listed directory api
Viewed main.py:1-91
Listed directory scripts
Listed directory docs
Viewed probar_cag.py:1-100
Viewed probar_cag.py:101-186
Viewed diagnostico_cache.py:1-76
Viewed document_repository.py:1-100
Searched for "fase 4"
Viewed bitacora.md:554-640
Listed directory data
Listed directory processed
Listed directory cag_cache
Viewed contexto.json:1-60

En este proyecto, **CAG (Cache-Augmented Generation)** es la estrategia en la cual **no se realiza una búsqueda ni recuperación dinámica de fragmentos por similitud** (como hace RAG). En su lugar, se precarga y estructura el **corpus documental completo** dentro de la ventana de contexto del modelo, manteniéndolo idéntico entre consultas para aprovechar el almacenamiento en caché.

---

### 1. El concepto clave: Las dos cachés del proyecto

El diseño distingue estrictamente dos niveles de caché definidos en [cag.py](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/app/services/cag.py#L8-L19):

1. **Caché de Aplicación (local)**: Es el archivo [contexto.json](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/data/processed/cag_cache/contexto.json) y la variable en memoria RAM. Es controlada al 100% por el sistema para no tener que reconstruir ni leer el corpus desde cero en cada consulta (pasa de ~4.5 ms a ~0.001 ms).
2. **Caché de Cómputo del Proveedor (KV Prompt Cache)**: La que ofrece el proveedor del LLM (ej. Groq / OpenAI / Anthropic). Al recibir un prefijo de tokens idéntico y determinista entre múltiples llamadas, el proveedor reutiliza los estados *Key-Value* calculados y solo cobra/procesa los tokens de la pregunta nueva (`tokens_cache_lectura`).

---

### 2. ¿Cómo usa el Agente el CAG en una consulta?

El flujo de ejecución del agente está centralizado en [Agente.responder](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/app/services/agent.py#L178-L210) y sigue estos pasos:

```
Pregunta del usuario
        │
        ▼
CAGService.preparar() ──► (Lee de RAM / contexto.json local)
        │
        ▼
Construcción del Prompt:
  [REGLAS_SISTEMA] + [Contexto Documental Fijo] + [Pregunta del Ciudadano]
        │
        ▼
Llamada al LLM (Groq / Provider)
  └── El LLM detecta el prefijo ya visto y lee de su Prompt Cache
        │
        ▼
Verificación local de citas / Grounding (citations.py)
  └── El sistema valida qué usó realmente el modelo contra los documentos
        │
        ▼
Respuesta final con fuentes verificadas y métricas de caché
```

---

### 3. ¿Cómo se carga todo a la caché? (Paso a paso)

El proceso de preparación e ingesta hacia la caché lo realiza el método [`CAGService.preparar()`](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/app/services/cag.py#L143-L250):

#### Paso 1: Ingesta del corpus base
Los documentos originales (`.html` / `.pdf` en `data/raw/`) son procesados y normalizados por [ingestion.py](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/app/services/ingestion.py) y almacenados en [DocumentoRepository](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/app/repositories/document_repository.py) dentro del archivo `data/processed/corpus.json`.

#### Paso 2: Cálculo de huella digital (`hash_corpus`)
En [`CAGService.hash_corpus()`](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/app/services/cag.py#L120-L138), el sistema genera un hash SHA-256 combinando la versión del formato y el `document_id:hash_contenido` de cada documento activo. 
* Si un documento cambia, se agrega o se borra, el hash cambia automáticamente.

#### Paso 3: Verificación de caché existente
Cuando se solicita el contexto:
1. **Memoria RAM (`self._en_memoria`)**: Si ya está en memoria y el hash del corpus coincide, devuelve una copia inmediata.
2. **Disco local (`_leer_cache`)**: Si no está en RAM, busca [contexto.json](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/data/processed/cag_cache/contexto.json). Si el hash coincide, lo carga a memoria.

#### Paso 4: Construcción en frío (si no hay caché o cambió el corpus)
Si la caché no existe o quedó invalidada (`force=True` o hash no coincide):
1. **Carga y filtrado**: Carga las entradas desde [DocumentoRepository.cargar_entradas](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/app/repositories/document_repository.py) y descarta documentos marcados como `no_accesible`.
2. **Ordenamiento determinista obligatorio**:
   ```python
   documentos.sort(key=lambda d: d.metadatos.document_id)
   ```
   > **¿Por qué?** Si el orden de los documentos variara entre ejecuciones, el prefijo de texto cambiaría byte a byte y la caché de cómputo del LLM **fallaría por completo**.
3. **Control del presupuesto de tokens**: Con [`_seleccionar_por_presupuesto()`](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/app/services/cag.py#L251-L273), verifica contra el límite configurado (`cag_max_context_tokens`, por defecto 6.000 tokens). Si un documento excede el presupuesto, se trunca limpiamente por documento entero para no generar citas rotas.
4. **Formateo y defensas de seguridad**: A cada documento se le añade su cabecera (Título, Organismo, Trámite, Vigencia) y se envuelve con etiquetas:
   ```xml
   <contexto_documental no_confiable>
   ... contenido oficial ...
   </contexto_documental>
   ```
   Esto previene ataques de inyección de prompts contenidos en las webs oficiales.

#### Paso 5: Persistencia
El contexto finalizado se guarda mediante [`_escribir_cache()`](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/app/services/cag.py#L325-L355) en [contexto.json](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/data/processed/cag_cache/contexto.json) conteniendo:
- El hash del corpus y timestamp de creación.
- El texto completo ya concatenado listo para enviar al prompt.
- La lista deserializada de documentos (para que el validador de citas local pueda auditar la respuesta del modelo).
- Métricas del contexto (tokens estimados, documentos incluidos/omitidos).

#### Paso 6: Calentamiento en el proveedor de LLM
La primera consulta que se envía al LLM envía este prefijo largo. El backend del proveedor procesa los ~3.700 tokens de entrada y guarda sus estados KV en caché. Las siguientes consultas que reutilicen ese mismo prefijo registrarán aciertos de lectura (`tokens_cache_lectura`), logrando el ahorro de costo y tiempo propio de CAG.

Para comprobar y experimentar con este ciclo directamente por terminal, el proyecto cuenta con los scripts [probar_cag.py](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/scripts/probar_cag.py) y [diagnostico_cache.py](file:///c:/Users/Usuario/Desktop/seminario/cag_seminario/scripts/diagnostico_cache.py).

Viewed probar_cag.py:165-186