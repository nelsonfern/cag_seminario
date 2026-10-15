# Guia de exposicion

**Números reales de la corrida ejecutada:** 23 de las 50 preguntas del
dataset. Tres de las cinco categorías. Todo lo de abajo sale de
`results/informe-parcial.json`, generado por
`scripts/informe_desde_log.py`.

Que mostrar en vivo, que decir, y que tener listo por si algo falla.

**Antes de empezar:** las demos que dependen del proveedor arrancan
lento. Lanzalas con antelacion. Un minuto de "cargando" en vivo se siente
como un fallo aunque no lo sea.

---

## 1. La apertura (30 segundos)

No empieces mostrando la interfaz. Empieza con la pregunta que organiza
todo:

> "Tengo seis documentos oficiales de un registro civil. Los puedo
> mandar enteros al modelo, o puedo buscar en ellos lo relevante y
> mandar solo eso. Es CAG y RAG. Los comparo con el mismo modelo, las
> mismas preguntas y el mismo corpus, y la pregunta no es cual es mas
> rapido: es cual deja de responder cosas que si estan en las fuentes."

Y una advertencia que es parte del trabajo:

> "El corpus es real y sale de sitios oficiales. Una de las paginas tiene
> el costo del matrimonio escrito como `$??`. No lo complete. Si lo
> complete, estaria inventando."

---

## 2. El resultado, y su limite (1 minuto)

Decirlo de entrada, no al final:

> "Corrieron 23 de las 50 preguntas, no las 50. El plan gratuito del
> proveedor da 8.000 tokens por minuto y en una de las consultas me
> pidió esperar 862 segundos. Después de 23 decidí no seguir esperando.
> El dataset tiene las 50, verificadas una por una contra las fuentes;
> lo que no corrió es la ejecución."

| | Acierto | Latencia p50 | Tokens de entrada |
|---|---|---|---|
| **CAG** | **21/23 — 91,3%** | 1.547 ms | 3.951 |
| **RAG** | 13/23 — 56,5% | 1.406 ms | 1.493 |

Por categoría:

| Categoría | CAG | RAG |
|---|---|---|
| Requisitos | 8/10 | 6/10 |
| Procedimiento | 10/10 | 7/10 |
| Multi-sección | **3/3** | **0/3** |

**Lo que se dice:**

> "La latencia es empate: 1.547 contra 1.406 milisegundos. RAG usa menos
> de la mitad de los tokens. Pero RAG acerta 56% y CAG 91%. Y donde la
> diferencia se abre es en multi-sección: RAG no acierta ninguna."

---

## 3. Demo en vivo: la palabra que rompe todo (3 minutos)

Esta es la demo principal. **No gasta un solo token** porque la
recuperación es local.

```powershell
.venv\Scripts\python.exe scripts\demo_recuperacion.py
```

```
PREGUNTA: ¿Qué documentación necesito para una partida de nacimiento?
   con la palabra 'partida'
 4. 0.652  dentro del corte    Requisitos del trámite
 6. 0.594  FUERA del corte     Inscripción de Nacimientos
  >> 'documentación requerida' salió en el puesto 8: NO LLEGA al corte

PREGUNTA: ¿Qué documentación necesito para inscribir un nacimiento?
   con la palabra 'inscribir'
 5. 0.627  dentro del corte    Documentación requerida   <<< ESTA ES LA RESPUESTA
  >> 'documentación requerida' salió en el puesto 5: LLEGA
```

**Lo que se dice:**

> "El fragmento que tiene la respuesta existe, está bien cortado y está
> indexado. Lo único que cambió fue una palabra. Con 'partida' queda en
> el puesto 8 y no llega al corte de cinco. Con 'inscribir' queda en el 5
> y llega. Una palabra decide si el ciudadano recibe una respuesta o un
> 'no tengo información'."

Y el vínculo con la tabla: **multi-sección es donde RAG cae a cero**,
porque la respuesta requiere combinar dos secciones del mismo documento y
el recuperador no trae las dos.

---

## 4. Demo en vivo: el modelo no era el problema (2 minutos)

```powershell
.venv\Scripts\python.exe scripts\experimento_modelo.py
```

| Lo que recibió el modelo | Respuesta |
|---|---|
| Sin el fragmento — lo que recibió RAG | *"No tengo información suficiente..."* |
| Con el fragmento | Los tres requisitos, correctos |

**Lo que se dice:**

> "Mismo modelo, mismo prompt, misma pregunta. Lo único que cambió fueron
> 679 caracteres con el fragmento que faltaba. El modelo podía responder.
> Lo que falló fue la recuperación, y eso no depende del modelo: si el
> fragmento no está en el prompt, ninguna respuesta correcta es posible."

**Si preguntan si es culpa del modelo:** no, y esto es la prueba. Donde el
modelo sí importa es en que se abstenga antes que adivinar. Si
confabulara, RAG ganaría esas consultas con respuestas inventadas y nadie
vería que el recuperador falló.

---

## 5. Demo en vivo: la comparación lado a lado (2 minutos)

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app
```

Abrir `http://localhost:8000`. Arranca en modo **comparar**.

Preguntar algo que esté en las dos, por ejemplo:

> ¿Cuánto cuesta la copia de acta certificada?

> ¿Qué documentación necesito para una partida de nacimiento?

La segunda es la de la demo 3: se ve cómo RAG se abstiene y CAG contesta,
en vivo y en la misma pantalla.

**Advertencia de ritmo:** el proveedor da 8.000 tokens por minuto y cada
consulta de CAG manda casi 4.000. Las esperas de 20 a 30 segundos son
normales y la interfaz lo avisa. Decirlo de antemano evita que se lea como
un fallo.

---

## 6. El costo (1 minuto)

```powershell
.venv\Scripts\python.exe scripts/costo_consulta.py
```

**Lo que se dice:**

> "Con los precios publicados de Groq la diferencia es de un centavo cada
> cien consultas. Con seis documentos ninguna de las dos es cara. Y CAG
> sólo gana en costo cuando la caché funciona, que el propio proveedor
> declara 'best effort': en este proyecto el mismo prefijo dio entre 0% y
> 98% de acierto entre corridas. Una tesis económica apoyada ahí se cae
> sola."

---

## 7. La tabla que sostiene la conclusión

```
                 Acierto    Latencia   Tokens   Corpus completo
CAG              21/23      1.547 ms   3.951    siempre
RAG              13/23      1.406 ms   1.493    nunca (se trunca en 9 docs)
Corte de CAG: 8 documentos. El noveno lo rompe.
```

**Lo que se dice:**

> "CAG no es más rápido ni más barato. La justificación es una sola: el
> corpus entra en la ventana de contexto, y a partir de nueve documentos
> deja de entrar. El costo de CAG no crece de forma gradual, tiene un
> borde."

Y el otro lado, que no hay que esconder:

> "RAG acertó dos consultas donde CAG falló: R05 y R10. No es un
> resultado de una sola cara."

---

## 8. Lo que esta corrida NO demuestra

Esto hay que decirlo antes de que lo pregunten.

1. **No dice nada sobre la honestidad del sistema.** Las diez preguntas
   de *fuera de corpus* —las que miden si el sistema se abstiene en vez
   de inventar— **no se ejecutaron**. Sin ellas no hay metric de
   abstención. La categoría es parte del dataset y queda pendiente de
   correr.

2. **No cubre preguntas ambiguas.** Diez preguntas más sin ejecutar.

3. **23 preguntas no son una muestra estadisticamente solida.** Es una
   diferencia grande —91% contra 56%—, pero con 23 casos el margen de
   error es amplio.

4. **La comparación favorece a CAG por construcción.** CAG manda los
   3.730 tokens del corpus; RAG manda 1.493. Es la definición de cada
   técnica, no un sesgo del experimento. Pero conviene decirlo: no se
   esta comparando "con mas contexto" contra "con menos" por casualidad.

5. **La latencia no incluye la espera del proveedor.** Una de las 23
   consultas de CAG tardó 891 segundos esperando que el proveedor liberara
   el cupo. Queda fuera de las cifras por eso. Si se incluyera, CAG
   parecería muchísimo peor, y eso tampoco sería justo: sería medir el
   plan gratuito, no la estrategia.

6. **No se probaron otras preguntas del mismo corpus.** Con otro recorte
   de cinco preguntas, las cifras podrían moverse.

---

## 9. Preguntas que seguro van a hacer

**"¿Por qué no usaste un modelo local?"**
La máquina tiene 5,8 GB de memoria y no entra uno que pase las pruebas
antihallucinación. El de 1,5 B fallaba 5 de 5; el remoto, 2 de 3.

**"RAG es lo que usa todo el mundo, ¿no?"**
Sí, y está bien para corpus grandes y cambiantes. El proyecto no dice que
RAG esté mal: dice que RAG tiene un modo de falla silencioso, y que con
seis documentos no compensa pagarlo.

**"¿Y si el recuperador fuera mejor?"**
El argumento se debilita, y hay que decirlo así. El hallazgo es sobre
este recuperador, con este corte, sobre este corpus. Un recuperador
híbrido con re-ranking sobre los 41 fragmentos daría otra respuesta. Eso
sería trabajo siguiente, y el proyecto no lo mide.

**"¿Por qué no corriste las 50?"**
Porque el proveedor gratuito limita a 8.000 tokens por minuto y una
consulta de CAG gasta 3.951. Después de 23 consultas me pidió esperar 862
segundos; el resto eran unas seis horas. Preferí 23 respuestas reales y
declaradas que 50 responses inventadas. El dataset completo está, y se
puede correr con un plan pago o con menos pausa entre llamadas.

**"¿El corpus es representativo?"**
No, y está declarado en el manifiesto. Es un recorte de seis páginas.
Sirve para el método, no como muestra del Registro Civil.

**"¿Cómo sabes que el modelo no inventa?"**
No se sabe, se mide. Por eso existen las diez preguntas de fuera de
corpus, y por eso hay que decir que no se corrieron.

---

## 10. Si algo falla en vivo

| Síntoma | Qué hacer |
|---|---|
| El proveedor da 429 | Esperar. Decirlo en voz alta: es parte del experimento. |
| La demo de recuperación tarda | Carga el modelo de embeddings. Lanzarla 2 minutos antes. |
| El puerto 8000 está ocupado | `--port 8010` |
| La API da 502 | El proveedor cayó. Es un fallo real, no una abstención: el sistema distingue los dos. |

**Lo más importante:** si algo falla, dejar que se vea y explicar que el
fallo técnico no es lo mismo que una abstención. Esa distinción es parte
del diseño y se nota en el código de estado HTTP.