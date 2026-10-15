# Por que CAG y no RAG en este proyecto

Esta es la pregunta que hay que responder en el seminario. La respuesta
corta es que **no hay una respuesta general**: CAG y RAG resuelven
problemas distintos, y la decision depende de una sola variable, que es
si el corpus entra en la ventana de contexto.

Todo lo de abajo sale de mediciones de este repositorio, no de opinion.

---

## 1. Primero, la premisa que hay que corregir

La pregunta era "si RAG es mas rapido y gasta menos tokens". La primera
mitad es falsa y la segunda es cierta.

**RAG no es mas rapido.** En la corrida de 23 preguntas, con la cache del
proveedor funcionando:

| | Acierto | Latencia p50 | Latencia p90 | Tokens mediana |
|---|---|---|---|---|
| **CAG** | **21/23 - 91,3%** | 1.547 ms | 2.124 ms | 3.951 |
| **RAG** | 13/23 - 56,5% | 1.406 ms | 1.922 ms | 1.493 |

Empate en latencia: 141 milisegundos de diferencia no significan nada
frente a la dispersion. RAG si gasta menos tokens, 1.493 contra 3.951, un
62% menos.

La medicion inicial de 30.706 ms para RAG era falsa: era la carga del
modelo de embeddings, no recuperacion.

Y el resultado no es de una sola cara: **RAG acerto dos consultas donde
CAG fallo**, R05 y R10.

Por categoria:

| Categoria | CAG | RAG |
|---|---|---|
| Requisitos | 8/10 | 6/10 |
| Procedimiento | 10/10 | 7/10 |
| Multi seccion | **3/3** | **0/3** |

Multi seccion es donde la diferencia se abre del todo: RAG no acierta
ninguna de las tres. Son preguntas que exigen combinar dos secciones
distintas, y el recuperador no trae las dos.

## 2. Y el costo no decide nada aca

Con los precios publicados por Groq para `gpt-oss-120b` —entrada
$0.150/1M, entrada cacheada $0.075/1M, salida $0.600/1M:

| Escenario | CAG | RAG | Diferencia por consulta |
|---|---|---|---|
| Cache funcionando | $0.000379 | $0.000289 | $0.000090 |
| Cache fria | $0.000667 | $0.000327 | $0.000340 |

En 100 consultas: **un centavo**. En 100.000 consultas: **nueve
dolares**.

Con seis documentos, ninguna de las dos estrategias es cara. Pedirle a
esta decision que se justifique por costo es pedirle una justificacion
falsa.

Ademas hay un detalle uncomfortable: **CAG solo gana en costo cuando la
cache funciona**, y Groq documenta que el acierto de cache es
"best effort", no garantizado. En este proyecto el mismo prefijo dio
0%, 14%, 73%, 81% y 98% de acierto en corridas distintas. Construir una
tesis economica sobre una cache que el propio proveedor declara
incierta seria apoyarse en lo mas fragil del sistema.

Se puede reproducir con `python scripts/costo_consulta.py`.

---

## 3. La pregunta sobre el modelo: se puede contestar con un experimento

La pregunta era si el avantage de CAG viene del modelo que se esta
usando. La respuesta es **no para este resultado, y si para otro
distinto**, asi que conviene separarlas.

### 3.1 La abstencion de RAG no es culpa del modelo

RAG se abstuvo de una pregunta cuya respuesta si esta en el corpus. Se
podia sospechar que el modelo era incapaz. Se armo el mismo prompt que
recibio RAG, con una sola variable distinta: 679 caracteres con el
fragmento que faltaba.

| Lo que recibio el modelo | Respuesta |
|---|---|
| Sin el fragmento — lo que recibio RAG | *"No tengo informacion suficiente en las fuentes consultadas."* |
| Con el fragmento | Los tres requisitos, correctos |

Mismo modelo, mismas reglas, misma pregunta. **El modelo podia
responder.** Lo que fallo fue la recuperacion.

Esto no depende del modelo. Es aritmetica de la informacion: si el
fragmento no esta en el prompt, ninguna respuesta correcta es posible. Un
modelo distinto se abstendria igual, o peor: inventaria algo.

Se reproduce con `python scripts/experimento_modelo.py`.

### 3.2 Donde el modelo si importa

Con honestidad, hay dos lugares donde el modelo importa:

**a) Que la falla sea visible.** La abstencion se ve porque el sistema
esta disenado para abstenerse antes que para adivinar. Si el modelo
confabulara, RAG "ganaria" esta consulta con una respuesta inventada y
nadie se enteraria de que el recuperador fallo. El modelo no cause el
problema, pero su disposicion a abstenerse es lo que lo vuelve
diagnosticable. Y esa disposicion viene de las reglas del prompt y de la
verificacion local, que son trabajo del disenio del sistema.

**b) Que CAG pueda aprovechar un contexto largo.** CAG manda 3.952
tokens y apuesta a que el modelo los aprovecha. `gpt-oss-120b` los
aprovecha. Un modelo chico puede empeorar con contexto largo, y en este
proyecto ya se vio: el modelo local de 1,5 B que se probo al principio
fallaba las pruebas antihallucinacion 5 de 5, contra 2 de 3 del
proveedor remoto.

Asi que la pregunta por el modelo tiene una respuesta distinta segun a
que resultado se refiera. Para la abstencion, no. Para el rendimiento de
CAG con contexto largo, si, y ahi CAG fue unadecision de riesgo
medido.

---

## 4. La justificacion que si se sostiene

La razon para elegir CAG no es que sea mas rapida, ni mas barata, ni que
el modelo sea mejor. Es una sola:

> **El corpus entra en la ventana de contexto.**

Los numeros:

| | |
|---|---|
| Ventana del modelo | 32.768 tokens |
| Presupuesto de CAG | 6.000 tokens |
| Uso actual | 3.952 tokens (66%) |
| Corpus disponible | 3.730 tokens (73% de lo que queda) |
| Promedio por documento | 622 tokens |
| **Documentos que entran** | **8** |
| **Documentos donde CAG se rompe** | **9** |

Seis documentos hoy. El noveno documento rompe CAG.

Y el punto importante no es el numero: es que **el costo de CAG no crece
de forma gradual, tiene un borde**. O entra todo el corpus y funciona,
o hay que truncar y se pierde informacion. No hay un punto medio.

La regla de decision que sale de ahi:

> **CAG cuando el corpus es chico y estable. RAG cuando el corpus es
> grande o cambia seguido.**

Con seis documentos de un registro civil que se actualiza pocas veces,
CAG es la opcion correcta. No porque sea superior en abstracto, sino
porque es la que **no puede perder informacion por no haberla buscado**.

---

## 5. El hallazgo que sostiene la tesis del seminario

Lo que el experimento del punto 3.1 demuestra, y que es mas interesante
que la latencia o el costo, es esto:

**La falla de RAG es silenciosa.**

No hay error, no hay aviso, no hay log. El sistema devuelve "no tengo
informacion suficiente" y eso es **indistinguible** de que el documento
realmente no trate el tema. En un sistema asi, nadie puede distinguir:

- "La fuente no lo dice"
- "El recuperador no lo encontró"

Para un ciudadano que pregunta que papeles necesita, esa indistinguibilidad
es el riesgo real: el sistema se abstain mucho, con toda honestidad, y
nadie va a saber por que.

CAG no tiene esa falla. No recupera nada, asi que no puede dejar pasar
un dato. Se trunca o funciona, y cuando se trunca lo dice.

Ese es el argumento del seminario, y no es "CAG es mejor". Es:

> **Con un corpus chico, la recuperacion es una fuente de falla que el
> sistema no puede ver. Mandar el corpus entero la elimina.**

Y tiene una segunda mitad, que conviene decir en voz alta: cuando el
corpus crezca a nueve documentos, ese argumento se invierte, y RAG pasa
a ser la unica opcion. El proyecto no demuestra que CAG sea superior.
Demuestra que **la eleccion depende del tamano del corpus**, y que esa
variable es la que hay que mirar antes de decidir.

---

## 6. Lo que este documento no demuestra

- **No demuestra que CAG sea mas rapido.** Los datos dan empate:
  1.547 ms contra 1.406 ms.
- **No demuestra que CAG sea mas barato.** RAG es mas barato incluso con
  la cache funcionando, con 62% menos tokens.
- **No demuestra que CAG sea mejor en general.** Con 9 documentos, CAG no
  funciona. Y en esta corrida RAG acerto 2 de 23 donde CAG fallo.
- **No demuestra nada sobre la honestidad del sistema.** Las diez
  preguntas de fuera de corpus, las que miden si el agente se abstiene en
  vez de inventar, **no se ejecutaron**. Es la limitacion mas grave de
  toda la corrida.
- **23 preguntas no son una muestra solida.** La diferencia es grande,
  pero el margen de error con 23 casos es amplio.
- **No cubre preguntas ambiguas.** Otras diez sin ejecutar.
- **La comparacion favorece a CAG por construccion.** CAG manda los
  3.730 tokens del corpus y RAG 1.493. Es la definicion de cada tecnica,
  no un sesgo del experimento, pero conviene decirlo en voz alta.

La corrida completa puede—invertirse, y habria que decirlo asi. Con 23 de
50 preguntas, y **ninguna** de las diez que miden honestidad, el resultado
sostiene una tesis sobre **cobertura** y no sobre **seguridad**.