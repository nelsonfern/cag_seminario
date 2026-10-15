# Guia de exposicion

Que mostrar en vivo, que decir, y que tener preparado por si algo falla.
Esta guia asume diez minutos de demostracion y cinco de preguntas.

**Antes de empezar:** las demos que dependan del proveedor de lenguaje
arrancan lento. Lanzalas con antelacion. Un minuto de "cargando" en vivo
se siente como un fallo aunque no lo sea.

---

## 1. La apertura (30 segundos)

No empieces mostrando la interfaz. Empieza con la pregunta que organiza
todo el proyecto:

> "Tengo seis documentos oficiales de un registro civil. Los puedo
> mandar enteros al modelo, o puedo buscar en ellos lo relevante y
> mandar solo eso. Es CAG y RAG. Los comparo con el mismo modelo, las
> mismas preguntas y el mismo corpus, y la pregunta no es cual es mas
> rapido: es cual deja de responder cosas que si estan en las fuentes."

Despues, una advertencia que es parte del trabajo:

> "El corpus es real y sale de sitios oficiales. Una de las paginas tiene
> el costo del matrimonio escrito como `$??`. No lo complete. Si lo
> complete, estaria inventando."

---

## 2. Demo en vivo: la palabra que rompe todo (3 minutos)

Esta es la demo principal. **No gasta un solo token** porque la
recuperacion es local.

```powershell
.venv\Scripts\python.exe scripts\demo_recuperacion.py
```

Que muestra, en pantalla:

```
PREGUNTA: ¿Qué documentación necesito para una partida de nacimiento?
   con la palabra 'partida'
   ...
 4. 0.652  dentro del corte    Requisitos del trámite
 6. 0.594  FUERA del corte     Inscripción de Nacimientos
  >> 'documentación requerida' salió en el puesto 8: NO LLEGA al corte

PREGUNTA: ¿Qué documentación necesito para inscribir un nacimiento?
   con la palabra 'inscribir'
  ...
 5. 0.627  dentro del corte    Documentación requerida   <<< ESTA ES LA RESPUESTA
  >> 'documentación requerida' salió en el puesto 5: LLEGA
```

**Lo que se dice:**

> "El fragmento que tiene la respuesta existe, esta bien cortado y esta
> indexado. Lo unico que cambio fue una palabra. Con 'partida' queda en
> el puesto 8 y no llega al corte. Con 'inscribir' queda en el 5 y
> llega. Una palabra decide si el ciudadano recibe una respuesta o un
> 'no tengo informacion'."

**Si preguntan por que cae tan bajo:** el corte es de cinco fragmentos.
El corpus tiene 41. El recuperador ordena por parecido semantico, y
"partida" tira hacia el documento de solicitud de partidas mientras la
pregunta quiere la de inscripcion de nacimiento.

---

## 3. Demo en vivo: el modelo no era el problema (2 minutos)

```powershell
.venv\Scripts\python.exe scripts/experimento_modelo.py
```

Muestra dos respuestas al mismo prompt:

| Lo que recibio el modelo | Respuesta |
|---|---|
| Sin el fragmento — lo que recibio RAG | *"No tengo informacion suficiente..."* |
| Con el fragmento | Los tres requisitos, correctos |

**Lo que se dice:**

> "Mismo modelo, mismo prompt, misma pregunta. Lo unico que cambio fueron
> 679 caracteres con el fragmento que faltaba. El modelo podia responder.
> Lo que fallo fue la recuperacion, y eso no depende del modelo: si el
> fragmento no esta en el prompt, ninguna respuesta correcta es posible."

**Si preguntan si esto es culpa del modelo:** no, y esta es la prueba.
Donde el modelo si importa es en que se abstenga antes que adivinar. Si
confabulara, RAG ganaria esta consulta con una respuesta inventada y
nadie veria que el recuperador fallo.

---

## 4. Demo en vivo: la comparacion lado a lado (2 minutos)

Levantar la aplicacion:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app
```

Abrir `http://localhost:8000`. La interfaz arranca en modo **comparar**.

Preguntar algo que este en las dos, por ejemplo:

> ¿Cuánto cuesta la copia de acta certificada?

Se ven las dos columnas con sus metricas debajo. Ahi conviene decir:

> "CAG mando 3.952 tokens, RAG 1.446. RAG usa un tercio. La latencia da
> empate. Pero miren lo que hay que mirar primero: si las dos responden
> con fuentes o alguna se abstiene."

**Advertencia de ritmo:** el proveedor gratuito da 8.000 tokens por minuto
y cada consulta de CAG manda casi 4.000. Las esperas de 20 a 30 segundos
son normales. La interfaz lo avisa. Decirlo de antemano evita que se lea
como un fallo.

---

## 5. El costo (1 minuto)

```powershell
.venv\Scripts\python.exe scripts/costo_consulta.py
```

**Lo que se dice:**

> "Con los precios publicados de Groq, la diferencia es de un centavo cada
> cien consultas. Con seis documentos, ninguna de las dos es cara. Y CAG
> solo gana en costo cuando la cache funciona, que el propio proveedor
> declara como 'best effort': en este proyecto el mismo prefijo dio entre
> 0% y 98% de acierto entre corridas. Una tesis economica apoyada ahi se
> cae sola."

---

## 6. La tabla que sostiene la conclusion

```
                 Latencia    Tokens    Cache    Corpus completo
CAG              1.599 ms    3.952     97,2%    siempre
RAG              1.624 ms    1.446     35,4%    nunca (se trunca en 9 docs)
Corte de CAG: 8 documentos. El noveno lo rompe.
```

**Lo que se dice:**

> "CAG no es mas rapido. No es mas barato. La justificacion es una sola:
> el corpus entra en la ventana de contexto, y a partir de nueve
> documentos deja de entrar. El costo de CAG no crece de forma gradual,
> tiene un borde."

---

## 7. Lo que el proyecto no demuestra

Esto hay que decirlo antes de que lo pregunten, no despues:

- **No demuestra que CAG sea mejor.** Con nueve documentos no funciona.
- **No demuestra que RAG sea peor en general.** Pierde informacion cuando
  el fragmento correcto cae fuera del corte; con otros corpora o con un
  corte mas grande puede no pasar.
- **La comparacion depende de la cache.** Con la cache fria CAG pierde en
  velocidad. Los numeros de arriba son con la cache caliente.
- **El corpus es chico.** Seis documentos. Los resultados no se
  generalizan a un registro civil completo, que tiene cientos de
  secciones.

---

## 8. Preguntas que seguro van a hacer

**"¿Por que no usaste un modelo local?"**
Porque la maquina del seminario tiene 5,8 GB de memoria y no entra uno
que pase las pruebas antihallucinacion. El de 1,5 B que se probo al
principio fallaba 5 de 5; el remoto, 2 de 3. Con el mismo corpus, mismo
modelo, misma pregunta.

**"RAG es lo que usa todo el mundo, ¿no?"**
Si, y esta bien para corpus grandes y cambiantes. El proyecto no dice que
RAG este mal: dice que RAG tiene un modo de falla silencioso, y que con
seis documentos no compensa pagarlo.

**"¿Y si el recuperador fuera mejor?"**
El argumento se debilita, y hay que decirlo asi. El finding es sobre este
recuperador, con este corte, sobre este corpus. Un recuperador hibrido con
re-ranking sobre los 41 fragmentosaria otra respuesta. Eso seria trabajo
siguiente, y el proyecto no lo mide.

**"¿El corpus es representativo?"**
No, y esta declarado en el manifiesto. Es un recorte de seis paginas. Sirve
para el metodo, no como muestra del Registro Civil.

**"¿Como sabes que el modelo no inventa?"**
No se sabe, se mide. Las diez preguntas fuera de corpus son para eso: si
el sistema contesta alguna, esta inventando. El resultado de esa medicion
esta en el informe del experimento.

---

## 9. Si algo falla en vivo

| Sintoma | Que hacer |
|---------|-----------|
| El proveedor da 429 | Esperar. El limite es de 8.000 tokens por minuto. Decirlo en voz alta: es parte del experimento. |
| La demo de recuperacion tarda | Carga el modelo de embeddings. Lanzarla 2 minutos antes. |
| RAG devuelvecontexto vacio | El indice no esta. `python scripts\demo_recuperacion.py` lo reconstruye solo. |
| El puerto 8000 esta ocupado | `uvicorn app.main:app --port 8010` |
| La API da 502 | El proveedor cayo. Es un fallo real, no una abstencion: el sistema distingue los dos. |

**Lo mas importante:** si el sistema falla, dejar que se vea y explicar
que el fallo tecnico no es lo mismo que una abstencion. Esa distincion es
parte del disenio y se nota en el codigo de estado HTTP.