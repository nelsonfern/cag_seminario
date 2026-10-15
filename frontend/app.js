/* TramitaIA - logica del cliente.
   Sin dependencias externas: fetch y DOM nativos.

   Dos modos:
     - comparar: la misma pregunta a CAG y a RAG, lado a lado
     - simple:   una sola estrategia, mas rapido y mas barato

   El modo comparar es el que hace visible el experimento. El simple esta
   para cuando solo se quiere una respuesta. */

(function () {
  "use strict";

  const API = "";              // mismo origen
  const ENDPOINT_CHAT = `${API}/chat`;
  const ENDPOINT_COMPARAR = `${API}/chat/compare`;
  const ENDPOINT_HEALTH = `${API}/health`;

  const $ = (id) => document.getElementById(id);

  const elementos = {
    formulario: $("formulario"),
    pregunta: $("pregunta"),
    enviar: $("enviar"),
    conversacion: $("conversacion"),
    vacio: $("vacio"),
    ejemplos: $("ejemplos"),
    estado: $("estado"),
    metricas: $("metricas-cuerpo"),
    selectorSimple: $("selector-simple"),
    notaLimite: $("nota-limite"),
  };

  let ocupada = false;

  // ------------------------------------------------------------------
  // Modo y estrategia
  // ------------------------------------------------------------------
  function modoSeleccionado() {
    const marcada = document.querySelector('input[name="modo"]:checked');
    return marcada ? marcada.value : "comparar";
  }

  function estrategiaSeleccionada() {
    const marcada = document.querySelector('input[name="estrategia"]:checked');
    return marcada ? marcada.value : "cag";
  }

  function marcarGrupo(nombre) {
    document.querySelectorAll(`input[name="${nombre}"]`).forEach((radio) => {
      radio.addEventListener("change", () => {
        radio.closest(".modo,.opcion").classList.toggle("activa", radio.checked);
      });
    });
  }
  marcarGrupo("modo");
  marcarGrupo("estrategia");

  document.querySelectorAll('input[name="modo"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      // El selector de estrategia solo tiene sentido en modo simple.
      elementos.selectorSimple.hidden = modoSeleccionado() !== "simple";
      elementos.notaLimite.hidden = modoSeleccionado() !== "comparar";
    });
  });

  // ------------------------------------------------------------------
  // Estado del servicio
  // ------------------------------------------------------------------
  async function comprobarEstado() {
    try {
      const r = await fetch(ENDPOINT_HEALTH);
      const datos = await r.json();
      const ok = datos.status === "ok";

      elementos.estado.className = "estado " + (ok ? "ok" : "error");
      elementos.estado.querySelector(".texto").textContent = ok
        ? `${datos.llm_provider} / ${datos.model || "sin modelo"} — ${datos.corpus_documentos} documentos`
        : "Sin proveedor configurado";
    } catch {
      elementos.estado.className = "estado error";
      elementos.estado.querySelector(".texto").textContent =
        "No se pudo contactar al servidor";
    }
  }

  // ------------------------------------------------------------------
  // Pintado
  // ------------------------------------------------------------------
  function burbuja(texto) {
    const div = document.createElement("div");
    div.className = "burbuja";
    div.textContent = texto;
    return div;
  }

  function pintarUsuario(texto) {
    const mensaje = document.createElement("div");
    mensaje.className = "mensaje usuario";
    mensaje.appendChild(burbuja(texto));
    elementos.conversacion.appendChild(mensaje);
  }

  function pintarAvisos(destino, avisos) {
    if (!avisos || !avisos.length) return;

    const caja = document.createElement("div");
    caja.className = "avisos";
    avisos.forEach((texto) => {
      const p = document.createElement("p");
      p.textContent = texto;
      caja.appendChild(p);
    });
    destino.appendChild(caja);
  }

  function pintarFuentes(destino, fuentes) {
    if (!fuentes || !fuentes.length) return;

    const detalles = document.createElement("details");
    detalles.className = "fuentes";

    const resumen = document.createElement("summary");
    resumen.textContent = `Fuentes oficiales consultadas (${fuentes.length})`;
    detalles.appendChild(resumen);

    const lista = document.createElement("ul");
    fuentes.forEach((fuente) => {
      const item = document.createElement("li");

      const titulo = document.createElement("span");
      titulo.className = "titulo";
      titulo.textContent = fuente.titulo;
      item.appendChild(titulo);

      if (fuente.url_origen) {
        const salto = document.createElement("br");
        const enlace = document.createElement("a");
        enlace.href = fuente.url_origen;
        enlace.target = "_blank";
        enlace.rel = "noopener noreferrer";
        enlace.textContent = "Ver en el sitio oficial";
        item.appendChild(salto);
        item.appendChild(enlace);
      }

      lista.appendChild(item);
    });

    detalles.appendChild(lista);
    destino.appendChild(detalles);
  }

  function pintarMetricasBasicas(destino, datos, etiqueta) {
    const uso = datos.token_usage || {};
    const est = datos.estrategia_datos || {};

    const caja = document.createElement("div");
    caja.className = "metricas-mini";

    const partes = [`${Math.round(datos.latency_ms)} ms`];
    if (uso.tokens_entrada) partes.push(`${uso.tokens_entrada} tokens de entrada`);
    if (uso.cache_disponible) {
      const pct = Math.round((100 * uso.tokens_cache_lectura) / uso.tokens_entrada);
      partes.push(`caché ${pct}%`);
    } else {
      partes.push("caché no informada");
    }
    if (est.estrategia === "cag" && est.documentos_en_contexto != null) {
      partes.push(`${est.documentos_en_contexto} documentos`);
    }
    if (est.estrategia === "rag" && est.fragmentos_recuperados != null) {
      partes.push(`${est.fragmentos_recuperados} fragmentos`);
    }

    caja.textContent = `${etiqueta}: ${partes.join(" · ")}`;
    destino.appendChild(caja);
  }

  function panelAgente(datos, etiqueta) {
    const panel = document.createElement("div");
    panel.className = "panel-agente" + (datos.grounded ? "" : " abstencion");

    const cabecera = document.createElement("div");
    cabecera.className = "panel-cabecera";
    cabecera.textContent = etiqueta;
    panel.appendChild(cabecera);

    panel.appendChild(burbuja(datos.answer));
    pintarMetricasBasicas(panel, datos, etiqueta);
    pintarFuentes(panel, datos.sources);
    pintarAvisos(panel, datos.warnings);

    return panel;
  }

  function pintarComparacion(datos) {
    const mensaje = document.createElement("div");
    mensaje.className = "mensaje comparacion";

    const titulo = document.createElement("span");
    titulo.className = "etiqueta-estrategia";
    titulo.textContent = "MISMA PREGUNTA, DOS ESTRATEGIAS";
    mensaje.appendChild(titulo);

    const columnas = document.createElement("div");
    columnas.className = "columnas";
    columnas.appendChild(panelAgente(datos.cag, "CAG"));
    columnas.appendChild(panelAgente(datos.rag, "RAG"));
    mensaje.appendChild(columnas);

    // Lectura de la comparacion en una frase: una tabla sin leer no dice nada.
    const lectura = document.createElement("p");
    lectura.className = "lectura";
    lectura.textContent = leerDiferencias(datos.diferencias);
    mensaje.appendChild(lectura);

    // Las notas del servidorexplain cuando un numero no es comparable.
    if (datos.diferencias && datos.diferencias.notas) {
      pintarAvisos(mensaje, datos.diferencias.notas);
    }

    elementos.conversacion.appendChild(mensaje);
    mensaje.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function leerDiferencias(d) {
    if (!d || d.latencia_cag_ms == null) return "";

    const partes = [];
    const difLat = d.diferencia_latencia_ms;

    // Con el indice ya construido, el tiempo de RAG es el real. Si no,
    // comparar los dos tiempos seria comparar dos arranques en frio.
    if (d.arranque_en_frio) {
      partes.push(
        "La latencia de RAG incluye reconstruir su indice, asi que los dos tiempos no son comparables todavia."
      );
    } else if (difLat < -200) {
      partes.push("En esta consulta CAG respondió más rápido.");
    } else if (difLat > 200) {
      partes.push("En esta consulta RAG respondió más rápido.");
    } else {
      partes.push("Los tiempos fueron similares.");
    }

    if (d.tokens_rag_menos_que_cag != null && d.tokens_rag_menos_que_cag > 200) {
      partes.push(
        `RAG mandó ${Math.round(d.tokens_rag_menos_que_cag)} tokens menos de contexto.`
      );
    }

    if (d.cache_reportada_cag && !d.cache_reportada_rag) {
      partes.push(
        "El proveedor cacheó el contexto de CAG y no el de RAG, porque en RAG el contexto cambia en cada consulta."
      );
    }

    if (d.grounded_cag !== d.grounded_rag) {
      partes.push(
        "Una se abstuvo y la otra no: los fragmentos recuperados alcanzan o no alcanzan la evidencia."
      );
    }

    return partes.join(" ");
  }

  function pintarSimple(datos, etiqueta) {
    const mensaje = document.createElement("div");
    mensaje.className = "mensaje agente" + (datos.grounded ? "" : " abstencion");

    const cabecera = document.createElement("span");
    cabecera.className = "etiqueta-estrategia";
    cabecera.textContent =
      etiqueta + (datos.grounded ? " — con fuentes" : " — sin evidencia suficiente");
    mensaje.appendChild(cabecera);

    mensaje.appendChild(burbuja(datos.answer));
    pintarMetricasBasicas(mensaje, datos, etiqueta);
    pintarFuentes(mensaje, datos.sources);
    pintarAvisos(mensaje, datos.warnings);

    elementos.conversacion.appendChild(mensaje);
    mensaje.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function pintarError(mensaje) {
    const div = document.createElement("div");
    div.className = "mensaje error";
    div.appendChild(burbuja(mensaje));
    elementos.conversacion.appendChild(div);
  }

  // ------------------------------------------------------------------
  // Espera
  // ------------------------------------------------------------------
  function mostrarEspera(texto) {
    let aviso = $("espera");
    if (!aviso) {
      aviso = document.createElement("div");
      aviso.id = "espera";
      aviso.className = "espera";
      elementos.conversacion.appendChild(aviso);
    }
    aviso.textContent = texto;
    return aviso;
  }

  function ocultarEspera() {
    const aviso = $("espera");
    if (aviso) aviso.remove();
  }

  /** Cuenta hacia atrás mientras espera la respuesta. */
  function iniciarEspera(segundosEstimados) {
    let restantes = segundosEstimados;
    const reloj = setInterval(() => {
      restantes -= 1;
      if (restantes <= 0) {
        mostrarEspera(
          "El proveedor tiene un límite de 8.000 tokens por minuto. " +
            "La consulta va a arrancar en cuanto se libere el cupo."
        );
        clearInterval(reloj);
      }
    }, 1000);
    return reloj;
  }

  // ------------------------------------------------------------------
  // Envío
  // ------------------------------------------------------------------
  function ocupar(texto) {
    ocupada = true;
    elementos.enviar.disabled = true;
    elementos.enviar.querySelector(".texto").hidden = true;
    elementos.enviar.querySelector(".cargando").hidden = false;
    mostrarEspera(texto);
  }

  function liberar() {
    ocupada = false;
    elementos.enviar.disabled = false;
    elementos.enviar.querySelector(".texto").hidden = false;
    elementos.enviar.querySelector(".cargando").hidden = true;
    ocultarEspera();
  }

  async function consultar(texto) {
    if (ocupada) return;

    const comparar = modoSeleccionado() === "comparar";
    ocupar(
      comparar
        ? "Consultando CAG y RAG. Puede tardar: son dos consultas al proveedor…"
        : "Consultando…"
    );

    const reloj = iniciarEspera(comparar ? 12 : 8);

    try {
      const url = comparar ? ENDPOINT_COMPARAR : ENDPOINT_CHAT;
      const cuerpo = comparar
        ? { question: texto }
        : { question: texto, strategy: estrategiaSeleccionada() };

      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cuerpo),
      });

      clearInterval(reloj);

      if (!r.ok) {
        let detalle = `El servidor respondió ${r.status}.`;
        try {
          const datos = await r.json();
          if (datos.detail) detalle = datos.detail;
        } catch {
          /* la respuesta no era JSON */
        }
        pintarError(
          `${detalle} Eso es un problema del servicio, no significa que no exista información.`
        );
        return;
      }

      const datos = await r.json();
      if (comparar) pintarComparacion(datos);
      else pintarSimple(datos, datos.strategy.toUpperCase());
    } catch {
      clearInterval(reloj);
      pintarError("No se pudo contactar al servidor.");
    } finally {
      liberar();
    }
  }

  // ------------------------------------------------------------------
  // Eventos
  // ------------------------------------------------------------------
  elementos.formulario.addEventListener("submit", (e) => {
    e.preventDefault();
    const texto = elementos.pregunta.value.trim();
    if (!texto) return;

    if (elementos.vacio) elementos.vacio.remove();
    pintarUsuario(texto);
    elementos.pregunta.value = "";
    consultar(texto);
  });

  elementos.ejemplos.addEventListener("click", (e) => {
    const boton = e.target.closest("button[data-pregunta]");
    if (!boton) return;
    elementos.pregunta.value = boton.dataset.pregunta;
    elementos.pregunta.focus();
    elementos.formulario.requestSubmit();
  });

  elementos.pregunta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      elementos.formulario.requestSubmit();
    }
  });

  comprobarEstado();
})();