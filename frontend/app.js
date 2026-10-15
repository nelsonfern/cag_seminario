/* TramitaIA - logica del cliente.
   Sin dependencias externas: fetch y DOM nativos.
   La app es un archivo estatico, se sirve desde cualquier servidor. */

(function () {
  "use strict";

  const API = "";              // mismo origen; cambiar si la API esta en otro host
  const ENDPOINT_CHAT = `${API}/chat`;
  const ENDPOINT_HEALTH = `${API}/health`;

  const elementos = {
    formulario: document.getElementById("formulario"),
    pregunta: document.getElementById("pregunta"),
    enviar: document.getElementById("enviar"),
    conversacion: document.getElementById("conversacion"),
    vacio: document.getElementById("vacio"),
    ejemplos: document.getElementById("ejemplos"),
    estado: document.getElementById("estado"),
    metricas: document.getElementById("metricas-cuerpo"),
  };

  let ocupada = false;

  // ------------------------------------------------------------------
  // Estrategia
  // ------------------------------------------------------------------
  function estrategiaSeleccionada() {
    const marcada = document.querySelector('input[name="estrategia"]:checked');
    return marcada ? marcada.value : "cag";
  }

  document.querySelectorAll('input[name="estrategia"]').forEach(function (radio) {
    radio.addEventListener("change", function () {
      document.querySelectorAll(".opcion").forEach(function (etiqueta) {
        etiqueta.classList.toggle("activa", etiqueta.contains(radio) && radio.checked);
      });
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
        ? `${datos.llm_provider} / ${datos.model || "sin modelo"} - ${datos.corpus_documentos} documentos`
        : "Sin proveedor configurado";
    } catch (e) {
      elementos.estado.className = "estado error";
      elementos.estado.querySelector(".texto").textContent = "No se pudo contactar al servidor";
    }
  }

  // ------------------------------------------------------------------
  // Pintado
  // ------------------------------------------------------------------
  function burbuja(clase, texto) {
    const div = document.createElement("div");
    div.className = "burbuja " + clase;
    div.textContent = texto;
    return div;
  }

  function pintarUsuario(texto) {
    const mensaje = document.createElement("div");
    mensaje.className = "mensaje usuario";
    mensaje.appendChild(burbuja("", texto));
    elementos.conversacion.appendChild(mensaje);
  }

  function pintarAgente(datos) {
    const mensaje = document.createElement("div");
    const sinRespaldo = !datos.grounded;
    mensaje.className =
      "mensaje agente" + (sinRespaldo ? " abstencion" : "");

    const etiqueta = document.createElement("span");
    etiqueta.className = "etiqueta-estrategia";
    etiqueta.textContent =
      (datos.strategy || "").toUpperCase() +
      (sinRespaldo ? " - sin evidencia suficiente" : " - con fuentes");
    mensaje.appendChild(etiqueta);

    mensaje.appendChild(burbuja("", datos.answer));

    // Las fuentes van aparte, con enlace a la pagina oficial.
    if (datos.sources && datos.sources.length) {
      const detalles = document.createElement("details");
      detalles.className = "fuentes";

      const resumen = document.createElement("summary");
      resumen.textContent =
        `Fuentes oficiales consultadas (${datos.sources.length})`;
      detalles.appendChild(resumen);

      const lista = document.createElement("ul");
      datos.sources.forEach(function (fuente) {
        const item = document.createElement("li");

        const titulo = document.createElement("span");
        titulo.className = "titulo";
        titulo.textContent = fuente.titulo;
        item.appendChild(titulo);

        const vigente = fuente.vigencia;
        if (vigente && vigente !== "vigente") {
          const aviso = document.createElement("span");
          aviso.className = "vigencia incierto";
          aviso.textContent = "Vigencia no confirmada: " + vigente;
          item.appendChild(aviso);
        }

        if (fuente.url_origen) {
          const enlace = document.createElement("a");
          enlace.href = fuente.url_origen;
          enlace.target = "_blank";
          enlace.rel = "noopener noreferrer";
          enlace.textContent = "Ver en el sitio oficial";
          item.appendChild(document.createElement("br"));
          item.appendChild(enlace);
        }

        lista.appendChild(item);
      });

      detalles.appendChild(lista);
      mensaje.appendChild(detalles);
    }

    // Los avisos del agente son lo mas importante: explican por que la
    // respuesta puede no ser confiable.
    if (datos.warnings && datos.warnings.length) {
      const avisos = document.createElement("div");
      avisos.className = "avisos";
      datos.warnings.forEach(function (texto) {
        const p = document.createElement("p");
        p.textContent = texto;
        avisos.appendChild(p);
      });
      mensaje.appendChild(avisos);
    }

    elementos.conversacion.appendChild(mensaje);
    mensaje.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function pintarError(mensaje) {
    const div = document.createElement("div");
    div.className = "mensaje error";
    div.appendChild(burbuja("", mensaje));
    elementos.conversacion.appendChild(div);
  }

  // ------------------------------------------------------------------
  // Metricas
  // ------------------------------------------------------------------
  function pintarMetricas(datos) {
    elementos.metricas.innerHTML = "";

    const uso = datos.token_usage || {};
    const estrategia = datos.estrategia_datos || {};
    const casteo = uso.cache_disponible
      ? `${uso.tokens_cache_lectura}/${uso.tokens_entrada}`
      : "no informada";

    const items = [
      { clave: "Latencia", valor: `${Math.round(datos.latency_ms)} ms` },
      { clave: "Tokens de entrada", valor: uso.tokens_entrada ?? "n/d" },
      { clave: "Tokens de salida", valor: uso.tokens_salida ?? "n/d" },
      {
        clave: "Cache del proveedor",
        valor: uso.cache_disponible ? `${casteo}` : "n/d",
        nota: uso.cache_disponible
          ? `${Math.round((100 * uso.tokens_cache_lectura) / uso.tokens_entrada)}% reutilizado`
          : "el proveedor no lo reporta",
      },
    ];

    if (estrategia.estrategia === "cag") {
      items.push({
        clave: "Documentos en contexto",
        valor: estrategia.documentos_en_contexto ?? "n/d",
        nota: estrategia.truncado
          ? `truncado, ${estrategia.documentos_omitidos} fuera`
          : `~${estrategia.tokens_contexto_estimados ?? "?"} tokens`,
      });
    } else if (estrategia.estrategia === "rag") {
      items.push({
        clave: "Fragmentos recuperados",
        valor: estrategia.fragmentos_recuperados ?? "n/d",
        nota: `recuperación ${Math.round(estrategia.recuperacion_ms ?? 0)} ms`,
      });
    }

    items.forEach(function (item) {
      const caja = document.createElement("div");
      caja.className = "metrica";

      const clave = document.createElement("span");
      clave.className = "clave";
      clave.textContent = item.clave;

      const valor = document.createElement("span");
      valor.className = "valor";
      valor.textContent = item.valor;

      caja.appendChild(clave);
      caja.appendChild(valor);

      if (item.nota) {
        const nota = document.createElement("span");
        nota.className = "nota";
        nota.textContent = item.nota;
        caja.appendChild(nota);
      }

      elementos.metricas.appendChild(caja);
    });
  }

  // ------------------------------------------------------------------
  // Envio
  // ------------------------------------------------------------------
  async function consultar(texto) {
    if (ocupada) return;
    ocupada = true;

    elementos.enviar.disabled = true;
    elementos.enviar.querySelector(".texto").hidden = true;
    elementos.enviar.querySelector(".cargando").hidden = false;

    try {
      const r = await fetch(ENDPOINT_CHAT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: texto, strategy: estrategiaSeleccionada() }),
      });

      if (!r.ok) {
        let detalle = `El servidor respondió ${r.status}.`;
        try {
          const cuerpo = await r.json();
          if (cuerpo.detail) detalle = cuerpo.detail;
        } catch (e) { /* la respuesta no era JSON */ }
        pintarError(
          detalle + " Eso es un problema del servicio, no significa que no exista información."
        );
        return;
      }

      const datos = await r.json();
      pintarAgente(datos);
      pintarMetricas(datos);
    } catch (e) {
      pintarError("No se pudo contactar al servidor.");
    } finally {
      ocupada = false;
      elementos.enviar.disabled = false;
      elementos.enviar.querySelector(".texto").hidden = false;
      elementos.enviar.querySelector(".cargando").hidden = true;
    }
  }

  // ------------------------------------------------------------------
  // Eventos
  // ------------------------------------------------------------------
  elementos.formulario.addEventListener("submit", function (e) {
    e.preventDefault();
    const texto = elementos.pregunta.value.trim();
    if (!texto) return;

    if (elementos.vacio) elementos.vacio.remove();
    pintarUsuario(texto);
    elementos.pregunta.value = "";
    consultar(texto);
  });

  elementos.ejemplos.addEventListener("change", function () {
    if (elementos.ejemplos.value) {
      elementos.pregunta.value = elementos.ejemplos.value;
      elementos.pregunta.focus();
    }
  });

  // Enter envia, Shift+Enter hace un salto de linea.
  elementos.pregunta.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      elementos.formulario.requestSubmit();
    }
  });

  comprobarEstado();
})();