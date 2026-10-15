"""Benchmark del proveedor de LLM local.

Mide lo que necesita el experimento CAG vs RAG:
  - tiempo de carga en frio del modelo,
  - velocidad de generacion (tokens/seg),
  - velocidad de prefill, que es lo que domina en CAG,
  - latencia con un contexto corto (parecido a RAG) y uno largo (parecido a CAG).

Uso:
    python scripts/benchmark_llm.py
    python scripts/benchmark_llm.py --model qwen2.5:1.5b-instruct --ctx 6000

Los numeros se imprimen en JSON para poder guardarlos en la bitacora.
No inventa resultados: si Ollama no responde, lo dice.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

# Texto sintetico en espanol, del tipo que produce el corpus documental.
# No contiene datos reales: solo sirve para medir volumen de tokens.
_FRAGMENTO = (
    "Para solicitar una partida de nacimiento en el Registro Civil de La Rioja "
    "el interesado deberá presentar documento nacional de identidad, partida de "
    "nacimiento de los padres o tutores, y acreditar el vinculo correspondiente. "
    "La solicitud puede realizarse de forma presencial en el domicilio central o "
    "por medio electronico. El plazo de entrega y el arancel aplicable son los "
    "que constan en la normativa vigente y deberian verificarse en la fuente "
    "oficial antes de informar al ciudadano. "
)

PREGUNTA = (
    "Segun los documentos proporcionados, que documentacion necesito para "
    "solicitar una partida de nacimiento y ante que organismo debo_presentarla?"
)


def _post(url: str, payload: dict, timeout: int = 900) -> dict:
    """POST JSON contra la API de Ollama."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _disponible(base_url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/version", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def _construir_contexto(tokens_objetivo: int) -> str:
    """Genera contexto de aproximadamente `tokens_objetivo` tokens.

    Se estima ~3.6 caracteres por token en espanol. La estimacion es
    aproximada a proposito: lo que se mide es la capacidad del equipo,
    no la precision del conteo.
    """
    caracteres = int(tokens_objetivo * 3.6)
    repetido = (_FRAGMENTO * (caracteres // len(_FRAGMENTO) + 1))[:caracteres]
    return (
        "CONTEXTO DOCUMENTAL:\n"
        "====================\n"
        f"{repetido}\n\n"
        "====================\n"
        "FIN DEL CONTEXTO. Responder usando solo la informacion anterior.\n"
    )


def _medir(
    base_url: str, model: str, contexto: str, max_tokens: int, num_ctx: int
) -> dict:
    """Ejecuta una inferencia y devuelve tiempos y conteos reales."""
    payload = {
        "model": model,
        "prompt": f"{contexto}\n\nPREGUNTA: {PREGUNTA}\n\nRESPUESTA:",
        "stream": False,
        "options": {
            # Determinismo: necesario para que el experimento sea reproducible.
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }

    inicio = time.perf_counter()
    resultado = _post(f"{base_url}/api/generate", payload)
    latencia_ms = (time.perf_counter() - inicio) * 1000

    # Ollama reporta duraciones en nanosegundos.
    tokens_prompt = resultado.get("prompt_eval_count", 0)
    tokens_eval = resultado.get("eval_count", 0)
    dur_prompt_ns = resultado.get("prompt_eval_duration", 0)
    dur_eval_ns = resultado.get("eval_duration", 0)

    return {
        "latencia_ms": round(latencia_ms, 1),
        "tokens_prompt": tokens_prompt,
        "tokens_generados": tokens_eval,
        # Prefill: procesar el contexto. Es lo que domina en CAG.
        "prefill_tok_s": round(tokens_prompt / (dur_prompt_ns / 1e9), 1)
        if dur_prompt_ns
        else None,
        # Generacion: producir la respuesta.
        "generacion_tok_s": round(tokens_eval / (dur_eval_ns / 1e9), 1) if dur_eval_ns else None,
        "tokens_por_segundo_total": round(tokens_prompt / (latencia_ms / 1000), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark del LLM local")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:1.5b-instruct")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--escenarios",
        type=int,
        nargs="+",
        default=[500, 2000, 4000],
        help="Tamanos de contexto a probar, en tokens estimados",
    )
    parser.add_argument(
        "--guardar",
        type=Path,
        default=Path("results/benchmark_llm.json"),
    )
    args = parser.parse_args()

    version = _disponible(args.base_url)
    if version is None:
        print(
            f"ERROR: Ollama no responde en {args.base_url}. "
            "Iniciarlo con `ollama serve` antes de correr el benchmark."
        )
        return 1

    print(f"Ollama {version['version']} | modelo={args.model} | num_ctx={args.num_ctx}")
    print("-" * 72)

    resultados: dict[str, object] = {
        "ollama_version": version["version"],
        "modelo": args.model,
        "num_ctx": args.num_ctx,
        "escenarios": {},
    }

    # --- Carga en frio: la primera llamada incluye leer el modelo a memoria.
    inicio = time.perf_counter()
    try:
        _post(
            f"{args.base_url}/api/generate",
            {
                "model": args.model,
                "prompt": "Responde con una sola palabra: ok",
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 4, "num_ctx": args.num_ctx},
            },
        )
        carga_frio_s = round(time.perf_counter() - inicio, 1)
        resultados["carga_frio_s"] = carga_frio_s
        print(f"Carga en frio (incluye leer el modelo de disco): {carga_frio_s}s")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"ERROR: el modelo '{args.model}' no se pudo cargar: {exc}")
        print("Ejecutar antes:  ollama pull " + args.model)
        return 1

    print("-" * 72)

    for tokens in args.escenarios:
        contexto = _construir_contexto(tokens)
        etiqueta = "RAG-like" if tokens <= 1500 else "CAG-like"
        try:
            m = _medir(args.base_url, args.model, contexto, args.max_tokens, args.num_ctx)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f"  contexto ~{tokens:>5} tokens: ERROR {exc}")
            continue

        m["etiqueta"] = etiqueta
        m["contexto_tokens_estimado"] = tokens
        resultados["escenarios"][str(tokens)] = m

        print(
            f"  contexto ~{tokens:>5} ({etiqueta:>8}): "
            f"lat={m['latencia_ms']:>8.0f}ms  "
            f"prefill={m['tokens_prompt']:>5}tok@{m['prefill_tok_s']:>6}tok/s  "
            f"gen={m['tokens_generados']:>4}tok@{m['generacion_tok_s']:>5}tok/s"
        )

    print("-" * 72)

    # --- Proyeccion del experimento completo: 50 preguntas x 2 estrategias.
    print("Proyeccion del experimento (50 preguntas x 2 estrategias = 100 llamadas):")
    for etiqueta in ("RAG-like", "CAG-like"):
        filas = [m for m in resultados["escenarios"].values() if m.get("etiqueta") == etiqueta]
        if not filas:
            continue
        # Se usa el peor caso medido, para no submestimar.
        peor = max(filas, key=lambda m: m["latencia_ms"])
        total_s = peor["latencia_ms"] * 100 / 1000
        print(
            f"  {etiqueta:>8}: peor caso {peor['latencia_ms'] / 1000:>6.1f}s por llamada "
            f"-> {total_s / 60:>6.1f} min en total"
        )

    args.guardar.parent.mkdir(parents=True, exist_ok=True)
    args.guardar.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nResultado guardado en {args.guardar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())