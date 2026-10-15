"""Balance de una corrida en curso o interrumpida.

El runner guarda `results/informe-parcial.json` despues de cada consulta,
pero si la corrida se corta antes de que llegara esa version, el log de
consola sigue teniendo todo. Este script reconstruye el balance desde ahi.

    python scripts\balance_parcial.py

Se lee el archivo que se indique, o `experimento_out.log` por defecto.
Un informe parcial es un informe real: dice cuantas consultas se hicieron
y cuales, y el titulo lo deja claro para que nadie lo presente como si
fuera la corrida completa.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app.evaluation.dataset import PREGUNTAS_BASE  # noqa: E402

ARCHIVO = RAIZ / (sys.argv[1] if len(sys.argv) > 1 else "experimento_out.log")

log = ARCHIVO.read_text(encoding="utf-8", errors="replace")
prog = re.findall(r"\[(\d+)/50\]\s+(\S+)\s+(CAG|RAG)\s+(ok|MAL|ERROR)", log)
detalle = re.findall(r"\[(\d+)/50\]\s+(\S+)\s+(CAG|RAG)\s+(?:ok|MAL)\s+(\d+)ms in=(\d+)", log)

cats = {p["id"]: p["categoria"] for p in PREGUNTAS_BASE}
d = defaultdict(lambda: {"ok": 0, "mal": 0})
lat = defaultdict(list)
tok = defaultdict(list)
fallos = []

for _, pid, est, marca in prog:
    c = cats.get(pid, "?")
    clave = (c, est)
    if est not in d[c]:
        d[c][est] = {"ok": 0, "mal": 0}
    d[c][est]["ok" if marca == "ok" else "mal"] += 1
    if marca == "MAL":
        fallos.append((pid, est))

for _, _, est, ms, tk in detalle:
    lat[est].append(int(ms))
    tok[est].append(int(tk))


def par(cat, est):
    v = d[cat].get(est, {"ok": 0, "mal": 0})
    t = v["ok"] + v["mal"]
    return f"{v['ok']}/{t}" if t else "-"


hechas = len({pid for _, pid, _, _ in prog})
print(f"RESULTADOS PARCIALES: {hechas} de 50 preguntas, ambas estrategias")
print("=" * 70)
print(f"{'categoria':<20}{'CAG':<14}{'RAG':<14}")
print("-" * 70)
tot = {"CAG": [0, 0], "RAG": [0, 0]}
for cat in ("requisitos", "procedimiento", "multi_seccion"):
    for est in ("CAG", "RAG"):
        v = d[cat].get(est, {"ok": 0, "mal": 0})
        tot[est][0] += v["ok"]
        tot[est][1] += v["ok"] + v["mal"]
    print(f"{cat:<20}{par(cat, 'CAG'):<14}{par(cat, 'RAG'):<14}")
print("-" * 70)
print(
    f"{'TOTAL':<20}{f'{tot["CAG"][0]}/{tot["CAG"][1]}':<14}{f'{tot["RAG"][0]}/{tot["RAG"][1]}':<14}"
)
print()
for est in ("CAG", "RAG"):
    L = sorted(lat[est])
    T = sorted(tok[est])
    if L:
        print(
            f"{est:<5} latencia mediana {L[len(L) // 2]:>6} ms | max {L[-1]:>6} ms"
            f" | tokens mediana {T[len(T) // 2]:>5}"
        )
print()
print(f"consultas donde CAG acerto y RAG no: {sorted({p for p, e in fallos if e == 'RAG'})}")
print(f"consultas donde RAG acerto y CAG no: {sorted({p for p, e in fallos if e == 'CAG'})}")
