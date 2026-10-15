"""Descarga de fuentes oficiales respetando sus condiciones de acceso.

El enunciado pide revisar las politicas de acceso antes de automatizar la
descarga. Este script:

1. Consulta y cachea el `robots.txt` de cada host.
2. Se niega a descargar lo que este prohibido, y lo reporta como bloqueado
   en vez de saltarselo en silencio.
3. Espera entre peticiones (`FETCH_DELAY_S`).
4. Guarda el HTML crudo en `data/raw/` para que el pipeline sea auditable.
5. Respeta una allowlist de rutas: estos sitios exponen endpoints internos
   (`/functions/`, `firebaseConfig.js`) que no son contenido publico.

Uso:
    python scripts/fetch_sources.py --listar
    python scripts/fetch_sources.py
    python scripts/fetch_sources.py --solo partidas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.repositories.document_repository import DocumentoRepository
from app.services.corpus import FuenteDeclarada, nombre_archivo_desde_url

USER_AGENT_POR_DEFECTO = "TramitaIA/0.1 (proyecto academico; uso educativo)"

# ---------------------------------------------------------------------------
# Fuentes del corpus inicial
# ---------------------------------------------------------------------------
# El recorte es deliberado: el enunciado pide limitar el corpus a tramites
# concretos en lugar de cargar toda la normativa provincial.
#
# `robot` queda en None cuando la pagina esta permitida y el sitio no
# tiene robots.txt. `bloqueada_robots` marca las que no se pueden automatizar:
# el enunciado contempla ese caso y pide incorporarlas manualmente
# conservando su URL de origen.
FUENTES: list[dict[str, str | bool | None]] = [
    {
        "tramite": "partidas",
        "url": "https://registrocivil.larioja.gob.ar/tramite-solicitud-partidas.html",
        "organismo": "Registro Civil de La Rioja",
        "robot": None,
        "nota": "Incluye costos citados de la Ley Impositiva 2024.",
    },
    {
        "tramite": "nacimientos",
        "url": "https://registrocivil.larioja.gob.ar/tramite-nacimientos.html",
        "organismo": "Registro Civil de La Rioja",
        "robot": None,
        "nota": None,
    },
    {
        "tramite": "matrimonios",
        "url": "https://registrocivil.larioja.gob.ar/tramite-matrimonios.html",
        "organismo": "Registro Civil de La Rioja",
        "robot": None,
        "nota": None,
    },
    {
        "tramite": "defunciones",
        "url": "https://registrocivil.larioja.gob.ar/tramite-defunciones.html",
        "organismo": "Registro Civil de La Rioja",
        "robot": None,
        "nota": None,
    },
    {
        "tramite": "rectificacion_partidas",
        "url": "https://registrocivil.larioja.gob.ar/tramite-rectificacion-partidas.html",
        "organismo": "Registro Civil de La Rioja",
        "robot": None,
        "nota": None,
    },
    {
        "tramite": "guia_tramites",
        "url": "https://registrocivil.larioja.gob.ar/guia-tramites.html",
        "organismo": "Registro Civil de La Rioja",
        "robot": None,
        "nota": "Pagina indice: sirve para el mapa de enlaces, no para contenido.",
    },
    {
        "tramite": "partidas",
        "url": "https://www.argentina.gob.ar/tramitar",
        "organismo": "Administracion Nacional",
        # Confirmado: el robots.txt declara `Disallow: /tramitar`.
        # No se automatiza la descarga. Puede incorporarse a mano guardando
        # el archivo en data/raw/ y manteniendo la URL en el manifiesto.
        "robot": False,
        "nota": "BLOQUEADA por robots.txt (Disallow: /tramitar). Incorporar manualmente.",
    },
]

# Rutas que nunca deben descargarse aunque una pagina las enlace.
PREFIJOS_PROHIBIDOS = (
    "/functions/",
    "/node_modules/",
    "/rgformulario/",
    "/firebaseConfig",
    "/auth.json",
    "/qa_",
)


def robots_para(url: str, user_agent: str, cache: dict[str, RobotFileParser]) -> RobotFileParser:
    """Devuelve el parser de robots.txt de un host, cacheado por host."""
    partes = urlparse(url)
    origen = f"{partes.scheme}://{partes.netloc}"

    if origen not in cache:
        parser = RobotFileParser()
        parser.set_url(f"{origen}/robots.txt")
        try:
            parser.read()
        except (urllib.error.URLError, OSError, TimeoutError):
            # Sin robots.txt accesible se asume permitido, que es lo que
            # hacen los rastreadores. La lectura igual queda registrada.
            print(f"  robots.txt no disponible en {origen}: se asume permitido")
        cache[origen] = parser

    return cache[origen]


def ruta_prohibida(url: str) -> bool:
    """Comprueba la allowlist de rutas internas del sitio."""
    ruta = urlparse(url).path
    return any(ruta.startswith(prefijo) for prefijo in PREFIJOS_PROHIBIDOS)


def descargar(url: str, destino: Path, user_agent: str, timeout: int = 45) -> int:
    """Descarga una pagina y la guarda. Devuelve los bytes escritos."""
    peticion = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
        # La codificacion se forz a UTF-8: los sitios argentinos suelen
        # declarar iso-8859-1 en el header y el texto sale con mojibake.
        crudo = respuesta.read()
        try:
            texto = crudo.decode("utf-8")
        except UnicodeDecodeError:
            texto = crudo.decode("latin-1")

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    return len(texto)


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga de fuentes de TramitaIA")
    parser.add_argument("--listar", action="store_true", help="Muestra las fuentes y sale")
    parser.add_argument("--solo", nargs="*", help="Tramites a incluir")
    parser.add_argument("--forzar", action="store_true", help="Redescarga aunque exista")
    args = parser.parse_args()

    settings = get_settings()
    user_agent = settings.fetch_user_agent or USER_AGENT_POR_DEFECTO
    respeta_robots = settings.fetch_respect_robots

    fuentes = FUENTES
    if args.solo:
        fuentes = [f for f in fuentes if f["tramite"] in args.solo]

    if args.listar:
        print(f"{'TRAMITE':<24} {'ROBOTS':<8} URL")
        print("-" * 100)
        for fuente in fuentes:
            veredicto = "BLOQUEADO" if fuente["robot"] is False else "permitido"
            print(f"{fuente['tramite']:<24} {veredicto:<8} {fuente['url']}")
        return 0

    cache_robots: dict[str, RobotFileParser] = {}
    repositorio = DocumentoRepository(settings.processed_dir)

    descargados = 0
    bloqueados = 0
    fallos = 0

    for fuente in fuentes:
        url = str(fuente["url"])
        tramite = str(fuente["tramite"])

        # Allowlist de rutas internas: se comprueba antes de robots porque
        # es una restriccion propia del proyecto.
        if ruta_prohibida(url):
            print(f"[OMITIDA] {url}  (ruta interna, fuera del contenido publico)")
            bloqueados += 1
            continue

        if respeta_robots:
            parser = robots_para(url, user_agent, cache_robots)
            permitido = parser.can_fetch(user_agent, url)
        else:
            permitido = True

        if not permitido:
            print(f"[BLOQUEADA] {url}  (robots.txt no lo permite)")
            repositorio_cargado = ManifiestoProxy(repositorio)
            repositorio_cargado.marcar_bloqueada(url, str(fuente.get("nota") or ""))
            bloqueados += 1
            time.sleep(settings.fetch_delay_s)
            continue

        destino = settings.raw_dir / nombre_archivo_desde_url(url)
        if destino.exists() and not args.forzar:
            print(f"[CACHE]    {url}  (ya existe, usar --forzar para repetir)")

        print(f"[BAJA]     {url}")
        try:
            bytes_escritos = descargar(url, destino, user_agent)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f"  ERROR al descargar: {exc}")
            fallos += 1
            time.sleep(settings.fetch_delay_s)
            continue

        try:
            declarada = FuenteDeclarada(
                url=url,
                tramite=tramite,
                organismo=str(fuente["organismo"]),
                observaciones=str(fuente["nota"]) if fuente.get("nota") else None,
                fecha_consulta=date.today(),
            )
            metadatos = repositorio.procesar_archivo(destino, declarada)
            print(
                f"  OK: {metadatos.document_id}  "
                f"{len(metadatos.titulo[:52])}  "
                f"actualizacion={metadatos.fecha_actualizacion or 'desconocida'}  "
                f"vigencia={metadatos.vigencia.value}  "
                f"hash={metadatos.hash_contenido[:12]}  {bytes_escritos}b"
            )
            descargados += 1
        except Exception as exc:
            print(f"  ERROR al ingerir: {exc}")
            fallos += 1

        # Espera entre peticiones: politely crawling, y ademas lo exigen
        # politicas como la de argentina.gob.ar (Crawl-delay: 10).
        time.sleep(settings.fetch_delay_s)

    print("-" * 60)
    print(f"descargados={descargados}  bloqueados={bloqueados}  fallos={fallos}")

    informe = repositorio.informe()
    print("\nEstado del corpus:")
    print(json.dumps(informe, indent=2, ensure_ascii=False))
    return 1 if fallos else 0


class ManifiestoProxy:
    """Envoltura minima para marcar fuentes bloqueadas sin repetir imports."""

    def __init__(self, repositorio: DocumentoRepository) -> None:
        self.repositorio = repositorio

    def marcar_bloqueada(self, url: str, nota: str) -> None:
        from app.models.schemas import EstadoFuente
        from app.services.corpus import Manifiesto

        manifiesto = Manifiesto.cargar(self.repositorio.ruta_manifiesto)
        if manifiesto.buscar_por_url(url) is None:
            manifiesto.registrar(
                {
                    "url": url,
                    "titulo": None,
                    "organismo": None,
                    "tramite": None,
                    "estado_verificacion": EstadoFuente.NO_ACCESIBLE.value,
                    "fecha_consulta": date.today().isoformat(),
                    "fecha_ultima_revision": date.today().isoformat(),
                    "fecha_incorporacion": date.today().isoformat(),
                    "observaciones": nota or "No se pudo acceder automaticamente.",
                }
            )
            manifiesto.marcar_estado(url, EstadoFuente.NO_ACCESIBLE, nota)
            manifiesto.guardar(self.repositorio.ruta_manifiesto)


if __name__ == "__main__":
    raise SystemExit(main())
