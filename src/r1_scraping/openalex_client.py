"""
Cliente HTTP para la API de OpenAlex.

Características:
  - Paginación automática por cursor (OpenAlex no usa ?page=N).
  - Pertenencia al "polite pool" de OpenAlex mediante el parámetro
    mailto, lo que garantiza mejor tasa de respuesta y estabilidad.
  - Almacenamiento incremental en JSONL: cada artículo es una línea JSON
    independiente; permite reanudar búsquedas y evita saturar la RAM
    con corpus grandes.
  - Generador de progreso compatible con Streamlit (yield de estado).

Uso básico (fuera de Streamlit):
    from src.r1_scraping.openalex_client import OpenAlexClient

    client = OpenAlexClient()
    records = client.fetch_all("generative artificial intelligence", max_results=500)
    # records es una lista de dicts con el JSON crudo de OpenAlex

Uso con almacenamiento JSONL:
    client = OpenAlexClient()
    client.fetch_and_save(
        query="generative artificial intelligence",
        output_path="data/raw/openalex_raw.jsonl",
        max_results=1000,
    )

Referencias:
  https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging
  https://docs.openalex.org/api-entities/works
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Generator, Iterator
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL        = "https://api.openalex.org/works"
DEFAULT_EMAIL   = "calderon@vivexia.co"   # polite pool de OpenAlex
PER_PAGE        = 200                      # máximo permitido por OpenAlex
DEFAULT_MAX     = 1000
RETRY_ATTEMPTS  = 3
RETRY_DELAY_S   = 2.0                      # segundos entre reintentos

# ─────────────────────────────────────────────────────────────────────────────
# Cliente principal
# ─────────────────────────────────────────────────────────────────────────────

class OpenAlexClient:
    """
    Cliente para la API de OpenAlex con soporte de paginación por cursor
    y almacenamiento incremental en JSONL.
    """

    def __init__(self, email: str = DEFAULT_EMAIL, per_page: int = PER_PAGE):
        """
        Parameters
        ----------
        email    : Correo para el polite pool de OpenAlex.  Mejora la
                   prioridad y el rate limit de las peticiones.
        per_page : Registros por página (máx 200).
        """
        self.email    = email
        self.per_page = min(per_page, 200)
        self.session  = requests.Session()
        self.session.headers.update({
            "User-Agent": f"bibliometrics-app/1.0 (mailto:{email})",
            "Accept": "application/json",
        })

    # ── API pública ──────────────────────────────────────────────────────────

    def fetch_all(
        self,
        query: str,
        max_results: int = DEFAULT_MAX,
        extra_filters: str | None = None,
    ) -> list[dict]:
        """
        Descarga hasta ``max_results`` artículos y los devuelve en memoria.

        Parameters
        ----------
        query         : Término de búsqueda (e.g. "generative artificial intelligence").
        max_results   : Límite máximo de registros a descargar.
        extra_filters : Filtros adicionales de OpenAlex en formato filter string
                        (e.g. "publication_year:>2019").

        Returns
        -------
        Lista de dicts con el JSON crudo de OpenAlex (objetos Work).
        """
        results: list[dict] = []
        for batch, _total, _cursor in self._paginate(query, max_results, extra_filters):
            results.extend(batch)
        return results

    def fetch_and_save(
        self,
        query: str,
        output_path: str | Path,
        max_results: int = DEFAULT_MAX,
        extra_filters: str | None = None,
        overwrite: bool = False,
    ) -> int:
        """
        Descarga artículos y los guarda de forma incremental en un archivo JSONL.

        Cada línea del archivo resultante es un objeto JSON completo (un Work).
        Si el archivo ya existe y ``overwrite=False``, se añaden los registros
        al final (modo append).

        Parameters
        ----------
        query         : Término de búsqueda.
        output_path   : Ruta del archivo JSONL de salida.
        max_results   : Límite máximo de registros.
        extra_filters : Filtros adicionales de OpenAlex.
        overwrite     : Si True, sobreescribe el archivo existente.

        Returns
        -------
        Número total de registros guardados en esta llamada.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "w" if overwrite else "a"
        saved = 0

        with path.open(mode, encoding="utf-8") as fh:
            for batch, _total, _cursor in self._paginate(query, max_results, extra_filters):
                for record in batch:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += len(batch)

        logger.info("fetch_and_save: %d registros guardados en %s", saved, path)
        return saved

    def fetch_with_progress(
        self,
        query: str,
        max_results: int = DEFAULT_MAX,
        extra_filters: str | None = None,
    ) -> Generator[tuple[list[dict], int, int], None, None]:
        """
        Generador que yield (batch, total_descargados, total_estimado).

        Diseñado para integrarse con la barra de progreso de Streamlit::

            for batch, downloaded, total in client.fetch_with_progress(query, 500):
                progress_bar.progress(downloaded / max(total, 1))
                status_text.text(f"Descargados {downloaded} de ~{total}")

        Yields
        ------
        (batch, total_descargados, total_estimado)
        """
        downloaded = 0
        for batch, total_estimated, _cursor in self._paginate(query, max_results, extra_filters):
            downloaded += len(batch)
            yield batch, downloaded, total_estimated

    # ── Paginación interna ───────────────────────────────────────────────────

    def _paginate(
        self,
        query: str,
        max_results: int,
        extra_filters: str | None,
    ) -> Iterator[tuple[list[dict], int, str | None]]:
        """
        Itera sobre las páginas de la API de OpenAlex usando cursor paging.

        OpenAlex devuelve un campo ``next_cursor`` en cada respuesta que
        se usa como parámetro ``cursor`` en la siguiente petición.
        El cursor inicial es ``"*"``.

        Yields
        ------
        (batch_records, total_count, next_cursor)
        """
        cursor      = "*"
        total_fetched = 0

        while True:
            remaining = max_results - total_fetched
            if remaining <= 0:
                break

            page_size = min(self.per_page, remaining)

            try:
                data, next_cursor = self._fetch_page(query, cursor, page_size, extra_filters)
            except requests.RequestException as exc:
                logger.error("Error en petición OpenAlex: %s", exc)
                raise

            records = data.get("results") or []
            total_count = (data.get("meta") or {}).get("count", 0)

            if not records:
                break

            yield records, total_count, next_cursor
            total_fetched += len(records)

            if not next_cursor:
                break

            cursor = next_cursor

    def _fetch_page(
        self,
        query: str,
        cursor: str,
        per_page: int,
        extra_filters: str | None,
    ) -> tuple[dict, str | None]:
        """
        Realiza una petición a un solo página de la API.

        Incluye reintentos automáticos en caso de error transitorio
        (HTTP 429, 500, 503).

        Returns
        -------
        (response_dict, next_cursor_or_None)
        """
        params: dict[str, str | int] = {
            "search":   query,
            "per-page": per_page,
            "cursor":   cursor,
            "mailto":   self.email,
        }
        if extra_filters:
            params["filter"] = extra_filters

        url = f"{BASE_URL}?{urlencode(params)}"
        logger.debug("GET %s", url)

        last_exc: Exception | None = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                next_cursor = (data.get("meta") or {}).get("next_cursor")
                return data, next_cursor

            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status in (429, 500, 503) and attempt < RETRY_ATTEMPTS:
                    wait = RETRY_DELAY_S * attempt
                    logger.warning(
                        "HTTP %d en intento %d/%d — esperando %.1fs",
                        status, attempt, RETRY_ATTEMPTS, wait,
                    )
                    time.sleep(wait)
                    last_exc = exc
                    continue
                raise

            except requests.RequestException as exc:
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_DELAY_S * attempt)
                    last_exc = exc
                    continue
                raise

        raise last_exc  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades JSONL
# ─────────────────────────────────────────────────────────────────────────────

def read_jsonl(path: str | Path) -> list[dict]:
    """
    Lee un archivo JSONL y devuelve la lista de objetos.

    Parameters
    ----------
    path : Ruta al archivo .jsonl

    Returns
    -------
    Lista de dicts (un dict por línea del archivo).
    Líneas vacías o mal formadas son ignoradas con advertencia.
    """
    path = Path(path)
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("JSONL línea %d ignorada (%s): %s", lineno, path, exc)

    logger.debug("read_jsonl: %d registros leídos desde %s", len(records), path)
    return records


def read_jsonl_slice(
    path: str | Path,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[dict], int]:
    """
    Lee una porción de un archivo JSONL para paginación en UI.

    Parameters
    ----------
    path   : Ruta al archivo .jsonl
    offset : Línea desde la que empezar (0-based).
    limit  : Número máximo de registros a devolver.

    Returns
    -------
    (slice_records, total_count)
    """
    path = Path(path)
    if not path.exists():
        return [], 0

    all_records = read_jsonl(path)
    total = len(all_records)
    return all_records[offset: offset + limit], total
