"""
data_extraction/ebsco_scraper.py — Scraper directo para EBSCO Discovery Service

Conexión directa al Descubridor EDS de la biblioteca de la Universidad del
Quindío (https://library.uniquindio.edu.co/databases) usando Playwright.

Flujo:
  1. Abre un navegador Chromium con contexto persistente (cookies guardadas).
  2. Navega a la URL de búsqueda EBSCO vía el proxy institucional.
  3. Si detecta el muro de login (IntelProxy), espera login manual del usuario.
  4. Extrae los resultados de búsqueda de la página de EBSCO EDS.
  5. Pagina hasta alcanzar max_results.
  6. Retorna un pd.DataFrame con STANDARD_COLUMNS.

Requisitos:
  pip install playwright
  playwright install chromium
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

try:
    from playwright.sync_api import sync_playwright, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
    
    # En entornos como Streamlit Cloud, Playwright necesita descargar el binario del navegador
    # Ejecutamos playwright install chromium automatizadamente para asegurar que exista
    os.system("playwright install chromium")
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from config import (
    DATA_DIR,
    DEFAULT_QUERY,
    MAX_RESULTS_PER_SOURCE,
    STANDARD_COLUMNS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Constantes EBSCO / UniQuindío ────────────────────────────────────────────

SESSION_DIR = DATA_DIR / "ebsco_session"

# Parámetros institucionales
EBSCO_CUSTID = "ns004363"
EBSCO_GROUPID = "main"
EBSCO_PROFILE = "eds"
EBSCO_AUTHTYPE = "ip,uid"

# URLs base
PROXY_BASE = "https://login.crai.referencistas.com/login?url="
EBSCO_SEARCH_BASE = "https://search.ebscohost.com/login.aspx"


class EBSCOScraper:
    """
    Scraper para EBSCO Discovery Service vía la biblioteca UniQuindío.

    Usa launch_persistent_context con limpieza automática de SingletonLock
    para permitir múltiples ejecuciones en la misma sesión de la app.
    """

    def __init__(
        self,
        query: str = DEFAULT_QUERY,
        max_results: int = MAX_RESULTS_PER_SOURCE,
        headless: bool = False,
        progress_callback=None,
        reset_session: bool = False,
    ):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright no está instalado. Ejecute:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )
        self.query = query
        self.max_results = max_results
        self.headless = headless
        self.progress_callback = progress_callback
        self.reset_session = reset_session

        # Limpiar sesión anterior si se solicita
        if self.reset_session and SESSION_DIR.exists():
            try:
                shutil.rmtree(SESSION_DIR)
                logger.info("Sesión de EBSCO eliminada (reset_session=True)")
            except Exception as e:
                logger.warning(f"No se pudo eliminar la sesión: {e}")

        SESSION_DIR.mkdir(parents=True, exist_ok=True)

    def _build_search_url(self) -> str:
        """Construye la URL de búsqueda EBSCO con proxy institucional."""
        ebsco_params = (
            f"direct=true"
            f"&authtype={EBSCO_AUTHTYPE}"
            f"&custid={EBSCO_CUSTID}"
            f"&groupid={EBSCO_GROUPID}"
            f"&profile={EBSCO_PROFILE}"
            f"&db={EBSCO_PROFILE}"
            f"&bquery={quote_plus(self.query)}"
        )
        target_url = f"{EBSCO_SEARCH_BASE}?{ebsco_params}"
        return f"{PROXY_BASE}{target_url}"

    def _cleanup_lock_files(self) -> None:
        """
        Elimina los archivos de bloqueo de Chromium que impiden
        abrir una segunda instancia con el mismo directorio de sesión.
        """
        lock_files = ["SingletonLock", "SingletonSocket", "SingletonCookie"]
        for name in lock_files:
            lock_path = SESSION_DIR / name
            if lock_path.exists():
                try:
                    lock_path.unlink()
                    logger.info(f"Lock eliminado: {name}")
                except Exception as e:
                    logger.warning(f"No se pudo eliminar {name}: {e}")

    # ── Detección de estado ────────────────────────────────────────────────────

    # URLs que indican que estamos en un muro de login ACTIVO (esperando credenciales)
    _LOGIN_INDICATORS = [
        "login.intelproxy.com/v2/inicio",    # Landing del proxy
        "accounts.google.com/",              # Login de Google
        "login.ebsco.com",                   # Login directo EBSCO
        "login.microsoftonline.com",         # Login Microsoft
        "myaccount.google.com",              # Verificación Google
    ]

    def _is_on_ebsco(self, page: Page) -> bool:
        """Detecta si estamos en el dominio de EBSCO (resultados)."""
        url = page.url.lower()
        # Patrones directos de EBSCO
        if "research-ebsco-com" in url:
            return True
        if "research.ebsco.com" in url:
            return True
        if "eds.p.ebscohost.com" in url:
            return True
        if "eds.s.ebscohost.com" in url:
            return True
        if "ebscohost.com" in url and "login.ebsco" not in url:
            return True
        if "/search/results" in url and "ebsco" in url:
            return True
        # Proxy ya redirigió al resultado final
        if "referencistas.com" in url and "login" not in url:
            return True
        return False

    def _is_on_login_page(self, page: Page) -> bool:
        """Detecta si estamos en un muro de autenticación activo."""
        url = page.url.lower()
        return any(indicator in url for indicator in self._LOGIN_INDICATORS)

    def _needs_login(self, page: Page) -> bool:
        """Detecta si estamos en un muro de autenticación REAL."""
        if self._is_on_ebsco(page):
            return False
        return self._is_on_login_page(page)

    # ── Manejo de login y redirects ────────────────────────────────────────────

    def _wait_for_redirect(self, page: Page, timeout_seconds: int = 25) -> str:
        """
        Espera a que la cadena de redirects del proxy termine.

        Returns:
            'ebsco' | 'login' | 'unknown'
        """
        logger.info("Esperando redirecciones del proxy...")
        start = time.time()
        while time.time() - start < timeout_seconds:
            try:
                if self._is_on_ebsco(page):
                    logger.info(f"En EBSCO: {page.url[:80]}")
                    return "ebsco"
                if self._needs_login(page):
                    logger.info(f"Login requerido: {page.url[:80]}")
                    return "login"
            except Exception:
                pass
            page.wait_for_timeout(1000)
        logger.warning(f"Redirect no resuelto en {timeout_seconds}s. URL: {page.url[:80]}")
        return "unknown"

    def _wait_for_manual_login(self, page: Page, timeout_seconds: int = 300) -> bool:
        """
        Espera a que el usuario complete el login y EBSCO cargue.
        
        Solo retorna True cuando _is_on_ebsco() detecta que estamos
        en la página de resultados de EBSCO. Sin atajos ni detección
        indirecta — esperamos a estar realmente en EBSCO.
        """
        # Traer el navegador al frente
        try:
            page.bring_to_front()
        except Exception:
            pass

        logger.info(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  LOGIN MANUAL REQUERIDO                                     ║\n"
            "║                                                             ║\n"
            "║  Se abrió un navegador Chromium. Por favor:                 ║\n"
            "║  1. Inicie sesión con su cuenta Google institucional        ║\n"
            "║  2. Complete cualquier verificación 2FA si aparece          ║\n"
            "║  3. Espere a que aparezcan los resultados de búsqueda      ║\n"
            "║                                                             ║\n"
            "║  La sesión se guardará para futuras ejecuciones.           ║\n"
            "║  Timeout: %d segundos                                      ║\n"
            "╚══════════════════════════════════════════════════════════════╝",
            timeout_seconds,
        )
        start = time.time()
        seen_urls = set()
        
        while time.time() - start < timeout_seconds:
            try:
                for ctx_page in page.context.pages:
                    try:
                        current_url = ctx_page.url
                    except Exception:
                        continue

                    # Logear URL cuando es nueva (para diagnóstico)
                    if current_url not in seen_urls:
                        elapsed = int(time.time() - start)
                        logger.info(f"[{elapsed}s] Tab URL: {current_url[:120]}")
                        seen_urls.add(current_url)

                    if self._is_on_ebsco(ctx_page):
                        logger.info(f"Login exitoso — EBSCO detectado: {current_url[:100]}")
                        return True
            except Exception as e:
                logger.debug(f"Error polling: {e}")
            page.wait_for_timeout(2000)

        logger.error("Timeout: No se completó el login en %d segundos.", timeout_seconds)
        return False

    # ── Espera de resultados ──────────────────────────────────────────────────

    def _wait_for_results_load(self, page: Page, timeout: int = 30000) -> bool:
        """Espera a que los resultados de búsqueda carguen."""
        selectors = [
            "a[class*='result-item-title']",
            "[class*='result-item-title-new']",
            "h3 a[class*='title']",
            "a[class*='title']",
        ]
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=timeout)
                logger.info(f"Resultados detectados: {sel}")
                return True
            except Exception:
                continue
        logger.warning("No se detectaron resultados.")
        return False

    # ── Scraping de resultados ────────────────────────────────────────────────

    def _scrape_current_page(self, page: Page) -> list[list]:
        """Extrae los registros de la página actual."""
        records = []

        if not self._wait_for_results_load(page):
            return records
        time.sleep(2)

        # Buscar todos los títulos
        title_links = page.query_selector_all("a[class*='result-item-title']")
        if not title_links:
            title_links = page.query_selector_all("h3 a")
        if not title_links:
            all_links = page.query_selector_all("a[class*='title']")
            title_links = [l for l in all_links if len((l.inner_text() or "").strip()) > 20]

        logger.info(f"Encontrados {len(title_links)} títulos.")

        for title_el in title_links:
            try:
                record = self._parse_result_from_title(title_el, page)
                if record:
                    records.append(record)
            except Exception as e:
                logger.debug(f"Error parseando: {e}")
        return records

    def _find_result_container(self, title_el):
        """Sube por el DOM para encontrar el contenedor del resultado."""
        try:
            container = title_el.evaluate_handle(
                """el => {
                    let node = el.parentElement;
                    for (let i = 0; i < 6; i++) {
                        if (!node) break;
                        if (node.children.length >= 3) return node;
                        node = node.parentElement;
                    }
                    return el.parentElement?.parentElement || el.parentElement;
                }"""
            )
            return container.as_element()
        except Exception:
            return None

    def _parse_result_from_title(self, title_el, page: Page) -> list | None:
        """Parsea un resultado EBSCO."""
        title = (title_el.inner_text() or "").strip()
        if not title or len(title) < 5:
            return None

        container = self._find_result_container(title_el)
        if not container:
            return [f"EBSCO_{hash(title)}", title, "", "", "", "", "", "", "EBSCO", "", 0, ""]

        container_text = ""
        try:
            container_text = (container.inner_text() or "")
        except Exception:
            pass

        authors = self._extract_authors(container)
        year = ""
        m = re.search(r"\b(19|20)\d{2}\b", container_text)
        if m:
            year = m.group(0)
        journal = self._extract_journal(container, container_text)
        abstract = self._extract_abstract(container, container_text)
        doi = self._extract_doi(container, container_text)
        keywords = self._extract_keywords(container)
        country = self._extract_country(container, container_text)

        record_id = f"EBSCO_{doi}" if doi else f"EBSCO_{hash(title)}"
        url = f"https://doi.org/{doi}" if doi else ""

        return [
            record_id, title, authors, year, abstract, keywords,
            journal, doi, "EBSCO", country, 0, url,
        ]

    def _extract_authors(self, container) -> str:
        try:
            for sel in [
                "a[class*='delimited-link-item']",
                "a[class*='metadata-content']",
                "a[class*='people-page-link']",
                "a[class*='author']",
            ]:
                els = container.query_selector_all(sel)
                if els:
                    names = []
                    for a in els[:10]:
                        name = (a.inner_text() or "").strip()
                        if name and not name.startswith("http") and len(name) < 80:
                            names.append(name)
                    if names:
                        return "; ".join(names)
        except Exception:
            pass
        return ""

    def _extract_journal(self, container, text: str) -> str:
        try:
            match = re.search(r"(?:En|In)\s*:\s*(.+?)(?:\.\s|\n|$)", text)
            if match:
                j = match.group(1).strip()
                return re.split(r"\s*,\s*vol\b|\s*,\s*Vol\b|\s*\d{4}\s*,", j)[0].strip()
        except Exception:
            pass
        try:
            for sel in ["[class*='source']", "[class*='journal']"]:
                el = container.query_selector(sel)
                if el:
                    t = (el.inner_text() or "").strip()
                    if t:
                        return t
        except Exception:
            pass
        return ""

    def _extract_abstract(self, container, container_text: str) -> str:
        try:
            blocks = container.query_selector_all("div, p, span")
            for b in blocks:
                t = (b.inner_text() or "").strip()
                if len(t) > 100 and t != container_text:
                    if not t.startswith("En:") and "Mostrar" not in t[:20]:
                        return t
        except Exception:
            pass
        return ""

    def _extract_doi(self, container, text: str) -> str:
        try:
            els = container.query_selector_all("a[href*='doi.org']")
            if els:
                href = els[0].get_attribute("href") or ""
                if "doi.org/" in href:
                    return href.split("doi.org/")[-1]
        except Exception:
            pass
        m = re.search(r"10\.\d{4,}/[^\s,;]+", text)
        if m:
            return m.group(0).rstrip(".")
        return ""

    def _extract_keywords(self, container) -> str:
        try:
            for sel in ["a[class*='subjects-content']", "a[class*='subject']", "[class*='keyword']"]:
                els = container.query_selector_all(sel)
                if els:
                    kws = [(k.inner_text() or "").strip() for k in els[:15]]
                    kws = [k for k in kws if k]
                    if kws:
                        return "; ".join(kws)
        except Exception:
            pass
        return ""

    def _extract_country(self, container, text: str) -> str:
        """Extrae país/lugar de publicación."""
        try:
            for sel in ["[class*='publisher']", "[class*='location']", "[class*='affiliation']"]:
                el = container.query_selector(sel)
                if el:
                    t = (el.inner_text() or "").strip()
                    if t:
                        return t
        except Exception:
            pass
        try:
            m = re.search(r"(?:Publisher|Editorial|Publicado por)\s*:\s*(.+?)(?:\n|$)", text)
            if m:
                return m.group(1).strip()
            m = re.search(r"(?:Published|Publicado)\s+(?:in|en)\s+(.+?)(?:\.|,|\n|$)", text, re.I)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
        return ""

    # ── Paginación ────────────────────────────────────────────────────────────

    def _load_more_results(self, page: Page) -> bool:
        """
        Intenta cargar más resultados clickeando 'Mostrar más resultados'.

        El botón está dentro de un scroll-container, así que scrolleamos
        al fondo primero y luego usamos JavaScript click para evitar
        problemas de visibilidad.
        """
        # Primero, scrollear al fondo de la página/scroll-container
        try:
            page.evaluate("""
                const scrollContainers = document.querySelectorAll('[class*="scroll-container"]');
                if (scrollContainers.length > 0) {
                    const container = scrollContainers[scrollContainers.length - 1];
                    container.scrollTop = container.scrollHeight;
                }
                window.scrollTo(0, document.body.scrollHeight);
            """)
            time.sleep(2)
        except Exception:
            pass

        # Contar resultados antes
        count_before = len(page.query_selector_all("a[class*='result-item-title'], a[class*='title']"))

        # Intentar encontrar y clickear el botón de paginación
        selectors = [
            "button.eb-pagination__button",
            "button[class*='pagination']",
        ]

        for sel in selectors:
            try:
                btn = page.query_selector(sel)
                if btn:
                    # Usar JavaScript click para evitar problemas de visibilidad
                    btn.evaluate("el => el.scrollIntoView({block: 'center'})")
                    time.sleep(0.5)
                    btn.evaluate("el => el.click()")
                    logger.info(f"Click en botón de paginación: {sel}")

                    # Esperar a que carguen nuevos resultados
                    time.sleep(5)
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        pass

                    count_after = len(page.query_selector_all("a[class*='result-item-title'], a[class*='title']"))
                    if count_after > count_before:
                        logger.info(f"Más resultados cargados: {count_before} → {count_after}")
                        return True
                    else:
                        logger.info(f"Botón clickeado pero sin nuevos resultados ({count_before} → {count_after})")
                        return False
            except Exception as e:
                logger.debug(f"Paginación ({sel}): {e}")
                continue

        # Fallback: intentar con Playwright locator (más resiliente)
        try:
            locator = page.locator("text=Mostrar más resultados").first
            if locator.count() > 0:
                locator.scroll_into_view_if_needed()
                time.sleep(0.5)
                locator.click(force=True)
                logger.info("Click via locator: 'Mostrar más resultados'")
                time.sleep(5)
                count_after = len(page.query_selector_all("a[class*='result-item-title'], a[class*='title']"))
                if count_after > count_before:
                    logger.info(f"Más resultados: {count_before} → {count_after}")
                    return True
        except Exception as e:
            logger.debug(f"Locator fallback: {e}")

        logger.info("No hay más resultados para cargar.")
        return False

    # ── Espera post-login ─────────────────────────────────────────────────────

    def _wait_for_ebsco_page(
        self, context: "BrowserContext", fallback_page: "Page", timeout: int = 60
    ) -> "Page":
        """
        Después del login, espera a que alguna tab del contexto esté en EBSCO.

        El proxy IntelProxy redirecciona varias veces: callback → CRAI → EBSCO.
        Esta función espera activamente hasta que la página EBSCO aparezca.
        """
        logger.info("Esperando a que EBSCO cargue tras login...")
        start = time.time()
        while time.time() - start < timeout:
            for ctx_page in context.pages:
                try:
                    if self._is_on_ebsco(ctx_page):
                        logger.info(f"EBSCO encontrado: {ctx_page.url[:100]}")
                        return ctx_page
                except Exception:
                    continue
            fallback_page.wait_for_timeout(2000)
        # Si no se encontró, intentar con la página actual (quizá cargó tarde)
        logger.warning(
            f"No se encontró EBSCO tras {timeout}s. "
            f"Usando página actual: {fallback_page.url[:80]}"
        )
        return fallback_page

    # ── Flujo principal ───────────────────────────────────────────────────────

    def fetch(self) -> pd.DataFrame:
        """
        Ejecuta el scraping completo de EBSCO EDS.

        Usa launch_persistent_context con limpieza de SingletonLock para
        permitir múltiples ejecuciones en la misma sesión.
        """
        logger.info(f"Iniciando scraper EBSCO — query: '{self.query}', max: {self.max_results}")

        # Limpiar locks de ejecuciones anteriores
        self._cleanup_lock_files()

        all_records = []

        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(SESSION_DIR),
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
                locale="es-CO",
                timezone_id="America/Bogota",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )

            # Usar la primera página si existe, o crear una nueva
            page = context.pages[0] if context.pages else context.new_page()

            try:
                search_url = self._build_search_url()
                logger.info(f"Navegando a: {search_url[:100]}...")
                page.goto(search_url, wait_until="networkidle", timeout=60000)

                # Esperar a que terminen las redirecciones
                state = self._wait_for_redirect(page, timeout_seconds=25)

                if state == "login":
                    logger.info("Login requerido. Esperando login manual...")
                    login_ok = self._wait_for_manual_login(page, timeout_seconds=300)
                    if not login_ok:
                        logger.error("No se completó el login. Abortando.")
                        context.close()
                        return pd.DataFrame(columns=STANDARD_COLUMNS)
                    # Después del login, esperar a que EBSCO cargue en alguna tab
                    page = self._wait_for_ebsco_page(context, page)
                elif state == "unknown":
                    logger.info("Estado desconocido. Esperando 10s...")
                    page.wait_for_timeout(10000)
                    if not self._is_on_ebsco(page):
                        login_ok = self._wait_for_manual_login(page, timeout_seconds=300)
                        if not login_ok:
                            context.close()
                            return pd.DataFrame(columns=STANDARD_COLUMNS)
                        page = self._wait_for_ebsco_page(context, page)

                # En EBSCO → esperar resultados
                logger.info(f"En EBSCO. URL: {page.url[:80]}")
                self._wait_for_results_load(page, timeout=30000)

                # Scraping
                iteration = 0
                max_iterations = 20

                while len(all_records) < self.max_results and iteration < max_iterations:
                    iteration += 1
                    logger.info(
                        f"Iteración {iteration} — "
                        f"Acumulados: {len(all_records)}/{self.max_results}"
                    )

                    records = self._scrape_current_page(page)
                    if not records:
                        logger.info("Sin resultados. Fin.")
                        break

                    existing_titles = {r[1] for r in all_records}
                    new_records = [r for r in records if r[1] not in existing_titles]

                    if not new_records:
                        logger.info("Sin resultados nuevos. Fin.")
                        break

                    all_records.extend(new_records)
                    logger.info(f"Iteración {iteration}: {len(new_records)} nuevos (total: {len(all_records)})")

                    if self.progress_callback:
                        self.progress_callback("ebsco", len(all_records))

                    if len(all_records) >= self.max_results:
                        break

                    if not self._load_more_results(page):
                        break
                    page.wait_for_timeout(2000)

            except Exception as e:
                logger.error(f"Error durante scraping EBSCO: {e}")
                import traceback
                traceback.print_exc()

            finally:
                try:
                    context.close()
                except Exception:
                    pass  # Navegador ya cerrado

        all_records = all_records[:self.max_results]
        df = pd.DataFrame(all_records, columns=STANDARD_COLUMNS)
        logger.info(f"EBSCO scraping completado: {len(df)} registros.")
        return df


# ─── Ejecución directa (para pruebas) ────────────────────────────────────────
if __name__ == "__main__":
    scraper = EBSCOScraper(
        query=DEFAULT_QUERY,
        max_results=10,
        headless=False,
    )
    df = scraper.fetch()
    print(f"\n{'='*60}")
    print(f"Resultados obtenidos: {len(df)}")
    print(f"{'='*60}")
    if not df.empty:
        print(df[["title", "authors", "year", "source_db"]].to_string())
    else:
        print("No se obtuvieron resultados.")
