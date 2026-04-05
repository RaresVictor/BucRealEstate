"""
Storia.ro apartment scraper for Bucharest.

Primary strategy: parse the __NEXT_DATA__ JSON embedded in every page
(stable, clean, avoids brittle CSS selectors).
Fallback: BeautifulSoup HTML extraction if JSON is absent.
"""

import json
import logging
import random
import re
import time
from urllib.parse import urljoin

import cloudscraper
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.storia.ro/ro/rezultate/vanzare/apartament/bucuresti?page={}"
BASE_DOMAIN = "https://www.storia.ro"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
}

_scraper = None


def _get_scraper() -> cloudscraper.CloudScraper:
    global _scraper
    if _scraper is None:
        _scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        _scraper.headers.update(HEADERS)
    return _scraper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_listing_urls(page_num: int) -> list[str]:
    """Return unique offer URLs found on the given listing page."""
    url = BASE_URL.format(page_num)
    scraper = _get_scraper()
    try:
        response = scraper.get(url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to fetch listing page {page_num}: {e}")
        return []

    # Try to extract URLs from __NEXT_DATA__ first (faster, no HTML parsing)
    urls = _urls_from_next_data(response.text)

    # Fallback: scan all anchors
    if not urls:
        soup = BeautifulSoup(response.text, "html.parser")
        seen: set[str] = set()
        for tag in soup.find_all("a", href=True):
            href: str = tag["href"]
            if "/ro/oferta/" in href:
                full = urljoin(BASE_DOMAIN, href).split("?")[0]
                if full not in seen:
                    seen.add(full)
                    urls.append(full)

    logger.info(f"Page {page_num}: found {len(urls)} listing URLs")
    time.sleep(random.uniform(1.0, 2.5))
    return urls


def scrape_listing(url: str) -> dict | None:
    """
    Scrape one listing page and return a raw dict with these keys:
        url, title, price_raw, address_raw,
        details_raw (dict), features_raw (list[str]), description
    Returns None on fetch/parse failure.
    """
    scraper = _get_scraper()
    try:
        response = scraper.get(url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to fetch listing {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Primary: __NEXT_DATA__ JSON
    result = _parse_from_next_data(soup, url)

    # Fallback: HTML scraping
    if result is None:
        logger.debug(f"No __NEXT_DATA__ for {url}, falling back to HTML parsing")
        result = _parse_from_html(soup, url)

    time.sleep(random.uniform(1.0, 2.5))
    return result


# ---------------------------------------------------------------------------
# __NEXT_DATA__ extraction helpers
# ---------------------------------------------------------------------------

def _get_next_data(html: str) -> dict | None:
    """Parse and return the __NEXT_DATA__ JSON object, or None."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        if script and script.string:
            return json.loads(script.string)
    except Exception:
        pass
    return None


def _urls_from_next_data(html: str) -> list[str]:
    """Extract offer URLs from the search-results __NEXT_DATA__ payload."""
    data = _get_next_data(html)
    if not data:
        return []
    try:
        # The listing items live under props.pageProps.data.searchAds.items
        # (structure may vary; we scan recursively for "url" fields that match)
        items = (
            data.get("props", {})
            .get("pageProps", {})
            .get("data", {})
            .get("searchAds", {})
            .get("items", [])
        )
        urls: list[str] = []
        seen: set[str] = set()
        for item in items:
            u = item.get("url") or item.get("slug")
            if u and "/ro/oferta/" in u:
                full = urljoin(BASE_DOMAIN, u).split("?")[0]
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        return urls
    except Exception:
        return []


def _parse_from_next_data(soup: BeautifulSoup, url: str) -> dict | None:
    """
    Build the raw listing dict from the __NEXT_DATA__ JSON embedded in the page.
    Returns None if the JSON is absent or doesn't contain an 'ad' object.
    """
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return None

    try:
        data = json.loads(script.string)
        ad = data["props"]["pageProps"]["ad"]
    except (KeyError, json.JSONDecodeError):
        return None

    # --- title ---
    title: str | None = ad.get("title")

    # --- price ---
    price_raw: str | None = None
    chars: list[dict] = ad.get("characteristics", [])
    char_map = {c["key"]: c for c in chars}
    if "price" in char_map:
        price_raw = char_map["price"].get("localizedValue")

    # --- address ---
    address_raw = _build_address_raw(ad)

    # --- details_raw: structured fields from characteristics ---
    details_raw = _build_details_raw(char_map, ad)

    # --- features_raw: flat list of amenity strings ---
    features_raw: list[str] = ad.get("features") or []

    # --- description (strip HTML tags) ---
    description_html: str = ad.get("description") or ""
    description = re.sub(r"<[^>]+>", " ", description_html).strip() or None

    return {
        "url": url,
        "title": title,
        "price_raw": price_raw,
        "address_raw": address_raw,
        "details_raw": details_raw,
        "features_raw": features_raw,
        "description": description,
    }


def _build_address_raw(ad: dict) -> str | None:
    """Construct a human-readable address string from the ad's location object."""
    loc = ad.get("location", {})
    if not loc:
        return None

    parts: list[str] = []

    # Prefer the most specific level from reverseGeocoding
    reverse = loc.get("reverseGeocoding", {}).get("locations", [])
    # Sort by specificity: district > sector > county
    level_order = {"district": 0, "sector": 1, "county": 2}
    sorted_locs = sorted(
        reverse, key=lambda x: level_order.get(x.get("locationLevel", ""), 99)
    )
    if sorted_locs:
        parts.append(sorted_locs[0].get("fullName", ""))

    # Add street if available (value is a string or null)
    address_obj = loc.get("address", {}) or {}
    street = address_obj.get("street")
    if street and isinstance(street, str):
        parts.insert(0, street)

    result = ", ".join(p for p in parts if p)
    return result or None


def _build_details_raw(char_map: dict, ad: dict) -> dict:
    """
    Build a details dict from the characteristics map and target object.
    Keys are normalised Romanian labels matching what parser.py expects.
    """
    details: dict[str, str] = {}

    # Mapping: storia key → Romanian label used by parser.py
    key_map = {
        "m":                   "suprafață utilă",
        "rooms_num":           "număr camere",
        "floor_no":            "etaj",
        "building_floors_num": "numărul de etaje",
        "build_year":          "an construcție",
        "heating":             "încălzire",
        "building_material":   "material construcție",
        "building_type":       "tip clădire",
        "construction_status": "stare construcție",
        "windows_type":        "tip geamuri",
        "building_ownership":  "tip proprietate",
        "price_per_m":         "preț pe mp",
    }

    for storia_key, ro_label in key_map.items():
        char = char_map.get(storia_key)
        if char:
            details[ro_label] = char.get("localizedValue") or char.get("value") or ""

    # Normalise floor: "floor_5" → "5", "floor_ground" → "parter", "floor_attic" → "mansardă"
    if "etaj" in details:
        raw_floor = details["etaj"]
        if "ground" in raw_floor:
            details["etaj"] = "parter"
        elif "attic" in raw_floor or "mansard" in raw_floor.lower():
            details["etaj"] = "mansardă"
        else:
            # Extract numeric part: "floor_5" → "5"
            num = re.sub(r"[^\d]", "", raw_floor)
            total = details.get("numărul de etaje", "")
            details["etaj"] = f"{num}/{total}" if total else num

    return details


# ---------------------------------------------------------------------------
# HTML fallback helpers
# ---------------------------------------------------------------------------

def _parse_from_html(soup: BeautifulSoup, url: str) -> dict | None:
    """Best-effort HTML extraction when __NEXT_DATA__ is unavailable."""
    try:
        title = _html_title(soup)
        price_raw = _html_price(soup)
        address_raw = _html_address(soup)
        details_raw = _html_details(soup)
        features_raw = _html_features(soup)
        description = _html_description(soup)
    except Exception as e:
        logger.warning(f"HTML parse error for {url}: {e}")
        return None

    return {
        "url": url,
        "title": title,
        "price_raw": price_raw,
        "address_raw": address_raw,
        "details_raw": details_raw,
        "features_raw": features_raw,
        "description": description,
    }


def _html_title(soup: BeautifulSoup) -> str | None:
    tag = soup.find(attrs={"data-cy": "adPageAdTitle"}) or soup.find("h1")
    return tag.get_text(strip=True) if tag else None


def _html_price(soup: BeautifulSoup) -> str | None:
    tag = soup.find(attrs={"aria-label": "Preț"}) or soup.find(
        attrs={"data-cy": "adPageHeaderPrice"}
    )
    if tag:
        return tag.get_text(strip=True) or None
    # Fallback: any element containing a price-like string
    for el in soup.find_all(["strong", "span"]):
        text = el.get_text(strip=True)
        if ("€" in text or "EUR" in text) and any(c.isdigit() for c in text):
            return text
    return None


def _html_address(soup: BeautifulSoup) -> str | None:
    # MapLink component
    tag = soup.find(attrs={"data-sentry-component": "MapLink"})
    if tag:
        return tag.get_text(strip=True) or None
    return None


def _html_details(soup: BeautifulSoup) -> dict:
    """Extract key-value pairs from the AdDetailsBase component."""
    details: dict[str, str] = {}
    container = soup.find(attrs={"data-sentry-component": "AdDetailsBase"})
    if not container:
        return details

    # Each item is a grid div with two child divs: label and value
    for grid in container.find_all(
        attrs={"data-sentry-element": "ItemGridContainer"}
    ):
        items = grid.find_all(attrs={"data-sentry-element": "Item"})
        if len(items) >= 2:
            key = items[0].get_text(strip=True).rstrip(":").lower()
            value = items[1].get_text(strip=True)
            if key and value:
                details[key] = value

    # AccordionSection contains additional parameters (building, features)
    for acc in soup.find_all(attrs={"data-sentry-component": "AccordionSection"}):
        for grid in acc.find_all(
            attrs={"data-sentry-element": "ItemGridContainer"}
        ):
            items = grid.find_all(attrs={"data-sentry-element": "Item"})
            if len(items) >= 2:
                key = items[0].get_text(strip=True).rstrip(":").lower()
                value = items[1].get_text(strip=True)
                if key and value and key not in details:
                    details[key] = value

    return details


def _html_features(soup: BeautifulSoup) -> list[str]:
    """Extract amenity feature strings from AccordionSection."""
    features: list[str] = []
    seen: set[str] = set()

    for acc in soup.find_all(attrs={"data-sentry-component": "AccordionSection"}):
        # Features are listed as plain text items (not key:value grids)
        for tag in acc.find_all(["li", "span"]):
            text = tag.get_text(strip=True)
            if text and text not in seen and len(text) < 80:
                seen.add(text)
                features.append(text)

    return features


def _html_description(soup: BeautifulSoup) -> str | None:
    container = soup.find(attrs={"data-sentry-component": "AdDescriptionBase"})
    if container:
        text = container.get_text(separator="\n", strip=True)
        # Strip the "Descriere" heading and trailing junk
        text = re.sub(r"^Descriere\s*", "", text, flags=re.IGNORECASE).strip()
        if len(text) > 50:
            return text
    return None
