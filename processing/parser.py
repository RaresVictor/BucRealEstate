"""
Parse raw string fields from the scraper into typed values.
"""

import re


def parse_price(raw: str | None) -> float | None:
    if not raw:
        return None
    digits = re.sub(r"[^\d]", "", raw)
    return float(digits) if digits else None


def parse_area(details: dict, description: str | None = None) -> float | None:
    keys = ["suprafață utilă", "suprafata utila", "suprafață construită",
            "suprafata construita", "suprafață", "suprafata", "area"]
    for k in keys:
        val = details.get(k)
        if val:
            m = re.search(r"(\d+(?:[.,]\d+)?)", val)
            if m:
                return float(m.group(1).replace(",", "."))
    if description:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*m[²2]", description)
        if m:
            return float(m.group(1).replace(",", "."))
    return None


def parse_rooms(details: dict, title: str | None = None) -> int | None:
    keys = ["număr camere", "numar camere", "camere", "nr. camere",
            "numărul de camere", "rooms_num"]
    for k in keys:
        val = details.get(k)
        if val:
            m = re.search(r"(\d+)", val)
            if m:
                return int(m.group(1))
    if title:
        m = re.search(r"(\d)\s*camer", title, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # "garsoniera" = 1 room
        if re.search(r"garsonier", title, re.IGNORECASE):
            return 1
    return None


def parse_floor(details: dict) -> tuple[int | None, int | None]:
    val = details.get("etaj") or details.get("floor_no")
    total_val = details.get("numărul de etaje") or details.get("numar etaje") or details.get("total_floors")

    total: int | None = None
    if total_val:
        m = re.search(r"(\d+)", str(total_val))
        if m:
            total = int(m.group(1))

    if not val:
        return None, total

    val_str = str(val).lower().strip()

    if "parter" in val_str or "ground" in val_str:
        return 0, total
    if "mansard" in val_str or "attic" in val_str:
        return total, total

    # "4/8" format
    m = re.match(r"(\d+)\s*/\s*(\d+)", val_str)
    if m:
        return int(m.group(1)), int(m.group(2))

    # "floor_5" → already normalised to "5/11" by scraper, but handle raw form
    m = re.search(r"(\d+)", val_str)
    if m:
        return int(m.group(1)), total

    return None, total


def parse_year_built(details: dict, description: str | None = None) -> int | None:
    keys = ["an construcție", "an constructie", "anul construcției",
            "anul constructiei", "build_year", "an construire"]
    for k in keys:
        val = details.get(k)
        if val:
            m = re.search(r"(1[89]\d{2}|20[012]\d)", str(val))
            if m:
                year = int(m.group(1))
                if 1900 <= year <= 2030:
                    return year
    if description:
        m = re.search(
            r"construit\s*(?:în|in)?\s*(19[4-9]\d|20[0-2]\d)",
            description,
            re.IGNORECASE,
        )
        if m:
            year = int(m.group(1))
            if 1900 <= year <= 2030:
                return year
    return None


def parse_compartmentare(details: dict) -> str | None:
    keys = ["compartimentare", "tip apartament", "compartmentare"]
    for k in keys:
        val = details.get(k)
        if val:
            val_l = val.lower()
            if "nedecomandat" in val_l or "studio" in val_l:
                return "nedecomandat"
            if "semidecomandat" in val_l or "semi" in val_l:
                return "semidecomandat"
            if "circular" in val_l:
                return "circular"
            if "decomandat" in val_l:
                return "decomandat"
    return None


def parse_is_penthouse(details: dict, title: str | None, description: str | None) -> bool:
    """
    True if the listing is a penthouse or duplex on the top floor.
    Criteria: explicitly called penthouse/duplex OR floor == total_floors AND area > 120m².
    """
    text = f"{title or ''} {description or ''}".lower()
    if "penthouse" in text or "duplex" in text:
        return True
    floor, total = parse_floor(details)
    area_val = details.get("suprafață utilă") or details.get("suprafata utila") or ""
    import re
    area_m = re.search(r"(\d+(?:[.,]\d+)?)", area_val)
    area = float(area_m.group(1).replace(",", ".")) if area_m else 0
    if floor is not None and total is not None and floor >= total and area > 120:
        return True
    return False


def parse_construction_status(details: dict) -> str | None:
    """Return normalized construction status: 'ready', 'under_construction', 'needs_renovation', or None."""
    val = (details.get("stare construcție") or details.get("stare constructie") or "").lower()
    if not val:
        return None
    if "construcție" in val or "constructie" in val or "finalizare" in val or "completion" in val:
        return "under_construction"
    if "gata" in val or "ready" in val or "utilizare" in val:
        return "ready"
    if "renov" in val or "necesit" in val:
        return "needs_renovation"
    return None


def parse_neighborhood_from_address(address_raw: str | None) -> str | None:
    """
    Extract the neighborhood string the seller wrote.
    Stored as neighborhood_raw for debugging only — never used as a model feature.
    """
    if not address_raw:
        return None
    # Storia's reverseGeocoding gives us "District, Sector, City"
    # The first part before the first comma is the most specific location
    parts = [p.strip() for p in address_raw.split(",")]
    if parts:
        first = parts[0]
        # Filter out sector-only strings
        if not re.match(r"^Sector(ul)?\s+\d", first, re.IGNORECASE):
            return first
    return None
