"""
Boolean / categorical feature extraction and listing enrichment.
"""

from geocoding.geocoding import validate_coords

from processing.parser import (
    parse_price,
    parse_area,
    parse_rooms,
    parse_floor,
    parse_year_built,
    parse_compartmentare,
    parse_construction_status,
    parse_is_penthouse,
    parse_neighborhood_from_address,
)

# Keywords that indicate CGI renders or explicitly unfurnished/unfinished state
_CGI_KEYWORDS = [
    # Actual phrases used on Storia.ro (confirmed from DB sample)
    "titlu de prezentare",      # "imaginile sunt cu titlu de prezentare"
    "titlul de prezentare",     # "poze cu titlul de prezentare"
    "caracter orientativ",      # "imagini cu caracter orientativ"
    "caracter informativ",      # "imagini cu caracter informativ"
    "propuneri de amenajare",   # "fotografiile reprezintă propuneri de amenajare"
    "scopul de prezentare",
    # Fallback: older / less common variants
    "randare", "render",
    "exemplu ilustrativ",
]

# Keywords that explicitly indicate the apartment is unfurnished/empty shell
_UNFURNISHED_KEYWORDS = [
    "nemobilat", "fara mobila", "fără mobilă", "fara mobilier", "fără mobilier",
    "mobilier optional", "mobilier nu este inclus", "fara furniture",
    "se vinde gol", "se vinde neechipat",
]

# Seismic risk per CLAUDE.md specification
HIGH_SEISMIC_RISK_NEIGHBORHOODS = {
    "Armeneasca", "Unirii", "Universitate", "Colentina", "Pantelimon",
    "Grivița", "Grivita", "Obor", "Dristor", "Iancului",
}


def has_feature(features_list: list, *keywords: str) -> bool:
    joined = " ".join(features_list).lower()
    return any(kw.lower() in joined for kw in keywords)


def parse_boolean_features(features_list: list, description: str | None) -> dict:
    desc = (description or "").lower()
    fl = [f.lower() for f in (features_list or [])]

    def in_desc(*kws):
        return any(kw in desc for kw in kws)

    def in_feat(*kws):
        return has_feature(fl, *kws)

    return {
        "has_parking": in_feat("parcare", "garaj") or in_desc("parcare", "garaj"),
        "has_balcony": in_feat("balcon", "terasa", "terasă") or in_desc("balcon", "terasă", "terasa"),
        "has_elevator": in_feat("lift", "elevator") or in_desc("lift", "elevator"),
        "has_ac": in_feat("aer condiționat", "aer conditionat", "aer conditio") or in_desc("aer condiționat", "aer conditionat"),
        "has_central_heating": (
            in_feat("centrală termică", "centrala termica", "centrala proprie", "centrala pe gaz")
            or in_desc("centrală termică", "centrala termica", "centrala proprie", "centrala pe gaz")
        ),
        "has_storage": in_feat("debara", "boxă", "boxa") or in_desc("debara", "boxă", "boxa"),
        "is_renovated": in_desc("renovat", "modernizat", "refăcut", "refacut", "renovata", "renovare"),
        "is_furnished": (
            in_feat("mobilat", "utilat") or in_desc("mobilat", "utilat", "mobilată", "mobilata")
        ),
    }


def is_cgi_listing(description: str | None) -> bool:
    """True if the listing uses CGI/rendered images presented as real interiors."""
    if not description:
        return False
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in _CGI_KEYWORDS)


def is_explicitly_unfurnished(description: str | None) -> bool:
    """True if the listing explicitly states the apartment is unfurnished/empty."""
    if not description:
        return False
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in _UNFURNISHED_KEYWORDS)


def compute_is_new_build(year_built: int | None, construction_status: str | None) -> int:
    """1 if the apartment is a new build (built/finishing 2020+) or under construction."""
    if construction_status == "under_construction":
        return 1
    if year_built is not None and year_built >= 2020:
        return 1
    return 0


def compute_seismic_risk(year_built: int | None, neighborhood: str | None) -> str:
    if year_built is None:
        return "unknown"
    in_high_risk = (neighborhood or "") in HIGH_SEISMIC_RISK_NEIGHBORHOODS
    if year_built < 1978:
        return "very_high" if in_high_risk else "high"
    if year_built < 1990:
        return "medium"
    return "low"


def compute_is_post_1977(year_built: int | None) -> int | None:
    if year_built is None:
        return None
    return 1 if year_built > 1977 else 0


def _validated_coords(lat, lon, address_raw: str | None) -> dict:
    """Validate Storia's coordinates; refine with Nominatim if suspicious."""
    if lat is None or lon is None:
        return {"lat": None, "lon": None}
    lat, lon, _ = validate_coords(float(lat), float(lon), address_raw)
    return {"lat": lat, "lon": lon}


def enrich_listing(raw: dict) -> dict:
    """
    Transform a raw scraper dict into a fully-enriched dict ready for DB insertion.
    Geographic fields (lat/lon, metro distance, etc.) are left None here and
    populated later by the geocoding job, UNLESS Storia already provided coordinates
    (which it does via __NEXT_DATA__ — we store those directly).
    """
    details = raw.get("details_raw") or {}
    features_list = raw.get("features_raw") or []
    description = raw.get("description")
    title = raw.get("title") or ""

    price_eur = parse_price(raw.get("price_raw"))
    area_sqm = parse_area(details, description)
    rooms = parse_rooms(details, title)
    floor, total_floors = parse_floor(details)
    year_built = parse_year_built(details, description)
    compartmentare = parse_compartmentare(details)
    construction_status = parse_construction_status(details)
    is_penthouse = parse_is_penthouse(details, title, description)
    neighborhood_raw = parse_neighborhood_from_address(raw.get("address_raw"))

    bool_feats = parse_boolean_features(features_list, description)
    seismic_risk = compute_seismic_risk(year_built, None)  # neighborhood not known yet
    is_post_1977 = compute_is_post_1977(year_built)

    # New build / CGI detection
    cgi = is_cgi_listing(description)
    explicitly_unfurnished = is_explicitly_unfurnished(description)
    is_new_build = compute_is_new_build(year_built, construction_status)

    # Fix misleading boolean flags:
    # 1. CGI renders on new builds → can't be renovated or furnished
    if is_new_build and cgi:
        bool_feats["is_renovated"] = False
        bool_feats["is_furnished"] = False
    # 2. Explicitly says unfurnished → override any keyword match
    if explicitly_unfurnished:
        bool_feats["is_furnished"] = False
    # 3. New build under construction → can't be renovated
    if construction_status == "under_construction":
        bool_feats["is_renovated"] = False

    price_per_sqm = None
    if price_eur and area_sqm and area_sqm > 0:
        price_per_sqm = round(price_eur / area_sqm, 2)

    return {
        # Identity
        "url": raw.get("url"),
        "title": title or None,
        "address_raw": raw.get("address_raw"),
        "price_raw": raw.get("price_raw"),
        "details_raw": details,
        "description": description,
        "neighborhood_raw": neighborhood_raw,

        # Parsed numerics
        "price_eur": price_eur,
        "price_per_sqm": price_per_sqm,
        "area_sqm": area_sqm,
        "rooms": rooms,
        "floor": floor,
        "total_floors": total_floors,
        "year_built": year_built,
        "compartmentare": compartmentare,

        # Boolean features
        **bool_feats,

        # Seismic
        "seismic_risk": seismic_risk,
        "is_post_1977": is_post_1977,

        # New build / CGI
        "is_new_build": is_new_build,
        "is_cgi_listing": int(cgi),

        # Penthouse/duplex
        "is_penthouse": int(is_penthouse),

        # Coordinates — validated against Nominatim if Storia's look wrong
        **_validated_coords(raw.get("lat"), raw.get("lon"), raw.get("address_raw")),

        # Geographic features derived from coords (populated by geocoding job)
        "neighborhood": None,
        "zone": None,
        "nearest_metro": None,
        "dist_metro_m": None,
        "dist_center_m": None,
    }
