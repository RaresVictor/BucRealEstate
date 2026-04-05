"""
Boolean / categorical feature extraction and listing enrichment.
"""

from processing.parser import (
    parse_price,
    parse_area,
    parse_rooms,
    parse_floor,
    parse_year_built,
    parse_compartmentare,
    parse_neighborhood_from_address,
)

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
    neighborhood_raw = parse_neighborhood_from_address(raw.get("address_raw"))

    bool_feats = parse_boolean_features(features_list, description)
    seismic_risk = compute_seismic_risk(year_built, None)  # neighborhood not known yet
    is_post_1977 = compute_is_post_1977(year_built)

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

        # Geographic (populated later by geocoding job)
        "lat": None,
        "lon": None,
        "neighborhood": None,
        "zone": None,
        "nearest_metro": None,
        "dist_metro_m": None,
        "dist_center_m": None,
    }
