"""
Geographic calculations: point-in-polygon, metro distance, center distance.
Primary coords come from Storia's __NEXT_DATA__; Nominatim is used as a
fallback when Storia's coordinates look wrong (pinned to a landmark/square).
"""

import re
import time

from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from shapely.geometry import Point, Polygon

from geocoding.neighborhoods import (
    CARTIER_TO_ZONA,
    METRO_STATIONS,
    NEIGHBORHOODS,
)

_geolocator = Nominatim(user_agent="bucharest-re-v1", timeout=10)

# Rough bounding box for Bucharest metropolitan area
_BBOX = {"lat_min": 44.32, "lat_max": 44.60, "lon_min": 25.92, "lon_max": 26.28}

# Build polygons once at import time
# Shapely uses (lon, lat) order
_POLYGONS: dict[str, Polygon] = {
    name: Polygon([(lon, lat) for lat, lon in coords])
    for name, coords in NEIGHBORHOODS.items()
}

# Bucharest center — Piața Unirii
_CENTER = (44.4268, 26.1025)

# Minimum plausible distance to a metro station for a residential building.
# Anything below this means Storia geocoded to the station plaza, not the building.
_MIN_METRO_DIST_M = 80


def _in_bucharest(lat: float, lon: float) -> bool:
    return (
        _BBOX["lat_min"] <= lat <= _BBOX["lat_max"]
        and _BBOX["lon_min"] <= lon <= _BBOX["lon_max"]
    )


def _is_suspicious(lat: float, lon: float, address_raw: str | None) -> bool:
    """Return True if Storia's coordinates look wrong and need refinement."""
    if not _in_bucharest(lat, lon):
        return True
    # Pinned to a metro station entrance
    raw_dist = min(
        geodesic((lat, lon), (slat, slon)).meters
        for _, (slat, slon) in METRO_STATIONS.items()
    )
    if raw_dist < _MIN_METRO_DIST_M:
        return True
    # Address has no street number — vague, geocoding is unreliable
    if address_raw and not re.search(r"\d", address_raw):
        return True
    return False


def _nominatim_refine(address_raw: str) -> tuple[float, float] | None:
    """Try Nominatim on progressively simplified versions of the address."""
    candidates = [
        f"{address_raw}, București, România",
        re.sub(r"\b(nr\.?|numărul)\s*\d+\w*", "", address_raw).strip() + ", București, România",
    ]
    for query in candidates:
        try:
            loc = _geolocator.geocode(query)
            time.sleep(0.5)
            if loc and _in_bucharest(loc.latitude, loc.longitude):
                return loc.latitude, loc.longitude
        except Exception:
            pass
    return None


def validate_coords(
    lat: float, lon: float, address_raw: str | None
) -> tuple[float, float, bool]:
    """
    Validate Storia's coordinates and refine with Nominatim if suspicious.

    Returns (lat, lon, was_refined).
    Falls back to original coords if Nominatim also fails.
    """
    if not _is_suspicious(lat, lon, address_raw):
        return lat, lon, False

    refined = _nominatim_refine(address_raw or "")
    if refined:
        ref_lat, ref_lon = refined
        # Only accept if meaningfully different (>100m) — avoids swapping one wrong
        # coord for another that's equally close
        diff = geodesic((lat, lon), (ref_lat, ref_lon)).meters
        if diff > 100:
            return ref_lat, ref_lon, True

    return lat, lon, False


def point_in_neighborhood(lat: float, lon: float) -> str | None:
    """Return the neighborhood name for (lat, lon), or nearest centroid fallback."""
    pt = Point(lon, lat)  # shapely is (lon, lat)
    for name, poly in _POLYGONS.items():
        if poly.contains(pt):
            return name
    return _nearest_neighborhood_centroid(lat, lon)


def _nearest_neighborhood_centroid(lat: float, lon: float) -> str | None:
    """Return the closest neighborhood by centroid distance, or None if >3 km away."""
    best_name: str | None = None
    best_dist = float("inf")
    for name, poly in _POLYGONS.items():
        c = poly.centroid
        dist = geodesic((lat, lon), (c.y, c.x)).meters
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name if best_dist <= 3000 else None


def get_zone(neighborhood: str | None) -> str | None:
    if not neighborhood:
        return None
    return CARTIER_TO_ZONA.get(neighborhood)


def get_nearest_metro(lat: float, lon: float) -> tuple[float, str]:
    """Return (distance_m, station_name) for the closest metro station."""
    best_name = ""
    best_dist = float("inf")
    for name, (slat, slon) in METRO_STATIONS.items():
        dist = geodesic((lat, lon), (slat, slon)).meters
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_dist, best_name


def get_distance_to_center(lat: float, lon: float) -> float:
    """Distance in metres to Piața Unirii."""
    return geodesic((lat, lon), _CENTER).meters
