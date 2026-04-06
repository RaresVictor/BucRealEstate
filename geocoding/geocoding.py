"""
Geographic calculations: point-in-polygon, metro distance, center distance.
Does NOT use GeoPy — coordinates come from Storia's own __NEXT_DATA__.
"""

from geopy.distance import geodesic
from shapely.geometry import Point, Polygon

from geocoding.neighborhoods import (
    CARTIER_TO_ZONA,
    METRO_STATIONS,
    NEIGHBORHOODS,
)

# Build polygons once at import time
# Shapely uses (lon, lat) order
_POLYGONS: dict[str, Polygon] = {
    name: Polygon([(lon, lat) for lat, lon in coords])
    for name, coords in NEIGHBORHOODS.items()
}

# Bucharest center — Piața Unirii
_CENTER = (44.4268, 26.1025)


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
