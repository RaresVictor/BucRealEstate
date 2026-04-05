"""
Database access layer. All SQL goes through here.
"""

import json
import os
import sqlite3

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def init_db(db_path: str = "real_estate.db") -> sqlite3.Connection:
    conn = get_connection(db_path)
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def get_connection(db_path: str = "real_estate.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def upsert_neighborhood(conn: sqlite3.Connection, name: str, zone: str | None) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO Neighborhoods (name, zone) VALUES (?, ?)",
        (name, zone),
    )
    row = conn.execute(
        "SELECT id FROM Neighborhoods WHERE name = ?", (name,)
    ).fetchone()
    return row["id"]


def insert_listing(conn: sqlite3.Connection, listing: dict) -> bool:
    """
    Insert a processed listing dict into the DB.
    Returns True if inserted, False if URL already existed.
    """
    # Resolve neighborhood_id
    neighborhood_id = None
    neighborhood = listing.get("neighborhood")
    if neighborhood:
        neighborhood_id = upsert_neighborhood(
            conn, neighborhood, listing.get("zone")
        )

    details_json = json.dumps(
        listing.get("details_raw") or {}, ensure_ascii=False
    )

    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO Listings (
                url, neighborhood_id,
                price_eur, price_per_sqm,
                title, area_sqm, rooms, floor, total_floors,
                year_built, compartmentare,
                has_parking, has_balcony, has_elevator, has_ac,
                has_central_heating, has_storage, is_renovated, is_furnished,
                seismic_risk, is_post_1977,
                lat, lon, nearest_metro, dist_metro_m, dist_center_m,
                description, address_raw, neighborhood_raw,
                price_raw, details_raw_json
            ) VALUES (
                :url, :neighborhood_id,
                :price_eur, :price_per_sqm,
                :title, :area_sqm, :rooms, :floor, :total_floors,
                :year_built, :compartmentare,
                :has_parking, :has_balcony, :has_elevator, :has_ac,
                :has_central_heating, :has_storage, :is_renovated, :is_furnished,
                :seismic_risk, :is_post_1977,
                :lat, :lon, :nearest_metro, :dist_metro_m, :dist_center_m,
                :description, :address_raw, :neighborhood_raw,
                :price_raw, :details_raw_json
            )
            """,
            {
                "url": listing.get("url"),
                "neighborhood_id": neighborhood_id,
                "price_eur": listing.get("price_eur"),
                "price_per_sqm": listing.get("price_per_sqm"),
                "title": listing.get("title"),
                "area_sqm": listing.get("area_sqm"),
                "rooms": listing.get("rooms"),
                "floor": listing.get("floor"),
                "total_floors": listing.get("total_floors"),
                "year_built": listing.get("year_built"),
                "compartmentare": listing.get("compartmentare"),
                "has_parking": int(listing.get("has_parking") or 0),
                "has_balcony": int(listing.get("has_balcony") or 0),
                "has_elevator": int(listing.get("has_elevator") or 0),
                "has_ac": int(listing.get("has_ac") or 0),
                "has_central_heating": int(listing.get("has_central_heating") or 0),
                "has_storage": int(listing.get("has_storage") or 0),
                "is_renovated": int(listing.get("is_renovated") or 0),
                "is_furnished": int(listing.get("is_furnished") or 0),
                "seismic_risk": listing.get("seismic_risk"),
                "is_post_1977": listing.get("is_post_1977"),
                "lat": listing.get("lat"),
                "lon": listing.get("lon"),
                "nearest_metro": listing.get("nearest_metro"),
                "dist_metro_m": listing.get("dist_metro_m"),
                "dist_center_m": listing.get("dist_center_m"),
                "description": listing.get("description"),
                "address_raw": listing.get("address_raw"),
                "neighborhood_raw": listing.get("neighborhood_raw"),
                "price_raw": listing.get("price_raw"),
                "details_raw_json": details_json,
            },
        )
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        conn.rollback()
        raise


def get_ungeocoded_listings(conn: sqlite3.Connection, limit: int = 100) -> list:
    return conn.execute(
        "SELECT * FROM Listings WHERE geocoded_at IS NULL LIMIT ?", (limit,)
    ).fetchall()


def update_geocoding(
    conn: sqlite3.Connection,
    listing_id: int,
    lat: float,
    lon: float,
    neighborhood: str,
    zone: str,
    dist_metro_m: float,
    nearest_metro: str,
    dist_center_m: float,
) -> None:
    neighborhood_id = upsert_neighborhood(conn, neighborhood, zone)
    conn.execute(
        """
        UPDATE Listings
        SET lat = ?, lon = ?, neighborhood_id = ?,
            dist_metro_m = ?, nearest_metro = ?, dist_center_m = ?,
            geocoded_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (lat, lon, neighborhood_id, dist_metro_m, nearest_metro, dist_center_m, listing_id),
    )
    conn.commit()


def get_listings_for_model(conn: sqlite3.Connection):
    import pandas as pd
    query = """
        SELECT
            l.*,
            n.name  AS neighborhood,
            n.zone  AS zone
        FROM Listings l
        LEFT JOIN Neighborhoods n ON l.neighborhood_id = n.id
        WHERE l.price_per_sqm IS NOT NULL
          AND l.lat IS NOT NULL
    """
    return pd.read_sql_query(query, conn)


def print_stats(conn: sqlite3.Connection) -> None:
    total = conn.execute("SELECT COUNT(*) FROM Listings").fetchone()[0]
    geocoded = conn.execute(
        "SELECT COUNT(*) FROM Listings WHERE geocoded_at IS NOT NULL"
    ).fetchone()[0]
    with_price = conn.execute(
        "SELECT COUNT(*) FROM Listings WHERE price_per_sqm IS NOT NULL"
    ).fetchone()[0]
    print(f"Total listings : {total:,}")
    print(f"Geocoded       : {geocoded:,}")
    print(f"With price/sqm : {with_price:,}")
    rows = conn.execute(
        "SELECT n.name, COUNT(*) AS cnt FROM Listings l "
        "JOIN Neighborhoods n ON l.neighborhood_id = n.id "
        "GROUP BY n.name ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    if rows:
        print("\nTop neighborhoods:")
        for r in rows:
            print(f"  {r['name']:25} {r['cnt']:>4}")
