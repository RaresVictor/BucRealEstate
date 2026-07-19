#!/usr/bin/env python3
"""
Pipeline orchestrator for the Bucharest Real Estate Valuation Engine.

Usage:
    python pipeline.py --scrape --pages 55   # scrape ~2000 listings
    python pipeline.py --fetch-coords        # backfill lat/lon from Storia pages
    python pipeline.py --geocode             # geocode via Nominatim (address-based)
    python pipeline.py --stats               # show DB statistics
    python pipeline.py --all --pages 55      # scrape + fetch-coords + geocode
"""

import argparse
import logging
import random
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

DB_PATH = "real_estate.db"


def phase_scrape(pages: int, db: str = DB_PATH) -> None:
    from scraper.storia_scraper import get_listing_urls, scrape_listing
    from processing.features import enrich_listing
    from database.db_manager import init_db, insert_listing, get_existing_urls

    conn = init_db(db)
    known_urls = get_existing_urls(conn)
    inserted = 0
    skipped_known = 0
    skipped_rental = 0
    failed = 0

    logger.info(f"Starting scrape: {pages} pages (~{pages * 37} listings), "
                f"{len(known_urls)} URLs already in DB")

    for page in range(1, pages + 1):
        urls = get_listing_urls(page)
        time.sleep(random.uniform(1.0, 2.5))
        if not urls:
            logger.warning(f"Page {page}: no URLs found, skipping")
            continue

        for url in urls:
            # Don't re-fetch listings we already have
            if url in known_urls:
                skipped_known += 1
                continue

            raw = scrape_listing(url)
            time.sleep(random.uniform(1.0, 2.5))
            if raw is None:
                failed += 1
                continue

            # Skip rentals that slipped through the vanzare URL filter
            url_lower = (raw.get("url") or "").lower()
            title_lower = (raw.get("title") or "").lower()
            if "inchiriere" in url_lower or "inchiriere" in title_lower or "închiriere" in title_lower:
                logger.debug(f"Skipping rental listing: {raw.get('url')}")
                skipped_rental += 1
                continue

            enriched = enrich_listing(raw)
            if insert_listing(conn, enriched):
                inserted += 1
                known_urls.add(url)
            else:
                skipped_known += 1

        logger.info(
            f"Page {page}/{pages} done | "
            f"inserted={inserted} known={skipped_known} "
            f"rentals={skipped_rental} failed={failed}"
        )

    conn.close()
    logger.info(f"Scrape complete: {inserted} new, {skipped_known} already known, "
                f"{skipped_rental} rentals, {failed} failed")


def _apply_geo(conn, listing_id: int, lat: float, lon: float,
               year_built: int | None) -> None:
    """Compute all coordinate-derived fields and persist them."""
    from geocoding.geocoding import (
        point_in_neighborhood, get_zone, get_nearest_metro, get_distance_to_center,
    )
    from processing.features import compute_seismic_risk
    from database.db_manager import update_geocoding

    neighborhood = point_in_neighborhood(lat, lon)
    zone = get_zone(neighborhood)
    dist_metro_m, nearest_metro = get_nearest_metro(lat, lon)
    dist_center_m = get_distance_to_center(lat, lon)
    # Seismic risk depends on the REAL neighborhood — recompute now that we know it
    seismic_risk = compute_seismic_risk(year_built, neighborhood)

    update_geocoding(
        conn,
        listing_id=listing_id,
        lat=lat,
        lon=lon,
        neighborhood=neighborhood,
        zone=zone,
        dist_metro_m=dist_metro_m,
        nearest_metro=nearest_metro,
        dist_center_m=dist_center_m,
        seismic_risk=seismic_risk,
    )


def phase_fetch_coords(db: str = DB_PATH) -> None:
    """
    For each listing without lat/lon, re-fetch the Storia page,
    pull coordinates from __NEXT_DATA__, then compute neighborhood/metro/center.
    Much faster than GeoPy — no rate limits, more accurate coords.
    """
    from scraper.storia_scraper import fetch_coords
    from geocoding.geocoding import validate_coords
    from database.db_manager import (
        get_connection, get_listings_missing_coords, mark_coords_failed,
    )

    conn = get_connection(db)
    updated = 0
    failed = 0

    while True:
        rows = get_listings_missing_coords(conn, limit=200)
        if not rows:
            break

        for row in rows:
            lat, lon = fetch_coords(row["url"])
            if lat is not None and lon is not None:
                lat, lon, refined = validate_coords(lat, lon, row["address_raw"])
                if refined:
                    logger.info(f"Coords refined via Nominatim for {row['url']}")
            if lat is None or lon is None:
                failed += 1
                mark_coords_failed(conn, row["id"])
                time.sleep(0.5)
                continue

            _apply_geo(conn, row["id"], lat, lon, row["year_built"])
            updated += 1
            time.sleep(random.uniform(0.5, 1.0))

        logger.info(f"Coords fetched: {updated} updated, {failed} failed so far…")

    conn.close()
    logger.info(f"fetch-coords complete: {updated} updated, {failed} failed")


def phase_geocode(batch: int, db: str = DB_PATH) -> None:
    """Address-based geocoding via Nominatim for listings Storia gave no coords for."""
    from geocoding.geocoding import get_coords_from_address
    from database.db_manager import (
        get_connection, get_ungeocoded_listings, mark_coords_failed,
    )

    conn = get_connection(db)
    total_updated = 0
    total_failed = 0

    while True:
        listings = get_ungeocoded_listings(conn, limit=batch)
        if not listings:
            break

        for listing in listings:
            coords = get_coords_from_address(listing["address_raw"])
            if not coords:
                logger.debug(f"Could not geocode: {listing['address_raw']}")
                # Mark it so we don't retry the same row forever
                mark_coords_failed(conn, listing["id"])
                total_failed += 1
                continue

            lat, lon = coords
            _apply_geo(conn, listing["id"], lat, lon, listing["year_built"])
            total_updated += 1
            time.sleep(1.0)  # Nominatim rate limit

        logger.info(f"Geocoded {total_updated} listings so far…")

    conn.close()
    logger.info(f"Geocoding complete: {total_updated} updated, {total_failed} failed")


def phase_stats(db: str = DB_PATH) -> None:
    from database.db_manager import get_connection, print_stats
    conn = get_connection(db)
    print_stats(conn)
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="BucRealEstate pipeline")
    parser.add_argument("--scrape", action="store_true")
    parser.add_argument("--fetch-coords", action="store_true",
                        help="Backfill lat/lon by re-fetching listing pages from Storia")
    parser.add_argument("--geocode", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pages", type=int, default=55,
                        help="Number of listing pages to scrape (default 55 ≈ 2000 listings)")
    parser.add_argument("--batch", type=int, default=100,
                        help="Geocoding batch size per iteration")
    parser.add_argument("--db", type=str, default=DB_PATH,
                        help="Path to SQLite database")
    args = parser.parse_args()
    db = args.db

    if not any([args.scrape, args.fetch_coords, args.geocode, args.stats, args.all]):
        parser.print_help()
        sys.exit(1)

    if args.scrape or args.all:
        phase_scrape(args.pages, db)

    if args.fetch_coords or args.all:
        phase_fetch_coords(db)

    if args.geocode or args.all:
        phase_geocode(args.batch, db)

    if args.stats:
        phase_stats(db)


if __name__ == "__main__":
    main()
