#!/usr/bin/env python3
"""
Pipeline orchestrator for the Bucharest Real Estate Valuation Engine.

Usage:
    python pipeline.py --scrape --pages 55   # scrape ~2000 listings
    python pipeline.py --geocode             # geocode ungeocoded listings
    python pipeline.py --stats               # show DB statistics
    python pipeline.py --all --pages 55      # scrape + geocode
"""

import argparse
import logging
import sys

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
    from database.db_manager import init_db, insert_listing

    conn = init_db(db)
    inserted = 0
    skipped = 0
    failed = 0

    logger.info(f"Starting scrape: {pages} pages (~{pages * 37} listings)")

    for page in range(1, pages + 1):
        urls = get_listing_urls(page)
        if not urls:
            logger.warning(f"Page {page}: no URLs found, skipping")
            continue

        for url in urls:
            raw = scrape_listing(url)
            if raw is None:
                failed += 1
                continue

            enriched = enrich_listing(raw)
            was_new = insert_listing(conn, enriched)
            if was_new:
                inserted += 1
            else:
                skipped += 1

        total = inserted + skipped + failed
        logger.info(
            f"Page {page}/{pages} done | "
            f"inserted={inserted} skipped={skipped} failed={failed} "
            f"(total processed={total})"
        )

    conn.close()
    logger.info(f"Scrape complete: {inserted} new, {skipped} duplicate, {failed} failed")


def phase_geocode(batch: int, db: str = DB_PATH) -> None:
    import time
    from database.db_manager import get_connection, get_ungeocoded_listings, update_geocoding

    # Geocoding imports — only needed for this phase
    try:
        from geocoding.geocoding import (
            get_coords_from_address,
            point_in_neighborhood,
            get_zone,
            get_nearest_metro,
            get_distance_to_center,
        )
    except ImportError as e:
        logger.error(f"Geocoding module not available: {e}")
        return

    conn = get_connection(db)
    total_updated = 0

    while True:
        listings = get_ungeocoded_listings(conn, limit=batch)
        if not listings:
            break

        for listing in listings:
            address = listing["address_raw"]
            coords = get_coords_from_address(address)
            if not coords:
                logger.debug(f"Could not geocode: {address}")
                continue

            lat, lon = coords
            neighborhood = point_in_neighborhood(lat, lon)
            zone = get_zone(neighborhood)
            dist_metro_m, nearest_metro = get_nearest_metro(lat, lon)
            dist_center_m = get_distance_to_center(lat, lon)

            update_geocoding(
                conn,
                listing_id=listing["id"],
                lat=lat,
                lon=lon,
                neighborhood=neighborhood or "Unknown",
                zone=zone or "Unknown",
                dist_metro_m=dist_metro_m,
                nearest_metro=nearest_metro,
                dist_center_m=dist_center_m,
            )
            total_updated += 1
            time.sleep(1.0)

        logger.info(f"Geocoded {total_updated} listings so far…")

    conn.close()
    logger.info(f"Geocoding complete: {total_updated} listings updated")


def phase_stats(db: str = DB_PATH) -> None:
    from database.db_manager import get_connection, print_stats
    conn = get_connection(db)
    print_stats(conn)
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="BucRealEstate pipeline")
    parser.add_argument("--scrape", action="store_true")
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

    if not any([args.scrape, args.geocode, args.stats, args.all]):
        parser.print_help()
        sys.exit(1)

    if args.scrape or args.all:
        phase_scrape(args.pages, db)

    if args.geocode or args.all:
        phase_geocode(args.batch, db)

    if args.stats:
        phase_stats(db)


if __name__ == "__main__":
    main()
