CREATE TABLE IF NOT EXISTS Neighborhoods (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT UNIQUE NOT NULL,
    zone            TEXT,
    avg_price_sqm   REAL
);

CREATE TABLE IF NOT EXISTS Listings (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    url                  TEXT UNIQUE NOT NULL,
    neighborhood_id      INTEGER REFERENCES Neighborhoods(id),

    -- Price (model target)
    price_eur            REAL,
    price_per_sqm        REAL,

    -- Parsed property fields
    title                TEXT,
    area_sqm             REAL,
    rooms                INTEGER,
    floor                INTEGER,
    total_floors         INTEGER,
    year_built           INTEGER,
    compartmentare       TEXT,

    -- Boolean features (0/1)
    has_parking          INTEGER DEFAULT 0,
    has_balcony          INTEGER DEFAULT 0,
    has_elevator         INTEGER DEFAULT 0,
    has_ac               INTEGER DEFAULT 0,
    has_central_heating  INTEGER DEFAULT 0,
    has_storage          INTEGER DEFAULT 0,
    is_renovated         INTEGER DEFAULT 0,
    is_furnished         INTEGER DEFAULT 0,

    -- Seismic features
    seismic_risk         TEXT,
    is_post_1977         INTEGER,

    -- New build / listing quality flags
    is_new_build         INTEGER DEFAULT 0,
    is_cgi_listing       INTEGER DEFAULT 0,
    is_penthouse         INTEGER DEFAULT 0,

    -- Geographic features (from coordinates, never from seller text)
    lat                  REAL,
    lon                  REAL,
    nearest_metro        TEXT,
    dist_metro_m         REAL,
    dist_center_m        REAL,

    -- Raw fields (for debugging and re-parsing)
    description          TEXT,
    address_raw          TEXT,
    neighborhood_raw     TEXT,
    price_raw            TEXT,
    details_raw_json     TEXT,

    scraped_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    geocoded_at          TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_neighborhood ON Listings(neighborhood_id);
CREATE INDEX IF NOT EXISTS idx_price_sqm    ON Listings(price_per_sqm);
CREATE INDEX IF NOT EXISTS idx_year_built   ON Listings(year_built);
CREATE INDEX IF NOT EXISTS idx_dist_metro   ON Listings(dist_metro_m);
CREATE INDEX IF NOT EXISTS idx_geocoded     ON Listings(geocoded_at);
