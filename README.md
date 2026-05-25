# Bucharest Real Estate Valuation Engine

Paste a Storia.ro apartment link and get an independent, data-driven estimate of
its fair price per square metre — with a prediction interval, an over/under/fair
verdict, and a comparison against real listings in the same neighbourhood.

The project is a full pipeline: a resilient scraper, a domain-aware cleaning and
feature-engineering layer, a spatial enrichment stage (neighbourhood, metro
distance, seismic risk), an XGBoost model with calibrated prediction intervals,
and a Streamlit app that ties it all together into a usable tool.




https://github.com/user-attachments/assets/f66302f2-9048-4cf2-957a-fb0ccaefbb87







---

## What it does

1. **Scrapes** apartment listings from Storia.ro, primarily by parsing the
   `__NEXT_DATA__` JSON embedded in each page (with a BeautifulSoup HTML fallback).
2. **Cleans & enriches** each listing — parses prices, areas, rooms, floors, and
   build years from messy Romanian text, and derives boolean amenities plus
   data-quality flags (CGI-render detection, explicit-unfurnished detection).
3. **Geocodes** each listing into spatial features: neighbourhood (point-in-polygon),
   zone, nearest-metro distance, distance to city centre, and seismic-risk band.
4. **Predicts** fair €/m² with an XGBoost regressor, plus a 10th–90th percentile
   interval from dedicated quantile models.
5. **Presents** the result in a Streamlit app: verdict banner, apartment details,
   neighbourhood price distribution, and comparable listings.

---

## Why it's more than "train a regressor"

A few design decisions worth highlighting:

- **Structured-data-first scraping.** Listings are read from the page's
  `__NEXT_DATA__` JSON rather than scraped from CSS selectors, which is far more
  stable; HTML parsing is only a fallback. The scraper also handles Cloudflare
  (via `cloudscraper`) and throttles itself with randomised delays.
- **Coordinates are validated, not trusted.** Storia's pins are often snapped to a
  metro plaza or a landmark. The geocoder flags suspicious coordinates (too close
  to a station, outside the city bounding box, or from an address with no street
  number) and refines them through Nominatim — accepting the result only if it
  moves the point more than 100 m.
- **Domain-aware data cleaning.** The enrichment layer detects CGI/render photos
  presented as real interiors and explicit "unfurnished" statements, then
  *corrects* misleading flags — a CGI new-build can't be "renovated" or "furnished".
- **Calibrated uncertainty.** Alongside the point estimate, two quantile models
  (p10 / p90) produce a prediction interval, and training verifies the interval's
  empirical coverage on the test set.
- **Spatial features only from coordinates.** Neighbourhood and distances are
  derived geometrically from validated lat/lon — never from the seller's free-text
  address, which is unreliable.

---

## Architecture

```
            Storia.ro
                │  scrape (__NEXT_DATA__ JSON, HTML fallback)
                ▼
        ┌───────────────┐     parse prices / area / rooms / year
        │  raw listing  │ ──▶ derive amenities + data-quality flags
        └───────────────┘     (CGI, unfurnished, new-build)
                │
                ▼  geocoding
        ┌───────────────┐     point-in-polygon → neighbourhood / zone
        │  enriched     │ ──▶ nearest metro, distance to centre
        │  listing      │     seismic-risk band, coord validation
        └───────────────┘
                │
                ▼
        ┌───────────────┐     normalised schema:
        │  SQLite DB    │     Listings  ←→  Neighborhoods
        └───────────────┘     (indexed on price, year, metro dist)
                │
                ▼
        ┌───────────────┐     XGBoost regressor (point estimate)
        │  model (.pkl) │ ──▶ p10 / p90 quantile models (interval)
        └───────────────┘     median imputation + one-hot categoricals
                │
                ▼
        ┌───────────────┐     paste URL → live scrape → predict
        │ Streamlit app │ ──▶ verdict, comps, neighbourhood distribution
        └───────────────┘
```

---

## Project structure

```
.
├── scraper/
│   └── storia_scraper.py      # __NEXT_DATA__ + HTML scraping, coord fetch
├── processing/
│   ├── parser.py              # typed parsing of price/area/rooms/floor/year
│   └── features.py            # amenities, CGI/unfurnished detection, enrichment
├── geocoding/
│   ├── geocoding.py           # point-in-polygon, metro/centre distance, validation
│   └── neighborhoods.py       # neighbourhood polygons, metro stations, zone map
├── database/
│   ├── db_manager.py          # all SQL access (insert, upsert, geo updates)
│   └── schema.sql             # Listings + Neighborhoods schema and indexes
├── modelML/
│   └── train.py               # XGBoost training, quantile models, metrics
├── app/
│   └── app.py                 # Streamlit valuation app
├── pipeline.py                # orchestrator: scrape / fetch-coords / geocode / stats
├── requirements.txt
└── real_estate.db             # SQLite database (generated)
```

---

## Setup

**Requirements:** Python 3.10+ (uses `X | None` type syntax).

```bash
git clone https://github.com/RaresVictor/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

The pipeline runs in phases via `pipeline.py`:

```bash
# 1. Scrape listings (≈37 per page)
python pipeline.py --scrape --pages 55

# 2. Backfill / validate coordinates and compute spatial features
python pipeline.py --fetch-coords

# 3. Inspect the database
python pipeline.py --stats

# Or run scrape + geocoding together
python pipeline.py --all --pages 55
```

Train the model once the database is populated:

```bash
python modelML/train.py
```

This writes `model.pkl`, `model_q10.pkl`, `model_q90.pkl`, `imputer.pkl`, and
`metadata.json` (feature columns + metrics) into `modelML/`.

Launch the app:

```bash
streamlit run app/app.py
```

Paste a `storia.ro/ro/oferta/...` link and analyse.

---

## Model

- **Target:** raw `price_per_sqm` (log-transforming the target reduced R², so it's
  left raw).
- **Features:** area, rooms, floor / total floors, build year, `is_post_1977`,
  new-build / penthouse / CGI flags, amenities, log-distance to metro and centre,
  raw lat/lon as continuous spatial signal, plus one-hot neighbourhood / zone /
  seismic-risk / nearest-metro.
- **Preprocessing:** IQR filter on the target, median imputation, one-hot categoricals.
- **Intervals:** separate p10 / p90 XGBoost quantile models, with coverage checked
  on the held-out test set.
- **Metrics** (MAE, RMSE, R², MAPE) and interval coverage are written to
  `metadata.json` and surfaced live in the app sidebar.

> Numbers in the sidebar come straight from your last training run — keep them
> honest and let them update themselves rather than hard-coding values in this README.

---

## Adding a screenshot

The app is the best thing about this project — show it. With the app running and a
listing analysed, take a screenshot of the verdict banner + neighbourhood
distribution, save it as `docs/screenshot.png`, and reference it near the top:

```markdown
![App screenshot](docs/screenshot.png)
```

A short screen-recording turned into a GIF (paste link → verdict appears) is even
better, since the live-scrape-to-prediction flow is the part that impresses.

---

## Possible next steps

- Track listings over time to estimate price *trends*, not just a cross-sectional snapshot.
- Add SHAP explanations so each estimate shows which features drove it.
- Expand beyond apartments / beyond Bucharest using the same pipeline.

---

*Personal project exploring web scraping, geospatial feature engineering, and
calibrated price prediction on real Bucharest market data.*
