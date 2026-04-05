# Bucharest Real Estate Valuation Engine — Context pentru Claude Code

## Ce este acest proiect

Un motor de evaluare imobiliară pentru București. Scrăpăm anunțuri de apartamente
de pe Storia.ro, le stocăm într-o bază de date SQLite normalizată, antrenăm un model
XGBoost să prezică prețul per metru pătrat, și expunem totul printr-o aplicație
Streamlit interactivă.

Proiectul are 5 faze:
1. Scraping (colectare anunțuri)
2. Geocoding (determinare cartier din coordonate, nu din textul vânzătorului)
3. Procesare (parsare câmpuri brute, feature engineering)
4. Bază de date (stocare normalizată în SQLite)
5. Model + Aplicație (XGBoost + Streamlit)

---

## Structura proiectului

```
bucharest-real-estate/
├── CLAUDE.md                  ← acest fișier
├── requirements.txt
├── pipeline.py                ← orchestrează toate fazele
├── real_estate.db             ← generat automat la primul run
│
├── scraper/
│   ├── __init__.py
│   └── storia_scraper.py
│
├── geocoding/
│   ├── __init__.py
│   ├── neighborhoods.py
│   └── geocoding.py
│
├── processing/
│   ├── __init__.py
│   ├── parser.py
│   └── features.py
│
├── database/
│   ├── __init__.py
│   ├── schema.sql
│   └── db_manager.py
│
├── model/
│   ├── __init__.py
│   └── train.py
│
└── app/
    └── streamlit_app.py
```

---

## Reguli absolute — citește cu atenție

### 1. Cartierul se determină DIN COORDONATE, niciodată din textul vânzătorului

Aceasta este cea mai importantă regulă a proiectului. Vânzătorii scriu orice:
"Centru" când apartamentul e pe Armeneasca, "lângă metrou" fără să specifice
care, sau nu dau nicio zonă, doar strada.

Soluția corectă:
- Se ia adresa brută din anunț
- Se geocodează cu GeoPy → obținem lat/lon
- Se face point-in-polygon cu shapely → obținem cartierul real
- Distanța la metrou, distanța față de centru, zona (Nord/Sud/Est/Vest/Centru)
  se calculează toate din coordonate

Textul vânzătorului se păstrează în câmpul `description` și `address_raw`
din baza de date, dar **nu devine niciodată feature pentru model**.

### 2. Scraping DOAR apartamente

URL-ul de bază este:
```
https://www.storia.ro/ro/rezultate/vanzare/apartament/bucuresti?page={}
```
Niciodată `/casa-si-vila/` sau alte tipuri de proprietăți.

### 3. Geocodingul se face SEPARAT de scraping

GeoPy face request-uri HTTP și e lent (1-2 secunde per adresă). Nu se apelează
în timpul scraping-ului. Fluxul corect:
1. Scraping → salvează tot în DB cu lat/lon = NULL
2. Job separat de geocoding → actualizează rândurile cu lat/lon = NULL

### 4. Păstrează întotdeauna datele brute

Câmpurile `description`, `address_raw`, `price_raw`, `details_raw_json` se
salvează exact cum vin de pe site. Dacă parsarea Regex eșuează la un câmp,
poți re-parsa fără să re-scrapi.

### 5. Selectoarele CSS de pe Storia se pot schimba

Storia.ro folosește clase generate (ex: `css-1wnihf5`) care se schimbă la
fiecare deploy. Selectoarele prioritare sunt `data-testid` și `aria-label`
care sunt mai stabile. Dacă scraperul nu găsește elemente, primul pas de
debugging este să inspectezi HTML-ul curent al site-ului.

---

## requirements.txt

```
beautifulsoup4>=4.12
requests>=2.31
cloudscraper>=1.2
shapely>=2.0
geopy>=2.4
pandas>=2.0
xgboost>=2.0
scikit-learn>=1.3
streamlit>=1.28
matplotlib>=3.7
seaborn>=0.12
geopandas>=0.14
```

---

## scraper/storia_scraper.py

### Ce face
Colectează URL-urile anunțurilor din paginile de listing, apoi intră în
fiecare anunț individual și extrage toate câmpurile disponibile.

### Funcții de implementat

**`get_listing_urls(page_num: int) -> list[str]`**
- Accesează `BASE_URL.format(page_num)`
- Extrage toate link-urile de tip `/ro/oferta/...`
- Returnează lista de URL-uri unice
- Adaugă `time.sleep(random.uniform(1.0, 2.5))` după fiecare pagină

**`scrape_listing(url: str) -> dict | None`**
- Accesează pagina anunțului individual
- Extrage și returnează un dict cu cheile:
  - `url` — URL-ul anunțului
  - `title` — titlul anunțului (ex: "Apartament 3 camere, Floreasca")
  - `price_raw` — prețul ca string brut (ex: "185 000 €")
  - `address_raw` — adresa ca string brut (ex: "Str. Armeneasca nr. 12, Sector 2")
  - `details_raw` — dict cu perechile cheie-valoare din tabelul de detalii tehnice
    (ex: {"suprafață utilă": "78 m²", "etaj": "4/8", "an construcție": "1982"})
  - `features_raw` — listă de strings cu caracteristicile bifate
    (ex: ["parcare", "balcon", "lift", "aer condiționat"])
  - `description` — textul complet al descrierii libere
- Returnează `None` dacă pagina nu se poate accesa sau parse
- Adaugă `time.sleep(random.uniform(1.0, 2.5))` după fiecare anunț

### Note de implementare
- Folosește `cloudscraper` în loc de `requests` dacă primești 403
- Headers: `User-Agent` de Chrome real
- Selectori de încercat în ordine: `data-testid`, `aria-label`, clase CSS
- Logează URL-urile care eșuează fără să oprească execuția (try/except)
- Nu urmări link-uri externe sau anunțuri din alte orașe

---

## geocoding/neighborhoods.py

### Ce face
Definește toate structurile de date geografice statice ale proiectului:
poligoanele cartierelor, stațiile de metrou, maparea cartier→zonă,
și cartierele cu risc seismic ridicat.

### Ce trebuie să conțină

**`NEIGHBORHOODS: dict[str, list[tuple[float, float]]]`**
Dicționar cu numele cartierului ca cheie și lista de coordonate (lat, lon)
ale poligonului simplificat ca valoare. Include minim:
Floreasca, Dorobanți, Aviației, Herăstrău, Armeneasca, Tineretului,
Militari, Titan, Drumul Taberei, Berceni, Colentina, Pantelimon,
Pipera, Băneasa, Universitate, Unirii, Cotroceni, Rahova, Grivița,
Vitan, Dristor, Iancului, Obor, Ștefan cel Mare, Floreasca (M1).

Notă: Poligoanele din acest fișier sunt aproximative/simplificate.
Pentru producție, se pot înlocui cu GeoJSON exact de pe Overpass API:
```python
# Query Overpass pentru granițele reale ale cartierelor din București:
# area["name"="București"]["boundary"="administrative"]->.b;
# relation(area.b)["boundary"="administrative"]["admin_level"="10"];
# out geom;
```

**`METRO_STATIONS: dict[str, tuple[float, float]]`**
Toate stațiile de metrou din București (M1-M5 + M6 Drumul Taberei),
cu coordonatele (lat, lon). Include toate stațiile active.

**`ZONE_MAP: dict[str, list[str]]`**
Maparea zonelor macro la lista de cartiere:
- "Nord": Floreasca, Dorobanți, Aviației, Herăstrău, Băneasa, Pipera, Colentina
- "Centru": Armeneasca, Universitate, Unirii, Cotroceni, Grivița
- "Sud": Tineretului, Berceni, Rahova
- "Est": Titan, Pantelimon, Dristor, Vitan
- "Vest": Militari, Drumul Taberei

**`CARTIER_TO_ZONA: dict[str, str]`**
Inversul lui ZONE_MAP — cartier → zonă. Se generează automat din ZONE_MAP.

**`HIGH_SEISMIC_RISK_NEIGHBORHOODS: set[str]`**
Cartierele cu risc seismic ridicat (sol slab + construcții vechi pre-1977):
Armeneasca, Unirii, Universitate, Colentina, Pantelimon, Grivița, Obor,
Dristor, Iancului — zone afectate de cutremurul din 1977.

---

## geocoding/geocoding.py

### Ce face
Transformă adrese text în coordonate și coordonate în informații geografice
structurate (cartier, zonă, distanțe). Acesta este modulul care rezolvă
problema vânzătorilor care scriu adrese incorecte sau vagi.

### Funcții de implementat

**`get_coords_from_address(address_raw: str) -> tuple[float, float] | None`**
Încearcă geocodarea adresei în mai mulți pași, de la cel mai specific la cel
mai general:
1. Adresa completă + ", București, România"
2. Strada fără număr + ", București, România"
3. Doar cartierul menționat + ", București, România"
Returnează (lat, lon) la primul succes, None dacă toate eșuează.
Adaugă `time.sleep(0.5)` între încercări pentru a respecta limita GeoPy.

**`_build_search_candidates(address_raw: str) -> list[str]`**
Funcție helper privată. Din adresa brută (ex: "Str. Armeneasca nr. 12, Sector 2")
generează lista de candidați pentru geocodare:
- Adresa completă
- Adresa fără "Sector X" și "nr. X" (curățată cu Regex)
- Doar prefixul stradal + numele străzii (ex: "Strada Armeneasca")
- Doar numele străzii (ex: "Armeneasca")

**`point_in_neighborhood(lat: float, lon: float) -> str | None`**
Determină cartierul real al unui punct geografic folosind point-in-polygon
cu biblioteca shapely. Verifică dacă punctul (lon, lat) — atenție la ordinea
shapely! — se află în interiorul vreunui poligon din NEIGHBORHOODS.
Dacă nu se potrivește cu niciun poligon exact, apelează
`_nearest_neighborhood_centroid()` ca fallback.
Aceasta este funcția cheie care rezolvă problema adreselor vagi.

**`_nearest_neighborhood_centroid(lat: float, lon: float) -> str | None`**
Fallback când punctul nu se află în niciun poligon definit (de obicei la
granița dintre cartiere). Calculează distanța geodezică față de centroidul
fiecărui poligon și returnează cel mai apropiat cartier.
Returnează None dacă distanța minimă > 3000m (punct în afara Bucureștiului).

**`get_zone(neighborhood: str | None) -> str | None`**
Lookup simplu în CARTIER_TO_ZONA. Returnează zona macro (Nord/Sud/etc.)
sau None dacă cartierul nu e în map.

**`get_nearest_metro(lat: float, lon: float) -> tuple[float, str]`**
Calculează distanța geodezică față de toate stațiile din METRO_STATIONS
și returnează (distanța_în_metri, numele_stației_celei_mai_apropiate).

**`get_distance_to_center(lat: float, lon: float) -> float`**
Distanța geodezică față de Piața Unirii (44.4268, 26.1025), considerată
centrul geografic al Bucureștiului. Returnează distanța în metri.

### Note de implementare
- shapely folosește (lon, lat) nu (lat, lon) — atenție la ordinea coordonatelor!
- Poligoanele se construiesc o singură dată la import și se cachează în
  `_POLYGONS` (dict module-level) pentru performanță
- geolocator = Nominatim(user_agent="bucharest-re-v1", timeout=10)
- Toate funcțiile cu GeoPy trebuie wrapped în try/except

---

## processing/parser.py

### Ce face
Parsează câmpurile brute extrase de scraper (strings nestructurate) în valori
tipizate folosind Regex. Primește `details_raw` (dict), `features_raw` (list)
și `description` (string) și returnează valori numerice/booleane curate.

### Funcții de implementat

**`parse_price(raw: str) -> float | None`**
Din "185 000 €" sau "185.000 EUR" sau "185000 euro" extrage 185000.0.
Elimină toate caracterele non-numerice cu `re.sub(r"[^\d]", "", raw)`.

**`parse_area(details: dict, description: str) -> float | None`**
Caută mai întâi în details pentru cheile: "suprafață utilă", "suprafata utila",
"suprafață construită", "suprafață". Fallback: caută în description după
pattern `(\d+(?:[.,]\d+)?)\s*m[²2]`. Returnează float în m².

**`parse_rooms(details: dict, title: str) -> int | None`**
Caută în details pentru cheile: "număr camere", "camere", "nr. camere".
Fallback: din titlu după pattern `(\d)\s*camer` (ex: "3 camere").

**`parse_floor(details: dict) -> tuple[int | None, int | None]`**
Returnează (etaj_curent, total_etaje). Din details, cheia "etaj".
Formate posibile: "4/8", "4 din 8", "parter", "mansardă", doar "4".
"parter" → etaj=0. "mansardă" → etaj=total_etaje.

**`parse_year_built(details: dict, description: str) -> int | None`**
Caută în details pentru cheile: "an construcție", "an constructie",
"anul construcției". Fallback în description: pattern
`construit\s*(?:în|in)?\s*(19[4-9]\d|20[0-2]\d)`.
Validează: returneaza None dacă anul e în afara [1900, 2025].

**`parse_compartmentare(details: dict) -> str | None`**
Caută în details: "compartimentare", "tip apartament".
Normalizează la: "decomandat", "semidecomandat", "nedecomandat", "circular".

**`parse_neighborhood_from_address(address_raw: str) -> str | None`**
Extrage cartierul menționat de vânzător din adresa brută.
IMPORTANT: Aceasta se salvează în `neighborhood_raw` (pentru debugging),
NU se folosește ca feature pentru model. Cartierul real vine din geocoding.

---

## processing/features.py

### Ce face
Calculează feature-uri booleane și categorice din lista de caracteristici
bifate și din textul descrierii. Combină informațiile din mai multe surse
pentru a crește recall-ul (un feature poate fi menționat în caracteristici
SAU în descriere).

### Funcții de implementat

**`has_feature(features_list: list, *keywords: str) -> bool`**
Helper. Returnează True dacă oricare dintre keywords apare ca substring
în oricare element din features_list. Case-insensitive.

**`parse_boolean_features(features_list: list, description: str) -> dict`**
Returnează un dict cu cheile:
- `has_parking`: parcare sau garaj în features SAU în description
- `has_balcony`: balcon sau terasă
- `has_elevator`: lift sau elevator
- `has_ac`: aer condiționat
- `has_central_heating`: centrală termică proprie (nu termoficare)
- `has_storage`: debara sau boxă
- `is_renovated`: renovat/modernizat/refăcut menționat în description
- `is_furnished`: mobilat/utilat în features sau description

**`compute_seismic_risk(year_built: int | None, neighborhood: str | None) -> str`**
Combină anul construcției cu cartierul pentru a determina riscul seismic.
Logica:
- year_built < 1978 ȘI neighborhood în HIGH_SEISMIC_RISK_NEIGHBORHOODS → "very_high"
- year_built < 1978 (altă zonă) → "high"
- 1978 ≤ year_built < 1990 → "medium" (normative îmbunătățite, dar tot vechi)
- year_built ≥ 1990 → "low"
- year_built is None → "unknown"
Returnează string din {"very_high", "high", "medium", "low", "unknown"}.

**`compute_is_post_1977(year_built: int | None) -> int | None`**
Returnează 1 dacă year_built > 1977, 0 dacă ≤ 1977, None dacă necunoscut.
Aceasta e o variabilă binară importantă pentru model — cutremurul din 1977
a determinat schimbări majore în normativele de construcție.

**`enrich_listing(raw: dict) -> dict`**
Funcția principală care combină parser.py + features.py + geocoding.py.
Primește dict-ul raw de la scraper și returnează dict-ul complet cu toate
feature-urile calculate, gata de inserare în DB.
Apelează în ordine:
1. Toate funcțiile din parser.py
2. parse_boolean_features()
3. compute_seismic_risk() și compute_is_post_1977()
4. get_coords_from_address() → lat, lon
5. point_in_neighborhood(lat, lon) → neighborhood (cartier REAL)
6. get_zone(neighborhood) → zone
7. get_nearest_metro(lat, lon) → dist_metro_m, nearest_metro
8. get_distance_to_center(lat, lon) → dist_center_m
9. Calculează price_per_sqm = price_eur / area_sqm

---

## database/schema.sql

### Ce face
Definește structura bazei de date SQLite. Trei tabele principale.

### Tabele de implementat

**`Neighborhoods`**
```sql
CREATE TABLE IF NOT EXISTS Neighborhoods (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT UNIQUE NOT NULL,  -- "Floreasca", "Armeneasca" etc.
    zone            TEXT,                  -- "Nord", "Sud", "Est", "Vest", "Centru"
    avg_price_sqm   REAL                   -- actualizat periodic cu AVG din Listings
);
```

**`Listings`** — tabelul principal, toate câmpurile pentru model
```sql
CREATE TABLE IF NOT EXISTS Listings (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    url                  TEXT UNIQUE NOT NULL,
    neighborhood_id      INTEGER REFERENCES Neighborhoods(id),

    -- Preț (target pentru model)
    price_eur            REAL,
    price_per_sqm        REAL,   -- TARGET: price_eur / area_sqm

    -- Proprietate — câmpuri parsate
    title                TEXT,
    area_sqm             REAL,
    rooms                INTEGER,
    floor                INTEGER,
    total_floors         INTEGER,
    year_built           INTEGER,
    compartmentare       TEXT,    -- decomandat/semidecomandat/nedecomandat/circular

    -- Features booleane (0/1) din caracteristici + descriere
    has_parking          INTEGER DEFAULT 0,
    has_balcony          INTEGER DEFAULT 0,
    has_elevator         INTEGER DEFAULT 0,
    has_ac               INTEGER DEFAULT 0,
    has_central_heating  INTEGER DEFAULT 0,
    has_storage          INTEGER DEFAULT 0,
    is_renovated         INTEGER DEFAULT 0,
    is_furnished         INTEGER DEFAULT 0,

    -- Features seismice
    seismic_risk         TEXT,    -- very_high/high/medium/low/unknown
    is_post_1977         INTEGER, -- 1 dacă year_built > 1977

    -- Features geografice (calculate din coordonate, nu din textul vânzătorului)
    lat                  REAL,
    lon                  REAL,
    nearest_metro        TEXT,
    dist_metro_m         REAL,
    dist_center_m        REAL,

    -- Date brute (pentru debugging și re-parsare fără re-scraping)
    description          TEXT,
    address_raw          TEXT,
    neighborhood_raw     TEXT,   -- ce A SCRIS vânzătorul (nu se folosește ca feature)
    price_raw            TEXT,
    details_raw_json     TEXT,   -- details_raw dict serializat ca JSON

    scraped_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    geocoded_at          TIMESTAMP  -- NULL până când rulează jobul de geocoding
);
```

**Indecși** — esențiali pentru performanța query-urilor:
```sql
CREATE INDEX IF NOT EXISTS idx_neighborhood  ON Listings(neighborhood_id);
CREATE INDEX IF NOT EXISTS idx_price_sqm     ON Listings(price_per_sqm);
CREATE INDEX IF NOT EXISTS idx_year_built    ON Listings(year_built);
CREATE INDEX IF NOT EXISTS idx_dist_metro    ON Listings(dist_metro_m);
CREATE INDEX IF NOT EXISTS idx_geocoded      ON Listings(geocoded_at);
```

---

## database/db_manager.py

### Ce face
Layer de acces la baza de date. Toate operațiunile SQL trec prin acest modul.
Restul codului nu scrie SQL direct.

### Funcții de implementat

**`init_db(db_path: str = "real_estate.db") -> sqlite3.Connection`**
Citește schema.sql și execută scriptul. Returnează conexiunea.

**`get_connection(db_path: str = "real_estate.db") -> sqlite3.Connection`**
Returnează o conexiune cu `row_factory = sqlite3.Row` (accesul la coloane
prin nume, nu index).

**`upsert_neighborhood(conn, name: str, zone: str | None) -> int`**
INSERT OR IGNORE în Neighborhoods, returnează id-ul rândului (existent sau nou).

**`insert_listing(conn, listing: dict) -> bool`**
INSERT OR IGNORE în Listings. Apelează upsert_neighborhood() intern
pentru a obține neighborhood_id. Serializează details_raw ca JSON.
Returnează True dacă s-a inserat, False dacă URL-ul exista deja.

**`get_ungeocoded_listings(conn, limit: int = 100) -> list`**
SELECT * FROM Listings WHERE geocoded_at IS NULL LIMIT limit.
Folosit de jobul de geocoding.

**`update_geocoding(conn, listing_id: int, lat: float, lon: float, neighborhood: str, zone: str, dist_metro_m: float, nearest_metro: str, dist_center_m: float) -> None`**
UPDATE Listings SET lat=?, lon=?, neighborhood_id=?, dist_metro_m=?, ...
WHERE id=?. Setează geocoded_at = CURRENT_TIMESTAMP.

**`get_listings_for_model(conn) -> pd.DataFrame`**
SELECT cu JOIN pe Neighborhoods, returnează DataFrame cu toate
feature-urile necesare pentru antrenare. Filtrează rândurile cu
price_per_sqm IS NULL sau lat IS NULL.

---

## pipeline.py

### Ce face
Orchestrează toate fazele proiectului. Script principal de rulat.
Acceptă argumente din linia de comandă pentru a rula doar anumite faze.

### Cum funcționează

```python
# Rulare completă:
python pipeline.py --all

# Doar scraping (fără geocoding):
python pipeline.py --scrape --pages 50

# Doar geocoding (pentru rândurile fără lat/lon):
python pipeline.py --geocode --batch 100

# Afișare statistici DB:
python pipeline.py --stats
```

### Faza 1: Scraping
```
pentru fiecare pagină 1..N:
    urls = get_listing_urls(page)
    pentru fiecare url:
        raw = scrape_listing(url)
        dacă raw nu e None:
            insert_listing(conn, raw)
            # lat/lon rămân NULL, geocoded_at rămâne NULL
```

### Faza 2: Geocoding (job separat, rulat după scraping)
```
cât timp există rânduri cu geocoded_at IS NULL:
    listings = get_ungeocoded_listings(conn, limit=100)
    pentru fiecare listing:
        coords = get_coords_from_address(listing["address_raw"])
        dacă coords:
            neighborhood = point_in_neighborhood(*coords)
            ...
            update_geocoding(conn, listing["id"], ...)
        sleep(1)  # respectă limita GeoPy
```

### Faza 3: Feature engineering
Aceasta rulează automat în insert_listing() — parser.py și features.py
sunt apelate înainte de inserare, deci DB-ul conține întotdeauna câmpurile
parsate. Singura excepție sunt câmpurile geografice (lat, lon, dist_metro etc.)
care se populează în faza de geocoding.

---

## model/train.py

### Ce face
Antrenează modelul XGBoost pe datele din SQLite și salvează modelul antrenat.

### Ce trebuie să conțină

**Features pentru model** (în ordinea importanței estimate):
```python
NUMERIC_FEATURES = [
    "area_sqm", "rooms", "floor", "total_floors",
    "year_built", "is_post_1977",
    "dist_metro_m", "dist_center_m",
    "has_parking", "has_balcony", "has_elevator",
    "has_ac", "has_central_heating", "is_renovated",
]

CATEGORICAL_FEATURES = [
    "neighborhood",      # cartierul real din geocoding
    "zone",              # Nord/Sud/Est/Vest/Centru
    "compartmentare",    # decomandat etc.
    "seismic_risk",      # very_high/high/medium/low
    "nearest_metro",     # stația de metrou cea mai apropiată
]

TARGET = "price_per_sqm"
```

**Pași de implementat:**
1. `get_listings_for_model(conn)` → DataFrame
2. One-hot encoding pentru CATEGORICAL_FEATURES
3. Imputare mediană pentru valorile lipsă în NUMERIC_FEATURES
4. Train/test split 80/20 cu random_state=42
5. Antrenare XGBoost cu hiperparametri de bază:
   ```python
   model = xgb.XGBRegressor(
       n_estimators=500, learning_rate=0.05,
       max_depth=6, subsample=0.8,
       colsample_bytree=0.8, random_state=42
   )
   ```
6. Evaluare: MAE, RMSE, R² pe test set
7. Salvare model: `pickle.dump(model, open("model/model.pkl", "wb"))`
8. Salvare lista de features (pentru a reconstrui DataFrame în Streamlit):
   `json.dump(feature_cols, open("model/feature_cols.json", "w"))`
9. Plot feature importance (top 15) salvat ca `model/feature_importance.png`

---

## app/streamlit_app.py

### Ce face
Interfață web interactivă pentru predicție în timp real.
Utilizatorul introduce caracteristicile apartamentului și primește
estimarea prețului/m² și valoarea totală estimată.

### Ce trebuie să conțină

**Layout:**
- Titlu: "Evaluare Imobiliară București"
- Două coloane de input:
  - Stânga: area_sqm, rooms, floor, year_built
  - Dreapta: neighborhood (dropdown), dist_metro_m (slider 0-3000m),
    has_parking, has_balcony (checkboxes)
- Buton "Estimează prețul"
- Output: st.metric pentru preț/m² și valoare totală
- Grafic feature importance din `model/feature_importance.png`

**Logică:**
1. Încarcă modelul din `model/model.pkl` cu `@st.cache_resource`
2. Încarcă `model/feature_cols.json`
3. La click pe buton:
   - Construiește DataFrame cu aceleași coloane ca la antrenare
   - Aplică one-hot encoding identic
   - Completează cu 0 coloanele lipsă
   - `price_per_sqm = model.predict(input_df)[0]`
   - Afișează rezultatul cu `st.metric`

---

## Statusul implementării

Bifează pe măsură ce completezi:

```
[ ] requirements.txt
[ ] scraper/storia_scraper.py
[ ] geocoding/neighborhoods.py
[ ] geocoding/geocoding.py
[ ] processing/parser.py
[ ] processing/features.py
[ ] database/schema.sql
[ ] database/db_manager.py
[ ] pipeline.py
[ ] model/train.py
[ ] app/streamlit_app.py
```

---

## Cum să rulezi proiectul

```bash
# 1. Instalează dependențele
pip install -r requirements.txt

# 2. Rulează scraping-ul (50 de pagini = ~1000 anunțuri)
python pipeline.py --scrape --pages 50

# 3. Rulează geocodingul pe datele colectate
python pipeline.py --geocode

# 4. Verifică statisticile
python pipeline.py --stats

# 5. Antrenează modelul (după ce ai minim 500 de anunțuri geocodate)
python model/train.py

# 6. Pornește aplicația Streamlit
streamlit run app/streamlit_app.py
```

---

## Debugging frecvent

**Scraperul returnează None pentru toate anunțurile:**
Selectoarele CSS s-au schimbat. Deschide un anunț în browser, F12,
inspectează elementele și actualizează selectoarele în storia_scraper.py.

**GeoPy returnează None pentru adrese:**
Adresa e prea vagă sau formatată ciudat. Verifică `_build_search_candidates()`
și adaugă mai multe variante de curățare.

**point_in_neighborhood() returnează None:**
Coordonatele sunt în afara poligoanelor definite. Fie poligonul e prea mic,
fie adresa a fost geocodată greșit. Verifică cu `_nearest_neighborhood_centroid()`.

**Modelul are MAE mare:**
Verifică distribuția lui price_per_sqm — probabil sunt outliere (anunțuri cu
prețuri greșite). Aplică IQR filtering: elimină rândurile unde
price_per_sqm < Q1 - 1.5*IQR sau > Q3 + 1.5*IQR înainte de antrenare.