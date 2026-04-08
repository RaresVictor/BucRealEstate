"""
Bucharest Real Estate Valuation App
Design: Teal/Slate — professional real estate analytics
"""

import json
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "modelML")
DB_PATH   = os.path.join(os.path.dirname(__file__), "..", "real_estate.db")

# ── Design system ────────────────────────────────────────────────────────────
# Colors: trust-teal primary, slate dark, clean light background
C_PRIMARY   = "#0F766E"
C_SECONDARY = "#14B8A6"
C_DARK      = "#134E4A"
C_BG        = "#F0FDFA"
C_BORDER    = "#CCFBF1"
C_MUTED     = "#5EEAD4"
C_GRAY      = "#6B7280"

# ── Custom CSS ───────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Josefin+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Josefin Sans', sans-serif !important; }
#MainMenu, footer, [data-testid="stToolbar"], .stDeployButton { display: none !important; }
[data-testid="stAppViewContainer"] { background: #F0FDFA; }

/* Header */
.re-header { text-align: center; padding: 2.5rem 1rem 1.75rem; border-bottom: 2px solid #CCFBF1; margin-bottom: 2rem; }
.re-header h1 { font-family: 'Cinzel', serif !important; font-size: 2.1rem; font-weight: 700; color: #0F766E; margin: 0; letter-spacing: 0.06em; }
.re-header p  { color: #0F766E; opacity: 0.7; font-size: 0.8rem; margin-top: 0.5rem; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 300; }

/* Section titles */
.sec-title { font-family: 'Cinzel', serif !important; font-size: 0.95rem; font-weight: 600; color: #0F766E; letter-spacing: 0.06em; padding-bottom: 0.5rem; border-bottom: 2px solid #CCFBF1; margin: 1.75rem 0 1rem; }

/* White card */
.card { background: #FFFFFF; border-radius: 14px; border: 1px solid #CCFBF1; padding: 1.25rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(15,118,110,0.07); }

/* Verdict banner */
.verdict-wrap { border-radius: 14px; padding: 1.5rem 2rem; display: flex; align-items: center; justify-content: space-around; gap: 1rem; margin-bottom: 0.75rem; }
.verdict-over  { background: linear-gradient(135deg,#FEF2F2,#FECACA); border: 1px solid #FCA5A5; }
.verdict-under { background: linear-gradient(135deg,#F0FDFA,#CCFBF1); border: 1px solid #5EEAD4; }
.verdict-fair  { background: linear-gradient(135deg,#FEFCE8,#FEF08A); border: 1px solid #FDE047; }
.v-block { text-align: center; }
.v-label { font-size: 0.65rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #6B7280; margin-bottom: 0.3rem; }
.v-value { font-size: 1.9rem; font-weight: 700; color: #134E4A; line-height: 1.1; }
.v-sub   { font-size: 0.72rem; color: #6B7280; margin-top: 0.2rem; }
.v-sep   { width: 1px; height: 56px; background: rgba(0,0,0,0.1); }
.v-badge { font-family: 'Cinzel', serif; font-size: 1rem; font-weight: 700; padding: 0.45rem 1.1rem; border-radius: 8px; white-space: nowrap; }
.badge-over  { color: #DC2626; background: rgba(220,38,38,0.1); }
.badge-under { color: #0F766E; background: rgba(15,118,110,0.12); }
.badge-fair  { color: #854D0E; background: rgba(133,77,14,0.1); }

/* Total info bar */
.total-bar { background: #FFFFFF; border: 1px solid #CCFBF1; border-radius: 10px; padding: 0.75rem 1.25rem; font-size: 0.85rem; color: #134E4A; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
.total-bar span { font-weight: 600; }
.total-bar .diff-pos { color: #DC2626; }
.total-bar .diff-neg { color: #0F766E; }

/* Detail grid */
.detail-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.65rem; margin-bottom: 0.65rem; }
.detail-grid-2 { grid-template-columns: repeat(4,1fr); }
.d-item { background: #F0FDFA; border: 1px solid #CCFBF1; border-radius: 10px; padding: 0.75rem 1rem; }
.d-label { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #0F766E; margin-bottom: 0.25rem; }
.d-value { font-size: 1rem; font-weight: 600; color: #134E4A; }

/* Feature pills */
.pills { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.75rem; }
.pill         { background: #CCFBF1; color: #0F766E; border: 1px solid #99F6E4; border-radius: 20px; padding: 0.2rem 0.7rem; font-size: 0.72rem; font-weight: 500; }
.pill-warn    { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
.pill-riskVH  { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
.pill-riskH   { background: #FFEDD5; color: #9A3412; border: 1px solid #FED7AA; }
.pill-riskM   { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
.pill-riskL   { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }

/* Stat row */
.stat-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.65rem; margin-bottom: 1rem; }
.s-item  { text-align: center; background: #F0FDFA; border: 1px solid #CCFBF1; border-radius: 10px; padding: 0.875rem 0.5rem; }
.s-value { font-size: 1.25rem; font-weight: 700; color: #0F766E; }
.s-label { font-size: 0.62rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7280; margin-top: 0.2rem; }

/* Percentile caption */
.pct-note { background: #F0FDFA; border-left: 3px solid #0F766E; border-radius: 0 8px 8px 0; padding: 0.6rem 1rem; font-size: 0.82rem; color: #134E4A; margin-bottom: 1rem; }

/* Input */
.stTextInput input { border-radius: 10px !important; border: 1.5px solid #5EEAD4 !important; background: #FFFFFF !important; font-family: 'Josefin Sans', sans-serif !important; color: #134E4A !important; }
.stTextInput input:focus { border-color: #0F766E !important; box-shadow: 0 0 0 3px rgba(15,118,110,0.15) !important; }
.stTextInput label { font-weight: 600 !important; color: #134E4A !important; letter-spacing: 0.06em !important; font-size: 0.75rem !important; text-transform: uppercase !important; }

/* Button */
.stButton > button { background: linear-gradient(135deg,#0F766E,#14B8A6) !important; color: #FFF !important; border: none !important; border-radius: 10px !important; font-family: 'Josefin Sans', sans-serif !important; font-weight: 600 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; font-size: 0.82rem !important; padding: 0.6rem 2rem !important; box-shadow: 0 2px 10px rgba(15,118,110,0.28) !important; transition: all 0.2s ease !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 18px rgba(15,118,110,0.38) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* Spinner */
.stSpinner > div { border-top-color: #0F766E !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #CCFBF1 !important; }

/* Divider */
hr { border-color: #CCFBF1 !important; margin: 1.5rem 0 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #134E4A !important; }
[data-testid="stSidebar"] * { color: #CCFBF1 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color: #FFF !important; font-family: 'Cinzel', serif !important; }
[data-testid="stSidebar"] [data-testid="metric-container"] { background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(94,234,212,0.25) !important; border-radius: 10px !important; }
[data-testid="stSidebar"] [data-testid="metric-container"] label { color: #5EEAD4 !important; font-size: 0.7rem !important; letter-spacing: 0.08em !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #FFF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(94,234,212,0.25) !important; }
[data-testid="stSidebar"] .stCaption { color: rgba(204,251,241,0.6) !important; }
</style>
"""

# ── Chart style ───────────────────────────────────────────────────────────────
def _chart_style():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "axes.grid": False,   # controlled per-axis with ax.grid(axis=...)
        "grid.alpha": 0.45,
        "grid.color": "#CCFBF1",
        "xtick.color": "#6B7280",
        "ytick.color": "#6B7280",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelcolor": "#6B7280",
        "axes.labelsize": 9,
    })


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    def _load(name):
        with open(os.path.join(MODEL_DIR, name), "rb") as f:
            return pickle.load(f)
    model    = _load("model.pkl")
    q10      = _load("model_q10.pkl")
    q90      = _load("model_q90.pkl")
    imputer  = _load("imputer.pkl")
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return model, q10, q90, imputer, meta


@st.cache_data(ttl=3600)
def load_market_data() -> pd.DataFrame:
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT l.price_per_sqm, l.area_sqm, l.rooms, l.year_built,
               l.floor, l.total_floors, l.dist_metro_m, l.address_raw,
               l.has_parking, l.has_balcony, l.is_renovated, l.is_furnished,
               l.nearest_metro, l.seismic_risk,
               n.name AS neighborhood, n.zone
        FROM Listings l
        LEFT JOIN Neighborhoods n ON l.neighborhood_id = n.id
        WHERE l.price_per_sqm IS NOT NULL
          AND l.lat IS NOT NULL AND l.lat != -1
        """,
        conn,
    )
    conn.close()
    return df


# ── Prediction ────────────────────────────────────────────────────────────────
def _feature_row(listing: dict, meta: dict) -> pd.DataFrame:
    feature_cols      = meta["feature_cols"]
    numeric_features  = meta["numeric_features"]
    categorical_features = meta["categorical_features"]
    row = {col: np.nan for col in feature_cols}
    for col in numeric_features:
        v = listing.get(col)
        if v is not None:
            row[col] = float(v)
    if listing.get("dist_metro_m") is not None:
        row["log_dist_metro"] = np.log1p(float(listing["dist_metro_m"]))
    if listing.get("dist_center_m") is not None:
        row["log_dist_center"] = np.log1p(float(listing["dist_center_m"]))
    for cat in categorical_features:
        val = listing.get(cat)
        if val:
            oh = f"{cat}_{val}"
            if oh in row:
                row[oh] = 1.0
            for col in feature_cols:
                if col.startswith(f"{cat}_") and col != oh and np.isnan(row.get(col, np.nan)):
                    row[col] = 0.0
    return pd.DataFrame([row])[feature_cols]


def predict_interval(listing, model, q10, q90, imputer, meta):
    df_row = _feature_row(listing, meta)
    df_imp = pd.DataFrame(imputer.transform(df_row), columns=meta["feature_cols"])
    pred = float(model.predict(df_imp)[0])
    lo   = float(q10.predict(df_imp)[0])
    hi   = float(q90.predict(df_imp)[0])
    return pred, min(lo, pred), max(hi, pred)


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _histogram(df_neigh: pd.DataFrame, listed: float | None, predicted: float):
    _chart_style()
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.hist(df_neigh["price_per_sqm"], bins=28, color="#5EEAD4", edgecolor="white",
            linewidth=0.5, alpha=0.85, zorder=2)
    ax.axvline(predicted, color=C_PRIMARY, linewidth=2.5, zorder=3,
               label=f"Estimare  {predicted:,.0f} €/m²")
    if listed:
        ax.axvline(listed, color="#DC2626", linewidth=2.5, linestyle="--", zorder=3,
                   label=f"Cerut  {listed:,.0f} €/m²")
    ax.grid(axis="y", alpha=0.45, color=C_BORDER)
    ax.spines["bottom"].set_color(C_BORDER)
    ax.set_xlabel("€/m²")
    ax.set_ylabel("Nr. anunțuri")
    ax.legend(fontsize=9, framealpha=0.95, edgecolor=C_BORDER)
    plt.tight_layout(pad=0.5)
    return fig


def _zone_bar(df_all: pd.DataFrame, current_nb: str | None):
    _chart_style()
    med = (
        df_all.groupby("neighborhood")["price_per_sqm"]
        .median().sort_values().dropna()
    )
    colors = [C_PRIMARY if idx == current_nb else "#CCFBF1" for idx in med.index]
    edge   = [C_DARK    if idx == current_nb else C_BORDER  for idx in med.index]
    fig, ax = plt.subplots(figsize=(8, max(4, len(med) * 0.3)))
    bars = ax.barh(med.index, med.values, color=colors, edgecolor=edge, linewidth=0.8)
    # Label the highlighted bar
    for bar, idx in zip(bars, med.index):
        if idx == current_nb:
            ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():,.0f}", va="center", ha="left",
                    fontsize=8.5, color=C_DARK, fontweight="600")
    ax.grid(axis="x", alpha=0.45, color=C_BORDER)
    ax.spines["bottom"].set_color(C_BORDER)
    ax.set_xlabel("Preț median €/m²")
    plt.tight_layout(pad=0.5)
    return fig


# ── HTML helpers ──────────────────────────────────────────────────────────────
def _verdict_html(listed: float | None, pred: float, lo: float, hi: float) -> str:
    if listed:
        pct = (listed - pred) / pred * 100
        if pct > 10:
            cls, badge_cls, badge_txt = "verdict-over",  "badge-over",  f"Supraevaluat {pct:+.1f}%"
        elif pct < -10:
            cls, badge_cls, badge_txt = "verdict-under", "badge-under", f"Sub piață {pct:+.1f}%"
        else:
            cls, badge_cls, badge_txt = "verdict-fair",  "badge-fair",  f"Preț corect {pct:+.1f}%"
    else:
        cls, badge_cls, badge_txt = "verdict-fair", "badge-fair", "—"

    listed_html = f"""
        <div class="v-block">
            <div class="v-label">Preț cerut</div>
            <div class="v-value">{listed:,.0f}</div>
            <div class="v-sub">€/m²</div>
        </div>
        <div class="v-sep"></div>""" if listed else ""

    return f"""
    <div class="verdict-wrap {cls}">
        {listed_html}
        <div class="v-block">
            <div class="v-label">Estimare model</div>
            <div class="v-value">{pred:,.0f}</div>
            <div class="v-sub">€/m² &nbsp;·&nbsp; interval {lo:,.0f} – {hi:,.0f}</div>
        </div>
        <div class="v-sep"></div>
        <div class="v-block">
            <div class="v-label">Verdict</div>
            <div class="v-badge {badge_cls}">{badge_txt}</div>
        </div>
    </div>"""


def _detail_item(label: str, value) -> str:
    return f"""<div class="d-item"><div class="d-label">{label}</div>
               <div class="d-value">{value if value not in (None, "?") else "—"}</div></div>"""


def _stat_item(value: str, label: str) -> str:
    return f"""<div class="s-item"><div class="s-value">{value}</div>
               <div class="s-label">{label}</div></div>"""


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluare Imobiliară București",
    page_icon="🏠",
    layout="centered",
)
st.html(CSS)

model, q10, q90, imputer, meta = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.html("""
<div class="re-header">
    <h1>Evaluare Imobiliară</h1>
    <p>București &nbsp;·&nbsp; Estimare independentă bazată pe date reale</p>
</div>
""")

# ── Input ─────────────────────────────────────────────────────────────────────
url = st.text_input(
    "Link anunț Storia.ro",
    placeholder="https://www.storia.ro/ro/oferta/...",
)

if url and not url.startswith("https://www.storia.ro/ro/oferta/"):
    st.error("Inserează un link valid de pe storia.ro/ro/oferta/...")
    st.stop()

analyze = st.button("Analizează", type="primary", use_container_width=False)

if url and analyze:
    # ── Scrape ────────────────────────────────────────────────────────────────
    with st.spinner("Se preiau datele anunțului..."):
        try:
            from scraper.storia_scraper import scrape_listing
            from processing.features import enrich_listing
            from geocoding.geocoding import (
                point_in_neighborhood, get_zone,
                get_nearest_metro, get_distance_to_center, validate_coords,
            )
            import sqlite3
            raw = scrape_listing(url.strip())
        except Exception as e:
            st.error(f"Eroare la preluarea anunțului: {e}")
            st.stop()

    if raw is None:
        st.error("Nu am putut accesa anunțul. Verifică link-ul și încearcă din nou.")
        st.stop()

    with st.spinner("Se procesează și se estimează prețul..."):
        enriched = enrich_listing(raw)
        lat, lon = enriched.get("lat"), enriched.get("lon")
        if lat and lon:
            enriched["neighborhood"] = point_in_neighborhood(lat, lon)
            enriched["zone"]         = get_zone(enriched["neighborhood"])
            enriched["dist_metro_m"], enriched["nearest_metro"] = get_nearest_metro(lat, lon)
            enriched["dist_center_m"] = get_distance_to_center(lat, lon)

        pred, lo, hi = predict_interval(enriched, model, q10, q90, imputer, meta)
        listed_psqm  = enriched.get("price_per_sqm")
        listed_price = enriched.get("price_eur")
        area         = enriched.get("area_sqm")
        neighborhood = enriched.get("neighborhood")
        zone         = enriched.get("zone")
        rooms        = enriched.get("rooms")

    # ── Title ─────────────────────────────────────────────────────────────────
    st.divider()
    st.html(f"""
    <div class="card">
        <div style="font-family:'Cinzel',serif;font-size:1.1rem;font-weight:600;color:#134E4A;">
            {raw.get("title") or "Anunț"}
        </div>
        <div style="font-size:0.8rem;color:#6B7280;margin-top:0.3rem;letter-spacing:0.04em;">
            📍 {enriched.get("address_raw") or "Adresă necunoscută"}
        </div>
    </div>
    """)

    # ── Verdict ───────────────────────────────────────────────────────────────
    st.html(_verdict_html(listed_psqm, pred, lo, hi))

    if area and listed_price:
        est_total  = pred * area
        diff_total = listed_price - est_total
        diff_cls   = "diff-pos" if diff_total > 0 else "diff-neg"
        st.html(f"""
        <div class="total-bar">
            <span>Preț total cerut: <b>{listed_price:,.0f} €</b></span>
            <span>Estimare totală: <b>{est_total:,.0f} €</b></span>
            <span class="{diff_cls}">Diferență: <b>{diff_total:+,.0f} €</b></span>
        </div>
        """)

    # ── Apartment details ──────────────────────────────────────────────────────
    st.html('<div class="sec-title">Detalii apartament</div>')

    floor       = enriched.get("floor")
    total_floors = enriched.get("total_floors")
    floor_str   = f"{floor}/{total_floors}" if floor is not None and total_floors else (str(floor) if floor is not None else None)
    dist_m      = enriched.get("dist_metro_m")

    st.html(f"""
    <div class="detail-grid">
        {_detail_item("Suprafață", f'{area:.0f} m²' if area else None)}
        {_detail_item("Camere", rooms)}
        {_detail_item("Etaj", floor_str)}
        {_detail_item("An construcție", enriched.get("year_built"))}
    </div>
    <div class="detail-grid">
        {_detail_item("Cartier", neighborhood)}
        {_detail_item("Zonă", zone)}
        {_detail_item("Metrou apropiat", enriched.get("nearest_metro"))}
        {_detail_item("Dist. metrou", f'{dist_m:.0f} m' if dist_m else None)}
    </div>
    """)

    # Feature pills
    pills = []
    if enriched.get("has_parking"):     pills.append('<span class="pill">Parcare</span>')
    if enriched.get("has_balcony"):     pills.append('<span class="pill">Balcon</span>')
    if enriched.get("has_elevator"):    pills.append('<span class="pill">Lift</span>')
    if enriched.get("has_ac"):          pills.append('<span class="pill">Aer condiționat</span>')
    if enriched.get("is_renovated"):    pills.append('<span class="pill">Renovat</span>')
    if enriched.get("is_furnished"):    pills.append('<span class="pill">Mobilat</span>')
    if enriched.get("is_new_build"):    pills.append('<span class="pill">Construcție nouă</span>')
    if enriched.get("is_cgi_listing"):  pills.append('<span class="pill pill-warn">Poze orientative</span>')

    risk = enriched.get("seismic_risk")
    risk_map = {
        "very_high": ("pill-riskVH", "Risc seismic foarte ridicat"),
        "high":      ("pill-riskH",  "Risc seismic ridicat"),
        "medium":    ("pill-riskM",  "Risc seismic mediu"),
        "low":       ("pill-riskL",  "Risc seismic scăzut"),
    }
    if risk and risk in risk_map:
        rc, rt = risk_map[risk]
        pills.append(f'<span class="pill {rc}">{rt}</span>')

    if pills:
        st.html(f'<div class="pills">{"".join(pills)}</div>')

    # ── Market statistics ──────────────────────────────────────────────────────
    df_all   = load_market_data()
    df_neigh = df_all[df_all["neighborhood"] == neighborhood] if neighborhood else pd.DataFrame()

    # 1. Piața în cartier
    if neighborhood and len(df_neigh) >= 5:
        st.html(f'<div class="sec-title">Piața în {neighborhood}</div>')

        med_nb = df_neigh["price_per_sqm"].median()
        pmin   = df_neigh["price_per_sqm"].min()
        pmax   = df_neigh["price_per_sqm"].max()

        st.html(f"""
        <div class="stat-row">
            {_stat_item(f"{med_nb:,.0f} €", "Median €/m²")}
            {_stat_item(str(len(df_neigh)), "Anunțuri")}
            {_stat_item(f"{pmin:,.0f} €", "Minim €/m²")}
            {_stat_item(f"{pmax:,.0f} €", "Maxim €/m²")}
        </div>
        """)

        if listed_psqm:
            pct = (df_neigh["price_per_sqm"] < listed_psqm).mean() * 100
            st.html(f"""
            <div class="pct-note">
                Prețul cerut este mai mare decât <b>{pct:.0f}%</b>
                din apartamentele listate în {neighborhood}.
            </div>
            """)

        st.pyplot(_histogram(df_neigh, listed_psqm, pred), use_container_width=True)

    # 2. Comparabile
    st.html('<div class="sec-title">Apartamente similare</div>')

    comp = df_all.copy()
    if rooms:
        comp = comp[comp["rooms"] == rooms]
    if area:
        comp = comp[(comp["area_sqm"] >= area * 0.75) & (comp["area_sqm"] <= area * 1.25)]

    comp_nb   = comp[comp["neighborhood"] == neighborhood] if neighborhood else pd.DataFrame()
    comp_zone = comp[comp["zone"] == zone] if zone else pd.DataFrame()

    if len(comp_nb) >= 3:
        comp_show, comp_scope = comp_nb,   f"cartierul {neighborhood}"
    elif len(comp_zone) >= 3:
        comp_show, comp_scope = comp_zone, f"zona {zone}"
    else:
        comp_show, comp_scope = comp,      "București"

    if len(comp_show) > 0:
        st.caption(f"{len(comp_show)} apartamente cu {rooms or '?'} camere și suprafață similară în {comp_scope}.")
        display = (
            comp_show[["address_raw", "area_sqm", "price_per_sqm", "year_built", "floor", "dist_metro_m"]]
            .rename(columns={
                "address_raw":   "Adresă",
                "area_sqm":      "m²",
                "price_per_sqm": "€/m²",
                "year_built":    "An",
                "floor":         "Etaj",
                "dist_metro_m":  "Metrou (m)",
            })
            .sort_values("€/m²")
            .head(10)
            .reset_index(drop=True)
        )
        display["€/m²"]       = display["€/m²"].map("{:,.0f}".format)
        display["Metrou (m)"] = display["Metrou (m)"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.caption("Nu sunt suficiente date comparabile în baza de date.")

    # 3. Prețuri pe cartiere
    st.html('<div class="sec-title">Prețuri mediane pe cartiere</div>')
    st.pyplot(_zone_bar(df_all, neighborhood), use_container_width=True)

    # ── Save to DB ────────────────────────────────────────────────────────────
    st.divider()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    exists = conn.execute("SELECT id FROM Listings WHERE url = ?", (url.strip(),)).fetchone()
    conn.close()

    if exists:
        st.success("Acest anunț este deja în baza de date.")
    else:
        if st.button("Adaugă în baza de date"):
            try:
                from database.db_manager import get_connection, insert_listing
                conn = get_connection(DB_PATH)
                insert_listing(conn, enriched)
                conn.close()
                st.success("Anunț adăugat în baza de date!")
            except Exception as e:
                st.error(f"Eroare la salvare: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model")
    m = meta["metrics"]
    st.metric("R²",   f"{m['r2']:.3f}")
    st.metric("MAE",  f"{m['mae']:.0f} €/m²")
    st.metric("MAPE", f"{m['mape']:.1f}%")

    iv = meta.get("interval", {})
    if iv:
        st.metric(
            "Acoperire interval",
            f"{iv['coverage']:.0%}",
            help="% cazuri din test unde prețul real a căzut în intervalul estimat.",
        )

    df_sb = load_market_data()
    st.divider()
    st.caption(f"Antrenat pe {len(df_sb):,} anunțuri din București")
    st.caption("XGBoost · 500 estimatori · IQR filtered")

    img = os.path.join(MODEL_DIR, "feature_importance.png")
    if os.path.exists(img):
        st.divider()
        st.markdown("### Feature Importance")
        st.image(img)
