"""
Bucharest Real Estate Valuation App
Design: Teal/Slate — professional real estate analytics
"""

import html as html_lib
import json
import os
import pickle
import sqlite3
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

/* Numbers never jiggle: tabular figures everywhere data lives */
.v-value, .s-value, .d-value, .total-bar, .range-lab { font-variant-numeric: tabular-nums; }

/* Motion: subtle entrance, fully disabled for reduced-motion users */
@media (prefers-reduced-motion: no-preference) {
  .card, .verdict-wrap, .total-bar, .step-card { animation: rise 0.25s ease-out both; }
  @keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}

/* Header */
.re-header { text-align: center; padding: 2.5rem 1rem 1.75rem; border-bottom: 2px solid #CCFBF1; margin-bottom: 2rem; }
.re-header h1 { font-family: 'Cinzel', serif !important; font-size: 2.1rem; font-weight: 700; color: #0F766E; margin: 0; letter-spacing: 0.06em; }
.re-header p  { color: #0F766E; opacity: 0.7; font-size: 0.8rem; margin-top: 0.5rem; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 300; }

/* Section titles */
.sec-title { font-family: 'Cinzel', serif !important; font-size: 0.95rem; font-weight: 600; color: #0F766E; letter-spacing: 0.06em; padding-bottom: 0.5rem; border-bottom: 2px solid #CCFBF1; margin: 1.75rem 0 1rem; }

/* White card */
.card { background: #FFFFFF; border-radius: 14px; border: 1px solid #CCFBF1; padding: 1.25rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(15,118,110,0.07); }

/* Verdict banner */
.verdict-wrap { border-radius: 14px; padding: 1.5rem 2rem; display: flex; align-items: center; justify-content: space-around; gap: 1rem; margin-bottom: 0.75rem; flex-wrap: wrap; }
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

/* Detail grid — responsive: 4 cols desktop, 2 on narrow screens */
.detail-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.65rem; margin-bottom: 0.65rem; }
@media (max-width: 640px) { .detail-grid { grid-template-columns: repeat(2,1fr); } }
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

/* Stat row — responsive */
.stat-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.65rem; margin-bottom: 1rem; }
@media (max-width: 640px) { .stat-row { grid-template-columns: repeat(2,1fr); } }
.s-item  { text-align: center; background: #F0FDFA; border: 1px solid #CCFBF1; border-radius: 10px; padding: 0.875rem 0.5rem; }
.s-value { font-size: 1.25rem; font-weight: 700; color: #0F766E; }
.s-label { font-size: 0.62rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7280; margin-top: 0.2rem; }

/* Percentile caption */
.pct-note { background: #F0FDFA; border-left: 3px solid #0F766E; border-radius: 0 8px 8px 0; padding: 0.6rem 1rem; font-size: 0.82rem; color: #134E4A; margin-bottom: 1rem; }

/* Input */
.stTextInput input { border-radius: 10px !important; border: 1.5px solid #5EEAD4 !important; background: #FFFFFF !important; font-family: 'Josefin Sans', sans-serif !important; color: #134E4A !important; }
.stTextInput input:focus { border-color: #0F766E !important; box-shadow: 0 0 0 3px rgba(15,118,110,0.15) !important; }
.stTextInput label { font-weight: 600 !important; color: #134E4A !important; letter-spacing: 0.06em !important; font-size: 0.75rem !important; text-transform: uppercase !important; }

/* Button — hover lift, press feedback, visible keyboard focus */
.stButton > button { background: linear-gradient(135deg,#0F766E,#14B8A6) !important; color: #FFF !important; border: none !important; border-radius: 10px !important; font-family: 'Josefin Sans', sans-serif !important; font-weight: 600 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; font-size: 0.82rem !important; padding: 0.6rem 2rem !important; min-height: 44px !important; box-shadow: 0 2px 10px rgba(15,118,110,0.28) !important; transition: transform 0.15s ease, box-shadow 0.15s ease !important; cursor: pointer !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 18px rgba(15,118,110,0.38) !important; }
.stButton > button:active { transform: scale(0.98) !important; }
.stButton > button:focus-visible { outline: 3px solid #0369A1 !important; outline-offset: 2px !important; }
.stTextInput input:focus-visible { outline: 3px solid #0369A1 !important; outline-offset: 1px !important; }

/* Price-position range bar */
.range-wrap { background: #FFFFFF; border: 1px solid #CCFBF1; border-radius: 10px; padding: 1rem 1.25rem 1.4rem; margin-bottom: 0.75rem; }
.range-title { font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #6B7280; margin-bottom: 0.9rem; }
.range-track { position: relative; height: 10px; border-radius: 6px; background: linear-gradient(90deg,#99F6E4,#5EEAD4 50%,#99F6E4); }
.range-marker { position: absolute; top: 50%; transform: translate(-50%,-50%); width: 4px; height: 22px; border-radius: 2px; }
.range-marker.est { background: #0F766E; }
.range-marker.ask { background: #DC2626; }
.range-lab { position: absolute; transform: translateX(-50%); font-size: 0.68rem; font-weight: 600; white-space: nowrap; }
.range-lab.est { color: #0F766E; top: 1.1rem; }
.range-lab.ask { color: #DC2626; top: -1.35rem; }
.range-ends { display: flex; justify-content: space-between; font-size: 0.68rem; color: #6B7280; margin-top: 1.7rem; }

/* Empty state: how-it-works steps */
.step-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.75rem; margin-top: 1.5rem; }
@media (max-width: 640px) { .step-grid { grid-template-columns: 1fr; } }
.step-card { background: #FFFFFF; border: 1px solid #CCFBF1; border-radius: 14px; padding: 1.25rem; text-align: center; box-shadow: 0 1px 4px rgba(15,118,110,0.07); }
.step-icon { display: inline-flex; align-items: center; justify-content: center; width: 42px; height: 42px; border-radius: 12px; background: #F0FDFA; border: 1px solid #CCFBF1; color: #0F766E; margin-bottom: 0.6rem; }
.step-num { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #14B8A6; margin-bottom: 0.25rem; }
.step-title { font-family: 'Cinzel', serif; font-size: 0.92rem; font-weight: 600; color: #134E4A; margin-bottom: 0.35rem; }
.step-desc { font-size: 0.78rem; color: #6B7280; line-height: 1.5; }

/* Methodology / disclaimer note */
.method-note { display: flex; gap: 0.6rem; align-items: flex-start; background: #FFFFFF; border: 1px solid #CCFBF1; border-left: 3px solid #14B8A6; border-radius: 0 10px 10px 0; padding: 0.7rem 1rem; font-size: 0.78rem; color: #134E4A; line-height: 1.5; margin-bottom: 1rem; }
.method-note svg { flex-shrink: 0; margin-top: 0.1rem; color: #0F766E; }

/* Address line with icon */
.addr-line { display: flex; align-items: center; gap: 0.35rem; font-size: 0.8rem; color: #6B7280; margin-top: 0.35rem; letter-spacing: 0.04em; }
.addr-line svg { flex-shrink: 0; color: #0F766E; }

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
    q_lo     = _load("model_q_lo.pkl")
    q_hi     = _load("model_q_hi.pkl")
    imputer  = _load("imputer.pkl")
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return model, q_lo, q_hi, imputer, meta


@st.cache_data(ttl=3600)
def load_market_data() -> pd.DataFrame:
    from database.db_manager import get_connection
    conn = get_connection(DB_PATH)
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
          AND l.lat IS NOT NULL
          AND l.coords_failed_at IS NULL
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


def predict_interval(listing, model, q_lo, q_hi, imputer, meta):
    df_row = _feature_row(listing, meta)
    df_imp = pd.DataFrame(imputer.transform(df_row), columns=meta["feature_cols"])
    pred = float(model.predict(df_imp)[0])
    qhat = float(meta.get("interval", {}).get("conformal_offset", 0.0))
    lo   = float(q_lo.predict(df_imp)[0]) - qhat
    hi   = float(q_hi.predict(df_imp)[0]) + qhat
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
    med_all = (
        df_all.groupby("neighborhood")["price_per_sqm"]
        .median().dropna().sort_values(ascending=False)
    )
    # Data density: top 15 neighborhoods, always including the current one
    med = med_all.head(15)
    if current_nb and current_nb in med_all.index and current_nb not in med.index:
        med = pd.concat([med, med_all.loc[[current_nb]]])
    med = med.sort_values()
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
def _esc(value) -> str:
    """Escape scraped/external text before injecting it into st.html."""
    return html_lib.escape(str(value), quote=True)


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
    shown = _esc(value) if value not in (None, "?") else "—"
    return f"""<div class="d-item"><div class="d-label">{label}</div>
               <div class="d-value">{shown}</div></div>"""


def _stat_item(value: str, label: str) -> str:
    return f"""<div class="s-item"><div class="s-value">{value}</div>
               <div class="s-label">{label}</div></div>"""


# Inline SVG icons (Lucide outlines) — consistent 1.5px stroke, no emoji
_SVG = {
    "map-pin": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>',
    "link": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>',
    "chart": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M7 16v-5"/><path d="M12 16V8"/><path d="M17 16v-3"/></svg>',
    "badge": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z"/><path d="m9 12 2 2 4-4"/></svg>',
    "info": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
}


def _range_bar_html(listed: float | None, pred: float, lo: float, hi: float) -> str:
    """Visual position of the model estimate (and asking price) within the p5–p95 band."""
    span = max(hi - lo, 1.0)
    pad = span * 0.15
    lo_e, hi_e = lo - pad, hi + pad
    if listed:
        lo_e, hi_e = min(lo_e, listed - pad * 0.5), max(hi_e, listed + pad * 0.5)

    def pct(v: float) -> float:
        return max(3.0, min(97.0, (v - lo_e) / (hi_e - lo_e) * 100))

    ask_html = ""
    if listed:
        p = pct(listed)
        ask_html = (
            f'<div class="range-marker ask" style="left:{p:.1f}%;"></div>'
            f'<div class="range-lab ask" style="left:{p:.1f}%;">cerut {listed:,.0f}</div>'
        )
    pe = pct(pred)
    return f"""
    <div class="range-wrap">
        <div class="range-title">Poziția prețului în intervalul estimat (€/m²)</div>
        <div style="padding-top:1.4rem;">
            <div class="range-track">
                <div class="range-marker est" style="left:{pe:.1f}%;"></div>
                <div class="range-lab est" style="left:{pe:.1f}%;">estimare {pred:,.0f}</div>
                {ask_html}
            </div>
        </div>
        <div class="range-ends"><span>{lo:,.0f} (p5)</span><span>{hi:,.0f} (p95)</span></div>
    </div>"""


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluare Imobiliară București",
    page_icon="🏠",
    layout="centered",
)
st.html(CSS)

model, q_lo, q_hi, imputer, meta = load_model()

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
url_clean = (url or "").strip()

if url_clean and not url_clean.startswith("https://www.storia.ro/ro/oferta/"):
    st.error("Inserează un link valid de pe storia.ro/ro/oferta/...")
    st.stop()

analyze = st.button("Analizează", type="primary", use_container_width=False)

if url_clean and analyze:
    # ── Scrape ────────────────────────────────────────────────────────────────
    with st.spinner("Se preiau datele anunțului..."):
        try:
            from scraper.storia_scraper import scrape_listing
            raw = scrape_listing(url_clean)
        except Exception as e:
            st.error(f"Eroare la preluarea anunțului: {e}")
            st.stop()

    if raw is None:
        st.error("Nu am putut accesa anunțul. Verifică link-ul și încearcă din nou.")
        st.stop()

    with st.spinner("Se procesează și se estimează prețul..."):
        from processing.features import enrich_listing, compute_seismic_risk
        from geocoding.geocoding import (
            point_in_neighborhood, get_zone,
            get_nearest_metro, get_distance_to_center,
        )
        enriched = enrich_listing(raw)
        lat, lon = enriched.get("lat"), enriched.get("lon")
        if lat and lon:
            enriched["neighborhood"] = point_in_neighborhood(lat, lon)
            enriched["zone"]         = get_zone(enriched["neighborhood"])
            enriched["dist_metro_m"], enriched["nearest_metro"] = get_nearest_metro(lat, lon)
            enriched["dist_center_m"] = get_distance_to_center(lat, lon)
            # Seismic risk depends on the REAL neighborhood — recompute it
            enriched["seismic_risk"] = compute_seismic_risk(
                enriched.get("year_built"), enriched["neighborhood"]
            )

        pred, lo, hi = predict_interval(enriched, model, q_lo, q_hi, imputer, meta)

    # Persist across Streamlit reruns — otherwise any widget interaction
    # (e.g. the save button below) would wipe the results.
    st.session_state["analysis"] = {
        "url": url_clean, "raw": raw, "enriched": enriched,
        "pred": pred, "lo": lo, "hi": hi,
    }

_analysis = st.session_state.get("analysis")
if _analysis and _analysis["url"] == url_clean:
    raw      = _analysis["raw"]
    enriched = _analysis["enriched"]
    pred, lo, hi = _analysis["pred"], _analysis["lo"], _analysis["hi"]

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
            {_esc(raw.get("title") or "Anunț")}
        </div>
        <div class="addr-line">
            {_SVG["map-pin"]} {_esc(enriched.get("address_raw") or "Adresă necunoscută")}
        </div>
    </div>
    """)

    # ── Verdict ───────────────────────────────────────────────────────────────
    st.html(_verdict_html(listed_psqm, pred, lo, hi))
    st.html(_range_bar_html(listed_psqm, pred, lo, hi))
    st.html(f"""
    <div class="method-note">
        {_SVG["info"]}
        <span>Estimarea se bazează pe <b>prețurile cerute</b> în anunțurile de pe piață,
        nu pe prețuri finale de tranzacție. Prețul real de vânzare este de obicei
        cu câteva procente sub cel cerut.</span>
    </div>""")

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

    # 3. Prețuri pe cartiere (top 15 + cartierul curent)
    st.html('<div class="sec-title">Top cartiere după preț median</div>')
    st.pyplot(_zone_bar(df_all, neighborhood), use_container_width=True)

    # ── Save to DB ────────────────────────────────────────────────────────────
    st.divider()
    from database.db_manager import get_connection, insert_listing
    conn = get_connection(DB_PATH)
    exists = conn.execute(
        "SELECT id FROM Listings WHERE url = ?", (_analysis["url"],)
    ).fetchone()

    if exists:
        conn.close()
        st.success("Acest anunț este deja în baza de date.")
    else:
        if st.button("Adaugă în baza de date"):
            try:
                insert_listing(conn, enriched)
                st.success("Anunț adăugat în baza de date!")
                load_market_data.clear()
            except Exception as e:
                st.error(f"Eroare la salvare: {e}")
        conn.close()
else:
    # ── Empty state: explain the product instead of a blank page ─────────────
    n_listings = meta.get("n_train")
    compare_txt = (
        f"Comparăm apartamentul cu {n_listings:,} anunțuri reale din București"
        if n_listings else "Comparăm apartamentul cu anunțuri reale din București"
    )
    st.html(f"""
    <div class="step-grid">
        <div class="step-card">
            <div class="step-icon">{_SVG["link"]}</div>
            <div class="step-num">Pasul 1</div>
            <div class="step-title">Lipește link-ul</div>
            <div class="step-desc">Copiază adresa unui anunț de apartament de pe
            Storia.ro și lipește-o în câmpul de mai sus.</div>
        </div>
        <div class="step-card">
            <div class="step-icon">{_SVG["chart"]}</div>
            <div class="step-num">Pasul 2</div>
            <div class="step-title">Modelul analizează</div>
            <div class="step-desc">{compare_txt}: cartier real, distanță metrou,
            an construcție, dotări, risc seismic.</div>
        </div>
        <div class="step-card">
            <div class="step-icon">{_SVG["badge"]}</div>
            <div class="step-num">Pasul 3</div>
            <div class="step-title">Primești verdictul</div>
            <div class="step-desc">Estimare €/m² cu interval de încredere,
            plus comparație directă cu piața din zonă.</div>
        </div>
    </div>
    <div class="method-note" style="margin-top:1rem;">
        {_SVG["info"]}
        <span>Estimările se bazează pe <b>prețurile cerute</b> în anunțuri,
        nu pe prețuri finale de tranzacție — un instrument de comparație
        cu piața, nu o evaluare oficială.</span>
    </div>
    """)

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

    st.divider()
    n_train = meta.get("n_train")
    if n_train:
        st.caption(f"Antrenat pe {n_train:,} anunțuri din București")
    st.caption("XGBoost · 500 estimatori · IQR filtered")

    img = os.path.join(MODEL_DIR, "feature_importance.png")
    if os.path.exists(img):
        st.divider()
        st.markdown("### Feature Importance")
        st.image(img)
