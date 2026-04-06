"""
Bucharest Real Estate Valuation App
------------------------------------
Paste a Storia.ro listing URL → get an independent price estimate.
"""

import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "modelML")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "real_estate.db")

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_q10.pkl"), "rb") as f:
        model_q10 = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_q90.pkl"), "rb") as f:
        model_q90 = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "imputer.pkl"), "rb") as f:
        imputer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return model, model_q10, model_q90, imputer, meta


@st.cache_data(ttl=3600)
def load_market_data() -> pd.DataFrame:
    """Load all trainable listings once; used for all statistics sections."""
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


def _build_feature_row(listing: dict, meta: dict) -> pd.DataFrame:
    feature_cols = meta["feature_cols"]
    numeric_features = meta["numeric_features"]
    categorical_features = meta["categorical_features"]

    row = {col: np.nan for col in feature_cols}

    for col in numeric_features:
        val = listing.get(col)
        if val is not None:
            row[col] = float(val)

    if listing.get("dist_metro_m") is not None:
        row["log_dist_metro"] = np.log1p(float(listing["dist_metro_m"]))
    if listing.get("dist_center_m") is not None:
        row["log_dist_center"] = np.log1p(float(listing["dist_center_m"]))

    for cat_col in categorical_features:
        val = listing.get(cat_col)
        if val:
            oh_col = f"{cat_col}_{val}"
            if oh_col in row:
                row[oh_col] = 1.0
            for col in feature_cols:
                if col.startswith(f"{cat_col}_") and col != oh_col:
                    if np.isnan(row.get(col, np.nan)):
                        row[col] = 0.0

    return pd.DataFrame([row])[feature_cols]


def predict_with_interval(listing: dict, model, model_q10, model_q90, imputer, meta):
    df_row = _build_feature_row(listing, meta)
    df_imp = pd.DataFrame(imputer.transform(df_row), columns=meta["feature_cols"])
    pred = float(model.predict(df_imp)[0])
    lo = float(model_q10.predict(df_imp)[0])
    hi = float(model_q90.predict(df_imp)[0])
    # Ensure interval contains the point estimate
    lo = min(lo, pred)
    hi = max(hi, pred)
    return pred, lo, hi


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _neighborhood_histogram(df_neigh: pd.DataFrame, listed_psqm: float | None, predicted_psqm: float):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(df_neigh["price_per_sqm"], bins=25, color="#93c5fd", edgecolor="white", linewidth=0.4)
    ax.axvline(predicted_psqm, color="#2563eb", linewidth=2, label=f"Estimare: {predicted_psqm:,.0f} €/m²")
    if listed_psqm:
        ax.axvline(listed_psqm, color="#dc2626", linewidth=2, linestyle="--", label=f"Cerut: {listed_psqm:,.0f} €/m²")
    ax.set_xlabel("Preț €/m²", fontsize=10)
    ax.set_ylabel("Nr. apartamente", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def _zone_bar_chart(df_all: pd.DataFrame, current_neighborhood: str | None):
    med = (
        df_all.groupby("neighborhood")["price_per_sqm"]
        .median()
        .sort_values(ascending=True)
        .dropna()
    )
    colors = ["#2563eb" if idx == current_neighborhood else "#bfdbfe" for idx in med.index]
    fig, ax = plt.subplots(figsize=(7, max(4, len(med) * 0.28)))
    ax.barh(med.index, med.values, color=colors)
    ax.set_xlabel("Preț median €/m²", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Evaluare Imobiliară București",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Evaluare Imobiliară București")
st.markdown(
    "Lipește un link de pe [Storia.ro](https://www.storia.ro) și află dacă prețul "
    "cerut este corect față de piață."
)

model, model_q10, model_q90, imputer, meta = load_model()

# ---------------------------------------------------------------------------
# URL input
# ---------------------------------------------------------------------------
url = st.text_input(
    "Link anunț Storia.ro",
    placeholder="https://www.storia.ro/ro/oferta/...",
)

if url and not url.startswith("https://www.storia.ro/ro/oferta/"):
    st.error("Te rog inserează un link valid de pe storia.ro/ro/oferta/...")
    st.stop()

if url and st.button("Analizează", type="primary"):
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

    with st.spinner("Se procesează datele..."):
        enriched = enrich_listing(raw)

        lat, lon = enriched.get("lat"), enriched.get("lon")
        if lat and lon:
            enriched["neighborhood"] = point_in_neighborhood(lat, lon)
            enriched["zone"] = get_zone(enriched["neighborhood"])
            enriched["dist_metro_m"], enriched["nearest_metro"] = get_nearest_metro(lat, lon)
            enriched["dist_center_m"] = get_distance_to_center(lat, lon)

        predicted_psqm, pred_lo, pred_hi = predict_with_interval(
            enriched, model, model_q10, model_q90, imputer, meta
        )
        listed_psqm = enriched.get("price_per_sqm")
        listed_price = enriched.get("price_eur")
        area = enriched.get("area_sqm")

    # -----------------------------------------------------------------------
    # Results — verdict
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader(raw.get("title") or "Anunț")
    st.caption(f"📍 {enriched.get('address_raw') or 'Adresă necunoscută'}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Preț cerut", f"{listed_psqm:,.0f} €/m²" if listed_psqm else "N/A")

    with col2:
        st.metric(
            "Estimare model",
            f"{predicted_psqm:,.0f} €/m²",
            help=f"Interval estimat: {pred_lo:,.0f} – {pred_hi:,.0f} €/m²",
        )
        st.caption(f"Interval: {pred_lo:,.0f} – {pred_hi:,.0f} €/m²")

    with col3:
        if listed_psqm:
            diff_pct = (listed_psqm - predicted_psqm) / predicted_psqm * 100
            delta_label = f"{diff_pct:+.1f}% față de estimare"
            if diff_pct > 10:
                st.metric("Verdict", "Supraevaluat 🔴", delta=delta_label, delta_color="inverse")
            elif diff_pct < -10:
                st.metric("Verdict", "Sub piață 🟢", delta=delta_label, delta_color="normal")
            else:
                st.metric("Verdict", "Preț corect 🟡", delta=delta_label, delta_color="off")

    if area and listed_price:
        estimated_total = predicted_psqm * area
        diff_total = listed_price - estimated_total
        st.info(
            f"**Preț total cerut:** {listed_price:,.0f} € — "
            f"**Estimare totală:** {estimated_total:,.0f} € — "
            f"**Diferență:** {diff_total:+,.0f} €"
        )

    # -----------------------------------------------------------------------
    # Apartment details
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Detalii apartament")

    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    dcol1.metric("Suprafață", f"{area:.0f} m²" if area else "?")
    dcol2.metric("Camere", enriched.get("rooms") or "?")
    floor = enriched.get("floor")
    total_floors = enriched.get("total_floors")
    dcol3.metric(
        "Etaj",
        f"{floor}/{total_floors}" if floor is not None and total_floors else (str(floor) if floor is not None else "?"),
    )
    dcol4.metric("An construcție", enriched.get("year_built") or "?")

    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    gcol1.metric("Cartier", enriched.get("neighborhood") or "?")
    gcol2.metric("Zonă", enriched.get("zone") or "?")
    gcol3.metric("Metrou apropiat", enriched.get("nearest_metro") or "?")
    dist_m = enriched.get("dist_metro_m")
    gcol4.metric("Distanță metrou", f"{dist_m:.0f} m" if dist_m else "?")

    features = []
    if enriched.get("has_parking"):   features.append("✅ Parcare")
    if enriched.get("has_balcony"):   features.append("✅ Balcon")
    if enriched.get("has_elevator"):  features.append("✅ Lift")
    if enriched.get("has_ac"):        features.append("✅ Aer condiționat")
    if enriched.get("is_renovated"):  features.append("✅ Renovat")
    if enriched.get("is_furnished"):  features.append("✅ Mobilat")
    if enriched.get("is_new_build"):  features.append("🏗️ Construcție nouă")
    if enriched.get("is_cgi_listing"): features.append("⚠️ Poze orientative")
    if features:
        st.markdown("  ".join(features))

    risk = enriched.get("seismic_risk")
    risk_labels = {
        "very_high": "🔴 Risc seismic foarte ridicat",
        "high": "🟠 Risc seismic ridicat",
        "medium": "🟡 Risc seismic mediu",
        "low": "🟢 Risc seismic scăzut",
    }
    if risk and risk != "unknown":
        st.caption(risk_labels.get(risk, ""))

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------
    neighborhood = enriched.get("neighborhood")
    zone = enriched.get("zone")
    rooms = enriched.get("rooms")
    df_all = load_market_data()

    st.divider()

    # --- 1. Piața în cartier ---
    df_neigh = df_all[df_all["neighborhood"] == neighborhood] if neighborhood else pd.DataFrame()

    if neighborhood and len(df_neigh) >= 5:
        st.subheader(f"Piața în {neighborhood}")

        med_neigh = df_neigh["price_per_sqm"].median()
        n_listings = len(df_neigh)
        pmin = df_neigh["price_per_sqm"].min()
        pmax = df_neigh["price_per_sqm"].max()

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Median €/m²", f"{med_neigh:,.0f}")
        sc2.metric("Nr. anunțuri", n_listings)
        sc3.metric("Min €/m²", f"{pmin:,.0f}")
        sc4.metric("Max €/m²", f"{pmax:,.0f}")

        if listed_psqm:
            pct = (df_neigh["price_per_sqm"] < listed_psqm).mean() * 100
            st.caption(
                f"Prețul cerut este mai mare decât **{pct:.0f}%** din apartamentele din {neighborhood}."
            )

        st.pyplot(_neighborhood_histogram(df_neigh, listed_psqm, predicted_psqm))

    # --- 2. Apartamente similare ---
    st.subheader("Apartamente similare")

    comp = df_all.copy()
    if rooms:
        comp = comp[comp["rooms"] == rooms]
    if area:
        comp = comp[(comp["area_sqm"] >= area * 0.75) & (comp["area_sqm"] <= area * 1.25)]

    # Prefer same neighborhood, fallback to same zone
    comp_neigh = comp[comp["neighborhood"] == neighborhood] if neighborhood else pd.DataFrame()
    comp_zone = comp[comp["zone"] == zone] if zone else pd.DataFrame()

    if len(comp_neigh) >= 3:
        comp_show = comp_neigh
        comp_label = f"cartierul {neighborhood}"
    elif len(comp_zone) >= 3:
        comp_show = comp_zone
        comp_label = f"zona {zone}"
    else:
        comp_show = comp
        comp_label = "București"

    if len(comp_show) > 0:
        st.caption(
            f"{len(comp_show)} apartamente cu {rooms or '?'} camere și suprafață similară în {comp_label}."
        )
        display = (
            comp_show[["address_raw", "area_sqm", "price_per_sqm", "year_built", "floor", "dist_metro_m"]]
            .rename(columns={
                "address_raw": "Adresă",
                "area_sqm": "Suprafață (m²)",
                "price_per_sqm": "€/m²",
                "year_built": "An",
                "floor": "Etaj",
                "dist_metro_m": "Dist. metrou (m)",
            })
            .sort_values("€/m²")
            .head(10)
            .reset_index(drop=True)
        )
        display["€/m²"] = display["€/m²"].map("{:,.0f}".format)
        display["Dist. metrou (m)"] = display["Dist. metrou (m)"].map(
            lambda x: f"{x:.0f}" if pd.notna(x) else "?"
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.caption("Nu sunt suficiente date comparabile în baza de date.")

    # --- 3. Prețuri pe cartiere ---
    st.subheader("Prețuri mediane pe cartiere")
    st.pyplot(_zone_bar_chart(df_all, neighborhood))

    # -----------------------------------------------------------------------
    # Save to DB
    # -----------------------------------------------------------------------
    st.divider()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    exists = conn.execute(
        "SELECT id FROM Listings WHERE url = ?", (url.strip(),)
    ).fetchone()
    conn.close()

    if exists:
        st.success("✅ Acest anunț este deja în baza de date.")
    else:
        if st.button("💾 Adaugă în baza de date"):
            try:
                from database.db_manager import get_connection, insert_listing
                conn = get_connection(DB_PATH)
                insert_listing(conn, enriched)
                conn.close()
                st.success("✅ Anunț adăugat în baza de date!")
            except Exception as e:
                st.error(f"Eroare la salvare: {e}")

# ---------------------------------------------------------------------------
# Sidebar — model info + feature importance
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Despre model")
    m = meta["metrics"]
    st.metric("R²", f"{m['r2']:.3f}")
    st.metric("MAE", f"{m['mae']:.0f} €/m²")
    st.metric("MAPE", f"{m['mape']:.1f}%")

    interval_info = meta.get("interval", {})
    if interval_info:
        st.metric(
            "Acoperire interval",
            f"{interval_info['coverage']:.0%}",
            help="Procentul cazurilor din test set în care prețul real a căzut în intervalul estimat.",
        )

    df_all_sidebar = load_market_data()
    st.caption(f"Antrenat pe {len(df_all_sidebar):,} anunțuri din București")
    st.caption("Model: XGBoost cu 500 estimatori")

    img_path = os.path.join(MODEL_DIR, "feature_importance.png")
    if os.path.exists(img_path):
        st.divider()
        st.subheader("Feature importance")
        st.image(img_path)
