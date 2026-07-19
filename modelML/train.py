"""
XGBoost model for predicting price per sqm of Bucharest apartments.

Key design choices from data analysis:
- Target: raw price_per_sqm (log transform hurt R², not used)
- Distances log-transformed (non-linear relationship confirmed)
- lat/lon included as features (continuous spatial signal > polygon categories)
- IQR filtering removes outliers before training
- compartmentare dropped (100% missing)
"""

import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.db_manager import get_connection, get_listings_for_model

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "real_estate.db")
OUT_DIR = os.path.dirname(__file__)

NUMERIC_FEATURES = [
    "area_sqm", "rooms", "floor", "total_floors", "year_built", "is_post_1977", "is_new_build", "is_penthouse", "is_cgi_listing",
    "log_dist_metro", "log_dist_center",
    "lat", "lon",
    "has_parking", "has_balcony", "has_elevator", "has_ac",
    "has_central_heating", "has_storage", "is_renovated", "is_furnished",
]

CATEGORICAL_FEATURES = [
    "neighborhood", "zone", "seismic_risk", "nearest_metro",
]

TARGET = "price_per_sqm"

# Quantile levels for the prediction interval (p5–p95 = 90% nominal coverage)
Q_LO, Q_HI = 0.05, 0.95


def load_data(db_path: str) -> pd.DataFrame:
    conn = get_connection(db_path)
    df = get_listings_for_model(conn)
    conn.close()

    # Drop relisted duplicates (same apartment under a different URL) so the
    # same unit can't land in both train and test splits.
    before = len(df)
    df = df.drop_duplicates(
        subset=["area_sqm", "rooms", "lat", "lon", "price_eur"], keep="first"
    )
    if len(df) < before:
        print(f"Dropped {before - len(df)} relisted duplicates")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Apply IQR filter, engineer features, encode categoricals (no imputation)."""
    Q1, Q3 = df[TARGET].quantile(0.25), df[TARGET].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[TARGET] >= Q1 - 1.5 * IQR) & (df[TARGET] <= Q3 + 1.5 * IQR)
    df = df[mask].copy()
    print(f"After IQR filter: {len(df):,} rows  ({(~mask).sum()} outliers removed)")

    # Log-transform distances
    df["log_dist_metro"] = np.log1p(df["dist_metro_m"])
    df["log_dist_center"] = np.log1p(df["dist_center_m"])

    y = df[TARGET].reset_index(drop=True)

    # One-hot encode categoricals
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=False)
    X_num = df[NUMERIC_FEATURES].copy()
    X = pd.concat([X_num, X_cat], axis=1).reset_index(drop=True)

    return X, y, list(X.columns)


def train(db_path: str = DB_PATH, out_dir: str = OUT_DIR) -> None:
    print("Loading data...")
    df = load_data(db_path)
    print(f"Loaded {len(df):,} listings")

    print("\nPreparing features...")
    X, y, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Impute with medians learned from the TRAINING split only (no test leakage)
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"\n=== Test set metrics ===")
    print(f"  MAE   : {mae:.1f} €/m²")
    print(f"  RMSE  : {rmse:.1f} €/m²")
    print(f"  R²    : {r2:.4f}")
    print(f"  MAPE  : {mape:.1f}%")

    # Feature importance plot (top 20)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top20 = importances.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    top20.plot(kind="barh", ax=ax, color="#2563eb")
    ax.set_title("XGBoost — Top 20 Feature Importances", fontsize=13)
    ax.set_xlabel("Importance (gain)")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"\nFeature importance plot saved.")

    # Quantile models for the 90% prediction interval (p5 / p95), conformalized:
    # raw quantile regressors undercover (77% observed vs 90% nominal), so we
    # hold out a calibration split and widen the interval by the conformity-score
    # quantile (CQR). The point model above still uses the full training data.
    print(f"\nTraining quantile models (p{int(Q_LO*100)} / p{int(Q_HI*100)}, conformalized)...")
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train.reset_index(drop=True), test_size=0.2, random_state=42
    )
    q_params = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0, n_jobs=-1,
    )
    model_q_lo = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=Q_LO, **q_params)
    model_q_hi = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=Q_HI, **q_params)
    model_q_lo.fit(X_fit, y_fit)
    model_q_hi.fit(X_fit, y_fit)

    # Conformity scores: how far outside the raw interval each calibration point falls
    cal_lo = model_q_lo.predict(X_cal)
    cal_hi = model_q_hi.predict(X_cal)
    scores = np.maximum(cal_lo - y_cal.values, y_cal.values - cal_hi)
    alpha = 1.0 - (Q_HI - Q_LO)  # 0.10 for a 90% interval
    n_cal = len(scores)
    q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    qhat = float(np.quantile(scores, q_level, method="higher"))
    print(f"  Conformal offset (qhat)       : {qhat:.0f} €/m²  ({n_cal} calibration rows)")

    lo_pred = model_q_lo.predict(X_test) - qhat
    hi_pred = model_q_hi.predict(X_test) + qhat
    coverage = np.mean((y_test.values >= lo_pred) & (y_test.values <= hi_pred))
    avg_width = np.mean(hi_pred - lo_pred)
    print(f"  90% interval coverage on test : {coverage:.1%}  (target ≥90%)")
    print(f"  Average interval width        : {avg_width:.0f} €/m²")

    # Save model and metadata
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(out_dir, "model_q_lo.pkl"), "wb") as f:
        pickle.dump(model_q_lo, f)
    with open(os.path.join(out_dir, "model_q_hi.pkl"), "wb") as f:
        pickle.dump(model_q_hi, f)

    with open(os.path.join(out_dir, "imputer.pkl"), "wb") as f:
        pickle.dump(imputer, f)

    metadata = {
        "feature_cols": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape},
        "interval": {
            "q_lo": Q_LO,
            "q_hi": Q_HI,
            "conformal_offset": qhat,
            "coverage": float(coverage),
            "avg_width": float(avg_width),
        },
        "iqr_bounds": {
            "Q1": float(df[TARGET].quantile(0.25)),
            "Q3": float(df[TARGET].quantile(0.75)),
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {out_dir}/model.pkl")
    print(f"Metadata saved to {out_dir}/metadata.json")

    # Top features summary
    print(f"\n=== Top 10 features ===")
    for feat, imp in importances.nlargest(10).items():
        print(f"  {feat:35} {imp:.4f}")
    print(f"\nInterval models saved to {out_dir}/model_q_lo.pkl, model_q_hi.pkl")


if __name__ == "__main__":
    train()
