"""
XGBoost model for predicting price per sqm of Bucharest apartments.

Key design choices from data analysis:
- Target: raw price_per_sqm (log transform hurt R², not used)
- Distances log-transformed (non-linear relationship confirmed)
- lat/lon included as features (continuous spatial signal > polygon categories)
- IQR filtering removes 7% outliers before training
- compartmentare dropped (100% missing)
"""

import json
import os
import pickle
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

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


def load_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT l.*, n.name AS neighborhood, n.zone AS zone
        FROM Listings l
        LEFT JOIN Neighborhoods n ON l.neighborhood_id = n.id
        WHERE l.price_per_sqm IS NOT NULL
          AND l.lat IS NOT NULL
          AND l.lat != -1
        """,
        conn,
    )
    conn.close()
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, SimpleImputer, list[str]]:
    """
    Apply IQR filter, engineer features, encode categoricals.
    Returns X, y, fitted imputer, and feature column names.
    """
    # IQR filter on target
    Q1, Q3 = df[TARGET].quantile(0.25), df[TARGET].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[TARGET] >= Q1 - 1.5 * IQR) & (df[TARGET] <= Q3 + 1.5 * IQR)
    df = df[mask].copy()
    print(f"After IQR filter: {len(df):,} rows  ({(~mask).sum()} outliers removed)")

    # Log-transform distances
    df["log_dist_metro"] = np.log1p(df["dist_metro_m"])
    df["log_dist_center"] = np.log1p(df["dist_center_m"])

    y = df[TARGET]

    # One-hot encode categoricals
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=False)
    X_num = df[NUMERIC_FEATURES].copy()
    X = pd.concat([X_num, X_cat], axis=1)

    feature_cols = list(X.columns)

    # Median imputation for missing numerics
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    return X_imp, y.reset_index(drop=True), imputer, feature_cols


def train(db_path: str = DB_PATH, out_dir: str = OUT_DIR) -> None:
    print("Loading data...")
    df = load_data(db_path)
    print(f"Loaded {len(df):,} listings")

    print("\nPreparing features...")
    X, y, imputer, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

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

    # Quantile models for prediction intervals (10th and 90th percentile)
    print("\nTraining quantile models (p10 / p90)...")
    q_params = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0, n_jobs=-1,
    )
    model_q10 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.05, **q_params)
    model_q90 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.95, **q_params)
    model_q10.fit(X_train, y_train)
    model_q90.fit(X_train, y_train)

    q10_pred = model_q10.predict(X_test)
    q90_pred = model_q90.predict(X_test)
    coverage = np.mean((y_test.values >= q10_pred) & (y_test.values <= q90_pred))
    avg_width = np.mean(q90_pred - q10_pred)
    print(f"  90% interval coverage on test : {coverage:.1%}  (target ≥90%)")
    print(f"  Average interval width        : {avg_width:.0f} €/m²")

    # Save model and metadata
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(out_dir, "model_q10.pkl"), "wb") as f:
        pickle.dump(model_q10, f)
    with open(os.path.join(out_dir, "model_q90.pkl"), "wb") as f:
        pickle.dump(model_q90, f)

    with open(os.path.join(out_dir, "imputer.pkl"), "wb") as f:
        pickle.dump(imputer, f)

    metadata = {
        "feature_cols": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape},
        "interval": {"coverage": float(coverage), "avg_width": float(avg_width)},
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
    print(f"\nInterval models saved to {out_dir}/model_q10.pkl, model_q90.pkl")


if __name__ == "__main__":
    train()
