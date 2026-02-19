from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    GradientBoostingClassifier = None
    RandomForestClassifier = None
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


@dataclass(slots=True)
class MLRiskConfig:
    model_path: Path = Path(__file__).resolve().parents[1] / "models" / "risk_model.pkl"
    random_state: int = 42
    default_threshold: float = 60.0
    preferred_model: str = "xgboost"
    shap_enabled: bool = True
    shap_max_rows: int = 2000


class MLRiskFusionModel:
    def __init__(self, config: MLRiskConfig | None = None) -> None:
        self.config = config or MLRiskConfig()
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_columns = [
            "vector_risk_score",
            "satellite_encroachment_score",
            "temporal_growth_score",
            "parcel_spillover_pct",
            "built_up_ratio",
            "dist_road_m",
            "dist_canal_m",
            "dist_lake_m",
        ]
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not self.config.model_path.exists():
            return
        with self.config.model_path.open("rb") as f:
            payload = pickle.load(f)
        self.model = payload.get("model")
        saved_features = payload.get("features")
        if saved_features:
            self.feature_columns = saved_features

    def _feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=df.index)
        for feature in self.feature_columns:
            series = df.get(feature, pd.Series(0.0, index=df.index)).copy()
            if series.dtype.kind not in "biufc":
                series = pd.to_numeric(series, errors="coerce")
            x[feature] = series.fillna(0.0)
        # Keep distance features bounded so fallback logic remains stable.
        for col in ["dist_road_m", "dist_canal_m", "dist_lake_m"]:
            if col in x:
                x[col] = x[col].clip(lower=0.0, upper=500.0)
        x["parcel_spillover_pct"] = x["parcel_spillover_pct"].clip(0.0, 1.0)
        x["built_up_ratio"] = x["built_up_ratio"].clip(0.0, 2.0)
        return x

    def train(
        self,
        df: gpd.GeoDataFrame,
        label_column: str = "known_violation",
        model_type: str | None = None,
    ) -> bool:
        if not SKLEARN_AVAILABLE:
            return False
        if label_column not in df.columns:
            return False

        labels = pd.to_numeric(df[label_column], errors="coerce").fillna(0).astype(int)
        x = self._feature_frame(df)
        if labels.nunique() < 2:
            return False

        model_choice = (model_type or self.config.preferred_model).lower()
        if model_choice == "xgboost" and XGBOOST_AVAILABLE:
            model = XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.config.random_state,
                eval_metric="logloss",
                tree_method="hist",
            )
        elif model_choice == "random_forest":
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            model = GradientBoostingClassifier(random_state=self.config.random_state)
        model.fit(x, labels)
        self.model = model

        payload = {"model": self.model, "features": self.feature_columns}
        with self.config.model_path.open("wb") as f:
            pickle.dump(payload, f)
        return True

    def predict(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        out = df.copy()
        x = self._feature_frame(out)

        if self.model is not None:
            probs = self.model.predict_proba(x)[:, 1]
            out["model_used"] = self.model.__class__.__name__.lower()
        else:
            # Deterministic fallback when training labels/model are unavailable.
            vector_score = (x["vector_risk_score"] / 100.0).clip(0.0, 1.0)
            satellite_score = (x["satellite_encroachment_score"] / 100.0).clip(0.0, 1.0)
            temporal_score = (x["temporal_growth_score"] / 100.0).clip(0.0, 1.0)
            dist_road_component = (1.0 - (x["dist_road_m"] / 80.0).clip(0.0, 1.0)).clip(0.0, 1.0)
            dist_canal_component = (1.0 - (x["dist_canal_m"] / 100.0).clip(0.0, 1.0)).clip(0.0, 1.0)
            dist_lake_component = (1.0 - (x["dist_lake_m"] / 120.0).clip(0.0, 1.0)).clip(0.0, 1.0)

            footpath_shop = pd.to_numeric(
                out.get("footpath_shop_prob", pd.Series(np.nan, index=out.index)), errors="coerce"
            ).fillna(0.0)
            street_vending = pd.to_numeric(
                out.get("street_vending_prob", pd.Series(np.nan, index=out.index)), errors="coerce"
            ).fillna(0.0)
            informal_structure = pd.to_numeric(
                out.get("informal_structure_prob", pd.Series(np.nan, index=out.index)), errors="coerce"
            ).fillna(0.0)
            illegal_settlement = pd.to_numeric(
                out.get("illegal_settlement_prob", pd.Series(np.nan, index=out.index)), errors="coerce"
            ).fillna(0.0)
            illegal_activity = pd.concat(
                [footpath_shop, street_vending, informal_structure, illegal_settlement],
                axis=1,
            ).max(axis=1)

            road_intrusion = ((x["dist_road_m"] <= 2.5) & (x["parcel_spillover_pct"] >= 0.15)).astype(float)
            severe_vector = (vector_score >= 0.70).astype(float)
            severe_satellite = (satellite_score >= 0.65).astype(float)
            severe_temporal = (temporal_score >= 0.55).astype(float)

            base = (
                0.31 * vector_score
                + 0.31 * satellite_score
                + 0.20 * temporal_score
                + 0.07 * x["parcel_spillover_pct"]
                + 0.04 * x["built_up_ratio"].clip(0.0, 1.0)
                + 0.03 * dist_road_component
                + 0.02 * dist_canal_component
                + 0.01 * dist_lake_component
            )
            interaction_boost = (
                0.12 * (severe_vector * severe_satellite)
                + 0.10 * (severe_satellite * severe_temporal)
                + 0.08 * road_intrusion
                + 0.06 * (illegal_activity >= 0.75).astype(float)
                + 0.05 * (x["parcel_spillover_pct"] >= 0.35).astype(float)
                + 0.05 * (x["built_up_ratio"] >= 0.90).astype(float)
            )
            probs = (base + interaction_boost).clip(0.0, 1.0)
            out["model_used"] = "weighted_fallback"

        out["final_violation_probability"] = (probs * 100.0).clip(0.0, 100.0)
        out["risk_category"] = pd.cut(
            out["final_violation_probability"],
            bins=[-np.inf, 30, 60, np.inf],
            labels=["Low", "Medium", "High"],
        ).astype(str)
        out["risk_flag"] = (out["final_violation_probability"] >= self.config.default_threshold).astype(int)

        out["shap_top_feature"] = "not_available"
        out["shap_values"] = "{}"
        if (
            self.model is not None
            and SHAP_AVAILABLE
            and self.config.shap_enabled
            and len(x) <= self.config.shap_max_rows
        ):
            try:
                explainer = shap.Explainer(self.model, x, feature_names=self.feature_columns)
                shap_values = explainer(x, check_additivity=False)
                values = np.asarray(shap_values.values)
                if values.ndim == 3:
                    values = values[:, :, 1]

                top_features: list[str] = []
                shap_json: list[str] = []
                for i in range(values.shape[0]):
                    contrib = {self.feature_columns[j]: float(values[i, j]) for j in range(values.shape[1])}
                    top = max(contrib.items(), key=lambda kv: abs(kv[1]))[0]
                    top_features.append(top)
                    shap_json.append(json.dumps(contrib, ensure_ascii=True))
                out["shap_top_feature"] = top_features
                out["shap_values"] = shap_json
            except Exception:
                pass

        return out


def run_ml_fusion(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    model = MLRiskFusionModel()
    return model.predict(df)
