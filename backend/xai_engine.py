from __future__ import annotations

import json
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass(slots=True)
class XAIConfig:
    include_rule_narrative: bool = True


class ExplainableAIEngine:
    def __init__(self, config: XAIConfig | None = None) -> None:
        self.config = config or XAIConfig()

    @staticmethod
    def _clip_pct(value: float) -> float:
        return max(0.0, min(100.0, value))

    def _factor_breakdown(self, row: pd.Series) -> dict[str, float]:
        vector = float(row.get("vector_risk_score", 0.0))
        satellite = float(row.get("satellite_encroachment_score", 0.0))
        temporal = float(row.get("temporal_growth_score", 0.0))
        total = max(vector + satellite + temporal, 1.0)
        out = {
            "vector_rules_pct": self._clip_pct((vector / total) * 100.0),
            "satellite_evidence_pct": self._clip_pct((satellite / total) * 100.0),
            "temporal_growth_pct": self._clip_pct((temporal / total) * 100.0),
        }
        if str(row.get("shap_top_feature", "not_available")) != "not_available":
            out["model_shap_top_feature"] = str(row.get("shap_top_feature"))
        return out

    def _rule_narrative(self, row: pd.Series) -> str:
        parts: list[str] = []
        if float(row.get("parcel_spillover_pct", 0.0)) > 0.0:
            parts.append(f"Parcel spillover: {float(row.get('parcel_spillover_pct', 0.0)) * 100:.1f}%")
        if float(row.get("dist_road_m", 999.0)) <= 6.0:
            parts.append(f"Road buffer proximity: {float(row.get('dist_road_m', 0.0)):.2f} m")
        if float(row.get("dist_canal_m", 999.0)) <= 25.0:
            parts.append(f"Canal buffer proximity: {float(row.get('dist_canal_m', 0.0)):.2f} m")
        if float(row.get("dist_lake_m", 999.0)) <= 60.0:
            parts.append(f"Lake buffer proximity: {float(row.get('dist_lake_m', 0.0)):.2f} m")
        if float(row.get("built_up_ratio", 0.0)) > 0.60:
            parts.append(f"Built-up ratio: {float(row.get('built_up_ratio', 0.0)):.2f}")
        if not parts:
            return "No major rule threshold triggered."
        return "; ".join(parts)

    def _legal_narrative(self, row: pd.Series) -> str:
        score = float(row.get("final_violation_probability", 0.0))
        category = row.get("risk_category", "Low")
        trigger_text = row.get("vector_triggers", "none")
        sat = row.get("satellite_evidence", "none")
        temporal = row.get("temporal_evidence", "stable")
        year = int(row.get("history_reference_year", 2020))
        shap_top = row.get("shap_top_feature", "not_available")
        model_line = (
            f" Model contribution leader (SHAP): {shap_top}."
            if str(shap_top) != "not_available"
            else ""
        )
        return (
            f"Parcel classified as {category} risk with violation probability {score:.1f}%. "
            f"Rule triggers: {trigger_text}. Satellite indicators: {sat}. "
            f"Temporal assessment since {year}: {temporal}.{model_line}"
        )

    def run(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        out = df.copy()
        vector = pd.to_numeric(out.get("vector_risk_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        satellite = (
            pd.to_numeric(out.get("satellite_encroachment_score", 0.0), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        temporal = pd.to_numeric(out.get("temporal_growth_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        total = np.maximum(vector + satellite + temporal, 1.0)

        vector_pct = np.clip((vector / total) * 100.0, 0.0, 100.0)
        satellite_pct = np.clip((satellite / total) * 100.0, 0.0, 100.0)
        temporal_pct = np.clip((temporal / total) * 100.0, 0.0, 100.0)
        shap_top = out.get("shap_top_feature", pd.Series("not_available", index=out.index)).astype(str).to_numpy()

        feature_contributions: list[str] = []
        top_factors: list[str] = []
        for v_pct, s_pct, t_pct, shap_name in zip(vector_pct, satellite_pct, temporal_pct, shap_top):
            payload: dict[str, float | str] = {
                "vector_rules_pct": float(v_pct),
                "satellite_evidence_pct": float(s_pct),
                "temporal_growth_pct": float(t_pct),
            }
            if shap_name != "not_available":
                payload["model_shap_top_feature"] = shap_name
            feature_contributions.append(json.dumps(payload, ensure_ascii=True))
            if v_pct >= s_pct and v_pct >= t_pct:
                top_factors.append("vector_rules_pct")
            elif s_pct >= v_pct and s_pct >= t_pct:
                top_factors.append("satellite_evidence_pct")
            else:
                top_factors.append("temporal_growth_pct")

        out["feature_contributions"] = feature_contributions
        out["xai_top_factor"] = top_factors

        spillover = pd.to_numeric(out.get("parcel_spillover_pct", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        dist_road = pd.to_numeric(out.get("dist_road_m", 999.0), errors="coerce").fillna(999.0).to_numpy(dtype=float)
        dist_canal = pd.to_numeric(out.get("dist_canal_m", 999.0), errors="coerce").fillna(999.0).to_numpy(dtype=float)
        dist_lake = pd.to_numeric(out.get("dist_lake_m", 999.0), errors="coerce").fillna(999.0).to_numpy(dtype=float)
        built_up = pd.to_numeric(out.get("built_up_ratio", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

        rule_narratives: list[str] = []
        for sp, dr, dc, dl, bu in zip(spillover, dist_road, dist_canal, dist_lake, built_up):
            parts: list[str] = []
            if float(sp) > 0.0:
                parts.append(f"Parcel spillover: {float(sp) * 100:.1f}%")
            if float(dr) <= 6.0:
                parts.append(f"Road buffer proximity: {float(dr):.2f} m")
            if float(dc) <= 25.0:
                parts.append(f"Canal buffer proximity: {float(dc):.2f} m")
            if float(dl) <= 60.0:
                parts.append(f"Lake buffer proximity: {float(dl):.2f} m")
            if float(bu) > 0.60:
                parts.append(f"Built-up ratio: {float(bu):.2f}")
            rule_narratives.append("; ".join(parts) if parts else "No major rule threshold triggered.")

        out["rule_narrative"] = rule_narratives

        probabilities = (
            pd.to_numeric(out.get("final_violation_probability", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        categories = out.get("risk_category", pd.Series("Low", index=out.index)).astype(str).to_numpy()
        triggers = out.get("vector_triggers", pd.Series("none", index=out.index)).astype(str).to_numpy()
        sat_evidence = out.get("satellite_evidence", pd.Series("none", index=out.index)).astype(str).to_numpy()
        temp_evidence = out.get("temporal_evidence", pd.Series("stable", index=out.index)).astype(str).to_numpy()
        ref_year = (
            pd.to_numeric(out.get("history_reference_year", 2020), errors="coerce")
            .fillna(2020)
            .astype(int)
            .to_numpy()
        )

        legal_narratives: list[str] = []
        for score, category, trigger_text, sat_text, temp_text, year, shap_name in zip(
            probabilities, categories, triggers, sat_evidence, temp_evidence, ref_year, shap_top
        ):
            model_line = (
                f" Model contribution leader (SHAP): {shap_name}." if str(shap_name) != "not_available" else ""
            )
            legal_narratives.append(
                f"Parcel classified as {category} risk with violation probability {float(score):.1f}%. "
                f"Rule triggers: {trigger_text}. Satellite indicators: {sat_text}. "
                f"Temporal assessment since {int(year)}: {temp_text}.{model_line}"
            )

        out["legal_narrative"] = legal_narratives
        return out


def run_xai(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return ExplainableAIEngine().run(df)
