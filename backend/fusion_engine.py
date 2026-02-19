from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from .config import get_config
    from .ml_model import MLRiskConfig, MLRiskFusionModel
    from .satellite_engine import SatelliteEngineConfig, SatelliteVisionEngine
    from .temporal_engine import TemporalChangeEngine, TemporalEngineConfig
    from .vector_engine import VectorComplianceEngine, VectorEngineConfig
    from .xai_engine import ExplainableAIEngine
except ImportError:
    from config import get_config
    from ml_model import MLRiskConfig, MLRiskFusionModel
    from satellite_engine import SatelliteEngineConfig, SatelliteVisionEngine
    from temporal_engine import TemporalChangeEngine, TemporalEngineConfig
    from vector_engine import VectorComplianceEngine, VectorEngineConfig
    from xai_engine import ExplainableAIEngine


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class FusionPipelineConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    output_dir: Path = Path(__file__).resolve().parent / "data"
    max_buildings: int | None = 50000
    pipeline_chunk_size: int = _env_int("URBANGUARD_PIPELINE_CHUNK_SIZE", 100000)
    parquet_row_group_size: int = _env_int("URBANGUARD_PARQUET_ROW_GROUP_SIZE", 50000)
    write_geojson: bool = _env_bool("URBANGUARD_WRITE_GEOJSON", True)
    geojson_max_rows: int = _env_int("URBANGUARD_GEOJSON_MAX_ROWS", 200000)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_chunk_size = max(0, int(self.pipeline_chunk_size))
        self.parquet_row_group_size = max(1000, int(self.parquet_row_group_size))
        self.geojson_max_rows = max(0, int(self.geojson_max_rows))


class UrbanGuardPipeline:
    def __init__(self, config: FusionPipelineConfig | None = None) -> None:
        self.config = config or FusionPipelineConfig()
        self.app_config = get_config()

    @staticmethod
    def _as_tags(value: Any) -> list[str]:
        if value is None:
            return []
        raw = str(value).strip()
        if not raw or raw.lower() in {"none", "stable", "nan"}:
            return []
        excluded = {
            "imagery_only_temporal",
            "no_imagery_source",
            "tile_fetch_failed",
            "sentinel_fetch_failed",
            "tile_compare_disabled",
            "not_sampled",
        }
        return [token.strip() for token in raw.split(",") if token.strip() and token.strip() not in excluded]

    def _post_process(self, explained: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        explained["parcel_id"] = explained.get("parcel_id", explained.get("building_id", explained.index))
        vector_triggers = explained.get("vector_triggers", pd.Series("", index=explained.index)).astype(str).to_numpy()
        satellite_evidence = explained.get("satellite_evidence", pd.Series("", index=explained.index)).astype(str).to_numpy()
        temporal_evidence = explained.get("temporal_evidence", pd.Series("", index=explained.index)).astype(str).to_numpy()
        footpath_flags = (
            pd.to_numeric(explained.get("footpath_shop_flag", pd.Series(0, index=explained.index)), errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        vending_flags = (
            pd.to_numeric(explained.get("street_vending_flag", pd.Series(0, index=explained.index)), errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        settlement_flags = (
            pd.to_numeric(
                explained.get("illegal_settlement_flag", pd.Series(0, index=explained.index)), errors="coerce"
            )
            .fillna(0)
            .astype(int)
            .to_numpy()
        )

        enc_types: list[str] = []
        for vt, se, te, fp_flag, sv_flag, is_flag in zip(
            vector_triggers, satellite_evidence, temporal_evidence, footpath_flags, vending_flags, settlement_flags
        ):
            tags: list[str] = []
            tags.extend(self._as_tags(vt))
            tags.extend(self._as_tags(se))
            tags.extend(self._as_tags(te))

            if int(fp_flag) == 1:
                tags.append("footpath_shop")
            if int(sv_flag) == 1:
                tags.append("street_vending")
            if int(is_flag) == 1:
                tags.append("illegal_settlement")

            ordered = list(dict.fromkeys(tags))
            enc_types.append(",".join(ordered) if ordered else "none")

        explained["encroachment_type"] = enc_types
        explained["violation_score"] = explained["final_violation_probability"]
        explained["before_after_image"] = explained.get(
            "temporal_evidence", pd.Series("not_available", index=explained.index)
        )
        explained["satellite_proof_image"] = explained.get(
            "satellite_evidence", pd.Series("not_available", index=explained.index)
        )
        return explained

    def _run_stage_chain(
        self,
        frame: gpd.GeoDataFrame,
        satellite_engine: SatelliteVisionEngine,
        temporal_engine: TemporalChangeEngine,
        ml_engine: MLRiskFusionModel,
        xai_engine: ExplainableAIEngine,
    ) -> gpd.GeoDataFrame:
        sat_df = satellite_engine.run(frame)
        temporal_df = temporal_engine.run(sat_df)
        ml_df = ml_engine.predict(temporal_df)
        explained = xai_engine.run(ml_df)
        return self._post_process(explained)

    def run(self) -> gpd.GeoDataFrame:
        vector_engine = VectorComplianceEngine(
            VectorEngineConfig(
                project_root=self.config.project_root,
                output_dir=self.config.output_dir,
                max_buildings=self.config.max_buildings,
            )
        )
        vector_df = vector_engine.run()

        sat_config = SatelliteEngineConfig.from_app_config(self.app_config)
        temporal_config = TemporalEngineConfig(project_root=self.config.project_root)
        ml_engine = MLRiskFusionModel(MLRiskConfig(model_path=self.app_config.model.risk_model_path))
        xai_engine = ExplainableAIEngine()

        chunk_size = self.config.pipeline_chunk_size
        chunked = chunk_size > 0 and len(vector_df) > chunk_size
        if chunked:
            chunk_count = int(math.ceil(len(vector_df) / float(chunk_size)))
            if sat_config.max_remote_tiles > 0:
                sat_config.max_remote_tiles = max(1, sat_config.max_remote_tiles // chunk_count)
            if temporal_config.max_tile_comparisons > 0:
                temporal_config.max_tile_comparisons = max(1, temporal_config.max_tile_comparisons // chunk_count)

        satellite_engine = SatelliteVisionEngine(sat_config)
        temporal_engine = TemporalChangeEngine(temporal_config)

        if not chunked:
            return self._run_stage_chain(
                frame=vector_df,
                satellite_engine=satellite_engine,
                temporal_engine=temporal_engine,
                ml_engine=ml_engine,
                xai_engine=xai_engine,
            )

        frames: list[gpd.GeoDataFrame] = []
        for start in range(0, len(vector_df), chunk_size):
            stop = min(len(vector_df), start + chunk_size)
            chunk = vector_df.iloc[start:stop].copy()
            chunk_out = self._run_stage_chain(
                frame=chunk,
                satellite_engine=satellite_engine,
                temporal_engine=temporal_engine,
                ml_engine=ml_engine,
                xai_engine=xai_engine,
            )
            frames.append(chunk_out)

        merged = pd.concat(frames, ignore_index=True)
        return gpd.GeoDataFrame(merged, geometry="geometry", crs=vector_df.crs)

    def save(self, results: gpd.GeoDataFrame) -> dict[str, Path]:
        parquet_path = self.config.output_dir / "urban_guard_results.parquet"
        geojson_path = self.config.output_dir / "urban_guard_results.geojson"
        summary_path = self.config.output_dir / "urban_guard_summary.json"

        results.to_parquet(parquet_path, row_group_size=self.config.parquet_row_group_size)
        write_geojson = self.config.write_geojson and len(results) <= self.config.geojson_max_rows
        if write_geojson:
            results.to_file(geojson_path, driver="GeoJSON")
        elif geojson_path.exists():
            geojson_path.unlink()

        summary = {
            "total_features": int(len(results)),
            "risk_distribution": results["risk_category"].value_counts(dropna=False).to_dict(),
            "mean_violation_probability": float(results["final_violation_probability"].mean()),
            "geojson_written": bool(write_geojson),
            "satellite_tiles_used": int(results.get("satellite_tile_used", False).sum()),
            "satellite_tile_sources": results.get("satellite_tile_source", "").value_counts().to_dict()
            if "satellite_tile_source" in results.columns
            else {},
            "config": {k: str(v) for k, v in asdict(self.config).items()},
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "parquet": parquet_path,
            "geojson": geojson_path,
            "summary": summary_path,
        }


def run_pipeline(max_buildings: int | None = 50000) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    pipeline = UrbanGuardPipeline(FusionPipelineConfig(max_buildings=max_buildings))
    results = pipeline.run()
    outputs = pipeline.save(results)
    report: dict[str, Any] = {
        "rows": int(len(results)),
        "high_risk_count": int((results["risk_category"] == "High").sum()),
        "avg_probability": float(np.round(results["final_violation_probability"].mean(), 2)),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    return results, report


if __name__ == "__main__":
    _, report = run_pipeline()
    print(json.dumps(report, indent=2))
