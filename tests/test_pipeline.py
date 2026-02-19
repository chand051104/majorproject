from __future__ import annotations

from backend.fusion_engine import run_pipeline


def test_pipeline_smoke() -> None:
    gdf, report = run_pipeline(max_buildings=60)
    assert len(gdf) > 0
    assert report["rows"] == len(gdf)
    required_columns = {
        "vector_risk_score",
        "satellite_encroachment_score",
        "temporal_growth_score",
        "temporal_image_mode",
        "final_violation_probability",
        "risk_category",
        "legal_narrative",
    }
    assert required_columns.issubset(gdf.columns)
    assert gdf["final_violation_probability"].between(0, 100).all()
