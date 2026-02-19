from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd

try:
    import uvicorn
except ImportError:
    uvicorn = None

try:
    from .config import get_config
    from .fusion_engine import run_pipeline
    from .ml_model import MLRiskConfig, MLRiskFusionModel
except ImportError:
    from config import get_config
    from fusion_engine import run_pipeline
    from ml_model import MLRiskConfig, MLRiskFusionModel


def _train_risk_model(args: argparse.Namespace) -> None:
    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    if data_path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(data_path)
    else:
        gdf = gpd.read_file(data_path)

    cfg = get_config()
    model = MLRiskFusionModel(
        MLRiskConfig(
            model_path=cfg.model.risk_model_path,
            preferred_model=args.model_type,
        )
    )
    ok = model.train(gdf, label_column=args.label, model_type=args.model_type)
    if not ok:
        raise RuntimeError(
            "Training failed. Ensure label column exists with at least 2 classes and sklearn dependencies are installed."
        )
    print(f"Risk model trained and saved to: {cfg.model.risk_model_path}")


def _run_pipeline(args: argparse.Namespace) -> None:
    _, report = run_pipeline(max_buildings=args.max_buildings)
    print(json.dumps(report, indent=2))


def _serve_api(args: argparse.Namespace) -> None:
    if uvicorn is None:
        raise RuntimeError("uvicorn is not installed. Install dependencies first.")
    uvicorn.run("backend.api:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UrbanGuard AI CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run-pipeline", help="Run end-to-end violation pipeline")
    run_cmd.add_argument("--max-buildings", type=int, default=50000)
    run_cmd.set_defaults(func=_run_pipeline)

    train_cmd = sub.add_parser("train-risk", help="Train risk model from labeled data")
    train_cmd.add_argument("--input", required=True, help="Path to labeled GeoJSON/Parquet")
    train_cmd.add_argument("--label", default="known_violation", help="Label column name")
    train_cmd.add_argument(
        "--model-type",
        default="xgboost",
        choices=["xgboost", "random_forest", "gradient_boosting"],
    )
    train_cmd.set_defaults(func=_train_risk_model)

    serve_cmd = sub.add_parser("serve", help="Run FastAPI server")
    serve_cmd.add_argument("--host", default="0.0.0.0")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.add_argument("--reload", action="store_true")
    serve_cmd.set_defaults(func=_serve_api)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

