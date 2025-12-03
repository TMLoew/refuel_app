"""Offline training entrypoint for the attendance + snack demand models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from .demand_model import MODEL_DIR, add_time_signals, save_models, train_models

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from frontend.streamlit_app.services.data_utils import DATA_FILE, build_enriched_history  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and persist Refuel demand models")
    parser.add_argument(
        "--csv",
        default=DATA_FILE,
        type=Path,
        help=f"Path to the telemetry CSV (default: {DATA_FILE})",
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        type=Path,
        help="Directory where trained models should be stored (default: model/)",
    )
    parser.add_argument(
        "--use-live-weather",
        action="store_true",
        help="Use the Open-Meteo API to enrich the dataset before training.",
    )
    return parser.parse_args()


def train_offline(csv_path: Path, model_dir: Path, use_live_weather: bool = False) -> Tuple[Path, Path]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Telemetry CSV not found: {csv_path}")
    history = build_enriched_history(csv_path=csv_path, use_weather_api=use_live_weather)
    if history.empty:
        raise RuntimeError("Telemetry CSV is empty; cannot train models.")
    feature_df = add_time_signals(history)
    models = train_models(feature_df)
    return save_models(models, model_dir=model_dir)


def main() -> None:
    args = _parse_args()
    checkin_path, snack_path = train_offline(args.csv, args.model_dir, args.use_live_weather)
    print("Saved models:")
    print(f"  - check-ins: {checkin_path}")
    print(f"  - snacks:    {snack_path}")


if __name__ == "__main__":
    main()
