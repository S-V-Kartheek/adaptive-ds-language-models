from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline
from .schemas import PipelineInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaptiveDS-LM pipeline runner")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where pipeline outputs will be generated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = PipelineInput.model_validate_json(config_path.read_text(encoding="utf-8"))
    output = run_pipeline(config, args.output_dir)
    print(json.dumps(output.model_dump(), indent=2))


if __name__ == "__main__":
    main()
