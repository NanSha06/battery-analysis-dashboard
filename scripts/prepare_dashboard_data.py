from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pipeline import build_and_export_dashboard_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Streamlit dashboard artifacts.")
    parser.add_argument(
        "--mat-dir",
        default="mat_files",
        help="Directory containing battery MAT files.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where dashboard artifacts will be written.",
    )
    parser.add_argument(
        "--nominal-capacity-ah",
        type=float,
        default=2.0,
        help="Nominal battery capacity used for SOC estimation.",
    )
    parser.add_argument(
        "--soh-threshold",
        type=float,
        default=0.8,
        help="SOH threshold used for RUL projection.",
    )
    parser.add_argument(
        "--ecm-sample-limit",
        type=int,
        default=50000,
        help="Sample limit used during ECM/EKF artifact generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_and_export_dashboard_artifacts(
        mat_dir=Path(args.mat_dir),
        output_dir=Path(args.output_dir),
        nominal_capacity_ah=args.nominal_capacity_ah,
        soh_threshold=args.soh_threshold,
        ecm_sample_limit=args.ecm_sample_limit,
    )
    print("Wrote dashboard artifacts:")
    for key, path in sorted(paths.items()):
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
