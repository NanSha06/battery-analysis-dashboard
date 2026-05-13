from __future__ import annotations

import argparse
import sys
import json
import hashlib
import datetime
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import yaml
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


def write_run_metadata(config: dict, metrics: dict, artifact_dir: Path):
    run_id = hashlib.md5(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:8]

    metadata = {
        "run_id":     run_id,
        "timestamp":  datetime.datetime.utcnow().isoformat(),
        "config":     config,
        "metrics":    metrics,
    }

    out_path = artifact_dir / f"run_{run_id}.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Also save as latest_run.json for easy access by app.py
    (artifact_dir / "latest_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[pipeline] Run metadata saved -> {out_path}")


# ---------------------------------------------------------------------------
# FIX 1 — ecm_metrics.json key alignment
#
# app.py reads:
#   ecm_metrics.get("mean_rmse_v")      <- wants "Voltage RMSE"
#   ecm_metrics.get("mean_ekf_rmse_v")  <- wants "EKF RMSE"
#
# pipeline.py writes:
#   { "rmse_v": ..., "ekf_rmse_v": ... }   (no "mean_" prefix)
#
# Root cause: naming convention changed in app.py but the pipeline artifact
# was never updated to match.  We patch the JSON after the pipeline run by
# injecting the aliased keys so both old and new readers keep working.
# ---------------------------------------------------------------------------
def patch_ecm_metrics(artifact_dir: Path) -> None:
    metrics_path = artifact_dir / "ecm_metrics.json"
    if not metrics_path.exists():
        print("[patch] WARNING: ecm_metrics.json not found, skipping patch.")
        return

    metrics: dict = json.loads(metrics_path.read_text(encoding="utf-8"))

    changed = False

    # Alias  rmse_v  ->  mean_rmse_v  (Voltage RMSE card)
    if "mean_rmse_v" not in metrics and "rmse_v" in metrics:
        metrics["mean_rmse_v"] = metrics["rmse_v"]
        changed = True

    # Alias  ekf_rmse_v  ->  mean_ekf_rmse_v  (EKF RMSE card)
    if "mean_ekf_rmse_v" not in metrics and "ekf_rmse_v" in metrics:
        metrics["mean_ekf_rmse_v"] = metrics["ekf_rmse_v"]
        changed = True

    if changed:
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(
            f"[patch] ecm_metrics.json patched: "
            f"mean_rmse_v={metrics.get('mean_rmse_v'):.6f}, "
            f"mean_ekf_rmse_v={metrics.get('mean_ekf_rmse_v'):.6f}"
        )
    else:
        print("[patch] ecm_metrics.json already has correct keys, no change needed.")


# ---------------------------------------------------------------------------
# FIX 2 — impedance_metrics.json missing impedance_rmse / phase_rmse_deg
#
# app.py reads (Global Validation Summary panel):
#   impedance_metrics[battery_id].get("impedance_rmse")    <- "Impedance RMSE"
#   impedance_metrics[battery_id].get("phase_rmse_deg")    <- "Phase RMSE"
#
# pipeline.py only writes:
#   { "growth_rate": ..., "max_impedance": ..., "drift_percent": ... }
#
# Root cause: the pipeline computes R0-vs-pulse-impedance validation but
# never persists the RMSE of that comparison.  The app.py *does* compute
# proper EIS-based impedance/phase RMSE inline for the per-battery tab, but
# that path requires live EIS reference frames which aren't available for
# these batteries, so it always falls back to NaN there too.
#
# Best available proxy: use the R0 vs estimated_impedance residuals that
# are already captured in r0_validation.json (field "rmse").  This is the
# same physical quantity — ECM model resistance vs pulse-derived impedance —
# and is what the global validation panel is intended to display.
#
# For phase_rmse_deg there is no equivalent signal in the current pipeline
# (no EIS phase data for these batteries), so we write NaN explicitly rather
# than leaving the key absent, which causes app.py to silently show "n/a".
# ---------------------------------------------------------------------------
def patch_impedance_metrics(artifact_dir: Path) -> None:
    imp_path = artifact_dir / "impedance_metrics.json"
    r0_val_path = artifact_dir / "r0_validation.json"

    if not imp_path.exists():
        print("[patch] WARNING: impedance_metrics.json not found, skipping patch.")
        return

    imp_metrics: dict = json.loads(imp_path.read_text(encoding="utf-8"))

    # Load R0 validation as the source for impedance RMSE values
    r0_validation: dict = {}
    if r0_val_path.exists():
        r0_validation = json.loads(r0_val_path.read_text(encoding="utf-8"))

    changed = False
    for battery_id, batt_metrics in imp_metrics.items():
        needs_imp_rmse = "impedance_rmse" not in batt_metrics
        needs_phase_rmse = "phase_rmse_deg" not in batt_metrics

        if not (needs_imp_rmse or needs_phase_rmse):
            continue  # battery already has both keys

        # impedance_rmse: pull from r0_validation[battery_id]["rmse"]
        if needs_imp_rmse:
            r0_rmse = (
                r0_validation.get(battery_id, {}).get("rmse", None)
            )
            if r0_rmse is not None and math.isfinite(float(r0_rmse)):
                batt_metrics["impedance_rmse"] = float(r0_rmse)
            else:
                # Fallback: derive from max_impedance as a rough scale estimate
                # (only used when r0_validation is absent)
                batt_metrics["impedance_rmse"] = float("nan")
            changed = True

        # phase_rmse_deg: no EIS phase data available for this dataset;
        # write NaN explicitly so app.py shows "n/a" intentionally rather
        # than crashing on a missing key.
        if needs_phase_rmse:
            batt_metrics["phase_rmse_deg"] = float("nan")
            changed = True

    if changed:
        imp_path.write_text(json.dumps(imp_metrics, indent=2), encoding="utf-8")
        summary = {
            bid: {
                "impedance_rmse": v.get("impedance_rmse"),
                "phase_rmse_deg": v.get("phase_rmse_deg"),
            }
            for bid, v in imp_metrics.items()
        }
        print(f"[patch] impedance_metrics.json patched: {summary}")
    else:
        print("[patch] impedance_metrics.json already has correct keys, no change needed.")


def main() -> None:
    # Load config
    config_path = ROOT / "config.yaml"
    config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            
    args = parse_args()
    
    # Merge CLI args with config
    mat_dir = Path(config_data.get("paths", {}).get("raw_data", args.mat_dir))
    output_dir = Path(config_data.get("paths", {}).get("artifacts", args.output_dir))
    nominal_cap = float(config_data.get("models", {}).get("nominal_capacity_ah", args.nominal_capacity_ah))
    soh_thresh = float(config_data.get("models", {}).get("soh_threshold", args.soh_threshold))
    
    paths = build_and_export_dashboard_artifacts(
        mat_dir=mat_dir,
        output_dir=output_dir,
        nominal_capacity_ah=nominal_cap,
        soh_threshold=soh_thresh,
        ecm_sample_limit=args.ecm_sample_limit,
    )

    # ------------------------------------------------------------------
    # Post-run artifact patches
    # These fix key-name mismatches between what pipeline.py writes and
    # what app.py reads, without modifying either of those source files.
    # ------------------------------------------------------------------
    artifact_dir = Path(args.output_dir)
    print("\n[patch] Running artifact compatibility patches...")

    # Fix 1: ecm_metrics.json — add mean_rmse_v / mean_ekf_rmse_v aliases
    patch_ecm_metrics(artifact_dir)

    # Fix 2: impedance_metrics.json — add impedance_rmse / phase_rmse_deg
    patch_impedance_metrics(artifact_dir)

    print("[patch] Done.\n")

    # Write run metadata
    config = {
        "nominal_capacity_ah": args.nominal_capacity_ah,
        "soh_threshold": args.soh_threshold,
        "ecm_sample_limit": args.ecm_sample_limit,
        "mat_dir": str(args.mat_dir),
    }
    # Placeholder metrics; in a full impl, we'd pull these from the 'paths' artifacts
    metrics = {"status": "success", "artifact_count": len(paths)}
    write_run_metadata(config, metrics, Path(args.output_dir))

    print("Wrote dashboard artifacts:")
    for key, path in sorted(paths.items()):
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()