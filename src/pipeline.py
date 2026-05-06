from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import numpy as np

from .data_loader import load_shadow_tables
from .ecm import (
    attach_ecm_state,
    ekf_voltage_error_metrics,
    voltage_error_metrics,
)
from .impedance_validation import (
    analyze_impedance_growth,
    process_battery_impedance,
    validate_r0,
)
from .state_estimators import build_shadow_state


def build_digital_shadow(
    mat_dir: str | Path,
    nominal_capacity_ah: float = 2.0,
    soh_threshold: float = 0.8,
    ecm_sample_limit: int | None = 50000,
) -> dict[str, object]:
    cycle_table, sample_table = load_shadow_tables(mat_dir)
    cycle_state, sample_state, soh_model = build_shadow_state(
        cycle_table=cycle_table,
        sample_table=sample_table,
        nominal_capacity_ah=nominal_capacity_ah,
        soh_threshold=soh_threshold,
    )

    ecm_input = sample_state
    if ecm_sample_limit is not None and len(sample_state) > ecm_sample_limit:
        ecm_input = sample_state.head(ecm_sample_limit).copy()

    sample_shadow, ecm_params, ocv_curve = attach_ecm_state(
        ecm_input,
        nominal_capacity_ah=nominal_capacity_ah,
    )
    metrics = voltage_error_metrics(sample_shadow)
    metrics.update(ekf_voltage_error_metrics(sample_shadow))

    cycle_shadow = cycle_state.copy()
    sample_shadow = sample_shadow.merge(
        cycle_shadow[
            [
                "battery_id",
                "cycle_index",
                "operation_number",
                "discharge_number",
                "soh",
                "soh_model",
                "rul_cycles",
            ]
        ],
        on=["battery_id", "cycle_index"],
        how="left",
    )

    return {
        "cycle_table": cycle_table,
        "sample_table": sample_table,
        "sample_state": sample_state,
        "cycle_shadow": cycle_shadow,
        "sample_shadow": sample_shadow,
        "ocv_curve": ocv_curve,
        "ecm_params": asdict(ecm_params),
        "ecm_metrics": metrics,
        "soh_model": soh_model,
    }


def export_shadow_tables(result: dict[str, object], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in ("cycle_table", "sample_table", "cycle_shadow", "sample_shadow", "ocv_curve"):
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"{key}.csv", index=False)


def export_dashboard_artifacts(result: dict[str, object], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    global_frames = ("cycle_table", "cycle_shadow", "ocv_curve", "impedance_trend", "impedance_curve", "aligned_r0", "eis_reference")

    for key in global_frames:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            path = output_dir / f"{key}.parquet"
            value.to_parquet(path, index=False)
            paths[key] = str(path)

    sample_dir = output_dir / "sample" / "by_battery"
    sample_shadow_dir = output_dir / "sample_shadow" / "by_battery"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_shadow_dir.mkdir(parents=True, exist_ok=True)

    sample_table = result.get("sample_table")
    if isinstance(sample_table, pd.DataFrame) and not sample_table.empty:
        for battery_id, group in sample_table.groupby("battery_id"):
            path = sample_dir / f"{battery_id}.parquet"
            group.to_parquet(path, index=False)
            paths[f"sample_{battery_id}"] = str(path)

    sample_shadow = result.get("sample_shadow")
    if isinstance(sample_shadow, pd.DataFrame) and not sample_shadow.empty:
        for battery_id, group in sample_shadow.groupby("battery_id"):
            path = sample_shadow_dir / f"{battery_id}.parquet"
            group.to_parquet(path, index=False)
            paths[f"sample_shadow_{battery_id}"] = str(path)

    metrics = result.get("ecm_metrics", {})
    metrics_path = output_dir / "ecm_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    paths["ecm_metrics"] = str(metrics_path)

    params = result.get("ecm_params", {})
    params_path = output_dir / "ecm_params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    paths["ecm_params"] = str(params_path)

    battery_metrics = result.get("battery_ecm_metrics", {})
    battery_metrics_path = output_dir / "battery_ecm_metrics.json"
    battery_metrics_path.write_text(json.dumps(battery_metrics, indent=2), encoding="utf-8")
    paths["battery_ecm_metrics"] = str(battery_metrics_path)

    battery_params = result.get("battery_ecm_params", {})
    battery_params_path = output_dir / "battery_ecm_params.json"
    battery_params_path.write_text(json.dumps(battery_params, indent=2), encoding="utf-8")
    paths["battery_ecm_params"] = str(battery_params_path)

    r0_val = result.get("r0_validation", {})
    if r0_val:
        r0_val_path = output_dir / "r0_validation.json"
        r0_val_path.write_text(json.dumps(r0_val, indent=2), encoding="utf-8")
        paths["r0_validation"] = str(r0_val_path)

    imp_met = result.get("impedance_metrics", {})
    if imp_met:
        imp_met_path = output_dir / "impedance_metrics.json"
        imp_met_path.write_text(json.dumps(imp_met, indent=2), encoding="utf-8")
        paths["impedance_metrics"] = str(imp_met_path)
        
    s_met = result.get("scaling_metrics", {})
    if s_met:
        s_met_path = output_dir / "scaling_metrics.json"
        s_met_path.write_text(json.dumps(s_met, indent=2), encoding="utf-8")
        paths["scaling_metrics"] = str(s_met_path)

    manifest = {
        "global_tables": {
            key: paths[key]
            for key in global_frames
            if key in paths
        },
        "sample_dir": str(sample_dir),
        "sample_shadow_dir": str(sample_shadow_dir),
        "metrics_path": str(metrics_path),
        "params_path": str(params_path),
        "battery_metrics_path": str(battery_metrics_path),
        "battery_params_path": str(battery_params_path),
        "scaling_metrics_path": str(s_met_path) if s_met else "",
    }
    if "r0_validation" in paths:
        manifest["r0_validation_path"] = paths["r0_validation"]
    if "impedance_metrics" in paths:
        manifest["impedance_metrics_path"] = paths["impedance_metrics"]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    return paths


def build_and_export_dashboard_artifacts(
    mat_dir: str | Path,
    output_dir: str | Path,
    nominal_capacity_ah: float = 2.0,
    soh_threshold: float = 0.8,
    ecm_sample_limit: int | None = 50000,
) -> dict[str, str]:
    result = build_digital_shadow(
        mat_dir=mat_dir,
        nominal_capacity_ah=nominal_capacity_ah,
        soh_threshold=soh_threshold,
        ecm_sample_limit=ecm_sample_limit,
    )

    sample_state = result.get("sample_state")
    cycle_shadow = result.get("cycle_shadow")
    if isinstance(sample_state, pd.DataFrame) and isinstance(cycle_shadow, pd.DataFrame):
        battery_frames: list[pd.DataFrame] = []
        battery_metrics: dict[str, dict[str, float]] = {}
        battery_params: dict[str, dict[str, float]] = {}

        for battery_id, group in sample_state.groupby("battery_id"):
            ecm_input = group.copy()
            if ecm_sample_limit is not None and len(ecm_input) > ecm_sample_limit:
                ecm_input = ecm_input.head(ecm_sample_limit).copy()

            sample_shadow_battery, ecm_params_battery, _ = attach_ecm_state(
                ecm_input,
                nominal_capacity_ah=nominal_capacity_ah,
            )
            sample_shadow_battery = sample_shadow_battery.merge(
                cycle_shadow[["battery_id", "cycle_index", "soh", "soh_model", "rul_cycles"]],
                on=["battery_id", "cycle_index"],
                how="left",
            )
            battery_frames.append(sample_shadow_battery)
            battery_params[battery_id] = asdict(ecm_params_battery)
            metrics = voltage_error_metrics(sample_shadow_battery)
            metrics.update(ekf_voltage_error_metrics(sample_shadow_battery))
            battery_metrics[battery_id] = metrics

        if battery_frames:
            result["sample_shadow"] = pd.concat(battery_frames, ignore_index=True)
            result["battery_ecm_params"] = battery_params
            result["battery_ecm_metrics"] = battery_metrics

            cycle_shadow = cycle_shadow.copy()
            cycle_shadow["r0"] = pd.NA
            cycle_shadow["r1"] = pd.NA
            cycle_shadow["c1"] = pd.NA
            cycle_shadow["r2"] = pd.NA
            cycle_shadow["c2"] = pd.NA
            cycle_shadow["cycle_voltage_mean_v"] = cycle_shadow.get("voltage_mean_v")
            cycle_shadow["cycle_current_mean_a"] = cycle_shadow.get("current_mean_a")
            cycle_shadow["cycle_temperature_mean_c"] = cycle_shadow.get("temperature_mean_c")
            cycle_shadow["cycle_ecm_mae_v"] = pd.NA
            cycle_shadow["cycle_ecm_rmse_v"] = pd.NA
            cycle_shadow["cycle_ekf_mae_v"] = pd.NA
            cycle_shadow["cycle_ekf_rmse_v"] = pd.NA

            for battery_id, params in battery_params.items():
                battery_mask = cycle_shadow["battery_id"] == battery_id
                battery_frame = cycle_shadow.loc[battery_mask].copy()
                if battery_frame.empty:
                    continue

                soh_reference = battery_frame["soh"].ffill().bfill().fillna(1.0)
                resistance_reference = battery_frame["total_resistance_ohm"].ffill().bfill()

                if resistance_reference.notna().any():
                    res_min = resistance_reference.min()
                    res_max = resistance_reference.max()
                    if pd.notna(res_min) and pd.notna(res_max) and abs(res_max - res_min) > 1e-12:
                        resistance_scale = (resistance_reference - res_min) / (res_max - res_min)
                    else:
                        resistance_scale = pd.Series(0.0, index=battery_frame.index)
                else:
                    resistance_scale = pd.Series(0.0, index=battery_frame.index)

                aging_scale = (1.0 - soh_reference).clip(lower=0.0).fillna(0.0)

                cycle_shadow.loc[battery_mask, "r0"] = float(params["r0"]) * (1.0 + 0.6 * resistance_scale)
                cycle_shadow.loc[battery_mask, "r1"] = float(params["r1"]) * (1.0 + 1.2 * resistance_scale)
                cycle_shadow.loc[battery_mask, "r2"] = float(params["r2"]) * (1.0 + 0.8 * resistance_scale)
                cycle_shadow.loc[battery_mask, "c1"] = float(params["c1"]) * (1.0 - 0.5 * aging_scale)
                cycle_shadow.loc[battery_mask, "c2"] = float(params["c2"]) * (1.0 - 0.3 * aging_scale)

                metrics = battery_metrics.get(battery_id, {})
                cycle_shadow.loc[battery_mask, "cycle_ecm_mae_v"] = metrics.get("mae_v")
                cycle_shadow.loc[battery_mask, "cycle_ecm_rmse_v"] = metrics.get("rmse_v")
                cycle_shadow.loc[battery_mask, "cycle_ekf_mae_v"] = metrics.get("ekf_mae_v")
                cycle_shadow.loc[battery_mask, "cycle_ekf_rmse_v"] = metrics.get("ekf_rmse_v")

            estimated_imp = process_battery_impedance(sample_state)
            if not estimated_imp.empty:
                cycle_shadow = cycle_shadow.merge(estimated_imp, on=["battery_id", "cycle_index"], how="left")
            else:
                cycle_shadow["estimated_impedance_ohm"] = np.nan

            r0_validation = {}
            impedance_metrics = {}
            trend_frames = []
            scaling_metrics = {}
            aligned_r0_frames = []
            eis_ref_frames = []

            for battery_id in battery_params.keys():
                battery_mask = cycle_shadow["battery_id"] == battery_id
                battery_frame = cycle_shadow.loc[battery_mask].copy()
                
                if "re_ohm" in battery_frame.columns:
                    valid_eis = battery_frame.dropna(subset=["re_ohm", "r0"])
                    if not valid_eis.empty:
                        r0_ref_series = valid_eis["re_ohm"].values
                        r0_pred_series = valid_eis["r0"].values
                        
                        r0_ref_mean = float(np.mean(r0_ref_series))
                        r0_pred_mean = float(np.mean(r0_pred_series))
                        
                        if r0_pred_mean > 0:
                            scale_factor = r0_ref_mean / r0_pred_mean
                        else:
                            scale_factor = 1.0
                            
                        cycle_shadow.loc[battery_mask, "r0_aligned"] = cycle_shadow.loc[battery_mask, "r0"] * scale_factor
                        
                        scaling_metrics[battery_id] = {
                            "mean_predicted_r0": r0_pred_mean,
                            "mean_eis_r0": r0_ref_mean,
                            "scale_factor": float(scale_factor),
                            "normalization_detected": bool(scale_factor > 10.0 or scale_factor < 0.1),
                            "unit_consistency": "Ohms (aligned)",
                            "outlier_counts": int(np.sum((r0_pred_series * scale_factor > 1.0) | (r0_pred_series * scale_factor < 0.0)))
                        }
                        
                        aligned_r0_frames.append(cycle_shadow.loc[battery_mask, ["battery_id", "cycle_index", "r0", "r0_aligned"]])
                        eis_ref_frames.append(valid_eis[["battery_id", "cycle_index", "re_ohm"]])
                    else:
                        cycle_shadow.loc[battery_mask, "r0_aligned"] = cycle_shadow.loc[battery_mask, "r0"]
                else:
                    cycle_shadow.loc[battery_mask, "r0_aligned"] = cycle_shadow.loc[battery_mask, "r0"]

                val_frame = cycle_shadow.loc[battery_mask].dropna(subset=["r0_aligned", "estimated_impedance_ohm"])
                if not val_frame.empty:
                    val = validate_r0(val_frame["r0_aligned"], val_frame["estimated_impedance_ohm"])
                    r0_validation[battery_id] = val
                    
                    trend = analyze_impedance_growth(val_frame["cycle_index"], val_frame["estimated_impedance_ohm"])
                    trend["battery_id"] = battery_id
                    trend_frames.append(trend)
                    
                    growth_rate = float(trend["growth_rate"].iloc[0]) if not trend.empty and "growth_rate" in trend else np.nan
                    max_imp = float(battery_frame["estimated_impedance_ohm"].max())
                    impedance_metrics[battery_id] = {
                        "growth_rate": growth_rate,
                        "max_impedance": max_imp,
                        "drift_percent": val["drift_percent"]
                    }

            result["cycle_shadow"] = cycle_shadow
            result["r0_validation"] = r0_validation
            result["impedance_metrics"] = impedance_metrics
            result["scaling_metrics"] = scaling_metrics
            
            if trend_frames:
                result["impedance_trend"] = pd.concat(trend_frames, ignore_index=True)
            else:
                result["impedance_trend"] = pd.DataFrame(columns=["battery_id", "cycle_index", "impedance", "rolling_avg", "growth_rate", "anomaly"])
                
            curve_cols = ["battery_id", "cycle_index", "r0_aligned", "estimated_impedance_ohm"]
            result["impedance_curve"] = cycle_shadow.dropna(subset=["r0_aligned", "estimated_impedance_ohm"])[curve_cols].rename(columns={"r0_aligned": "r0"}) if not cycle_shadow.empty else pd.DataFrame()
            
            result["aligned_r0"] = pd.concat(aligned_r0_frames, ignore_index=True) if aligned_r0_frames else pd.DataFrame()
            result["eis_reference"] = pd.concat(eis_ref_frames, ignore_index=True) if eis_ref_frames else pd.DataFrame()

    return export_dashboard_artifacts(result, output_dir)
