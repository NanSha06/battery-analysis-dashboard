from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_shadow_tables
from .ecm import (
    attach_ecm_state,
    adaptive_r0,
    ekf_voltage_error_metrics,
    voltage_error_metrics,
    interpolate_ocv,
)
from .state_estimators import build_shadow_state
from .rul import fit_stress_coefficients, fit_pooled_rul
from .features import cluster_operating_regimes
from .recommendations import get_charge_recommendation
from .calibration import compute_soh_calibration
from .impedance_validation import (
    analyze_impedance_growth,
    process_battery_impedance,
    validate_r0,
    detect_multivariate_anomalies,
)


CACHE_DIR = Path("artifacts/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(battery_id: str, cycle_index: int) -> Path:
    return CACHE_DIR / f"{battery_id}_cycle_{cycle_index:04d}.parquet"


def compute_cycle_features(battery_id: str, cycle_index: int, cycle_data: pd.DataFrame, sample_data: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for the per-cycle processing logic. 
    This will be expanded as we integrate more features.
    """
    # For now, just return the sample data for this cycle
    # In a real implementation, this would include EKF, SOC estimation, etc.
    return sample_data.copy()


def load_or_compute_cycle(battery_id: str, cycle_index: int, cycle_data: pd.DataFrame, sample_data: pd.DataFrame) -> pd.DataFrame:
    cache_path = get_cache_key(battery_id, cycle_index)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    result = compute_cycle_features(battery_id, cycle_index, cycle_data, sample_data)
    result.to_parquet(cache_path, index=False)
    return result


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

    # --- Group 3 Operating Regime ---
    cycle_state = cluster_operating_regimes(cycle_state)

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
                "soh_model_pred",
                "soh_model_upper",
                "soh_model_lower",
                "rul_cycles",
                "rul_cycles_gpr",
                "rul_p10",
                "rul_p90",
                "rul_mc_p5",
                "rul_mc_p25",
                "rul_mc_p75",
                "rul_mc_p95",
                "rul_arrhenius",
                "knee_cycle",
                "post_knee_degradation_rate",
                "operating_regime",
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
            value.to_csv(output_dir / f"{key}.csv", index=False, encoding="utf-8")


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

    reg_stat = result.get("regime_stats", {})
    if reg_stat:
        reg_stat_path = output_dir / "regime_stats.json"
        reg_stat_path.write_text(json.dumps(reg_stat, indent=2), encoding="utf-8")
        paths["regime_stats"] = str(reg_stat_path)

    # Export calibration artifacts
    calibration_data = result.get("calibration", {})
    for bid, cal_df in calibration_data.items():
        if isinstance(cal_df, pd.DataFrame):
            cal_path = output_dir / f"calibration_{bid}.parquet"
            cal_df.to_parquet(cal_path, index=False)
            paths[f"calibration_{bid}"] = str(cal_path)

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
        "regime_stats_path": str(output_dir / "regime_stats.json") if "regime_stats" in paths else "",
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
        all_batt_cycles: dict[str, pd.DataFrame] = {}
        all_batt_ocv: dict[str, pd.DataFrame] = {}

        for battery_id, group in sample_state.groupby("battery_id"):
            ecm_input = group.copy()
            if ecm_sample_limit is not None and len(ecm_input) > ecm_sample_limit:
                ecm_input = ecm_input.head(ecm_sample_limit).copy()

            sample_shadow_battery, ecm_params_battery, ocv_curve_battery = attach_ecm_state(
                ecm_input,
                nominal_capacity_ah=nominal_capacity_ah,
            )
            sample_shadow_battery = sample_shadow_battery.merge(
                cycle_shadow[
                    [
                        "battery_id",
                        "cycle_index",
                        "soh",
                        "soh_model",
                        "soh_model_pred",
                        "soh_model_upper",
                        "soh_model_lower",
                        "rul_cycles",
                        "rul_cycles_gpr",
                        "rul_p10",
                        "rul_p90",
                        "rul_mc_p5",
                        "rul_mc_p25",
                        "rul_mc_p75",
                        "rul_mc_p95",
                        "rul_arrhenius",
                        "knee_cycle",
                        "post_knee_degradation_rate",
                        "operating_regime",
                    ]
                ],
                on=["battery_id", "cycle_index"],
                how="left",
            )
            battery_frames.append(sample_shadow_battery)
            battery_params[battery_id] = asdict(ecm_params_battery)
            all_batt_ocv[battery_id] = ocv_curve_battery
            metrics = voltage_error_metrics(sample_shadow_battery)
            metrics.update(ekf_voltage_error_metrics(sample_shadow_battery))
            battery_metrics[battery_id] = metrics
            all_batt_cycles[battery_id] = cycle_shadow[cycle_shadow["battery_id"] == battery_id]

        # 1. Fit Global Stress Coefficients & Pooled RUL
        stress_coeffs = fit_stress_coefficients(all_batt_cycles)
        pooled_rul = fit_pooled_rul(all_batt_cycles)
        
        # 2. Multivariate Anomaly Detection
        cycle_shadow["anomaly"] = detect_multivariate_anomalies(cycle_shadow)

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

                # --- Adaptive R0: physics-based per-cycle computation ---
                soh_series = battery_frame["soh"].ffill().bfill().fillna(1.0) if "soh" in battery_frame.columns else pd.Series(1.0, index=battery_frame.index)
                
                # We need ECMParameters instance
                from .ecm import ECMParameters
                base_params = ECMParameters(**params) if isinstance(params, dict) else params

                from .ecm import get_dynamic_params
                dynamic_params = get_dynamic_params(
                    battery_frame["discharge_soc_mean"].fillna(
                        battery_frame.get("charge_soc_mean", pd.Series(0.5, index=battery_frame.index))
                    ).fillna(0.5),
                    battery_frame["temperature_mean_c"].ffill().bfill().fillna(25.0),
                    soh_series,
                    base_params,
                )

                cycle_shadow.loc[battery_mask, "r0"] = dynamic_params["r0_dynamic"].values
                cycle_shadow.loc[battery_mask, "r1"] = dynamic_params["r1_dynamic"].values
                cycle_shadow.loc[battery_mask, "r2"] = dynamic_params["r2_dynamic"].values
                cycle_shadow.loc[battery_mask, "c1"] = dynamic_params["c1_dynamic"].values
                cycle_shadow.loc[battery_mask, "c2"] = dynamic_params["c2_dynamic"].values

                # --- Group 1 Physics Features ---
                battery_rows = cycle_shadow.loc[battery_mask].copy()
                
                # 1. State of Power (SOP)
                # Use mean discharge SOC if available, else mean charge SOC, else 0.5
                soc_sop = battery_rows["discharge_soc_mean"].fillna(battery_rows["charge_soc_mean"]).fillna(0.5).to_numpy()
                batt_ocv = all_batt_ocv.get(battery_id, pd.DataFrame())
                ocv_sop = interpolate_ocv(soc_sop, batt_ocv)
                r_total = (battery_rows["r0"] + battery_rows["r1"] + battery_rows["r2"]).to_numpy(dtype=float)
                v_min = 3.0
                # SOP = ((V_min - OCV(SOC)) / (R0 + R1 + R2)) * V_min
                with np.errstate(divide="ignore", invalid="ignore"):
                    sop_values = ((v_min - ocv_sop) / np.where(r_total > 1e-6, r_total, np.nan)) * v_min
                cycle_shadow.loc[battery_mask, "sop_w"] = sop_values

                # 2. Lithium Plating Risk Index
                # plating_risk = clip((charge_rate_c / max(temperature_c, 1)) * (1 - soc), 0, 1)
                charge_current = battery_rows["charge_current_mean_a"].fillna(0.0).to_numpy()
                charge_rate_c = np.abs(charge_current) / nominal_capacity_ah
                charge_temp = battery_rows["charge_temp_mean_c"].fillna(25.0).to_numpy()
                charge_soc = battery_rows["charge_soc_mean"].fillna(0.0).to_numpy()
                
                risk = (charge_rate_c / np.maximum(charge_temp, 1.0)) * (1.0 - charge_soc)
                cycle_shadow.loc[battery_mask, "plating_risk"] = np.clip(risk, 0.0, 1.0)

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
                        # Calculate total aligned resistance for better pulse validation
                        cycle_shadow.loc[battery_mask, "r_total_aligned"] = (
                            cycle_shadow.loc[battery_mask, "r0"] + 
                            cycle_shadow.loc[battery_mask, "r1"] + 
                            cycle_shadow.loc[battery_mask, "r2"]
                        ) * scale_factor
                        
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

                imp_col = (
                    "estimated_impedance_smoothed_ohm"
                    if "estimated_impedance_smoothed_ohm" in cycle_shadow.columns
                    else "estimated_impedance_ohm"
                )
                val_frame = cycle_shadow.loc[battery_mask].dropna(subset=["r_total_aligned", imp_col])
                if not val_frame.empty:
                    # Compare Total Model Resistance to Total Pulse Impedance
                    val = validate_r0(val_frame["r_total_aligned"], val_frame[imp_col])
                    r0_validation[battery_id] = val
                    
                    trend = analyze_impedance_growth(val_frame["cycle_index"], val_frame[imp_col])
                    trend["battery_id"] = battery_id
                    trend_frames.append(trend)
                    
                    growth_rate = float(trend["growth_rate"].iloc[0]) if not trend.empty and "growth_rate" in trend else np.nan
                    max_imp = float(battery_frame[imp_col].max())
                    impedance_metrics[battery_id] = {
                        "growth_rate": growth_rate,
                        "max_impedance": max_imp,
                        "drift_percent": val["drift_percent"]
                    }

            result["cycle_shadow"] = cycle_shadow
            result["r0_validation"] = r0_validation
            result["impedance_metrics"] = impedance_metrics
            result["scaling_metrics"] = scaling_metrics
            result["global_models"] = {
                "stress_coeffs": stress_coeffs,
                "pooled_rul_r2": pooled_rul.get("loco_r2", []),
            }
            
            if trend_frames:
                result["impedance_trend"] = pd.concat(trend_frames, ignore_index=True)
            else:
                result["impedance_trend"] = pd.DataFrame(columns=["battery_id", "cycle_index", "impedance", "rolling_avg", "growth_rate", "anomaly"])
                
            # --- Group 3 Operating Regime Stats ---
            regime_stats = {}
            for regime, group in cycle_shadow[cycle_shadow["cycle_type"] == "discharge"].groupby("operating_regime"):
                group = group.sort_values(["battery_id", "cycle_index"])
                group["soh_diff"] = group.groupby("battery_id")["soh"].diff()
                group["cycle_diff"] = group.groupby("battery_id")["cycle_index"].diff()
                
                with np.errstate(divide="ignore", invalid="ignore"):
                    rates = group["soh_diff"] / group["cycle_diff"]
                
                regime_stats[int(regime)] = {
                    "mean_degradation_rate": float(rates.mean()),
                    "std_degradation_rate": float(rates.std()),
                    "mean_temperature": float(group["temperature_mean_c"].mean()),
                    "mean_current": float(group["current_mean_a"].mean()),
                    "cycle_count": int(len(group))
                }
            result["regime_stats"] = regime_stats
            result["cycle_shadow"] = cycle_shadow # Update with regime labels
            
            # --- Group 4 Calibration ---
            calibration_map = {}
            for bid in battery_params.keys():
                batt_cycles = cycle_shadow[cycle_shadow["battery_id"] == bid]
                calibration_map[bid] = compute_soh_calibration(batt_cycles)
            result["calibration"] = calibration_map
                
            imp_col_global = (
                "estimated_impedance_smoothed_ohm"
                if "estimated_impedance_smoothed_ohm" in cycle_shadow.columns
                else "estimated_impedance_ohm"
            )
            curve_cols = ["battery_id", "cycle_index", "r0_aligned", imp_col_global]
            result["impedance_curve"] = cycle_shadow.dropna(subset=["r0_aligned", imp_col_global])[curve_cols].rename(columns={"r0_aligned": "r0", imp_col_global: "estimated_impedance_ohm"}) if not cycle_shadow.empty else pd.DataFrame()
            
            result["aligned_r0"] = pd.concat(aligned_r0_frames, ignore_index=True) if aligned_r0_frames else pd.DataFrame()
            result["eis_reference"] = pd.concat(eis_ref_frames, ignore_index=True) if eis_ref_frames else pd.DataFrame()

    return export_dashboard_artifacts(result, output_dir)
