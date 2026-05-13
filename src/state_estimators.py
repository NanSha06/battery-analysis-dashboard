from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from .features import (
    add_cycle_features,
    add_cycle_shape_features,
    add_lag_features,
    add_sample_features,
    compute_cycle_efficiency,
    summarize_discharge_cycles,
    summarize_charge_cycles,
)
from .rul import add_rul_estimates


def add_cycle_counters(
    cycle_table: pd.DataFrame,
    sample_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle_frame = cycle_table.copy()
    sample_frame = sample_table.copy()

    if cycle_frame.empty:
        return cycle_frame, sample_frame

    cycle_frame["operation_number"] = cycle_frame.groupby("battery_id").cumcount() + 1
    cycle_frame["discharge_number"] = np.nan

    discharge_mask = cycle_frame["cycle_type"] == "discharge"
    cycle_frame.loc[discharge_mask, "discharge_number"] = (
        cycle_frame.loc[discharge_mask]
        .groupby("battery_id")
        .cumcount()
        + 1
    )

    counter_cols = ["battery_id", "cycle_index", "operation_number", "discharge_number"]
    sample_frame = sample_frame.merge(
        cycle_frame[counter_cols],
        on=["battery_id", "cycle_index"],
        how="left",
    )
    return cycle_frame, sample_frame


def estimate_soc_coulomb_counting(
    sample_table: pd.DataFrame,
    nominal_capacity_ah: float = 2.0,
) -> pd.DataFrame:
    if sample_table.empty:
        return sample_table.copy()

    frame = add_sample_features(sample_table)
    frame["soc"] = np.nan

    for (battery_id, cycle_index), group in frame.groupby(["battery_id", "cycle_index"]):
        idx = group.index
        current = group["current_a"].fillna(0.0).to_numpy(dtype=float)
        dt_s = group["dt_s"].fillna(0.0).to_numpy(dtype=float)
        voltage = group["voltage_v"].to_numpy(dtype=float)
        cycle_type = str(group["cycle_type"].iloc[0]).lower()

        # --- OCV-based SOC seed (replaces hard-coded 0.1 / 1.0) ---
        v_first = float(voltage[0]) if np.isfinite(voltage[0]) else np.nan
        soc_from_ocv = float(
            np.interp(v_first,
                      [3.0, 3.5, 3.7, 3.9, 4.0, 4.2],
                      [0.0, 0.2, 0.5, 0.75, 0.9, 1.0])
        ) if np.isfinite(v_first) else (0.1 if cycle_type == "charge" else 1.0)

        soc = np.zeros(len(group), dtype=float)
        soc[0] = float(np.clip(soc_from_ocv, 0.0, 1.0))

        direction = 1.0 if cycle_type == "charge" else -1.0

        for i in range(1, len(group)):
            delta_ah = current[i] * dt_s[i] / 3600.0
            soc[i] = float(np.clip(soc[i - 1] + direction * delta_ah / nominal_capacity_ah, 0.0, 1.0))

            # Cutoff-based drift correction: anchor to 0.0 / 1.0 at physical limits
            if cycle_type == "discharge" and np.isfinite(voltage[i]) and voltage[i] <= 3.0:
                soc[i] = 0.0
            elif cycle_type == "charge" and np.isfinite(voltage[i]) and voltage[i] >= 4.2:
                soc[i] = 1.0

        frame.loc[idx, "soc"] = soc

    return frame


def estimate_soc_ocv(
    voltage: pd.Series | np.ndarray,
    ocv_points: pd.DataFrame | None = None,
) -> np.ndarray:
    voltage_values = np.asarray(voltage, dtype=float)
    if ocv_points is None or ocv_points.empty:
        default_ocv = np.asarray([3.20, 3.55, 3.75, 3.95, 4.20], dtype=float)
        default_soc = np.asarray([0.00, 0.25, 0.50, 0.75, 1.00], dtype=float)
        return np.interp(voltage_values, default_ocv, default_soc, left=0.0, right=1.0)

    curve = ocv_points.dropna(subset=["ocv_v", "soc"]).sort_values("ocv_v")
    if curve.empty:
        return np.full_like(voltage_values, fill_value=np.nan, dtype=float)
    return np.interp(
        voltage_values,
        curve["ocv_v"].to_numpy(dtype=float),
        curve["soc"].to_numpy(dtype=float),
        left=float(curve["soc"].iloc[0]),
        right=float(curve["soc"].iloc[-1]),
    )


def apply_soc_anchor(
    sample_table: pd.DataFrame,
    ocv_points: pd.DataFrame | None = None,
    alpha: float = 0.85,
    current_epsilon: float = 0.05,
    voltage_slope_epsilon: float = 0.002,
) -> pd.DataFrame:
    if sample_table.empty:
        return sample_table.copy()

    frame = sample_table.copy()
    frame["soc_raw"] = frame["soc"] if "soc" in frame else np.nan
    frame["soc_ocv"] = estimate_soc_ocv(frame["voltage_v"], ocv_points)
    frame["voltage_slope_v_per_s"] = (
        frame.groupby(["battery_id", "cycle_index"])["voltage_v"].diff()
        / frame.groupby(["battery_id", "cycle_index"])["time_s"].diff().replace(0, np.nan)
    )
    frame["soc_anchor_window"] = (
        frame["current_a"].abs().le(current_epsilon)
        & frame["voltage_slope_v_per_s"].abs().le(voltage_slope_epsilon)
    )
    frame["soc_corrected"] = frame["soc_raw"]
    anchor_mask = frame["soc_anchor_window"] & frame["soc_raw"].notna() & frame["soc_ocv"].notna()
    frame.loc[anchor_mask, "soc_corrected"] = (
        alpha * frame.loc[anchor_mask, "soc_raw"]
        + (1.0 - alpha) * frame.loc[anchor_mask, "soc_ocv"]
    ).clip(0.0, 1.0)
    return frame


def estimate_soh_regression(cycle_table: pd.DataFrame) -> tuple[pd.DataFrame, BayesianRidge | None]:
    frame = cycle_table.copy()
    discharge = frame[frame["cycle_type"] == "discharge"].copy()
    if discharge.empty:
        return frame, None

    feature_cols = [
        "duration_s",
        "voltage_min_v",
        "voltage_max_v",
        "temperature_mean_c",
        "temperature_max_c",
        "total_resistance_ohm",
        "rct_delta",
        "re_delta",
        "ic_median",
        "v_discharge_slope",
        "t80_frac",
        "charge_discharge_asym",
        "voltage_rolling_var",
        "soh_lag_1",
        "soh_delta_1",
        "soh_rolling_var_10",
        "dod",
    ]
    # Filter features that are actually present
    feature_cols = [c for c in feature_cols if c in discharge.columns]
    
    usable = discharge.dropna(subset=["soh"] + feature_cols).copy()
    frame["soh_model"] = np.nan
    frame["soh_model_pred"] = np.nan
    frame["soh_model_lower"] = np.nan
    frame["soh_model_upper"] = np.nan
    
    if len(usable) < 5:
        return frame, None

    X = usable[feature_cols]
    y = usable["soh"]

    model = BayesianRidge()
    model.fit(X, y)
    
    X_all = discharge[feature_cols].fillna(0.0)
    soh_pred, soh_std = model.predict(X_all, return_std=True)
    
    # --- Group 4 Uncertainty Widening ---
    # Multiply soh_std by (1 / soh_pred.clip(0.1, 1.0))
    soh_std = soh_std * (1.0 / np.clip(soh_pred, 0.1, 1.0))
    
    frame["soh_model"] = np.nan
    frame["soh_model_pred"] = np.nan
    frame["soh_model_lower"] = np.nan
    frame["soh_model_upper"] = np.nan
    
    frame.loc[discharge.index, "soh_model"] = soh_pred
    frame.loc[discharge.index, "soh_model_pred"] = soh_pred
    frame.loc[discharge.index, "soh_model_lower"] = soh_pred - 1.65 * soh_std
    frame.loc[discharge.index, "soh_model_upper"] = soh_pred + 1.65 * soh_std
    
    return frame, model


def build_shadow_state(
    cycle_table: pd.DataFrame,
    sample_table: pd.DataFrame,
    nominal_capacity_ah: float = 2.0,
    soh_threshold: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, BayesianRidge | None]:
    sample_state = estimate_soc_coulomb_counting(sample_table, nominal_capacity_ah=nominal_capacity_ah)
    
    # 1. Basic cycle features and SOH fusion
    cycle_state = add_cycle_features(cycle_table)
    
    # 2. Add efficiency
    efficiency = compute_cycle_efficiency(sample_state)
    if not efficiency.empty:
        cycle_state = cycle_state.merge(efficiency, on=["battery_id", "cycle_index"], how="left")
    
    # 3. Shape features from sample data
    cycle_state = add_cycle_shape_features(cycle_state, sample_state)
    
    # 4. Temporal lag features
    cycle_state = add_lag_features(cycle_state)
    
    # 5. Summaries (moved up for SOH features)
    cycle_state, sample_state = add_cycle_counters(cycle_state, sample_state)
    
    discharge_summary = summarize_discharge_cycles(sample_state)
    if not discharge_summary.empty:
        cols_to_use = [c for c in discharge_summary.columns if c not in cycle_state.columns or c in ["battery_id", "cycle_index"]]
        cycle_state = cycle_state.merge(
            discharge_summary[cols_to_use],
            on=["battery_id", "cycle_index"],
            how="left",
        )

    charge_summary = summarize_charge_cycles(sample_state)
    if not charge_summary.empty:
        cols_to_use = [c for c in charge_summary.columns if c not in cycle_state.columns or c in ["battery_id", "cycle_index"]]
        cycle_state = cycle_state.merge(
            charge_summary[cols_to_use],
            on=["battery_id", "cycle_index"],
            how="left",
        )

    # 6. SOH regression model
    cycle_state, soh_model = estimate_soh_regression(cycle_state)
    
    cycle_state = add_rul_estimates(cycle_state, soh_threshold=soh_threshold)
    return cycle_state, sample_state, soh_model
