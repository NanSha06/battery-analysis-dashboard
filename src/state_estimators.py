from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .features import add_cycle_features, add_sample_features, summarize_discharge_cycles
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

    for (_, cycle_index), group in frame.groupby(["battery_id", "cycle_index"]):
        idx = group.index
        current = group["current_a"].fillna(0.0).to_numpy(dtype=float)
        dt_s = group["dt_s"].fillna(0.0).to_numpy(dtype=float)
        cycle_type = str(group["cycle_type"].iloc[0]).lower()
        soc = np.zeros(len(group), dtype=float)

        if cycle_type == "charge":
            soc[0] = 0.1
            direction = 1.0
        else:
            soc[0] = 1.0
            direction = -1.0

        for i in range(1, len(group)):
            delta_ah = current[i] * dt_s[i] / 3600.0
            soc[i] = np.clip(soc[i - 1] + direction * delta_ah / nominal_capacity_ah, 0.0, 1.0)

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


def estimate_soh_regression(cycle_table: pd.DataFrame) -> tuple[pd.DataFrame, Ridge | None]:
    frame = add_cycle_features(cycle_table)
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
    ]
    usable = discharge.dropna(subset=["soh"]).copy()
    X = usable[feature_cols].fillna(0.0)
    y = usable["soh"]

    if len(usable) < 3:
        frame["soh_model"] = np.nan
        return frame, None

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    frame["soh_model"] = np.nan
    frame.loc[usable.index, "soh_model"] = model.predict(X)
    return frame, model


def build_shadow_state(
    cycle_table: pd.DataFrame,
    sample_table: pd.DataFrame,
    nominal_capacity_ah: float = 2.0,
    soh_threshold: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, Ridge | None]:
    sample_state = estimate_soc_coulomb_counting(sample_table, nominal_capacity_ah=nominal_capacity_ah)
    cycle_state, soh_model = estimate_soh_regression(cycle_table)
    cycle_state, sample_state = add_cycle_counters(cycle_state, sample_state)
    discharge_summary = summarize_discharge_cycles(sample_state)

    if not discharge_summary.empty:
        cycle_state = cycle_state.merge(
            discharge_summary,
            on=["battery_id", "cycle_index"],
            how="left",
        )

    cycle_state = add_rul_estimates(cycle_state, soh_threshold=soh_threshold)
    return cycle_state, sample_state, soh_model
