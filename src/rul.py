from __future__ import annotations

import numpy as np
import pandas as pd


def fit_linear_degradation(cycle_index: np.ndarray, soh: np.ndarray) -> tuple[float, float]:
    x = np.asarray(cycle_index, dtype=float)
    y = np.asarray(soh, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def estimate_eol_cycle(
    cycle_index: np.ndarray,
    soh: np.ndarray,
    soh_threshold: float = 0.8,
) -> float:
    slope, intercept = fit_linear_degradation(cycle_index, soh)
    if not np.isfinite(slope) or abs(slope) < 1e-10:
        return np.nan
    return float((soh_threshold - intercept) / slope)


def add_rul_estimates(
    cycle_table: pd.DataFrame,
    soh_threshold: float = 0.8,
) -> pd.DataFrame:
    if cycle_table.empty:
        return cycle_table.copy()

    frame = cycle_table.copy()
    frame["estimated_eol_cycle"] = np.nan
    frame["rul_cycles"] = np.nan

    discharge = frame[frame["cycle_type"] == "discharge"].copy()
    if discharge.empty or "soh" not in discharge:
        return frame

    for battery_id, group in discharge.groupby("battery_id"):
        eol_cycle = estimate_eol_cycle(group["cycle_index"], group["soh"], soh_threshold=soh_threshold)
        
        if "temperature_max_c" in group and "total_resistance_ohm" in group and np.isfinite(eol_cycle):
            temp_stress = np.clip((group["temperature_max_c"].mean() - 30.0) / 20.0, 0.0, 0.15)
            res_stress = 0.0
            if group["total_resistance_ohm"].notna().any():
                res_max = group["total_resistance_ohm"].max()
                res_min = group["total_resistance_ohm"].min()
                if res_min > 0:
                    res_stress = np.clip((res_max - res_min) / res_min * 0.1, 0.0, 0.15)
            eol_cycle = eol_cycle * (1.0 - float(temp_stress) - float(res_stress))
            
        mask = (frame["battery_id"] == battery_id) & (frame["cycle_type"] == "discharge")
        frame.loc[mask, "estimated_eol_cycle"] = eol_cycle
        frame.loc[mask, "rul_cycles"] = eol_cycle - frame.loc[mask, "cycle_index"]

    return frame
