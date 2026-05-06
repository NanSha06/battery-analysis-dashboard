from __future__ import annotations

import numpy as np
import pandas as pd


def add_cycle_features(cycle_table: pd.DataFrame, w1: float = 0.8, w2: float = 0.2) -> pd.DataFrame:
    if cycle_table.empty:
        return cycle_table.copy()

    frame = cycle_table.copy()
    frame["total_resistance_ohm"] = frame["re_ohm"] + frame["rct_ohm"]
    frame["re_rct_ratio"] = frame["re_ohm"] / frame["rct_ohm"].replace(0, np.nan)

    initial_capacity = (
        frame.loc[frame["cycle_type"] == "discharge"]
        .groupby("battery_id")["capacity_ah"]
        .transform("first")
    )
    initial_resistance = (
        frame.loc[frame["cycle_type"] == "discharge"]
        .groupby("battery_id")["total_resistance_ohm"]
        .transform("first")
    )
    
    discharge_mask = frame["cycle_type"] == "discharge"
    frame.loc[discharge_mask, "initial_capacity_ah"] = initial_capacity
    frame.loc[discharge_mask, "initial_resistance_ohm"] = initial_resistance
    
    capacity_norm = frame.loc[discharge_mask, "capacity_ah"] / frame.loc[discharge_mask, "initial_capacity_ah"]
    resistance_norm = frame.loc[discharge_mask, "initial_resistance_ohm"] / frame.loc[discharge_mask, "total_resistance_ohm"].replace(0, np.nan)
    resistance_norm = resistance_norm.fillna(1.0)
    
    frame.loc[discharge_mask, "soh"] = w1 * capacity_norm + w2 * resistance_norm

    frame["rct_delta"] = frame.groupby("battery_id")["rct_ohm"].diff()
    frame["re_delta"] = frame.groupby("battery_id")["re_ohm"].diff()
    frame["temperature_rise_c"] = frame["temperature_max_c"] - frame["temperature_mean_c"]
    return frame

def add_sample_features(sample_table: pd.DataFrame) -> pd.DataFrame:
    if sample_table.empty:
        return sample_table.copy()

    frame = sample_table.copy()
    frame["dt_s"] = frame.groupby(["battery_id", "cycle_index"])["time_s"].diff().fillna(0.0)
    frame["power_w"] = frame["voltage_v"] * frame["current_a"]
    frame["abs_current_a"] = frame["current_a"].abs()
    return frame


def summarize_discharge_cycles(sample_table: pd.DataFrame) -> pd.DataFrame:
    if sample_table.empty:
        return pd.DataFrame()

    discharge = sample_table[sample_table["cycle_type"] == "discharge"].copy()
    if discharge.empty:
        return pd.DataFrame()

    discharge["incremental_energy_wh"] = (
        discharge["power_w"] * discharge["dt_s"] / 3600.0
    )

    summary = (
        discharge.groupby(["battery_id", "cycle_index"], as_index=False)
        .agg(
            discharge_duration_s=("time_s", lambda x: float(x.max() - x.min())),
            voltage_mean_v=("voltage_v", "mean"),
            voltage_start_v=("voltage_v", "first"),
            voltage_end_v=("voltage_v", "last"),
            discharge_energy_wh=("incremental_energy_wh", "sum"),
            temp_min_c=("temperature_c", "min"),
            temp_max_c=("temperature_c", "max"),
            current_mean_a=("current_a", "mean"),
        )
    )
    summary["voltage_drop_v"] = summary["voltage_start_v"] - summary["voltage_end_v"]
    summary["temp_rise_c"] = summary["temp_max_c"] - summary["temp_min_c"]
    return summary


def compute_cycle_efficiency(sample_table: pd.DataFrame) -> pd.DataFrame:
    if sample_table.empty:
        return pd.DataFrame()

    frame = add_sample_features(sample_table)
    frame["incremental_ah"] = frame["current_a"].abs() * frame["dt_s"] / 3600.0

    charge = (
        frame[frame["cycle_type"] == "charge"]
        .groupby(["battery_id", "cycle_index"], as_index=False)
        .agg(charge_ah=("incremental_ah", "sum"))
        .sort_values(["battery_id", "cycle_index"])
    )
    discharge = (
        frame[frame["cycle_type"] == "discharge"]
        .groupby(["battery_id", "cycle_index"], as_index=False)
        .agg(discharge_ah=("incremental_ah", "sum"))
        .sort_values(["battery_id", "cycle_index"])
    )

    rows: list[dict[str, float | str | int]] = []
    for battery_id, discharge_group in discharge.groupby("battery_id", sort=False):
        charge_group = charge[charge["battery_id"] == battery_id]
        charge_records = charge_group.to_dict("records")
        charge_pos = 0

        for discharge_row in discharge_group.to_dict("records"):
            while (
                charge_pos + 1 < len(charge_records)
                and charge_records[charge_pos + 1]["cycle_index"] < discharge_row["cycle_index"]
            ):
                charge_pos += 1

            if not charge_records or charge_records[charge_pos]["cycle_index"] > discharge_row["cycle_index"]:
                charge_ah = np.nan
                charge_cycle_index = np.nan
            else:
                charge_ah = float(charge_records[charge_pos]["charge_ah"])
                charge_cycle_index = int(charge_records[charge_pos]["cycle_index"])

            discharge_ah = float(discharge_row["discharge_ah"])
            efficiency = discharge_ah / charge_ah if charge_ah and np.isfinite(charge_ah) else np.nan
            efficiency_plausible = bool(np.isfinite(efficiency) and 0.0 < efficiency <= 1.2)
            rows.append(
                {
                    "battery_id": battery_id,
                    "cycle_index": int(discharge_row["cycle_index"]),
                    "charge_cycle_index": charge_cycle_index,
                    "charge_ah": charge_ah,
                    "discharge_ah": discharge_ah,
                    "coulombic_efficiency": efficiency if efficiency_plausible else np.nan,
                    "coulombic_efficiency_raw": efficiency,
                    "coulombic_efficiency_plausible": efficiency_plausible,
                }
            )

    return pd.DataFrame(rows)


def compute_efficiency_trends(
    efficiency_table: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    if efficiency_table.empty:
        return efficiency_table.copy()

    frame = efficiency_table.sort_values(["battery_id", "cycle_index"]).copy()
    grouped = frame.groupby("battery_id")["coulombic_efficiency"]
    frame["coulombic_efficiency_rollmean"] = grouped.transform(
        lambda series: series.rolling(window=window, min_periods=2).mean()
    )
    frame["coulombic_efficiency_decline_rate"] = grouped.transform(
        lambda series: series.rolling(window=window, min_periods=2).apply(
            lambda values: np.polyfit(np.arange(len(values)), values, 1)[0]
            if np.isfinite(values).sum() >= 2
            else np.nan,
            raw=True,
        )
    )
    return frame
