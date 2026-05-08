from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def optimise_soh_weights(cycle_df: pd.DataFrame) -> tuple[float, float]:
    """
    Find (w1, w2) that maximise correlation between fused SOH and
    measured capacity ratio on discharge cycles.
    Returns (w1, w2) with w1 + w2 = 1.
    """
    discharge = cycle_df[cycle_df["cycle_type"] == "discharge"].dropna(
        subset=["capacity_ah", "total_resistance_ohm"]
    ).copy()
    
    if len(discharge) < 5:
        return 0.8, 0.2

    C0 = discharge["capacity_ah"].iloc[0]
    R0 = discharge["total_resistance_ohm"].iloc[0]
    cap_ratio = discharge["capacity_ah"] / C0
    res_ratio = R0 / discharge["total_resistance_ohm"]
    target = cap_ratio  # ground truth: pure capacity ratio

    best_score, best_w1 = -np.inf, 0.8
    for w1 in np.arange(0.5, 1.0, 0.05):
        w2 = 1.0 - w1
        fused = w1 * cap_ratio + w2 * res_ratio
        score = np.corrcoef(fused, target)[0, 1]
        if score > best_score:
            best_score, best_w1 = score, w1

    return float(best_w1), round(float(1.0 - best_w1), 2)


def add_cycle_features(cycle_table: pd.DataFrame, w1: float | None = None, w2: float | None = None) -> pd.DataFrame:
    if cycle_table.empty:
        return cycle_table.copy()

    frame = cycle_table.copy()
    if "total_resistance_ohm" not in frame.columns:
        frame["total_resistance_ohm"] = frame["re_ohm"] + frame["rct_ohm"]
    
    frame["re_rct_ratio"] = frame["re_ohm"] / frame["rct_ohm"].replace(0, np.nan)
    
    # Re/Rct slope per battery (discharge cycles)
    frame["re_rct_slope"] = np.nan
    for bid, group in frame[frame["cycle_type"] == "discharge"].groupby("battery_id"):
        valid = group.dropna(subset=["re_rct_ratio", "cycle_index"])
        if len(valid) > 2:
            slope = np.polyfit(valid["cycle_index"], valid["re_rct_ratio"], 1)[0]
            frame.loc[frame["battery_id"] == bid, "re_rct_slope"] = float(slope)

    if w1 is None or w2 is None:
        w1, w2 = optimise_soh_weights(frame)

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
    frame["soh_w1"] = w1
    frame["soh_w2"] = w2

    frame["rct_delta"] = frame.groupby("battery_id")["rct_ohm"].diff()
    frame["re_delta"] = frame.groupby("battery_id")["re_ohm"].diff()
    frame["temperature_rise_c"] = frame["temperature_max_c"] - frame["temperature_mean_c"]
    return frame


def add_cycle_shape_features(cycle_df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-cycle shape features computed from raw sample vectors.
    Merge result onto cycle_df on 'cycle_index'.
    """
    records = []
    # Ensure columns exist before grouping
    if "cycle_index" not in sample_df.columns or "time_s" not in sample_df.columns:
        return cycle_df.copy()

    for (battery_id, cycle_id), grp in sample_df.groupby(["battery_id", "cycle_index"]):
        grp = grp.sort_values("time_s")
        V = grp["voltage_v"].values
        I = grp["current_a"].values
        
        # 1. dV/dQ — incremental capacity; use median of non-zero-Q region
        # Q_cumulative in Ah
        dt = np.diff(grp["time_s"].values, prepend=grp["time_s"].iloc[0])
        dq_vec = np.abs(I) * dt / 3600.0
        Q_cumulative = np.cumsum(dq_vec)

        dq = np.diff(Q_cumulative)
        dv = np.diff(V)
        with np.errstate(divide="ignore", invalid="ignore"):
            dvdq = np.where(np.abs(dq) > 1e-6, dv / dq, np.nan)
        ic_median = float(np.nanmedian(dvdq)) if not np.all(np.isnan(dvdq)) else np.nan

        # 2. Voltage discharge slope (linear fit slope over discharge period)
        discharge_mask = I < -0.05
        if discharge_mask.sum() > 5:
            v_slope = float(np.polyfit(np.where(discharge_mask)[0], V[discharge_mask], 1)[0])
        else:
            v_slope = np.nan

        # 3. Time to reach 80% capacity cutoff (fraction of cycle duration)
        total_Q = Q_cumulative[-1]
        if total_Q > 0:
            idx_80 = np.searchsorted(Q_cumulative, 0.8 * total_Q)
            duration = grp["time_s"].iloc[-1] - grp["time_s"].iloc[0]
            if duration > 0:
                t80_frac = float((grp["time_s"].iloc[min(idx_80, len(grp)-1)] - grp["time_s"].iloc[0]) / duration)
            else:
                t80_frac = np.nan
        else:
            t80_frac = np.nan

        # 4. Charge–discharge duration asymmetry ratio
        charge_time    = grp.loc[grp["current_a"] > 0.05, "time_s"].count()
        discharge_time = grp.loc[grp["current_a"] < -0.05, "time_s"].count()
        asym_ratio = (
            float(charge_time / discharge_time)
            if discharge_time > 0 else np.nan
        )

        # 5. Rolling variance of voltage (within cycle) — captures noise growth
        v_rolling_var = float(pd.Series(V).rolling(10).var().mean())

        records.append({
            "battery_id":     battery_id,
            "cycle_index":    cycle_id,
            "ic_median":      ic_median,
            "v_discharge_slope": v_slope,
            "t80_frac":       t80_frac,
            "charge_discharge_asym": asym_ratio,
            "voltage_rolling_var":   v_rolling_var,
        })

    if not records:
        return cycle_df.copy()
        
    shape_df = pd.DataFrame(records)
    return cycle_df.merge(shape_df, on=["battery_id", "cycle_index"], how="left")


def add_lag_features(cycle_df: pd.DataFrame, lags: list[int] = [1, 3, 5, 10]) -> pd.DataFrame:
    """
    Appends lag and rolling features for SOH and Coulombic efficiency.
    Only computed on discharge cycles; NaN-filled for others.
    """
    if cycle_df.empty:
        return cycle_df.copy()
        
    df = cycle_df.copy().sort_values(["battery_id", "cycle_index"])
    discharge_mask = df["cycle_type"] == "discharge"

    for lag in lags:
        df.loc[discharge_mask, f"soh_lag_{lag}"] = (
            df.loc[discharge_mask].groupby("battery_id")["soh"].shift(lag)
        )
        if "coulombic_efficiency" in df.columns:
            df.loc[discharge_mask, f"ce_lag_{lag}"] = (
                df.loc[discharge_mask].groupby("battery_id")["coulombic_efficiency"].shift(lag)
            )

    # Rolling variance of SOH (momentum signal)
    df.loc[discharge_mask, "soh_rolling_var_10"] = (
        df.loc[discharge_mask].groupby("battery_id")["soh"]
          .transform(lambda x: x.rolling(window=10, min_periods=3).var())
    )

    # Cycle-over-cycle SOH delta
    df.loc[discharge_mask, "soh_delta_1"] = (
        df.loc[discharge_mask].groupby("battery_id")["soh"].diff(1)
    )

    return df


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
            discharge_soc_mean=("soc", "mean"),
            dod=("soc", lambda x: float(x.max() - x.min())),
        )
    )
    summary["voltage_drop_v"] = summary["voltage_start_v"] - summary["voltage_end_v"]
    summary["temp_rise_c"] = summary["temp_max_c"] - summary["temp_min_c"]
    return summary


def summarize_charge_cycles(sample_table: pd.DataFrame) -> pd.DataFrame:
    if sample_table.empty:
        return pd.DataFrame()

    charge = sample_table[sample_table["cycle_type"] == "charge"].copy()
    if charge.empty:
        return pd.DataFrame()
    
    if "soc" not in charge.columns:
        charge["soc"] = np.nan

    summary = (
        charge.groupby(["battery_id", "cycle_index"], as_index=False)
        .agg(
            charge_duration_s=("time_s", lambda x: float(x.max() - x.min())),
            charge_current_mean_a=("current_a", "mean"),
            charge_temp_mean_c=("temperature_c", "mean"),
            charge_soc_mean=("soc", "mean"),
        )
    )
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


def cluster_operating_regimes(cycle_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Groups cycles into regimes based on operating conditions.
    """
    frame = cycle_df.copy()
    discharge = frame[frame["cycle_type"] == "discharge"].copy()
    
    features = ["temperature_mean_c", "current_mean_a", "duration_s", "dod"]
    # Ensure features exist
    features = [f for f in features if f in discharge.columns]
    
    if len(discharge) < n_clusters or not features:
        frame["operating_regime"] = 0
        return frame
        
    X = discharge[features].fillna(discharge[features].median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    frame["operating_regime"] = np.nan
    frame.loc[discharge.index, "operating_regime"] = clusters.astype(float)
    frame["operating_regime"] = frame["operating_regime"].fillna(0).astype(int)
    
    return frame
