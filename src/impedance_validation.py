import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from scipy.stats import pearsonr

def detect_current_pulses(current: np.ndarray | pd.Series, threshold: float = 0.5) -> np.ndarray:
    """
    Identify sudden current transitions that represent discharge/load pulses.
    Returns the indices of the pulse edges.
    """
    current_arr = np.asarray(current, dtype=float)
    if len(current_arr) < 2:
        return np.array([], dtype=int)
        
    delta_i = np.diff(current_arr)
    # Detect sudden changes (pulses) greater than threshold
    pulse_indices = np.where(np.abs(delta_i) > threshold)[0]
    
    return pulse_indices

def estimate_impedance(
    voltage: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    pulse_indices: np.ndarray,
    smooth_window: int = 3,
) -> np.ndarray:
    """
    Estimate impedance Z_est = delta_V / delta_I at the given pulse indices.

    Uses a centred moving-window average over ``smooth_window`` adjacent
    pulse points to suppress single-sample noise spikes.
    """
    v_arr = np.asarray(voltage, dtype=float)
    i_arr = np.asarray(current, dtype=float)

    raw_impedances: list[float] = []

    for idx in pulse_indices:
        if idx + 1 >= len(v_arr):
            continue

        # Use a small window around the pulse edge for robust ΔV/ΔI
        lo = max(0, idx - smooth_window // 2)
        hi = min(len(v_arr) - 1, idx + 1 + smooth_window // 2)

        delta_v = float(np.mean(v_arr[idx + 1 : hi + 1]) - np.mean(v_arr[lo : idx + 1]))
        delta_i = float(np.mean(i_arr[idx + 1 : hi + 1]) - np.mean(i_arr[lo : idx + 1]))

        if abs(delta_i) < 1e-6:
            continue

        z = abs(delta_v / delta_i)

        # Physical range gate
        if 1e-4 <= z <= 5.0:
            raw_impedances.append(z)

    if len(raw_impedances) == 0:
        return np.array([], dtype=float)

    raw = np.array(raw_impedances, dtype=float)

    # --- Adaptive outlier rejection (median ± 2.5 MAD) ---
    med = float(np.median(raw))
    mad = float(np.median(np.abs(raw - med)))
    if mad > 1e-9:
        mask = np.abs(raw - med) <= 2.5 * mad
        raw = raw[mask]

    return raw if len(raw) > 0 else np.array([med], dtype=float)

import warnings

def validate_r0(
    r0_values: np.ndarray | pd.Series,
    impedance_values: np.ndarray | pd.Series,
    cycle_types: np.ndarray | pd.Series | None = None,
    iqr_multiplier: float = 2.0,
) -> dict[str, float]:
    """Validate ECM-derived R0 against transient pulse impedance.

    Improvements over the original:
    - Filters to discharge cycles only when cycle_types is supplied.
    - Rejects outliers via IQR instead of hard physical bounds alone.
    - Reports per-SOC-decile RMSE breakdown (keys: rmse_soc_d0 … rmse_soc_d9).
    """
    r0_arr = np.asarray(r0_values, dtype=float)
    imp_arr = np.asarray(impedance_values, dtype=float)

    # --- Discharge filter ---
    if cycle_types is not None:
        ct = np.asarray(cycle_types, dtype=str)
        discharge_mask = ct == "discharge"
        r0_arr = r0_arr[discharge_mask]
        imp_arr = imp_arr[discharge_mask]

    # --- Basic physical bounds ---
    if np.any(r0_arr < 0.0):
        warnings.warn("Validation Warning: R0 < 0 detected.")
    if np.any(r0_arr > 1.0):
        warnings.warn("Validation Warning: R0 > 1 Ω. Check units.")

    valid_mask = np.isfinite(r0_arr) & np.isfinite(imp_arr)
    r0_valid = r0_arr[valid_mask]
    imp_valid = imp_arr[valid_mask]

    _NAN_RESULT = {
        "mae": np.nan, "rmse": np.nan,
        "correlation": np.nan, "trend_consistency": np.nan,
        "drift_percent": np.nan,
    }

    if len(r0_valid) < 2:
        return _NAN_RESULT

    # --- IQR outlier rejection on the impedance signal ---
    q1, q3 = np.percentile(imp_valid, [25, 75])
    iqr = q3 - q1
    keep = (imp_valid >= q1 - iqr_multiplier * iqr) & (imp_valid <= q3 + iqr_multiplier * iqr)
    r0_valid = r0_valid[keep]
    imp_valid = imp_valid[keep]

    if len(r0_valid) < 2:
        return _NAN_RESULT

    mae = float(mean_absolute_error(imp_valid, r0_valid))
    rmse = float(np.sqrt(mean_squared_error(imp_valid, r0_valid)))

    corr = np.nan
    if np.var(r0_valid) > 1e-12 and np.var(imp_valid) > 1e-12:
        corr, _ = pearsonr(imp_valid, r0_valid)

    mean_imp = float(np.mean(imp_valid))
    drift_percent = float(np.mean(r0_valid - imp_valid) / mean_imp * 100.0) if mean_imp > 1e-6 else np.nan
    if np.isfinite(drift_percent) and abs(drift_percent) > 200.0:
        warnings.warn(f"Scaling drift {drift_percent:.1f}%.")

    trend_consistency = np.nan
    if len(r0_valid) > 5:
        tr0 = np.polyfit(np.arange(len(r0_valid)), r0_valid, 1)[0]
        timp = np.polyfit(np.arange(len(imp_valid)), imp_valid, 1)[0]
        trend_consistency = 1.0 if tr0 * timp > 0 else 0.0

    result: dict[str, float] = {
        "mae": mae,
        "rmse": rmse,
        "correlation": float(corr),
        "trend_consistency": float(trend_consistency),
        "drift_percent": float(drift_percent) if np.isfinite(drift_percent) else np.nan,
    }

    # --- Per-SOC-decile RMSE (requires positional alignment; use imp as proxy) ---
    deciles = np.array_split(np.argsort(imp_valid), 10)
    for d_idx, idx_set in enumerate(deciles):
        if len(idx_set) < 2:
            result[f"rmse_decile_{d_idx}"] = np.nan
        else:
            result[f"rmse_decile_{d_idx}"] = float(
                np.sqrt(np.mean((r0_valid[idx_set] - imp_valid[idx_set]) ** 2))
            )

    return result

def analyze_impedance_growth(cycles: np.ndarray | pd.Series, impedance: np.ndarray | pd.Series) -> pd.DataFrame:
    """
    Analyze degradation trend over cycles based on estimated impedance.
    """
    df = pd.DataFrame({
        "cycle_index": np.asarray(cycles, dtype=float),
        "impedance": np.asarray(impedance, dtype=float)
    }).dropna()
    
    if len(df) < 2:
        return pd.DataFrame(columns=["cycle_index", "impedance", "rolling_avg", "growth_rate", "anomaly"])
        
    df = df.sort_values("cycle_index").reset_index(drop=True)
    df["rolling_avg"] = df["impedance"].rolling(window=min(10, len(df)), min_periods=1).mean()
    
    # Calculate global growth rate (slope) via polyfit
    if len(df) > 5:
        slope, _ = np.polyfit(df["cycle_index"], df["impedance"], 1)
        df["growth_rate"] = slope
    else:
        df["growth_rate"] = np.nan
        
    # Detect anomalies (e.g. spikes > 2 std dev from rolling mean)
    std_dev = df["impedance"].std()
    df["anomaly"] = (np.abs(df["impedance"] - df["rolling_avg"]) > 2 * std_dev).astype(int)
    
    return df

def smooth_impedance_series(
    impedance_values: np.ndarray | pd.Series,
    window: int = 7,
) -> np.ndarray:
    """Apply robust outlier rejection, rolling median, and EMA smoothing.

    The output is the impedance signal used by adaptive scaling,
    validation, R0 tracking, and Nyquist/Bode calibration.
    """
    arr = np.asarray(impedance_values, dtype=float)
    if len(arr) < 2:
        return arr

    s = pd.Series(arr)
    valid = s.replace([np.inf, -np.inf], np.nan)
    global_med = valid.median()
    global_mad = (valid - global_med).abs().median()
    if pd.notna(global_mad) and global_mad > 1e-9:
        valid = valid.mask((valid - global_med).abs() > 3.0 * global_mad)
    valid = valid.interpolate(limit_direction="both").fillna(global_med)

    med = valid.rolling(window=window, min_periods=1, center=True).median()
    residual = (valid - med).abs()
    local_mad = residual.rolling(window=window, min_periods=1, center=True).median()
    stable = valid.mask(residual > (3.0 * local_mad.replace(0.0, np.nan))).interpolate(limit_direction="both").fillna(med)

    alpha = 2.0 / (window + 2)
    ema = med.ewm(alpha=alpha, adjust=False).mean()
    smooth = (0.35 * stable + 0.65 * ema).ewm(alpha=alpha, adjust=False).mean()

    return smooth.to_numpy(dtype=float)


def process_battery_impedance(sample_table: pd.DataFrame) -> pd.DataFrame:
    """
    Process sample-level data for a battery to extract cycle-level impedance
    estimates with smoothing to reduce oscillatory noise.
    """
    if sample_table.empty:
        return pd.DataFrame(columns=[
            "battery_id", "cycle_index",
            "estimated_impedance_ohm", "estimated_impedance_smoothed_ohm",
        ])

    results: list[dict] = []

    for (battery_id, cycle_index), group in sample_table.groupby(
        ["battery_id", "cycle_index"], sort=False
    ):
        if len(group) < 10:
            continue

        pulses = detect_current_pulses(group["current_a"])
        if len(pulses) == 0:
            continue

        impedances = estimate_impedance(
            group["voltage_v"], group["current_a"], pulses
        )

        if len(impedances) > 0:
            # Use median (robust to outliers) instead of mean
            median_imp = float(np.median(impedances))
            results.append({
                "battery_id": battery_id,
                "cycle_index": cycle_index,
                "estimated_impedance_ohm": median_imp,
            })

    df = pd.DataFrame(results)

    if not df.empty:
        # Apply cross-cycle smoothing per battery
        smoothed_parts = []
        for _bid, grp in df.groupby("battery_id", sort=False):
            grp = grp.sort_values("cycle_index").copy()
            grp["estimated_impedance_smoothed_ohm"] = smooth_impedance_series(
                grp["estimated_impedance_ohm"].values
            )
            smoothed_parts.append(grp)
        df = pd.concat(smoothed_parts, ignore_index=True)
    else:
        df["estimated_impedance_smoothed_ohm"] = np.nan

    return df


ANOMALY_FEATURES = [
    "estimated_impedance_smoothed_ohm",  # robust transient impedance
    "soh",                      # state of health
    "temperature_mean_c",       # thermal stress
    "coulombic_efficiency",     # efficiency signal
    "total_resistance_ohm",     # DC resistance
]


def detect_multivariate_anomalies(cycle_df: pd.DataFrame, contamination: float = 0.05) -> pd.Series:
    """
    Returns a boolean Series indexed like cycle_df: True = anomaly.
    contamination=0.05 flags ~5% of cycles as anomalous.
    """
    if cycle_df.empty:
        return pd.Series([], dtype=bool)

    # Filter only existing features
    present_features = [f for f in ANOMALY_FEATURES if f in cycle_df.columns]
    feat_df = cycle_df[present_features].dropna()
    
    if len(feat_df) < 20:
        return pd.Series(False, index=cycle_df.index)

    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    preds = clf.fit_predict(feat_df.values)   # -1 = anomaly, 1 = normal
    anomaly_mask = pd.Series(preds == -1, index=feat_df.index)
    return anomaly_mask.reindex(cycle_df.index, fill_value=False)
