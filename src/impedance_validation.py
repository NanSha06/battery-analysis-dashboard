import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

def estimate_impedance(voltage: np.ndarray | pd.Series, current: np.ndarray | pd.Series, pulse_indices: np.ndarray) -> np.ndarray:
    """
    Estimate impedance Z_est = delta_V / delta_I at the given pulse indices.
    """
    v_arr = np.asarray(voltage, dtype=float)
    i_arr = np.asarray(current, dtype=float)
    
    impedance_est = []
    
    for idx in pulse_indices:
        if idx + 1 >= len(v_arr):
            continue
            
        delta_v = v_arr[idx + 1] - v_arr[idx]
        delta_i = i_arr[idx + 1] - i_arr[idx]
        
        # Avoid divide-by-zero or extremely small current changes
        if abs(delta_i) < 1e-6:
            continue
            
        z = abs(delta_v / delta_i)
        
        # Ignore physically unrealistic values (e.g. > 10 Ohms or < 1e-5 Ohms)
        if 1e-5 <= z <= 10.0:
            impedance_est.append(z)
            
    return np.array(impedance_est, dtype=float)

import warnings

def validate_r0(r0_values: np.ndarray | pd.Series, impedance_values: np.ndarray | pd.Series) -> dict[str, float]:
    """
    Validate ECM-derived R0 against transient estimated impedance.
    """
    r0_arr = np.asarray(r0_values, dtype=float)
    imp_arr = np.asarray(impedance_values, dtype=float)
    
    # Unit Validation Checks & Warnings
    if np.any(r0_arr < 0.0):
        warnings.warn("Validation Warning: R0 < 0 detected. Physical resistance cannot be negative.")
    if np.any(r0_arr > 1.0):
        warnings.warn("Validation Warning: R0 > 1 Ω detected. Ensure values are in Ohms, not normalized units.")
        
    valid_mask = np.isfinite(r0_arr) & np.isfinite(imp_arr)
    r0_valid = r0_arr[valid_mask]
    imp_valid = imp_arr[valid_mask]
    
    if np.any((imp_valid < 1e-5) | (imp_valid > 10.0)):
        warnings.warn("Validation Warning: Impedance magnitude unrealistic (<1e-5 or >10 Ohms).")
    
    if len(r0_valid) < 2:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "correlation": np.nan,
            "trend_consistency": np.nan,
            "drift_percent": np.nan
        }
        
    mae = mean_absolute_error(imp_valid, r0_valid)
    rmse = np.sqrt(mean_squared_error(imp_valid, r0_valid))
    
    # Check if variance is sufficient for correlation
    if np.var(r0_valid) > 1e-12 and np.var(imp_valid) > 1e-12:
        corr, _ = pearsonr(imp_valid, r0_valid)
    else:
        corr = np.nan
        
    # Relative drift (mean percentage error)
    mean_imp = np.mean(imp_valid)
    if mean_imp > 1e-6:
        drift_percent = float(np.mean(r0_valid - imp_valid) / mean_imp * 100.0)
        if abs(drift_percent) > 200.0:
            warnings.warn(f"Validation Warning: Scaling drift exceeds threshold ({drift_percent:.1f}%). Check scaling/normalizers.")
    else:
        drift_percent = np.nan
        
    # Trend consistency (do they slope in the same direction?)
    if len(r0_valid) > 5:
        trend_r0 = np.polyfit(np.arange(len(r0_valid)), r0_valid, 1)[0]
        trend_imp = np.polyfit(np.arange(len(imp_valid)), imp_valid, 1)[0]
        trend_consistency = 1.0 if (trend_r0 * trend_imp > 0) else 0.0
    else:
        trend_consistency = np.nan
        
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "correlation": float(corr),
        "trend_consistency": float(trend_consistency),
        "drift_percent": float(drift_percent)
    }

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

def process_battery_impedance(sample_table: pd.DataFrame) -> pd.DataFrame:
    """
    Process sample-level data for a battery to extract cycle-level impedance estimates.
    """
    if sample_table.empty:
        return pd.DataFrame(columns=["battery_id", "cycle_index", "estimated_impedance_ohm"])
        
    results = []
    
    for (battery_id, cycle_index), group in sample_table.groupby(["battery_id", "cycle_index"], sort=False):
        if len(group) < 10:
            continue
            
        pulses = detect_current_pulses(group["current_a"])
        if len(pulses) == 0:
            continue
            
        impedances = estimate_impedance(group["voltage_v"], group["current_a"], pulses)
        
        if len(impedances) > 0:
            mean_imp = float(np.mean(impedances))
            results.append({
                "battery_id": battery_id,
                "cycle_index": cycle_index,
                "estimated_impedance_ohm": mean_imp
            })
            
    return pd.DataFrame(results)
