from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern, DotProduct
from kneed import KneeLocator
from scipy.optimize import curve_fit


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


def fit_stress_coefficients(all_battery_data: dict) -> dict:
    """
    all_battery_data: {battery_id: cycle_df with columns
                       ['temperature_mean_c', 'total_resistance_ohm',
                        'soh', 'cycle_index']}
    Returns dict of fitted coefficients.
    """
    X_rows, y_rows = [], []

    for bid, df in all_battery_data.items():
        discharge = df[df["cycle_type"] == "discharge"].copy()
        if len(discharge) < 20: continue
        
        T_mean = discharge["temperature_mean_c"].mean()
        R_min  = discharge["total_resistance_ohm"].min()
        R_max  = discharge["total_resistance_ohm"].max()

        sigma_temp = np.clip((T_mean - 30) / 20, 0, 0.15)
        sigma_res  = np.clip((R_max - R_min) / R_min * 0.1, 0, 0.15)

        # Ground truth: actual EOL cycle from data
        eol_actual_series = discharge[discharge["soh"] <= 0.80]["cycle_index"]
        if eol_actual_series.empty: continue
        eol_actual = eol_actual_series.min()
        
        # Linear model EOL
        eol_linear = estimate_eol_cycle(discharge["cycle_index"], discharge["soh"])

        X_rows.append([sigma_temp, sigma_res])
        y_rows.append((eol_actual - eol_linear) / max(eol_linear, 1))

    if not X_rows:
        return {"coef_temp": -1.0, "coef_res": -1.0, "intercept": 0.0}

    X, y = np.array(X_rows, dtype=float), np.array(y_rows, dtype=float)
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    if len(X) < 2:
        return {"coef_temp": -1.0, "coef_res": -1.0, "intercept": 0.0}

    model = Ridge(alpha=0.1).fit(X, y)

    return {
        "coef_temp": float(model.coef_[0]),
        "coef_res":  float(model.coef_[1]),
        "intercept": float(model.intercept_),
    }


def fit_gpr_rul(discharge_df: pd.DataFrame, soh_threshold: float = 0.80) -> dict:
    """
    Returns:
        rul_median:   array of per-cycle median RUL
        rul_p10:      10th-percentile RUL (pessimistic)
        rul_p90:      90th-percentile RUL (optimistic)
        eol_median:   median EOL cycle
    """
    df = discharge_df.dropna(subset=["soh"]).copy()
    if len(df) < 5:
        return {}
        
    X  = df["cycle_index"].values.reshape(-1, 1)
    y  = df["soh"].values

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=50, length_scale_bounds=(5, 500), nu=1.5)
        + DotProduct(sigma_0=0.0, sigma_0_bounds="fixed")
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        alpha=1e-6,
    )
    gpr.fit(X, y)

    # Predict SOH out to 2x the current max cycle
    max_cycle = int(X.max())
    future_cycles = np.arange(int(X.min()), max_cycle * 2).reshape(-1, 1)
    soh_pred, soh_std = gpr.predict(future_cycles, return_std=True)

    def eol_from_curve(curve):
        below = np.where(curve <= soh_threshold)[0]
        return int(future_cycles[below[0]]) if len(below) else int(future_cycles[-1])

    eol_median = eol_from_curve(soh_pred)
    eol_p10    = eol_from_curve(soh_pred - 1.28 * soh_std)  # pessimistic
    eol_p90    = eol_from_curve(soh_pred + 1.28 * soh_std)  # optimistic

    # --- Group 4 Monte Carlo RUL ---
    n_samples = 500
    # Use the covariance matrix for proper sampling if possible, 
    # but instructed to use np.random.normal(soh_pred, soh_std)
    # This assumes independent noise per future point, which is slightly different 
    # than trajectory sampling but matches the prompt.
    soh_samples = np.random.normal(
        soh_pred.reshape(-1, 1), 
        soh_std.reshape(-1, 1), 
        size=(len(future_cycles), n_samples)
    )
    
    mc_eols = []
    for s in range(n_samples):
        curve = soh_samples[:, s]
        mc_eols.append(eol_from_curve(curve))
    mc_eols = np.array(mc_eols)
    
    eol_mc_p5  = np.percentile(mc_eols, 5)
    eol_mc_p25 = np.percentile(mc_eols, 25)
    eol_mc_p75 = np.percentile(mc_eols, 75)
    eol_mc_p95 = np.percentile(mc_eols, 95)

    current_cycles = df["cycle_index"].values
    rul_median = np.maximum(eol_median - current_cycles, 0)
    rul_p10    = np.maximum(eol_p10    - current_cycles, 0)
    rul_p90    = np.maximum(eol_p90    - current_cycles, 0)
    
    rul_mc_p5  = np.maximum(eol_mc_p5  - current_cycles, 0)
    rul_mc_p25 = np.maximum(eol_mc_p25 - current_cycles, 0)
    rul_mc_p75 = np.maximum(eol_mc_p75 - current_cycles, 0)
    rul_mc_p95 = np.maximum(eol_mc_p95 - current_cycles, 0)

    return {
        "rul_median": rul_median,
        "rul_p10":    rul_p10,
        "rul_p90":    rul_p90,
        "rul_mc_p5":  rul_mc_p5,
        "rul_mc_p25": rul_mc_p25,
        "rul_mc_p75": rul_mc_p75,
        "rul_mc_p95": rul_mc_p95,
        "eol_median": eol_median,
    }


def fit_segmented_gpr_rul(
    discharge_df: pd.DataFrame,
    knee_cycle: int,
    soh_threshold: float = 0.80,
) -> dict:
    """Fit separate GPR models before and after the degradation knee.

    The post-knee segment has fewer points and a steeper slope; a dedicated
    fit with a shorter length-scale captures it far better than a global GPR.
    """
    df = discharge_df.dropna(subset=["soh"]).copy()

    pre = df[df["cycle_index"] < knee_cycle]
    post = df[df["cycle_index"] >= knee_cycle]

    if len(post) < 5:
        # Not enough post-knee data - fall back to global GPR
        return fit_gpr_rul(df, soh_threshold=soh_threshold)

    # Post-knee kernel: shorter length-scale, no DotProduct (already trending)
    post_kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=15, length_scale_bounds=(2, 100), nu=1.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )
    gpr_post = GaussianProcessRegressor(
        kernel=post_kernel, n_restarts_optimizer=5, normalize_y=True, alpha=1e-6
    )
    X_post = post["cycle_index"].values.reshape(-1, 1)
    y_post = post["soh"].values
    gpr_post.fit(X_post, y_post)

    # Project from the knee onwards
    max_cycle = int(df["cycle_index"].max())
    future = np.arange(knee_cycle, max_cycle * 2).reshape(-1, 1)
    soh_p, soh_s = gpr_post.predict(future, return_std=True)

    def _eol(curve: np.ndarray) -> int:
        below = np.where(curve <= soh_threshold)[0]
        return int(future[below[0]]) if len(below) else int(future[-1])

    eol_median = _eol(soh_p)
    eol_p10 = _eol(soh_p - 1.28 * soh_s)
    eol_p90 = _eol(soh_p + 1.28 * soh_s)

    # Monte Carlo
    n_samples = 500
    soh_mc = np.random.normal(soh_p.reshape(-1, 1), soh_s.reshape(-1, 1),
                              (len(future), n_samples))
    mc_eols = [_eol(soh_mc[:, s]) for s in range(n_samples)]
    mc_eols = np.array(mc_eols)

    current_cycles = df["cycle_index"].values
    return {
        "rul_median": np.maximum(eol_median - current_cycles, 0),
        "rul_p10": np.maximum(eol_p10 - current_cycles, 0),
        "rul_p90": np.maximum(eol_p90 - current_cycles, 0),
        "rul_mc_p5": np.maximum(np.percentile(mc_eols, 5) - current_cycles, 0),
        "rul_mc_p25": np.maximum(np.percentile(mc_eols, 25) - current_cycles, 0),
        "rul_mc_p75": np.maximum(np.percentile(mc_eols, 75) - current_cycles, 0),
        "rul_mc_p95": np.maximum(np.percentile(mc_eols, 95) - current_cycles, 0),
        "eol_median": eol_median,
    }


def fit_arrhenius_rul(
    cycle_index: np.ndarray,
    soh: np.ndarray,
    temp_c: np.ndarray,
    soh_threshold: float = 0.8,
) -> np.ndarray:
    """
    Fits Q_loss = A * exp(-Ea / (R * T)) * cycle^0.5
    Returns estimated RUL array.
    """
    R = 8.314
    T_k = np.asarray(temp_c, dtype=float) + 273.15
    cycles = np.asarray(cycle_index, dtype=float)
    y = 1.0 - np.asarray(soh, dtype=float)  # Q_loss
    
    mask = np.isfinite(cycles) & np.isfinite(y) & np.isfinite(T_k)
    if mask.sum() < 5:
        return np.full_like(cycle_index, np.nan, dtype=float)

    def model_func(xdata, A, Ea):
        c, t = xdata
        return A * np.exp(-Ea / (R * t)) * np.sqrt(c)

    try:
        popt, _ = curve_fit(
            model_func,
            (cycles[mask], T_k[mask]),
            y[mask],
            p0=[1e-3, 20000],
            bounds=(0, [1.0, 1e6])
        )
        A_fit, Ea_fit = popt
        
        # Project forward to find EOL
        future = np.arange(1, 2000)
        T_ref = np.mean(T_k[mask])
        q_loss_pred = A_fit * np.exp(-Ea_fit / (R * T_ref)) * np.sqrt(future)
        eol_idx = np.where(q_loss_pred >= (1.0 - soh_threshold))[0]
        eol_cycle = future[eol_idx[0]] if len(eol_idx) else 1000
        
        return np.maximum(eol_cycle - cycles, 0)
    except:
        return np.full_like(cycle_index, np.nan, dtype=float)


def fit_pooled_rul(all_battery_dfs: dict) -> dict:
    """
    all_battery_dfs: {battery_id: discharge cycle_df}
    Returns fitted pipeline + LOCO CV scores.
    """
    frames = []
    for bid, df in all_battery_dfs.items():
        d = df[df["cycle_type"] == "discharge"].copy()
        d["battery_id"] = bid
        frames.append(d)

    if not frames:
        return {}

    pooled = pd.concat(frames).dropna(subset=["soh", "cycle_index"])

    FEATURES = [
        "cycle_index", "temperature_mean_c", "total_resistance_ohm",
        "soh_lag_1", "soh_delta_1", "soh_rolling_var_10",
    ]
    # Filter only existing features
    FEATURES = [f for f in FEATURES if f in pooled.columns]
    
    if len(pooled) < 20 or len(pooled["battery_id"].unique()) < 2:
        return {}

    X = pooled[FEATURES].fillna(0).values
    y = pooled["soh"].values
    groups = pooled["battery_id"].values

    logo = LeaveOneGroupOut()
    model = Ridge(alpha=1.0)
    cv_scores = []
    
    try:
        for train_idx, test_idx in logo.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            cv_scores.append(float(model.score(X[test_idx], y[test_idx])))
    except:
        cv_scores = [0.0]

    # Final fit on all data
    model.fit(X, y)

    return {
        "model":     model,
        "features":  FEATURES,
        "loco_r2":   cv_scores,   # one score per held-out battery
    }


def add_rul_estimates(
    cycle_table: pd.DataFrame,
    soh_threshold: float = 0.8,
    stress_coeffs: dict | None = None,
) -> pd.DataFrame:
    if cycle_table.empty:
        return cycle_table.copy()

    frame = cycle_table.copy()
    frame["estimated_eol_cycle"] = np.nan
    frame["rul_cycles"] = np.nan
    frame["rul_cycles_gpr"] = np.nan
    frame["rul_p10"] = np.nan
    frame["rul_p90"] = np.nan
    frame["rul_mc_p5"] = np.nan
    frame["rul_mc_p25"] = np.nan
    frame["rul_mc_p75"] = np.nan
    frame["rul_mc_p95"] = np.nan
    frame["rul_arrhenius"] = np.nan
    frame["knee_cycle"] = np.nan
    frame["post_knee_degradation_rate"] = np.nan

    discharge = frame[frame["cycle_type"] == "discharge"].copy()
    if discharge.empty or "soh" not in discharge:
        return frame

    for battery_id, group in discharge.groupby("battery_id"):
        eol_cycle = estimate_eol_cycle(group["cycle_index"], group["soh"], soh_threshold=soh_threshold)
        
        # Stress correction
        if "temperature_max_c" in group and "total_resistance_ohm" in group and np.isfinite(eol_cycle):
            T_mean = group["temperature_max_c"].mean()
            R_min = group["total_resistance_ohm"].min()
            R_max = group["total_resistance_ohm"].max()
            
            sigma_temp = np.clip((T_mean - 30.0) / 20.0, 0.0, 0.15)
            sigma_res = 0.0
            if R_min > 0:
                sigma_res = np.clip((R_max - R_min) / R_min * 0.1, 0.0, 0.15)
            
            if stress_coeffs:
                adj_factor = 1 + stress_coeffs["coef_temp"] * sigma_temp + stress_coeffs["coef_res"] * sigma_res + stress_coeffs["intercept"]
                eol_cycle = eol_cycle * np.clip(adj_factor, 0.7, 1.3)
            else:
                eol_cycle = eol_cycle * (1.0 - float(sigma_temp) - float(sigma_res))
            
        mask = (frame["battery_id"] == battery_id) & (frame["cycle_type"] == "discharge")
        frame.loc[mask, "estimated_eol_cycle"] = eol_cycle
        frame.loc[mask, "rul_cycles"] = eol_cycle - frame.loc[mask, "cycle_index"]
        
        # 3. Knee-point Detection
        if len(group) > 20:
            try:
                kl = KneeLocator(group["cycle_index"].values, group["soh"].values, 
                                 curve='concave', direction='decreasing')
                if kl.knee:
                    knee = int(kl.knee)
                    frame.loc[mask, "knee_cycle"] = knee
                    post_knee = group[group["cycle_index"] >= knee]
                    if len(post_knee) > 5:
                        slope, _ = fit_linear_degradation(post_knee["cycle_index"], post_knee["soh"])
                        frame.loc[mask, "post_knee_degradation_rate"] = slope
                        
                        # Update RUL if past knee
                        current_cycles = frame.loc[mask, "cycle_index"]
                        past_knee_mask = (current_cycles >= knee)
                        if past_knee_mask.any():
                            # Intersection of current SOH with post-knee slope
                            # y - y0 = m(x - x0) => 0.8 - soh = slope * (eol - cycle)
                            # eol = cycle + (0.8 - soh) / slope
                            current_soh = group.loc[past_knee_mask, "soh"]
                            if abs(slope) > 1e-10:
                                eol_past = current_cycles[past_knee_mask] + (soh_threshold - current_soh) / slope
                                frame.loc[mask.values & past_knee_mask.values, "rul_cycles"] = np.maximum(eol_past - current_cycles[past_knee_mask], 0)
            except:
                pass

        # GPR RUL
        knee_val = frame.loc[mask, "knee_cycle"].dropna()
        if not knee_val.empty and np.isfinite(knee_val.iloc[0]):
            gpr_results = fit_segmented_gpr_rul(group, int(knee_val.iloc[0]), soh_threshold)
        else:
            gpr_results = fit_gpr_rul(group, soh_threshold=soh_threshold)
        if gpr_results:
            frame.loc[mask, "rul_cycles_gpr"] = gpr_results["rul_median"]
            frame.loc[mask, "rul_p10"] = gpr_results["rul_p10"]
            frame.loc[mask, "rul_p90"] = gpr_results["rul_p90"]
            frame.loc[mask, "rul_mc_p5"] = gpr_results["rul_mc_p5"]
            frame.loc[mask, "rul_mc_p25"] = gpr_results["rul_mc_p25"]
            frame.loc[mask, "rul_mc_p75"] = gpr_results["rul_mc_p75"]
            frame.loc[mask, "rul_mc_p95"] = gpr_results["rul_mc_p95"]

            if "mean_soc_ekf_std" in frame.columns:
                ekf_std_series = frame.loc[mask, "mean_soc_ekf_std"].fillna(0.0)
                ekf_scale = float(1.0 + 2.0 * ekf_std_series.mean())
                frame.loc[mask, "rul_p10"] = gpr_results["rul_p10"] / ekf_scale
                frame.loc[mask, "rul_p90"] = gpr_results["rul_p90"] * ekf_scale
                frame.loc[mask, "rul_mc_p5"] = gpr_results["rul_mc_p5"] / ekf_scale
                frame.loc[mask, "rul_mc_p95"] = gpr_results["rul_mc_p95"] * ekf_scale

        # 4. Arrhenius Model
        if "temperature_mean_c" in group:
            frame.loc[mask, "rul_arrhenius"] = fit_arrhenius_rul(
                group["cycle_index"].values, 
                group["soh"].values, 
                group["temperature_mean_c"].values,
                soh_threshold=soh_threshold
            )

    return frame
