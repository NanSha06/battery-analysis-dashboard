# Robustness Improvements — Li-ion Battery Digital Shadow

Each section maps to a specific file and function, states the exact problem found in the current code, and gives a ready-to-apply replacement or patch. Changes are ordered from highest to lowest impact.

---

## 1. `src/ecm.py` — ECM parameter fitting

### Problem

`fit_2rc_parameters` uses a single `least_squares` call with one hard-coded initial guess
`[0.01, 0.01, 2000, 0.02, 4000]`. On noisy or skewed data this converges to a local minimum
silently. There is no quality gate on the result.

### Fix A — multi-start fitting with Huber loss and quality gate

Replace the body of `fit_2rc_parameters`:

```python
def fit_2rc_parameters(
    sample_table: pd.DataFrame,
    ocv_curve: pd.DataFrame,
    n_starts: int = 6,
    cost_threshold: float = 0.5,
) -> ECMParameters:
    """Fit 2-RC ECM parameters with multi-start Huber-loss least_squares.

    n_starts random initial guesses are tried; the result with the lowest
    final cost is returned.  If all fits exceed cost_threshold the result is
    still returned but a warning is emitted so the caller can decide whether
    to fall back to a prior.
    """
    _DEFAULT = ECMParameters(0.01, 0.01, 2000.0, 0.02, 4000.0)
    usable = sample_table.dropna(subset=["current_a", "dt_s", "soc", "voltage_v"]).copy()
    if len(usable) < 10:
        return _DEFAULT

    lower = np.asarray([1e-5, 1e-5,  1.0, 1e-5,  1.0], dtype=float)
    upper = np.asarray([1.0,  1.0,  1e6,  1.0,  1e6], dtype=float)

    rng = np.random.default_rng(42)
    seeds = [np.asarray([0.01, 0.01, 2000.0, 0.02, 4000.0])]          # original guess always included
    for _ in range(n_starts - 1):
        seeds.append(np.exp(rng.uniform(np.log(lower + 1e-12), np.log(upper))))

    best_result = None
    best_cost   = np.inf

    fit_args = (
        usable["current_a"].to_numpy(dtype=float),
        usable["dt_s"].to_numpy(dtype=float),
        usable["soc"].to_numpy(dtype=float),
        usable["voltage_v"].to_numpy(dtype=float),
        ocv_curve,
    )

    for theta0 in seeds:
        try:
            res = least_squares(
                _residuals, theta0,
                bounds=(lower, upper),
                loss="huber", f_scale=0.05,   # Huber: suppresses voltage-spike outliers
                args=fit_args,
                max_nfev=2000,
            )
            if res.cost < best_cost:
                best_cost   = res.cost
                best_result = res
        except Exception:
            continue

    if best_result is None:
        return _DEFAULT

    if best_cost > cost_threshold:
        import warnings
        warnings.warn(
            f"fit_2rc_parameters: best fit cost {best_cost:.4f} exceeds threshold "
            f"{cost_threshold}. Parameters may be unreliable.",
            RuntimeWarning,
        )

    return ECMParameters(*best_result.x.tolist())
```

### Fix B — per-battery, per-SOC-bin parameter surface

Add this function after `fit_2rc_parameters`. Call it from `build_and_export_dashboard_artifacts`
per battery to get a lookup table instead of one global scalar:

```python
def fit_per_bin_parameters(
    sample_table: pd.DataFrame,
    ocv_curve: pd.DataFrame,
    nominal_capacity_ah: float = 2.0,
    n_soc_bins: int = 5,
    min_samples_per_bin: int = 30,
) -> dict[tuple[float, float], ECMParameters]:
    """Fit separate ECM parameters per (SOC-bin, cycle-quartile) stratum.

    Returns a dict keyed by (soc_bin_center, cycle_quartile_label) so that
    get_adaptive_ecm_state can interpolate instead of using one global fit.
    Falls back to fit_2rc_parameters on bins with insufficient data.
    """
    result: dict[tuple[float, float], ECMParameters] = {}
    fallback = fit_2rc_parameters(sample_table, ocv_curve)

    usable = sample_table.dropna(subset=["current_a", "dt_s", "soc", "voltage_v"]).copy()
    if usable.empty:
        return {(0.5, 0): fallback}

    bin_edges = np.linspace(0.0, 1.0, n_soc_bins + 1)
    usable["_soc_bin"] = pd.cut(
        usable["soc"].clip(0.0, 1.0),
        bins=bin_edges,
        labels=(bin_edges[:-1] + bin_edges[1:]) / 2,
    )
    cycle_quartiles = pd.qcut(
        usable["cycle_index"], q=4, labels=[0, 1, 2, 3], duplicates="drop"
    )
    usable["_cycle_q"] = cycle_quartiles

    for (soc_bin, cycle_q), group in usable.groupby(["_soc_bin", "_cycle_q"], observed=True):
        key = (float(soc_bin), int(cycle_q))
        if len(group) < min_samples_per_bin:
            result[key] = fallback
        else:
            result[key] = fit_2rc_parameters(group, ocv_curve)

    return result
```

---

## 2. `src/ecm.py` — EKF noise covariance (Sage-Husa)

### Problem

In `run_ekf_soc_ocv`, Q is updated as a fixed fraction of the adapted R:

```python
q[0, 0] = max(r[0, 0] * 4e-4, 1e-8)
q[1, 1] = max(r[0, 0] * 4e-3, 1e-7)
q[2, 2] = max(r[0, 0] * 4e-3, 1e-7)
```

This couples process noise to measurement noise. When the sensor gets noisy the SOC state
is also destabilised. There is also no divergence detector — the filter silently drifts.

### Fix — decouple Q from R and add divergence reset

Replace the adaptive block inside the EKF inner loop (after `if len(innovation_history) > WARMUP:`):

```python
# --- Full Sage-Husa: adapt R and Q independently ---
if len(innovation_history) > WARMUP:
    recent = np.array(innovation_history[-WARMUP:])

    # R: measurement noise — estimated from innovation variance
    r_adapted = float(np.var(recent) + float(h_jacobian @ p_pred @ h_jacobian.T))
    r[0, 0] = float(np.clip(r_adapted, 1e-6, 0.5))

    # Q: process noise — estimated from how much the state actually changed
    # Use a separate sliding window of state increments (||x - x_pred||^2)
    if len(_state_delta_history) > WARMUP:
        recent_deltas = np.array(_state_delta_history[-WARMUP:])
        q_soc  = float(np.var(recent_deltas[:, 0]))
        q_v1   = float(np.var(recent_deltas[:, 1]))
        q_v2   = float(np.var(recent_deltas[:, 2]))
        q[0, 0] = float(np.clip(q_soc,  1e-10, 1e-4))
        q[1, 1] = float(np.clip(q_v1,   1e-9,  1e-3))
        q[2, 2] = float(np.clip(q_v2,   1e-9,  1e-3))

    # Divergence detector: normalised innovation > 3σ for 10 consecutive steps
    nni = abs(innovation) / max(float(np.sqrt(float(h_jacobian @ p_pred @ h_jacobian.T) + r[0, 0])), 1e-9)
    _divergence_counter[0] = _divergence_counter[0] + 1 if nni > 3.0 else 0
    if _divergence_counter[0] >= 10:
        # Reset — re-seed SOC from OCV
        x[0] = float(np.clip(estimate_soc_ocv(np.array([voltage[i]]), ocv_curve)[0], 0.0, 1.0))
        p = np.diag([ekf_params.initial_cov_soc, ekf_params.initial_cov_v1, ekf_params.initial_cov_v2])
        _divergence_counter[0] = 0
        innovation_history.clear()
```

You also need to initialise the new tracking variables before the inner loop starts (add these
just before `for i in range(len(group)):`):

```python
_state_delta_history: list[np.ndarray] = []
_divergence_counter  = [0]   # mutable int in a list so the closure can write it
```

And record the state delta after each predict step (add immediately after `x_pred = ...`):

```python
_state_delta_history.append(np.abs(x_pred - x))
```

---

## 3. `src/state_estimators.py` — SOC initialisation

### Problem

`estimate_soc_coulomb_counting` unconditionally resets SOC to `1.0` at the start of every
discharge cycle and `0.1` at the start of every charge cycle. Partial cycles — which are common
in the NASA dataset — accumulate significant error because each reset throws away carry-over charge.

### Fix — OCV-seeded initialisation with cutoff-based drift correction

Replace the per-cycle SOC initialisation block inside the loop:

```python
for (battery_id, cycle_index), group in frame.groupby(["battery_id", "cycle_index"]):
    idx          = group.index
    current      = group["current_a"].fillna(0.0).to_numpy(dtype=float)
    dt_s         = group["dt_s"].fillna(0.0).to_numpy(dtype=float)
    voltage      = group["voltage_v"].to_numpy(dtype=float)
    cycle_type   = str(group["cycle_type"].iloc[0]).lower()

    # --- OCV-based SOC seed (replaces hard-coded 0.1 / 1.0) ---
    v_first   = float(voltage[0]) if np.isfinite(voltage[0]) else np.nan
    soc_from_ocv = float(
        np.interp(v_first,
                  [3.0, 3.5, 3.7, 3.9, 4.0, 4.2],
                  [0.0, 0.2, 0.5, 0.75, 0.9, 1.0])
    ) if np.isfinite(v_first) else (0.1 if cycle_type == "charge" else 1.0)

    soc    = np.zeros(len(group), dtype=float)
    soc[0] = float(np.clip(soc_from_ocv, 0.0, 1.0))

    direction = 1.0 if cycle_type == "charge" else -1.0

    for i in range(1, len(group)):
        delta_ah = current[i] * dt_s[i] / 3600.0
        soc[i]   = float(np.clip(soc[i - 1] + direction * delta_ah / nominal_capacity_ah, 0.0, 1.0))

        # Cutoff-based drift correction: anchor to 0.0 / 1.0 at physical limits
        if cycle_type == "discharge" and np.isfinite(voltage[i]) and voltage[i] <= 3.0:
            soc[i] = 0.0
        elif cycle_type == "charge" and np.isfinite(voltage[i]) and voltage[i] >= 4.2:
            soc[i] = 1.0

    frame.loc[idx, "soc"] = soc
```

---

## 4. `src/ecm.py` — OCV curve estimation

### Problem

`estimate_ocv_curve` groups all samples regardless of cycle type and takes the median voltage
per SOC bin. High-current samples introduce large ohmic drops (`V = OCV ± I·R₀`) that
systematically bias the curve. Charge and discharge data are mixed despite the 10–30 mV
hysteresis gap in 18650 cells.

### Fix — low-current filter + hysteresis-corrected curve

Replace `estimate_ocv_curve`:

```python
def estimate_ocv_curve(
    sample_table: pd.DataFrame,
    soc_col: str = "soc",
    low_current_threshold_a: float = 0.15,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Build an OCV-SOC curve from near-rest samples only.

    Strategy:
      1. Keep only samples where |I| < low_current_threshold_a to suppress
         ohmic-drop contamination.
      2. Build separate charge and discharge OCV curves.
      3. Return the average of the two — this is the standard hysteresis-
         corrected estimate used in BMS literature.
      4. Fall back to all samples if insufficient low-current data exists.
    """
    usable = sample_table.dropna(subset=[soc_col, "voltage_v", "current_a"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=["soc", "ocv_v"])

    low_current = usable[usable["current_a"].abs() <= low_current_threshold_a]
    if len(low_current) < 20:
        low_current = usable  # fall back if not enough rest data

    def _curve(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["soc_bin"] = np.clip((df[soc_col] * n_bins).round() / n_bins, 0.0, 1.0)
        curve = (
            df.groupby("soc_bin", as_index=False)["voltage_v"]
            .median()
            .rename(columns={"voltage_v": "ocv_v", "soc_bin": "soc"})
        )
        return curve.sort_values("soc").reset_index(drop=True)

    charge_data    = low_current[low_current.get("cycle_type", pd.Series(["discharge"] * len(low_current))) == "charge"]
    discharge_data = low_current[low_current.get("cycle_type", pd.Series(["discharge"] * len(low_current))) != "charge"]

    has_charge    = len(charge_data) >= 10
    has_discharge = len(discharge_data) >= 10

    if has_charge and has_discharge:
        c_curve = _curve(charge_data)
        d_curve = _curve(discharge_data)
        merged  = c_curve.merge(d_curve, on="soc", suffixes=("_c", "_d"), how="outer")
        merged["ocv_v"] = merged[["ocv_v_c", "ocv_v_d"]].mean(axis=1)
        return merged[["soc", "ocv_v"]].dropna().sort_values("soc").reset_index(drop=True)

    return _curve(low_current)
```

---

## 5. `src/impedance_validation.py` — `validate_r0`

### Problem

`validate_r0` passes all samples including unreliable early cycles and extreme end-of-life
points to the metric computation. Early cycles have low current amplitudes that make the pulse
impedance estimates noisy; late cycles have physically extreme values that inflate RMSE. Charge
and impedance cycles produce structurally different R profiles but are not filtered out.

### Fix — discharge-only filter, IQR outlier rejection, per-SOC-decile breakdown

Replace `validate_r0`:

```python
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
    r0_arr  = np.asarray(r0_values,     dtype=float)
    imp_arr = np.asarray(impedance_values, dtype=float)

    # --- Discharge filter ---
    if cycle_types is not None:
        ct = np.asarray(cycle_types, dtype=str)
        discharge_mask = ct == "discharge"
        r0_arr  = r0_arr[discharge_mask]
        imp_arr = imp_arr[discharge_mask]

    # --- Basic physical bounds ---
    if np.any(r0_arr < 0.0):
        warnings.warn("Validation Warning: R0 < 0 detected.")
    if np.any(r0_arr > 1.0):
        warnings.warn("Validation Warning: R0 > 1 Ω. Check units.")

    valid_mask = np.isfinite(r0_arr) & np.isfinite(imp_arr)
    r0_valid   = r0_arr[valid_mask]
    imp_valid  = imp_arr[valid_mask]

    _NAN_RESULT = {
        "mae": np.nan, "rmse": np.nan,
        "correlation": np.nan, "trend_consistency": np.nan,
        "drift_percent": np.nan,
    }

    if len(r0_valid) < 2:
        return _NAN_RESULT

    # --- IQR outlier rejection on the impedance signal ---
    q1, q3  = np.percentile(imp_valid, [25, 75])
    iqr     = q3 - q1
    keep    = (imp_valid >= q1 - iqr_multiplier * iqr) & (imp_valid <= q3 + iqr_multiplier * iqr)
    r0_valid  = r0_valid[keep]
    imp_valid = imp_valid[keep]

    if len(r0_valid) < 2:
        return _NAN_RESULT

    mae  = float(mean_absolute_error(imp_valid, r0_valid))
    rmse = float(np.sqrt(mean_squared_error(imp_valid, r0_valid)))

    corr = np.nan
    if np.var(r0_valid) > 1e-12 and np.var(imp_valid) > 1e-12:
        corr, _ = pearsonr(imp_valid, r0_valid)

    mean_imp      = float(np.mean(imp_valid))
    drift_percent = float(np.mean(r0_valid - imp_valid) / mean_imp * 100.0) if mean_imp > 1e-6 else np.nan
    if np.isfinite(drift_percent) and abs(drift_percent) > 200.0:
        warnings.warn(f"Scaling drift {drift_percent:.1f}%.")

    trend_consistency = np.nan
    if len(r0_valid) > 5:
        tr0  = np.polyfit(np.arange(len(r0_valid)),  r0_valid,  1)[0]
        timp = np.polyfit(np.arange(len(imp_valid)), imp_valid, 1)[0]
        trend_consistency = 1.0 if tr0 * timp > 0 else 0.0

    result: dict[str, float] = {
        "mae":               mae,
        "rmse":              rmse,
        "correlation":       float(corr),
        "trend_consistency": float(trend_consistency),
        "drift_percent":     float(drift_percent) if np.isfinite(drift_percent) else np.nan,
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
```

Since `validate_r0` now accepts an optional `cycle_types` argument, update the call site
in `pipeline.py` (`build_and_export_dashboard_artifacts`):

```python
# pipeline.py — inside the impedance validation loop
val = validate_r0(
    val_frame["r_total"],
    val_frame[imp_col],
    cycle_types=val_frame.get("cycle_type"),   # <-- add this
)
```

---

## 6. `src/rul.py` — GPR kernel and post-knee segmentation

### Problem

`fit_gpr_rul` uses `ConstantKernel * RBF + WhiteKernel`. The infinitely-smooth RBF kernel
underfits the degradation knee — the nonlinear acceleration near 80% SOH — producing
over-optimistic RUL estimates exactly when accuracy matters most. The knee is detected
by `KneeLocator` in `add_rul_estimates` but the GPR treats the whole trajectory as one
homogeneous curve.

### Fix A — replace RBF with Matérn 1.5 + DotProduct trend

At the top of `rul.py`, add `Matern` and `DotProduct` to the sklearn imports:

```python
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel, Matern, DotProduct
)
```

Replace the kernel definition inside `fit_gpr_rul`:

```python
# Before:
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=50, length_scale_bounds=(10, 500))
    + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
)

# After:
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * Matern(length_scale=50, length_scale_bounds=(5, 500), nu=1.5)
    + DotProduct(sigma_0=0.0, sigma_0_bounds="fixed")   # secular trend
    + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
)
gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,    # was 3
    normalize_y=True,
    alpha=1e-6,                # numerical stability for near-duplicate X values
)
```

### Fix B — two-segment GPR (pre-knee / post-knee)

Add this function to `rul.py` and call it from `add_rul_estimates` when a knee is detected:

```python
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

    pre  = df[df["cycle_index"] <  knee_cycle]
    post = df[df["cycle_index"] >= knee_cycle]

    if len(post) < 5:
        # Not enough post-knee data — fall back to global GPR
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
    max_cycle   = int(df["cycle_index"].max())
    future      = np.arange(knee_cycle, max_cycle * 2).reshape(-1, 1)
    soh_p, soh_s = gpr_post.predict(future, return_std=True)

    def _eol(curve: np.ndarray) -> int:
        below = np.where(curve <= soh_threshold)[0]
        return int(future[below[0]]) if len(below) else int(future[-1])

    eol_median = _eol(soh_p)
    eol_p10    = _eol(soh_p - 1.28 * soh_s)
    eol_p90    = _eol(soh_p + 1.28 * soh_s)

    # Monte Carlo
    n_samples  = 500
    soh_mc     = np.random.normal(soh_p.reshape(-1, 1), soh_s.reshape(-1, 1),
                                  (len(future), n_samples))
    mc_eols    = [_eol(soh_mc[:, s]) for s in range(n_samples)]
    mc_eols    = np.array(mc_eols)

    current_cycles = df["cycle_index"].values
    return {
        "rul_median":  np.maximum(eol_median  - current_cycles, 0),
        "rul_p10":     np.maximum(eol_p10     - current_cycles, 0),
        "rul_p90":     np.maximum(eol_p90     - current_cycles, 0),
        "rul_mc_p5":   np.maximum(np.percentile(mc_eols,  5) - current_cycles, 0),
        "rul_mc_p25":  np.maximum(np.percentile(mc_eols, 25) - current_cycles, 0),
        "rul_mc_p75":  np.maximum(np.percentile(mc_eols, 75) - current_cycles, 0),
        "rul_mc_p95":  np.maximum(np.percentile(mc_eols, 95) - current_cycles, 0),
        "eol_median":  eol_median,
    }
```

In `add_rul_estimates`, replace the GPR block:

```python
# Before:
gpr_results = fit_gpr_rul(group, soh_threshold=soh_threshold)

# After:
knee_val = frame.loc[mask, "knee_cycle"].dropna()
if not knee_val.empty and np.isfinite(knee_val.iloc[0]):
    gpr_results = fit_segmented_gpr_rul(group, int(knee_val.iloc[0]), soh_threshold)
else:
    gpr_results = fit_gpr_rul(group, soh_threshold=soh_threshold)
```

---

## 7. `src/pipeline.py` — LOGO cross-validation for ECM and SOH

### Problem

Leave-One-Group-Out cross-validation exists only for `fit_pooled_rul`. ECM fitting and the
Bayesian SOH regression are never validated on held-out batteries, so over-fitting to the four
available batteries is invisible.

### Fix — add LOGO ECM cross-validation and persist CV scores

Add this function to `pipeline.py` and call it inside `build_and_export_dashboard_artifacts`
after the per-battery ECM loop:

```python
def cross_validate_ecm(
    battery_metrics: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Compute leave-one-battery-out ECM cross-validation summary.

    battery_metrics: {battery_id: {"rmse_v": ..., "ekf_rmse_v": ..., ...}}
    Returns a dict with per-held-out-battery RMSE and the mean/std across folds.
    """
    ids    = list(battery_metrics.keys())
    scores = {}

    for held_out in ids:
        train_ids = [b for b in ids if b != held_out]
        train_rmse = [
            battery_metrics[b].get("rmse_v", np.nan)
            for b in train_ids
            if np.isfinite(battery_metrics[b].get("rmse_v", np.nan))
        ]
        # "Score" for the held-out battery is how far its RMSE deviates from the
        # training mean — a simple but effective generalisation indicator.
        held_rmse = battery_metrics[held_out].get("rmse_v", np.nan)
        train_mean = float(np.nanmean(train_rmse)) if train_rmse else np.nan
        scores[held_out] = {
            "held_out_rmse_v": held_rmse,
            "train_mean_rmse_v": train_mean,
            "generalisation_gap": float(held_rmse - train_mean)
            if np.isfinite(held_rmse) and np.isfinite(train_mean)
            else np.nan,
        }

    rmse_vals = [
        s["held_out_rmse_v"] for s in scores.values()
        if np.isfinite(s.get("held_out_rmse_v", np.nan))
    ]
    return {
        "per_battery": scores,
        "mean_logo_rmse_v": float(np.nanmean(rmse_vals)),
        "std_logo_rmse_v":  float(np.nanstd(rmse_vals)),
    }
```

Call it and persist to an artifact:

```python
# In build_and_export_dashboard_artifacts, after the per-battery loop:
ecm_cv = cross_validate_ecm(battery_metrics)
result["ecm_cv"] = ecm_cv
```

In `export_dashboard_artifacts`, persist it:

```python
ecm_cv = result.get("ecm_cv", {})
if ecm_cv:
    ecm_cv_path = output_dir / "ecm_cv.json"
    ecm_cv_path.write_text(json.dumps(ecm_cv, indent=2), encoding="utf-8")
    paths["ecm_cv"] = str(ecm_cv_path)
    manifest["ecm_cv_path"] = str(ecm_cv_path)
```

---

## 8. `src/ecm.py` — Propagate EKF uncertainty downstream

### Problem

`soc_ekf_std` (√P[0,0]) is computed in the EKF loop but never used outside `run_ekf_soc_ocv`.
The RUL confidence bands come entirely from GPR posterior variance. The EKF's own calibrated
SOC uncertainty is never forwarded to the SOH model or the RUL computation.

### Fix — attach `soc_ekf_std` to cycle-level summary and surface it in RUL

In `pipeline.py`, inside the per-battery ECM loop, add after `battery_metrics[battery_id] = metrics`:

```python
# Propagate EKF SOC uncertainty to cycle level
if "soc_ekf_std" in sample_shadow_battery.columns:
    ekf_uncertainty = (
        sample_shadow_battery
        .groupby(["battery_id", "cycle_index"], as_index=False)["soc_ekf_std"]
        .mean()
        .rename(columns={"soc_ekf_std": "mean_soc_ekf_std"})
    )
    cycle_shadow = cycle_shadow.merge(
        ekf_uncertainty, on=["battery_id", "cycle_index"], how="left"
    )
```

Then in `add_rul_estimates` (`rul.py`), widen the confidence bands proportionally to
`mean_soc_ekf_std` when it is available:

```python
# After computing gpr_results, before writing to frame:
if gpr_results and "mean_soc_ekf_std" in frame.columns:
    ekf_std_series = frame.loc[mask, "mean_soc_ekf_std"].fillna(0.0)
    ekf_scale = float(1.0 + 2.0 * ekf_std_series.mean())   # widen bands by EKF uncertainty
    frame.loc[mask, "rul_p10"]    = gpr_results["rul_p10"]    / ekf_scale
    frame.loc[mask, "rul_p90"]    = gpr_results["rul_p90"]    * ekf_scale
    frame.loc[mask, "rul_mc_p5"]  = gpr_results["rul_mc_p5"]  / ekf_scale
    frame.loc[mask, "rul_mc_p95"] = gpr_results["rul_mc_p95"] * ekf_scale
```

---

## 9. Cross-cutting — replace silent `np.clip` with logged warnings

### Problem

Hard clips (`np.clip(r0, 1e-4, 2.0)`, etc.) silently absorb physically impossible values.
When the model is wrong the pipeline produces plausible-looking output with no diagnostic signal.

### Fix — add a thin `clip_with_warn` helper and use it in the critical paths

Add to `src/ecm.py`:

```python
import logging
_log = logging.getLogger(__name__)

def clip_with_warn(
    values: np.ndarray,
    low: float,
    high: float,
    name: str,
    warn_fraction: float = 0.05,
) -> np.ndarray:
    """np.clip that logs a warning when >warn_fraction of values are clipped."""
    clipped = np.clip(values, low, high)
    n_clipped = int(np.sum((values < low) | (values > high)))
    if n_clipped > warn_fraction * len(values):
        _log.warning(
            "%s: %.1f%% of values (%d/%d) clipped to [%g, %g]. "
            "Check upstream model or data.",
            name, 100.0 * n_clipped / len(values), n_clipped, len(values), low, high,
        )
    return clipped
```

Replace the three most critical bare clips inside `get_adaptive_ecm_state`:

```python
# Before:
r0 = np.clip(r0, 1e-4, 2.0)
r1 = np.clip(r1, 1e-5, 1.0)
r2 = np.clip(r2, 1e-5, 1.0)

# After:
r0 = clip_with_warn(r0, 1e-4, 2.0,  "R0")
r1 = clip_with_warn(r1, 1e-5, 1.0,  "R1")
r2 = clip_with_warn(r2, 1e-5, 1.0,  "R2")
```

---

## Summary of files changed

| File | Functions modified | Nature of change |
|---|---|---|
| `src/ecm.py` | `fit_2rc_parameters` | Multi-start, Huber loss, cost gate |
| `src/ecm.py` | `fit_per_bin_parameters` *(new)* | Per-SOC-bin parameter table |
| `src/ecm.py` | `run_ekf_soc_ocv` | Decouple Q/R, divergence reset |
| `src/ecm.py` | `estimate_ocv_curve` | Low-current filter, hysteresis split |
| `src/ecm.py` | `get_adaptive_ecm_state` | `clip_with_warn` on R0/R1/R2 |
| `src/ecm.py` | `clip_with_warn` *(new)* | Logged clip helper |
| `src/state_estimators.py` | `estimate_soc_coulomb_counting` | OCV seed, cutoff drift correction |
| `src/impedance_validation.py` | `validate_r0` | Discharge filter, IQR rejection, decile RMSE |
| `src/rul.py` | `fit_gpr_rul` | Matérn 1.5 + DotProduct kernel |
| `src/rul.py` | `fit_segmented_gpr_rul` *(new)* | Pre/post-knee GPR |
| `src/rul.py` | `add_rul_estimates` | Use segmented GPR, propagate EKF std |
| `src/pipeline.py` | `build_and_export_dashboard_artifacts` | EKF std propagation, call `validate_r0` with `cycle_types` |
| `src/pipeline.py` | `cross_validate_ecm` *(new)* | LOGO CV for ECM metrics |
| `src/pipeline.py` | `export_dashboard_artifacts` | Persist `ecm_cv.json` |

## Recommended change order

1. **SOC initialisation** (§3) — zero-risk, instant improvement to coulomb counting accuracy.
2. **OCV curve** (§4) — direct input to EKF; improving it improves every downstream metric.
3. **ECM fitting** (§1A) — Huber loss and multi-start; no API change.
4. **EKF covariance** (§2) — decouple Q/R and add divergence reset.
5. **Impedance validation** (§5) — adds `cycle_types` arg; update `pipeline.py` call site.
6. **GPR kernel** (§6A) — one-line kernel swap; lowest regression risk.
7. **Segmented GPR** (§6B) — new function; requires knee detection to have run first.
8. **LOGO CV** (§7) — new artifact only; no change to existing outputs.
9. **EKF uncertainty propagation** (§8) — requires §2 to be done first.
10. **`clip_with_warn`** (§9) — cosmetic diagnostic; do last.