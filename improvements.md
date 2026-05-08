# Data-Driven Improvements — Li-ion Battery Digital Shadow

A precise, ordered implementation plan to make the project more data-driven. Each change identifies the exact file, the constant or logic being replaced, and the code/approach to use.

---

## 1. Learn SOH Fusion Weights from Data

**File:** `src/features.py`  
**Current behaviour:** `w1 = 0.8`, `w2 = 0.2` are hardcoded in the SOH fusion formula.

**Change:** Replace with a data-optimised weight found via cross-validated grid search.

```python
# features.py — replace hardcoded weights with optimised values
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np

def optimise_soh_weights(cycle_df: pd.DataFrame) -> tuple[float, float]:
    """
    Find (w1, w2) that maximise correlation between fused SOH and
    measured capacity ratio on discharge cycles.
    Returns (w1, w2) with w1 + w2 = 1.
    """
    discharge = cycle_df[cycle_df["cycle_type"] == "discharge"].dropna(
        subset=["capacity_ah", "total_resistance_ohm"]
    ).copy()

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

    return best_w1, round(1.0 - best_w1, 2)


def compute_soh(cycle_df: pd.DataFrame) -> pd.Series:
    w1, w2 = optimise_soh_weights(cycle_df)
    # ... rest of existing formula using w1, w2
```

**Persist** the discovered weights alongside `scaling_metrics.json` so they are reproducible across runs.

---

## 2. Adaptive EKF Noise Covariances

**File:** `src/ecm.py` → `run_ekf_soc_ocv()`  
**Current behaviour:** `process_var_soc=1e-6`, `process_var_v1=1e-5`, `process_var_v2=1e-5`, `measurement_var_v=2.5e-3` are fixed tuning parameters.

**Change:** Implement Sage-Husa innovation-based covariance adaptation. After the first `warmup_cycles` cycles of data, update `R` (measurement noise) from the rolling innovation variance and scale `Q` proportionally.

```python
# ecm.py — inside run_ekf_soc_ocv(), after the main EKF loop initialisation

WARMUP = 50          # samples before adaptation begins
ADAPT_LR = 0.02      # learning rate for covariance update

innovation_history = []

# --- Inside the per-timestep loop, after computing y_k ---
innovation_history.append(float(y_k))

if len(innovation_history) > WARMUP:
    recent = np.array(innovation_history[-WARMUP:])
    R_adapted = float(np.var(recent) + (H @ P_pred @ H.T))
    R = max(R_adapted, 1e-6)               # floor to prevent collapse
    # Scale Q proportionally so signal-to-noise ratio is preserved
    Q[0, 0] = max(R * 4e-4, 1e-8)         # SOC process noise
    Q[1, 1] = max(R * 4e-3, 1e-7)         # V_RC1 process noise
    Q[2, 2] = max(R * 4e-3, 1e-7)         # V_RC2 process noise
```

**Persist** the final `R` and diagonal of `Q` per battery into the exported JSON artifacts for audit.

---

## 3. Data-Fitted Stress Correction Coefficients for RUL

**File:** `src/rul.py`  
**Current behaviour:** Stress penalty coefficients (`/20`, `* 0.1`, `0.15` clips) are hand-tuned constants.

**Change:** Fit a small linear regression that maps `[sigma_temp, sigma_res]` → observed RUL error (difference between linear-extrapolated RUL and actual remaining cycles). This requires at least two batteries to train on and one to validate.

```python
# rul.py — new function

from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
import numpy as np

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
        T_mean = discharge["temperature_mean_c"].mean()
        R_min  = discharge["total_resistance_ohm"].min()
        R_max  = discharge["total_resistance_ohm"].max()

        sigma_temp = np.clip((T_mean - 30) / 20, 0, 0.15)
        sigma_res  = np.clip((R_max - R_min) / R_min * 0.1, 0, 0.15)

        # Ground truth: actual EOL cycle from data
        eol_actual = discharge[discharge["soh"] <= 0.80]["cycle_index"].min()
        # Linear model EOL (existing fit_linear_degradation output)
        eol_linear = fit_linear_degradation(discharge)["eol_cycle"]

        X_rows.append([sigma_temp, sigma_res])
        y_rows.append((eol_actual - eol_linear) / max(eol_linear, 1))

    X, y = np.array(X_rows), np.array(y_rows)
    model = Ridge(alpha=0.1).fit(X, y)

    return {
        "coef_temp": float(model.coef_[0]),
        "coef_res":  float(model.coef_[1]),
        "intercept": float(model.intercept_),
    }
```

Replace the hardcoded `(1 - sigma_temp - sigma_res)` multiplier with:

```python
adj_factor = 1 + coef_temp * sigma_temp + coef_res * sigma_res + intercept
n_eol_adj  = n_eol * np.clip(adj_factor, 0.7, 1.3)
```

Persist coefficients to `stress_coefficients.json` via `pipeline.py`.

---

## 4. Upgrade RUL to Gaussian Process Regression

**File:** `src/rul.py`  
**Current behaviour:** First-order polynomial fit with single point EOL estimate.

**Change:** Add a GPR-based RUL estimator that runs alongside the linear model and outputs a median + 80% prediction interval. Use the linear model as the GPR mean prior.

```python
# rul.py — add alongside existing fit_linear_degradation()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def fit_gpr_rul(discharge_df: pd.DataFrame, soh_threshold: float = 0.80) -> dict:
    """
    Returns:
        rul_median:   array of per-cycle median RUL
        rul_p10:      10th-percentile RUL (pessimistic)
        rul_p90:      90th-percentile RUL (optimistic)
        eol_median:   median EOL cycle
    """
    df = discharge_df.dropna(subset=["soh"]).copy()
    X  = df["cycle_index"].values.reshape(-1, 1)
    y  = df["soh"].values

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=50, length_scale_bounds=(10, 500))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                   normalize_y=True)
    gpr.fit(X, y)

    # Predict SOH out to 2x the current max cycle
    future_cycles = np.arange(X.min(), X.max() * 2).reshape(-1, 1)
    soh_pred, soh_std = gpr.predict(future_cycles, return_std=True)

    def eol_from_curve(curve):
        below = np.where(curve <= soh_threshold)[0]
        return int(future_cycles[below[0]]) if len(below) else int(future_cycles[-1])

    eol_median = eol_from_curve(soh_pred)
    eol_p10    = eol_from_curve(soh_pred - 1.28 * soh_std)  # pessimistic
    eol_p90    = eol_from_curve(soh_pred + 1.28 * soh_std)  # optimistic

    current_cycles = df["cycle_index"].values
    rul_median = np.maximum(eol_median - current_cycles, 0)
    rul_p10    = np.maximum(eol_p10    - current_cycles, 0)
    rul_p90    = np.maximum(eol_p90    - current_cycles, 0)

    return {
        "rul_median": rul_median,
        "rul_p10":    rul_p10,
        "rul_p90":    rul_p90,
        "eol_median": eol_median,
        "eol_p10":    eol_p10,
        "eol_p90":    eol_p90,
    }
```

**Wire into pipeline:** call `fit_gpr_rul()` in `pipeline.py` and export `rul_gpr_{battery_id}.parquet`. Update `app.py` RUL chart to show the confidence band using `go.Scatter` with `fill='tonexty'`.

---

## 5. Within-Cycle Shape Features for SOH Regression

**File:** `src/features.py` → `add_cycle_features()`  
**Current behaviour:** 8 cycle-level aggregate features fed to Ridge regression.

**Change:** Add 5 new features derived from within-cycle time-series in `sample_table`.

```python
# features.py — extend add_cycle_features() to accept sample_table

def add_cycle_shape_features(cycle_df: pd.DataFrame,
                              sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-cycle shape features computed from raw sample vectors.
    Merge result onto cycle_df on 'cycle_index'.
    """
    records = []
    for cycle_id, grp in sample_df.groupby("cycle_index"):
        grp = grp.sort_values("time_s")
        V = grp["voltage_v"].values
        I = grp["current_a"].values
        Q_cumulative = np.cumsum(np.abs(I) * np.diff(grp["time_s"].values,
                                                      prepend=0)) / 3600

        # 1. dV/dQ — incremental capacity; use median of non-zero-Q region
        dq = np.diff(Q_cumulative)
        dv = np.diff(V)
        with np.errstate(divide="ignore", invalid="ignore"):
            dvdq = np.where(np.abs(dq) > 1e-6, dv / dq, np.nan)
        ic_median = float(np.nanmedian(dvdq))

        # 2. Voltage discharge slope (linear fit slope over discharge period)
        discharge_mask = I < -0.05
        v_slope = (
            float(np.polyfit(np.where(discharge_mask)[0], V[discharge_mask], 1)[0])
            if discharge_mask.sum() > 5 else np.nan
        )

        # 3. Time to reach 80% capacity cutoff (fraction of cycle duration)
        total_Q = Q_cumulative[-1]
        if total_Q > 0:
            idx_80 = np.searchsorted(Q_cumulative, 0.8 * total_Q)
            t80_frac = float(grp["time_s"].iloc[min(idx_80, len(grp)-1)]
                             / grp["time_s"].iloc[-1])
        else:
            t80_frac = np.nan

        # 4. Charge–discharge duration asymmetry ratio
        charge_time    = grp.loc[I > 0.05, "time_s"].count()
        discharge_time = grp.loc[I < -0.05, "time_s"].count()
        asym_ratio = (
            charge_time / discharge_time
            if discharge_time > 0 else np.nan
        )

        # 5. Rolling variance of voltage (within cycle) — captures noise growth
        v_rolling_var = float(pd.Series(V).rolling(10).var().mean())

        records.append({
            "cycle_index":    cycle_id,
            "ic_median":      ic_median,
            "v_discharge_slope": v_slope,
            "t80_frac":       t80_frac,
            "charge_discharge_asym": asym_ratio,
            "voltage_rolling_var":   v_rolling_var,
        })

    shape_df = pd.DataFrame(records)
    return cycle_df.merge(shape_df, on="cycle_index", how="left")
```

Add these 5 columns to the `features` list in the Ridge regression in `state_estimators.py`.

---

## 6. Lag Features on the Cycle Table

**File:** `src/features.py`  
**Current behaviour:** No temporal lag features; model treats each cycle independently.

**Change:** Add a `add_lag_features()` function called after `add_cycle_features()`.

```python
# features.py — call after add_cycle_features()

def add_lag_features(cycle_df: pd.DataFrame,
                     lags: list[int] = [1, 3, 5, 10]) -> pd.DataFrame:
    """
    Appends lag and rolling features for SOH and Coulombic efficiency.
    Only computed on discharge cycles; NaN-filled for others.
    """
    df = cycle_df.copy().sort_values("cycle_index")
    discharge_mask = df["cycle_type"] == "discharge"

    for lag in lags:
        df.loc[discharge_mask, f"soh_lag_{lag}"] = (
            df.loc[discharge_mask, "soh"].shift(lag)
        )
        df.loc[discharge_mask, f"ce_lag_{lag}"] = (
            df.loc[discharge_mask, "coulombic_efficiency"].shift(lag)
        )

    # Rolling variance of SOH (momentum signal)
    df.loc[discharge_mask, "soh_rolling_var_10"] = (
        df.loc[discharge_mask, "soh"]
          .rolling(window=10, min_periods=3)
          .var()
          .values
    )

    # Cycle-over-cycle SOH delta
    df.loc[discharge_mask, "soh_delta_1"] = (
        df.loc[discharge_mask, "soh"].diff(1)
    )

    return df
```

Add the non-NaN lag columns to the Ridge feature list. Use `dropna()` before fitting to avoid training on early-cycle NaN rows.

---

## 7. Cross-Battery Pooled Degradation Model

**File:** `src/rul.py` and `src/pipeline.py`  
**Current behaviour:** Each battery's RUL is estimated independently.

**Change:** Train a pooled Ridge regression on all 4 batteries with a `battery_id` dummy variable, then use leave-one-battery-out CV to evaluate generalisation.

```python
# rul.py — new function

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd, numpy as np

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

    pooled = pd.concat(frames).dropna(subset=["soh", "cycle_index"])

    FEATURES = [
        "cycle_index", "temperature_mean_c", "total_resistance_ohm",
        "soh_lag_1", "soh_delta_1", "soh_rolling_var_10",
        "coulombic_efficiency",
    ]
    X = pooled[FEATURES].fillna(0).values
    y = pooled["soh"].values
    groups = pooled["battery_id"].values

    logo = LeaveOneGroupOut()
    model = Ridge(alpha=1.0)
    cv_scores = []
    for train_idx, test_idx in logo.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        cv_scores.append(model.score(X[test_idx], y[test_idx]))

    # Final fit on all data
    model.fit(X, y)

    return {
        "model":     model,
        "features":  FEATURES,
        "loco_r2":   cv_scores,   # one score per held-out battery
    }
```

Expose LOCO R² scores in the **Diagnostics** tab of the dashboard.

---

## 8. Multivariate Anomaly Detection

**File:** `src/impedance_validation.py`  
**Current behaviour:** 2-sigma rule on impedance alone.

**Change:** Add an `IsolationForest` trained on 5 multivariate features per cycle, replacing the univariate sigma check as the primary anomaly flag.

```python
# impedance_validation.py — add to process_battery_impedance()

from sklearn.ensemble import IsolationForest
import numpy as np

ANOMALY_FEATURES = [
    "z_est_ohm",             # transient impedance
    "soh",                   # state of health
    "temperature_mean_c",    # thermal stress
    "coulombic_efficiency",  # efficiency signal
    "total_resistance_ohm",  # DC resistance
]

def detect_multivariate_anomalies(cycle_df: pd.DataFrame,
                                  contamination: float = 0.05) -> pd.Series:
    """
    Returns a boolean Series indexed like cycle_df: True = anomaly.
    contamination=0.05 flags ~5% of cycles as anomalous.
    """
    feat_df = cycle_df[ANOMALY_FEATURES].dropna()
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
```

Store `anomaly_if` column alongside the existing `anomaly_zscore` column in the exported parquet. Surface both in the **Diagnostics** tab so users can compare them.

---

## 9. Bayesian SOH Uncertainty + EKF Covariance in Dashboard

**File:** `src/state_estimators.py` and `app.py`

### 9a. Bayesian Ridge for SOH prediction intervals

```python
# state_estimators.py — replace Ridge with BayesianRidge

from sklearn.linear_model import BayesianRidge

# Replace:
#   model = Ridge(alpha=1.0)
# With:
model = BayesianRidge()
model.fit(X_train, y_train)

soh_pred, soh_std = model.predict(X_all, return_std=True)
# Store soh_pred ± 1.65 * soh_std as the 90% interval
cycle_df["soh_model_pred"]   = soh_pred
cycle_df["soh_model_lower"]  = soh_pred - 1.65 * soh_std
cycle_df["soh_model_upper"]  = soh_pred + 1.65 * soh_std
```

### 9b. EKF SOC uncertainty band

```python
# ecm.py — after each EKF timestep, record diagonal of P
soc_std_trace.append(float(np.sqrt(P[0, 0])))   # SOC variance

# Export as column soc_ekf_std alongside soc_ekf
```

### 9c. Dashboard — Battery Health tab

In `app.py`, for the SOH degradation chart, add shaded band:

```python
# app.py — SOH tab
fig.add_trace(go.Scatter(
    x=pd.concat([df["cycle_index"], df["cycle_index"][::-1]]),
    y=pd.concat([df["soh_model_upper"], df["soh_model_lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(99,110,250,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="SOH 90% CI",
))
```

---

## 10. Schema Validation on Ingestion

**File:** `src/data_loader.py`  
**Current behaviour:** No validation; bad MAT files produce silent NaNs.

**Change:** Add `pandera` schema checks at the end of `load_shadow_tables()`.

```bash
pip install pandera
```

```python
# data_loader.py — add after constructing cycle_table and sample_table

import pandera as pa
from pandera import Column, DataFrameSchema, Check

CYCLE_SCHEMA = DataFrameSchema({
    "cycle_index":           Column(int,   Check.ge(0)),
    "cycle_type":            Column(str,   Check.isin(["charge", "discharge", "impedance"])),
    "capacity_ah":           Column(float, Check.ge(0),   nullable=True),
    "temperature_mean_c":    Column(float, Check.between(-20, 80), nullable=True),
    "total_resistance_ohm":  Column(float, Check.ge(0),   nullable=True),
})

SAMPLE_SCHEMA = DataFrameSchema({
    "cycle_index":  Column(int,   Check.ge(0)),
    "voltage_v":    Column(float, Check.between(0, 5)),
    "current_a":    Column(float, Check.between(-10, 10)),
    "temperature_c":Column(float, Check.between(-20, 80), nullable=True),
})

def validate_tables(cycle_table, sample_table, battery_id):
    try:
        CYCLE_SCHEMA.validate(cycle_table,  lazy=True)
        SAMPLE_SCHEMA.validate(sample_table, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(
            f"[{battery_id}] Schema validation failed — "
            f"{len(e.failure_cases)} violations:\n{e.failure_cases}"
        )
```

Call `validate_tables(cycle_table, sample_table, battery_id)` at the bottom of `load_shadow_tables()` before returning.

---

## 11. Incremental Processing Cache

**File:** `src/pipeline.py`  
**Current behaviour:** Full re-processing on every run.

**Change:** Add a cycle-level cache keyed on `(battery_id, cycle_index)`. Only cycles not present in the cache are reprocessed; cached results are loaded from parquet.

```python
# pipeline.py — wrap the per-cycle processing block

CACHE_DIR = Path("artifacts/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_key(battery_id: str, cycle_index: int) -> Path:
    return CACHE_DIR / f"{battery_id}_cycle_{cycle_index:04d}.parquet"

def load_or_compute_cycle(battery_id, cycle_index, cycle_data, sample_data):
    cache_path = get_cache_key(battery_id, cycle_index)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    result = compute_cycle_features(cycle_data, sample_data)  # existing logic
    result.to_parquet(cache_path, index=False)
    return result
```

On a clean run the cache is cold and behaviour is identical. On re-runs (e.g. after changing only the dashboard), only uncached cycles (new data) are recomputed.

---

## 12. Experiment Tracking

**File:** `scripts/prepare_dashboard_data.py`  
**Current behaviour:** No record of which model version produced an artifact.

**Change:** Write a `run_metadata.json` to `artifacts/` on every pipeline execution.

```python
# scripts/prepare_dashboard_data.py — append at end of main()

import json, hashlib, datetime
from pathlib import Path

def write_run_metadata(config: dict, metrics: dict, artifact_dir: Path):
    run_id = hashlib.md5(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:8]

    metadata = {
        "run_id":     run_id,
        "timestamp":  datetime.datetime.utcnow().isoformat(),
        "config":     config,    # e.g. soh weights, EKF params, model versions
        "metrics":    metrics,   # e.g. LOCO R², ECM RMSE per battery
    }

    out_path = artifact_dir / f"run_{run_id}.json"
    out_path.write_text(json.dumps(metadata, indent=2))
    print(f"[pipeline] Run metadata saved → {out_path}")
```

Surface the most recent `run_id` and key metrics in the **Overview** tab of the Streamlit dashboard.

---

## Summary of File Changes

| File | Changes |
|------|---------|
| `src/features.py` | Data-optimised SOH weights; shape features; lag features |
| `src/state_estimators.py` | BayesianRidge for SOH; expose prediction intervals |
| `src/ecm.py` | Sage-Husa adaptive EKF covariance; export `soc_ekf_std` |
| `src/rul.py` | GPR-based RUL with CI; pooled cross-battery model; fitted stress coefficients |
| `src/impedance_validation.py` | IsolationForest multivariate anomaly detection |
| `src/data_loader.py` | Pandera schema validation on ingestion |
| `src/pipeline.py` | Cycle-level parquet cache; persist all new coefficients to JSON |
| `scripts/prepare_dashboard_data.py` | Run metadata / experiment tracking |
| `app.py` | SOH confidence band; RUL uncertainty band; LOCO R² in Diagnostics |

> **Suggested implementation order:** 10 (validation) → 11 (cache) → 1 (SOH weights) → 5–6 (features) → 4 (GPR RUL) → 2 (adaptive EKF) → 3 (stress coefficients) → 7 (pooled model) → 8 (anomaly detection) → 9 (uncertainty in dashboard) → 12 (experiment tracking)