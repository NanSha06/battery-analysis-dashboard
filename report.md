# Li-ion Battery Digital Shadow — Technical Report

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Data Pipeline Architecture](#3-data-pipeline-architecture)
4. [Parameter Calculations & Mathematics](#4-parameter-calculations--mathematics)
   - 4.1 [Data Ingestion](#41-data-ingestion)
   - 4.2 [State of Charge (SOC)](#42-state-of-charge-soc)
   - 4.3 [State of Health (SOH)](#43-state-of-health-soh)
   - 4.4 [Remaining Useful Life (RUL)](#44-remaining-useful-life-rul)
   - 4.5 [Equivalent Circuit Model (ECM)](#45-equivalent-circuit-model-ecm)
   - 4.6 [Extended Kalman Filter (EKF)](#46-extended-kalman-filter-ekf)
   - 4.7 [Impedance Validation](#47-impedance-validation)
   - 4.8 [R0 Scale Alignment](#48-r0-scale-alignment)
   - 4.9 [Coulombic Efficiency](#49-coulombic-efficiency)
   - 4.10 [End of Life (EOL)](#410-end-of-life-eol)
   - 4.11 [Dynamic ECM Parameters](#411-dynamic-ecm-parameters)
   - 4.12 [Anomaly Detection](#412-anomaly-detection)
5. [Module Reference](#5-module-reference)
6. [Dashboard Visualization](#6-dashboard-visualization)

---

## 1. Project Overview

This project implements a **Digital Shadow** (a data-driven digital twin) for NASA Li-ion batteries (B0005, B0006, B0007, B0018). It ingests raw `.mat` telemetry files, constructs physics-informed battery state models, and renders an interactive Streamlit dashboard for real-time health monitoring, degradation tracking, and predictive maintenance.

The system combines:
- **Coulomb counting** for SOC estimation
- **Weighted capacity–resistance fusion** for SOH
- **Linear extrapolation with stress correction** for RUL
- **2-RC Equivalent Circuit Modeling** for voltage reconstruction
- **Extended Kalman Filtering** for real-time state correction
- **Transient impedance analysis** for ECM validation
- **EIS-based scale alignment** for physical unit consistency

---

## 2. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | Core implementation |
| Data Format | NASA `.mat` files | Raw battery telemetry |
| Data I/O | `scipy.io.loadmat` | MAT file parsing |
| DataFrames | `pandas`, `numpy` | Tabular data processing |
| Optimization | `scipy.optimize.least_squares` | ECM parameter fitting |
| Statistics | `scipy.stats.pearsonr` | Correlation analysis |
| ML | `scikit-learn` (Ridge regression) | SOH regression model |
| Visualization | `plotly` (dark theme) | Interactive charts |
| Dashboard | `Streamlit` | Web-based UI |
| Serialization | `parquet`, `json` | Artifact storage |

**File structure:**
```
src/
├── data_loader.py          # MAT ingestion & table construction
├── features.py             # Feature engineering (SOH, efficiency)
├── state_estimators.py     # SOC, SOH, shadow state builder
├── rul.py                  # RUL estimation & EOL prediction
├── ecm.py                  # 2-RC ECM, EKF, OCV curve
├── impedance_validation.py # Transient impedance & R0 validation
├── pipeline.py             # Orchestration & artifact export
├── dashboard_data.py       # Dashboard data loading utilities
app.py                      # Streamlit dashboard
scripts/prepare_dashboard_data.py  # CLI entry point
```

---

## 3. Data Pipeline Architecture

```
NASA .mat files
      │
      ▼
┌─────────────────┐
│  data_loader.py  │  Parse MAT → cycle_table + sample_table
└────────┬────────┘
         ▼
┌─────────────────────┐
│ state_estimators.py  │  SOC (Coulomb counting) → SOH (Ridge) → RUL
└────────┬────────────┘
         ▼
┌─────────────────┐
│     ecm.py       │  Fit 2-RC params → Simulate voltage → Run EKF
└────────┬────────┘
         ▼
┌──────────────────────────┐
│ impedance_validation.py  │  Transient Z → R0 validation → Growth trends
└────────┬─────────────────┘
         ▼
┌─────────────────┐
│   pipeline.py    │  Scale alignment → Artifact export (parquet/json)
└────────┬────────┘
         ▼
┌─────────────────┐
│     app.py       │  Streamlit dashboard (6 tabs)
└─────────────────┘
```

---

## 4. Parameter Calculations & Mathematics

### 4.1 Data Ingestion

**Source:** `src/data_loader.py`

Each NASA `.mat` file contains a structured array of charge/discharge/impedance cycles. Per cycle, the following raw vectors are extracted:

| Field | Variable | Unit |
|-------|----------|------|
| `Voltage_measured` | `voltage_v` | V |
| `Current_measured` | `current_a` | A |
| `Temperature_measured` | `temperature_c` | °C |
| `Capacity` | `capacity_ah` | Ah |
| `Re` | `re_ohm` | Ω |
| `Rct` | `rct_ohm` | Ω |

**Cycle-level aggregates** are also computed:
- `duration_s` = time of last sample − time of first sample
- `voltage_min_v`, `voltage_max_v` = min/max of voltage vector
- `current_mean_a` = mean of current vector
- `temperature_mean_c`, `temperature_max_c` = mean/max of temperature vector

---

### 4.2 State of Charge (SOC)

**Source:** `src/state_estimators.py`

#### Method 1: Coulomb Counting

SOC is estimated by integrating current over time:

$$\text{SOC}_k = \text{SOC}_{k-1} + \frac{I_k \Delta t}{3600 Q_{\text{nom}}}$$

Where:
- $I_k$ = current at timestep $k$ (A)
- $\Delta t$ = time difference `dt_s` (s)
- $Q_{\text{nom}}$ = nominal capacity = 2.0 Ah
- Direction is $+1$ for charge, $-1$ for discharge
- SOC is clipped to $[0.0, 1.0]$

Initial conditions: $\text{SOC}_0 = 0.1$ (charge), $\text{SOC}_0 = 1.0$ (discharge)

#### Method 2: OCV Lookup

SOC is estimated by inverting the Open Circuit Voltage curve:

$$\text{SOC} = f_{\text{OCV}}^{-1}(V_{\text{measured}})$$

The OCV curve is built by binning SOC values into 5% intervals and taking the **median voltage** per bin.

#### Method 3: SOC Anchor Correction

During rest periods ($|I| < 0.05$ A and $|dV/dt| < 0.002$ V/s), the Coulomb-counted SOC is blended with the OCV-derived SOC:

$$\text{SOC}_{\text{corrected}} = \alpha \cdot \text{SOC}_{\text{coulomb}} + (1 - \alpha) \cdot \text{SOC}_{\text{OCV}}$$

Where $\alpha = 0.85$ (default). This corrects long-term Coulomb counting drift.

---

### 4.3 State of Health (SOH)

**Source:** `src/features.py`

SOH uses a **weighted capacity–resistance fusion**:

$$\text{SOH}_k = w_1 \left( \frac{C_k}{C_0} \right) + w_2 \left( \frac{R_0}{R_k} \right)$$

Where:
- $C_k$ = discharge capacity at cycle $k$ (Ah)
- $C_0$ = initial discharge capacity (Ah)
- $R_k$ = total resistance at cycle $k = R_e + R_{ct}$ (Ω)
- $R_0$ = initial total resistance (Ω)
- $w_1 = 0.8$, $w_2 = 0.2$

**Rationale:** Capacity fade is the primary degradation indicator, but resistance growth provides a complementary signal. As the battery ages, capacity decreases and resistance increases, both driving SOH downward.

#### SOH Regression Model

A **Ridge regression** model is also trained on discharge cycles to predict SOH from operational features:

**Features:** `duration_s`, `voltage_min_v`, `voltage_max_v`, `temperature_mean_c`, `temperature_max_c`, `total_resistance_ohm`, `rct_delta`, `re_delta`

$$\text{SOH}_{\text{model}} = \beta_0 + \sum_{i=1}^n \beta_i x_i$$

Regularization: $\alpha = 1.0$ (L2 penalty)

---

### 4.4 Remaining Useful Life (RUL)

**Source:** `src/rul.py`

#### Linear Degradation Model

A first-order polynomial is fitted to SOH vs. cycle index on discharge cycles:

$$\text{SOH}(n) = m \cdot n + b$$

Where $m$ = slope (negative, representing degradation rate), $b$ = intercept.

**EOL cycle** is estimated by solving for the threshold:

$$n_{\text{EOL}} = \frac{\text{SOH}_{\text{threshold}} - b}{m}$$

Default threshold: $\text{SOH}_{\text{threshold}} = 0.80$

**RUL** at any cycle $n$:

$$\text{RUL} = n_{\text{EOL}} - n$$

#### Stress-Adjusted Correction

If thermal and resistance data are available, the predicted EOL is adjusted:

$$n_{\text{EOL}}^{\text{adj}} = n_{\text{EOL}} \cdot (1 - \sigma_{\text{temp}} - \sigma_{\text{res}})$$

Where:
- $\sigma_{\text{temp}} = \text{clip}\left(\frac{T_{\text{mean}} - 30}{20}, 0, 0.15\right)$
- $\sigma_{\text{res}} = \text{clip}\left(\frac{R_{\text{max}} - R_{\text{min}}}{R_{\text{min}}} \cdot 0.1, 0, 0.15\right)$

This penalizes batteries operating at elevated temperatures or exhibiting large resistance swings.

---

### 4.5 Equivalent Circuit Model (ECM)

**Source:** `src/ecm.py`

The battery is modeled as a **2-RC Thévenin equivalent circuit**:

```
    R0        R1         R2
──/\/\/──┬──/\/\/──┬──/\/\/──┬──
         │        │         │
        C1       C2       OCV(SOC)
         │        │         │
─────────┴────────┴─────────┴──
```

#### Terminal Voltage Equation

$$V_{\text{term}}(k) = V_{\text{OCV}}(\text{SOC}_k) - I_k R_0 - V_{\text{RC1}}(k) - V_{\text{RC2}}(k)$$

Where the RC branch voltages evolve as:

$$V_{\text{RC},j}(k) = a_j V_{\text{RC},j}(k-1) + R_j (1 - a_j) I_k$$

$$a_j = \exp\left(-\frac{\Delta t}{R_j C_j}\right)$$

**Parameters:** `R0`, `R1`, `C1`, `R2`, `C2`

| Parameter | Physical Meaning | Typical Range |
|-----------|-----------------|---------------|
| R0 | Ohmic resistance (electrolyte, contacts) | 1e-5 – 1.0 Ω |
| R1 | Charge-transfer resistance (fast) | 1e-5 – 1.0 Ω |
| C1 | Double-layer capacitance (fast) | 1 – 1e6 F |
| R2 | Diffusion resistance (slow) | 1e-5 – 1.0 Ω |
| C2 | Diffusion capacitance (slow) | 1 – 1e6 F |

#### Parameter Fitting

Parameters are identified via **bounded nonlinear least squares** (`scipy.optimize.least_squares`):

$$\theta^* = \arg\min_{\theta} \sum_{k} \left( V_{\text{model}}(k, \theta) - V_{\text{meas}}(k) \right)^2$$

Initial guess: $\theta_0 = [0.01, 0.01, 2000, 0.02, 4000]$

Bounds: $R \in [10^{-5}, 1.0]$, $C \in [1.0, 10^6]$

#### Error Metrics

$$\text{MAE} = \frac{1}{N} \sum_{k=1}^N |V_{\text{model},k} - V_{\text{meas},k}|$$

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{k=1}^N (V_{\text{model},k} - V_{\text{meas},k})^2}$$

---

### 4.6 Extended Kalman Filter (EKF)

**Source:** `src/ecm.py` → `run_ekf_soc_ocv()`

The EKF recursively estimates the battery state $\mathbf{x} = [SOC, V_{RC1}, V_{RC2}]^T$ by fusing the ECM model prediction with voltage measurements.

#### State Prediction (Time Update)

$$\text{SOC}_{k|k-1} = \text{clip}\left(\text{SOC}_{k-1} - \frac{I_k \Delta t}{3600 Q_{\text{nom}}}, 0, 1\right)$$

$$V_{\text{RC},j,k|k-1} = a_j V_{\text{RC},j,k-1} + b_j I_k$$

State transition Jacobian:

$$\mathbf{F} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & a_1 & 0 \\ 0 & 0 & a_2 \end{bmatrix}$$

Covariance prediction:

$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}$$

#### Measurement Update

Predicted measurement:

$$\hat{V}_k = V_{\text{OCV}}(\text{SOC}_{k|k-1}) - I_k R_0 - V_{\text{RC1},k|k-1} - V_{\text{RC2},k|k-1}$$

Measurement Jacobian:

$$\mathbf{H} = \begin{bmatrix} \frac{\partial V_{\text{OCV}}}{\partial \text{SOC}} & -1 & -1 \end{bmatrix}$$

Innovation, Kalman gain, and state update:

$$\mathbf{y}_k = V_{\text{meas},k} - \hat{V}_k$$

$$\mathbf{S} = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + R$$

$$\mathbf{K} = \mathbf{P}_{k|k-1} \mathbf{H}^T \mathbf{S}^{-1}$$

$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K} \mathbf{y}_k$$

$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P}_{k|k-1}$$

#### Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `process_var_soc` | 1e-6 | SOC process noise |
| `process_var_v1` | 1e-5 | V_RC1 process noise |
| `process_var_v2` | 1e-5 | V_RC2 process noise |
| `measurement_var_v` | 2.5e-3 | Voltage measurement noise |

---

### 4.7 Impedance Validation

**Source:** `src/impedance_validation.py`

#### Transient Impedance Estimation

Current pulses are detected where $|\Delta I| > 0.5$ A (configurable threshold). At each pulse edge:

$$Z_{\text{est}} = \frac{|\Delta V|}{|\Delta I|} = \frac{|V_{k+1} - V_k|}{|I_{k+1} - I_k|}$$

Values are filtered to the physically realistic range $[10^{-5}, 10.0]$ Ω.

#### R0 Validation Metrics

The ECM-derived R0 (aligned) is compared against the transient impedance estimates:

| Metric | Formula |
|--------|---------|
| MAE | $\frac{1}{N} \sum |R_0 - Z_{\text{est}}|$ |
| RMSE | $\sqrt{\frac{1}{N} \sum (R_0 - Z_{\text{est}})^2}$ |
| Correlation | Pearson $r$ between $R_0$ and $Z_{\text{est}}$ series |
| Drift % | $\frac{\bar{R}_0 - \bar{Z}_{\text{est}}}{\bar{Z}_{\text{est}}} \times 100$ |
| Trend Consistency | 1.0 if both slopes have same sign, else 0.0 |

#### Impedance Growth Analysis

A rolling average (window=10) and linear slope are computed over the cycle life to track degradation:

$$\text{growth\_rate} = m \quad \text{where} \quad Z(n) = m \cdot n + b$$

Anomalies are flagged when $|Z_k - \bar{Z}_{\text{rolling}}| > 2\sigma$.

---

### 4.8 R0 Scale Alignment

**Source:** `src/pipeline.py`

The ECM fitting operates on normalized current/voltage vectors, producing R0 values that are orders of magnitude smaller than physical resistance. The pipeline corrects this:

$$\gamma = \frac{\bar{R}_{\text{EIS}}}{\bar{R}_{0,\text{pred}}}$$

$$R_{0,\text{aligned}} = R_{0,\text{pred}} \cdot \gamma$$

Where $R_{\text{EIS}}$ is the `re_ohm` column from NASA EIS measurements (~0.045–0.062 Ω). This is computed per battery and persisted in `scaling_metrics.json`.

---

### 4.9 Coulombic Efficiency

**Source:** `src/features.py`

$$\eta_{\text{CE}} = \frac{Q_{\text{discharge}}}{Q_{\text{charge}}}$$

Where:

$$Q = \sum_k \frac{|I_k| \Delta t_k}{3600} \quad \text{(Ah)}$$

Only values in $[0, 1.2]$ are considered plausible. A rolling mean (window=10) and decline rate (rolling polyfit slope) are computed for trend analysis.

---

### 4.10 End of Life (EOL)

**Source:** `app.py`

EOL status is determined by two NASA-defined thresholds:

| Condition | Threshold |
|-----------|-----------|
| Capacity EOL | $C_{discharge} \leq 1.4$ Ah |
| SOH EOL | $SOH \leq 0.70$ |

**Status labels:**

| Status | Condition |
|--------|-----------|
| Healthy | SOH > 0.85 |
| Watch | 0.80 < SOH ≤ 0.85 |
| Warning | SOH ≤ 0.80 |
| Reached | SOH ≤ 0.70 or Capacity ≤ 1.4 Ah |

---

### 4.11 Dynamic ECM Parameters

**Source:** `src/ecm.py` → `get_dynamic_params()`

ECM parameters are modulated by operating conditions:

**Stress factors:**

$$\sigma_{\text{SOC}} = \text{clip}(|\text{SOC} - 0.5| \cdot 2, 0, 1)$$

$$\sigma_{\text{temp}} = \text{clip}\left(\frac{T_{\text{ref}} - T}{25}, -0.5, 1.5\right)$$

$$\sigma_{\text{aging}} = \text{clip}(1 - \text{SOH}, 0, 0.6)$$

**Resistance scaling:**

$$R_{\text{dyn}} = R_{\text{base}} \cdot (1 + 0.25\sigma_{\text{SOC}} + 0.20\sigma_{\text{temp}} + 1.20\sigma_{\text{aging}})$$

**Capacitance scaling:**

$$C_{\text{dyn}} = C_{\text{base}} \cdot \text{clip}\left(1 - 0.45\sigma_{\text{aging}} + 0.08 \frac{T - T_{\text{ref}}}{25}, 0.2, 1.5\right)$$

---

### 4.12 Anomaly Detection

Impedance anomalies are detected using a **2-sigma deviation** from the rolling average:

$$\text{anomaly}_k = \begin{cases} 1 & \text{if } |Z_k - \bar{Z}_{\text{roll},k}| > 2\sigma_Z \\ 0 & \text{otherwise} \end{cases}$$

Validation warnings are triggered for:
- $R_0 < 0$ (physically impossible)
- $R_0 > 1.0$ Ω (potential unit error)
- Scaling drift > 200% (misaligned normalizer)

---

## 5. Module Reference

| Module | Key Functions | Output |
|--------|--------------|--------|
| `data_loader.py` | `load_shadow_tables()` | `cycle_table`, `sample_table` |
| `features.py` | `add_cycle_features()`, `compute_cycle_efficiency()` | SOH, resistance deltas, efficiency |
| `state_estimators.py` | `estimate_soc_coulomb_counting()`, `build_shadow_state()` | SOC, SOH, RUL, cycle counters |
| `rul.py` | `fit_linear_degradation()`, `estimate_eol_cycle()` | EOL cycle, RUL per cycle |
| `ecm.py` | `fit_2rc_parameters()`, `simulate_2rc_ecm()`, `run_ekf_soc_ocv()` | ECM params, model voltage, EKF SOC |
| `impedance_validation.py` | `process_battery_impedance()`, `validate_r0()` | Transient Z, validation metrics |
| `pipeline.py` | `build_and_export_dashboard_artifacts()` | All parquet/json artifacts |
| `dashboard_data.py` | `load_global_tables()`, `load_battery_table()` | Dashboard data loaders |

---

## 6. Dashboard Visualization

The Streamlit dashboard is organized into **6 tabs**:

| Tab | Contents |
|-----|----------|
| **Overview** | AI insight summary, multi-battery comparison (SOH, RUL bars), EOL prediction plot |
| **Battery Health** | SOH degradation curves, RUL projection across all batteries |
| **ECM & Impedance** | R0 validation, transient impedance growth, EIS vs Model R0, scaling diagnostics, ECM parameter evolution |
| **Electrochemical Insights** | OCV curve, voltage reconstruction (measured vs ECM vs EKF), SOC traces, thermal profiles, dynamic resistance/capacitance, coulombic efficiency |
| **Diagnostics** | Cycle-level voltage/SOC/thermal plots, residual analysis, ECM metrics JSON |
| **Data Explorer** | Raw sample data table, cycle-level ECM table, CSV download |

**Design system:** Dark charcoal theme (`#0e1117`), glassmorphism KPI cards with hover animations, `plotly_dark` chart template, responsive grid layouts.

---

*Generated for the Li-ion Battery Digital Shadow project — NASA Battery Dataset (B0005, B0006, B0007, B0018)*
