# Li-ion Digital Shadow Implementation

## Objective
Build a notebook-first Li-ion battery digital shadow that loads NASA battery aging `.mat` files from `mat_files/`, normalizes the cycle data, and estimates:

- `SOC` from current integration and voltage anchors
- `SOH` from measured and modeled capacity fade
- `RUL` from degradation trajectories
- terminal voltage dynamics with a `2RC` Thevenin ECM
- temperature and current trajectories from measured data

## Scope
Version 1 is an offline analytical digital shadow, not a live deployed service. It uses the available battery measurements to reconstruct internal state and aging behavior cycle by cycle.

## Dataset Assumptions
Based on `mat_files/README.txt`, each battery file contains a top-level `cycle` structure with:

- `type`: `charge`, `discharge`, or `impedance`
- `ambient_temperature`
- `time`
- `data`: per-cycle measurements

Relevant signals:

- Charge/discharge: `Voltage_measured`, `Current_measured`, `Temperature_measured`, `Time`
- Discharge only: `Capacity`
- Impedance only: `Re`, `Rct`, `Battery_impedance`, `Rectified_impedance`

## Architecture
The implementation is split into a small reusable Python package plus notebooks.

### Project layout
- `src/data_loader.py`: MAT parsing and table normalization
- `src/features.py`: cycle-level and sample-level feature engineering
- `src/state_estimators.py`: `SOC`, `SOH`, `RUL`, and digital-shadow assembly
- `src/ecm.py`: `2RC` ECM simulation and parameter fitting
- `src/rul.py`: degradation-curve and EOL projection helpers
- `notebooks/01_load_and_flatten_mat.ipynb`: ingestion walkthrough
- `notebooks/02_feature_engineering_and_targets.ipynb`: features and health targets
- `notebooks/03_soc_soh_rul_and_ecm.ipynb`: digital shadow, ECM fit, and validation

## Digital Shadow State
At any time step, the shadow will track:

- Measured inputs: `current`, `terminal_voltage`, `temperature`, `dt`
- Estimated states: `soc`, `soh`, `rul_cycles`, `v_rc1`, `v_rc2`
- Slowly varying parameters: `r0`, `r1`, `c1`, `r2`, `c2`
- Metadata: `battery_id`, `cycle_index`, `cycle_type`, `timestamp`

## Modeling Strategy

### 1. Data ingestion
Parse each `.mat` battery file into:

- sample-level rows for time-series modeling
- cycle-level rows for degradation modeling

Both outputs use a common schema so all batteries can be concatenated.

### 2. SOC
Baseline SOC uses coulomb counting:

`SOC_t = clip(SOC_{t-1} - I_t * dt / (3600 * C_nominal), 0, 1)`

Notes:
- positive current during discharge reduces SOC
- drift is corrected with simple voltage anchors near full charge / end-of-discharge
- if the file lacks a clean anchor, the cycle is initialized from its starting condition

### 3. SOH
For discharge cycles:

`SOH = capacity_ah / initial_capacity_ah`

Primary explanatory features:
- discharge energy
- discharge duration
- voltage drop and mean voltage
- temperature rise
- `Re`, `Rct`, `Re + Rct`

Version 1 exposes both:
- measured SOH from capacity
- modeled SOH from regression against engineered features

### 4. RUL
End of life is defined by the NASA criterion:

`capacity <= 1.4 Ah`

or equivalently:

`SOH <= 0.70`

For compatibility with common battery-health reporting, the implementation also reports cycles remaining to a configurable threshold such as `SOH <= 0.80`.

Version 1 RUL uses a trend fit over historical SOH:
- linear degradation baseline
- polynomial fallback if the linear fit is degenerate

### 5. 2RC ECM
The terminal-voltage model is:

`V_t = OCV(SOC) - I * R0 - V1 - V2`

with RC branch dynamics:

`dV1/dt = -V1 / (R1*C1) + I / C1`

`dV2/dt = -V2 / (R2*C2) + I / C2`

Discrete-time update for step `dt`:

`a1 = exp(-dt / (R1*C1))`

`a2 = exp(-dt / (R2*C2))`

`V1_k = a1 * V1_{k-1} + R1 * (1 - a1) * I_k`

`V2_k = a2 * V2_{k-1} + R2 * (1 - a2) * I_k`

The first implementation fits `R0`, `R1`, `C1`, `R2`, `C2` by minimizing voltage reconstruction error against measured terminal voltage.

### 6. Temperature
Version 1 uses measured temperature directly in the shadow state and derives:
- `temp_rise`
- mean and max temperature
- coupling between thermal rise and resistance growth

This keeps the shadow physically interpretable without forcing a thermal PDE or RC thermal model too early.

## Validation

### Ingestion checks
- required fields exist per cycle type
- cycle counts are non-zero
- time vectors are monotonic

### SOC checks
- bounded in `[0, 1]`
- decreases during discharge and increases during charge on average

### SOH checks
- starts near `1.0`
- trends downward over aging cycles

### RUL checks
- remaining-cycle estimates approach zero near observed EOL

### ECM checks
- report voltage `MAE` and `RMSE`
- plot measured vs reconstructed terminal voltage

## Milestones
1. Build parsers and normalized tables for all battery files.
2. Create feature and target pipelines for cycle-level modeling.
3. Implement baseline `SOC`, measured `SOH`, and projected `RUL`.
4. Fit a `2RC` ECM and reconstruct terminal voltage.
5. Combine outputs into a single digital-shadow dataframe and notebook workflow.

## Constraints and Risks
- MATLAB struct parsing can differ by file format version.
- OCV estimation is approximate without dedicated rest segments.
- RUL generalization is limited by the small number of batteries.
- ECM fitting can become unstable without parameter bounds.

## Initial Success Criteria
- A notebook can load battery files from `mat_files/`.
- Sample-level and cycle-level tables are generated successfully.
- The digital shadow reports `SOC`, `SOH`, `RUL`, current, voltage, and temperature.
- The `2RC` ECM reconstructs terminal voltage with finite, reasonable error.



## 7. EOL Threshold Calculation Plan

### Objective

Estimate when the Li-ion battery reaches **End of Life (EOL)** using measurable degradation signals and configurable thresholds.

### Standard EOL Definition

Battery reaches EOL when usable performance falls below acceptable limits. Common thresholds:

* **Capacity-based EOL**:
  `Capacity <= 80% of initial capacity`

* **NASA dataset threshold** (already used):
  `Capacity <= 1.4 Ah`

* **SOH-based EOL**:
  `SOH <= 0.80` or `SOH <= 0.70`

* **Resistance-based EOL** (advanced):
  `Internal Resistance >= 150% of initial resistance`

---

### Inputs Required

From each cycle:

* Measured discharge capacity
* Initial fresh capacity
* SOH trend
* Cycle count
* Internal resistance (`Re`, `Rct`)
* Temperature stress indicators

---

### Step 1: Compute Dynamic SOH Per Cycle

[
SOH_i = \frac{Capacity_i}{Capacity_{initial}}
]

Track decline over cycles.

---

### Step 2: Detect EOL Threshold Crossing

Find first cycle `k` where:

[
SOH_k \le Threshold
]

Examples:

* 80% threshold → commercial battery replacement point
* 70% threshold → NASA aging benchmark

---

### Step 3: Predict Future EOL Cycle

Fit degradation curve using historical SOH:

#### Linear Model

SOH(c)=a-bc

Solve for threshold:

[
Threshold = a - b c_{EOL}
]

[
c_{EOL} = \frac{a-Threshold}{b}
]

#### Polynomial / Exponential Model (optional)

Use if nonlinear fade appears.

---

### Step 4: Remaining Useful Life (RUL)

[
RUL = c_{EOL} - c_{current}
]

Where:

* `c_EOL` = predicted end cycle
* `c_current` = present cycle

---

### Step 5: Confidence Band

Use rolling regression / bootstrap to estimate:

* Best case EOL
* Expected EOL
* Worst case EOL

---

### Output Columns to Add

| Column              | Meaning                     |
| ------------------- | --------------------------- |
| eol_threshold       | Configured SOH threshold    |
| predicted_eol_cycle | Expected failure cycle      |
| remaining_cycles    | RUL                         |
| eol_status          | Healthy / Warning / Reached |
| confidence_low      | Conservative estimate       |
| confidence_high     | Optimistic estimate         |

---

### Alert Logic

* **Healthy:** SOH > 85%
* **Warning:** 80% > SOH > Threshold
* **Critical:** SOH <= Threshold

---

### Notebook Integration

Add in `03_soc_soh_rul_and_ecm.ipynb`

1. Load SOH history
2. Fit degradation model
3. Predict EOL cycle
4. Plot threshold crossing
5. Show remaining life gauge

---

### Visualization Ideas

* SOH vs Cycles curve with EOL line
* RUL countdown chart
* Capacity fade trajectory
* Multi-battery comparison dashboard

---

### Success Criteria

* Predict threshold crossing before actual failure
* Error within ±20 cycles (baseline target)
* Stable forecasts after 30% lifecycle history


## 4. Physics-Based Enhancements

### Objective

Improve the fidelity and interpretability of the digital shadow by extending the existing `2RC` equivalent-circuit model with state-dependent parameters, voltage-anchor corrections, and charge-efficiency tracking. These additions reduce estimator drift and better represent real battery behavior across aging cycles. 

### Scope

This section extends the current modules:

* `src/ecm.py`
* `src/state_estimators.py`
* `src/features.py`

The enhancements remain compatible with the notebook-first offline analytical workflow already defined in the implementation plan. 

---

## 4.1 Adaptive ECM Parameters

### Motivation

The current `2RC` Thevenin ECM uses fixed parameters:

* `r0`
* `r1`
* `c1`
* `r2`
* `c2`

In practical Li-ion cells, these values vary with:

* `soc`
* temperature
* aging level (`soh`)
* load conditions

Using fixed values can reduce voltage reconstruction accuracy as the battery degrades. 

### Proposed Model

Replace constant parameters with state-dependent functions:

r_0=f(SOC,T,SOH)

Additional mappings:

* `r1 = f(soc, temperature)`
* `c1 = f(soh)`
* `r2 = f(soc, temperature)`
* `c2 = f(soh, temperature)`

### Implementation Plan

#### Data Sources

Use existing measured signals:

* terminal voltage
* current
* temperature
* cycle index
* estimated `soc`
* estimated `soh`

#### Parameter Estimation Methods

Versioned options:

1. **Lookup tables** using SOC and temperature bins
2. **Polynomial regression**
3. **Tree-based regressors** (Random Forest / XGBoost)
4. **Online recursive fitting** for future live deployment

#### Module Updates

Add to `src/ecm.py`:

* `get_dynamic_params(soc, temp, soh)`
* `fit_parameter_surface(df)`

### Expected Outputs

Per time step:

* `r0_dynamic`
* `r1_dynamic`
* `c1_dynamic`
* `r2_dynamic`
* `c2_dynamic`

### Validation

* Compare voltage `MAE` before vs after dynamic ECM
* Verify resistance growth trends upward with aging
* Ensure parameters remain within physical bounds

---

## 4.2 OCV-SOC Lookup Engine

### Motivation

Baseline SOC currently uses coulomb counting, which can accumulate integration drift over long horizons. Voltage-based anchoring improves long-term SOC stability. 

### Proposed Model

Use open-circuit voltage during low-current rest windows:

SOC=g(OCV)

Where:

* `OCV` = relaxed terminal voltage
* `g(.)` = lookup/interpolation mapping

### Implementation Plan

#### Step 1: Rest Segment Detection

Detect intervals where:

* `|current| < epsilon`
* voltage slope is small
* signal noise is low

#### Step 2: Build Lookup Table

Create battery-specific or fleet-average table:

| OCV  | SOC  |
| ---- | ---- |
| 4.20 | 1.00 |
| 3.95 | 0.75 |
| 3.75 | 0.50 |
| 3.55 | 0.25 |
| 3.20 | 0.00 |

#### Step 3: SOC Correction Logic

Combine coulomb counting with anchors:

`SOC_corrected = alpha * SOC_counted + (1-alpha) * SOC_ocv`

#### Module Updates

Add to `src/state_estimators.py`:

* `estimate_soc_ocv(voltage)`
* `apply_soc_anchor(df)`

### Expected Outputs

* `soc_raw`
* `soc_ocv`
* `soc_corrected`

### Validation

* SOC remains within `[0,1]`
* Reduced long-cycle drift
* Better charge/discharge endpoint alignment

---

## 4.3 Coulombic Efficiency Tracking

### Motivation

Charge inefficiency increases as Li-ion batteries age. Monitoring coulombic efficiency provides an early degradation indicator before sharp capacity loss appears.

### Definition

\eta=\frac{Q_{discharge}}{Q_{charge}}

Where:

* `Q_charge` = amp-hour input during charge
* `Q_discharge` = amp-hour output during discharge

Ideal healthy cells trend close to `1.0`.

### Implementation Plan

#### Per-Cycle Computation

For each paired charge/discharge cycle:

* integrate charge current over time
* integrate discharge current over time
* compute efficiency ratio

#### Derived Features

Store:

* `coulombic_efficiency`
* rolling mean efficiency
* rolling efficiency decline rate

#### Module Updates

Add to `src/features.py`:

* `compute_cycle_efficiency(df)`
* `compute_efficiency_trends(df)`

### Expected Use Cases

Use as predictor in:

* SOH regression
* RUL forecasting
* anomaly detection

### Validation

* Values remain in plausible range
* Long-term decline correlates with capacity fade
* Detect abrupt drops after stress events

---

## Integration into Digital Shadow State

Extend shadow dataframe with:

* `r0_dynamic`
* `r1_dynamic`
* `c1_dynamic`
* `r2_dynamic`
* `c2_dynamic`
* `soc_ocv`
* `soc_corrected`
* `coulombic_efficiency`

---

## Notebook Additions

Update `03_soc_soh_rul_and_ecm.ipynb` to include:

1. Dynamic ECM parameter fitting
2. OCV-SOC calibration plots
3. Coulombic efficiency trends
4. Voltage reconstruction comparison (fixed vs adaptive ECM)
5. Impact on RUL forecast accuracy

---

## Success Criteria

* Lower voltage reconstruction error than fixed-parameter ECM
* Reduced SOC drift over long cycles
* Coulombic efficiency trends provide early degradation signal
* Improved SOH / RUL prediction performance when added as features
