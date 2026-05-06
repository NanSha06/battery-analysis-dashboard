# IMPLEMENTATION PROMPT — Add Impedance Validation Layer to Battery Digital Shadow System

You are a senior battery analytics engineer and Python architect.

Enhance the existing Battery Digital Shadow & Analytics platform by implementing an impedance-informed ECM validation pipeline and integrating it into the Streamlit dashboard.

The current project already contains:

* ECM parameter extraction
* R0 estimation
* SOC/SOH estimation
* RUL prediction
* Streamlit dashboard
* Parquet artifact generation

The project structure currently includes:

```text
src/
 ├── data_loader.py
 ├── features.py
 ├── ecm.py
 ├── rul.py
 ├── state_estimators.py
 ├── pipeline.py
 └── dashboard_data.py
```

Implement the following enhancements carefully and modularly.

---

1. CREATE NEW MODULE

---

Create:

```text
src/impedance_validation.py
```

This module will:

* estimate transient impedance
* validate ECM-derived R0 values
* analyze resistance growth trends
* generate impedance metrics
* export validation artifacts

---

2. IMPLEMENT IMPEDANCE ESTIMATION

---

Use transient voltage/current response to estimate impedance.

Use:

Z_{est} = \frac{\Delta V}{\Delta I}

Implement:

```python
def detect_current_pulses(current, threshold=0.5):
```

Responsibilities:

* identify sudden current transitions
* detect discharge/load pulses
* return pulse indices

Then implement:

```python
def estimate_impedance(voltage, current, pulse_indices):
```

Responsibilities:

* compute delta V
* compute delta I
* estimate impedance
* filter invalid/noisy values
* return impedance series

Requirements:

* use NumPy/Pandas
* handle divide-by-zero safely
* ignore unstable spikes
* include smoothing option

---

3. VALIDATE ECM R0 VALUES

---

Implement:

```python
def validate_r0(r0_values, impedance_values):
```

Compute:

* MAE
* RMSE
* correlation coefficient
* relative drift
* trend consistency

Use:

* sklearn.metrics
* scipy.stats

Return dictionary:

```python
{
    "mae": ...,
    "rmse": ...,
    "correlation": ...,
    "trend_consistency": ...,
    "drift_percent": ...
}
```

---

4. IMPLEMENT IMPEDANCE TREND ANALYSIS

---

Implement:

```python
def analyze_impedance_growth(cycles, impedance):
```

Responsibilities:

* analyze degradation trend
* compute resistance growth rate
* detect abnormal spikes
* estimate aging progression

Return:

* rolling averages
* degradation slope
* anomaly markers

---

5. MODIFY ECM PIPELINE

---

Update:

```text
src/pipeline.py
```

New flow:

```text
Load Data
   ↓
Feature Engineering
   ↓
ECM Parameter Extraction
   ↓
Impedance Estimation
   ↓
R0 Validation
   ↓
SOH/RUL Enhancement
   ↓
Artifact Export
```

Pipeline must:

* call impedance validation functions
* merge outputs into cycle-level analytics
* export metrics

---

6. CREATE NEW ARTIFACTS

---

Inside:

```text
artifacts/
```

Generate:

```text
impedance_curve.parquet
r0_validation.json
impedance_metrics.json
impedance_trend.parquet
```

Each artifact should contain:

* timestamps/cycles
* impedance estimates
* ECM R0 values
* validation metrics
* degradation trends

---

7. ENHANCE SOH ESTIMATION

---

Modify SOH logic to include resistance growth.

Use hybrid SOH:

SOH = w_1 \cdot Capacity_{norm} + w_2 \cdot \frac{R_{0,initial}}{R_0}

Requirements:

* configurable weights
* normalized resistance contribution
* resistance-aware health estimation

---

8. ENHANCE RUL PREDICTION

---

Incorporate:

* impedance growth
* resistance drift
* degradation slope

into RUL forecasting.

The system should use:

* capacity fade
* impedance increase
* thermal stress indicators

to improve prediction realism.

---

9. DASHBOARD INTEGRATION

---

Update:

```text
app.py
```

Add new dashboard section:

# “Impedance & ECM Validation”

Include the following visualizations:

---

## A. Impedance vs Cycle Plot

Show:

* cycle number on x-axis
* impedance on y-axis
* rolling average trendline

Purpose:

* visualize degradation progression

---

## B. R0 vs Estimated Impedance

Overlay:

* ECM-derived R0
* estimated impedance

Display:

* validation consistency
* divergence regions

---

## C. Validation Metrics Cards

Display:

* RMSE
* MAE
* Correlation
* Drift %
* Resistance growth %

Use:

* Streamlit metric cards

---

## D. Resistance Growth Trend

Plot:

* resistance increase over lifecycle
* anomaly markers

Purpose:

* aging diagnostics

---

## E. SOH vs Impedance Relationship

Scatter plot:

* SOH
* impedance

Purpose:

* visualize degradation correlation

---

10. STREAMLIT UI REQUIREMENTS

---

Dashboard must:

* remain responsive
* support cached artifact loading
* use clean layouts
* include section headers
* include scientific explanations
* show equations/tooltips where useful

Add expandable explanations:

* what impedance means
* why R0 matters
* how degradation affects resistance

---

11. ENGINEERING REQUIREMENTS

---

Code must:

* be modular
* production-quality
* fully commented
* type hinted
* exception-safe
* numerically stable

Use:

* NumPy
* Pandas
* SciPy
* scikit-learn
* Plotly or Altair for charts
* Streamlit caching

---

12. OPTIONAL ADVANCED FEATURES

---

If possible, also implement:

* anomaly detection on resistance spikes
* adaptive pulse thresholding
* Savitzky–Golay smoothing
* confidence intervals
* impedance heatmaps

---

13. EXPECTED FINAL OUTCOME

---

The final system should evolve from:

```text
data-driven battery analytics
```

into:

```text
physics-informed battery intelligence platform
```

The dashboard should clearly demonstrate:

* physically validated ECM parameters
* resistance-aware degradation
* impedance-informed SOH/RUL estimation
* electrochemical interpretability
* digital shadow realism

```
```
