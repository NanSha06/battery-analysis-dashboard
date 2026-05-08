You are a senior battery systems engineer and Python developer working on a Li-ion Battery Digital Shadow built on the NASA dataset (B0005, B0006, B0007, B0018).

The existing codebase has these modules:
- src/data_loader.py — MAT ingestion + Pandera validation
- src/features.py — SOH fusion, shape features, lag features, efficiency
- src/state_estimators.py — SOC, BayesianRidge SOH, shadow assembly
- src/rul.py — linear RUL, GPR RUL, pooled LOCO model, stress correction
- src/ecm.py — 2RC ECM, EKF, dynamic parameters
- src/impedance_validation.py — transient impedance, IsolationForest anomaly detection
- src/pipeline.py — orchestration, parquet/JSON artifact export
- app.py — Streamlit dashboard (6 tabs, glassmorphism KPI cards, plotly dark)

Add the following features in this exact order. After each group, the pipeline must still run end-to-end without errors before moving to the next group.

---

GROUP 1 — PHYSICS (src/ecm.py, src/features.py)

1. State of Power (SOP): compute per-cycle peak deliverable power using
   SOP = ((V_min - OCV(SOC)) / (R0 + R1 + R2)) * V_min
   where V_min = 3.0V. Add sop_w column to cycle_shadow. Use existing ECM params and OCV curve.

2. Lithium plating risk index: add plating_risk column to cycle_shadow.
   plating_risk = clip((charge_rate_c / max(temperature_c, 1)) * (1 - soc), 0, 1)
   where charge_rate_c = current_mean_a / nominal_capacity_ah. Flag cycles where plating_risk > 0.4.

3. Re/Rct ratio trend: re_rct_ratio already exists in features.py. Extend it to compute
   re_rct_slope per battery (linear polyfit slope over discharge cycles). Add to cycle_shadow.
   This is a degradation mode indicator — rising ratio means electrolyte degradation dominates.

---

GROUP 2 — DEGRADATION MODELING (src/rul.py)

4. Knee-point detection: use the kneed library (add to requirements.txt).
   For each battery's discharge SOH series, detect the knee cycle using KneeLocator
   with curve='concave', direction='decreasing'. Store knee_cycle and
   post_knee_degradation_rate (slope after the knee) in cycle_shadow.
   In add_rul_estimates, if the current cycle is past the knee, use the steeper
   post-knee slope for EOL extrapolation instead of the global linear fit.

5. Depth of discharge (DoD): compute per discharge cycle as
   dod = soc_corrected.max() - soc_corrected.min() from sample_shadow.
   Add dod column to cycle_shadow via pipeline merge. Use as a feature in
   the BayesianRidge SOH model in state_estimators.py.

6. Arrhenius semi-empirical capacity fade: add fit_arrhenius_rul() to rul.py.
   Model: Q_loss = A * exp(-Ea / (R * T_kelvin)) * cycle_index^0.5
   Fit A and Ea using scipy.optimize.curve_fit on (cycle_index, soh, temperature_mean_c).
   Store rul_arrhenius column alongside rul_cycles and rul_cycles_gpr in cycle_shadow.

---

GROUP 3 — OPERATIONAL INTELLIGENCE (src/features.py, src/pipeline.py)

7. Operating regime clustering: fit KMeans(n_clusters=3) on discharge cycles using
   [temperature_mean_c, current_mean_a, duration_s, dod] (StandardScaler first).
   Add operating_regime column (0, 1, 2) to cycle_shadow.
   Compute per-regime degradation rates and store in a regime_stats.json artifact.

8. Charge protocol recommendation: add get_charge_recommendation(soh, rul_cycles,
   temperature_mean_c, plating_risk) to a new src/recommendations.py module.
   Returns a dict with action (one of: 'normal', 'reduce_crate', 'reduce_voltage',
   'inspect', 'replace') and reason string. Thresholds:
   - plating_risk > 0.4 → reduce_crate
   - soh < 0.80 and rul_cycles < 20 → inspect
   - soh < 0.70 → replace
   - temperature_mean_c > 45 → reduce_voltage
   - else → normal

---

GROUP 4 — UNCERTAINTY & RELIABILITY (src/rul.py, src/state_estimators.py)

9. Monte Carlo RUL simulation: in fit_gpr_rul(), after fitting the GPR, draw
   n_samples=500 trajectories by sampling from the posterior:
   soh_samples = np.random.normal(soh_pred, soh_std, size=(500, len(future_cycles)))
   For each sample find the EOL crossing. Store rul_mc_p5, rul_mc_p25,
   rul_mc_p75, rul_mc_p95 per discharge cycle in addition to existing p10/p90.

10. Uncertainty widening with age: multiply soh_std by (1 / soh_pred.clip(0.1, 1.0))
    before computing soh_model_lower/upper in estimate_soh_regression(). This widens
    the confidence band as SOH degrades toward EOL.

11. Calibration check: add compute_soh_calibration(cycle_df) to a new
    src/calibration.py module. For each decile of predicted SOH, compute the
    fraction of observed SOH values that fall inside the 90% CI. Return a
    calibration_df with columns [decile, coverage, expected_coverage].
    Export as calibration_{battery_id}.parquet from pipeline.py.

---

GROUP 5 — DASHBOARD (app.py)

12. Maintenance decision panel: add a coloured action card at the top of the
    Overview tab (above the KPI grid) that calls get_charge_recommendation() and
    displays the action with matching colour (green=normal, yellow=reduce,
    orange=inspect, red=replace) and the reason string.

13. What-if panel: in the Battery Health tab, add two st.slider controls:
    - Operating temperature offset: -10°C to +20°C
    - Charge C-rate multiplier: 0.5x to 2.0x
    Rerun add_rul_estimates with modified stress_coeffs derived from the sliders
    and overlay the adjusted RUL projection on the existing RUL chart as a dashed line.

14. Operating regime chart: in the Diagnostics tab, add a scatter plot of
    cycle_index vs soh coloured by operating_regime. Add a second chart showing
    per-regime mean SOH degradation rate as a bar chart.

15. Calibration plot: in the Diagnostics tab, add a reliability diagram showing
    actual coverage vs expected coverage from calibration_{battery_id}.parquet.
    A perfectly calibrated model sits on the diagonal y=x line.

---

GROUP 6 — INFRASTRUCTURE (tests/, requirements.txt)

16. Unit tests: create tests/ directory with pytest fixtures. Write tests for:
    - test_lag_features_no_bleed: verify soh_lag_1 never bleeds across battery_id boundaries
    - test_gpr_rul_index_alignment: verify rul_cycles_gpr has same length as discharge cycles
    - test_validate_tables_drops_invalid: verify out-of-range voltage rows are dropped
    - test_sop_positive: verify sop_w > 0 for all discharge cycles
    - test_calibration_coverage: verify calibration_df coverage values are in [0, 1]
    Use small synthetic DataFrames (5 batteries, 20 cycles each) as fixtures.

17. Update requirements.txt: add kneed, pytest, scipy (currently missing).

---

CONSTRAINTS:
- Do not rename any existing columns that app.py already references
- All file writes must use encoding='utf-8'
- All new columns must be NaN-safe (use .fillna() or nullable checks before operations)
- New artifacts must be registered in the manifest dict in pipeline.py
- After all groups are done, run the full pipeline: python scripts/prepare_dashboard_data.py and confirm it completes without errors before touching app.py