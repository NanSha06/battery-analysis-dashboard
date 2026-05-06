# FIX PROMPT — Correct ECM R0 Scaling & EIS Validation Errors

You are a senior battery modeling engineer specializing in ECM fitting, EIS validation, and physics-informed battery analytics.

The current Li-ion Digital Shadow system already implements:

* ECM parameter extraction
* impedance validation
* EIS real-part comparison
* Streamlit dashboard visualization

However, the current dashboard shows the following issue:

```text id="94p59t"
Correlation between ECM R0 and EIS Re(Z) is high (~0.96),
but ECM R0 magnitude is nearly zero while measured
EIS resistance is around 0.045–0.062 Ω.
```

This indicates:

* strong trend alignment
* incorrect physical scaling
* normalization/unit mismatch
* improperly denormalized ECM outputs

The goal is to fix the ECM R0 estimation so that:

* predicted R0 values are physically realistic
* magnitudes align with EIS-derived resistance
* RMSE/MAE/drift improve significantly
* dashboard plots overlap correctly

---

1. ROOT CAUSE ANALYSIS

---

Investigate all stages where R0 may be scaled incorrectly.

Check:

* feature normalization
* voltage scaling
* current scaling
* unit conversions
* denormalization logic
* ECM optimization output
* artifact serialization

Specifically verify:

* whether voltage/current are normalized before ECM fitting
* whether R0 is stored in normalized feature space
* whether inverse transforms are missing

---

2. FIX ECM R0 COMPUTATION

---

Ensure ECM computes physical ohmic resistance using:

R_0 = \frac{\Delta V}{\Delta I}

Requirements:

* use REAL volts
* use REAL amps
* do NOT use normalized features
* preserve units in ohms

Add safeguards:

* divide-by-zero handling
* transient noise filtering
* smoothing support

---

3. FIX EIS VALIDATION PIPELINE

---

Ensure EIS-derived reference resistance is computed correctly.

Use:

R_0 \approx \Re(Z_{high\ frequency})

Implementation requirements:

* extract ONLY real impedance
* select highest-frequency region
* average top-k high-frequency points
* reject outliers
* preserve Ω units

Correct implementation:

```python id="it9k9m"
top_k = np.argsort(freqs)[-5:]
r0_ref = np.mean(np.real(Z_complex[top_k]))
```

---

4. IMPLEMENT AUTOMATIC SCALE ALIGNMENT

---

If legacy ECM outputs remain normalized, automatically align scales before validation.

Add:

```python id="9a5pwa"
scale_factor = np.mean(r0_ref_series) / np.mean(r0_pred_series)

r0_pred_series = r0_pred_series * scale_factor
```

Requirements:

* avoid divide-by-zero
* handle NaNs safely
* log scale factor
* persist aligned outputs

---

5. ADD UNIT VALIDATION CHECKS

---

Implement assertions to verify:

* voltage units are volts
* current units are amps
* impedance units are ohms

Add warnings if:

* R0 < 0
* R0 > 1 Ω
* impedance magnitude unrealistic
* scaling drift exceeds threshold

---

6. IMPROVE VALIDATION METRICS

---

Compute:

* RMSE
* MAE
* Pearson correlation
* percentage drift
* scale mismatch factor

Display before/after comparison.

Expected improvements:

* MAE decreases substantially
* RMSE decreases substantially
* drift approaches realistic values
* visual overlap improves

---

7. FIX DASHBOARD VISUALIZATION

---

Update dashboard plots so:

* ECM R0 and EIS Re(Z) share same y-scale
* normalized traces are removed
* legends clearly distinguish:

  * raw ECM
  * scaled ECM
  * measured EIS Re(Z)

Add:

* unit labels (Ω)
* scaling diagnostics
* alignment status badges

---

8. ADD DEBUGGING PANEL

---

Create expandable section:

# “ECM Scaling Diagnostics”

Display:

* mean predicted R0
* mean EIS R0
* scale factor
* normalization detected?
* unit consistency
* outlier counts

---

9. PERSIST FIXED OUTPUTS

---

Export:

* aligned_r0.parquet
* eis_reference.parquet
* scaling_metrics.json

Ensure artifacts store:

* physical resistance values
* non-normalized outputs
* validated Ω units

---

10. EXPECTED FINAL RESULT

---

After fixes:

* ECM R0 magnitude should align with EIS Re(Z)
* validation plots should visually overlap
* MAE/RMSE should improve dramatically
* drift should become physically meaningful
* dashboard should show realistic resistance evolution

The final system should become:

```text id="4ll5o9"
physics-calibrated ECM validation platform
```

instead of:

* trend-only matching
* normalized-space comparison
* visually misleading validation

```
```
