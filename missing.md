Enhance the existing Li-ion Battery Digital Shadow Streamlit dashboard by adding missing research-grade ECM validation features while keeping the current UI/theme unchanged.

Add the following:

1. Voltage Validation
- Display RMSE, MAE, Max Error, Residual Mean, and R² for ECM voltage prediction
- Add residual plots:
  - residual vs time
  - residual histogram
  - residual boxplot

2. SOC Validation
- Add SOC RMSE and SOC drift metrics
- Add SOC residual plot

3. Impedance Validation
- Add Nyquist plot (Experimental EIS vs ECM)
- Add Bode magnitude and phase plots
- Compute impedance RMSE and phase error

4. Dynamic ECM Diagnostics
- Compute and plot ECM time constants:
  tau1 = R1*C1
  tau2 = R2*C2
- Add parameter drift analysis

5. Temperature Validation
- Add RMSE vs temperature plot
- Plot resistance vs temperature

6. Uncertainty Visualization
- Add EKF covariance/confidence interval visualization
- Add prediction uncertainty bands

7. Statistical Summary
- Create a validation summary panel showing:
  RMSE
  MAE
  Max Error
  Correlation
  R²
  Drift %

8. AI Insights
- Add automatic interpretation text explaining whether validation quality is:
  Excellent / Good / Weak

Use Plotly interactive charts and ensure all metrics update dynamically per selected battery.