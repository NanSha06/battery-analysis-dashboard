import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add get_adaptive_ecm_params(frame)
# Let's add it before sanitize_ecm_impedance_params
new_func = """def get_adaptive_ecm_params(frame: pd.DataFrame) -> dict[str, float]:
    \"\"\"Extract adaptive ECM parameters cleanly from the dataframe medians.\"\"\"
    return {
        "r0": _finite_median(frame, "r0") or 0.01,
        "r1": _finite_median(frame, "r1") or 0.01,
        "r2": _finite_median(frame, "r2") or 0.02,
        "c1": _finite_median(frame, "c1") or 2000.0,
        "c2": _finite_median(frame, "c2") or 4000.0,
    }

def sanitize_ecm_impedance_params("""
content = content.replace("def sanitize_ecm_impedance_params(", new_func)

# 2. Fix Nyquist/Bode rendering in build_impedance_validation
# It currently uses:
# clean_params = { "r0": _finite_median(cycle_frame, "r0") or params.get("r0", 0.01), ... }
# We will use get_adaptive_ecm_params!
biv_start = content.find("def build_impedance_validation")
biv_end = content.find("def add_physics_features")

old_biv = content[biv_start:biv_end]

new_biv = """def build_impedance_validation(cycle_frame: pd.DataFrame, params: dict[str, float]) -> tuple[dict[str, object], go.Figure, go.Figure, pd.DataFrame]:
    frequencies = np.geomspace(0.01, 10000.0, 300)
    
    # Build clean_params directly from adaptive frame medians
    clean_params = get_adaptive_ecm_params(cycle_frame)
    clean_params["warburg_aw"] = params.get("warburg_aw", 0.015)
    
    clean_params, param_warnings = sanitize_ecm_impedance_params(clean_params, cycle_frame)
    z_model = ecm_impedance_response(clean_params, frequencies)
    z_real = np.real(z_model)
    z_imag = np.imag(z_model)
    z_mag = np.sqrt(z_real**2 + z_imag**2)
    z_phase = np.degrees(np.arctan2(z_imag, z_real))
    eis_frame = cycle_frame.dropna(subset=["re_ohm", "rct_ohm"]).copy() if {"re_ohm", "rct_ohm"}.issubset(cycle_frame.columns) else pd.DataFrame()

    nyquist = go.Figure()
    nyquist.add_trace(go.Scatter(x=z_real, y=-z_imag, mode="lines", name="ECM 2RC (Adaptive)", line=dict(color="#58a6ff", width=3)))
    if not eis_frame.empty:
        nyquist.add_trace(go.Scatter(x=eis_frame["re_ohm"], y=np.zeros(len(eis_frame)), mode="markers", name="Experimental EIS Re", marker=dict(color="#f2cc60", size=8)))
        nyquist.add_trace(go.Scatter(x=eis_frame["re_ohm"] + eis_frame["rct_ohm"], y=eis_frame["rct_ohm"] / 2.0, mode="markers", name="Experimental EIS Arc", marker=dict(color="#ff7b72", size=8)))
    
    # Add annotations for high/low freq
    nyquist.add_annotation(x=z_real[-1], y=-z_imag[-1], text="High Freq", showarrow=True, arrowhead=2, ax=20, ay=-30, font=dict(color="#a5d6ff"))
    nyquist.add_annotation(x=z_real[0], y=-z_imag[0], text="Low Freq", showarrow=True, arrowhead=2, ax=-20, ay=-30, font=dict(color="#a5d6ff"))
    nyquist.update_layout(title="Nyquist: Experimental EIS vs Adaptive ECM", xaxis_title="Z real (Ohm)", yaxis_title="-Z imag (Ohm)", height=480, plot_bgcolor='rgba(13,17,23,1)', paper_bgcolor='rgba(13,17,23,1)', font=dict(color='#c9d1d9'))

    bode = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, subplot_titles=("Magnitude", "Phase"))
    bode.add_trace(go.Scatter(x=frequencies, y=z_mag, mode="lines", name="ECM |Z|", line=dict(color="#58a6ff", width=3)), row=1, col=1)
    bode.add_trace(go.Scatter(x=frequencies, y=z_phase, mode="lines", name="ECM Phase", line=dict(color="#7ee787", width=3)), row=2, col=1)
    
    exp_mag = np.array([], dtype=float)
    exp_phase = np.array([], dtype=float)
    if not eis_frame.empty:
        exp_mag = np.sqrt((eis_frame["re_ohm"] + eis_frame["rct_ohm"]) ** 2 + (eis_frame["rct_ohm"] / 2.0) ** 2).to_numpy(dtype=float)
        exp_phase = -np.degrees(np.arctan2((eis_frame["rct_ohm"] / 2.0).to_numpy(dtype=float), (eis_frame["re_ohm"] + eis_frame["rct_ohm"]).to_numpy(dtype=float)))
        f_exp = np.geomspace(frequencies.min(), frequencies.max(), len(eis_frame))
        bode.add_trace(go.Scatter(x=f_exp, y=exp_mag, mode="markers", name="Experimental |Z|", marker=dict(color="#f2cc60", size=8)), row=1, col=1)
        bode.add_trace(go.Scatter(x=f_exp, y=exp_phase, mode="markers", name="Experimental Phase", marker=dict(color="#ff7b72", size=8)), row=2, col=1)
    
    bode.update_xaxes(type="log", title_text="Frequency (Hz)", row=2, col=1, gridcolor='rgba(48,54,61,1)')
    bode.update_yaxes(title_text="|Z| (Ohm)", row=1, col=1, gridcolor='rgba(48,54,61,1)')
    bode.update_yaxes(title_text="Phase (deg)", row=2, col=1, gridcolor='rgba(48,54,61,1)')
    bode.update_layout(title="Bode: Adaptive Magnitude and Phase", height=580, hovermode="x unified", plot_bgcolor='rgba(13,17,23,1)', paper_bgcolor='rgba(13,17,23,1)', font=dict(color='#c9d1d9'))

    diagnostics = pd.DataFrame(
        {
            "frequency_hz": frequencies,
            "re_z_ohm": z_real,
            "im_z_ohm": z_imag,
            "abs_z_ohm": z_mag,
            "phase_deg": z_phase,
        }
    )

    high_freq_intercept = float(z_real[-1])
    low_freq_impedance = float(z_mag[0])
    min_magnitude = float(np.nanmin(z_mag))
    warnings = list(dict.fromkeys(param_warnings))
    
    # We remove verbose string warnings, replacing them with simple status keys if needed.
    # The user asked to remove verbose warnings and use compact status badges in UI.
    # So we'll just keep metrics and let UI handle them.
    insights = []
    
    if eis_frame.empty:
        metrics: dict[str, object] = {"impedance_rmse": np.nan, "phase_rmse_deg": np.nan}
    else:
        f_exp = np.geomspace(frequencies.min(), frequencies.max(), len(exp_mag))
        model_mag = np.interp(np.log10(f_exp), np.log10(frequencies), z_mag)
        model_phase = np.interp(np.log10(f_exp), np.log10(frequencies), z_phase)
        impedance_rmse = float(np.sqrt(np.mean((model_mag - exp_mag) ** 2)))
        phase_rmse = float(np.sqrt(np.mean((model_phase - exp_phase) ** 2)))
        metrics = {
            "impedance_rmse": impedance_rmse,
            "phase_rmse_deg": phase_rmse,
            "phase_error_deg": phase_rmse,
        }

    total_r = clean_params["r0"] + clean_params["r1"] + clean_params["r2"]
    clean_params.setdefault("warburg_aw", float(np.clip(total_r * 0.18, 0.005, 0.15)))
    metrics.update(
        {
            "r0_ohm": clean_params["r0"],
            "r1_ohm": clean_params["r1"],
            "r2_ohm": clean_params["r2"],
            "c1_f": clean_params["c1"],
            "c2_f": clean_params["c2"],
            "warburg_aw": clean_params.get("warburg_aw", 0.0),
            "tau1_s": clean_params["r1"] * clean_params["c1"],
            "tau2_s": clean_params["r2"] * clean_params["c2"],
            "high_freq_intercept_ohm": high_freq_intercept,
            "low_freq_impedance_ohm": low_freq_impedance,
            "min_magnitude_ohm": min_magnitude,
            "total_dc_resistance_ohm": total_r,
        }
    )
    imp_col = (
        "estimated_impedance_smoothed_ohm"
        if "estimated_impedance_smoothed_ohm" in cycle_frame.columns
        else "estimated_impedance_ohm"
    )
    exp_imp_med = _finite_median(cycle_frame, imp_col)
    
    if exp_imp_med is not None and exp_imp_med > 0:
        metrics["impedance_scaling_error"] = float(abs(total_r - exp_imp_med) / exp_imp_med * 100.0)
        metrics["r0_tracking_error"] = float(abs(clean_params["r0"] - exp_imp_med) / exp_imp_med * 100.0)
    else:
        metrics["impedance_scaling_error"] = np.nan
        metrics["r0_tracking_error"] = np.nan
    return metrics, nyquist, bode, diagnostics

"""

content = content.replace(old_biv, new_biv)

# 3. Remove add_physics_features entirely, and rewrite build_physics_summary
apf_start = content.find("def add_physics_features")
apf_end = content.find("def render_maintenance_panel")

old_apf = content[apf_start:apf_end]

new_apf = """def build_physics_summary(physics_shadow: pd.DataFrame) -> pd.DataFrame:
    if physics_shadow.empty:
        return pd.DataFrame()

    columns = [
        "cycle_index",
        "cycle_type",
        "r0",
        "r1",
        "r2",
        "c1",
        "c2",
    ]
    avail_cols = [c for c in columns if c in physics_shadow.columns]
    return (
        physics_shadow[avail_cols]
        .groupby(["cycle_index", "cycle_type"], as_index=False)
        .mean(numeric_only=True)
    )

"""
content = content.replace(old_apf, new_apf)

# 4. Remove `physics_shadow = add_physics_features(detail_shadow, ocv_curve, selected_ecm_params)`
# Replace with `physics_shadow = detail_shadow.copy()`
content = content.replace("physics_shadow = add_physics_features(detail_shadow, ocv_curve, selected_ecm_params)", "physics_shadow = detail_shadow.copy()")

# 5. Fix tabs names
tabs_old = '''    overview_tab, health_tab, ecm_tab, electro_tab, diagnostics_tab, data_tab = st.tabs([
        "Overview",
        "Battery Health",
        "ECM & Impedance",
        "Electrochemical Insights",
        "Diagnostics",
        "Data Explorer"
    ])'''
tabs_new = '''    overview_tab, ecm_tab, soc_tab, impedance_tab, health_tab, diagnostics_tab, data_tab = st.tabs([
        "Fleet Overview",
        "ECM Validation",
        "SOC & EKF",
        "Impedance & EIS",
        "Aging & SOH",
        "Diagnostics",
        "Research Metrics"
    ])'''
content = content.replace(tabs_old, tabs_new)

# Tab mappings
tab_map_old = '''    comparison_tab = overview_tab
    eol_tab = overview_tab
    soh_tab = health_tab
    rul_tab = health_tab
    impedance_tab = ecm_tab
    cycle_detail_tab = diagnostics_tab
    residual_tab = diagnostics_tab
    parameters_tab = ecm_tab
    ocv_tab = electro_tab
    voltage_tab = electro_tab
    soc_tab = electro_tab
    thermal_tab = electro_tab
    physics_tab = electro_tab'''

tab_map_new = '''    comparison_tab = overview_tab
    eol_tab = health_tab
    soh_tab = health_tab
    rul_tab = health_tab
    cycle_detail_tab = diagnostics_tab
    residual_tab = diagnostics_tab
    parameters_tab = ecm_tab
    ocv_tab = soc_tab
    voltage_tab = ecm_tab
    thermal_tab = diagnostics_tab
    physics_tab = data_tab'''
content = content.replace(tab_map_old, tab_map_new)

# 6. Global Validation Summary (at the bottom or top of Overview)
# We can put it in overview_tab just after the insight summary.

global_summary_code = """
        st.markdown("### Global Validation Summary")
        val_rmse_v = ecm_metrics.get("mean_rmse_v", np.nan)
        val_ekf_v = ecm_metrics.get("mean_ekf_rmse_v", np.nan)
        
        rating = "Moderate"
        color = "#d29922"
        if val_rmse_v < 0.02 and val_ekf_v < 0.02:
            rating = "Excellent"
            color = "#3fb950"
        elif val_rmse_v < 0.05 and val_ekf_v < 0.05:
            rating = "Good"
            color = "#58a6ff"
        elif val_rmse_v > 0.1 or val_ekf_v > 0.1:
            rating = "Weak"
            color = "#f85149"
            
        st.markdown(f'''
        <div style="background: rgba(30,34,42,0.4); border-left: 4px solid {color}; padding: 16px; border-radius: 8px; margin-bottom: 24px; font-size: 1.05rem;">
            <strong>System Rating:</strong> {rating}
        </div>
        ''', unsafe_allow_html=True)
        
        gv_cols = st.columns(4)
        gv_cols[0].metric("Voltage RMSE", format_kpi_value(val_rmse_v, suffix=" V", digits=4))
        gv_cols[1].metric("EKF RMSE", format_kpi_value(val_ekf_v, suffix=" V", digits=4))
        gv_cols[2].metric("Impedance RMSE", format_kpi_value(global_data.get("impedance_metrics", {}).get(selected_battery, {}).get("impedance_rmse", np.nan), suffix=" Ω", digits=4))
        gv_cols[3].metric("Phase RMSE", format_kpi_value(global_data.get("impedance_metrics", {}).get(selected_battery, {}).get("phase_rmse_deg", np.nan), suffix=" °", digits=2))
"""
# insert inside `with comparison_tab:`
content = content.replace('        st.markdown("### AI Insight Summary")', global_summary_code + '\n        st.markdown("### AI Insight Summary")')

# 7. Diagnostic Panels inside impedance_tab or diagnostics_tab
# The user wants "A. Voltage Validation, B. Impedance Validation, C. Adaptive ECM Diagnostics, D. Electrochemical Insights" as collapsible sections.
# Let's replace the whole `with impedance_tab:`
imp_tab_start = content.find("    with impedance_tab:")
imp_tab_end = content.find("    with cycle_detail_tab:")

old_imp_tab = content[imp_tab_start:imp_tab_end]

new_imp_tab = """    with impedance_tab:
        st.markdown("### Impedance & EIS Validation")
        r0_val = global_data.get("r0_validation", {}).get(selected_battery, {})
        imp_met = global_data.get("impedance_metrics", {}).get(selected_battery, {})
        imp_curve = global_data.get("impedance_curve", pd.DataFrame())
        imp_curve_batt = imp_curve[imp_curve["battery_id"] == selected_battery].copy() if not imp_curve.empty else pd.DataFrame()
        
        imp_validation, nyquist_fig, bode_fig, bode_diagnostics = build_impedance_validation(detail_cycle_shadow, selected_ecm_params)
        
        with st.expander("B. Impedance Validation", expanded=True):
            val_cols = st.columns(4)
            val_cols[0].metric("Impedance RMSE", format_kpi_value(imp_validation.get("impedance_rmse"), suffix=" Ω", digits=4))
            val_cols[1].metric("Phase RMSE", format_kpi_value(imp_validation.get("phase_rmse_deg"), suffix=" °", digits=2))
            val_cols[2].metric("Scaling Error", format_kpi_value(imp_validation.get("impedance_scaling_error"), suffix=" %", digits=2))
            val_cols[3].metric("R0 Tracking Error", format_kpi_value(imp_validation.get("r0_tracking_error"), suffix=" %", digits=2))
            
            st.plotly_chart(nyquist_fig, use_container_width=True)
            st.plotly_chart(bode_fig, use_container_width=True)

        with st.expander("C. Adaptive ECM Diagnostics", expanded=True):
            if not imp_curve_batt.empty:
                st.markdown("#### Adaptive R0 vs Estimated Impedance")
                imp_col = (
                    "estimated_impedance_smoothed_ohm"
                    if "estimated_impedance_smoothed_ohm" in imp_curve_batt.columns
                    else "estimated_impedance_ohm"
                )
                
                # Apply smoothing to adaptive R0 for the plot
                imp_curve_batt["r0_smooth"] = imp_curve_batt["r0"].rolling(window=3, min_periods=1).mean()
                
                r0_fig = go.Figure()
                r0_fig.add_trace(go.Scatter(x=imp_curve_batt["cycle_index"], y=imp_curve_batt["r0_smooth"], mode="lines", name="Adaptive R0 (Smoothed)", line=dict(color="#58a6ff", width=3)))
                r0_fig.add_trace(go.Scatter(x=imp_curve_batt["cycle_index"], y=imp_curve_batt[imp_col], mode="lines", name="Transient Impedance (Smoothed)", line=dict(color="#f85149", width=2, dash="dash")))
                
                # Correlation
                valid = imp_curve_batt.dropna(subset=["r0_smooth", imp_col])
                if not valid.empty and len(valid) > 2:
                    corr = np.corrcoef(valid["r0_smooth"], valid[imp_col])[0, 1]
                    r0_fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text=f"Correlation: {corr:.2f}", showarrow=False, font=dict(color="#c9d1d9", size=14), bgcolor="rgba(13,17,23,0.8)")

                r0_fig.update_layout(title="ECM R0 vs Transient Impedance", xaxis_title="Cycle", yaxis_title="Resistance (Ω)", height=450, plot_bgcolor='rgba(13,17,23,1)', paper_bgcolor='rgba(13,17,23,1)', font=dict(color='#c9d1d9'))
                st.plotly_chart(r0_fig, use_container_width=True)

"""
content = content.replace(old_imp_tab, new_imp_tab)

# 8. Add A. Voltage Validation to `voltage_tab`
# Find `with voltage_tab:`
volt_tab_start = content.find("    with voltage_tab:")
volt_tab_end = content.find("    with soc_tab:")

old_volt_tab = content[volt_tab_start:volt_tab_end]

new_volt_tab = """    with voltage_tab:
        if detail_shadow.empty:
            st.info("No detail samples were found for the selected cycle range.")
        else:
            with st.expander("A. Voltage Validation", expanded=True):
                render_validation_summary(validation_metrics["voltage"], "Voltage Validation", "V")
                voltage_fig = px.line(
                    detail_shadow,
                    x="time_s",
                    y=["voltage_v", "voltage_model_v", "voltage_ekf_v"],
                    title=f"Measured vs Adaptive ECM/EKF Voltage: {selected_battery}",
                )
                st.plotly_chart(voltage_fig, use_container_width=True)
                residual_line, residual_hist, residual_box = build_voltage_validation_plots(detail_shadow)
                st.plotly_chart(residual_line, use_container_width=True)
                residual_cols = st.columns(2)
                with residual_cols[0]:
                    st.plotly_chart(residual_hist, use_container_width=True)
                with residual_cols[1]:
                    st.plotly_chart(residual_box, use_container_width=True)

"""
content = content.replace(old_volt_tab, new_volt_tab)

# 9. Clean up Physics / Data tab
# Replaced `physics_tab = electro_tab` with `physics_tab = data_tab`
# The physics_tab now uses `r0`, `r1` instead of `r0_dynamic`, `r1_dynamic`.
phys_tab_start = content.find("    with physics_tab:")
phys_tab_end = content.find("    with impedance_tab:")

old_phys_tab = content[phys_tab_start:phys_tab_end]

new_phys_tab = """    with physics_tab:
        if physics_shadow.empty:
            st.info("No detail samples were found for the selected cycle range.")
        else:
            with st.expander("D. Electrochemical Insights", expanded=True):
                st.markdown("#### Adaptive ECM Parameters History")
                dynamic_cols = st.columns(2)
                with dynamic_cols[0]:
                    cols_to_plot = [c for c in ["r0", "r1", "r2"] if c in physics_shadow.columns]
                    dynamic_resistance_fig = px.line(
                        physics_shadow,
                        x="time_s",
                        y=cols_to_plot,
                        title="State-dependent Resistance Parameters",
                        labels={"time_s": "Time (s)", "value": "Ohm", "variable": "Parameter"},
                    )
                    dynamic_resistance_fig.update_layout(height=380, hovermode="x unified")
                    st.plotly_chart(dynamic_resistance_fig, use_container_width=True)
                with dynamic_cols[1]:
                    cols_to_plot = [c for c in ["c1", "c2"] if c in physics_shadow.columns]
                    dynamic_capacitance_fig = px.line(
                        physics_shadow,
                        x="time_s",
                        y=cols_to_plot,
                        title="State-dependent Capacitance Parameters",
                        labels={"time_s": "Time (s)", "value": "Farad", "variable": "Parameter"},
                    )
                    dynamic_capacitance_fig.update_layout(height=380, hovermode="x unified")
                    st.plotly_chart(dynamic_capacitance_fig, use_container_width=True)

            st.markdown("#### Physics Feature Summary")
            st.dataframe(build_physics_summary(physics_shadow), use_container_width=True)

"""
content = content.replace(old_phys_tab, new_phys_tab)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
