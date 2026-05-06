import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Fix Plotly deprecation warning (revert width='stretch' to use_container_width=True for st.plotly_chart)
code = code.replace("st.plotly_chart(soh_compare_fig, width='stretch')", "st.plotly_chart(soh_compare_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(rul_compare_fig, width='stretch')", "st.plotly_chart(rul_compare_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(error_compare_fig, width='stretch')", "st.plotly_chart(error_compare_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(eol_fig, width='stretch')", "st.plotly_chart(eol_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(summary_fig, width='stretch')", "st.plotly_chart(summary_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(rul_fig, width='stretch')", "st.plotly_chart(rul_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(ocv_fig, width='stretch')", "st.plotly_chart(ocv_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(voltage_fig, width='stretch')", "st.plotly_chart(voltage_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(soc_fig, width='stretch')", "st.plotly_chart(soc_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(thermo_fig, width='stretch')", "st.plotly_chart(thermo_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(soc_anchor_fig, width='stretch')", "st.plotly_chart(soc_anchor_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(dynamic_resistance_fig, width='stretch')", "st.plotly_chart(dynamic_resistance_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(dynamic_capacitance_fig, width='stretch')", "st.plotly_chart(dynamic_capacitance_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(efficiency_fig, width='stretch')", "st.plotly_chart(efficiency_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(imp_fig, width='stretch')", "st.plotly_chart(imp_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(r0_fig, width='stretch')", "st.plotly_chart(r0_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(eis_fig, width='stretch')", "st.plotly_chart(eis_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(soh_imp_fig, width='stretch')", "st.plotly_chart(soh_imp_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(cycle_voltage_fig, width='stretch')", "st.plotly_chart(cycle_voltage_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(cycle_soc_fig, width='stretch')", "st.plotly_chart(cycle_soc_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(cycle_thermal_fig, width='stretch')", "st.plotly_chart(cycle_thermal_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(residual_fig, width='stretch')", "st.plotly_chart(residual_fig, use_container_width=True)")
code = code.replace("st.plotly_chart(parameter_signal_fig, width='stretch')", "st.plotly_chart(parameter_signal_fig, use_container_width=True)")
code = code.replace("width='stretch'", "use_container_width=True") # Fallback to cover everything that might have been missed

# 2. Fix raw HTML appearing in KPI cards (remove indentation that makes it a markdown code block)
kpi_loop_old = '''    grid_html = \'<div class="kpi-grid">\'
    for card in cards:
        grid_html += f"""
        <div class="kpi-card" style="--accent: {card['accent']};">
            <div class="kpi-label">{card['label']}</div>
            <div class="kpi-value">{card['value']}</div>
            <div class="kpi-subtext">{card['subtext']}</div>
        </div>
        """
    grid_html += \'</div>\''''

kpi_loop_new = '''    grid_html = \'<div class="kpi-grid">\'
    for card in cards:
        grid_html += f"""<div class="kpi-card" style="--accent: {card['accent']};">
<div class="kpi-label">{card['label']}</div>
<div class="kpi-value">{card['value']}</div>
<div class="kpi-subtext">{card['subtext']}</div>
</div>"""
    grid_html += \'</div>\''''
code = code.replace(kpi_loop_old, kpi_loop_new)

# 3. Clean overlapping plots and noisy ECM data
# In app.py, where `detail_shadow` is used, we need to sort values by time_s and reset_index.
# And we apply rolling mean to smooth SOC and voltage lines.
detail_shadow_process = '''    detail_shadow = get_battery_sample_shadow(
        artifact_dir,
        battery_id=selected_battery,
        start_cycle=cycle_range[0],
        end_cycle=cycle_range[1],
    )'''
detail_shadow_new = '''    detail_shadow = get_battery_sample_shadow(
        artifact_dir,
        battery_id=selected_battery,
        start_cycle=cycle_range[0],
        end_cycle=cycle_range[1],
    )
    if not detail_shadow.empty:
        detail_shadow = detail_shadow.sort_values("time_s").reset_index(drop=True)
        # Downsample if too large to prevent overlap
        if len(detail_shadow) > 5000:
            detail_shadow = detail_shadow.iloc[::max(1, len(detail_shadow)//5000)]
        # Smooth noisy sensor data
        for col in ["voltage_v", "voltage_model_v", "voltage_ekf_v", "current_a", "temperature_c", "soc", "soc_ekf"]:
            if col in detail_shadow.columns:
                detail_shadow[col] = detail_shadow[col].rolling(window=10, min_periods=1).mean()'''
code = code.replace(detail_shadow_process, detail_shadow_new)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
