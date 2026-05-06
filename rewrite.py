import re
with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

tabs_pattern = r'\(\s*comparison_tab,[\s\S]*?\]\s*\)'

new_tabs = '''overview_tab, health_tab, ecm_tab, electro_tab, diagnostics_tab, data_tab = st.tabs([
        "Overview",
        "Battery Health",
        "ECM & Impedance",
        "Electrochemical Insights",
        "Diagnostics",
        "Data Explorer"
    ])
    
    comparison_tab = overview_tab
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

code = re.sub(tabs_pattern, new_tabs, code, count=1)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
