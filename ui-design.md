# UI REDESIGN PROMPT — Li-ion Digital Shadow Dashboard

Redesign the existing Streamlit-based Li-ion Digital Shadow dashboard into a modern, clean, enterprise-grade battery intelligence interface.

The current dashboard contains:

* SOH/RUL analytics
* ECM modeling
* impedance validation
* EIS comparison
* degradation trends
* anomaly detection
* battery lifecycle analytics

The new UI should feel like a combination of:

* Tesla battery telemetry
* Grafana
* Apple Health
* industrial AI analytics dashboards

while maintaining scientific and research-oriented functionality.

---

## CORE UI GOALS

The dashboard should become:

* cleaner
* more modular
* insight-driven
* less cluttered
* visually balanced
* easier to navigate

Focus on:

* information hierarchy
* spacing
* readability
* responsive layouts
* minimalism
* professional dark theme

---

## LAYOUT REDESIGN

Replace the long scrolling layout with modular sections and tabs.

Create tabs such as:

* Overview
* Battery Health
* ECM & Impedance
* Electrochemical Insights
* Diagnostics
* Data Explorer

Use:

* `st.tabs()`
* `st.columns()`
* `st.container()`

Avoid stacking too many charts vertically.

---

## TOP KPI SECTION

Create a premium KPI summary row for:

* SOH
* RUL
* SOC
* Cycles
* Health Status

Use:

* modern metric cards
* icons
* subtle gradients
* rounded containers
* status badges
* tiny trend indicators

Example style:

* healthy → green
* aging → yellow
* critical → red

---

## CHART DESIGN IMPROVEMENTS

Improve all visualizations by:

* reducing visual clutter
* softening gridlines
* increasing whitespace
* improving axis readability
* using consistent color palettes
* increasing line clarity

Recommended color system:

* SOH → green
* RUL → orange
* ECM → blue
* Impedance → cyan
* Anomalies → red

Charts should look modern and minimal, not default Plotly outputs.

---

## DASHBOARD ORGANIZATION

Prioritize insights in this order:

1. Battery Health
2. Degradation Trends
3. Impedance & ECM Validation
4. Diagnostics & Anomalies
5. Raw Technical Details

Move technical/debugging information into:

* expanders
* collapsible panels
* optional diagnostics sections

---

## SIDEBAR REDESIGN

Use the sidebar for:

* battery selection
* cycle range
* smoothing options
* anomaly sensitivity
* chart toggles
* export/download options

Keep the main view focused on insights.

---

## INSIGHT PANELS

Add AI-style insight summaries such as:

* “Impedance increased by 18% over lifecycle”
* “Strong ECM–EIS correlation detected”
* “SOH degradation accelerated after cycle 400”
* “No critical anomalies detected”

These should appear as clean cards or alert panels.

---

## ECM & IMPEDANCE SECTION

Redesign impedance validation visuals into a cleaner grid layout.

Use:

* side-by-side plots
* aligned scales
* compact validation cards
* clear legends
* hover tooltips

Add concise scientific explanations for:

* impedance
* R0
* EIS validation
* degradation behavior

---

## THEME & STYLING

Use:

* deep charcoal/slate backgrounds
* soft glowing accents
* glassmorphism/light shadows
* rounded borders
* clean typography
* subtle animations if possible

Avoid:

* pure black backgrounds
* dense text blocks
* excessive labels
* oversized legends

---

## USER EXPERIENCE

The dashboard should tell a story:

Overview
→ Health degradation
→ Electrochemical behavior
→ ECM validation
→ Predictive maintenance insights

The interface should feel like an:

* AI-powered battery intelligence platform
* digital twin monitoring system
* industrial predictive analytics dashboard

not just a collection of charts.

---

## FINAL RESULT

The final UI should be:

* elegant
* futuristic
* readable
* modular
* research-grade
* industry-inspired
* suitable for demos, publications, and portfolio showcases

while preserving all existing analytical functionality.

```
```
