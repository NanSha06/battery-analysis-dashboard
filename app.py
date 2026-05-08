from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.templates.default = "plotly_dark"
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dashboard_data import (  # noqa: E402
    available_battery_ids,
    load_battery_table,
    load_battery_ecm_params,
    load_battery_metrics,
    load_ecm_params,
    load_global_tables,
    load_manifest,
    load_metrics,
    summarize_battery,
)
from src.ecm import ECMParameters, get_dynamic_params  # noqa: E402
from src.features import compute_cycle_efficiency, compute_efficiency_trends  # noqa: E402
from src.state_estimators import apply_soc_anchor  # noqa: E402
from src.recommendations import get_charge_recommendation # noqa: E402
from src.rul import add_rul_estimates # noqa: E402


st.set_page_config(page_title="Li-ion Digital Shadow", layout="wide")


NASA_EOL_CAPACITY_AH = 1.4
NASA_EOL_SOH = 0.70
WARNING_SOH = 0.80
HEALTHY_SOH = 0.85



CARD_CSS = '''
<style>
/* Dashboard Redesign CSS */
.stApp {
    background-color: #0e1117;
    color: #c9d1d9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* KPI Cards styling with Glassmorphism */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.2rem;
    margin: 1rem 0 2rem 0;
}
.kpi-card {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid var(--accent);
    border-radius: 12px;
    padding: 1.2rem;
    background: rgba(30, 34, 42, 0.4);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
    background: rgba(30, 34, 42, 0.6);
}
.kpi-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.2rem;
    text-shadow: 0 0 10px rgba(255,255,255,0.1);
}
.kpi-subtext {
    font-size: 0.8rem;
    color: #8b949e;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #161b22;
    border-radius: 8px 8px 0px 0px;
    padding: 10px 24px;
    color: #8b949e;
    font-weight: 600;
    border: 1px solid #30363d;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background-color: #21262d;
    color: #58a6ff;
    border-top: 3px solid #58a6ff;
}

/* Hide default streamlit padding at the top */
.block-container {
    padding-top: 2rem !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #161b22;
    border-radius: 8px;
    color: #c9d1d9;
}
</style>
'''



@st.cache_data(show_spinner=False)
def get_manifest(artifact_dir: str) -> dict:
    return load_manifest(artifact_dir)


@st.cache_data(show_spinner=False)
def get_global_data(artifact_dir: str) -> dict:
    data = load_global_tables(artifact_dir)
    data["ecm_metrics"] = load_metrics(artifact_dir)
    data["ecm_params"] = load_ecm_params(artifact_dir)
    data["battery_ecm_metrics"] = load_battery_metrics(artifact_dir)
    data["battery_ecm_params"] = load_battery_ecm_params(artifact_dir)
    try:
        manifest = load_manifest(artifact_dir)
        import json
        if "r0_validation_path" in manifest:
            with open(manifest["r0_validation_path"], "r", encoding="utf-8") as f:
                data["r0_validation"] = json.load(f)
        else:
            data["r0_validation"] = {}
        if "impedance_metrics_path" in manifest:
            with open(manifest["impedance_metrics_path"], "r", encoding="utf-8") as f:
                data["impedance_metrics"] = json.load(f)
        else:
            data["impedance_metrics"] = {}
        if "scaling_metrics_path" in manifest and manifest["scaling_metrics_path"]:
            with open(manifest["scaling_metrics_path"], "r", encoding="utf-8") as f:
                data["scaling_metrics"] = json.load(f)
        else:
            data["scaling_metrics"] = {}
            
        if "regime_stats_path" in manifest and manifest["regime_stats_path"]:
            with open(manifest["regime_stats_path"], "r", encoding="utf-8") as f:
                data["regime_stats"] = json.load(f)
        else:
            data["regime_stats"] = {}
            
        # Load calibration data per battery
        data["calibration"] = {}
        for bid in available_battery_ids(artifact_dir, table_kind="sample_shadow"):
            cal_path = Path(artifact_dir) / f"calibration_{bid}.parquet"
            if cal_path.exists():
                data["calibration"][bid] = pd.read_parquet(cal_path)

    except Exception:
        data["r0_validation"] = {}
        data["impedance_metrics"] = {}
        data["scaling_metrics"] = {}
        data["regime_stats"] = {}
        data["calibration"] = {}
    return data


@st.cache_data(show_spinner=False)
def get_run_metadata(artifact_dir: str) -> dict:
    import json
    path = Path(artifact_dir) / "latest_run.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@st.cache_data(show_spinner=False)
def get_battery_ids(artifact_dir: str) -> list[str]:
    return available_battery_ids(artifact_dir, table_kind="sample_shadow")


@st.cache_data(show_spinner=False)
def get_battery_sample_shadow(
    artifact_dir: str,
    battery_id: str,
    start_cycle: int,
    end_cycle: int,
):
    return load_battery_table(
        artifact_dir,
        battery_id=battery_id,
        table_kind="sample_shadow",
        cycle_range=(start_cycle, end_cycle),
    )


def build_parameter_signal_plot(
    cycle_shadow: pd.DataFrame,
):
    required_cols = [
        "cycle_index",
        "cycle_type",
        "discharge_number",
        "r0",
        "r1",
        "r2",
        "c1",
        "c2",
    ]
    plot_frame = cycle_shadow[required_cols].copy()
    parameter_cols = ["r0", "r1", "r2", "c1", "c2"]
    for column in parameter_cols:
        plot_frame[column] = pd.to_numeric(plot_frame[column], errors="coerce")

    # Constrain ranges to realistic values to avoid chart skew
    for col in ["r0", "r1", "r2"]:
        plot_frame[col] = plot_frame[col].clip(lower=1e-5, upper=2.0)
    for col in ["c1", "c2"]:
        plot_frame[col] = plot_frame[col].clip(lower=1.0, upper=100000.0)

    plot_frame = plot_frame.dropna(subset=parameter_cols, how="all")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Resistance Parameters", "Capacitance Parameters"),
    )

    colors = {
        "r0": "#2563eb",
        "r1": "#0f766e",
        "r2": "#b45309",
        "c1": "#7c3aed",
        "c2": "#be123c",
    }
    labels = {
        "r0": "R0",
        "r1": "R1",
        "r2": "R2",
        "c1": "C1",
        "c2": "C2",
    }

    for column in ("r0", "r1", "r2"):
        fig.add_trace(
            go.Scatter(
                x=plot_frame["cycle_index"],
                y=plot_frame[column],
                mode="lines",
                name=labels[column],
                line={"width": 2.2, "color": colors[column]},
                customdata=plot_frame[["cycle_type", "discharge_number"]],
                hovertemplate=(
                    "Cycle %{x}<br>"
                    "Type %{customdata[0]}<br>"
                    "Discharge %{customdata[1]}<br>"
                    f"{labels[column]}: %{{y:.6f}} ohm<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    for column in ("c1", "c2"):
        fig.add_trace(
            go.Scatter(
                x=plot_frame["cycle_index"],
                y=plot_frame[column],
                mode="lines",
                name=labels[column],
                line={"width": 2.2, "color": colors[column]},
                customdata=plot_frame[["cycle_type", "discharge_number"]],
                hovertemplate=(
                    "Cycle %{x}<br>"
                    "Type %{customdata[0]}<br>"
                    "Discharge %{customdata[1]}<br>"
                    f"{labels[column]}: %{{y:.2f}} F<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title="Cycle-level ECM Parameters",
        height=560,
        hovermode="x unified",
        legend_title_text="Parameter",
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
    )
    fig.update_xaxes(title_text="Cycle Index", row=2, col=1)
    fig.update_yaxes(title_text="Ohm", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="Farad", row=2, col=1, rangemode="tozero")
    return fig


def format_kpi_value(value: float | int | None, suffix: str = "", digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{float(value):.{digits}f}{suffix}"


def latest_soc_value(detail_shadow: pd.DataFrame) -> float:
    if detail_shadow.empty:
        return float("nan")

    soc_cols = [column for column in ("soc_ekf", "soc") if column in detail_shadow.columns]
    if not soc_cols:
        return float("nan")

    ordered = detail_shadow.sort_values(["cycle_index", "sample_index"])
    for column in soc_cols:
        series = ordered[column].dropna()
        if not series.empty:
            return float(series.iloc[-1])
    return float("nan")


def calculate_eol_summary(
    cycle_shadow: pd.DataFrame,
    battery_id: str,
    soh_threshold: float = NASA_EOL_SOH,
    capacity_threshold_ah: float = NASA_EOL_CAPACITY_AH,
) -> dict[str, float | str]:
    battery_frame = cycle_shadow[
        (cycle_shadow["battery_id"] == battery_id)
        & (cycle_shadow["cycle_type"] == "discharge")
    ].copy()

    if battery_frame.empty:
        return {
            "eol_threshold": soh_threshold,
            "capacity_threshold_ah": capacity_threshold_ah,
            "observed_eol_cycle": float("nan"),
            "predicted_eol_cycle": float("nan"),
            "remaining_cycles": float("nan"),
            "eol_status": "n/a",
            "latest_cycle": float("nan"),
        }

    battery_frame = battery_frame.sort_values("cycle_index")
    latest = battery_frame.iloc[-1]
    latest_cycle = float(latest["cycle_index"])
    latest_soh = float(latest["soh"]) if pd.notna(latest.get("soh")) else float("nan")
    latest_capacity = (
        float(latest["capacity_ah"]) if pd.notna(latest.get("capacity_ah")) else float("nan")
    )
    initial_capacity = (
        float(battery_frame["initial_capacity_ah"].dropna().iloc[0])
        if "initial_capacity_ah" in battery_frame and not battery_frame["initial_capacity_ah"].dropna().empty
        else float(battery_frame["capacity_ah"].dropna().iloc[0])
        if "capacity_ah" in battery_frame and not battery_frame["capacity_ah"].dropna().empty
        else float("nan")
    )
    capacity_equivalent_soh = (
        capacity_threshold_ah / initial_capacity
        if np.isfinite(initial_capacity) and initial_capacity > 0
        else float("nan")
    )
    effective_soh_threshold = (
        max(soh_threshold, capacity_equivalent_soh)
        if np.isfinite(capacity_equivalent_soh)
        else soh_threshold
    )

    observed_candidates = []
    if "soh" in battery_frame:
        soh_crossing = battery_frame[battery_frame["soh"] <= soh_threshold]
        if not soh_crossing.empty:
            observed_candidates.append(float(soh_crossing.iloc[0]["cycle_index"]))
    if "capacity_ah" in battery_frame:
        capacity_crossing = battery_frame[battery_frame["capacity_ah"] <= capacity_threshold_ah]
        if not capacity_crossing.empty:
            observed_candidates.append(float(capacity_crossing.iloc[0]["cycle_index"]))

    observed_eol_cycle = min(observed_candidates) if observed_candidates else float("nan")

    usable = battery_frame.dropna(subset=["cycle_index", "soh"])
    predicted_eol_cycle = float("nan")
    if len(usable) >= 2:
        slope, intercept = np.polyfit(
            usable["cycle_index"].to_numpy(dtype=float),
            usable["soh"].to_numpy(dtype=float),
            1,
        )
        if np.isfinite(slope) and slope < -1e-10:
            predicted_eol_cycle = float((effective_soh_threshold - intercept) / slope)

    remaining_cycles = (
        predicted_eol_cycle - latest_cycle
        if np.isfinite(predicted_eol_cycle)
        else float("nan")
    )

    eol_reached = (
        np.isfinite(observed_eol_cycle)
        or (np.isfinite(latest_soh) and latest_soh <= effective_soh_threshold)
        or (np.isfinite(latest_capacity) and latest_capacity <= capacity_threshold_ah)
    )
    if eol_reached:
        status = "Reached"
    elif np.isfinite(latest_soh) and latest_soh <= WARNING_SOH:
        status = "Warning"
    elif np.isfinite(latest_soh) and latest_soh > HEALTHY_SOH:
        status = "Healthy"
    else:
        status = "Watch"

    return {
        "eol_threshold": effective_soh_threshold,
        "nasa_soh_threshold": soh_threshold,
        "capacity_equivalent_soh": capacity_equivalent_soh,
        "capacity_threshold_ah": capacity_threshold_ah,
        "observed_eol_cycle": observed_eol_cycle,
        "predicted_eol_cycle": predicted_eol_cycle,
        "remaining_cycles": remaining_cycles,
        "eol_status": status,
        "latest_cycle": latest_cycle,
    }


def build_eol_plot(
    cycle_shadow: pd.DataFrame,
    battery_id: str,
    eol_summary: dict[str, float | str],
):
    battery_frame = cycle_shadow[
        (cycle_shadow["battery_id"] == battery_id)
        & (cycle_shadow["cycle_type"] == "discharge")
    ].copy()
    fig = px.line(
        battery_frame,
        x="cycle_index",
        y="soh",
        markers=True,
        title=f"NASA EOL Projection: {battery_id}",
        labels={"cycle_index": "Cycle Index", "soh": "SOH"},
    )
    effective_threshold = eol_summary["eol_threshold"]
    fig.add_hline(
        y=effective_threshold,
        line_dash="dash",
        line_color="#be123c",
        annotation_text="NASA EOL threshold",
    )
    if (
        isinstance(eol_summary["capacity_equivalent_soh"], float)
        and np.isfinite(eol_summary["capacity_equivalent_soh"])
        and abs(eol_summary["capacity_equivalent_soh"] - NASA_EOL_SOH) > 1e-6
    ):
        fig.add_hline(
            y=NASA_EOL_SOH,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="SOH 0.70 benchmark",
        )
    fig.add_hline(
        y=WARNING_SOH,
        line_dash="dot",
        line_color="#b45309",
        annotation_text="Warning SOH 0.80",
    )

    predicted_eol = eol_summary["predicted_eol_cycle"]
    if isinstance(predicted_eol, float) and np.isfinite(predicted_eol):
        fig.add_vline(
            x=predicted_eol,
            line_dash="dash",
            line_color="#7c3aed",
            annotation_text="Predicted EOL",
        )

    observed_eol = eol_summary["observed_eol_cycle"]
    if isinstance(observed_eol, float) and np.isfinite(observed_eol):
        fig.add_vline(
            x=observed_eol,
            line_dash="solid",
            line_color="#dc2626",
            annotation_text="Observed EOL",
        )

    fig.update_layout(height=460, hovermode="x unified")
    return fig


def build_battery_comparison_frame(
    cycle_shadow: pd.DataFrame,
    battery_ids: list[str],
    battery_ecm_metrics: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for battery_id in battery_ids:
        summary = summarize_battery(cycle_shadow, battery_id)
        eol_summary = calculate_eol_summary(cycle_shadow, battery_id)
        metrics = battery_ecm_metrics.get(battery_id, {})
        rows.append(
            {
                "battery_id": battery_id,
                "latest_soh": summary["latest_soh"],
                "latest_rul_cycles": summary["latest_rul_cycles"],
                "nasa_eol_status": eol_summary["eol_status"],
                "nasa_predicted_eol_cycle": eol_summary["predicted_eol_cycle"],
                "nasa_remaining_cycles": eol_summary["remaining_cycles"],
                "nasa_observed_eol_cycle": eol_summary["observed_eol_cycle"],
                "latest_capacity_ah": summary["latest_capacity_ah"],
                "discharge_cycles": summary["discharge_cycles"],
                "ecm_mae_v": metrics.get("mae_v"),
                "ekf_mae_v": metrics.get("ekf_mae_v"),
            }
        )
    return pd.DataFrame(rows)


def build_residual_plot(detail_shadow: pd.DataFrame, selected_battery: str):
    residual_cols = [
        column
        for column in ("voltage_error_v", "voltage_residual_ekf_v")
        if column in detail_shadow.columns
    ]
    plot_frame = detail_shadow[["time_s", "cycle_index", *residual_cols]].copy()
    plot_frame = plot_frame.dropna(subset=residual_cols, how="all")
    long_frame = plot_frame.melt(
        id_vars=["time_s", "cycle_index"],
        value_vars=residual_cols,
        var_name="residual_type",
        value_name="residual_v",
    )
    label_map = {
        "voltage_error_v": "ECM residual",
        "voltage_residual_ekf_v": "EKF innovation",
    }
    long_frame["residual_type"] = long_frame["residual_type"].map(label_map)
    fig = px.line(
        long_frame,
        x="time_s",
        y="residual_v",
        color="residual_type",
        hover_data=["cycle_index"],
        title=f"Voltage Residuals: {selected_battery}",
        labels={"time_s": "Time (s)", "residual_v": "Residual (V)", "residual_type": "Series"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(height=460, hovermode="x unified")
    return fig


def add_physics_features(
    detail_shadow: pd.DataFrame,
    ocv_curve: pd.DataFrame,
    ecm_params: dict[str, float],
) -> pd.DataFrame:
    if detail_shadow.empty:
        return detail_shadow.copy()

    frame = apply_soc_anchor(detail_shadow, ocv_curve)
    params = ECMParameters(
        r0=float(ecm_params.get("r0", 0.01)),
        r1=float(ecm_params.get("r1", 0.01)),
        c1=float(ecm_params.get("c1", 2000.0)),
        r2=float(ecm_params.get("r2", 0.02)),
        c2=float(ecm_params.get("c2", 4000.0)),
    )
    dynamic = get_dynamic_params(
        frame["soc_corrected"].ffill().bfill().fillna(0.5),
        frame["temperature_c"].ffill().bfill().fillna(25.0),
        frame["soh"].ffill().bfill().fillna(1.0),
        params,
    )
    return pd.concat([frame.reset_index(drop=True), dynamic.reset_index(drop=True)], axis=1)


def build_physics_summary(physics_shadow: pd.DataFrame) -> pd.DataFrame:
    if physics_shadow.empty:
        return pd.DataFrame()

    columns = [
        "cycle_index",
        "cycle_type",
        "soc_raw",
        "soc_ocv",
        "soc_corrected",
        "r0_dynamic",
        "r1_dynamic",
        "r2_dynamic",
        "c1_dynamic",
        "c2_dynamic",
    ]
    return (
        physics_shadow[columns]
        .groupby(["cycle_index", "cycle_type"], as_index=False)
        .mean(numeric_only=True)
    )


def render_maintenance_panel(
    summary: dict[str, float],
    selected_battery: str,
    battery_cycle_shadow: pd.DataFrame,
) -> None:
    # Get latest discharge cycle data
    discharge = battery_cycle_shadow[battery_cycle_shadow["cycle_type"] == "discharge"].sort_values("cycle_index")
    if discharge.empty:
        return
        
    latest = discharge.iloc[-1]
    rec = get_charge_recommendation(
        soh=float(latest.get("soh", 1.0)),
        rul_cycles=float(latest.get("rul_cycles", 100.0)),
        temperature_mean_c=float(latest.get("temperature_mean_c", 25.0)),
        plating_risk=float(latest.get("plating_risk", 0.0)),
    )
    
    colors = {
        "normal": "#3fb950",
        "reduce_crate": "#d29922",
        "reduce_voltage": "#d29922",
        "inspect": "#f85149",
        "replace": "#f85149",
    }
    actions = {
        "normal": "NORMAL OPERATION",
        "reduce_crate": "REDUCE CHARGE RATE",
        "reduce_voltage": "REDUCE PEAK VOLTAGE",
        "inspect": "DETAILED INSPECTION REQUIRED",
        "replace": "REPLACEMENT RECOMMENDED",
    }
    
    st.markdown(f"""
    <div style="background: rgba(30,34,42,0.4); border: 1px solid {colors[rec['action']]}; border-left: 8px solid {colors[rec['action']]}; padding: 20px; border-radius: 12px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Maintenance Decision for {selected_battery}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ffffff;">{actions[rec['action']]}</div>
                <div style="font-size: 1rem; color: #c9d1d9; margin-top: 8px;">{rec['reason']}</div>
            </div>
            <div style="background: {colors[rec['action']]}; color: #ffffff; padding: 8px 16px; border-radius: 20px; font-weight: 700; font-size: 0.9rem;">
                {rec['action'].replace('_', ' ').upper()}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(
    summary: dict[str, float],
    latest_soc: float,
    selected_battery: str,
    eol_summary: dict[str, float | str],
) -> None:
    cards = [
        {
            "label": "SOH",
            "value": format_kpi_value(summary["latest_soh"], digits=3),
            "subtext": "Latest discharge health",
            "accent": "#2563eb",
        },
        {
            "label": "RUL",
            "value": format_kpi_value(summary["latest_rul_cycles"], suffix=" cycles", digits=1),
            "subtext": "Projected to selected threshold",
            "accent": "#0f766e",
        },
        {
            "label": "SOC",
            "value": format_kpi_value(latest_soc, digits=3),
            "subtext": "Latest value in selected range",
            "accent": "#7c3aed",
        },
        {
            "label": "Cycles",
            "value": format_kpi_value(summary["discharge_cycles"]),
            "subtext": f"Discharge cycles for {selected_battery}",
            "accent": "#b45309",
        },
        {
            "label": "EOL",
            "value": str(eol_summary["eol_status"]),
            "subtext": (
                "Pred "
                f"{format_kpi_value(eol_summary['predicted_eol_cycle'], digits=1)}"
                " cycle"
            ),
            "accent": "#be123c",
        },
    ]

    st.markdown(CARD_CSS, unsafe_allow_html=True)
    grid_html = '<div class="kpi-grid">'
    for card in cards:
        grid_html += f"""<div class="kpi-card" style="--accent: {card['accent']};">
<div class="kpi-label">{card['label']}</div>
<div class="kpi-value">{card['value']}</div>
<div class="kpi-subtext">{card['subtext']}</div>
</div>"""
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)
    
    # Anomaly Alert
    if "anomaly" in summary and summary["anomaly"] > 0:
        st.warning(f"⚠️ {int(summary['anomaly'])} anomalous cycles detected in this battery's history.")


def main() -> None:
    st.title("Li-ion Battery Digital Shadow")

    default_artifact_dir = str(ROOT / "artifacts")
    artifact_dir = st.sidebar.text_input("Artifact directory", value=default_artifact_dir)

    try:
        manifest = get_manifest(artifact_dir)
        global_data = get_global_data(artifact_dir)
        battery_ids = get_battery_ids(artifact_dir)
        run_meta = get_run_metadata(artifact_dir)
    except FileNotFoundError:
        st.error(
            "Dashboard artifacts were not found. Run "
            "`python scripts/prepare_dashboard_data.py` first."
        )
        st.stop()

    cycle_shadow = global_data["cycle_shadow"]
    ocv_curve = global_data["ocv_curve"]
    ecm_metrics = global_data["ecm_metrics"]
    ecm_params = global_data["ecm_params"]
    battery_ecm_metrics = global_data["battery_ecm_metrics"]
    battery_ecm_params = global_data["battery_ecm_params"]

    st.sidebar.markdown("### Data Fetch Mode")
    st.sidebar.caption("Global summaries are loaded eagerly. Battery traces are fetched lazily.")

    if not battery_ids:
        st.warning("No battery partitions were found in the artifact directory.")
        st.stop()

    selected_battery = st.sidebar.selectbox("Battery", battery_ids)
    battery_cycle_shadow = cycle_shadow[cycle_shadow["battery_id"] == selected_battery].copy()
    cycle_min = int(battery_cycle_shadow["cycle_index"].min())
    cycle_max = int(battery_cycle_shadow["cycle_index"].max())
    cycle_range = st.sidebar.slider(
        "Cycle range",
        min_value=cycle_min,
        max_value=cycle_max,
        value=(cycle_min, cycle_max),
    )

    summary = summarize_battery(cycle_shadow, selected_battery)
    eol_summary = calculate_eol_summary(cycle_shadow, selected_battery)
    detail_shadow = get_battery_sample_shadow(
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
                detail_shadow[col] = detail_shadow[col].rolling(window=10, min_periods=1).mean()
    detail_cycle_shadow = battery_cycle_shadow[
        (battery_cycle_shadow["cycle_index"] >= cycle_range[0])
        & (battery_cycle_shadow["cycle_index"] <= cycle_range[1])
    ].copy()
    selected_ecm_params = battery_ecm_params.get(selected_battery, ecm_params)
    physics_shadow = add_physics_features(detail_shadow, ocv_curve, selected_ecm_params)
    efficiency_table = compute_efficiency_trends(compute_cycle_efficiency(detail_shadow))

    render_maintenance_panel(summary, selected_battery, battery_cycle_shadow)
    render_kpi_cards(summary, latest_soc_value(detail_shadow), selected_battery, eol_summary)

    if run_meta:
        with st.expander("🚀 Run Metadata", expanded=False):
            st.json(run_meta)

    st.caption(
        "`cycle_type` tags each row as `charge`, `discharge`, or `impedance`. "
        "`discharge_number` is the derived aging-cycle count and only increments on discharge cycles."
    )

    st.subheader("Plots")
    overview_tab, health_tab, ecm_tab, electro_tab, diagnostics_tab, data_tab = st.tabs([
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
    physics_tab = electro_tab

    with comparison_tab:
        # --- Group 6 Fleet Safety Audit ---
        with st.expander("🛡️ Fleet Safety Audit (Fleet-wide Risk Scan)", expanded=True):
            high_plating = cycle_shadow[cycle_shadow["plating_risk"] > 0.5]["battery_id"].unique()
            low_sop = cycle_shadow[cycle_shadow["sop_w"] < 5.0]["battery_id"].unique()
            critical = set(high_plating) | set(low_sop)
            
            if not critical:
                st.success("✅ All batteries in the fleet are operating within safe bounds.")
            else:
                st.error(f"⚠️ {len(critical)} batteries require immediate attention.")
                for bid in critical:
                    risks = []
                    if bid in high_plating: risks.append("High Plating Risk")
                    if bid in low_sop: risks.append("Low SOP (Power Fade)")
                    st.markdown(f"- **{bid}**: {', '.join(risks)}")

        st.markdown("### AI Insight Summary")
        if np.isfinite(summary["latest_soh"]):
            soh_percent = summary["latest_soh"] * 100
            if soh_percent > 85:
                status_msg = "healthy, with no critical anomalies detected."
                status_color = "#3fb950"
            elif soh_percent > 70:
                status_msg = "showing moderate degradation. RUL estimates suggest routine maintenance soon."
                status_color = "#d29922"
            else:
                status_msg = "in critical condition. Impedance growth has accelerated."
                status_color = "#f85149"
        else:
            status_msg = "being analyzed."
            status_color = "#58a6ff"
            
        st.markdown(f'''
        <div style="background: rgba(30,34,42,0.4); border-left: 4px solid {status_color}; padding: 16px; border-radius: 8px; margin-bottom: 24px; font-size: 1.05rem;">
            <strong>Insight:</strong> The selected battery is {status_msg} 
            The latest SOH is tracking consistently with ECM models.
        </div>
        ''', unsafe_allow_html=True)
        
        comparison_selection = st.multiselect(
            "Batteries",
            battery_ids,
            default=battery_ids,
            key="comparison_batteries",
        )
        if not comparison_selection:
            st.info("Select at least one battery to compare.")
        else:
            comparison_frame = build_battery_comparison_frame(
                cycle_shadow,
                comparison_selection,
                battery_ecm_metrics,
            )
            comparison_cols = st.columns(2)
            with comparison_cols[0]:
                soh_compare_fig = px.bar(
                    comparison_frame,
                    x="battery_id",
                    y="latest_soh",
                    title="Latest SOH",
                    labels={"battery_id": "Battery", "latest_soh": "SOH"},
                    color="battery_id",
                )
                soh_compare_fig.update_layout(showlegend=False, height=360)
                st.plotly_chart(soh_compare_fig, use_container_width=True)
            with comparison_cols[1]:
                rul_compare_fig = px.bar(
                    comparison_frame,
                    x="battery_id",
                    y="latest_rul_cycles",
                    title="Latest RUL",
                    labels={"battery_id": "Battery", "latest_rul_cycles": "RUL (cycles)"},
                    color="battery_id",
                )
                rul_compare_fig.update_layout(showlegend=False, height=360)
                st.plotly_chart(rul_compare_fig, use_container_width=True)

            error_compare_fig = px.bar(
                comparison_frame.melt(
                    id_vars=["battery_id"],
                    value_vars=["ecm_mae_v", "ekf_mae_v"],
                    var_name="metric",
                    value_name="mae_v",
                ),
                x="battery_id",
                y="mae_v",
                color="metric",
                barmode="group",
                title="Voltage Reconstruction Error",
                labels={"battery_id": "Battery", "mae_v": "MAE (V)", "metric": "Metric"},
            )
            error_compare_fig.update_layout(height=380)
            st.plotly_chart(error_compare_fig, use_container_width=True)
            st.dataframe(comparison_frame, use_container_width=True)

    with eol_tab:
        eol_cols = st.columns(4)
        eol_cols[0].metric("EOL Status", str(eol_summary["eol_status"]))
        eol_cols[1].metric(
            "Predicted EOL Cycle",
            format_kpi_value(eol_summary["predicted_eol_cycle"], digits=1),
        )
        eol_cols[2].metric(
            "Remaining Cycles",
            format_kpi_value(eol_summary["remaining_cycles"], digits=1),
        )
        eol_cols[3].metric(
            "Observed EOL Cycle",
            format_kpi_value(eol_summary["observed_eol_cycle"], digits=1),
        )
        st.caption(
            "NASA EOL is reached when discharge capacity is `<= 1.4 Ah` "
            "or SOH is `<= 0.70`. For prediction, `1.4 Ah` is converted to the "
            "battery-specific equivalent SOH threshold, then the earlier threshold is used."
        )
        eol_fig = build_eol_plot(cycle_shadow, selected_battery, eol_summary)
        st.plotly_chart(eol_fig, use_container_width=True)

    with soh_tab:
        discharge_cycles = cycle_shadow[cycle_shadow["cycle_type"] == "discharge"].copy()
        batt_soh = discharge_cycles[discharge_cycles["battery_id"] == selected_battery].copy()
        
        fig = go.Figure()
        
        # Uncertainty Band
        if "soh_model_upper" in batt_soh.columns and "soh_model_lower" in batt_soh.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([batt_soh["cycle_index"], batt_soh["cycle_index"][::-1]]),
                y=pd.concat([batt_soh["soh_model_upper"], batt_soh["soh_model_lower"][::-1]]),
                fill='toself',
                fillcolor='rgba(37, 99, 235, 0.2)',
                line_color='rgba(255,255,255,0)',
                showlegend=False,
                name="SOH 90% CI",
            ))
            
        fig.add_trace(go.Scatter(
            x=batt_soh["cycle_index"],
            y=batt_soh["soh"],
            mode='lines+markers',
            name="Observed SOH",
            line=dict(color="#2563eb", width=2)
        ))
        
        if "soh_model_pred" in batt_soh.columns:
            fig.add_trace(go.Scatter(
                x=batt_soh["cycle_index"],
                y=batt_soh["soh_model_pred"],
                mode='lines',
                name="Bayesian Model",
                line=dict(color="#60a5fa", width=2, dash='dash')
            ))
            
        fig.update_layout(title=f"SOH Degradation & Bayesian Uncertainty: {selected_battery}",
                          xaxis_title="Cycle Index", yaxis_title="SOH", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with rul_tab:
        batt_rul = discharge_cycles[discharge_cycles["battery_id"] == selected_battery].copy()
        fig = go.Figure()
        
        if "rul_p90" in batt_rul.columns and "rul_p10" in batt_rul.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([batt_rul["cycle_index"], batt_rul["cycle_index"][::-1]]),
                y=pd.concat([batt_rul["rul_p90"], batt_rul["rul_p10"][::-1]]),
                fill='toself',
                fillcolor='rgba(15, 118, 110, 0.2)',
                line_color='rgba(255,255,255,0)',
                showlegend=False,
                name="RUL 80% CI",
            ))
            
        fig.add_trace(go.Scatter(
            x=batt_rul["cycle_index"],
            y=batt_rul["rul_cycles"],
            mode='lines',
            name="Linear RUL",
            line=dict(color="#0f766e", width=2)
        ))
        
        if "rul_cycles_gpr" in batt_rul.columns:
            fig.add_trace(go.Scatter(
                x=batt_rul["cycle_index"],
                y=batt_rul["rul_cycles_gpr"],
                mode='lines',
                name="GPR Median RUL",
                line=dict(color="#2dd4bf", width=2, dash='dot')
            ))
            
        fig.update_layout(title=f"RUL Projection & GPR Uncertainty: {selected_battery}",
                          xaxis_title="Cycle Index", yaxis_title="Remaining Cycles", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Group 5 What-if Simulation ---
        with st.expander("🛠️ Stress Scenario What-if Simulation", expanded=False):
            st.markdown("Simulate how changes in operating stress would affect the linear RUL projection.")
            sim_cols = st.columns(2)
            temp_adj = sim_cols[0].slider("Temperature Delta (°C)", -10.0, 10.0, 0.0, 1.0)
            crate_adj = sim_cols[1].slider("C-Rate Scaling factor", 0.5, 2.0, 1.0, 0.1)
            
            sim_df = batt_rul.copy()
            if "temperature_max_c" in sim_df.columns:
                sim_df["temperature_max_c"] += temp_adj
            # C-rate scaling affects current
            if "current_mean_a" in sim_df.columns:
                sim_df["current_mean_a"] *= crate_adj
                
            sim_df = add_rul_estimates(sim_df, soh_threshold=NASA_EOL_SOH)
            
            whatif_fig = go.Figure()
            whatif_fig.add_trace(go.Scatter(x=batt_rul["cycle_index"], y=batt_rul["rul_cycles"], name="Baseline RUL", line=dict(color="#666")))
            whatif_fig.add_trace(go.Scatter(x=sim_df["cycle_index"], y=sim_df["rul_cycles"], name="Simulated RUL", line=dict(color="#f85149", width=3)))
            whatif_fig.update_layout(title="Stress Scenario Impact on RUL", xaxis_title="Cycle", yaxis_title="RUL (cycles)", height=400)
            st.plotly_chart(whatif_fig, use_container_width=True)

    with ocv_tab:
        ocv_fig = px.line(ocv_curve, x="soc", y="ocv_v", title="OCV vs SOC")
        st.plotly_chart(ocv_fig, use_container_width=True)

    with voltage_tab:
        if detail_shadow.empty:
            st.info("No detail samples were found for the selected cycle range.")
        else:
            voltage_fig = px.line(
                detail_shadow,
                x="time_s",
                y=["voltage_v", "voltage_model_v", "voltage_ekf_v"],
                title=f"Measured vs ECM/EKF Voltage: {selected_battery}",
            )
            st.plotly_chart(voltage_fig, use_container_width=True)

    with soc_tab:
        if detail_shadow.empty:
            st.info("No detail samples were found for the selected cycle range.")
        else:
            soc_fig = px.line(
                detail_shadow,
                x="time_s",
                y=["soc", "soc_ekf"],
                title=f"SOC vs EKF SOC: {selected_battery}",
            )
            st.plotly_chart(soc_fig, use_container_width=True)

    with thermal_tab:
        if detail_shadow.empty:
            st.info("No detail samples were found for the selected cycle range.")
        else:
            thermo_fig = px.line(
                detail_shadow,
                x="time_s",
                y=["current_a", "temperature_c"],
                title=f"Current and Temperature: {selected_battery}",
            )
            st.plotly_chart(thermo_fig, use_container_width=True)

    with physics_tab:
        if physics_shadow.empty:
            st.info("No detail samples were found for the selected cycle range.")
        else:
            st.markdown("#### OCV-SOC Anchor")
            soc_anchor_fig = px.line(
                physics_shadow,
                x="time_s",
                y=["soc_raw", "soc_ocv", "soc_corrected"],
                title=f"SOC Raw vs OCV Anchor vs Corrected: {selected_battery}",
                labels={"time_s": "Time (s)", "value": "SOC", "variable": "SOC Series"},
            )
            soc_anchor_fig.update_layout(height=420, hovermode="x unified")
            st.plotly_chart(soc_anchor_fig, use_container_width=True)

            st.markdown("#### Dynamic ECM Parameters")
            dynamic_cols = st.columns(2)
            with dynamic_cols[0]:
                dynamic_resistance_fig = px.line(
                    physics_shadow,
                    x="time_s",
                    y=["r0_dynamic", "r1_dynamic", "r2_dynamic"],
                    title="State-dependent Resistance Parameters",
                    labels={"time_s": "Time (s)", "value": "Ohm", "variable": "Parameter"},
                )
                dynamic_resistance_fig.update_layout(height=380, hovermode="x unified")
                st.plotly_chart(dynamic_resistance_fig, use_container_width=True)
            with dynamic_cols[1]:
                dynamic_capacitance_fig = px.line(
                    physics_shadow,
                    x="time_s",
                    y=["c1_dynamic", "c2_dynamic"],
                    title="State-dependent Capacitance Parameters",
                    labels={"time_s": "Time (s)", "value": "Farad", "variable": "Parameter"},
                )
                dynamic_capacitance_fig.update_layout(height=380, hovermode="x unified")
                st.plotly_chart(dynamic_capacitance_fig, use_container_width=True)

            st.markdown("#### Coulombic Efficiency")
            if efficiency_table.empty:
                st.info("No paired charge/discharge cycles were found in the selected range.")
            else:
                efficiency_fig = px.line(
                    efficiency_table,
                    x="cycle_index",
                    y=["coulombic_efficiency", "coulombic_efficiency_rollmean"],
                    markers=True,
                    title="Coulombic Efficiency Trend",
                    labels={"cycle_index": "Cycle Index", "value": "Efficiency", "variable": "Series"},
                )
                efficiency_fig.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(efficiency_fig, use_container_width=True)
                st.dataframe(efficiency_table, use_container_width=True)

            st.markdown("#### Operational Safety & Power Limits")
            safety_grid = st.columns(2)
            with safety_grid[0]:
                if "sop_w" in detail_cycle_shadow.columns:
                    sop_fig = px.line(
                        detail_cycle_shadow.dropna(subset=["sop_w"]), 
                        x="cycle_index", y="sop_w", 
                        title="State of Power (SOP) Evolution",
                        labels={"cycle_index": "Cycle", "sop_w": "Peak Power (W)"}
                    )
                    st.plotly_chart(sop_fig, use_container_width=True)
            with safety_grid[1]:
                if "plating_risk" in detail_cycle_shadow.columns:
                    risk_fig = px.line(
                        detail_cycle_shadow.dropna(subset=["plating_risk"]), 
                        x="cycle_index", y="plating_risk", 
                        title="Lithium Plating Risk Index",
                        labels={"cycle_index": "Cycle", "plating_risk": "Risk Score (0-1)"}
                    )
                    st.plotly_chart(risk_fig, use_container_width=True)

            st.markdown("#### Physics Feature Summary")
            st.dataframe(build_physics_summary(physics_shadow), use_container_width=True)

    with impedance_tab:
        st.markdown("### Impedance & ECM Validation")
        r0_val = global_data.get("r0_validation", {}).get(selected_battery, {})
        imp_met = global_data.get("impedance_metrics", {}).get(selected_battery, {})
        scale_met = global_data.get("scaling_metrics", {}).get(selected_battery, {})
        imp_trend = global_data.get("impedance_trend", pd.DataFrame())
        imp_curve = global_data.get("impedance_curve", pd.DataFrame())
        
        imp_trend_batt = imp_trend[imp_trend["battery_id"] == selected_battery].copy() if not imp_trend.empty else pd.DataFrame()
        imp_curve_batt = imp_curve[imp_curve["battery_id"] == selected_battery].copy() if not imp_curve.empty else pd.DataFrame()
            
        with st.expander("About Impedance Validation", expanded=False):
            st.markdown(
                "**What is Impedance?** Transient impedance is the immediate voltage response to a sudden current pulse. "
                "Tracking this shows physical resistance growth inside the cell.\n\n"
                "**Why R0 matters?** R0 in the ECM represents the pure ohmic drop. Comparing estimated impedance with R0 validates "
                "our ECM parameter fitting."
            )

        st.markdown("#### Validation Metrics")
        val_cols = st.columns(5)
        val_cols[0].metric("RMSE", format_kpi_value(r0_val.get("rmse"), suffix=" Ω", digits=4))
        val_cols[1].metric("MAE", format_kpi_value(r0_val.get("mae"), suffix=" Ω", digits=4))
        val_cols[2].metric("Correlation", format_kpi_value(r0_val.get("correlation"), digits=3))
        val_cols[3].metric("Drift", format_kpi_value(r0_val.get("drift_percent"), suffix=" %", digits=2))
        val_cols[4].metric("Res Growth", format_kpi_value(imp_met.get("growth_rate"), suffix=" Ω/cyc", digits=6))
        
        if not imp_trend_batt.empty:
            st.markdown("#### Resistance Growth Trend")
            imp_fig = px.line(
                imp_trend_batt, x="cycle_index", y=["impedance", "rolling_avg"],
                title="Transient Impedance Evolution",
                labels={"cycle_index": "Cycle", "value": "Impedance (Ω)"}
            )
            anomalies = imp_trend_batt[imp_trend_batt["anomaly"] == 1]
            if not anomalies.empty:
                imp_fig.add_scatter(x=anomalies["cycle_index"], y=anomalies["impedance"],
                                    mode="markers", marker=dict(color="red", size=8), name="Anomaly")
            st.plotly_chart(imp_fig, use_container_width=True)
            
        if scale_met:
            with st.expander("ECM Scaling Diagnostics", expanded=False):
                diag_cols = st.columns(3)
                diag_cols[0].metric("Mean Predicted R0", format_kpi_value(scale_met.get("mean_predicted_r0"), suffix=" Ω", digits=6))
                diag_cols[0].metric("Mean EIS R0", format_kpi_value(scale_met.get("mean_eis_r0"), suffix=" Ω", digits=6))
                diag_cols[1].metric("Scale Factor", format_kpi_value(scale_met.get("scale_factor"), digits=2))
                diag_cols[1].metric("Normalization Detected", str(scale_met.get("normalization_detected")))
                diag_cols[2].metric("Unit Consistency", str(scale_met.get("unit_consistency")))
                diag_cols[2].metric("Outlier Counts", format_kpi_value(scale_met.get("outlier_counts")))
            
        if not imp_curve_batt.empty:
            st.markdown("#### R0 vs Estimated Impedance")
            r0_fig = px.line(
                imp_curve_batt, x="cycle_index", y=["r0", "estimated_impedance_ohm"],
                title="ECM R0 vs Transient Impedance",
                labels={"cycle_index": "Cycle", "value": "Resistance (Ω)"}
            )
            st.plotly_chart(r0_fig, use_container_width=True)
            
        st.markdown("#### EIS Re vs Model R0")
        r0_col = "r0_aligned" if "r0_aligned" in battery_cycle_shadow.columns else "r0"
        
        if "re_ohm" in battery_cycle_shadow.columns and r0_col in battery_cycle_shadow.columns:
            eis_frame = battery_cycle_shadow.dropna(subset=["re_ohm", r0_col]).copy()
            if not eis_frame.empty:
                r0_ref_series = eis_frame["re_ohm"].values
                r0_pred_series = eis_frame[r0_col].values
                
                r0_ref_mean = np.mean(r0_ref_series)
                validation_error = np.mean(np.abs(r0_pred_series - r0_ref_series))
                
                if len(r0_ref_series) > 1:
                    correlation = np.corrcoef(r0_pred_series, r0_ref_series)[0, 1]
                else:
                    correlation = np.nan
                    
                st.info(
                    f"**Complex Impedance (EIS) Validation:** "
                    f"R0_ref = {r0_ref_mean:.5f} Ω | Mean Error = {validation_error:.5f} Ω | Corr = {correlation:.4f}"
                )

            eis_fig = px.line(
                battery_cycle_shadow, x="cycle_index", y=r0_col,
                title="Measured EIS Re vs Scaled Model R0",
                labels={"cycle_index": "Cycle", "value": "Resistance (Ω)"}
            )
            eis_fig.data[0].name = "Scaled Model R0"
            eis_fig.data[0].showlegend = True
            
            re_frame = battery_cycle_shadow.dropna(subset=["re_ohm"])
            if not re_frame.empty:
                eis_fig.add_scatter(
                    x=re_frame["cycle_index"], 
                    y=re_frame["re_ohm"], 
                    mode="markers+lines", 
                    name="Measured Re (EIS)",
                    marker=dict(size=6)
                )
            
            st.plotly_chart(eis_fig, use_container_width=True)
            
        st.markdown("#### SOH vs Impedance Relationship")
        if not imp_trend_batt.empty:
            soh_imp = battery_cycle_shadow.merge(imp_trend_batt, on=["battery_id", "cycle_index"])
            if not soh_imp.empty and "soh" in soh_imp.columns:
                soh_imp_fig = px.scatter(
                    soh_imp, x="impedance", y="soh", color="cycle_index",
                    title="SOH vs Transient Impedance",
                    labels={"impedance": "Impedance (Ω)", "soh": "SOH"}
                )
                st.plotly_chart(soh_imp_fig, use_container_width=True)

    with cycle_detail_tab:
        if detail_shadow.empty:
            st.info("No detail samples were found for the selected cycle range.")
        else:
            cycle_options = sorted(detail_shadow["cycle_index"].dropna().unique().tolist())
            selected_cycle = st.selectbox(
                "Cycle",
                cycle_options,
                index=0,
                key="cycle_detail_cycle",
            )
            cycle_frame = detail_shadow[detail_shadow["cycle_index"] == selected_cycle].copy()
            cycle_cols = st.columns(3)
            cycle_cols[0].metric("Samples", f"{len(cycle_frame):,}")
            cycle_cols[1].metric(
                "Mean Voltage",
                format_kpi_value(cycle_frame["voltage_v"].mean(), suffix=" V", digits=3),
            )
            cycle_cols[2].metric(
                "Max Temp",
                format_kpi_value(cycle_frame["temperature_c"].max(), suffix=" C", digits=2),
            )

            cycle_voltage_fig = px.line(
                cycle_frame,
                x="time_s",
                y=["voltage_v", "voltage_model_v", "voltage_ekf_v"],
                title=f"Voltage Detail: Cycle {selected_cycle}",
            )
            st.plotly_chart(cycle_voltage_fig, use_container_width=True)

            cycle_state_cols = st.columns(2)
            with cycle_state_cols[0]:
                cycle_soc_fig = px.line(
                    cycle_frame,
                    x="time_s",
                    y=["soc", "soc_ekf"],
                    title="SOC Detail",
                )
                st.plotly_chart(cycle_soc_fig, use_container_width=True)
            with cycle_state_cols[1]:
                cycle_thermal_fig = px.line(
                    cycle_frame,
                    x="time_s",
                    y=["current_a", "temperature_c"],
                    title="Current and Temperature Detail",
                )
                st.plotly_chart(cycle_thermal_fig, use_container_width=True)

    with residual_tab:
        if detail_shadow.empty:
            st.info("No detail samples were found for the selected cycle range.")
        else:
            residual_fig = build_residual_plot(detail_shadow, selected_battery)
            st.plotly_chart(residual_fig, use_container_width=True)

    with parameters_tab:
        if detail_cycle_shadow.empty:
            st.info("No cycle-level rows were found for the selected cycle range.")
        else:
            parameter_signal_fig = build_parameter_signal_plot(detail_cycle_shadow)
            st.plotly_chart(parameter_signal_fig, use_container_width=True)
            st.caption(
                "This plot uses raw `cycle_index` on the x-axis and shows `cycle_type` tags "
                "in hover details."
            )

    with data_tab:
        if detail_shadow.empty:
            st.info("No detail samples were found for the selected cycle range.")
        else:
            download_cols = st.columns(2)
            download_cols[0].download_button(
                "Download Sample CSV",
                data=detail_shadow.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_battery}_samples_{cycle_range[0]}_{cycle_range[1]}.csv",
                mime="text/csv",
            )
            download_cols[1].download_button(
                "Download Cycle CSV",
                data=detail_cycle_shadow.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_battery}_cycles_{cycle_range[0]}_{cycle_range[1]}.csv",
                mime="text/csv",
            )

            with st.expander("View Raw Sample Data", expanded=False):
                st.dataframe(
                    detail_shadow[
                        [
                            "battery_id",
                            "cycle_index",
                            "cycle_type",
                            "discharge_number",
                            "time_s",
                            "voltage_v",
                            "voltage_model_v",
                            "voltage_ekf_v",
                            "current_a",
                            "temperature_c",
                            "soc",
                            "soc_ekf",
                            "soh",
                            "rul_cycles",
                        ]
                    ],
                    use_container_width=True,
                )

        with st.expander("View Cycle-level ECM Table", expanded=False):
            st.dataframe(
                detail_cycle_shadow[
                    [
                        "battery_id",
                        "cycle_index",
                        "cycle_type",
                        "discharge_number",
                        "cycle_voltage_mean_v",
                        "cycle_current_mean_a",
                        "cycle_temperature_mean_c",
                        "r1",
                        "r2",
                        "c1",
                        "c2",
                        "cycle_ecm_mae_v",
                        "cycle_ekf_mae_v",
                    ]
                ].dropna(how="all"),
                use_container_width=True,
            )

    with diagnostics_tab:
        st.markdown("### Technical Diagnostics Logs")
        
        # Anomaly Visualization
        st.markdown("#### Multivariate Anomaly Scores")
        if "anomaly" in battery_cycle_shadow.columns:
            anom_df = battery_cycle_shadow[battery_cycle_shadow["cycle_type"] == "discharge"].copy()
            anom_fig = px.scatter(
                anom_df, x="cycle_index", y="soh", color="anomaly",
                color_discrete_map={True: "red", False: "#2563eb"},
                title="SOH with Flagged Anomalies (Isolation Forest)",
            )
            st.plotly_chart(anom_fig, use_container_width=True)
            
        # --- Group 5 Operating Regimes & Calibration ---
        diag_grid = st.columns(2)
        with diag_grid[0]:
            st.markdown("#### Operating Regimes")
            if "operating_regime" in battery_cycle_shadow.columns:
                regime_df = battery_cycle_shadow[battery_cycle_shadow["cycle_type"] == "discharge"].copy()
                regime_fig = px.scatter(
                    regime_df, x="temperature_mean_c", y="current_mean_a", 
                    color="operating_regime", size="dod",
                    title="Discharge Conditions by Regime",
                    labels={"temperature_mean_c": "Temp (°C)", "current_mean_a": "Current (A)"}
                )
                st.plotly_chart(regime_fig, use_container_width=True)
            else:
                st.info("Operating regime data not available.")

        with diag_grid[1]:
            st.markdown("#### SOH Model Calibration")
            cal_df = global_data.get("calibration", {}).get(selected_battery)
            if cal_df is not None and not cal_df.empty:
                cal_fig = go.Figure()
                cal_fig.add_trace(go.Scatter(x=cal_df["expected_coverage"], y=cal_df["coverage"], mode='markers+lines', name="Actual Coverage"))
                cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name="Perfect Calibration"))
                cal_fig.update_layout(title="90% CI Reliability Diagram", xaxis_title="Expected", yaxis_title="Observed", height=400)
                st.plotly_chart(cal_fig, use_container_width=True)
            else:
                st.info("Calibration data not available for this battery.")
        with st.expander("Global & Battery ECM Metrics", expanded=False):
            diagnostics_cols = st.columns(2)
            diagnostics_cols[0].json(
                battery_ecm_metrics.get(selected_battery, ecm_metrics)
            )
            diagnostics_cols[1].json(
                battery_ecm_params.get(selected_battery, ecm_params)
            )

    with st.expander("Artifact Manifest"):
        st.json(manifest)


if __name__ == "__main__":
    main()
