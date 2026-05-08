# Li-ion Battery Digital Shadow Platform (V3)

A data-driven digital twin and diagnostic platform for Li-ion batteries, featuring advanced state estimation, probabilistic health forecasting, and multivariate anomaly detection.

![Dashboard Preview](https://img.shields.io/badge/Status-Operational-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)

## 🌟 Key Features

### 🔍 Autonomous Physics-Based Diagnostics
- **State of Power (SOP):** Real-time peak power forecasting using state-dependent resistance.
- **Lithium Plating Index:** Risk-based safety monitoring for high-rate/low-temp charging.
- **Knee-point Detection:** Automated identification of accelerated aging phases (Kneedle algorithm).
- **Sage-Husa Adaptive EKF:** High-fidelity SOC tracking with dynamic noise adaptation.

### 📈 Reliability & Risk Forecasting
- **Monte Carlo GPR Simulation:** Probabilistic RUL trajectories with p5 through p95 confidence bands.
- **Arrhenius Capacity Fade:** Semi-empirical thermal degradation modeling for long-term health.
- **SOH Calibration Analytics:** Automated coverage checks and reliability diagrams for model validation.

### 🛡️ Operational Intelligence
- **Maintenance Decision Engine:** Rule-based protocol recommendations (Normal, Reduce C-rate, Replace).
- **What-if Scenario Simulator:** Interactive stress testing for temperature and C-rate impacts on RUL.
- **Operating Regime Clustering:** KMeans analysis of usage patterns to identify high-degradation behaviors.
- **Safety Audit CLI:** Rapid fleet-wide risk screening for plating and power fade.

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Prepare Data Artifacts
The pipeline precomputes all analytical metrics, uncertainty bands, and anomaly flags.
```bash
python scripts/prepare_dashboard_data.py --mat-dir mat_files --output-dir artifacts
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

## 📋 Data Format Specifications

The platform supports multiple ingestion formats. Data should be placed in the `mat_files/` directory (or as configured in `config.yaml`).

### 1. NASA .mat (Legacy)
Standard NASA Prognostics Data Repository format. The pipeline automatically extracts cycles, temperatures, and impedances.

### 2. JSON Format
JSON files should follow this structure:
```json
{
  "battery_id": "B0005",
  "cycles": [
    {
      "cycle_index": 0,
      "cycle_type": "discharge",
      "data": {
        "Time": [0, 1, 2],
        "Voltage_measured": [4.2, 4.1, 4.0],
        "Current_measured": [-1, -1, -1]
      }
    }
  ]
}
```

### 3. Excel (.xlsx)
Workbooks should contain a `cycle_index` column. All other columns are treated as time-series telemetry.

## ⚙️ Configuration (`config.yaml`)

Manage paths and model thresholds without touching code:

```yaml
paths:
  raw_data: "mat_files"
  artifacts: "artifacts"
models:
  soh_threshold: 0.70
  nominal_capacity_ah: 2.0
```

## 🐳 Docker Deployment

To run the platform in a containerized environment:

```bash
docker build -t battery-shadow .
docker run -p 8501:8501 battery-shadow
```

## 🧪 Testing

Run the test suite using `pytest`:

```bash
export PYTHONPATH=$PYTHONPATH:.
pytest tests/
```

## 📖 Documentation
For a detailed mathematical and architectural breakdown, see the [Technical Report](report.md).

---
*Developed for the Li-ion Battery Digital Shadow project — NASA Battery Dataset (B0005, B0006, B0007, B0018)*
