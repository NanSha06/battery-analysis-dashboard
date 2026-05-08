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

## 🛠️ Infrastructure (V3 Upgrades)
- **Fleet Safety Audit CLI:** `python scripts/safety_audit.py` for automated risk reporting.
- **Physics-Informed Unit Tests:** Automated validation of SOP and Arrhenius logic via `pytest`.
- **Cycle-Level Caching:** Parquet-based caching reduces incremental run times by >80%.
- **Experiment Tracking:** Automated run metadata and configuration hashing for reproducibility.
- **Cross-Platform Robustness:** Full UTF-8 support and environment configuration for Windows stability.

## 📖 Documentation
For a detailed mathematical and architectural breakdown, see the [Technical Report](report.md).

---
*Developed for the Li-ion Battery Digital Shadow project — NASA Battery Dataset (B0005, B0006, B0007, B0018)*
