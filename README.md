# Li-ion Battery Digital Shadow Platform (V2)

A data-driven digital twin and diagnostic platform for Li-ion batteries, featuring advanced state estimation, probabilistic health forecasting, and multivariate anomaly detection.

![Dashboard Preview](https://img.shields.io/badge/Status-Operational-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)

## 🌟 Key Features

### 🔍 Advanced State Estimation
- **Sage-Husa Adaptive EKF:** Real-time SOC tracking with dynamic covariance adjustment for varying noise environments.
- **2-RC Equivalent Circuit Modeling:** High-fidelity voltage reconstruction with state-dependent parameter modulation.
- **OCV-SOC Anchor Correction:** Automated drift correction using electrochemical rest-period anchors.

### 📈 Probabilistic Health & RUL
- **Bayesian SOH Fusion:** Multi-signal health estimation (Capacity + Resistance) with **90% Confidence Intervals**.
- **Gaussian Process Regression (GPR):** Non-linear Remaining Useful Life (RUL) projections with probabilistic uncertainty bands.
- **Pooled Degradation Modeling:** Cross-battery fleet analysis using Leave-One-Group-Out (LOGO) cross-validation.

### 🛡️ Diagnostic Intelligence
- **Multivariate Anomaly Detection:** Isolation Forest model identifying complex degradation signatures across health and thermal domains.
- **Transient Impedance Validation:** Real-time R0 validation against physical voltage-pulse responses.
- **Schema Enforcement:** Strict data quality control using **Pandera** validation during ingestion.

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

## 🛠️ Infrastructure (V2 Upgrades)
- **Cycle-Level Caching:** Parquet-based caching reduces incremental run times by >80%.
- **Experiment Tracking:** Automated run metadata and configuration hashing for reproducibility.
- **Cross-Platform Robustness:** Full UTF-8 support and environment configuration for Windows stability.

## 📖 Documentation
For a detailed mathematical and architectural breakdown, see the [Technical Report](report.md).

---
*Developed for the Li-ion Battery Digital Shadow project — NASA Battery Dataset (B0005, B0006, B0007, B0018)*
