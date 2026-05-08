from __future__ import annotations

import numpy as np
import pandas as pd

def compute_soh_calibration(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes calibration metrics for SOH predictions.
    Checks what fraction of actual SOH values fall within the 90% CI.
    """
    # Only use discharge cycles with valid actual and predicted SOH
    df = cycle_df[cycle_df["cycle_type"] == "discharge"].dropna(
        subset=["soh", "soh_model_pred", "soh_model_lower", "soh_model_upper"]
    ).copy()
    
    if df.empty:
        return pd.DataFrame(columns=["decile", "coverage", "expected_coverage"])
    
    # Define deciles based on predicted SOH
    df["decile"] = pd.qcut(df["soh_model_pred"], 10, labels=False, duplicates='drop')
    
    # Check if actual SOH is within the 90% CI
    df["in_90_ci"] = (df["soh"] >= df["soh_model_lower"]) & (df["soh"] <= df["soh_model_upper"])
    
    calibration = (
        df.groupby("decile")
        .agg(
            coverage=("in_90_ci", "mean"),
        )
        .reset_index()
    )
    
    calibration["expected_coverage"] = 0.90
    
    # Map decile index to actual range for clarity if needed, 
    # but prompt asks for [decile, coverage, expected_coverage]
    return calibration
