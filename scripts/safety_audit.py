import argparse
import pandas as pd
import json
from pathlib import Path

def run_audit(artifact_dir: str):
    artifact_path = Path(artifact_dir)
    cycle_table_path = artifact_path / "cycle_shadow.parquet"
    
    if not cycle_table_path.exists():
        print(f"Error: {cycle_table_path} not found.")
        return

    df = pd.read_parquet(cycle_table_path)
    
    # Identify batteries at risk
    high_plating = df[df["plating_risk"] > 0.5]["battery_id"].unique()
    low_sop = df[df["sop_w"] < 5.0]["battery_id"].unique()
    
    critical_batteries = set(high_plating) | set(low_sop)
    
    print("=== BATTERY SAFETY AUDIT REPORT ===")
    print(f"Total batteries analyzed: {len(df['battery_id'].unique())}")
    print(f"Batteries with high plating risk (> 0.5): {len(high_plating)}")
    print(f"Batteries with low SOP (< 5.0W): {len(low_sop)}")
    print("-" * 35)
    
    if not critical_batteries:
        print("RESULT: ALL BATTERIES OPERATING WITHIN SAFE BOUNDS.")
    else:
        print(f"RESULT: {len(critical_batteries)} BATTERIES REQUIRE IMMEDIATE ATTENTION.")
        for bid in critical_batteries:
            status = []
            if bid in high_plating: status.append("High Plating Risk")
            if bid in low_sop: status.append("Low SOP (Power Fade)")
            print(f" - {bid}: {', '.join(status)}")
    
    print("-" * 35)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()
    run_audit(args.artifact_dir)
