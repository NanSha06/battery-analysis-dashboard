import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import build_cycle_table, BatteryDataset

def test_build_cycle_table_empty():
    datasets = {}
    df = build_cycle_table(datasets)
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_build_cycle_table_mock():
    mock_data = {
        "Time": np.array([0, 1, 2]),
        "Voltage_measured": np.array([4.2, 4.1, 4.0]),
        "Current_measured": np.array([-1, -1, -1]),
        "Temperature_measured": np.array([25, 26, 27]),
        "Capacity": 1.8
    }
    dataset = BatteryDataset(
        battery_id="MOCK_BATT",
        cycles=[{
            "battery_id": "MOCK_BATT",
            "cycle_index": 0,
            "cycle_type": "discharge",
            "timestamp": None,
            "ambient_temperature": 25,
            "data": mock_data
        }]
    )
    df = build_cycle_table({"MOCK_BATT": dataset})
    assert len(df) == 1
    assert df.iloc[0]["battery_id"] == "MOCK_BATT"
    assert df.iloc[0]["capacity_ah"] == 1.8
    assert df.iloc[0]["voltage_min_v"] == 4.0
