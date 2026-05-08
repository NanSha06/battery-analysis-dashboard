from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import json
import yaml
import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from scipy.io import loadmat


REQUIRED_BATTERY_IDS = ("B0005", "B0006", "B0007", "B0018")


@dataclass
class BatteryDataset:
    battery_id: str
    cycles: list[dict[str, Any]]


def _to_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        if value.size == 1:
            return _to_scalar(value.reshape(-1)[0])
        return value.squeeze()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _to_string(value: Any) -> str:
    value = _to_scalar(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.reshape(-1).tolist()).strip()
        return str(value.tolist())
    return str(value).strip()


def _to_vector(value: Any) -> np.ndarray:
    value = _to_scalar(value)
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.asarray([], dtype=float)
    return np.asarray([float(value)], dtype=float)


def _to_datetime(value: Any) -> datetime | None:
    raw = _to_scalar(value)
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1).tolist()
    if not isinstance(raw, (list, tuple)) or len(raw) < 6:
        return None
    try:
        year, month, day, hour, minute, second = [int(float(v)) for v in raw[:6]]
        second = max(0, min(second, 59))
        return datetime(year, month, day, hour, minute, second)
    except (TypeError, ValueError):
        return None


def _extract_mat_root(mat_dict: dict[str, Any], battery_id: str) -> Any:
    if battery_id in mat_dict:
        return mat_dict[battery_id]
    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue
        if hasattr(value, "dtype") or hasattr(value, "_fieldnames"):
            return value
    raise KeyError(f"Could not find root struct for {battery_id}")


def _field_value(struct: Any, field_name: str) -> Any:
    if struct is None:
        return np.nan
    if hasattr(struct, field_name):
        return getattr(struct, field_name)
    if isinstance(struct, np.void) and field_name in struct.dtype.names:
        return struct[field_name]
    if isinstance(struct, dict):
        return struct.get(field_name, np.nan)
    return np.nan


def _iter_cycles(root_struct: Any) -> list[Any]:
    cycle_value = _field_value(root_struct, "cycle")
    cycle_value = _to_scalar(cycle_value)
    if isinstance(cycle_value, np.ndarray):
        return list(cycle_value.reshape(-1))
    if isinstance(cycle_value, list):
        return cycle_value
    return [cycle_value]


def load_battery_cycles(file_path: str | Path) -> BatteryDataset:
    file_path = Path(file_path)
    battery_id = file_path.stem
    ext = file_path.suffix.lower()

    try:
        if ext == ".mat":
            return _load_mat_battery(file_path, battery_id)
        elif ext == ".json":
            return _load_json_battery(file_path, battery_id)
        elif ext == ".xlsx":
            return _load_excel_battery(file_path, battery_id)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return BatteryDataset(battery_id=battery_id, cycles=[])


def _load_mat_battery(mat_path: Path, battery_id: str) -> BatteryDataset:
    mat_dict = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    root = _extract_mat_root(mat_dict, battery_id)
    cycles = []

    for index, cycle in enumerate(_iter_cycles(root)):
        cycle_type = _to_string(_field_value(cycle, "type")).lower()
        cycle_time = _to_datetime(_field_value(cycle, "time"))
        ambient_temperature = _to_scalar(_field_value(cycle, "ambient_temperature"))
        data = _field_value(cycle, "data")
        measurements = {}

        field_names = getattr(data, "_fieldnames", None)
        if field_names:
            for field_name in field_names:
                measurements[field_name] = _field_value(data, field_name)
        elif isinstance(data, np.void) and data.dtype.names:
            for field_name in data.dtype.names:
                measurements[field_name] = data[field_name]

        cycles.append(
            {
                "battery_id": battery_id,
                "cycle_index": index,
                "cycle_type": cycle_type,
                "timestamp": cycle_time,
                "ambient_temperature": ambient_temperature,
                "data": measurements,
            }
        )

    return BatteryDataset(battery_id=battery_id, cycles=cycles)


def _load_json_battery(path: Path, battery_id: str) -> BatteryDataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expected format: {"battery_id": "...", "cycles": [...]}
    return BatteryDataset(
        battery_id=data.get("battery_id", battery_id),
        cycles=data.get("cycles", [])
    )


def _load_excel_battery(path: Path, battery_id: str) -> BatteryDataset:
    # Minimal Excel support: each sheet is a cycle or one sheet with cycle_index col
    df = pd.read_excel(path)
    cycles = []
    if "cycle_index" in df.columns:
        for idx, group in df.groupby("cycle_index"):
            cycles.append({
                "battery_id": battery_id,
                "cycle_index": int(idx),
                "cycle_type": group["cycle_type"].iloc[0] if "cycle_type" in group.columns else "unknown",
                "data": {col: group[col].values for col in group.columns if col not in ["battery_id", "cycle_index", "cycle_type"]}
            })
    return BatteryDataset(battery_id=battery_id, cycles=cycles)


def load_all_batteries(data_dir: str | Path) -> dict[str, BatteryDataset]:
    data_dir = Path(data_dir)
    datasets: dict[str, BatteryDataset] = {}
    
    # Try different extensions
    for ext in [".mat", ".json", ".xlsx"]:
        for path in sorted(data_dir.glob(f"*{ext}")):
            if path.stem in datasets: continue
            datasets[path.stem] = load_battery_cycles(path)
            
    return datasets


def build_cycle_table(datasets: dict[str, BatteryDataset]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in datasets.values():
        for cycle in dataset.cycles:
            data = cycle["data"]
            time_values = _to_vector(data.get("Time"))
            voltage = _to_vector(data.get("Voltage_measured"))
            current = _to_vector(data.get("Current_measured"))
            temperature = _to_vector(data.get("Temperature_measured"))

            rows.append(
                {
                    "battery_id": cycle["battery_id"],
                    "cycle_index": cycle["cycle_index"],
                    "cycle_type": cycle["cycle_type"],
                    "timestamp": cycle["timestamp"],
                    "ambient_temperature": cycle["ambient_temperature"],
                    "sample_count": len(time_values),
                    "duration_s": float(time_values[-1] - time_values[0]) if len(time_values) > 1 else np.nan,
                    "capacity_ah": _to_scalar(data.get("Capacity")),
                    "re_ohm": _to_scalar(data.get("Re")),
                    "rct_ohm": _to_scalar(data.get("Rct")),
                    "total_resistance_ohm": float((_to_scalar(data.get("Re")) or 0.0) + (_to_scalar(data.get("Rct")) or 0.0)),
                    "voltage_min_v": float(np.min(voltage)) if len(voltage) else np.nan,
                    "voltage_max_v": float(np.max(voltage)) if len(voltage) else np.nan,
                    "current_mean_a": float(np.mean(current)) if len(current) else np.nan,
                    "temperature_mean_c": float(np.mean(temperature)) if len(temperature) else np.nan,
                    "temperature_max_c": float(np.max(temperature)) if len(temperature) else np.nan,
                }
            )

    cycle_table = pd.DataFrame(rows)
    if not cycle_table.empty:
        cycle_table = cycle_table.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)
    return cycle_table


def build_sample_table(datasets: dict[str, BatteryDataset]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset in datasets.values():
        for cycle in dataset.cycles:
            data = cycle["data"]
            time_values = _to_vector(data.get("Time"))
            if len(time_values) == 0:
                continue

            fields = {
                "voltage_v": _to_vector(data.get("Voltage_measured")),
                "current_a": _to_vector(data.get("Current_measured")),
                "temperature_c": _to_vector(data.get("Temperature_measured")),
                "load_current_a": _to_vector(data.get("Current_charge")),
                "load_voltage_v": _to_vector(data.get("Voltage_charge")),
            }

            size = len(time_values)
            for sample_index in range(size):
                record = {
                    "battery_id": cycle["battery_id"],
                    "cycle_index": cycle["cycle_index"],
                    "cycle_type": cycle["cycle_type"],
                    "timestamp": cycle["timestamp"],
                    "ambient_temperature": cycle["ambient_temperature"],
                    "sample_index": sample_index,
                    "time_s": float(time_values[sample_index]),
                }
                for output_name, vector in fields.items():
                    record[output_name] = float(vector[sample_index]) if sample_index < len(vector) else np.nan
                records.append(record)

    sample_table = pd.DataFrame(records)
    if not sample_table.empty:
        sample_table = sample_table.sort_values(
            ["battery_id", "cycle_index", "sample_index"]
        ).reset_index(drop=True)
    return sample_table


CYCLE_SCHEMA = DataFrameSchema({
    "cycle_index":           Column(int,   Check.ge(0)),
    "cycle_type":            Column(str,   Check.isin(["charge", "discharge", "impedance"])),
    "capacity_ah":           Column(float, Check.ge(0),   nullable=True),
    "temperature_mean_c":    Column(float, Check.between(-20, 80), nullable=True),
    "total_resistance_ohm":  Column(float, Check.ge(0),   nullable=True),
})

SAMPLE_SCHEMA = DataFrameSchema({
    "cycle_index":  Column(int,   Check.ge(0)),
    "voltage_v":    Column(float, Check.between(0, 5)),
    "current_a":    Column(float, Check.between(-10, 10)),
    "temperature_c":Column(float, Check.between(-20, 80), nullable=True),
})


def validate_tables(cycle_table: pd.DataFrame, sample_table: pd.DataFrame, battery_id: str = "Batch") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates tables and drops invalid rows.
    """
    try:
        clean_cycle = CYCLE_SCHEMA.validate(cycle_table, lazy=True)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
        print(f"[{battery_id}] Cycle table has validation violations. Dropping invalid rows.")
        if hasattr(err, "failure_cases"):
            clean_cycle = cycle_table.drop(err.failure_cases.index, errors="ignore")
        else:
            clean_cycle = cycle_table.dropna() # Fallback

    try:
        clean_sample = SAMPLE_SCHEMA.validate(sample_table, lazy=True)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
        print(f"[{battery_id}] Sample table has validation violations. Dropping invalid rows.")
        if hasattr(err, "failure_cases"):
            invalid_indices = err.failure_cases["index"].dropna().unique()
            clean_sample = sample_table.drop(invalid_indices, errors="ignore")
        else:
            clean_sample = sample_table.dropna(subset=["voltage_v", "current_a"])

    return clean_cycle, clean_sample


def load_shadow_tables(mat_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = load_all_batteries(mat_dir)
    cycle_table = build_cycle_table(datasets)
    sample_table = build_sample_table(datasets)
    
    cycle_table, sample_table = validate_tables(cycle_table, sample_table)
    
    return cycle_table, sample_table
