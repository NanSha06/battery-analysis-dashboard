from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def artifact_root(root_dir: str | Path) -> Path:
    return Path(root_dir)


def load_manifest(root_dir: str | Path) -> dict[str, Any]:
    path = artifact_root(root_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Dashboard manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_global_tables(root_dir: str | Path) -> dict[str, pd.DataFrame]:
    manifest = load_manifest(root_dir)
    global_tables = {}
    for name, path_str in manifest.get("global_tables", {}).items():
        global_tables[name] = pd.read_parquet(path_str)
    return global_tables


def load_metrics(root_dir: str | Path) -> dict[str, float]:
    manifest = load_manifest(root_dir)
    metrics_path = Path(manifest["metrics_path"])
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_ecm_params(root_dir: str | Path) -> dict[str, float]:
    manifest = load_manifest(root_dir)
    params_path = Path(manifest["params_path"])
    return json.loads(params_path.read_text(encoding="utf-8"))


def load_battery_metrics(root_dir: str | Path) -> dict[str, dict[str, float]]:
    manifest = load_manifest(root_dir)
    path = Path(manifest["battery_metrics_path"])
    return json.loads(path.read_text(encoding="utf-8"))


def load_battery_ecm_params(root_dir: str | Path) -> dict[str, dict[str, float]]:
    manifest = load_manifest(root_dir)
    path = Path(manifest["battery_params_path"])
    return json.loads(path.read_text(encoding="utf-8"))


def load_ecm_consistency(root_dir: str | Path) -> dict[str, Any]:
    manifest = load_manifest(root_dir)
    path_str = manifest.get("ecm_consistency_path")
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def available_battery_ids(root_dir: str | Path, table_kind: str = "sample_shadow") -> list[str]:
    manifest = load_manifest(root_dir)
    base_dir_key = "sample_shadow_dir" if table_kind == "sample_shadow" else "sample_dir"
    base_dir = Path(manifest[base_dir_key])
    if not base_dir.exists():
        return []
    return sorted(path.stem for path in base_dir.glob("*.parquet"))


def load_battery_table(
    root_dir: str | Path,
    battery_id: str,
    table_kind: str = "sample_shadow",
    cycle_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    manifest = load_manifest(root_dir)
    base_dir_key = "sample_shadow_dir" if table_kind == "sample_shadow" else "sample_dir"
    path = Path(manifest[base_dir_key]) / f"{battery_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Battery artifact not found: {path}")

    frame = pd.read_parquet(path)
    if cycle_range is not None and "cycle_index" in frame.columns:
        start_cycle, end_cycle = cycle_range
        frame = frame[(frame["cycle_index"] >= start_cycle) & (frame["cycle_index"] <= end_cycle)].copy()
    return frame.reset_index(drop=True)


def summarize_battery(cycle_shadow: pd.DataFrame, battery_id: str) -> dict[str, float]:
    battery_frame = cycle_shadow[cycle_shadow["battery_id"] == battery_id].copy()
    discharge = battery_frame[battery_frame["cycle_type"] == "discharge"].copy()
    if discharge.empty:
        return {
            "latest_soh": float("nan"),
            "latest_rul_cycles": float("nan"),
            "latest_capacity_ah": float("nan"),
            "discharge_cycles": 0,
        }

    latest = discharge.sort_values("cycle_index").iloc[-1]
    return {
        "latest_soh": float(latest.get("soh", float("nan"))),
        "latest_rul_cycles": float(latest.get("rul_cycles", float("nan"))),
        "latest_capacity_ah": float(latest.get("capacity_ah", float("nan"))),
        "discharge_cycles": int(len(discharge)),
    }
