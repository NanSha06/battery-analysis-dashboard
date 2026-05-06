from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class ECMParameters:
    r0: float
    r1: float
    c1: float
    r2: float
    c2: float


@dataclass
class EKFParameters:
    process_var_soc: float = 1e-6
    process_var_v1: float = 1e-5
    process_var_v2: float = 1e-5
    measurement_var_v: float = 2.5e-3
    initial_cov_soc: float = 1e-2
    initial_cov_v1: float = 1e-2
    initial_cov_v2: float = 1e-2


def estimate_ocv_curve(sample_table: pd.DataFrame, soc_col: str = "soc") -> pd.DataFrame:
    usable = sample_table.dropna(subset=[soc_col, "voltage_v"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=["soc", "ocv_v"])

    usable["soc_bin"] = np.clip((usable[soc_col] * 20).round() / 20.0, 0.0, 1.0)
    ocv_curve = usable.groupby("soc_bin", as_index=False)["voltage_v"].median()
    ocv_curve.columns = ["soc", "ocv_v"]
    return ocv_curve.sort_values("soc").reset_index(drop=True)


def interpolate_ocv(soc: np.ndarray, ocv_curve: pd.DataFrame) -> np.ndarray:
    if ocv_curve.empty:
        return np.full_like(soc, fill_value=np.nan, dtype=float)
    return np.interp(
        soc,
        ocv_curve["soc"].to_numpy(dtype=float),
        ocv_curve["ocv_v"].to_numpy(dtype=float),
        left=float(ocv_curve["ocv_v"].iloc[0]),
        right=float(ocv_curve["ocv_v"].iloc[-1]),
    )


def ocv_from_soc(soc_value: float, ocv_curve: pd.DataFrame) -> float:
    if ocv_curve.empty:
        return np.nan
    clipped_soc = float(np.clip(soc_value, 0.0, 1.0))
    return float(
        np.interp(
            clipped_soc,
            ocv_curve["soc"].to_numpy(dtype=float),
            ocv_curve["ocv_v"].to_numpy(dtype=float),
            left=float(ocv_curve["ocv_v"].iloc[0]),
            right=float(ocv_curve["ocv_v"].iloc[-1]),
        )
    )


def ocv_slope_from_soc(soc_value: float, ocv_curve: pd.DataFrame) -> float:
    if ocv_curve.empty or len(ocv_curve) < 2:
        return 0.0
    soc_points = ocv_curve["soc"].to_numpy(dtype=float)
    ocv_points = ocv_curve["ocv_v"].to_numpy(dtype=float)
    gradients = np.gradient(ocv_points, soc_points)
    clipped_soc = float(np.clip(soc_value, float(soc_points.min()), float(soc_points.max())))
    return float(
        np.interp(
            clipped_soc,
            soc_points,
            gradients,
            left=float(gradients[0]),
            right=float(gradients[-1]),
        )
    )


def get_dynamic_params(
    soc: pd.Series | np.ndarray,
    temperature_c: pd.Series | np.ndarray,
    soh: pd.Series | np.ndarray,
    base_params: ECMParameters,
    reference_temperature_c: float = 25.0,
) -> pd.DataFrame:
    soc_values = np.asarray(soc, dtype=float)
    temp_values = np.asarray(temperature_c, dtype=float)
    soh_values = np.asarray(soh, dtype=float)

    soc_stress = np.clip(np.abs(soc_values - 0.5) * 2.0, 0.0, 1.0)
    temp_stress = np.clip((reference_temperature_c - temp_values) / 25.0, -0.5, 1.5)
    aging_stress = np.clip(1.0 - soh_values, 0.0, 0.6)

    r_scale = 1.0 + 0.25 * soc_stress + 0.20 * temp_stress + 1.20 * aging_stress
    c_scale = np.clip(1.0 - 0.45 * aging_stress + 0.08 * (temp_values - reference_temperature_c) / 25.0, 0.2, 1.5)

    return pd.DataFrame(
        {
            "r0_dynamic": np.clip(base_params.r0 * r_scale, 1e-5, 1.0),
            "r1_dynamic": np.clip(base_params.r1 * (1.0 + 0.35 * soc_stress + 0.15 * temp_stress), 1e-5, 1.0),
            "c1_dynamic": np.clip(base_params.c1 * c_scale, 1.0, 1e6),
            "r2_dynamic": np.clip(base_params.r2 * (1.0 + 0.25 * soc_stress + 0.20 * temp_stress), 1e-5, 1.0),
            "c2_dynamic": np.clip(base_params.c2 * c_scale, 1.0, 1e6),
        }
    )


def fit_parameter_surface(sample_table: pd.DataFrame, base_params: ECMParameters) -> pd.DataFrame:
    if sample_table.empty:
        return pd.DataFrame(columns=["soc_bin", "temperature_bin_c", "r0_dynamic", "r1_dynamic", "c1_dynamic", "r2_dynamic", "c2_dynamic"])

    working = sample_table.copy()
    if "soh" not in working:
        working["soh"] = 1.0
    dynamic = get_dynamic_params(
        working["soc"].ffill().bfill().fillna(0.5),
        working["temperature_c"].ffill().bfill().fillna(25.0),
        working["soh"].ffill().bfill().fillna(1.0),
        base_params,
    )
    working = pd.concat([working.reset_index(drop=True), dynamic.reset_index(drop=True)], axis=1)
    working["soc_bin"] = (working["soc"].clip(0.0, 1.0) * 10).round() / 10.0
    working["temperature_bin_c"] = (working["temperature_c"] / 5.0).round() * 5.0
    return (
        working.groupby(["soc_bin", "temperature_bin_c"], as_index=False)[
            ["r0_dynamic", "r1_dynamic", "c1_dynamic", "r2_dynamic", "c2_dynamic"]
        ]
        .mean()
        .sort_values(["soc_bin", "temperature_bin_c"])
        .reset_index(drop=True)
    )


def simulate_2rc_ecm(
    current_a: np.ndarray,
    dt_s: np.ndarray,
    soc: np.ndarray,
    params: ECMParameters,
    ocv_curve: pd.DataFrame,
) -> pd.DataFrame:
    current_a = np.asarray(current_a, dtype=float)
    dt_s = np.asarray(dt_s, dtype=float)
    soc = np.asarray(soc, dtype=float)
    ocv = interpolate_ocv(soc, ocv_curve)

    v1 = np.zeros_like(current_a, dtype=float)
    v2 = np.zeros_like(current_a, dtype=float)
    terminal_v = np.zeros_like(current_a, dtype=float)

    for i in range(len(current_a)):
        if i > 0:
            a1 = np.exp(-max(dt_s[i], 0.0) / max(params.r1 * params.c1, 1e-9))
            a2 = np.exp(-max(dt_s[i], 0.0) / max(params.r2 * params.c2, 1e-9))
            v1[i] = a1 * v1[i - 1] + params.r1 * (1.0 - a1) * current_a[i]
            v2[i] = a2 * v2[i - 1] + params.r2 * (1.0 - a2) * current_a[i]
        terminal_v[i] = ocv[i] - current_a[i] * params.r0 - v1[i] - v2[i]

    return pd.DataFrame(
        {
            "ocv_v": ocv,
            "v_rc1": v1,
            "v_rc2": v2,
            "voltage_model_v": terminal_v,
        }
    )


def _residuals(
    theta: np.ndarray,
    current_a: np.ndarray,
    dt_s: np.ndarray,
    soc: np.ndarray,
    measured_v: np.ndarray,
    ocv_curve: pd.DataFrame,
) -> np.ndarray:
    params = ECMParameters(*theta)
    simulated = simulate_2rc_ecm(current_a, dt_s, soc, params, ocv_curve)
    return simulated["voltage_model_v"].to_numpy(dtype=float) - measured_v


def fit_2rc_parameters(
    sample_table: pd.DataFrame,
    ocv_curve: pd.DataFrame,
) -> ECMParameters:
    usable = sample_table.dropna(subset=["current_a", "dt_s", "soc", "voltage_v"]).copy()
    if usable.empty:
        return ECMParameters(0.01, 0.01, 2000.0, 0.02, 4000.0)

    theta0 = np.asarray([0.01, 0.01, 2000.0, 0.02, 4000.0], dtype=float)
    lower = np.asarray([1e-5, 1e-5, 1.0, 1e-5, 1.0], dtype=float)
    upper = np.asarray([1.0, 1.0, 1e6, 1.0, 1e6], dtype=float)

    result = least_squares(
        _residuals,
        theta0,
        bounds=(lower, upper),
        args=(
            usable["current_a"].to_numpy(dtype=float),
            usable["dt_s"].to_numpy(dtype=float),
            usable["soc"].to_numpy(dtype=float),
            usable["voltage_v"].to_numpy(dtype=float),
            ocv_curve,
        ),
    )
    return ECMParameters(*result.x.tolist())


def run_ekf_soc_ocv(
    sample_table: pd.DataFrame,
    params: ECMParameters,
    ocv_curve: pd.DataFrame,
    nominal_capacity_ah: float = 2.0,
    ekf_params: EKFParameters | None = None,
) -> pd.DataFrame:
    if sample_table.empty:
        return sample_table.copy()

    ekf_params = ekf_params or EKFParameters()
    frame = sample_table.copy().reset_index(drop=True)

    frame["soc_ekf"] = np.nan
    frame["ocv_ekf_v"] = np.nan
    frame["v_rc1_ekf"] = np.nan
    frame["v_rc2_ekf"] = np.nan
    frame["voltage_ekf_v"] = np.nan
    frame["voltage_residual_ekf_v"] = np.nan

    q = np.diag(
        [
            ekf_params.process_var_soc,
            ekf_params.process_var_v1,
            ekf_params.process_var_v2,
        ]
    )
    r = np.asarray([[ekf_params.measurement_var_v]], dtype=float)

    for (_, _), group in frame.groupby(["battery_id", "cycle_index"], sort=False):
        idx = group.index.to_numpy()
        current = group["current_a"].fillna(0.0).to_numpy(dtype=float)
        dt_s = group["dt_s"].fillna(0.0).to_numpy(dtype=float)
        voltage = group["voltage_v"].to_numpy(dtype=float)
        seed_soc = group["soc"].ffill().bfill().fillna(0.5).to_numpy(dtype=float)

        x = np.asarray([float(seed_soc[0]), 0.0, 0.0], dtype=float)
        p = np.diag(
            [
                ekf_params.initial_cov_soc,
                ekf_params.initial_cov_v1,
                ekf_params.initial_cov_v2,
            ]
        )

        soc_out = np.zeros(len(group), dtype=float)
        ocv_out = np.zeros(len(group), dtype=float)
        v1_out = np.zeros(len(group), dtype=float)
        v2_out = np.zeros(len(group), dtype=float)
        v_out = np.zeros(len(group), dtype=float)
        residual_out = np.zeros(len(group), dtype=float)

        for i in range(len(group)):
            dt = max(float(dt_s[i]), 0.0)
            ik = float(current[i])

            a1 = np.exp(-dt / max(params.r1 * params.c1, 1e-9))
            a2 = np.exp(-dt / max(params.r2 * params.c2, 1e-9))
            b1 = params.r1 * (1.0 - a1)
            b2 = params.r2 * (1.0 - a2)

            soc_pred = float(np.clip(x[0] - ik * dt / (3600.0 * nominal_capacity_ah), 0.0, 1.0))
            v1_pred = float(a1 * x[1] + b1 * ik)
            v2_pred = float(a2 * x[2] + b2 * ik)
            x_pred = np.asarray([soc_pred, v1_pred, v2_pred], dtype=float)

            f = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, a1, 0.0],
                    [0.0, 0.0, a2],
                ],
                dtype=float,
            )
            p_pred = f @ p @ f.T + q

            ocv_pred = ocv_from_soc(x_pred[0], ocv_curve)
            dh_dsoc = ocv_slope_from_soc(x_pred[0], ocv_curve)
            h = np.asarray([[ocv_pred - ik * params.r0 - x_pred[1] - x_pred[2]]], dtype=float)
            h_jacobian = np.asarray([[dh_dsoc, -1.0, -1.0]], dtype=float)

            if np.isfinite(voltage[i]):
                innovation = np.asarray([[float(voltage[i])]], dtype=float) - h
                s = h_jacobian @ p_pred @ h_jacobian.T + r
                k = p_pred @ h_jacobian.T @ np.linalg.pinv(s)
                x = x_pred + (k @ innovation).reshape(-1)
                x[0] = float(np.clip(x[0], 0.0, 1.0))
                p = (np.eye(3) - k @ h_jacobian) @ p_pred
                residual_value = float(innovation[0, 0])
            else:
                x = x_pred
                p = p_pred
                residual_value = np.nan

            ocv_value = ocv_from_soc(x[0], ocv_curve)
            voltage_value = float(ocv_value - ik * params.r0 - x[1] - x[2])

            soc_out[i] = x[0]
            ocv_out[i] = ocv_value
            v1_out[i] = x[1]
            v2_out[i] = x[2]
            v_out[i] = voltage_value
            residual_out[i] = residual_value

        frame.loc[idx, "soc_ekf"] = soc_out
        frame.loc[idx, "ocv_ekf_v"] = ocv_out
        frame.loc[idx, "v_rc1_ekf"] = v1_out
        frame.loc[idx, "v_rc2_ekf"] = v2_out
        frame.loc[idx, "voltage_ekf_v"] = v_out
        frame.loc[idx, "voltage_residual_ekf_v"] = residual_out

    return frame


def attach_ecm_state(
    sample_table: pd.DataFrame,
    nominal_capacity_ah: float = 2.0,
    run_ekf: bool = True,
    ekf_params: EKFParameters | None = None,
) -> tuple[pd.DataFrame, ECMParameters, pd.DataFrame]:
    if sample_table.empty:
        empty_curve = pd.DataFrame(columns=["soc", "ocv_v"])
        return sample_table.copy(), ECMParameters(0.01, 0.01, 2000.0, 0.02, 4000.0), empty_curve

    ocv_curve = estimate_ocv_curve(sample_table)
    params = fit_2rc_parameters(sample_table, ocv_curve)
    simulated = simulate_2rc_ecm(
        current_a=sample_table["current_a"].fillna(0.0).to_numpy(dtype=float),
        dt_s=sample_table["dt_s"].fillna(0.0).to_numpy(dtype=float),
        soc=sample_table["soc"].ffill().fillna(0.5).to_numpy(dtype=float),
        params=params,
        ocv_curve=ocv_curve,
    )
    frame = sample_table.copy()
    frame = pd.concat([frame.reset_index(drop=True), simulated.reset_index(drop=True)], axis=1)
    frame["voltage_error_v"] = frame["voltage_model_v"] - frame["voltage_v"]
    if run_ekf:
        frame = run_ekf_soc_ocv(
            frame,
            params=params,
            ocv_curve=ocv_curve,
            nominal_capacity_ah=nominal_capacity_ah,
            ekf_params=ekf_params,
        )
    return frame, params, ocv_curve


def estimate_cycle_ecm_parameters(
    sample_table: pd.DataFrame,
    nominal_capacity_ah: float = 2.0,
    min_samples: int = 20,
    max_samples_per_cycle: int = 300,
) -> pd.DataFrame:
    if sample_table.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []

    for (battery_id, cycle_index), group in sample_table.groupby(["battery_id", "cycle_index"], sort=False):
        cycle_type = str(group["cycle_type"].iloc[0]).lower()
        if cycle_type != "discharge" or len(group) < min_samples:
            continue

        working_group = group.copy()
        if len(working_group) > max_samples_per_cycle:
            positions = np.linspace(0, len(working_group) - 1, max_samples_per_cycle).astype(int)
            working_group = working_group.iloc[positions].copy()

        ocv_curve = estimate_ocv_curve(working_group)
        params = fit_2rc_parameters(working_group, ocv_curve)
        simulated = simulate_2rc_ecm(
            current_a=working_group["current_a"].fillna(0.0).to_numpy(dtype=float),
            dt_s=working_group["dt_s"].fillna(0.0).to_numpy(dtype=float),
            soc=working_group["soc"].ffill().fillna(0.5).to_numpy(dtype=float),
            params=params,
            ocv_curve=ocv_curve,
        )
        enriched = pd.concat([working_group.reset_index(drop=True), simulated.reset_index(drop=True)], axis=1)
        enriched["voltage_error_v"] = enriched["voltage_model_v"] - enriched["voltage_v"]
        metrics = voltage_error_metrics(enriched)

        rows.append(
            {
                "battery_id": battery_id,
                "cycle_index": int(cycle_index),
                "cycle_type": cycle_type,
                "operation_number": float(group["operation_number"].iloc[0]) if "operation_number" in group else np.nan,
                "discharge_number": float(group["discharge_number"].iloc[0]) if "discharge_number" in group else np.nan,
                "r0": float(params.r0),
                "r1": float(params.r1),
                "c1": float(params.c1),
                "r2": float(params.r2),
                "c2": float(params.c2),
                "cycle_voltage_mean_v": float(group["voltage_v"].mean()) if "voltage_v" in group else np.nan,
                "cycle_current_mean_a": float(group["current_a"].mean()) if "current_a" in group else np.nan,
                "cycle_temperature_mean_c": float(group["temperature_c"].mean()) if "temperature_c" in group else np.nan,
                "cycle_ecm_mae_v": float(metrics["mae_v"]),
                "cycle_ecm_rmse_v": float(metrics["rmse_v"]),
                "cycle_ekf_mae_v": np.nan,
                "cycle_ekf_rmse_v": np.nan,
            }
        )

    return pd.DataFrame(rows)


def voltage_error_metrics(sample_table: pd.DataFrame) -> dict[str, float]:
    usable = sample_table.dropna(subset=["voltage_model_v", "voltage_v"]).copy()
    if usable.empty:
        return {"mae_v": np.nan, "rmse_v": np.nan}

    error = usable["voltage_model_v"] - usable["voltage_v"]
    return {
        "mae_v": float(np.mean(np.abs(error))),
        "rmse_v": float(np.sqrt(np.mean(error**2))),
    }


def ekf_voltage_error_metrics(sample_table: pd.DataFrame) -> dict[str, float]:
    usable = sample_table.dropna(subset=["voltage_ekf_v", "voltage_v"]).copy()
    if usable.empty:
        return {"ekf_mae_v": np.nan, "ekf_rmse_v": np.nan}

    error = usable["voltage_ekf_v"] - usable["voltage_v"]
    return {
        "ekf_mae_v": float(np.mean(np.abs(error))),
        "ekf_rmse_v": float(np.sqrt(np.mean(error**2))),
    }
