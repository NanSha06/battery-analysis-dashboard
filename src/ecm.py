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


def adaptive_r0(
    soc: np.ndarray,
    temperature_c: np.ndarray,
    soh: np.ndarray,
    base_r0: float,
    reference_temperature_c: float = 25.0,
) -> np.ndarray:
    """Compute state-dependent R0 = f(SOC, Temperature, SOH).

    Uses nonlinear stress factors calibrated so that the resulting
    impedance magnitude lands in the 0.06–0.25 Ω band typically
    observed in 18650 Li-ion cells (NASA B0005-type).
    """
    soc_arr = np.asarray(soc, dtype=float)
    temp_arr = np.asarray(temperature_c, dtype=float)
    soh_arr = np.asarray(soh, dtype=float)

    # --- SOC stress: U-shaped – extremes increase impedance ---
    soc_deviation = np.clip(np.abs(soc_arr - 0.5) * 2.0, 0.0, 1.0)
    soc_factor = 1.0 + 1.8 * soc_deviation**1.5          # up to 2.8x at SOC 0/1

    # --- Temperature stress: Arrhenius-inspired ---
    delta_t = np.clip((reference_temperature_c - temp_arr), -15.0, 40.0)
    temp_factor = np.exp(0.025 * delta_t)                  # ~1.65x at 5°C, ~0.78x at 35°C

    # --- Aging stress: exponential growth below SOH 0.85 ---
    degradation = np.clip(1.0 - soh_arr, 0.0, 0.5)
    aging_factor = 1.0 + 4.5 * degradation**1.3            # up to ~3.2x at SOH 0.5

    r0_scaled = base_r0 * soc_factor * temp_factor * aging_factor
    return np.clip(r0_scaled, 1e-4, 2.0)


def _finite_series(values: pd.Series | np.ndarray | float, default: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([float(arr)], dtype=float)
    if np.isfinite(arr).any():
        fill_value = float(np.nanmedian(arr[np.isfinite(arr)]))
    else:
        fill_value = default
    return np.where(np.isfinite(arr), arr, fill_value)


def _column_or_default(frame: pd.DataFrame, columns: list[str], default: float) -> np.ndarray:
    for column in columns:
        if column in frame.columns:
            return _finite_series(frame[column], default)
    return np.full(len(frame), default, dtype=float)


def _base_param(base_params: ECMParameters, name: str) -> float:
    return float(getattr(base_params, name))


def get_adaptive_ecm_state(
    frame: pd.DataFrame,
    base_params: ECMParameters,
    reference_temperature_c: float = 25.0,
    impedance_column: str = "estimated_impedance_smoothed_ohm",
) -> dict[str, np.ndarray]:
    """Authoritative adaptive ECM state provider.

    All simulation, EKF, exported artifacts, and frequency-domain summaries
    should consume these columns or this provider. Static fitted parameters are
    only used here as the baseline that is adapted by SOC, SOH, temperature,
    cycle aging, current stress, and smoothed impedance evidence.
    """
    if frame.empty:
        return {
            "r0": np.asarray([], dtype=float),
            "r1": np.asarray([], dtype=float),
            "r2": np.asarray([], dtype=float),
            "c1": np.asarray([], dtype=float),
            "c2": np.asarray([], dtype=float),
            "warburg_aw": np.asarray([], dtype=float),
            "tau1": np.asarray([], dtype=float),
            "tau2": np.asarray([], dtype=float),
        }

    soc_values = _column_or_default(frame, ["soc", "discharge_soc_mean", "charge_soc_mean"], 0.5)
    temp_values = _column_or_default(frame, ["temperature_c", "temperature_mean_c", "cycle_temperature_mean_c"], reference_temperature_c)
    soh_values = _column_or_default(frame, ["soh", "soh_model", "soh_model_pred"], 1.0)
    current_values = _column_or_default(frame, ["current_a", "current_mean_a", "cycle_current_mean_a", "charge_current_mean_a"], 0.0)
    cycle_values = _column_or_default(frame, ["discharge_number", "cycle_index"], 0.0)

    dynamic = get_dynamic_params(
        soc_values,
        temp_values,
        soh_values,
        base_params,
        reference_temperature_c=reference_temperature_c,
    )

    cycle_span = float(np.nanmax(cycle_values) - np.nanmin(cycle_values)) if len(cycle_values) else 0.0
    if cycle_span > 1e-9:
        cycle_aging = (cycle_values - np.nanmin(cycle_values)) / cycle_span
    else:
        cycle_aging = np.zeros(len(frame), dtype=float)
    current_stress = np.clip(np.abs(current_values) / 2.0, 0.0, 2.0)

    r0 = dynamic["r0_dynamic"].to_numpy(dtype=float)
    r1 = dynamic["r1_dynamic"].to_numpy(dtype=float)
    r2 = dynamic["r2_dynamic"].to_numpy(dtype=float)
    c1 = dynamic["c1_dynamic"].to_numpy(dtype=float)
    c2 = dynamic["c2_dynamic"].to_numpy(dtype=float)

    stress_scale = 1.0 + 0.18 * current_stress + 0.12 * cycle_aging
    r0 = r0 * stress_scale
    r1 = r1 * (1.0 + 0.08 * current_stress + 0.10 * cycle_aging)
    r2 = r2 * (1.0 + 0.06 * current_stress + 0.12 * cycle_aging)
    c_scale = np.clip(1.0 - 0.08 * cycle_aging, 0.35, 1.25)
    c1 = c1 * c_scale
    c2 = c2 * c_scale

    imp_col = impedance_column if impedance_column in frame.columns else "estimated_impedance_ohm"
    if imp_col in frame.columns:
        impedance = _finite_series(frame[imp_col], np.nan)
        valid = np.isfinite(impedance) & (impedance > 1e-5)
        total_dynamic = np.clip(r0 + r1 + r2, 1e-6, None)
        target_scale = np.ones(len(frame), dtype=float)
        target_scale[valid] = np.clip(impedance[valid] / total_dynamic[valid], 0.35, 3.0)
        r0 = r0 * (0.75 + 0.25 * target_scale)
        r1 = r1 * (0.70 + 0.30 * target_scale)
        r2 = r2 * (0.70 + 0.30 * target_scale)

    r0 = np.clip(r0, 1e-4, 2.0)
    r1 = np.clip(r1, 1e-5, 1.0)
    r2 = np.clip(r2, 1e-5, 1.0)
    c1 = np.clip(c1, 1.0, 1e6)
    c2 = np.clip(c2, 1.0, 1e6)
    tau1 = r1 * c1
    tau2 = r2 * c2
    warburg_aw = np.clip((r0 + r1 + r2) * 0.18, 0.005, 0.15)

    return {
        "r0": r0,
        "r1": r1,
        "r2": r2,
        "c1": c1,
        "c2": c2,
        "warburg_aw": warburg_aw,
        "tau1": tau1,
        "tau2": tau2,
    }


def attach_adaptive_ecm_columns(frame: pd.DataFrame, base_params: ECMParameters) -> pd.DataFrame:
    enriched = frame.copy()
    state = get_adaptive_ecm_state(enriched, base_params)
    for column, values in state.items():
        enriched[column] = values
    return enriched


def get_dynamic_params(
    soc: pd.Series | np.ndarray,
    temperature_c: pd.Series | np.ndarray,
    soh: pd.Series | np.ndarray,
    base_params: ECMParameters,
    reference_temperature_c: float = 25.0,
) -> pd.DataFrame:
    """Return sample-level dynamic ECM parameters scaled by operating conditions.

    The scaling factors are calibrated to reproduce experimentally observed
    impedance magnitudes (0.09–0.18 Ω range for NASA 18650 cells).
    """
    soc_values = np.asarray(soc, dtype=float)
    temp_values = np.asarray(temperature_c, dtype=float)
    soh_values = np.asarray(soh, dtype=float)

    # --- Stress components (shared across R/C) ---
    soc_stress = np.clip(np.abs(soc_values - 0.5) * 2.0, 0.0, 1.0)
    temp_stress = np.clip((reference_temperature_c - temp_values) / 25.0, -0.5, 1.5)
    aging_stress = np.clip(1.0 - soh_values, 0.0, 0.6)

    # --- Adaptive R0 (primary fix for the scale mismatch) ---
    r0_dyn = adaptive_r0(
        soc_values,
        temp_values,
        soh_values,
        _base_param(base_params, "r0"),
        reference_temperature_c,
    )

    # --- R1/R2 scaling – stronger than before to lift polarization arcs ---
    r1_scale = 1.0 + 1.4 * soc_stress + 0.6 * temp_stress + 3.0 * aging_stress
    r2_scale = 1.0 + 1.0 * soc_stress + 0.8 * temp_stress + 3.5 * aging_stress

    # --- Capacitance scaling (aging reduces C → faster dynamics) ---
    c_scale = np.clip(
        1.0 - 0.55 * aging_stress + 0.10 * (temp_values - reference_temperature_c) / 25.0,
        0.15, 1.8,
    )

    return pd.DataFrame(
        {
            "r0_dynamic": r0_dyn,
            "r1_dynamic": np.clip(_base_param(base_params, "r1") * r1_scale, 1e-5, 1.0),
            "c1_dynamic": np.clip(_base_param(base_params, "c1") * c_scale, 1.0, 1e6),
            "r2_dynamic": np.clip(_base_param(base_params, "r2") * r2_scale, 1e-5, 1.0),
            "c2_dynamic": np.clip(_base_param(base_params, "c2") * c_scale, 1.0, 1e6),
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
    r0_override: np.ndarray | None = None,
    r1_override: np.ndarray | None = None,
    r2_override: np.ndarray | None = None,
    c1_override: np.ndarray | None = None,
    c2_override: np.ndarray | None = None,
) -> pd.DataFrame:
    """Simulate 2-RC ECM terminal voltage.

    Parameters
    ----------
    r0_override : optional per-sample R0 array. When provided, uses
        the persisted adaptive per-sample R0 history.
    """
    current_a = np.asarray(current_a, dtype=float)
    dt_s = np.asarray(dt_s, dtype=float)
    soc = np.asarray(soc, dtype=float)
    ocv = interpolate_ocv(soc, ocv_curve)

    if r0_override is not None:
        r0_arr = np.asarray(r0_override, dtype=float)
    else:
        state_frame = pd.DataFrame({"soc": soc})
        r0_arr = get_adaptive_ecm_state(state_frame, params)["r0"]
    if len(r0_arr) != len(current_a):
        raise ValueError("r0_override length must match current_a length")
    r1_arr = np.asarray(r1_override, dtype=float) if r1_override is not None else np.full_like(current_a, _base_param(params, "r1"), dtype=float)
    r2_arr = np.asarray(r2_override, dtype=float) if r2_override is not None else np.full_like(current_a, _base_param(params, "r2"), dtype=float)
    c1_arr = np.asarray(c1_override, dtype=float) if c1_override is not None else np.full_like(current_a, _base_param(params, "c1"), dtype=float)
    c2_arr = np.asarray(c2_override, dtype=float) if c2_override is not None else np.full_like(current_a, _base_param(params, "c2"), dtype=float)
    if any(len(arr) != len(current_a) for arr in (r1_arr, r2_arr, c1_arr, c2_arr)):
        raise ValueError("ECM parameter override lengths must match current_a length")

    v1 = np.zeros_like(current_a, dtype=float)
    v2 = np.zeros_like(current_a, dtype=float)
    terminal_v = np.zeros_like(current_a, dtype=float)

    for i in range(len(current_a)):
        if i > 0:
            r1_i = float(r1_arr[i])
            r2_i = float(r2_arr[i])
            c1_i = float(c1_arr[i])
            c2_i = float(c2_arr[i])
            a1 = np.exp(-max(dt_s[i], 0.0) / max(r1_i * c1_i, 1e-9))
            a2 = np.exp(-max(dt_s[i], 0.0) / max(r2_i * c2_i, 1e-9))
            v1[i] = a1 * v1[i - 1] + r1_i * (1.0 - a1) * current_a[i]
            v2[i] = a2 * v2[i - 1] + r2_i * (1.0 - a2) * current_a[i]
        terminal_v[i] = ocv[i] - current_a[i] * r0_arr[i] - v1[i] - v2[i]

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
    r0_override: np.ndarray | None = None,
) -> pd.DataFrame:
    if sample_table.empty:
        return sample_table.copy()

    ekf_params = ekf_params or EKFParameters()
    frame = sample_table.copy().reset_index(drop=True)

    if r0_override is not None:
        _r0_full = np.asarray(r0_override, dtype=float)
    elif "r0" in frame.columns:
        _r0_full = _finite_series(frame["r0"], 0.01)
    else:
        _r0_full = get_adaptive_ecm_state(frame, params)["r0"]
    if len(_r0_full) != len(frame):
        raise ValueError("r0_override length must match sample_table length")

    frame["soc_ekf"] = np.nan
    frame["soc_ekf_std"] = np.nan
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
        r0_group = _r0_full[idx]  # per-sample adaptive R0 for this group
        r1_group = _finite_series(group["r1"], _base_param(params, "r1")) if "r1" in group.columns else np.full(len(group), _base_param(params, "r1"))
        r2_group = _finite_series(group["r2"], _base_param(params, "r2")) if "r2" in group.columns else np.full(len(group), _base_param(params, "r2"))
        c1_group = _finite_series(group["c1"], _base_param(params, "c1")) if "c1" in group.columns else np.full(len(group), _base_param(params, "c1"))
        c2_group = _finite_series(group["c2"], _base_param(params, "c2")) if "c2" in group.columns else np.full(len(group), _base_param(params, "c2"))

        x = np.asarray([float(seed_soc[0]), 0.0, 0.0], dtype=float)
        p = np.diag(
            [
                ekf_params.initial_cov_soc,
                ekf_params.initial_cov_v1,
                ekf_params.initial_cov_v2,
            ]
        )

        soc_out = np.zeros(len(group), dtype=float)
        soc_std_out = np.zeros(len(group), dtype=float)
        ocv_out = np.zeros(len(group), dtype=float)
        v1_out = np.zeros(len(group), dtype=float)
        v2_out = np.zeros(len(group), dtype=float)
        v_out = np.zeros(len(group), dtype=float)
        residual_out = np.zeros(len(group), dtype=float)
        
        innovation_history = []
        WARMUP = 50

        for i in range(len(group)):
            dt = max(float(dt_s[i]), 0.0)
            ik = float(current[i])

            r1_i = float(r1_group[i])
            r2_i = float(r2_group[i])
            c1_i = float(c1_group[i])
            c2_i = float(c2_group[i])
            a1 = np.exp(-dt / max(r1_i * c1_i, 1e-9))
            a2 = np.exp(-dt / max(r2_i * c2_i, 1e-9))
            b1 = r1_i * (1.0 - a1)
            b2 = r2_i * (1.0 - a2)

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
            h_val = float(ocv_pred - ik * r0_group[i] - x_pred[1] - x_pred[2])
            h_jacobian = np.asarray([[dh_dsoc, -1.0, -1.0]], dtype=float)

            if np.isfinite(voltage[i]):
                innovation = float(voltage[i]) - h_val
                innovation_history.append(innovation)
                
                # --- Adaptive EKF (Sage-Husa) ---
                if len(innovation_history) > WARMUP:
                    recent = np.array(innovation_history[-WARMUP:])
                    R_adapted = float(np.var(recent) + (h_jacobian @ p_pred @ h_jacobian.T))
                    r[0, 0] = max(R_adapted, 1e-6)
                    
                    # Scale Q proportionally
                    q[0, 0] = max(r[0, 0] * 4e-4, 1e-8)
                    q[1, 1] = max(r[0, 0] * 4e-3, 1e-7)
                    q[2, 2] = max(r[0, 0] * 4e-3, 1e-7)

                s = h_jacobian @ p_pred @ h_jacobian.T + r
                k = p_pred @ h_jacobian.T @ np.linalg.pinv(s)
                x = x_pred + (k * innovation).reshape(-1)
                x[0] = float(np.clip(x[0], 0.0, 1.0))
                p = (np.eye(3) - k @ h_jacobian) @ p_pred
                residual_value = innovation
            else:
                x = x_pred
                p = p_pred
                residual_value = np.nan

            ocv_value = ocv_from_soc(x[0], ocv_curve)
            voltage_value = float(ocv_value - ik * r0_group[i] - x[1] - x[2])

            soc_out[i] = x[0]
            soc_std_out[i] = float(np.sqrt(p[0, 0]))
            ocv_out[i] = ocv_value
            v1_out[i] = x[1]
            v2_out[i] = x[2]
            v_out[i] = voltage_value
            residual_out[i] = residual_value

        frame.loc[idx, "soc_ekf"] = soc_out
        frame.loc[idx, "soc_ekf_std"] = soc_std_out
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
    
    adaptive_state = get_adaptive_ecm_state(sample_table, params)
    r0_arr = adaptive_state["r0"]

    simulated = simulate_2rc_ecm(
        current_a=sample_table["current_a"].fillna(0.0).to_numpy(dtype=float),
        dt_s=sample_table["dt_s"].fillna(0.0).to_numpy(dtype=float),
        soc=sample_table["soc"].ffill().fillna(0.5).to_numpy(dtype=float),
        params=params,
        ocv_curve=ocv_curve,
        r0_override=r0_arr,
        r1_override=adaptive_state["r1"],
        r2_override=adaptive_state["r2"],
        c1_override=adaptive_state["c1"],
        c2_override=adaptive_state["c2"],
    )
    frame = sample_table.copy()
    
    for column, values in adaptive_state.items():
        frame[column] = values
    
    frame = pd.concat([frame.reset_index(drop=True), simulated.reset_index(drop=True)], axis=1)
    frame["voltage_error_v"] = frame["voltage_model_v"] - frame["voltage_v"]
    if run_ekf:
        frame = run_ekf_soc_ocv(
            frame,
            params=params,
            ocv_curve=ocv_curve,
            nominal_capacity_ah=nominal_capacity_ah,
            ekf_params=ekf_params,
            r0_override=r0_arr,
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
        
        dynamic_state = get_adaptive_ecm_state(working_group, params)
        r0_arr = dynamic_state["r0"]

        simulated = simulate_2rc_ecm(
            current_a=working_group["current_a"].fillna(0.0).to_numpy(dtype=float),
            dt_s=working_group["dt_s"].fillna(0.0).to_numpy(dtype=float),
            soc=working_group["soc"].ffill().fillna(0.5).to_numpy(dtype=float),
            params=params,
            ocv_curve=ocv_curve,
            r0_override=r0_arr,
            r1_override=dynamic_state["r1"],
            r2_override=dynamic_state["r2"],
            c1_override=dynamic_state["c1"],
            c2_override=dynamic_state["c2"],
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
                "r0": float(np.nanmedian(dynamic_state["r0"])),
                "r1": float(np.nanmedian(dynamic_state["r1"])),
                "c1": float(np.nanmedian(dynamic_state["c1"])),
                "r2": float(np.nanmedian(dynamic_state["r2"])),
                "c2": float(np.nanmedian(dynamic_state["c2"])),
                "warburg_aw": float(np.nanmedian(dynamic_state["warburg_aw"])),
                "tau1": float(np.nanmedian(dynamic_state["tau1"])),
                "tau2": float(np.nanmedian(dynamic_state["tau2"])),
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


def _finite_median(frame: pd.DataFrame, column: str, default: float | None = None) -> float | None:
    if column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return default
    return float(values.median())


def get_frequency_domain_params(frame: pd.DataFrame) -> dict[str, float]:
    """Return Nyquist/Bode parameters from persisted adaptive artifact state."""
    params = {
        "r0": _finite_median(frame, "r0", 0.01),
        "r1": _finite_median(frame, "r1", 0.01),
        "r2": _finite_median(frame, "r2", 0.02),
        "c1": _finite_median(frame, "c1", 2000.0),
        "c2": _finite_median(frame, "c2", 4000.0),
        "warburg_aw": _finite_median(frame, "warburg_aw", None),
    }
    total_r = float(params["r0"] + params["r1"] + params["r2"])
    if params["warburg_aw"] is None or not np.isfinite(params["warburg_aw"]):
        params["warburg_aw"] = float(np.clip(total_r * 0.18, 0.005, 0.15))
    return {key: float(value) for key, value in params.items()}


def validate_ecm_consistency(
    sample_frame: pd.DataFrame | None = None,
    cycle_frame: pd.DataFrame | None = None,
    dashboard_source: str | None = None,
) -> dict[str, object]:
    """Detect split-pipeline ECM regressions before rendering/export."""
    warnings: list[str] = []

    for name, frame in {"sample": sample_frame, "cycle": cycle_frame}.items():
        if frame is None or frame.empty:
            continue
        missing = [col for col in ["r0", "r1", "r2", "c1", "c2"] if col not in frame.columns]
        if missing:
            warnings.append(f"{name} artifact is missing adaptive ECM columns: {', '.join(missing)}")
        if "r0" in frame.columns:
            r0 = pd.to_numeric(frame["r0"], errors="coerce").dropna()
            if len(r0) > 8:
                mean_r0 = float(r0.mean())
                cv = float(r0.std() / mean_r0) if abs(mean_r0) > 1e-12 else 0.0
                if cv < 0.005:
                    warnings.append(f"{name} adaptive R0 variance is unrealistically low (cv={cv:.4f}).")

    if sample_frame is not None and cycle_frame is not None and not sample_frame.empty and not cycle_frame.empty:
        common = {"battery_id", "cycle_index", "r0"}
        if common.issubset(sample_frame.columns) and common.issubset(cycle_frame.columns):
            sample_r0 = sample_frame.groupby(["battery_id", "cycle_index"], as_index=False)["r0"].median()
            cycle_r0 = cycle_frame[["battery_id", "cycle_index", "r0"]].copy()
            merged = sample_r0.merge(cycle_r0, on=["battery_id", "cycle_index"], suffixes=("_sample", "_cycle"))
            if not merged.empty:
                delta = np.nanmedian(np.abs(merged["r0_sample"] - merged["r0_cycle"]))
                scale = max(float(np.nanmedian(np.abs(merged["r0_sample"]))), 1e-9)
                if float(delta / scale) > 0.25:
                    warnings.append("Sample and cycle adaptive R0 histories diverge by more than 25%.")

    if dashboard_source:
        mutation_tokens = [
            "apply_adaptive_ecm_to_cycles",
            "adaptive_r0(",
            "get_dynamic_params(",
            "r0_aligned",
            "sanitize_ecm_impedance_params",
        ]
        for token in mutation_tokens:
            if token in dashboard_source:
                warnings.append(f"Dashboard source still contains rendering-side ECM mutation token: {token}")
        if "params.r0" in dashboard_source:
            warnings.append("Dashboard source still contains static params.r0 access.")

    return {"ok": not warnings, "warnings": warnings}
