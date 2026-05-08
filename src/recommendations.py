from __future__ import annotations

import numpy as np

def get_charge_recommendation(
    soh: float,
    rul_cycles: float,
    temperature_mean_c: float,
    plating_risk: float
) -> dict[str, str]:
    """
    Returns a recommendation dict with 'action' and 'reason'.
    """
    if plating_risk > 0.4:
        return {
            "action": "reduce_crate",
            "reason": f"High lithium plating risk ({plating_risk:.2f}) detected."
        }
    
    if soh < 0.70:
        return {
            "action": "replace",
            "reason": f"SOH ({soh:.2f}) is below replacement threshold (0.70)."
        }
    
    if soh < 0.80 and rul_cycles < 20:
        return {
            "action": "inspect",
            "reason": f"Low SOH ({soh:.2f}) and near-EOL RUL ({rul_cycles:.0f} cycles) suggest inspection."
        }
    
    if temperature_mean_c > 45.0:
        return {
            "action": "reduce_voltage",
            "reason": f"High operating temperature ({temperature_mean_c:.1f}°C) detected."
        }
    
    return {
        "action": "normal",
        "reason": "Operating parameters are within safe nominal ranges."
    }
