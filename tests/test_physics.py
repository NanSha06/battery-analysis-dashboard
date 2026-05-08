import numpy as np
import pandas as pd
import pytest
from src.rul import fit_arrhenius_rul

def test_sop_calculation_logic():
    # Simulate SOP calculation logic from pipeline.py
    # SOP = ((V_min - OCV(SOC)) / R_total) * V_min
    v_min = 3.0
    ocv_sop = np.array([3.6, 3.8, 4.0])
    r_total = np.array([0.1, 0.2, 0.15])
    
    sop_expected = ((v_min - ocv_sop) / r_total) * v_min
    
    # Check values
    assert len(sop_expected) == 3
    assert sop_expected[0] == ((3.0 - 3.6) / 0.1) * 3.0 # -18.0
    assert np.all(sop_expected < 0) # Because V_min < OCV in discharge

def test_arrhenius_fitting_smoke():
    # Create synthetic degradation data
    cycles = np.arange(1, 100)
    # Q_loss = A * exp(-Ea/(RT)) * sqrt(cycle)
    # Let A=0.01, Ea=20000, T=300, R=8.314
    # Q_loss = 0.01 * exp(-20000/(8.314*300)) * sqrt(cycle)
    # Q_loss = 0.01 * 0.000329 * sqrt(cycle)
    
    T_k = 300.0
    T_c = T_k - 273.15
    y_loss = 0.001 * np.sqrt(cycles)
    soh = 1.0 - y_loss
    temp = np.full_like(cycles, T_c)
    
    rul = fit_arrhenius_rul(cycles, soh, temp, soh_threshold=0.8)
    
    assert len(rul) == len(cycles)
    assert np.all(np.isfinite(rul))
    assert rul[0] > rul[-1] # RUL should decrease
