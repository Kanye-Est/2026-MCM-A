#!/usr/bin/env python3
"""
Validate the Grand Unified DAE solver against the 3-days battery
consumption dataset.

Note: The 3-days dataset has SOC values that appear synthetic (1.8% drop
over 24h at ~1.1W implies ~400 Ah capacity). We use the power profiles
as realistic inputs and demonstrate model behavior with physical parameters.

Outputs:
  - Model-predicted discharge trajectories
  - RMSE metrics (relative to model-consistent baseline)
  - Data files for figure generation
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import csv
from grand_unified_solver import (
    BatteryParams, SolverParams, run_simulation, make_P_comp_interpolator
)


# ─────────────────────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset',
                        '3-days-battery-consumption-data')

def load_day(filename):
    """Load a day CSV file.  Returns dict with arrays."""
    path = os.path.join(DATA_DIR, filename)
    t_min, brightness, app, network, power, soc = [], [], [], [], [], []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                t_min.append(float(row[0].strip()))
                brightness.append(float(row[1].strip()))
                app.append(float(row[2].strip()))
                network.append(float(row[3].strip()))
                power.append(float(row[4].strip()))
                soc.append(float(row[5].strip()))
            except (ValueError, IndexError):
                continue
    return {
        't_min': np.array(t_min),
        't_s': np.array(t_min) * 60.0,
        'brightness': np.array(brightness),
        'app': np.array(app),
        'network': np.array(network),
        'power': np.array(power),
        'soc': np.array(soc),
    }


def classify_scenario(power):
    """Classify each sample into Light/Medium/Heavy."""
    labels = []
    for p in power:
        if p < 1.0:
            labels.append('Light')
        elif p < 1.5:
            labels.append('Medium')
        else:
            labels.append('Heavy')
    return labels


# ─────────────────────────────────────────────────────────────
# 2. Compute "Ideal" SOC Trajectory for Comparison
# ─────────────────────────────────────────────────────────────

def compute_simple_soc_trajectory(t_data, P_data, z0, C_nom, V_nom=3.7, eta=0.92):
    """Simple Coulomb-counting SOC trajectory (linear model baseline).

    dz/dt = -P / (V * eta * C * 3600)

    This is the "linear model" that ignores:
      - OCV curve effects
      - Temperature dependence
      - Internal resistance voltage drop
    """
    z = np.zeros_like(t_data)
    z[0] = z0
    for i in range(1, len(t_data)):
        dt = t_data[i] - t_data[i-1]
        P_avg = 0.5 * (P_data[i] + P_data[i-1])
        I_avg = P_avg / (V_nom * eta)
        dz = -I_avg * dt / (C_nom * 3600.0)
        z[i] = z[i-1] + dz
    return np.clip(z, 0, 1)


# ─────────────────────────────────────────────────────────────
# 3. Run Validation - Model vs Linear Baseline
# ─────────────────────────────────────────────────────────────

def validate_day(day_data, label, p=None, sp=None):
    """Run solver on one day's data and compare with linear baseline."""
    if p is None:
        p = BatteryParams()
    if sp is None:
        sp = SolverParams()

    t_data = day_data['t_s']
    P_data = day_data['power']

    # Shift time to start at 0
    t_data_shifted = t_data - t_data[0]

    # Create interpolator for P_comp
    P_func = make_P_comp_interpolator(t_data_shifted, P_data)

    # Run DAE model simulation
    z0 = 1.0  # Start fully charged
    T0 = p.T_amb
    t_end = t_data_shifted[-1]

    result = run_simulation(P_func, z0, T0, t_end, p, sp)

    # Interpolate predicted values at data time points
    z_dae = np.interp(t_data_shifted, result.t, result.z)
    T_pred = np.interp(t_data_shifted, result.t, result.T)
    I_pred = np.interp(t_data_shifted, result.t, result.I)
    V_pred = np.interp(t_data_shifted, result.t, result.V_term)

    # Compute linear baseline for comparison
    z_linear = compute_simple_soc_trajectory(
        t_data_shifted, P_data, z0, p.C_nom * p.SOH)

    # Compute deviation: DAE model vs Linear baseline
    # This shows the "nonlinear effects" captured by the full model
    deviation = (z_dae - z_linear) * 100.0  # percentage points

    # Metrics of model vs linear baseline
    rmse = np.sqrt(np.mean(deviation**2))
    mae = np.mean(np.abs(deviation))
    max_dev = np.max(np.abs(deviation))

    # Classify by scenario
    scenarios = classify_scenario(P_data)
    metrics = {}
    for scen in ['Light', 'Medium', 'Heavy']:
        mask = np.array([s == scen for s in scenarios])
        if mask.sum() > 0:
            scen_dev = deviation[mask]
            metrics[scen] = {
                'n': int(mask.sum()),
                'rmse': float(np.sqrt(np.mean(scen_dev**2))),
                'mae': float(np.mean(np.abs(scen_dev))),
                'max_dev': float(np.max(np.abs(scen_dev))),
            }

    metrics['Overall'] = {
        'n': len(deviation),
        'rmse': float(rmse),
        'mae': float(mae),
        'max_dev': float(max_dev),
    }

    return {
        'label': label,
        't_data': t_data_shifted,
        'P_data': P_data,
        'z_dae': z_dae,
        'z_linear': z_linear,
        'T_pred': T_pred,
        'I_pred': I_pred,
        'V_pred': V_pred,
        'deviation': deviation,
        'metrics': metrics,
        'full_result': result,
    }


# ─────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("3-Day Battery Consumption Dataset Validation")
    print("DAE Model vs Linear Coulomb-Counting Baseline")
    print("=" * 60)

    # Load all days
    day1 = load_day('day1_battery_usage_data.csv')
    day2 = load_day('day2_battery_usage_data.csv')
    day3 = load_day('day3_battery_usage_data.csv')

    print(f"\nDay 1: {len(day1['t_min'])} samples, "
          f"Power: [{day1['power'].min():.2f}, {day1['power'].max():.2f}] W, "
          f"Duration: {(day1['t_min'].max()-day1['t_min'].min())/60:.1f} h")
    print(f"Day 2: {len(day2['t_min'])} samples, "
          f"Power: [{day2['power'].min():.2f}, {day2['power'].max():.2f}] W, "
          f"Duration: {(day2['t_min'].max()-day2['t_min'].min())/60:.1f} h")
    print(f"Day 3: {len(day3['t_min'])} samples, "
          f"Power: [{day3['power'].min():.2f}, {day3['power'].max():.2f}] W, "
          f"Duration: {(day3['t_min'].max()-day3['t_min'].min())/60:.1f} h")

    # Use default parameters (3.4 Ah Galaxy S10)
    p = BatteryParams()
    sp = SolverParams()

    print(f"\nModel parameters:")
    print(f"  C_nom = {p.C_nom} Ah, SOH = {p.SOH}")
    print(f"  V_cutoff = {p.V_cutoff} V")
    print(f"  R_base = {p.R_base*1000:.1f} mOhm, K_rise = {p.K_rise*1000:.1f} mOhm")

    # Run validation
    results = []
    for day_data, label in [(day1, 'Day 1'), (day2, 'Day 2'), (day3, 'Day 3')]:
        print(f"\nSimulating {label}...")
        r = validate_day(day_data, label, p, sp)
        results.append(r)

        print(f"  DAE Model Results:")
        print(f"    Final SOC: {r['z_dae'][-1]:.4f}")
        print(f"    Final T:   {r['T_pred'][-1] - 273.15:.2f} °C")
        print(f"    Final V:   {r['V_pred'][-1]:.3f} V")
        print(f"  Linear Baseline SOC: {r['z_linear'][-1]:.4f}")
        print(f"  DAE vs Linear Deviation: RMSE={r['metrics']['Overall']['rmse']:.3f}%, "
              f"Max={r['metrics']['Overall']['max_dev']:.3f}%")

    # Summary: DAE captures additional effects
    print("\n" + "=" * 60)
    print("MODEL COMPARISON: DAE vs Linear Baseline")
    print("=" * 60)
    print("\nThe DAE model captures nonlinear effects that the linear model ignores:")
    print("  - OCV curve (steeper at low SOC)")
    print("  - Temperature-dependent resistance")
    print("  - PMIC efficiency curve")
    print("  - Voltage feedback on current")

    print(f"\n{'Day':10s} {'z_final(DAE)':>14s} {'z_final(Lin)':>14s} {'Delta (%)':>12s}")
    print("-" * 55)
    for r in results:
        delta = (r['z_dae'][-1] - r['z_linear'][-1]) * 100
        print(f"{r['label']:10s} {r['z_dae'][-1]:14.4f} {r['z_linear'][-1]:14.4f} {delta:12.3f}")

    # Avalanche demonstration: run until cutoff
    print("\n" + "=" * 60)
    print("AVALANCHE EFFECT DEMONSTRATION")
    print("=" * 60)
    print("Running constant 1.5W load until cutoff...")

    result_avalanche = run_simulation(
        lambda t: 1.5,
        z0=1.0,
        T0=298.15,
        t_end=3600*24,  # up to 24h
        p=p,
        sp=sp,
    )

    # Find when voltage hits cutoff
    cutoff_idx = np.where(result_avalanche.V_term <= p.V_cutoff + 0.001)[0]
    if len(cutoff_idx) > 0:
        t_cutoff = result_avalanche.t[cutoff_idx[0]] / 3600
        z_cutoff = result_avalanche.z[cutoff_idx[0]]
        print(f"  Cutoff reached at t = {t_cutoff:.2f} h, SOC = {z_cutoff:.4f}")
    else:
        print(f"  Cutoff not reached in simulation window")
        print(f"  Final: t = {result_avalanche.t[-1]/3600:.2f} h, "
              f"SOC = {result_avalanche.z[-1]:.4f}, "
              f"V = {result_avalanche.V_term[-1]:.3f} V")

    # Demonstrate avalanche: plot dz/dt vs z
    # At low SOC, dz/dt should accelerate
    z_vals = result_avalanche.z
    t_vals = result_avalanche.t
    dz_dt = np.gradient(z_vals, t_vals)

    low_soc_mask = z_vals < 0.2
    if low_soc_mask.sum() > 2:
        avg_dz_high = np.mean(dz_dt[z_vals > 0.5])
        avg_dz_low = np.mean(dz_dt[low_soc_mask])
        acceleration = avg_dz_low / avg_dz_high if avg_dz_high != 0 else 1
        print(f"  Avalanche acceleration factor: {acceleration:.2f}x "
              f"(|dz/dt| at z<0.2 vs z>0.5)")
    else:
        print(f"  (Simulation ended before low SOC region)")

    # Save results for figure generation
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        fname = f"validation_{r['label'].replace(' ', '_').lower()}.npz"
        np.savez(os.path.join(output_dir, fname),
                 t_data=r['t_data'],
                 P_data=r['P_data'],
                 z_dae=r['z_dae'],
                 z_linear=r['z_linear'],
                 T_pred=r['T_pred'],
                 I_pred=r['I_pred'],
                 V_pred=r['V_pred'],
                 deviation=r['deviation'],
                 t_sim=r['full_result'].t,
                 z_sim=r['full_result'].z,
                 T_sim=r['full_result'].T,
                 I_sim=r['full_result'].I,
                 V_sim=r['full_result'].V_term)

    # Save avalanche demonstration
    np.savez(os.path.join(output_dir, 'avalanche_demo.npz'),
             t=result_avalanche.t,
             z=result_avalanche.z,
             T=result_avalanche.T,
             I=result_avalanche.I,
             V=result_avalanche.V_term,
             P=result_avalanche.P_comp,
             eta=result_avalanche.eta)

    print(f"\nResults saved to {output_dir}/")
    return results


if __name__ == '__main__':
    results = main()
