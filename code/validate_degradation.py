#!/usr/bin/env python3
"""
Validate the Grand Unified DAE solver against the smartphone battery
degradation dataset, demonstrating SOH sensitivity.

Runs the model at SOH = 1.0, 0.9, 0.8 and cross-validates with
degradation data (battery_health_percent vs charge_cycles, age, temp).

Outputs:
  - TTE comparison across SOH levels
  - R_int scaling with aging
  - Data files for figure generation
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import csv
from grand_unified_solver import (
    BatteryParams, SolverParams, run_simulation,
    V_oc, R_int
)


# ─────────────────────────────────────────────────────────────
# 1. Load Degradation Dataset
# ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset',
                        'smartphone_battery_degradation')

def load_degradation_data():
    """Load the degradation CSV."""
    path = os.path.join(DATA_DIR, 'smartphone_battery_degradation_data.csv')
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'age_months': float(row['age_months']),
                'charge_cycles': float(row['charge_cycles']),
                'screen_time_hrs_day': float(row['screen_time_hrs_day']),
                'fast_charge_percent': float(row['fast_charge_percent']),
                'avg_temp_celsius': float(row['avg_temp_celsius']),
                'full_discharge_count': float(row['full_discharge_count']),
                'battery_health_percent': float(row['battery_health_percent']),
            })
    return rows


def fit_soh_model(data):
    """Fit a simple SOH degradation model.

    SOH = 100 - a1*cycles - a2*age_months - a3*max(T-35, 0)

    Uses least-squares fit.
    """
    n = len(data)
    A = np.zeros((n, 4))
    b = np.zeros(n)

    for i, d in enumerate(data):
        A[i, 0] = 1.0
        A[i, 1] = d['charge_cycles']
        A[i, 2] = d['age_months']
        A[i, 3] = max(d['avg_temp_celsius'] - 35.0, 0.0)
        b[i] = d['battery_health_percent']

    # Least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    b_pred = A @ coeffs
    rmse = np.sqrt(np.mean((b - b_pred)**2))

    return {
        'intercept': coeffs[0],
        'cycle_coeff': coeffs[1],
        'age_coeff': coeffs[2],
        'temp_coeff': coeffs[3],
        'rmse': rmse,
        'b_pred': b_pred,
        'b_actual': b,
    }


# ─────────────────────────────────────────────────────────────
# 2. SOH Sensitivity Analysis
# ─────────────────────────────────────────────────────────────

def run_soh_sweep(P_const=1.5, soh_values=None):
    """Run the DAE solver at multiple SOH levels.

    Returns dict of SOH -> SimResult for TTE comparison.
    """
    if soh_values is None:
        soh_values = [1.0, 0.9, 0.8]

    sp = SolverParams()
    results = {}

    for soh in soh_values:
        p = BatteryParams(SOH=soh)
        result = run_simulation(
            P_comp_func=lambda t: P_const,
            z0=1.0,
            T0=298.15,
            t_end=3600 * 24,  # up to 24h
            p=p,
            sp=sp,
        )
        results[soh] = result

    return results


def compute_tte(result, V_cutoff=3.0):
    """Time-to-empty: time when V_term <= V_cutoff."""
    cutoff_idx = np.where(result.V_term <= V_cutoff + 0.001)[0]
    if len(cutoff_idx) > 0:
        return result.t[cutoff_idx[0]]
    return result.t[-1]


def compute_rint_scaling(soh_values=None):
    """Compute R_int at reference conditions across SOH levels."""
    if soh_values is None:
        soh_values = [1.0, 0.9, 0.8, 0.7, 0.6]

    p = BatteryParams()
    z_ref = 0.5
    T_ref = 298.15

    results = {}
    R_fresh = R_int(z_ref, T_ref, BatteryParams(SOH=1.0))

    for soh in soh_values:
        p_aged = BatteryParams(SOH=soh)
        R = R_int(z_ref, T_ref, p_aged)
        results[soh] = {
            'R_int': R,
            'R_ratio': R / R_fresh,
        }

    return results


# ─────────────────────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Battery Degradation Analysis & SOH Sensitivity")
    print("=" * 60)

    # Load and analyze degradation data
    data = load_degradation_data()
    print(f"\nLoaded {len(data)} degradation samples")

    # Statistics
    health = np.array([d['battery_health_percent'] for d in data])
    cycles = np.array([d['charge_cycles'] for d in data])
    ages = np.array([d['age_months'] for d in data])
    temps = np.array([d['avg_temp_celsius'] for d in data])

    print(f"\nDataset Statistics:")
    print(f"  Battery Health: {health.min():.1f}% - {health.max():.1f}% "
          f"(mean={health.mean():.1f}%)")
    print(f"  Charge Cycles:  {cycles.min():.0f} - {cycles.max():.0f} "
          f"(mean={cycles.mean():.0f})")
    print(f"  Age (months):   {ages.min():.0f} - {ages.max():.0f} "
          f"(mean={ages.mean():.1f})")
    print(f"  Avg Temp (°C):  {temps.min():.1f} - {temps.max():.1f} "
          f"(mean={temps.mean():.1f})")

    # Fit SOH model
    print("\n--- SOH Degradation Model ---")
    fit = fit_soh_model(data)
    print(f"  SOH = {fit['intercept']:.2f} "
          f"+ ({fit['cycle_coeff']:.4f}) × cycles "
          f"+ ({fit['age_coeff']:.4f}) × age_months "
          f"+ ({fit['temp_coeff']:.4f}) × max(T-35,0)")
    print(f"  Fit RMSE: {fit['rmse']:.2f}%")

    # SOH sensitivity sweep
    print("\n--- SOH Sensitivity Sweep ---")
    soh_levels = [1.0, 0.95, 0.9, 0.85, 0.8]
    sweep_results = run_soh_sweep(P_const=1.5, soh_values=soh_levels)

    p_ref = BatteryParams(SOH=1.0)
    tte_baseline = compute_tte(sweep_results[1.0], p_ref.V_cutoff)

    print(f"\n{'SOH':>6s} {'C_eff (Ah)':>12s} {'R_int ratio':>12s} "
          f"{'TTE (h)':>10s} {'TTE reduction':>14s}")
    print("-" * 60)

    r_scaling = compute_rint_scaling(soh_levels)

    for soh in soh_levels:
        tte = compute_tte(sweep_results[soh], p_ref.V_cutoff)
        c_eff = 3.4 * soh
        r_ratio = r_scaling[soh]['R_ratio']
        tte_red = (1 - tte / tte_baseline) * 100 if tte_baseline > 0 else 0
        print(f"{soh:6.2f} {c_eff:12.2f} {r_ratio:12.2f}× "
              f"{tte/3600:10.2f} {tte_red:13.1f}%")

    # Multi-load comparison
    print("\n--- Multi-Load SOH Analysis ---")
    loads = [0.8, 1.5, 2.5]
    for P in loads:
        print(f"\n  P_comp = {P} W:")
        sweep = run_soh_sweep(P_const=P, soh_values=[1.0, 0.9, 0.8])
        tte_base = compute_tte(sweep[1.0], p_ref.V_cutoff)
        for soh in [1.0, 0.9, 0.8]:
            tte = compute_tte(sweep[soh], p_ref.V_cutoff)
            red = (1 - tte/tte_base) * 100 if tte_base > 0 else 0
            print(f"    SOH={soh:.1f}: TTE = {tte/3600:.2f} h  "
                  f"(reduction = {red:.1f}%)")

    # Save results for figure generation
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Save sweep results
    for soh in soh_levels:
        r = sweep_results[soh]
        np.savez(os.path.join(output_dir, f'soh_sweep_{int(soh*100)}.npz'),
                 t=r.t, z=r.z, T=r.T, I=r.I, V=r.V_term)

    # Save degradation data
    np.savez(os.path.join(output_dir, 'degradation_data.npz'),
             health=health, cycles=cycles, ages=ages, temps=temps,
             health_pred=fit['b_pred'],
             fit_intercept=fit['intercept'],
             fit_cycle=fit['cycle_coeff'],
             fit_age=fit['age_coeff'],
             fit_temp=fit['temp_coeff'],
             fit_rmse=fit['rmse'])

    print(f"\nResults saved to {output_dir}/")
    return sweep_results, fit


if __name__ == '__main__':
    sweep_results, fit = main()
