#!/usr/bin/env python3
"""
Grand Unified DAE Solver for Smartphone Battery Discharge.

Dual-layer numerical solver:
  - Outer loop: RK4 with adaptive time stepping for [z(t), T(t)]
  - Inner loop: Newton-Raphson to solve implicit current I from
                algebraic constraint g(I) = 0

References:
  - Plett (2015) combined OCV model
  - Arrhenius internal-resistance model (Section 4.3)
  - Bernardi heat-generation equation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────
# 1. Model Parameters
# ─────────────────────────────────────────────────────────────

@dataclass
class BatteryParams:
    """All physical parameters for the Grand Unified Model."""

    # OCV Plett combined model (Table tab:ocv_params)
    K: np.ndarray = field(default_factory=lambda: np.array(
        [3.3930, 1.6143, -2.1838, 1.5588, 0.0, 0.0567, -0.0027]))

    # Internal resistance (fresh pack + Arrhenius)
    R_base: float = 0.030       # Ω  (30 mΩ)
    K_rise: float = 0.015       # Ω  (15 mΩ)
    alpha_R: float = 20.0       # steepness factor
    B_arr: float = 3500.0       # K  (Arrhenius activation energy / R_gas)
    T_ref: float = 298.15       # K

    # Battery cell
    C_nom: float = 3.4          # Ah  (Galaxy S10: 3400 mAh)
    eta_c: float = 0.995        # Coulombic efficiency
    V_cutoff: float = 3.0       # V

    # Thermal
    C_th: float = 30.0          # J/K
    R_th: float = 20.0          # K/W
    dV_dT: float = 1e-4         # V/K  (entropic coefficient, constant approx)
    T_amb: float = 298.15       # K  (ambient temperature)

    # PMIC efficiency
    eta_peak: float = 0.92
    a1: float = 0.05
    a2: float = 0.02
    I_opt: float = 1.0          # A
    I_max: float = 3.0          # A

    # State of Health
    SOH: float = 1.0            # 1.0 = fresh, 0.8 = aged


@dataclass
class SolverParams:
    """Numerical solver configuration."""
    eps_z: float = 1e-3         # max SOC change per step
    eps_T: float = 0.5          # max temperature change per step (K)
    dt_max: float = 60.0        # maximum time step (s)
    dt_min: float = 0.01        # minimum time step (s)
    NR_tol: float = 1e-6        # Newton-Raphson convergence (W)
    NR_max_iter: int = 30       # max NR iterations
    V_nom: float = 3.7          # V  (nominal for initial guess)


# ─────────────────────────────────────────────────────────────
# 2. Core Model Functions
# ─────────────────────────────────────────────────────────────

def V_oc(z: float, T: float, p: BatteryParams) -> float:
    """Plett combined OCV model with temperature correction.

    V_oc(z, T) = K0 + K1*z + K2*z^2 + K3*z^3 + K4/z
                 + K5*ln(z) + K6*ln(1-z) + (T - T_ref)*dV/dT
    """
    z_c = np.clip(z, 1e-6, 1.0 - 1e-6)
    K = p.K
    V_ref = (K[0] + K[1]*z_c + K[2]*z_c**2 + K[3]*z_c**3
             + K[4]/z_c + K[5]*np.log(z_c) + K[6]*np.log(1.0 - z_c))
    return V_ref + (T - p.T_ref) * p.dV_dT


def dVoc_dz(z: float, T: float, p: BatteryParams) -> float:
    """Partial derivative dV_oc/dz (needed for sensitivity analysis)."""
    z_c = np.clip(z, 1e-6, 1.0 - 1e-6)
    K = p.K
    return (K[1] + 2*K[2]*z_c + 3*K[3]*z_c**2
            - K[4]/z_c**2 + K[5]/z_c + K[6]/(1.0 - z_c))


def R_int(z: float, T: float, p: BatteryParams) -> float:
    """Arrhenius-type internal resistance.

    R_int(z, T) = [R_base + K_rise * exp(-alpha*z)] * exp[B*(1/T - 1/T_ref)]
    Scales with 1/SOH for aging.
    """
    z_c = np.clip(z, 1e-6, 1.0 - 1e-6)
    R_soc = p.R_base + p.K_rise * np.exp(-p.alpha_R * z_c)
    f_temp = np.exp(p.B_arr * (1.0/T - 1.0/p.T_ref))
    return R_soc * f_temp / p.SOH


def eta_conv(I: float, p: BatteryParams) -> float:
    """PMIC converter efficiency curve.

    eta(I) = eta_peak - a1*(I/I_opt - 1)^2 - a2*(I/I_max)^3
    """
    return (p.eta_peak
            - p.a1 * (I/p.I_opt - 1.0)**2
            - p.a2 * (I/p.I_max)**3)


def deta_dI(I: float, p: BatteryParams) -> float:
    """Derivative of PMIC efficiency w.r.t. current."""
    return (-2.0 * p.a1 * (I/p.I_opt - 1.0) / p.I_opt
            - 3.0 * p.a2 * (I/p.I_max)**2 / p.I_max)


# ─────────────────────────────────────────────────────────────
# 3. Algebraic Constraint and Newton-Raphson
# ─────────────────────────────────────────────────────────────

def g_constraint(I: float, z: float, T: float,
                 P_comp: float, p: BatteryParams) -> float:
    """Algebraic constraint:  g(I) = I * V_term * eta(I) - P_comp = 0.

    Where V_term = V_oc(z,T) - I * R_int(z,T).
    """
    Voc = V_oc(z, T, p)
    Rint = R_int(z, T, p)
    V_term = Voc - I * Rint
    eta = eta_conv(I, p)
    return I * V_term * eta - P_comp


def dg_dI(I: float, z: float, T: float, p: BatteryParams) -> float:
    """Derivative dg/dI for Newton-Raphson.

    g = I*(Voc - I*Rint)*eta
    dg/dI = (Voc - 2*I*Rint)*eta + I*(Voc - I*Rint)*deta/dI
    """
    Voc = V_oc(z, T, p)
    Rint = R_int(z, T, p)
    eta = eta_conv(I, p)
    de = deta_dI(I, p)
    return (Voc - 2.0*I*Rint) * eta + I * (Voc - I*Rint) * de


def solve_current_NR(z: float, T: float, P_comp: float,
                     p: BatteryParams, sp: SolverParams,
                     I_guess: Optional[float] = None) -> float:
    """Newton-Raphson solver for battery current I.

    Solves g(I) = I * V_term(I) * eta(I) - P_comp = 0.

    Returns:
        I: discharge current (A), positive = discharge.
    """
    if P_comp <= 0:
        return 0.0

    # Initial guess
    if I_guess is None:
        I = P_comp / (sp.V_nom * p.eta_peak)
    else:
        I = I_guess

    I = max(I, 1e-6)

    for _ in range(sp.NR_max_iter):
        gval = g_constraint(I, z, T, P_comp, p)
        dgval = dg_dI(I, z, T, p)

        if abs(dgval) < 1e-15:
            break

        delta = gval / dgval
        I_new = I - delta

        # Prevent negative or excessively large current
        I_new = np.clip(I_new, 1e-6, p.I_max * 2.0)
        I = I_new

        if abs(gval) < sp.NR_tol:
            break

    return I


# ─────────────────────────────────────────────────────────────
# 4. State Derivatives (ODEs)
# ─────────────────────────────────────────────────────────────

def state_derivatives(z: float, T: float, I: float,
                      p: BatteryParams) -> tuple:
    """Compute dz/dt and dT/dt.

    dz/dt = -eta_c * I / (C_nom * SOH)            [SOC depletion]
    dT/dt = (Q_gen - Q_cool) / C_th                [thermal dynamics]

    Q_gen = I^2 * R_int  +  I * T * dV_oc/dT      [Joule + entropic]
    Q_cool = (T - T_amb) / R_th                    [convective cooling]
    """
    C_eff = p.C_nom * p.SOH * 3600.0  # convert Ah to As (Coulombs)
    dz_dt = -p.eta_c * I / C_eff

    Rint = R_int(z, T, p)
    Q_joule = I**2 * Rint
    Q_entropic = I * T * p.dV_dT
    Q_cool = (T - p.T_amb) / p.R_th

    dT_dt = (Q_joule + Q_entropic - Q_cool) / p.C_th

    return dz_dt, dT_dt


# ─────────────────────────────────────────────────────────────
# 5. Adaptive Time Step
# ─────────────────────────────────────────────────────────────

def adaptive_dt(dz_dt: float, dT_dt: float, sp: SolverParams) -> float:
    """Compute adaptive time step from rate constraints."""
    dt = sp.dt_max

    if abs(dz_dt) > 1e-15:
        dt = min(dt, sp.eps_z / abs(dz_dt))

    if abs(dT_dt) > 1e-15:
        dt = min(dt, sp.eps_T / abs(dT_dt))

    return max(dt, sp.dt_min)


# ─────────────────────────────────────────────────────────────
# 6. RK4 Integration (Outer Loop)
# ─────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """Container for simulation results."""
    t: np.ndarray        # time (s)
    z: np.ndarray        # SOC
    T: np.ndarray        # temperature (K)
    I: np.ndarray        # current (A)
    V_term: np.ndarray   # terminal voltage (V)
    P_comp: np.ndarray   # computational power demand (W)
    eta: np.ndarray      # PMIC efficiency


def run_simulation(P_comp_func: Callable[[float], float],
                   z0: float, T0: float,
                   t_end: float,
                   p: BatteryParams = None,
                   sp: SolverParams = None) -> SimResult:
    """Run the full DAE simulation.

    Args:
        P_comp_func: function P_comp(t) returning power demand in Watts
        z0: initial SOC (0 to 1)
        T0: initial temperature (K)
        t_end: simulation end time (s)
        p: battery parameters
        sp: solver parameters

    Returns:
        SimResult with time histories of all states.
    """
    if p is None:
        p = BatteryParams()
    if sp is None:
        sp = SolverParams()

    # Pre-allocate (will trim later)
    max_steps = int(t_end / sp.dt_min) + 10000
    max_steps = min(max_steps, 5_000_000)  # safety cap

    t_arr = np.zeros(max_steps)
    z_arr = np.zeros(max_steps)
    T_arr = np.zeros(max_steps)
    I_arr = np.zeros(max_steps)
    V_arr = np.zeros(max_steps)
    P_arr = np.zeros(max_steps)
    eta_arr = np.zeros(max_steps)

    # Initial conditions
    t_arr[0] = 0.0
    z_arr[0] = z0
    T_arr[0] = T0

    P0 = P_comp_func(0.0)
    I0 = solve_current_NR(z0, T0, P0, p, sp)
    I_arr[0] = I0
    V_arr[0] = V_oc(z0, T0, p) - I0 * R_int(z0, T0, p)
    P_arr[0] = P0
    eta_arr[0] = eta_conv(I0, p)

    n = 0
    I_prev = I0

    while n < max_steps - 1:
        t_n = t_arr[n]
        z_n = z_arr[n]
        T_n = T_arr[n]

        if t_n >= t_end:
            break

        # Current power demand
        P_n = P_comp_func(t_n)

        # Solve current at current state
        I_n = solve_current_NR(z_n, T_n, P_n, p, sp, I_guess=I_prev)

        # Check terminal voltage
        V_term_n = V_oc(z_n, T_n, p) - I_n * R_int(z_n, T_n, p)
        if V_term_n < p.V_cutoff:
            I_arr[n] = I_n
            V_arr[n] = V_term_n
            P_arr[n] = P_n
            eta_arr[n] = eta_conv(I_n, p)
            n += 1
            break

        # Compute derivatives for adaptive step
        dz_n, dT_n = state_derivatives(z_n, T_n, I_n, p)
        dt = adaptive_dt(dz_n, dT_n, sp)
        dt = min(dt, t_end - t_n)

        # RK4 integration
        def rk4_rhs(z_s, T_s, t_s):
            P_s = P_comp_func(t_s)
            I_s = solve_current_NR(z_s, T_s, P_s, p, sp, I_guess=I_prev)
            dz_s, dT_s = state_derivatives(z_s, T_s, I_s, p)
            return dz_s, dT_s, I_s

        dz1, dT1, I_k1 = rk4_rhs(z_n, T_n, t_n)
        dz2, dT2, _ = rk4_rhs(z_n + 0.5*dt*dz1, T_n + 0.5*dt*dT1,
                               t_n + 0.5*dt)
        dz3, dT3, _ = rk4_rhs(z_n + 0.5*dt*dz2, T_n + 0.5*dt*dT2,
                               t_n + 0.5*dt)
        dz4, dT4, _ = rk4_rhs(z_n + dt*dz3, T_n + dt*dT3,
                               t_n + dt)

        z_new = z_n + (dt / 6.0) * (dz1 + 2*dz2 + 2*dz3 + dz4)
        T_new = T_n + (dt / 6.0) * (dT1 + 2*dT2 + 2*dT3 + dT4)

        # Clamp SOC
        z_new = np.clip(z_new, 0.0, 1.0)

        # Store
        I_prev = I_k1
        n += 1
        t_arr[n] = t_n + dt
        z_arr[n] = z_new
        T_arr[n] = T_new

        I_new = solve_current_NR(z_new, T_new, P_comp_func(t_n + dt),
                                 p, sp, I_guess=I_prev)
        I_arr[n] = I_new
        V_arr[n] = V_oc(z_new, T_new, p) - I_new * R_int(z_new, T_new, p)
        P_arr[n] = P_comp_func(t_n + dt)
        eta_arr[n] = eta_conv(I_new, p)
        I_prev = I_new

        # Stop if SOC depleted
        if z_new <= 1e-6:
            n += 1
            break

    # Trim arrays
    n = max(n, 1)
    return SimResult(
        t=t_arr[:n],
        z=z_arr[:n],
        T=T_arr[:n],
        I=I_arr[:n],
        V_term=V_arr[:n],
        P_comp=P_arr[:n],
        eta=eta_arr[:n],
    )


# ─────────────────────────────────────────────────────────────
# 7. Convenience: Run with P_comp time series
# ─────────────────────────────────────────────────────────────

def make_P_comp_interpolator(t_data: np.ndarray,
                             P_data: np.ndarray) -> Callable[[float], float]:
    """Create a linear interpolator for P_comp(t) from discrete data.

    Args:
        t_data: time points in seconds
        P_data: power values in Watts

    Returns:
        Callable that returns interpolated power at any time t.
    """
    def P_func(t: float) -> float:
        if t <= t_data[0]:
            return float(P_data[0])
        if t >= t_data[-1]:
            return float(P_data[-1])
        idx = np.searchsorted(t_data, t, side='right') - 1
        idx = min(idx, len(t_data) - 2)
        frac = (t - t_data[idx]) / (t_data[idx+1] - t_data[idx])
        return float(P_data[idx] + frac * (P_data[idx+1] - P_data[idx]))
    return P_func


# ─────────────────────────────────────────────────────────────
# 8. Quick self-test
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Grand Unified DAE Solver - Self Test")
    print("=" * 50)

    p = BatteryParams()
    sp = SolverParams()

    # Constant 1.5 W load
    result = run_simulation(
        P_comp_func=lambda t: 1.5,
        z0=1.0,
        T0=298.15,
        t_end=3600.0 * 6,  # 6 hours
        p=p,
        sp=sp,
    )

    print(f"Simulation completed: {len(result.t)} steps")
    print(f"  Final time:    {result.t[-1]/3600:.2f} h")
    print(f"  Final SOC:     {result.z[-1]:.4f}")
    print(f"  Final T:       {result.T[-1] - 273.15:.2f} °C")
    print(f"  Final V_term:  {result.V_term[-1]:.3f} V")
    print(f"  Final I:       {result.I[-1]:.3f} A")
    print(f"  Min V_term:    {result.V_term.min():.3f} V")
    print(f"  Max T:         {(result.T.max() - 273.15):.2f} °C")
    print(f"  SOC monotonic: {np.all(np.diff(result.z[:len(result.t)]) <= 1e-12)}")
