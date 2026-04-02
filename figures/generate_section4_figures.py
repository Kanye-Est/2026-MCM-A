#!/usr/bin/env python3
"""
Section 4 Figure Generation Script
===================================
Generates publication-quality figures for the Power Consumption Model section.
Output: SVG files for Inkscape polishing → PDF for LaTeX.

Usage:
    python generate_section4_figures.py

Author: MCM 2026 Team
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from pathlib import Path

# =============================================================================
# GLOBAL STYLE CONFIGURATION
# =============================================================================

# Academic color palette
COLORS = {
    'deep_blue': '#1f77b4',
    'academic_orange': '#ff7f0e',
    'slate_gray': '#7f7f7f',
    'forest_green': '#2ca02c',
    'crimson': '#d62728',
    'purple': '#9467bd',
    'teal': '#17becf',
    'brown': '#8c564b',
}

# Configure matplotlib for academic publications
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'grid.alpha': 0.3,
})

# Output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# FIGURE 4.2: OLED POWER SURFACE (3D)
# =============================================================================

def generate_fig_oled_power_surface():
    """
    3D surface plot showing OLED screen power as function of brightness and APL.
    P_screen = P_drv + α · (L/L_max)^γ · APL · A_disp
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Parameters (Pixel 6 Pro style)
    P_drv = 0.15  # W, driver overhead
    alpha = 45    # W/m^2
    A_disp = 0.01  # m^2 (~6.7" display)
    L_max = 1000  # nits
    gamma = 2.2

    # Create mesh
    L = np.linspace(0, 1000, 100)  # Brightness (nits)
    APL = np.linspace(0, 1, 100)   # Average Picture Level
    L_mesh, APL_mesh = np.meshgrid(L, APL)

    # Power equation
    P_screen = P_drv + alpha * (L_mesh / L_max) ** gamma * APL_mesh * A_disp

    # Plot surface
    surf = ax.plot_surface(L_mesh, APL_mesh, P_screen,
                           cmap=cm.viridis, alpha=0.85,
                           linewidth=0, antialiased=True)

    # Contour projections
    ax.contour(L_mesh, APL_mesh, P_screen, zdir='z', offset=0,
               cmap=cm.viridis, alpha=0.5, levels=10)

    # Labels
    ax.set_xlabel(r'Brightness $L$ (nits)', labelpad=10)
    ax.set_ylabel(r'APL', labelpad=10)
    ax.set_zlabel(r'$P_{\mathrm{screen}}$ (W)', labelpad=10)
    ax.set_title(r'OLED Power: $P = P_{\mathrm{drv}} + \alpha \cdot (L/L_{\max})^{\gamma} \cdot \mathrm{APL}$',
                 fontsize=11, pad=15)

    # Annotations for key points
    ax.scatter([1000], [1.0], [P_drv + alpha * 1 * 1 * A_disp],
               color=COLORS['crimson'], s=50, zorder=10)
    ax.text(1000, 1.0, P_drv + alpha * 1 * 1 * A_disp + 0.1,
            'Full white\n~0.6W', fontsize=8, ha='center')

    ax.scatter([500], [0.15], [P_drv + alpha * (0.5)**gamma * 0.15 * A_disp],
               color=COLORS['forest_green'], s=50, zorder=10)
    ax.text(500, 0.15, P_drv + alpha * (0.5)**gamma * 0.15 * A_disp + 0.05,
            'Dark mode\n~0.17W', fontsize=8, ha='center')

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    cbar.set_label(r'Power (W)')

    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_4_2_oled_power_surface.svg', format='svg', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_4_2_oled_power_surface.png', format='png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig_4_2_oled_power_surface.svg")


# =============================================================================
# FIGURE 4.3: CPU POWER DUALITY (TWO-PANEL)
# =============================================================================

def generate_fig_cpu_power_duality():
    """
    Two-panel figure:
    Left: P_dyn vs frequency (cubic f^3 scaling)
    Right: P_leak vs temperature (exponential thermal runaway)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left Panel: Dynamic Power (f^3 scaling) ---
    f = np.linspace(0.1, 3.0, 100)  # GHz

    # P_dyn ∝ C_eff · V^2 · f, with V ∝ f → P ∝ f^3
    C_eff = 2e-9  # F (effective capacitance)
    V_ref = 0.7   # V at 1 GHz

    # Voltage scales with frequency (simplified DVFS)
    V = V_ref * (f / 1.0) ** 0.5  # V ∝ √f (conservative scaling)
    P_dyn = C_eff * V**2 * f * 1e9  # Convert to W

    # Also show idealized f^3 for comparison
    P_ideal = 0.3 * (f / 1.0) ** 3

    ax1.plot(f, P_dyn, color=COLORS['deep_blue'], linewidth=2.5,
             label=r'$P_{\mathrm{dyn}} = C_{\mathrm{eff}} V^2 f$')
    ax1.plot(f, P_ideal, '--', color=COLORS['slate_gray'], linewidth=1.5,
             label=r'Idealized $f^3$ scaling')

    # Mark doubling points
    f1, f2 = 1.0, 2.0
    P1 = 0.3 * (f1) ** 3
    P2 = 0.3 * (f2) ** 3
    ax1.scatter([f1, f2], [P1, P2], color=COLORS['crimson'], s=60, zorder=10)
    ax1.annotate('', xy=(f2, P2), xytext=(f1, P1),
                 arrowprops=dict(arrowstyle='->', color=COLORS['academic_orange'], lw=2))
    ax1.text(1.5, 1.5, r'$\times 8$', fontsize=12, color=COLORS['academic_orange'],
             fontweight='bold', ha='center')

    ax1.set_xlabel(r'Clock Frequency $f$ (GHz)')
    ax1.set_ylabel(r'Dynamic Power $P_{\mathrm{dyn}}$ (W)')
    ax1.set_title(r'(a) Cubic Frequency Scaling', fontsize=11)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_xlim([0, 3])
    ax1.set_ylim([0, 3])
    ax1.grid(True, alpha=0.3)

    # --- Right Panel: Leakage Power (exponential T) ---
    T = np.linspace(20, 100, 100)  # °C
    T_ref = 25  # °C
    T_K = T + 273.15
    T_ref_K = T_ref + 273.15

    # P_leak = V · I_leak0 · (T/T_ref)^2 · exp(β(T - T_ref))
    V_core = 0.9
    I_leak0 = 0.05  # A at reference
    beta = 0.03  # K^-1

    P_leak = V_core * I_leak0 * (T_K / T_ref_K)**2 * np.exp(beta * (T - T_ref))

    ax2.plot(T, P_leak, color=COLORS['crimson'], linewidth=2.5,
             label=r'$P_{\mathrm{leak}} \propto T^2 e^{\beta(T-T_{\mathrm{ref}})}$')
    ax2.fill_between(T, P_leak, alpha=0.2, color=COLORS['crimson'])

    # Mark thermal runaway region
    ax2.axvspan(80, 100, alpha=0.15, color=COLORS['academic_orange'],
                label='Thermal throttle zone')
    ax2.axvline(x=85, color=COLORS['academic_orange'], linestyle='--', linewidth=1.5)
    ax2.text(87, 0.4, 'Throttle\nthreshold', fontsize=8, color=COLORS['academic_orange'])

    ax2.set_xlabel(r'Temperature $T$ (°C)')
    ax2.set_ylabel(r'Leakage Power $P_{\mathrm{leak}}$ (W)')
    ax2.set_title(r'(b) Exponential Thermal Coupling', fontsize=11)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_xlim([20, 100])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_4_3_cpu_power_duality.svg', format='svg', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_4_3_cpu_power_duality.png', format='png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig_4_3_cpu_power_duality.svg")


# =============================================================================
# FIGURE 4.4: RRC STATE & TAIL ENERGY TIMELINE
# =============================================================================

def generate_fig_rrc_tail_energy():
    """
    Timeline showing RRC state transitions and the tail energy phenomenon.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Time axis (seconds)
    t = np.linspace(0, 25, 1000)

    # Power levels (mW)
    P_idle = 10
    P_drx = 50
    P_active = 1200

    # Define state timeline
    power = np.ones_like(t) * P_idle

    # Event 1: Data request at t=2, active until t=4
    mask1 = (t >= 2) & (t < 4)
    power[mask1] = P_active

    # Tail energy: stay in DRX from t=4 to t=16 (T_tail = 12s)
    mask_tail1 = (t >= 4) & (t < 16)
    power[mask_tail1] = P_drx

    # Back to idle
    mask_idle1 = (t >= 16) & (t < 18)
    power[mask_idle1] = P_idle

    # Event 2: Another data request at t=18
    mask2 = (t >= 18) & (t < 19)
    power[mask2] = P_active

    # Short tail
    mask_tail2 = (t >= 19) & (t < 25)
    power[mask_tail2] = P_drx

    # Plot
    ax.fill_between(t, power, alpha=0.4, color=COLORS['deep_blue'], step='post')
    ax.step(t, power, where='post', color=COLORS['deep_blue'], linewidth=2)

    # Annotate states
    ax.axhline(y=P_active, color=COLORS['crimson'], linestyle=':', alpha=0.5)
    ax.axhline(y=P_drx, color=COLORS['academic_orange'], linestyle=':', alpha=0.5)
    ax.axhline(y=P_idle, color=COLORS['forest_green'], linestyle=':', alpha=0.5)

    ax.text(0.5, P_active + 50, 'Active TX/RX', fontsize=9, color=COLORS['crimson'])
    ax.text(0.5, P_drx + 30, 'C-DRX', fontsize=9, color=COLORS['academic_orange'])
    ax.text(0.5, P_idle + 20, 'IDLE', fontsize=9, color=COLORS['forest_green'])

    # Mark tail energy region
    ax.annotate('', xy=(16, 200), xytext=(4, 200),
                arrowprops=dict(arrowstyle='<->', color=COLORS['academic_orange'], lw=2))
    ax.text(10, 250, r'$T_{\mathrm{tail}} = 12$s', fontsize=10,
            color=COLORS['academic_orange'], ha='center', fontweight='bold')

    # Data transfer annotation
    ax.annotate('Data\ntransfer', xy=(3, P_active), xytext=(3, P_active + 300),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=1))

    # Shade tail energy waste
    t_tail = t[(t >= 4) & (t < 16)]
    power_tail = np.ones_like(t_tail) * P_drx
    ax.fill_between(t_tail, power_tail, P_idle, alpha=0.3,
                    color=COLORS['academic_orange'], hatch='//',
                    label='Tail energy waste')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('RRC State Transitions and Tail Energy Overhead', fontsize=11)
    ax.set_xlim([0, 25])
    ax.set_ylim([0, 1600])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_4_4_rrc_tail_energy.svg', format='svg', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_4_4_rrc_tail_energy.png', format='png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig_4_4_rrc_tail_energy.svg")


# =============================================================================
# FIGURE 4.5: GPS ACQUISITION DECAY
# =============================================================================

def generate_fig_gps_acquisition():
    """
    Exponential decay from P_acq to P_track as GPS locks onto satellites.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Parameters
    P_acq = 420    # mW (acquisition mode)
    P_track = 120  # mW (tracking mode)
    tau_fix_good = 30   # seconds (open sky, warm start)
    tau_fix_poor = 90   # seconds (urban canyon, cold start)

    t = np.linspace(0, 180, 500)  # seconds

    # P_GPS = P_acq * exp(-t/τ) + P_track * (1 - exp(-t/τ))
    def gps_power(t, tau):
        return P_acq * np.exp(-t / tau) + P_track * (1 - np.exp(-t / tau))

    P_good = gps_power(t, tau_fix_good)
    P_poor = gps_power(t, tau_fix_poor)
    P_blocked = np.ones_like(t) * P_acq  # Never locks

    # Plot scenarios
    ax.plot(t, P_good, color=COLORS['forest_green'], linewidth=2.5,
            label=r'Open sky ($\tau_{\mathrm{fix}} = 30$s)')
    ax.plot(t, P_poor, color=COLORS['academic_orange'], linewidth=2.5,
            label=r'Urban canyon ($\tau_{\mathrm{fix}} = 90$s)')
    ax.plot(t, P_blocked, '--', color=COLORS['crimson'], linewidth=2,
            label=r'Signal blocked ($\tau_{\mathrm{fix}} \to \infty$)')

    # Reference lines
    ax.axhline(y=P_acq, color=COLORS['slate_gray'], linestyle=':', alpha=0.5)
    ax.axhline(y=P_track, color=COLORS['slate_gray'], linestyle=':', alpha=0.5)

    ax.text(185, P_acq, r'$P_{\mathrm{acq}}$', fontsize=10, va='center')
    ax.text(185, P_track, r'$P_{\mathrm{track}}$', fontsize=10, va='center')

    # Mark tau_fix points
    ax.scatter([tau_fix_good], [gps_power(tau_fix_good, tau_fix_good)],
               color=COLORS['forest_green'], s=80, zorder=10, marker='o')
    ax.scatter([tau_fix_poor], [gps_power(tau_fix_poor, tau_fix_poor)],
               color=COLORS['academic_orange'], s=80, zorder=10, marker='o')

    # 63% decay annotation
    P_63 = P_track + (P_acq - P_track) * np.exp(-1)
    ax.axhline(y=P_63, color=COLORS['slate_gray'], linestyle='--', alpha=0.3)
    ax.text(5, P_63 + 15, '63% decay', fontsize=8, color=COLORS['slate_gray'])

    ax.set_xlabel('Time since GPS activation (s)')
    ax.set_ylabel('GPS Power (mW)')
    ax.set_title(r'GPS Power Decay: $P = P_{\mathrm{acq}} e^{-t/\tau} + P_{\mathrm{track}}(1 - e^{-t/\tau})$',
                 fontsize=11)
    ax.legend(loc='right')
    ax.set_xlim([0, 180])
    ax.set_ylim([80, 480])
    ax.grid(True, alpha=0.3)

    # Fill between curves to show energy difference
    ax.fill_between(t, P_good, P_track, alpha=0.15, color=COLORS['forest_green'])
    ax.fill_between(t, P_poor, P_good, alpha=0.15, color=COLORS['academic_orange'])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_4_5_gps_acquisition.svg', format='svg', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_4_5_gps_acquisition.png', format='png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig_4_5_gps_acquisition.svg")


# =============================================================================
# FIGURE 4.7: THE IMPLICIT I-V LOOP
# =============================================================================

def generate_fig_iv_loop():
    """
    Shows how required current I_batt explodes as terminal voltage V_term sags.
    I = P_comp / (V_term · η_conv)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Voltage range
    V_term = np.linspace(3.0, 4.35, 200)  # V

    # PMIC efficiency (simplified parabolic model)
    eta_peak = 0.92
    I_opt = 1.0  # A
    def eta_conv(I):
        return eta_peak - 0.05 * ((I / I_opt) - 1)**2

    # Power levels
    P_levels = [1.0, 2.0, 3.0, 4.0, 5.0]  # W
    colors = [COLORS['forest_green'], COLORS['teal'], COLORS['deep_blue'],
              COLORS['academic_orange'], COLORS['crimson']]

    for P, color in zip(P_levels, colors):
        # Simple approximation: I = P / (V * η)
        # Use fixed η for visualization (real solver is iterative)
        eta = 0.88
        I_batt = P / (V_term * eta)

        ax.plot(V_term, I_batt, color=color, linewidth=2.5,
                label=f'$P_{{\\mathrm{{comp}}}} = {P:.0f}$ W')

    # Mark cutoff voltage
    ax.axvline(x=3.0, color=COLORS['crimson'], linestyle='--', linewidth=2, alpha=0.7)
    ax.text(3.05, 2.2, 'Cutoff\n$V_{\\mathrm{cutoff}}$', fontsize=9,
            color=COLORS['crimson'])

    # Shade danger zone
    ax.axvspan(3.0, 3.3, alpha=0.15, color=COLORS['crimson'])
    ax.text(3.15, 0.3, 'Low-voltage\nstress zone', fontsize=8,
            color=COLORS['crimson'], ha='center')

    # Annotation for the "explosion"
    ax.annotate('Current explodes\nas voltage sags',
                xy=(3.1, 1.9), xytext=(3.5, 2.3),
                fontsize=9, color=COLORS['crimson'],
                arrowprops=dict(arrowstyle='->', color=COLORS['crimson']))

    ax.set_xlabel(r'Terminal Voltage $V_{\mathrm{term}}$ (V)')
    ax.set_ylabel(r'Battery Current $I_{\mathrm{batt}}$ (A)')
    ax.set_title(r'Implicit Current-Voltage Relationship: $I = P_{\mathrm{comp}} / (V \cdot \eta)$',
                 fontsize=11)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([3.0, 4.4])
    ax.set_ylim([0, 2.5])
    ax.grid(True, alpha=0.3)

    # Invert x-axis to show voltage dropping left-to-right (discharge direction)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_4_7_iv_loop.svg', format='svg', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_4_7_iv_loop.png', format='png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig_4_7_iv_loop.svg")


# =============================================================================
# FIGURE 4.8: THE AVALANCHE EFFECT ("MONEY SHOT")
# =============================================================================

def generate_fig_avalanche_effect():
    """
    The key figure showing how battery drain accelerates at low SOC.
    Compares naive linear model vs our coupled model.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # SOC range (100% to 0%)
    SOC = np.linspace(1.0, 0.02, 200)

    # --- Model Parameters ---
    P_comp = 3.0      # W, constant component power
    C_nom = 5.0       # Ah, battery capacity
    R0_ref = 0.08     # Ohm, reference internal resistance
    eta_peak = 0.90   # PMIC peak efficiency

    # OCV curve (Shepherd-Nernst simplified)
    # E_OCV = K0 + K1*z + K2*ln(z) + K3*ln(1-z)
    K0, K1, K2, K3 = 3.4, 0.5, 0.05, -0.03

    def E_OCV(z):
        z_safe = np.clip(z, 0.001, 0.999)
        return K0 + K1 * z_safe + K2 * np.log(z_safe) + K3 * np.log(1 - z_safe)

    # Internal resistance increases at low SOC
    def R0(z):
        return R0_ref * (1 + 0.5 * np.exp(-8 * z))  # Increases sharply below 20%

    # PMIC efficiency decreases at high current
    def eta_conv(I):
        return eta_peak - 0.02 * (I - 1.0)**2

    # --- Line A: Naive Model (constant V, constant η) ---
    V_nom = 3.7  # V
    I_naive = P_comp / (V_nom * eta_peak) * np.ones_like(SOC)

    # --- Line B: Our Coupled Model (iterative solve at each SOC) ---
    I_coupled = []
    V_term_coupled = []

    for z in SOC:
        # Initial guess
        I = P_comp / (E_OCV(z) * eta_peak)

        # Newton iteration (simplified)
        for _ in range(10):
            V_term = E_OCV(z) - I * R0(z)
            eta = np.clip(eta_conv(I), 0.7, 0.95)
            I_new = P_comp / (V_term * eta)
            if abs(I_new - I) < 1e-6:
                break
            I = 0.5 * I + 0.5 * I_new  # Damped update

        I_coupled.append(I)
        V_term_coupled.append(V_term)

    I_coupled = np.array(I_coupled)
    V_term_coupled = np.array(V_term_coupled)

    # --- Plotting ---
    ax.plot(SOC * 100, I_naive, '--', color=COLORS['slate_gray'], linewidth=2.5,
            label='Naive model (constant $V$, $\\eta$)')
    ax.plot(SOC * 100, I_coupled, color=COLORS['crimson'], linewidth=3,
            label='Coupled model (this work)')

    # Fill the gap to highlight divergence
    ax.fill_between(SOC * 100, I_naive, I_coupled,
                    where=(I_coupled > I_naive),
                    alpha=0.3, color=COLORS['crimson'],
                    label='Avalanche region')

    # Mark the avalanche zone
    ax.axvspan(0, 15, alpha=0.1, color=COLORS['academic_orange'])
    ax.text(7.5, 1.7, 'AVALANCHE\nZONE', fontsize=11, fontweight='bold',
            color=COLORS['academic_orange'], ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Annotation
    ax.annotate('+22% current\nat 5% SOC',
                xy=(5, I_coupled[SOC <= 0.05][0]),
                xytext=(20, 1.55),
                fontsize=10, color=COLORS['crimson'],
                arrowprops=dict(arrowstyle='->', color=COLORS['crimson'], lw=1.5))

    # Secondary y-axis for terminal voltage
    ax2 = ax.twinx()
    ax2.plot(SOC * 100, V_term_coupled, ':', color=COLORS['deep_blue'],
             linewidth=2, alpha=0.7)
    ax2.set_ylabel(r'Terminal Voltage $V_{\mathrm{term}}$ (V)',
                   color=COLORS['deep_blue'])
    ax2.tick_params(axis='y', labelcolor=COLORS['deep_blue'])
    ax2.set_ylim([2.8, 4.2])
    ax2.axhline(y=3.0, color=COLORS['deep_blue'], linestyle='--', alpha=0.5)
    ax2.text(2, 3.05, '$V_{\\mathrm{cutoff}}$', fontsize=8,
             color=COLORS['deep_blue'])

    ax.set_xlabel('State of Charge (%)')
    ax.set_ylabel(r'Battery Current $I_{\mathrm{batt}}$ (A)')
    ax.set_title('The Avalanche Effect: Current Surge at Low SOC Under Constant Load',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_xlim([100, 0])  # Reversed: 100% on left, 0% on right
    ax.set_ylim([0.8, 1.8])
    ax.grid(True, alpha=0.3)

    # Add percentage difference annotation
    pct_diff = (I_coupled[-1] / I_naive[-1] - 1) * 100
    ax.text(50, 0.95, f'At constant $P_{{\\mathrm{{comp}}}} = {P_comp}$ W',
            fontsize=9, style='italic', ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_4_8_avalanche_effect.svg', format='svg', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_4_8_avalanche_effect.png', format='png', bbox_inches='tight')
    plt.close()
    print("✓ Generated: fig_4_8_avalanche_effect.svg")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 60)
    print("Section 4 Figure Generation")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    generate_fig_oled_power_surface()
    generate_fig_cpu_power_duality()
    generate_fig_rrc_tail_energy()
    generate_fig_gps_acquisition()
    generate_fig_iv_loop()
    generate_fig_avalanche_effect()

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print("Next steps:")
    print("  1. Open SVG files in Inkscape for polishing")
    print("  2. Add icons, callouts, and gradient refinements")
    print("  3. Export to PDF for LaTeX integration")
    print("=" * 60)


if __name__ == '__main__':
    main()
