#!/usr/bin/env python3
"""
Generate all figures for Section 6–8: Numerical Results, Case Studies,
Sensitivity & Uncertainty Analysis.

MCM High-End Academic Visualization Style applied throughout.

Figures:
  1. fig_6_1_soc_trajectory.pdf      - SOC comparison with zoomed inset + deviation
  2. fig_6_2_deviation_distribution.pdf - Residual histogram
  3. fig_6_3_avalanche_effect.pdf    - 4-panel avalanche demonstration
  4. fig_6_4_soh_sensitivity.pdf     - TTE vs SOH
  5. fig_6_5_thermal_evolution.pdf   - Temperature dynamics
  6. fig_6_6_power_profile.pdf       - Input power profile
  7. fig_6_7_ocv_curve.pdf           - Plett OCV model
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

# =============================================================================
# MCM HIGH-END ACADEMIC STYLE
# =============================================================================

MCM = {
    'primary':    '#26A69A',   # Teal (main model curves)
    'secondary':  '#78909C',   # Slate Grey (baselines, references)
    'accent':     '#E65100',   # Deep Orange — ONLY for critical highlights (sparingly)
    'bg_cyan':    '#E0F7FA',   # Light Cyan fill
    'bg_rose':    '#FCE4EC',   # Light Rose fill
    'edge':       '#37474F',   # Blue Grey 800 (axes)
    'deep_teal':  '#00796B',   # Darker teal for emphasis text
    'warm_grey':  '#8D6E63',   # Warm Brown-Grey (voltage curves)
    'purple':     '#7E57C2',   # Muted Purple (secondary data)
    'rose_accent':'#E91E63',   # Rose — only for critical markers
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.edgecolor': MCM['edge'],
    'axes.grid': True,
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'inherit',
    'figure.autolayout': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'lines.linewidth': 2.0,
})

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def _save(fig, name):
    """Save figure in multiple formats."""
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'),
                    bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Generated: {name}")


def load_npz(name):
    path = os.path.join(FIG_DIR, name)
    if os.path.exists(path):
        return np.load(path)
    print(f"  Warning: {path} not found")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6.1: SOC TRAJECTORY — DAE vs LINEAR  (KEY IMPROVEMENT)
# ─────────────────────────────────────────────────────────────────────────────

def fig_soc_trajectory():
    """
    Three-part figure:
      (a) Full SOC trajectory with zoomed inset at low SOC
      (b) Absolute deviation (DAE − Linear) highlighting the nonlinear zone
    """
    data = load_npz('validation_day_1.npz')
    if data is None:
        return

    t_h = data['t_data'] / 3600
    z_dae = data['z_dae'] * 100
    z_linear = data['z_linear'] * 100
    deviation = z_dae - z_linear  # DAE drains faster → negative at low SOC

    # ── Create figure with GridSpec ──
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.3], hspace=0.28)

    # ══════════════════════════════════════════════════════════════
    # (a) MAIN PANEL — Full SOC trajectory
    # ══════════════════════════════════════════════════════════════
    ax_main = fig.add_subplot(gs[0])

    ax_main.plot(t_h, z_dae, color=MCM['primary'], linewidth=2.5,
                 label=r'DAE Model (Proposed)', zorder=5)
    ax_main.plot(t_h, z_linear, color=MCM['secondary'], linewidth=2.0,
                 linestyle='--', label=r'Linear Baseline', zorder=4)

    # Shade the low-SOC avalanche zone (< 20%)
    ax_main.axhspan(0, 20, color=MCM['accent'], alpha=0.08, zorder=1)
    ax_main.axhline(20, color=MCM['accent'], linestyle=':', linewidth=1.0,
                    alpha=0.6, zorder=2)
    ax_main.text(1.0, 21.5, r'Avalanche Zone ($z < 20\%$)',
                 fontsize=9, color=MCM['accent'], fontstyle='italic')

    ax_main.set_xlabel(r'Time (hours)', fontsize=12)
    ax_main.set_ylabel(r'State of Charge (\%)', fontsize=12)
    ax_main.set_title(r'(a) SOC Trajectory: DAE Model vs Linear Baseline ($P = 1.5$ W equivalent)',
                      fontsize=12, fontweight='bold', pad=12)
    ax_main.set_xlim(0, t_h[z_dae > 0.5].max() + 1)
    ax_main.set_ylim(-2, 105)
    ax_main.legend(loc='upper right', fontsize=10,
                   framealpha=0.9, edgecolor=MCM['edge'])

    # ── ZOOMED INSET: low SOC region ──
    # Find where SOC < 30% to determine time window
    low_mask = z_dae < 30
    if low_mask.any():
        t_low_start = t_h[low_mask].min() - 0.5
        t_low_end = t_h[z_dae > 0.5].max() + 0.3

        axins = inset_axes(ax_main, width="45%", height="50%",
                           loc='center', bbox_to_anchor=(0.05, 0.18, 0.55, 0.55),
                           bbox_transform=ax_main.transAxes)

        axins.plot(t_h, z_dae, color=MCM['primary'], linewidth=2.5, zorder=5)
        axins.plot(t_h, z_linear, color=MCM['secondary'], linewidth=2.0,
                   linestyle='--', zorder=4)

        # Shade between the curves to show deviation
        axins.fill_between(t_h, z_dae, z_linear,
                           where=(z_dae < 30),
                           color=MCM['accent'], alpha=0.25, zorder=3,
                           label='Deviation')

        axins.set_xlim(t_low_start, t_low_end)
        axins.set_ylim(-2, 32)
        axins.set_xlabel('Time (h)', fontsize=8)
        axins.set_ylabel('SOC (%)', fontsize=8)
        axins.tick_params(labelsize=8)
        axins.set_title('Zoom: Low SOC Region', fontsize=9, fontweight='bold',
                        color=MCM['accent'])

        # Box styling
        for spine in axins.spines.values():
            spine.set_edgecolor(MCM['accent'])
            spine.set_linewidth(1.5)

        # Draw connector lines from inset to main plot
        mark_inset(ax_main, axins, loc1=1, loc2=3,
                   fc='none', ec=MCM['accent'], linewidth=1.2, linestyle='--')

        # Annotate the max deviation point
        max_dev_idx = np.argmax(np.abs(deviation))
        axins.annotate(
            f'Max $\\Delta$ = {abs(deviation[max_dev_idx]):.1f}%',
            xy=(t_h[max_dev_idx], z_dae[max_dev_idx]),
            xytext=(t_h[max_dev_idx] - 2, z_dae[max_dev_idx] + 10),
            fontsize=9, fontweight='bold', color=MCM['accent'],
            arrowprops=dict(arrowstyle='->', color=MCM['accent'], linewidth=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=MCM['accent'], alpha=0.9)
        )

    # ══════════════════════════════════════════════════════════════
    # (b) DEVIATION PANEL — highlights where DAE ≠ Linear
    # ══════════════════════════════════════════════════════════════
    ax_dev = fig.add_subplot(gs[1])

    ax_dev.fill_between(t_h, deviation, 0,
                        where=(deviation >= 0), color=MCM['bg_cyan'],
                        alpha=0.8, zorder=2)
    ax_dev.fill_between(t_h, deviation, 0,
                        where=(deviation < 0), color=MCM['bg_rose'],
                        alpha=0.8, zorder=2)
    ax_dev.plot(t_h, deviation, color=MCM['primary'], linewidth=2.0, zorder=5)
    ax_dev.axhline(0, color=MCM['edge'], linewidth=0.8, zorder=1)

    # Mark RMSE band
    rmse = np.sqrt(np.mean(deviation**2))
    ax_dev.axhline(rmse, color=MCM['secondary'], linestyle='--', linewidth=1.2,
                   label=f'RMSE = {rmse:.2f}%')
    ax_dev.axhline(-rmse, color=MCM['secondary'], linestyle='--', linewidth=1.2)

    # Shade avalanche zone on time axis
    if low_mask.any():
        t_aval_start = t_h[low_mask].min()
        ax_dev.axvspan(t_aval_start, t_h.max(), color=MCM['accent'],
                       alpha=0.08, zorder=1, label=r'Avalanche zone ($z<20\%$)')

    ax_dev.set_xlabel(r'Time (hours)', fontsize=12)
    ax_dev.set_ylabel(r'$\Delta$SOC: DAE $-$ Linear (\%)', fontsize=11)
    ax_dev.set_title(r'(b) Model Deviation — Nonlinear effects concentrate at low SOC',
                     fontsize=12, fontweight='bold', pad=8)
    ax_dev.set_xlim(0, t_h[z_dae > 0.5].max() + 1)
    ax_dev.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Stats box
    max_dev = np.max(np.abs(deviation))
    stats = (f'RMSE = {rmse:.2f}%\n'
             f'Max |Δ| = {max_dev:.2f}%\n'
             f'Mean = {deviation.mean():.2f}%')
    ax_dev.text(0.72, 0.50, stats, transform=ax_dev.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=MCM['primary'], alpha=0.9),
                fontfamily='monospace')

    _save(fig, 'fig_6_1_soc_trajectory')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6.2: DEVIATION DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def fig_deviation_distribution():
    all_deviation = []
    for day in ['day_1', 'day_2', 'day_3']:
        data = load_npz(f'validation_{day}.npz')
        if data is not None:
            all_deviation.extend(data['deviation'])
    if not all_deviation:
        return

    all_deviation = np.array(all_deviation)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    n, bins, patches = ax.hist(all_deviation, bins=30, color=MCM['primary'],
                               alpha=0.75, edgecolor='white', linewidth=0.8)

    ax.axvline(0, color=MCM['accent'], linestyle='--', linewidth=2.0,
               label='Zero deviation', zorder=10)
    ax.axvline(all_deviation.mean(), color=MCM['accent'], linestyle='-',
               linewidth=2.0, label=f'Mean = {all_deviation.mean():.3f}%', zorder=10)

    ax.set_xlabel(r'SOC Deviation: DAE $-$ Linear (\%)', fontsize=12)
    ax.set_ylabel(r'Frequency', fontsize=12)
    ax.set_title(r'Distribution of Model Deviation from Linear Baseline',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    rmse = np.sqrt(np.mean(all_deviation**2))
    stats = (f'RMSE = {rmse:.3f}%\n'
             f'MAE = {np.mean(np.abs(all_deviation)):.3f}%\n'
             f'$N$ = {len(all_deviation)}')
    ax.text(0.98, 0.92, stats, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=MCM['primary'], alpha=0.9))

    plt.tight_layout()
    _save(fig, 'fig_6_2_deviation_distribution')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6.3: AVALANCHE EFFECT  (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

def fig_avalanche_effect():
    data = load_npz('avalanche_demo.npz')
    if data is None:
        return

    t_h = data['t'] / 3600
    z = data['z'] * 100
    I = data['I']
    V = data['V']

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(r'Avalanche Effect at $P = 1.5$ W Constant Load',
                 fontsize=14, fontweight='bold', y=1.02)

    # ── (a) SOC vs Time ──
    ax = axes[0, 0]
    ax.plot(t_h, z, color=MCM['primary'], linewidth=2.5)
    ax.axhline(20, color=MCM['rose_accent'], linestyle='--', linewidth=1.5,
               alpha=0.8, label=r'$z = 20\%$ threshold')
    ax.axhspan(0, 20, color=MCM['bg_rose'], alpha=0.3)
    ax.set_xlabel(r'Time (hours)')
    ax.set_ylabel(r'SOC (\%)')
    ax.set_title(r'(a) SOC Trajectory', fontweight='bold')
    ax.set_xlim(0, t_h.max())
    ax.set_ylim(-2, 105)
    ax.legend(loc='upper right', fontsize=9)

    # ── (b) Current vs Time ──
    ax = axes[0, 1]
    ax.plot(t_h, I, color=MCM['purple'], linewidth=2.5)

    # Mark acceleration region
    z_raw = data['z']
    aval_mask = z_raw < 0.20
    if aval_mask.any():
        t_aval = t_h[aval_mask].min()
        ax.axvspan(t_aval, t_h.max(), color=MCM['bg_rose'], alpha=0.3)
        ax.axvline(t_aval, color=MCM['rose_accent'], linestyle=':', linewidth=1.2)
        ax.text(t_aval + 0.1, I.min() + (I.max()-I.min())*0.15,
                r'$z < 20\%$', fontsize=9, color=MCM['rose_accent'], fontstyle='italic')

    ax.set_xlabel(r'Time (hours)')
    ax.set_ylabel(r'$I_{\mathrm{batt}}$ (A)')
    ax.set_title(r'(b) Discharge Current', fontweight='bold')
    ax.set_xlim(0, t_h.max())

    # ── (c) Current vs SOC — KEY avalanche plot ──
    ax = axes[1, 0]
    # Color-code by SOC level
    for i in range(len(z)-1):
        color = MCM['accent'] if z[i] < 20 else MCM['primary']
        ax.plot(z[i:i+2], I[i:i+2], color=color, linewidth=2.5)

    ax.axvline(20, color=MCM['rose_accent'], linestyle='--', linewidth=1.5,
               alpha=0.8, label=r'Avalanche zone')
    ax.axvspan(0, 20, color=MCM['bg_rose'], alpha=0.3)

    # Annotate the surge
    surge_idx = np.argmin(z)
    ax.annotate(
        f'Current surge\n{I.max():.3f} A (+{(I.max()/I.min()-1)*100:.0f}%)',
        xy=(z[surge_idx]+1, I.max()*0.97),
        xytext=(40, I.max()*0.95),
        fontsize=10, fontweight='bold', color=MCM['accent'],
        arrowprops=dict(arrowstyle='->', color=MCM['accent'], linewidth=1.8),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=MCM['accent'], alpha=0.9)
    )

    ax.set_xlabel(r'SOC (\%)')
    ax.set_ylabel(r'$I_{\mathrm{batt}}$ (A)')
    ax.set_title(r'(c) Current vs SOC — Avalanche Effect', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(loc='center right', fontsize=9)

    # ── (d) Terminal Voltage vs Time ──
    ax = axes[1, 1]
    ax.plot(t_h, V, color=MCM['warm_grey'], linewidth=2.5)
    ax.axhline(3.0, color=MCM['rose_accent'], linestyle='--', linewidth=1.5,
               alpha=0.8, label=r'$V_{\mathrm{cutoff}} = 3.0$ V')
    ax.fill_between(t_h, V, 3.0, where=(V <= 3.05),
                    color=MCM['bg_rose'], alpha=0.4)

    # Mark cutoff time
    cutoff_mask = V <= 3.001
    if cutoff_mask.any():
        t_cutoff = t_h[cutoff_mask].min()
        ax.annotate(
            f'TTE = {t_cutoff:.2f} h',
            xy=(t_cutoff, 3.0),
            xytext=(t_cutoff - 2.5, 3.4),
            fontsize=10, fontweight='bold', color=MCM['deep_teal'],
            arrowprops=dict(arrowstyle='->', color=MCM['deep_teal'], linewidth=1.8),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=MCM['deep_teal'], alpha=0.9)
        )

    ax.set_xlabel(r'Time (hours)')
    ax.set_ylabel(r'$V_{\mathrm{term}}$ (V)')
    ax.set_title(r'(d) Terminal Voltage', fontweight='bold')
    ax.set_xlim(0, t_h.max())
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    _save(fig, 'fig_6_3_avalanche_effect')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6.4: SOH SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def fig_soh_sensitivity():
    soh_data = {}
    for soh in [100, 95, 90, 85, 80]:
        data = load_npz(f'soh_sweep_{soh}.npz')
        if data is not None:
            soh_data[soh] = data
    if not soh_data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── (a) SOC Trajectories ──
    ax = axes[0]
    # Create a teal-to-warm gradient for degradation severity
    soh_colors = {
        100: '#00796B',   # Deep Teal
        95:  '#26A69A',   # Medium Teal
        90:  '#78909C',   # Slate Grey
        85:  '#A1887F',   # Warm Grey
        80:  '#E65100',   # Orange (only worst case)
    }

    for soh in sorted(soh_data.keys(), reverse=True):
        data = soh_data[soh]
        t_h = data['t'] / 3600
        lw = 2.5 if soh in [100, 80] else 1.8
        ls = '-' if soh == 100 else '-'
        ax.plot(t_h, data['z'] * 100, color=soh_colors[soh], linewidth=lw,
                linestyle=ls, label=f'SOH = {soh}%')

    # Shade the low-SOC zone
    ax.axhspan(0, 20, color=MCM['accent'], alpha=0.05)

    ax.set_xlabel(r'Time (hours)', fontsize=12)
    ax.set_ylabel(r'SOC (\%)', fontsize=12)
    ax.set_title(r'(a) SOC Trajectory vs State of Health', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 105)

    # ── (b) TTE Bar Chart ──
    ax = axes[1]
    soh_vals = sorted(soh_data.keys(), reverse=True)
    ttes = []
    for soh in soh_vals:
        data = soh_data[soh]
        V = data['V']
        t = data['t']
        cutoff_idx = np.where(V <= 3.001)[0]
        ttes.append(t[cutoff_idx[0]] / 3600 if len(cutoff_idx) > 0 else t[-1] / 3600)

    bar_colors = [soh_colors[s] for s in soh_vals]
    bars = ax.bar([str(s) for s in soh_vals], ttes, color=bar_colors,
                  edgecolor='white', linewidth=1.5, width=0.65)

    # Add percentage labels and connector lines
    tte_100 = ttes[0]
    for i, (bar, tte, soh) in enumerate(zip(bars, ttes, soh_vals)):
        reduction = (1 - tte / tte_100) * 100
        label = f'{tte:.1f}h' if reduction == 0 else f'{tte:.1f}h\n({reduction:+.0f}%)'
        color = MCM['edge'] if soh != 80 else MCM['accent']
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=color)

    ax.set_xlabel(r'State of Health (\%)', fontsize=12)
    ax.set_ylabel(r'Time-to-Empty (hours)', fontsize=12)
    ax.set_title(r'(b) TTE Reduction with Battery Aging', fontweight='bold')
    ax.set_ylim(0, max(ttes) * 1.25)

    plt.tight_layout()
    _save(fig, 'fig_6_4_soh_sensitivity')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6.5: THERMAL EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def fig_thermal_evolution():
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    data = load_npz('avalanche_demo.npz')
    if data is None:
        return

    t_h = data['t'] / 3600
    T_C = data['T'] - 273.15
    z = data['z'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Create teal→rose colormap for temperature gradient
    teal_rose_cmap = LinearSegmentedColormap.from_list(
        'teal_rose', [MCM['primary'], MCM['warm_grey'], MCM['rose_accent']]
    )

    # ── (a) Temperature vs Time ──
    ax = axes[0]

    # Thermal zone bands
    ax.axhspan(25.0, 25.2, color=MCM['bg_cyan'], alpha=0.5, label='Nominal (25.0–25.2°C)')
    ax.axhspan(25.2, 25.4, color='#FFF9C4', alpha=0.5, label='Elevated (25.2–25.4°C)')  # Light yellow
    ax.axhspan(25.4, T_C.max() + 0.1, color=MCM['bg_rose'], alpha=0.5, label='High (>25.4°C)')

    # Gradient-colored line using LineCollection
    points = np.array([t_h, T_C]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Normalize temperature for coloring
    T_norm = (T_C - T_C.min()) / (T_C.max() - T_C.min() + 1e-9)
    lc = LineCollection(segments, cmap=teal_rose_cmap, linewidth=3.0)
    lc.set_array(T_norm[:-1])
    ax.add_collection(lc)

    ax.axhline(25, color=MCM['secondary'], linestyle='--', linewidth=1.5,
               alpha=0.7, label=r'$T_{\mathrm{amb}} = 25$ °C')

    # Find inflection point (where heating accelerates — near low SOC)
    # Use second derivative to find inflection
    dT = np.gradient(T_C, t_h)
    d2T = np.gradient(dT, t_h)
    inflection_idx = np.argmax(d2T[len(d2T)//2:]) + len(d2T)//2  # Look in second half

    ax.annotate(
        'Heating accelerates\n(low SOC region)',
        xy=(t_h[inflection_idx], T_C[inflection_idx]),
        xytext=(t_h[inflection_idx] - 2.5, T_C[inflection_idx] + 0.15),
        fontsize=9, fontweight='bold', color=MCM['deep_teal'],
        arrowprops=dict(arrowstyle='->', color=MCM['deep_teal'], linewidth=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=MCM['deep_teal'], alpha=0.9)
    )

    ax.set_xlabel(r'Time (hours)', fontsize=12)
    ax.set_ylabel(r'Battery Temperature (°C)', fontsize=12)
    ax.set_title(r'(a) Temperature Evolution', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, t_h.max())
    ax.set_ylim(T_C.min() - 0.05, T_C.max() + 0.1)
    ax.autoscale_view()

    # ── (b) Temperature vs SOC ──
    ax = axes[1]

    # Thermal zone bands (horizontal)
    ax.axhspan(25.0, 25.2, color=MCM['bg_cyan'], alpha=0.5)
    ax.axhspan(25.2, 25.4, color='#FFF9C4', alpha=0.5)
    ax.axhspan(25.4, T_C.max() + 0.1, color=MCM['bg_rose'], alpha=0.5)

    # Gradient-colored line
    points_b = np.array([z, T_C]).T.reshape(-1, 1, 2)
    segments_b = np.concatenate([points_b[:-1], points_b[1:]], axis=1)
    lc_b = LineCollection(segments_b, cmap=teal_rose_cmap, linewidth=3.0)
    lc_b.set_array(T_norm[:-1])
    ax.add_collection(lc_b)

    ax.axhline(25, color=MCM['secondary'], linestyle='--', linewidth=1.5,
               alpha=0.7, label=r'Ambient')

    # Vertical dashed line at SOC=20% marking avalanche onset
    ax.axvline(20, color=MCM['rose_accent'], linestyle='--', linewidth=1.5,
               alpha=0.8, label=r'Avalanche onset ($z=20\%$)')
    ax.axvspan(0, 20, color=MCM['bg_rose'], alpha=0.15)

    ax.set_xlabel(r'SOC (\%)', fontsize=12)
    ax.set_ylabel(r'Battery Temperature (°C)', fontsize=12)
    ax.set_title(r'(b) Temperature vs SOC', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(T_C.min() - 0.05, T_C.max() + 0.1)
    ax.invert_xaxis()
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.autoscale_view()

    plt.tight_layout()
    _save(fig, 'fig_6_5_thermal_evolution')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6.6: POWER PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def fig_power_profile():
    data = load_npz('validation_day_1.npz')
    if data is None:
        return

    t_h = data['t_data'] / 3600
    P = data['P_data']

    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.fill_between(t_h, P, alpha=0.2, color=MCM['primary'])
    ax.plot(t_h, P, color=MCM['primary'], linewidth=2.0)

    # Usage zone bands
    ax.axhspan(0, 1.0, color='#2E7D32', alpha=0.06)  # Forest Green
    ax.axhspan(1.0, 1.5, color='#FFA726', alpha=0.06)
    ax.axhspan(1.5, P.max()*1.1, color=MCM['accent'], alpha=0.06)

    ax.axhline(1.0, color=MCM['secondary'], linestyle=':', linewidth=1.0, alpha=0.6)
    ax.axhline(1.5, color=MCM['secondary'], linestyle=':', linewidth=1.0, alpha=0.6)

    ax.text(0.5, 0.7, 'Light', fontsize=9, color='#2E7D32', fontstyle='italic')
    ax.text(0.5, 1.2, 'Medium', fontsize=9, color='#E65100', fontstyle='italic')
    ax.text(0.5, 1.7, 'Heavy', fontsize=9, color=MCM['accent'], fontstyle='italic')

    ax.set_xlabel(r'Time (hours)', fontsize=12)
    ax.set_ylabel(r'Power (W)', fontsize=12)
    ax.set_title(r'Power Consumption Profile (Day 1)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, t_h.max())
    ax.set_ylim(0, P.max() * 1.15)

    plt.tight_layout()
    _save(fig, 'fig_6_6_power_profile')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6.7: OCV CURVE
# ─────────────────────────────────────────────────────────────────────────────

def fig_ocv_curve():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
    try:
        from grand_unified_solver import V_oc, BatteryParams
    except ImportError:
        print("  Warning: grand_unified_solver not importable, skipping OCV curve")
        return

    p = BatteryParams()
    z = np.linspace(0.01, 0.99, 200)
    V = [V_oc(zi, 298.15, p) for zi in z]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(z * 100, V, color=MCM['primary'], linewidth=2.5)

    # Shade steep region
    steep_mask = z < 0.15
    ax.fill_between(z[steep_mask] * 100, V[:steep_mask.sum()],
                    min(V), color=MCM['bg_rose'], alpha=0.4)

    ax.axhline(3.0, color=MCM['secondary'], linestyle='--', linewidth=1.5,
               alpha=0.8, label=r'$V_{\mathrm{cutoff}} = 3.0$ V')

    ax.annotate(
        'Steep OCV gradient\n(nonlinear zone)',
        xy=(8, 3.3), xytext=(30, 3.35),
        fontsize=10, fontweight='bold', color=MCM['deep_teal'],
        arrowprops=dict(arrowstyle='->', color=MCM['deep_teal'], linewidth=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=MCM['deep_teal'], alpha=0.9)
    )

    ax.set_xlabel(r'State of Charge (\%)', fontsize=12)
    ax.set_ylabel(r'Open-Circuit Voltage (V)', fontsize=12)
    ax.set_title(r'Plett Combined OCV Model: $V_{\mathrm{oc}}(z, T_{\mathrm{ref}})$',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    _save(fig, 'fig_6_7_ocv_curve')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Generating Section 6–8 Figures (MCM Academic Style)")
    print("=" * 60)
    os.makedirs(FIG_DIR, exist_ok=True)

    fig_soc_trajectory()
    fig_deviation_distribution()
    fig_avalanche_effect()
    fig_soh_sensitivity()
    fig_thermal_evolution()
    fig_power_profile()
    fig_ocv_curve()

    print("\n  All figures saved to:", FIG_DIR)


if __name__ == '__main__':
    main()
