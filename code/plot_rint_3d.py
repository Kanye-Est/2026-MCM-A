"""
3D Surface Plot: Internal Resistance Model R_int(z, T)

Model equations:
    R_int(z, T) = R_soc(z) * f_temp(T)

Where:
    R_soc(z) = R_base + K_rise * exp(-alpha * z)
    f_temp(T) = exp[B * (1/T - 1/T_ref)]

Style: MCM High-End Academic Visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# =============================================================================
# MCM HIGH-END ACADEMIC STYLE CONFIGURATION
# =============================================================================

# MCM "Smartphone Tech" Color Palette
COLORS = {
    'primary': '#0D47A1',      # Deep University Blue - main model
    'secondary': '#78909C',    # Slate Grey - baseline/reference
    'accent': '#E65100',       # Deep Academic Orange - highlights
    'bg_fill': '#E3F2FD',      # Very Light Blue - background
    'edge': '#263238',         # Dark Charcoal - axes
}

# Global matplotlib style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.edgecolor': COLORS['edge'],
    'axes.grid': True,
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'inherit',
    'figure.autolayout': True,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

OUTPUT_DIR = Path(__file__).parent.parent / 'figures'

# =============================================================================
# MODEL PARAMETERS (from main.tex Section 5.4.3)
# =============================================================================
R_base = 30      # mΩ - Plateau baseline
K_rise = 15      # mΩ - Surge magnitude at z=0
alpha = 20       # Steepness factor (confines surge to z < 0.15)
B = 3500         # K - Arrhenius activation parameter
T_ref = 298.15   # K (25°C) - Reference temperature

# =============================================================================
# CREATE GRID
# =============================================================================
z = np.linspace(0.05, 1.0, 100)       # SOC: 5% to 100%
T = np.linspace(273.15, 323.15, 100)  # Temperature: 0°C to 50°C
Z, T_grid = np.meshgrid(z, T)

# Convert to Celsius for display
T_celsius = T_grid - 273.15

# =============================================================================
# MODEL EQUATIONS
# =============================================================================
R_soc = R_base + K_rise * np.exp(-alpha * Z)
f_temp = np.exp(B * (1/T_grid - 1/T_ref))
R_int = R_soc * f_temp

# =============================================================================
# CREATE CUSTOM COLORMAP (MCM Academic Style)
# =============================================================================
# Blue-to-Orange gradient matching the MCM palette
cmap_colors = [
    COLORS['primary'],    # Deep Blue at low values
    '#1976D2',            # Medium Blue
    '#42A5F5',            # Light Blue
    '#FFB74D',            # Light Orange
    COLORS['accent']      # Deep Orange at high values
]
mcm_cmap = LinearSegmentedColormap.from_list('mcm_academic', cmap_colors, N=256)

# =============================================================================
# CREATE FIGURE
# =============================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set background pane colors
ax.xaxis.set_pane_color((0.95, 0.97, 1.0, 0.9))
ax.yaxis.set_pane_color((0.95, 0.97, 1.0, 0.9))
ax.zaxis.set_pane_color((0.95, 0.97, 1.0, 0.9))

# Plot main surface
surf = ax.plot_surface(Z, T_celsius, R_int,
                       cmap=mcm_cmap, alpha=0.9,
                       linewidth=0, antialiased=True,
                       rcount=80, ccount=80)

# =============================================================================
# CONTOUR PROJECTIONS ON WALLS
# =============================================================================

# Bottom wall (z-projection): shows overall 2D contour shape
ax.contour(Z, T_celsius, R_int, zdir='z', offset=0,
           cmap=mcm_cmap, alpha=0.7, levels=12, linewidths=1.2)

# Back wall (SOC=0.05, x-projection): shows T-dependence at low SOC
ax.contour(Z, T_celsius, R_int, zdir='x', offset=0.05,
           cmap=mcm_cmap, alpha=0.7, levels=12, linewidths=1.2)

# Side wall (T=50°C, y-projection): shows SOC surge at high temperature
ax.contour(Z, T_celsius, R_int, zdir='y', offset=50,
           cmap=mcm_cmap, alpha=0.7, levels=12, linewidths=1.2)

# =============================================================================
# KEY POINT ANNOTATIONS
# =============================================================================

# Maximum resistance point: low SOC + low temperature
z_max, T_max_C = 0.05, 0
T_max_K = T_max_C + 273.15
R_max = (R_base + K_rise * np.exp(-alpha * z_max)) * np.exp(B * (1/T_max_K - 1/T_ref))
ax.scatter([z_max], [T_max_C], [R_max], color='white', s=100, zorder=10,
           edgecolors=COLORS['accent'], linewidths=2.5, marker='o')
ax.text(z_max + 0.12, T_max_C + 5, R_max + 10,
        f'Peak: {R_max:.0f} mΩ\n(Low SOC, Cold)',
        fontsize=10, ha='left', color=COLORS['accent'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['accent'], alpha=0.9))

# High temp, high SOC point (minimum resistance region)
z_min, T_min_C = 0.95, 50
T_min_K = T_min_C + 273.15
R_min = (R_base + K_rise * np.exp(-alpha * z_min)) * np.exp(B * (1/T_min_K - 1/T_ref))
ax.scatter([z_min], [T_min_C], [R_min], color='white', s=100, zorder=10,
           edgecolors=COLORS['primary'], linewidths=2.5, marker='o')
ax.text(z_min - 0.30, T_min_C - 5, R_min + 35,
        f'Min: {R_min:.1f} mΩ\n(High SOC, Warm)',
        fontsize=10, ha='center', color=COLORS['edge'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['primary'], alpha=0.9))

# =============================================================================
# LABELS AND FORMATTING
# =============================================================================
ax.set_xlabel(r'SOC $z$', labelpad=12, fontsize=13)
ax.set_ylabel(r'Temperature (°C)', labelpad=12, fontsize=13)
ax.set_zlabel(r'$R_{\mathrm{int}}$ (m$\Omega$)', labelpad=12, fontsize=13)
ax.set_title(r'Internal Resistance: $R_{\mathrm{int}}(z, T) = R_{\mathrm{soc}}(z) \cdot f_{\mathrm{temp}}(T)$',
             fontsize=14, pad=20, fontweight='bold')

# Set axis limits to align projections
ax.set_xlim(0.05, 1.0)
ax.set_ylim(0, 50)
ax.set_zlim(0, R_int.max() * 1.15)

# Viewing angle: ROTATED 180° to show temperature effect prominently
# elev=25, azim=40 — now viewing from opposite side
ax.view_init(elev=25, azim=40)

# Colorbar with MCM styling
cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=18, pad=0.12)
cbar.set_label(r'$R_{\mathrm{int}}$ (m$\Omega$)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# =============================================================================
# SAVE FIGURES
# =============================================================================
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_rint_3d.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_DIR / 'fig_rint_3d.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_DIR / 'fig_rint_3d.svg', format='svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ Saved: {OUTPUT_DIR / 'fig_rint_3d.png'}")
print(f"✓ Saved: {OUTPUT_DIR / 'fig_rint_3d.pdf'}")
print(f"✓ Saved: {OUTPUT_DIR / 'fig_rint_3d.svg'}")

# Print key values for verification
print(f"\nKey resistance values:")
print(f"  Peak (z=0.05, T=0°C):     {R_max:.1f} mΩ")
print(f"  Minimum (z=0.95, T=50°C): {R_min:.1f} mΩ")
print(f"  Temperature factor range: {f_temp.min():.2f}x to {f_temp.max():.2f}x")
print(f"  Ratio (max/min): {R_max/R_min:.1f}x")

plt.close()
