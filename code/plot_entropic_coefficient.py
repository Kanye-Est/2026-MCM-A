import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import UnivariateSpline
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection

# ── Load Data ──────────────────────────────────────────────────────────
df = pd.read_csv("dataset/temperature_entropy_table.csv")
soc = df["SOC"].values
ec = df["Entropy_Coeff_mV_per_K"].values

# ── Color Palette (muted, eye-friendly) ───────────────────────────────
COLOR_MAIN   = "#4A7C8A"   # teal-slate
COLOR_FILL_P = "#A8D0DB"   # light teal (positive region)
COLOR_FILL_N = "#D4A8B0"   # dusty rose  (negative region)
COLOR_ZERO   = "#8B8B8B"   # warm gray
COLOR_ANNO   = "#5B6770"   # charcoal blue
COLOR_GRID   = "#D6DDE1"   # pale silver
COLOR_SPINE  = "#9AABAF"   # soft steel
COLOR_BG     = "#FAFBFC"   # near-white
COLOR_ACCENT = "#C27C6E"   # muted terracotta for key points
COLOR_GRAD_A = "#4A7C8A"
COLOR_GRAD_B = "#7EAAB5"

# ── Figure Setup ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
fig.patch.set_facecolor(COLOR_BG)
ax.set_facecolor(COLOR_BG)

# ── Gradient-colored line via LineCollection ──────────────────────────
# Create segments for gradient effect based on y-value
points = np.column_stack([soc, ec])
segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

# Normalize ec values for colormap
norm = plt.Normalize(ec.min(), ec.max())
# Build a custom muted colormap
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "muted_teal_rose",
    ["#B07080", "#8B8B8B", "#4A7C8A"],  # negative → zero → positive
)
lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.4, zorder=5)
lc.set_array(ec[:-1])
ax.add_collection(lc)

# ── Shaded regions ───────────────────────────────────────────────────
ax.fill_between(soc, ec, 0, where=(ec >= 0), interpolate=True,
                color=COLOR_FILL_P, alpha=0.25, zorder=2, label="Endothermic region")
ax.fill_between(soc, ec, 0, where=(ec < 0), interpolate=True,
                color=COLOR_FILL_N, alpha=0.25, zorder=2, label="Exothermic region")

# ── Zero reference line ──────────────────────────────────────────────
ax.axhline(y=0, color=COLOR_ZERO, linewidth=0.9, linestyle="--", alpha=0.7, zorder=3)

# ── Key feature annotations ──────────────────────────────────────────
# Peak (maximum)
idx_max = np.argmax(ec)
ax.plot(soc[idx_max], ec[idx_max], 'o', color=COLOR_ACCENT, markersize=7,
        markeredgecolor='white', markeredgewidth=1.2, zorder=8)
ax.annotate(
    f"Peak: ({soc[idx_max]:.2f}, {ec[idx_max]:.3f})",
    xy=(soc[idx_max], ec[idx_max]),
    xytext=(soc[idx_max] + 0.12, ec[idx_max] + 0.03),
    fontsize=8.5, color=COLOR_ANNO, fontfamily="serif",
    arrowprops=dict(arrowstyle="-|>", color=COLOR_ANNO, lw=1.0,
                    connectionstyle="arc3,rad=-0.15"),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_GRID,
              alpha=0.9),
    zorder=9
)

# Trough (minimum)
idx_min = np.argmin(ec)
ax.plot(soc[idx_min], ec[idx_min], 'o', color=COLOR_ACCENT, markersize=7,
        markeredgecolor='white', markeredgewidth=1.2, zorder=8)
ax.annotate(
    f"Trough: ({soc[idx_min]:.2f}, {ec[idx_min]:.3f})",
    xy=(soc[idx_min], ec[idx_min]),
    xytext=(soc[idx_min] - 0.22, ec[idx_min] - 0.06),
    fontsize=8.5, color=COLOR_ANNO, fontfamily="serif",
    arrowprops=dict(arrowstyle="-|>", color=COLOR_ANNO, lw=1.0,
                    connectionstyle="arc3,rad=0.15"),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_GRID,
              alpha=0.9),
    zorder=9
)

# Zero-crossing point
zero_cross_idx = np.where(np.diff(np.sign(ec)))[0][0]
soc_zero = soc[zero_cross_idx] + (soc[zero_cross_idx+1] - soc[zero_cross_idx]) * \
           (-ec[zero_cross_idx]) / (ec[zero_cross_idx+1] - ec[zero_cross_idx])
ax.plot(soc_zero, 0, 's', color=COLOR_ZERO, markersize=7,
        markeredgecolor='white', markeredgewidth=1.2, zorder=8)
ax.annotate(
    f"Zero crossing\nSOC ≈ {soc_zero:.3f}",
    xy=(soc_zero, 0),
    xytext=(soc_zero + 0.13, 0.08),
    fontsize=8.5, color=COLOR_ANNO, fontfamily="serif",
    arrowprops=dict(arrowstyle="-|>", color=COLOR_ANNO, lw=1.0,
                    connectionstyle="arc3,rad=-0.2"),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_GRID,
              alpha=0.9),
    zorder=9
)

# ── Region labels (subtle) ──────────────────────────────────────────
ax.text(0.15, 0.25, "Endothermic\n(∂OCV/∂T > 0)", fontsize=9, color="#5A8A8A",
        fontfamily="serif", fontstyle="italic", ha="center", alpha=0.7, zorder=4)
ax.text(0.72, -0.15, "Exothermic\n(∂OCV/∂T < 0)", fontsize=9, color="#8A5A65",
        fontfamily="serif", fontstyle="italic", ha="center", alpha=0.7, zorder=4)

# ── Axes styling ─────────────────────────────────────────────────────
ax.set_xlabel("State of Charge (SOC)", fontsize=12, fontfamily="serif",
              color="#3A4A50", labelpad=8)
ax.set_ylabel("Entropic Coefficient  $\\frac{\\partial U_{OCV}}{\\partial T}$  (mV/K)",
              fontsize=12, fontfamily="serif", color="#3A4A50", labelpad=8)
ax.set_title("Entropic Coefficient as a Function of State of Charge",
             fontsize=14, fontfamily="serif", fontweight="bold", color="#2C3E44",
             pad=16)

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(ec.min() - 0.05, ec.max() + 0.08)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.025))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

ax.tick_params(axis='both', which='major', labelsize=9.5, colors="#4A5A60",
               direction='in', length=5, width=0.8)
ax.tick_params(axis='both', which='minor', direction='in', length=2.5, width=0.5,
               colors="#7A8A90")

for spine in ax.spines.values():
    spine.set_color(COLOR_SPINE)
    spine.set_linewidth(0.8)

ax.grid(True, which='major', color=COLOR_GRID, linewidth=0.6, alpha=0.8, zorder=0)
ax.grid(True, which='minor', color=COLOR_GRID, linewidth=0.3, alpha=0.4, zorder=0)

# ── Legend ────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLOR_MAIN, lw=2.2, label="$\\partial U_{OCV}/\\partial T$"),
    plt.Rectangle((0, 0), 1, 1, fc=COLOR_FILL_P, alpha=0.35, label="Endothermic region"),
    plt.Rectangle((0, 0), 1, 1, fc=COLOR_FILL_N, alpha=0.35, label="Exothermic region"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ACCENT,
           markersize=7, label="Critical points"),
]
legend = ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5,
                   frameon=True, fancybox=True, framealpha=0.92,
                   edgecolor=COLOR_GRID, prop={"family": "serif"})
legend.get_frame().set_linewidth(0.6)

# ── Subtle data source note ─────────────────────────────────────────
fig.text(0.98, 0.01, "Data: Li-ion cell entropic coefficient measurement",
         fontsize=7, color="#9AABAF", fontfamily="serif", ha="right", va="bottom")

# ── Save ─────────────────────────────────────────────────────────────
plt.tight_layout()
fig.savefig("figures/entropic_coefficient_vs_soc.png", dpi=300,
            bbox_inches="tight", facecolor=COLOR_BG, edgecolor='none')
fig.savefig("figures/entropic_coefficient_vs_soc.pdf",
            bbox_inches="tight", facecolor=COLOR_BG, edgecolor='none')
print("Saved to figures/entropic_coefficient_vs_soc.png and .pdf")
plt.close()
