import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys

df = pd.read_csv("benchmark.csv")
UNIT = "ms" 

n_rows_vals = sorted(df["n_rows"].unique())   # one panel per n_rows
n_cols_vals = sorted(df["n_cols"].unique())   # x-axis within each panel

GRID_COLS = 2
GRID_ROWS = int(np.ceil(len(n_rows_vals) / GRID_COLS))

fig, axes = plt.subplots(
    GRID_ROWS, GRID_COLS,
    figsize=(4.5 * GRID_COLS, 3.4 * GRID_ROWS),
    squeeze=False,
)

width = 0.4
colors = {
    ("sklearn", "fit"):       "#c1272d", 
    ("sklearn", "transform"): "#f4a3a3", 
    ("rust", "fit"):          "#1f5fb0", 
    ("rust", "transform"):    "#a3c4f4", 
}

for i, nr in enumerate(n_rows_vals):
    ax = axes[i // GRID_COLS][i % GRID_COLS]
    sub = df[df["n_rows"] == nr].sort_values("n_cols")
    x = np.arange(len(sub))

    # sklearn: fit on the bottom, transform stacked on top
    ax.bar(x - width / 2, sub["baseline_fit"], width,
           color=colors[("sklearn", "fit")])
    ax.bar(x - width / 2, sub["baseline_transform"], width,
           bottom=sub["baseline_fit"], color=colors[("sklearn", "transform")])

    # rust: same stacking
    ax.bar(x + width / 2, sub["rust_fit"], width,
           color=colors[("rust", "fit")])
    ax.bar(x + width / 2, sub["rust_transform"], width,
           bottom=sub["rust_fit"], color=colors[("rust", "transform")])

    ax.set_xticks(x)
    ax.set_xticklabels(sub["n_cols"])
    ax.set_title(f"n_rows = {nr:,}")
    ax.set_xlabel("n_cols")
    ax.set_ylabel(f"time ({UNIT})")
    ax.grid(True, axis="y", ls=":", alpha=0.5)

# turn off any empty panels
for j in range(len(n_rows_vals), GRID_ROWS * GRID_COLS):
    axes[j // GRID_COLS][j % GRID_COLS].axis("off")

legend_handles = [
    Patch(color=colors[("sklearn", "fit")],       label="sklearn fit"),
    Patch(color=colors[("sklearn", "transform")], label="sklearn transform"),
    Patch(color=colors[("rust", "fit")],          label="rust fit"),
    Patch(color=colors[("rust", "transform")],    label="rust transform"),
]
fig.legend(handles=legend_handles, ncol=4, loc="upper center",
           bbox_to_anchor=(0.5, 1.0))
fig.suptitle("sklearn vs Rust MinMaxScaler  (fit + transform stacked)",
             fontsize=14, y=1.06)
fig.tight_layout()
fig.savefig(f"results/benchmark_{sys.argv[1]}.png", dpi=150, bbox_inches="tight")
plt.show()
