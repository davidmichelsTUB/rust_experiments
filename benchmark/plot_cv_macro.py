import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys

df = pd.read_csv("benchmark/results/macrobenchmark.csv")
df = df[(df["n_jobs"] == 24) | (df["version"] == "sklearn")]
print(df)
UNIT = "ms"

length_vals = sorted(df["dataset_length"].unique())
word_vals = sorted(df["n_unique_words"].unique())

GRID_COLS = 3
GRID_ROWS = int(np.ceil(len(length_vals) / GRID_COLS))

fig, axes = plt.subplots(
    GRID_ROWS,
    GRID_COLS,
    figsize=(4.5 * GRID_COLS, 3.4 * GRID_ROWS),
    squeeze=False,
)

width = 0.4
colors = {
    "sklearn": "#c1272d",
    "rust": "#1f5fb0",
}

for i, length in enumerate(length_vals):
    ax = axes[i // GRID_COLS][i % GRID_COLS]
    sub = df[df["dataset_length"] == length]
    x = np.arange(len(word_vals))

    skl = [
        sub[(sub.n_unique_words == w) & (sub.version == "sklearn")].time_ms.iloc[0]
        for w in word_vals
    ]
    rust = [
        sub[(sub.n_unique_words == w) & (sub.version == "rust")].time_ms.iloc[0]
        for w in word_vals
    ]

    ax.bar(x - width / 2, skl, width, color=colors["sklearn"])
    ax.bar(x + width / 2, rust, width, color=colors["rust"])

    for xi, s, r in zip(x, skl, rust):
        speedup = s / r
        ax.text(
            xi + width / 2,
            r,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(word_vals)
    ax.set_title(f"n_documents = {length:,}")
    ax.set_xlabel("Unique Words")
    ax.set_ylabel(f"time ({UNIT})")
    ax.grid(True, axis="y", ls=":", alpha=0.5)

for j in range(len(length_vals), GRID_ROWS * GRID_COLS):
    axes[j // GRID_COLS][j % GRID_COLS].axis("off")

legend_handles = [
    Patch(color=colors["sklearn"], label="sklearn"),
    Patch(color=colors["rust"], label="rust"),
]
fig.legend(
    handles=legend_handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.04)
)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(f"benchmark/results/cv_sklearn_vs_rust.png", dpi=300, bbox_inches="tight")
plt.show()
