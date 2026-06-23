"""Plot rust vs sklearn fit times and accuracies from results/col_row_iter.csv.

Each of fit_time / predict_time / acc is a 2-element list [rust, sklearn].

Figure 1 (times): left = fix column, rows vs time; right = fix rows, cols vs time.
Figure 2 (accuracies): same layout for accuracy.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = "results/col_row_iter.csv"


def parse_pair(cell):
    """Extract the two floats from a list-like cell (handles np.float64(...))."""
    s = re.sub(r"np\.float64", "", str(cell))  # avoid matching the '64'
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(nums[0]), float(nums[1])


df = pd.read_csv(CSV)
df[["fit_rust", "fit_sklearn"]] = df["fit_time"].apply(
    lambda c: pd.Series(parse_pair(c))
)
df[["acc_rust", "acc_sklearn"]] = df["acc"].apply(lambda c: pd.Series(parse_pair(c)))

row_vals = sorted(df["rows"].unique())
col_vals = sorted(df["cols"].unique())

cmap = plt.get_cmap("viridis")
col_colors = {c: cmap(i / max(1, len(col_vals) - 1)) for i, c in enumerate(col_vals)}
row_colors = {r: cmap(i / max(1, len(row_vals) - 1)) for i, r in enumerate(row_vals)}


def style_legend(ax, color_map, color_title):
    """Legend combining color groups and rust/sklearn linestyles."""
    color_handles = [
        Line2D([0], [0], color=color_map[k], lw=2, label=str(k)) for k in color_map
    ]
    style_handles = [
        Line2D([0], [0], color="gray", lw=2, ls="-", marker="o", label="rust"),
        Line2D([0], [0], color="gray", lw=2, ls="--", marker="s", label="sklearn"),
    ]
    leg1 = ax.legend(handles=color_handles, title=color_title, fontsize=8,
                     loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, title="impl", fontsize=8, loc="lower right")


# ----------------------------------------------------------------------------
# Figure 1: times
# ----------------------------------------------------------------------------
fig1, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

# (a) fix column -> rows vs time, color = column
for c in col_vals:
    sub = df[df["cols"] == c].sort_values("rows")
    ax_a.plot(sub["rows"], sub["fit_rust"], color=col_colors[c], ls="-", marker="o")
    ax_a.plot(sub["rows"], sub["fit_sklearn"], color=col_colors[c], ls="--", marker="s")
ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlabel("rows (n samples)")
ax_a.set_ylabel("fit time [s]")
ax_a.set_title("Fit time vs rows (fixed #columns)")
style_legend(ax_a, col_colors, "cols")

# (b) fix rows -> cols vs time, color = rows
for r in row_vals:
    sub = df[df["rows"] == r].sort_values("cols")
    ax_b.plot(sub["cols"], sub["fit_rust"], color=row_colors[r], ls="-", marker="o")
    ax_b.plot(sub["cols"], sub["fit_sklearn"], color=row_colors[r], ls="--", marker="s")
ax_b.set_xscale("log")
ax_b.set_yscale("log")
ax_b.set_xlabel("cols (n features)")
ax_b.set_ylabel("fit time [s]")
ax_b.set_title("Fit time vs cols (fixed #rows)")
style_legend(ax_b, row_colors, "rows")

fig1.suptitle("Rust vs scikit-learn fit time", fontsize=14)
fig1.tight_layout()
fig1.savefig("results/fit_time_col_row.png", dpi=150)

# ----------------------------------------------------------------------------
# Figure 2: accuracies
# ----------------------------------------------------------------------------
fig2, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(14, 6))

for c in col_vals:
    sub = df[df["cols"] == c].sort_values("rows")
    ax_c.plot(sub["rows"], sub["acc_rust"], color=col_colors[c], ls="-", marker="o")
    ax_c.plot(sub["rows"], sub["acc_sklearn"], color=col_colors[c], ls="--", marker="s")
ax_c.set_xscale("log")
ax_c.set_xlabel("rows (n samples)")
ax_c.set_ylabel("accuracy")
ax_c.set_title("Accuracy vs rows (fixed #columns)")
style_legend(ax_c, col_colors, "cols")

for r in row_vals:
    sub = df[df["rows"] == r].sort_values("cols")
    ax_d.plot(sub["cols"], sub["acc_rust"], color=row_colors[r], ls="-", marker="o")
    ax_d.plot(sub["cols"], sub["acc_sklearn"], color=row_colors[r], ls="--", marker="s")
ax_d.set_xscale("log")
ax_d.set_xlabel("cols (n features)")
ax_d.set_ylabel("accuracy")
ax_d.set_title("Accuracy vs cols (fixed #rows)")
style_legend(ax_d, row_colors, "rows")

fig2.suptitle("Rust vs scikit-learn accuracy", fontsize=14)
fig2.tight_layout()
fig2.savefig("results/accuracy_col_row.png", dpi=150)

print("wrote results/fit_time_col_row.png and results/accuracy_col_row.png")
