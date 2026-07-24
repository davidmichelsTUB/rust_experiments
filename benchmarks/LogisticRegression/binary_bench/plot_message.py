import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_benchmark_data(title, xlabel, ylabel, x, y_rust, y_sklearn):
    width = 0.4
    x_pos = np.arange(len(x))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_rust = ax.bar(x_pos - width/2, y_rust, width, label="Rust")
    bars_sklearn = ax.bar(x_pos + width/2, y_sklearn, width, label="Scikit-Learn")

    ax.bar_label(bars_rust, fmt="%.2f")
    ax.bar_label(bars_sklearn, fmt="%.2f")
    ax.set_yscale('log')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    fig.tight_layout()
    return fig


def main():
    df = pd.read_csv(Path(__file__).parent / "message.csv")
    rows = df["rows"].to_numpy()
    fit_time = np.array(list(df["fit_time"].apply(lambda x : np.array([float(t) for t in x[1:-1].split(", ")]))))

    fig = plot_benchmark_data(
        title="Fit Time at 100 columns",
        xlabel="Number of Rows",
        ylabel="Time (s)",
        x=rows,
        y_rust=fit_time[:, 0],
        y_sklearn=fit_time[:, 1]
    )

    fig.savefig(Path(__file__).parent / "plot.png")

if __name__ == "__main__":
    main()