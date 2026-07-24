
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_results(results, title, xlabel, ylabel, ms=True):
    rows = results["Rows"]
    imgs = {
        "Rust": np.array(results["Rust_Time_ms"]) * 0.001 if ms else 1,
        "Scikit-Learn": np.array(results["Scikit_Learn_Time_ms"]) * 0.001 if ms else 1,
    }


    fig, ax = plt.subplots(layout='constrained')

    res = ax.grouped_bar(imgs, tick_labels=rows, group_spacing=1)
    for container in res.bar_containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

    # Add some text for labels, title, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend(loc='upper left', ncols=3)

    fig.savefig(f"{'_'.join(title.lower().split(' '))}.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--csv_path", type=str, help="Path to results.csv")
    parser.add_argument("--title", type=str, help="Title of the plot")
    parser.add_argument("--xlabel", type=str, help="X-axis label")
    parser.add_argument("--ylabel", type=str, help="Y-axis label")
    args = parser.parse_args()

    # Load results from CSV
    results = pd.read_csv(args.csv_path).to_dict(orient="list")

    plot_results(results, args.title, args.xlabel, args.ylabel)

if __name__ == "__main__":
    main()

