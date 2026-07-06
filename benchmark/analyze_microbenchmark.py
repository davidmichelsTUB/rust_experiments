import ast
import pandas as pd


def parse(path):
    rows = []
    current = None
    with open(path) as f:
        text = f.read().split("STARTANALYSIS", 1)[-1]
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            current = dict(ast.literal_eval(line)["Config"])
            rows.append(current)
        elif ":" in line and current is not None:
            key, val = line.split(":", 1)
            current[key.strip()] = float(val.replace("ms", "").strip())

    return pd.DataFrame(rows)


def compute_relative(df):
    cols = [
        x
        for x in df.columns
        if x != "n_unique_words" and x != "dataset_length" and "python" not in x
    ]
    rel = (df[cols].div(df["fit_total"], axis=0)).round(3)
    df[[f"{c}_rel" for c in cols]] = rel
    return df


def compute_overhead(df):
    df["python_overhead"] = df["python_total"] - df["fit_total"]
    df["rust_python_function_overhead"] = (
        df["python_total"] - df["rust_python_function_total"]
    )
    df["python_overhead_rel"] = (df["python_overhead"] / df["python_total"]).round(3)
    df["rust_python_function_overhead"] = (
        df["rust_python_function_overhead"] / df["python_total"]
    ).round(3)
    return df


if __name__ == "__main__":
    df = parse("benchmark/results/microbenchmark_cv.txt").sort_values(
        "dataset_length", ascending=False
    )
    df = compute_relative(df)
    df = compute_overhead(df)
    df.to_csv("benchmark/results/microbenchmark.csv")
    print(df)
