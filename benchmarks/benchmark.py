import sys, os

sys.path.insert(0, os.path.abspath(".."))  # project root

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from adapters.LogisticRegression import LogisticRegression as RustLogisticRegression
import timeit
import pandas as pd

_ROWS = [
    100,
    500,
    1000,
    5000,
    10000,
    # 25_000,
    # 50_000,
    # 100_000,
    # 500_000,
    # 1_000_000,
    # 5_000_000
]

_COLUMNS = [
    10,
    50,
    100,
    500,
    1000,
    # 5000,
    # 10_000,
    # 100_000,
    # 500_000,
]

_MAX_ITERS = [1000]


import time
import statistics
import argparse
from pathlib import Path


def bench_binary(
    rows: int,
    cols: int,
    max_iters: int,
    kernel: str = "fused_parallel",
    repeats: int = 4,
):
    X, y = make_classification(
        n_samples=rows,
        n_features=cols,
        n_informative=cols // 2,
        n_redundant=cols // 4,
        random_state=42,
    )

    results = {
        "rows": rows,
        "cols": cols,
        "max_iters": max_iters,
        "kernel": kernel,
        "repeats": repeats,
    }

    def time_it(fn):
        """Best-of-N wall-clock timing. Returns (seconds, last_result).

        One untimed warm-up absorbs thread-pool spin-up / allocation / page
        faults; min() over the repeats strips OS scheduling jitter. The result
        is captured so callers never recompute just to score it.
        """
        result = fn()  # warm-up, not timed
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn()
            ts.append(time.perf_counter() - t0)
        mean = sum(ts) / len(ts)
        std = statistics.stdev(ts) if len(ts) > 1 else 0.0
        return mean, std, result

    # Rust implementation
    rust_model = RustLogisticRegression(max_iter=max_iters, fit_intercept=False)
    rust_fit_time, rust_fit_std, _ = time_it(lambda: rust_model.fit(X, y))
    rust_predict_time, rust_predict_std, rust_proba = time_it(
        lambda: rust_model.predict_proba(X, kernel=kernel)
    )
    rust_acc = ((rust_proba >= 0.5).astype(int) == y).mean()

    # scikit-learn implementation
    sklearn_model = SklearnLogisticRegression(max_iter=max_iters, solver="lbfgs", fit_intercept=False)
    sklearn_fit_time, sklearn_fit_std, _ = time_it(lambda: sklearn_model.fit(X, y))
    sklearn_predict_time, sklearn_predict_std, sklearn_pp = time_it(lambda: sklearn_model.predict_proba(X))
    sklearn_proba = sklearn_pp[:, 1]  # (n, 2) -> positive-class column
    sklearn_acc = ((sklearn_proba >= 0.5).astype(int) == y).mean()

    results["fit_time"] = [rust_fit_time, sklearn_fit_time]
    results["fit_std"] = [rust_fit_std, sklearn_fit_std]
    results["predict_time"] = [rust_predict_time, sklearn_predict_time]
    results["predict_std"] = [rust_predict_std, sklearn_predict_std]
    results["acc"] = [rust_acc, sklearn_acc]
    results["fit_rust_speedup"] = sklearn_fit_time / rust_fit_time
    results["predict_rust_speedup"] = sklearn_predict_time / rust_predict_time

    return results


def main():
    argparser = argparse.ArgumentParser(
        description="Benchmark Rust vs scikit-learn Logistic Regression"
    )
    argparser.add_argument(
        "--save",
        type=str,
        default="benchmark_results.csv",
        help="Path to save the benchmark results CSV file",
    )
    args = argparser.parse_args()
    df = []
    for rows in _ROWS:
        for cols in _COLUMNS:
            for max_iters in _MAX_ITERS:
                print(f"Benchmarking: rows={rows}, cols={cols}, max_iters={max_iters}")
                result = bench_binary(
                    rows, cols, max_iters, kernel="fused_parallel", repeats=4
                )
                print(result["acc"])
                df.append(result)

    df = pd.DataFrame(df)

    p = Path(args.save)
    p.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.save, index=False)


if __name__ == "__main__":
    main()