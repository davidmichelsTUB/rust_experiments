import sys, os

sys.path.insert(0, os.path.abspath("../../"))  # project root

import adapters.QuantileTransformer as ParallelQuantileTransformer
from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer
import numpy as np
import time
import statistics
import pandas as pd

from pathlib import Path


params = {
    "rows" : [100, 1000, 10000, 100000, 500000, 1000000, 5000000],
    "cols" : [500],
    "kwargs" : [{"n_quantiles" : a, "output_distribution" : b, "random_state" : 42} for a in [1000] for b in ['normal', 'uniform']],
    "njobs" : [1,2,4,8,16]
}


def main(repeats: int = 4):

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
     
    
    for njobs in params["njobs"]:
        for cols in params["cols"]:
            for kwargs in params["kwargs"]:
                for rows in params["rows"]:
                    print(f"Rows: {rows}, Cols: {cols}")
                    X = np.random.rand(rows, cols).astype(np.float64)

                    # Fit and transform using sklearn
                    sklearn_transformer = SklearnQuantileTransformer(**kwargs)
                    sklearn_mean, sklearn_std, _ = time_it(lambda: sklearn_transformer.fit_transform(X))

                    # Fit and transform using parallel implementation
                    parallel_transformer = ParallelQuantileTransformer(**kwargs, n_jobs=njobs)
                    parallel_mean, parallel_std, _ = time_it(lambda: parallel_transformer.fit_transform(X))
                    # Compare results


                    results = {
                        "rows": rows,
                        "cols": cols,
                        "sklearn_mean": sklearn_mean,
                        "sklearn_std": sklearn_std,
                        "parallel_mean": parallel_mean,
                        "parallel_std": parallel_std,
                        "error_quantiles": ((sklearn_transformer.quantiles_ - parallel_transformer.quantiles_)**2).mean(),
                        "kwargs": kwargs,
                        "njobs": njobs
                    }

                    if not Path(__file__).parent.joinpath("results_fit_transform.csv").exists():
                        results_csv = pd.DataFrame([results])
                        results_csv.to_csv(Path(__file__).parent / "results_fit_transform.csv", index=False)
                    else:
                        results_csv = pd.read_csv(Path(__file__).parent / "results_fit_transform.csv")
                        results_csv = pd.concat([results_csv, pd.DataFrame([results])], ignore_index=True)
                        results_csv.to_csv(Path(__file__).parent / "results_fit_transform.csv", index=False)


if __name__ == "__main__":
    main(repeats=4)