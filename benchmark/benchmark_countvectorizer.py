from __future__ import annotations
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer
from sklearn.datasets import fetch_20newsgroups
from adapter import RustyCountVectorizer
import time
import os
import pandas as pd
from typing import Callable


N_REPEATS = 5
N_CHUNKS_LIST = [1, os.cpu_count()]

def load_corpus() -> list[str]:
    ds = fetch_20newsgroups(data_home="data")
    
    return ds.get("data")


def run_trials(
    vectorizer: Callable, corpus: list[str], n: int
) -> list[tuple[float, float]]:
    results = []
    for _ in range(n):
        cv = vectorizer()
        t0 = time.perf_counter()
        cv.fit(corpus)
        fit_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        cv.transform(corpus)
        transform_ms = (time.perf_counter() - t0) * 1000
        results.append((fit_ms, transform_ms))
    return results


def main() -> None:
    print("Loading corpus…")
    corpus = load_corpus()
    corpus = corpus + corpus + corpus + corpus
    print(f"Loaded {len(corpus):,} documents")

    out_path = HERE / "results" / "countvectorizer.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("backend,n_chunks,run,fit_ms,transform_ms\n")

        print(f"\nsklearn  n={len(corpus):,}")
        for run, (fit_ms, transform_ms) in enumerate(
            run_trials(SKCountVectorizer, corpus, N_REPEATS), 1
        ):
            print(f"  run {run}  fit={fit_ms:8.2f}ms  transform={transform_ms:8.2f}ms")
            f.write(f"sklearn,1,{run},{fit_ms:.3f},{transform_ms:.3f}\n")


        for n_chunks in N_CHUNKS_LIST:
            print(f"\nrust chunks={n_chunks}  n={len(corpus):,}")
            for run, (fit_ms, transform_ms) in enumerate(
                run_trials(lambda c=n_chunks: RustyCountVectorizer(n_jobs=c), corpus, N_REPEATS), 1
            ):
                print(f"  run {run}  fit={fit_ms:8.2f}ms  transform={transform_ms:8.2f}ms")
                f.write(f"rust,{n_chunks},{run},{fit_ms:.3f},{transform_ms:.3f}\n")

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
