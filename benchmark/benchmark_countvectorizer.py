from __future__ import annotations
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer
from adapter import RustyCountVectorizer, HAVE_RUST
import time
import os
import pandas as pd
from typing import Callable
N_REPEATS = 5
N_CHUNKS_LIST = [1, os.cpu_count()]


def load_corpus(path: Path) -> list[str]:
    df = pd.read_csv(path, usecols=["text"])
    return df["text"].dropna().tolist()


def run_trials(
    factory: Callable, corpus: list[str], n: int
) -> list[tuple[float, float]]:
    results = []
    for _ in range(n):
        cv = factory()

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
    corpus = load_corpus(HERE.parent / "data" / "news.csv")
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

        if HAVE_RUST:
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
