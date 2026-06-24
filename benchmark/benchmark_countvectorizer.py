from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer
from adapter import RustyCountVectorizer, HAVE_RUST
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

N_WARMUP_RUNS = 3
N_REPEATS = 5
N_CHUNKS_LIST = [1, os.cpu_count()]
CORPUS_SIZES = [1_000, 10_000, 50_000, 150_000]


def load_corpus(path: str) -> list[str]:
    df = pd.read_csv(path, usecols=["text"])
    return df["text"].dropna().tolist()


def time_vectorizer(cv, corpus: list[str]) -> tuple[float, float]:
    t0 = time.perf_counter()
    cv.fit(corpus)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    cv.transform(corpus)
    transform_time = time.perf_counter() - t0
    return fit_time, transform_time


def warmup(corpus: list[str]) -> None:
    small = corpus[:10_000]
    for _ in range(N_WARMUP_RUNS):
        time_vectorizer(SKCountVectorizer(), small)
        if HAVE_RUST:
            time_vectorizer(RustyCountVectorizer(n_jobs=os.cpu_count()), small)


def main() -> None:
    print("Loading corpus…")
    full_corpus = load_corpus("data/news.csv")
    print(f"Loaded {len(full_corpus):,} documents")

    print("Warming up…")
    warmup(full_corpus)

    out_path = "benchmark_countvectorizer.csv"
    with open(out_path, "w") as f:
        f.write("n_docs,backend,n_chunks,fit_ms,transform_ms\n")

    for n_docs in tqdm(CORPUS_SIZES, desc="corpus sizes"):
        corpus = full_corpus[:n_docs]

        sk_results = [time_vectorizer(SKCountVectorizer(), corpus) for _ in range(N_REPEATS)]
        sk_fit, sk_transform = np.mean(sk_results, axis=0)
        with open(out_path, "a") as f:
            f.write(f"{n_docs},sklearn,1,{sk_fit*1000:.3f},{sk_transform*1000:.3f}\n")
        print(f"  sklearn         n={n_docs:>7,}  fit={sk_fit*1000:8.2f}ms  transform={sk_transform*1000:8.2f}ms")

        if HAVE_RUST:
            for n_chunks in N_CHUNKS_LIST:
                rust_results = [
                    time_vectorizer(RustyCountVectorizer(n_jobs=n_chunks), corpus)
                    for _ in range(N_REPEATS)
                ]
                r_fit, r_transform = np.mean(rust_results, axis=0)
                with open(out_path, "a") as f:
                    f.write(f"{n_docs},rust,{n_chunks},{r_fit*1000:.3f},{r_transform*1000:.3f}\n")
                print(f"  rust chunks={n_chunks:<3} n={n_docs:>7,}  fit={r_fit*1000:8.2f}ms  transform={r_transform*1000:8.2f}ms")

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
