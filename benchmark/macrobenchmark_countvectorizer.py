from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer
from benchmark_adapter import RustyCountVectorizer
import time

from cvbenchmarkutils import warmup, get_synthetic_data

N_REPS = 5


def benchmark(cv, data, n_jobs=1):
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        cv.fit_transform(data, n_jobs)
    return (time.perf_counter() - t0) * 1000 / N_REPS


def main():
    warmup()
    n_jobs_list = [1, 2, 4, 8, 24]
    n_unique_words_list = [1_000, 10_000, 50_000]
    dataset_length_list = [10_000, 100_000, 1_000_000]

    out_path = Path("benchmark/results/macrobenchmark.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("n_jobs,n_unique_words,dataset_length,version,time_ms\n")
        for n_jobs in n_jobs_list:
            for n_unique_words in n_unique_words_list:
                for dataset_length in dataset_length_list:
                    data = get_synthetic_data(
                        n_unique_words=n_unique_words,
                        max_length=10,
                        n_sentences=dataset_length,
                        max_sentence_len=100,
                    )

                    for version, factory in [
                        ("rust", RustyCountVectorizer),
                        ("sklearn", SKCountVectorizer),
                    ]:
                        cv = factory()
                        if version == "sklearn":
                            if n_jobs != 1:
                                continue
                            else:
                                elapsed = benchmark(cv, data)
                        else:
                            elapsed = benchmark(cv, data, n_jobs)
                        f.write(
                            f"{n_jobs},{n_unique_words},{dataset_length},{version},{elapsed:.9f}\n"
                        )


if __name__ == "__main__":
    main()
