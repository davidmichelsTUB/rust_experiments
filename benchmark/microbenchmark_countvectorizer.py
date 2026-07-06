from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer
from adapter import RustyCountVectorizer
import time
import string
import random
import sys

from .cvbenchmarkutils import warmup, get_synthetic_data


def main():
    warmup()
    print("#" * 20)
    print("STARTANALYSIS")
    n_unique_words_list = [100, 1_000, 10_000, 50_000]
    dataset_length_list = [10_000, 100_000, 1_000_000]
    for n_unique_words in n_unique_words_list:
        for dataset_length in dataset_length_list:
            output = {
                "Config": {
                    "n_unique_words": n_unique_words,
                    "dataset_length": dataset_length,
                }
            }
            print(output)
            data = get_synthetic_data(
                n_unique_words=n_unique_words,
                max_length=10,
                n_sentences=dataset_length,
                max_sentence_len=100,
            )
            cv = RustyCountVectorizer()
            t0 = time.perf_counter()
            out = cv.fit_transform(data)
            total = time.perf_counter() - t0
            print(f"python_total: {total * 1000:.9f}")
            print("\n\n")


if __name__ == "__main__":
    main()
