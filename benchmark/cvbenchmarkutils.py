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


def get_synthetic_data(n_unique_words, max_length, n_sentences, max_sentence_len):
    words = []
    for _ in range(n_unique_words):
        word_len = random.choice(list(range(2, max_length)))
        word = "".join(random.choices(string.ascii_letters, k=word_len))
        words.append(word)
        sentences = []

    for _ in range(n_sentences):
        sentence_len = random.choice(list(range(1, max_sentence_len)))
        sentence = " ".join(random.choices(words, k=sentence_len)) + "."
        sentences.append(sentence)

    return sentences


def warmup():
    cv = RustyCountVectorizer()
    cv.fit_transform(get_synthetic_data(100, 10000, 100, 42))
    cv.fit_transform(get_synthetic_data(100, 10000, 100, 42))
    cv.fit_transform(get_synthetic_data(100, 10000, 100, 42))
    cv.fit_transform(get_synthetic_data(100, 10000, 100, 42))
