import unittest
import pytest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from adapter import RustyCountVectorizer


NEWS_PATH = "data/news.csv"


def load_news():
    with open(NEWS_PATH,"r") as f:
        content = f.readlines()
        text = [l.split(",")[0] for l in content]
    return text
class TestAdapterCountVectorizer(unittest.TestCase):
    
    
    def test_vocab(self):
        x = load_news()[:1000]
        sk = CountVectorizer()
        sk_out = sk.fit(x)
        rusty = RustyCountVectorizer()
        rusty_out = rusty.fit(x)
        
        # print(sk_out.vocabulary_)
        # print(rusty_out.vocabulary_)
        assert (sk_out.vocabulary_ == rusty_out.vocabulary_)
    def test_rusty_scaler(self):
        x = load_news()[:1000]
        sk = CountVectorizer()
        sk_out = sk.fit_transform(x)
        rusty = RustyCountVectorizer()
        rusty_out = rusty.fit_transform(x)
        assert (sk_out == rusty_out).all()
            